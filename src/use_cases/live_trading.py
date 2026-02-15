"""
v4.0 라이브 트레이딩 엔진 — 시그널→주문→모니터링 루프

기존 backtest_engine + signal_engine + position_sizer 로직을 라이브 환경으로 통합.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import yaml

from src.entities.trading_models import (
    ExitReason,
    LivePosition,
    Order,
    OrderStatus,
)
from src.use_cases.ports import BalancePort, CurrentPricePort, OrderPort
from src.use_cases.position_tracker import PositionTracker
from src.use_cases.safety_guard import SafetyGuard

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """실전 자동매매 엔진"""

    def __init__(
        self,
        order_port: OrderPort,
        balance_port: BalancePort,
        price_port: CurrentPricePort,
        tracker: PositionTracker,
        guard: SafetyGuard,
        config: dict | None = None,
    ):
        self.order_port = order_port
        self.balance_port = balance_port
        self.price_port = price_port
        self.tracker = tracker
        self.guard = guard

        self.config = config or {}
        live_cfg = self.config.get("live_trading", {})
        order_cfg = live_cfg.get("order", {})
        pos_cfg = live_cfg.get("position", {})
        monitor_cfg = live_cfg.get("monitor", {})

        self.default_order_type = order_cfg.get("default_type", "limit")
        self.slippage_ticks = order_cfg.get("slippage_ticks", 2)
        self.max_retry = order_cfg.get("max_retry", 3)
        self.retry_delay = order_cfg.get("retry_delay_sec", 5)
        self.cancel_after_sec = order_cfg.get("cancel_unfilled_after_sec", 300)

        self.max_positions = pos_cfg.get("max_positions", 4)
        self.initial_capital = pos_cfg.get("initial_capital", 50_000_000)

        self.monitor_interval = monitor_cfg.get("interval_sec", 60)

        # 포지션 사이저 초기화
        from src.position_sizer import PositionSizer
        self.sizer = PositionSizer(self.config)

    # ──────────────────────────────────────────
    # 매수 실행
    # ──────────────────────────────────────────

    def execute_buy_signals(self, signals: list[dict]) -> list[dict]:
        """
        시그널 기반 매수 실행.

        Args:
            signals: signal_engine.calculate_signal() 결과 리스트
                     각 dict에 ticker, entry_price, atr_value, grade, trigger_type 등 포함

        Returns:
            [{"ticker", "order", "position", "success"}, ...]
        """
        results = []

        # 1. 안전장치 체크
        balance = self.balance_port.get_available_cash()
        daily_loss = self._calc_daily_loss_pct()
        total_loss = self._calc_total_loss_pct(balance)

        state = self.guard.check_all(daily_loss, total_loss)
        if not state.can_trade:
            logger.warning("[매수] 안전장치 활성 — 매매 중단")
            return results

        if state.must_liquidate:
            self.guard.emergency_liquidate(self.tracker, self.order_port)
            return results

        # 2. 포지션 수 체크
        current_count = len(self.tracker.positions)
        available_slots = self.max_positions - current_count
        if available_slots <= 0:
            logger.info("[매수] 최대 포지션 도달 (%d/%d)", current_count, self.max_positions)
            return results

        # 3. 시그널 정렬 (등급 A→C, zone_score 높은 순)
        grade_order = {"A": 0, "B": 1, "C": 2, "F": 3}
        sorted_signals = sorted(
            [s for s in signals if s.get("signal", False)],
            key=lambda s: (grade_order.get(s.get("grade", "F"), 3), -s.get("zone_score", 0)),
        )

        # 4. 매수 실행 (가용 슬롯만큼)
        for sig in sorted_signals[:available_slots]:
            ticker = sig.get("ticker", "")
            if not ticker:
                continue

            # 이미 보유 중인 종목 스킵
            if any(p.ticker == ticker for p in self.tracker.positions):
                logger.info("[매수] %s 이미 보유 중 — 스킵", ticker)
                continue

            result = self._execute_single_buy(sig, balance)
            results.append(result)

            if result.get("success"):
                available_slots -= 1
                balance -= result.get("investment", 0)

        return results

    def _execute_single_buy(self, signal: dict, available_cash: float) -> dict:
        """단일 종목 매수 실행"""
        ticker = signal["ticker"]
        entry_price = int(signal.get("entry_price", 0))

        # 현재가 조회
        if entry_price <= 0:
            price_data = self.price_port.fetch_current_price(ticker)
            entry_price = price_data.get("current_price", 0)

        if entry_price <= 0:
            return {"ticker": ticker, "success": False, "reason": "현재가 조회 실패"}

        # 포지션 사이징
        grade_ratio = {"A": 1.0, "B": 0.67, "C": 0.33}.get(signal.get("grade", "C"), 0.33)
        stage_pct = signal.get("entry_stage_pct", 0.4)

        portfolio_risk = sum(
            (p.entry_price - p.stop_loss) * p.shares
            for p in self.tracker.positions
            if p.stop_loss < p.entry_price
        )

        sizing = self.sizer.calculate(
            account_balance=available_cash,
            entry_price=entry_price,
            atr_value=signal.get("atr_value", entry_price * 0.02),
            grade_ratio=grade_ratio,
            current_portfolio_risk=portfolio_risk,
            stage_pct=stage_pct,
        )

        shares = sizing["shares"]
        if shares <= 0:
            return {"ticker": ticker, "success": False, "reason": "계산 수량 0"}

        # 주문 가격 결정 (지정가: 현재가 + 슬리피지)
        order_price = entry_price + self.slippage_ticks if self.default_order_type == "limit" else 0

        # 주문 실행 (재시도 로직)
        order = None
        for attempt in range(self.max_retry):
            if self.default_order_type == "limit":
                order = self.order_port.buy_limit(ticker, order_price, shares)
            else:
                order = self.order_port.buy_market(ticker, shares)

            if order.status != OrderStatus.FAILED:
                self.guard.reset_api_failure()
                break

            logger.warning("[매수] %s 주문 실패 (%d/%d) — %s", ticker, attempt + 1, self.max_retry, order.message)
            if self.guard.record_api_failure():
                return {"ticker": ticker, "success": False, "reason": "API 연속 실패"}
            time.sleep(self.retry_delay)

        if order is None or order.status == OrderStatus.FAILED:
            return {"ticker": ticker, "success": False, "reason": order.message if order else "주문 실패"}

        # 체결 대기
        filled_order = self._wait_for_fill(order)

        if filled_order.status in (OrderStatus.FILLED, OrderStatus.PENDING):
            # 포지션 등록
            pos = self.tracker.add_position(filled_order, signal)
            # 텔레그램 알림
            self._send_buy_alert(filled_order, signal, sizing)
            return {
                "ticker": ticker,
                "order": filled_order,
                "position": pos,
                "investment": sizing["investment"],
                "success": True,
            }
        else:
            return {"ticker": ticker, "success": False, "reason": "체결 실패/취소"}

    def _wait_for_fill(self, order: Order, timeout: int | None = None) -> Order:
        """체결 대기 (timeout초 후 미체결 시 취소)"""
        timeout = timeout or self.cancel_after_sec
        start = time.time()

        while time.time() - start < timeout:
            status = self.order_port.get_order_status(order.order_id)
            if status.status == OrderStatus.FILLED:
                order.status = OrderStatus.FILLED
                order.filled_quantity = status.filled_quantity or order.quantity
                order.filled_price = status.filled_price or order.price
                return order
            time.sleep(5)

        # 타임아웃 → 취소
        logger.warning("[매수] %s 미체결 %d초 → 취소", order.ticker, timeout)
        self.order_port.cancel(order)
        order.status = OrderStatus.CANCELLED
        return order

    # ──────────────────────────────────────────
    # 매도 실행
    # ──────────────────────────────────────────

    def execute_sell_signals(self) -> list[dict]:
        """청산 조건 기반 매도 실행"""
        results = []
        exits = self.tracker.check_exit_conditions()

        for pos, reason, quantity in exits:
            result = self._execute_single_sell(pos, reason, quantity)
            results.append(result)

        return results

    def _execute_single_sell(
        self, pos: LivePosition, reason: ExitReason, quantity: int,
    ) -> dict:
        """단일 종목 매도 실행"""
        ticker = pos.ticker
        logger.info(
            "[매도] %s %d주 (사유=%s)", ticker, quantity, reason.value,
        )

        # 긴급/손절은 시장가, 나머지는 지정가
        if reason in (ExitReason.EMERGENCY, ExitReason.STOP_LOSS, ExitReason.PCT_STOP):
            order = self.order_port.sell_market(ticker, quantity)
        else:
            sell_price = int(pos.current_price)
            order = self.order_port.sell_limit(ticker, sell_price, quantity)

        if order.status == OrderStatus.FAILED:
            # 실패 시 시장가 재시도
            logger.warning("[매도] %s 지정가 실패 → 시장가 재시도", ticker)
            order = self.order_port.sell_market(ticker, quantity)

        if order.status != OrderStatus.FAILED:
            self.tracker.apply_partial_exit(pos, quantity, reason)
            self._send_sell_alert(pos, order, reason, quantity)

        return {
            "ticker": ticker,
            "shares": quantity,
            "reason": reason.value,
            "order": order,
            "success": order.status != OrderStatus.FAILED,
        }

    # ──────────────────────────────────────────
    # 장중 모니터링 루프
    # ──────────────────────────────────────────

    def monitor_loop(self, duration_sec: int = 0) -> None:
        """
        장중 실시간 모니터링.

        Args:
            duration_sec: 0이면 15:20까지 무한 루프, 양수면 해당 초 동안만 실행
        """
        logger.info("[모니터] 장중 모니터링 시작 (간격=%d초)", self.monitor_interval)
        start_time = time.time()
        cycle = 0

        while True:
            cycle += 1
            try:
                # 안전장치 체크
                if self.guard.check_stop_signal():
                    logger.warning("[모니터] STOP.signal 감지 — 모니터링 중단")
                    break

                if self.guard.check_reboot_trigger():
                    logger.info("[모니터] reboot.trigger 감지 — 재시작")
                    break

                # 현재가 갱신
                self.tracker.update_prices(self.price_port)

                # 청산 조건 체크 + 자동 매도
                sells = self.execute_sell_signals()
                if sells:
                    for s in sells:
                        logger.info(
                            "[모니터] 자동 매도: %s %d주 (%s) → %s",
                            s["ticker"], s["shares"], s["reason"],
                            "성공" if s["success"] else "실패",
                        )

                # 10분마다 상태 로그
                if cycle % 10 == 0:
                    summary = self.tracker.get_summary()
                    logger.info(
                        "[모니터] 보유 %d종목, 평가 %s원, 수익률 %.1f%%",
                        summary["count"],
                        f"{summary['total_eval']:,}",
                        summary["total_pnl_pct"],
                    )

                # 손실 한도 체크
                daily_loss = self._calc_daily_loss_pct()
                total_loss = self._calc_total_loss_pct(
                    self.balance_port.get_available_cash()
                )
                state = self.guard.check_all(daily_loss, total_loss)

                if state.must_liquidate:
                    logger.critical("[모니터] 총 손실 한도 초과 — 긴급 청산!")
                    self.guard.emergency_liquidate(self.tracker, self.order_port)
                    break

                if not state.can_trade:
                    logger.warning("[모니터] 안전장치 활성 — 매매 중단 (모니터링은 계속)")

            except Exception as e:
                logger.error("[모니터] 오류: %s", e)
                if self.guard.record_api_failure():
                    logger.critical("[모니터] API 연속 실패 — 모니터링 중단")
                    break

            # 종료 조건
            if duration_sec > 0 and (time.time() - start_time) >= duration_sec:
                logger.info("[모니터] 설정 시간(%d초) 도달 — 종료", duration_sec)
                break

            # 15:20 장중 모니터링 종료
            now = datetime.now()
            if now.hour >= 15 and now.minute >= 20:
                logger.info("[모니터] 15:20 — 장중 모니터링 종료")
                break

            time.sleep(self.monitor_interval)

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _calc_daily_loss_pct(self) -> float:
        """당일 손실률 계산"""
        summary = self.tracker.get_summary()
        total_pnl = summary.get("total_pnl", 0)
        if self.initial_capital <= 0:
            return 0.0
        return total_pnl / self.initial_capital

    def _calc_total_loss_pct(self, available_cash: float) -> float:
        """총 손실률 계산 (예수금 + 평가 vs 초기 자본)"""
        summary = self.tracker.get_summary()
        total_value = available_cash + summary.get("total_eval", 0)
        if self.initial_capital <= 0:
            return 0.0
        return (total_value - self.initial_capital) / self.initial_capital

    def _send_buy_alert(self, order: Order, signal: dict, sizing: dict) -> None:
        """매수 알림 (텔레그램)"""
        try:
            from src.telegram_sender import send_trade_alert
            alert_data = {
                "ticker": order.ticker,
                "name": signal.get("name", ""),
                "action": "BUY",
                "price": order.filled_price or order.price,
                "shares": order.filled_quantity or order.quantity,
                "grade": signal.get("grade", ""),
                "trigger_type": signal.get("trigger_type", ""),
                "stop_loss": sizing.get("stop_loss", 0),
                "target": sizing.get("target", 0),
                "investment": sizing.get("investment", 0),
                "pct_of_account": sizing.get("pct_of_account", 0),
            }
            send_trade_alert(alert_data, "BUY")
        except Exception as e:
            logger.debug("[알림] 매수 알림 실패: %s", e)

    def _send_sell_alert(
        self, pos: LivePosition, order: Order, reason: ExitReason, quantity: int,
    ) -> None:
        """매도 알림 (텔레그램)"""
        try:
            from src.telegram_sender import send_trade_alert
            alert_data = {
                "ticker": pos.ticker,
                "name": pos.name,
                "action": "SELL",
                "price": pos.current_price,
                "shares": quantity,
                "entry_price": pos.entry_price,
                "pnl_pct": pos.unrealized_pnl_pct,
                "exit_reason": reason.value,
                "grade": pos.grade,
            }
            send_trade_alert(alert_data, "SELL")
        except Exception as e:
            logger.debug("[알림] 매도 알림 실패: %s", e)


def create_live_engine(config_path: str = "config/settings.yaml") -> LiveTradingEngine:
    """설정 파일로부터 LiveTradingEngine 인스턴스 생성"""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    adapter = KisOrderAdapter()
    tracker = PositionTracker(config)
    guard = SafetyGuard(config)

    return LiveTradingEngine(
        order_port=adapter,
        balance_port=adapter,
        price_port=adapter,
        tracker=tracker,
        guard=guard,
        config=config,
    )
