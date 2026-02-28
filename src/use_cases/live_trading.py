"""
v4.0 라이브 트레이딩 엔진 — 시그널→주문→모니터링 루프

기존 backtest_engine + signal_engine + position_sizer 로직을 라이브 환경으로 통합.
v4.2: 손절 후 Shakeout 사후 판별 알림 (손절 자체는 절대 막지 않음)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

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
        self.cash_reserve_pct = pos_cfg.get("cash_reserve_pct", 0.20)

        # 분할매수 비율 (서보성 원칙)
        split_cfg = pos_cfg.get("split_buy", {})
        self.split_buy_1st = split_cfg.get("entry_1st", 0.50)
        self.split_buy_2nd = split_cfg.get("entry_2nd", 0.30)
        self.split_buy_3rd = split_cfg.get("entry_3rd", 0.20)

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

        # 2. 현금 유보 적용 (서보성 원칙: 20% 유보)
        reserved = balance * self.cash_reserve_pct
        usable_cash = balance - reserved
        if usable_cash <= 0:
            logger.info("[매수] 현금 유보 후 가용 잔고 0원 (유보 %.0f%%)", self.cash_reserve_pct * 100)
            return results
        logger.info("[매수] 잔고 %s원, 유보 %s원(%.0f%%), 가용 %s원",
                    f"{balance:,.0f}", f"{reserved:,.0f}",
                    self.cash_reserve_pct * 100, f"{usable_cash:,.0f}")

        # 3. 포지션 수 체크 (KOSPI 레짐 캡 적용)
        current_count = len(self.tracker.positions)
        regime_slots = self._get_kospi_regime_slots()
        effective_max = min(self.max_positions, regime_slots)
        available_slots = effective_max - current_count
        if available_slots <= 0:
            logger.info("[매수] 최대 포지션 도달 (%d/%d, 레짐 %d슬롯)",
                        current_count, effective_max, regime_slots)
            return results

        # 4. 시그널 정렬 (등급 A→C, zone_score 높은 순)
        grade_order = {"A": 0, "B": 1, "C": 2, "F": 3}
        sorted_signals = sorted(
            [s for s in signals if s.get("signal", False)],
            key=lambda s: (grade_order.get(s.get("grade", "F"), 3), -s.get("zone_score", 0)),
        )

        # 5. 매수 실행 (가용 슬롯만큼)
        for sig in sorted_signals[:available_slots]:
            ticker = sig.get("ticker", "")
            if not ticker:
                continue

            # 이미 보유 중인 종목 스킵
            if any(p.ticker == ticker for p in self.tracker.positions):
                logger.info("[매수] %s 이미 보유 중 — 스킵", ticker)
                continue

            result = self._execute_single_buy(sig, usable_cash)
            results.append(result)

            if result.get("success"):
                available_slots -= 1
                usable_cash -= result.get("investment", 0)

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

        # 포지션 사이징 (서보성 원칙: 1차 진입 비율 적용)
        grade_ratio = {"A": 1.0, "B": 0.67, "C": 0.33}.get(signal.get("grade", "C"), 0.33)
        stage_pct = self.split_buy_1st  # 1차 진입: 배정 비중의 50%

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
        """체결 대기 (timeout초 후 미체결 시 취소, API 오류 시 재시도)"""
        timeout = timeout or self.cancel_after_sec
        start = time.time()
        api_errors = 0

        while time.time() - start < timeout:
            try:
                status = self.order_port.get_order_status(order.order_id)
                api_errors = 0  # 성공 시 리셋

                if status.status == OrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = status.filled_quantity or order.quantity
                    order.filled_price = status.filled_price or order.price
                    return order

                if status.status == OrderStatus.FAILED:
                    logger.warning("[대기] %s 상태조회 FAILED 응답 — 재시도", order.ticker)
            except Exception as e:
                api_errors += 1
                logger.warning(
                    "[대기] %s 상태조회 오류 (%d회): %s", order.ticker, api_errors, e,
                )
                if api_errors >= 5:
                    logger.error("[대기] %s API 오류 5회 연속 — 취소 시도", order.ticker)
                    break

            time.sleep(5)

        # 타임아웃 또는 API 오류 → 취소
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

            # 손절 매도 성공 시 → Shakeout 사후 판별 알림
            if (
                result.get("success")
                and reason in (ExitReason.STOP_LOSS, ExitReason.PCT_STOP)
            ):
                self._shakeout_post_sell_alert(pos)

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

    # ──────────────────────────────────────────
    # Shakeout 사후 판별 (손절은 이미 실행된 후)
    # ──────────────────────────────────────────

    def _shakeout_post_sell_alert(self, pos: LivePosition) -> None:
        """
        손절 매도 완료 후 shakeout 가능성을 사후 판별하여 텔레그램 알림.
        손절 자체는 절대 막지 않음 — 정보 제공 목적.
        """
        try:
            from src.detect_shakeout import detect_shakeout

            result = detect_shakeout(
                ticker=pos.ticker,
                current_price=pos.current_price,
                entry_price=pos.entry_price,
            )

            # SHAKEOUT 또는 UNCERTAIN일 때만 알림
            if result.verdict in ("SHAKEOUT", "UNCERTAIN"):
                drop_pct = (pos.current_price / pos.entry_price - 1) * 100
                alert_text = result.to_alert_text(pos.ticker, pos.name, drop_pct)
                # 사후 판별임을 명시
                alert_text += "\n\n⚠️ 손절은 이미 실행됨. 3일 내 회복 시 재진입 검토."

                try:
                    from src.telegram_sender import send_message
                    send_message(alert_text)
                except Exception as e:
                    logger.debug("[알림] Shakeout 사후 알림 실패: %s", e)

                logger.info(
                    "[Shakeout] %s 사후 판별: %s (%d/4점) — 재진입 모니터링 권장",
                    pos.ticker, result.verdict, result.score,
                )

        except Exception as e:
            logger.debug("[Shakeout] %s 사후 판별 실패: %s", pos.ticker, e)


    # ──────────────────────────────────────────
    # KOSPI 레짐 기반 슬롯 캡
    # ──────────────────────────────────────────

    def _get_kospi_regime_slots(self) -> int:
        """KOSPI 레짐 기반 최대 슬롯 수 반환.

        BULL=5, CAUTION=3, BEAR=2, CRISIS=0.
        데이터 없거나 계산 실패 시 settings의 max_positions 반환.
        """
        try:
            import pandas as pd

            kospi_path = Path("data/kospi_index.csv")
            if not kospi_path.exists():
                logger.warning("[레짐] kospi_index.csv 없음 → 기본값 %d", self.max_positions)
                return self.max_positions

            df = pd.read_csv(kospi_path, parse_dates=["Date"])
            df = df.sort_values("Date").tail(252)
            if len(df) < 60:
                return self.max_positions

            close = df["close"].iloc[-1]
            ma20 = df["close"].tail(20).mean()
            ma60 = df["close"].tail(60).mean()

            # RV20 백분위
            returns = df["close"].pct_change().dropna()
            rv20 = returns.tail(20).std()
            rv_series = returns.rolling(20).std().dropna()
            rv_pct = (rv_series < rv20).mean() if len(rv_series) > 0 else 0.5

            if ma20 == 0 or ma60 == 0:
                regime, slots = "CAUTION", 3
            elif close > ma20:
                regime, slots = ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
            elif close > ma60:
                regime, slots = "BEAR", 2
            else:
                regime, slots = "CRISIS", 0

            logger.info("[레짐] KOSPI %.0f (MA20=%.0f, MA60=%.0f, RV%%ile=%.0f%%) → %s(%d슬롯)",
                        close, ma20, ma60, rv_pct * 100, regime, slots)
            return slots
        except Exception as e:
            logger.warning("[레짐] KOSPI 레짐 계산 실패: %s → 기본값 %d", e, self.max_positions)
            return self.max_positions


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
