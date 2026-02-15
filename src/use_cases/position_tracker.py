"""
v4.1 포지션 트래커 — 보유종목 관리 + PnL 계산 + 적응형 청산

v4.0 → v4.1 변경:
  - 적응형 청산 통합 (건강한 조정 vs 위험한 조정 판별)
  - 일일 보유 점수 통합 (HoldScore → 손절/보유일 동적 조정)
  - check_exit_conditions_adaptive() 추가 (DataFrame 기반 강화 판정)

backtest_engine.py의 4단계 부분청산/손절/트레일링 로직을 라이브용으로 재구현.
data/positions.json에 영속 저장.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.entities.trading_models import (
    ExitReason,
    LivePosition,
    Order,
    OrderStatus,
)
from src.use_cases.adaptive_exit import AdaptiveExitManager
from src.use_cases.daily_hold_scorer import DailyHoldScorer
from src.use_cases.ports import CurrentPricePort

logger = logging.getLogger(__name__)

POSITIONS_FILE = Path("data/positions.json")
TRADES_FILE = Path("data/trades_history.json")
PERFORMANCE_FILE = Path("data/daily_performance.json")

# 4단계 부분청산 R배수 (backtest_engine과 동일)
PARTIAL_EXIT_R = [2, 4, 8, 10]
PARTIAL_EXIT_PCT = 0.25


class PositionTracker:
    """보유 포지션 관리 + 청산 조건 판정"""

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        self.positions: list[LivePosition] = []
        self.config = config or {}
        exit_cfg = self.config.get("quant_engine", {}).get("exit", {})
        self.partial_exit_r = exit_cfg.get("partial_exit_r", PARTIAL_EXIT_R)
        self.partial_exit_pct = exit_cfg.get("partial_exit_pct", PARTIAL_EXIT_PCT)
        self.max_hold_days = exit_cfg.get("max_hold_days", 20)
        self.atr_stop_mult = self.config.get("backtest", {}).get(
            "trailing_stop_atr_mult", 2.0
        )

        # v4.1: 적응형 청산
        adaptive_cfg = self.config.get("adaptive_exit", {})
        self.adaptive_exit_enabled = adaptive_cfg.get("enabled", False)
        if self.adaptive_exit_enabled:
            self.adaptive_exit = AdaptiveExitManager(config_path)
            self.hold_scorer = DailyHoldScorer(config_path)
        else:
            self.adaptive_exit = None
            self.hold_scorer = None

        self.load_positions()

    # ──────────────────────────────────────────
    # 영속화
    # ──────────────────────────────────────────

    def load_positions(self) -> None:
        """data/positions.json 에서 로드"""
        if POSITIONS_FILE.exists():
            try:
                raw = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
                self.positions = [LivePosition.from_dict(d) for d in raw]
                logger.info("[포지션] %d개 포지션 로드 완료", len(self.positions))
            except Exception as e:
                logger.error("[포지션] 로드 실패: %s", e)
                self.positions = []
        else:
            self.positions = []

    def save_positions(self) -> None:
        """data/positions.json 에 저장"""
        POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self.positions]
        POSITIONS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ──────────────────────────────────────────
    # 포지션 추가/제거
    # ──────────────────────────────────────────

    def add_position(self, order: Order, signal: dict) -> LivePosition | None:
        """체결된 주문 + 시그널 정보로 신규 포지션 등록"""
        if order.status not in (OrderStatus.FILLED, OrderStatus.PENDING):
            logger.warning("[포지션] 주문 미체결 — 포지션 등록 건너뜀: %s", order.ticker)
            return None

        price = order.filled_price if order.filled_price > 0 else order.price
        shares = order.filled_quantity if order.filled_quantity > 0 else order.quantity

        pos = LivePosition(
            ticker=order.ticker,
            name=signal.get("name", ""),
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            entry_price=float(price),
            shares=shares,
            current_price=float(price),
            stop_loss=float(signal.get("stop_loss", price * 0.95)),
            target_price=float(signal.get("target_price", price * 1.10)),
            atr_value=float(signal.get("atr_value", 0)),
            grade=signal.get("grade", "C"),
            trigger_type=signal.get("trigger_type", "confirm"),
            stop_loss_pct=float(signal.get("stop_loss_pct", 0.05)),
            highest_price=float(price),
            trailing_stop=float(price - signal.get("atr_value", 0) * self.atr_stop_mult),
            partial_exits_done=0,
            initial_shares=shares,
            news_grade=signal.get("news_grade", ""),
            max_hold_days=signal.get("max_hold_days", self.max_hold_days),
        )

        self.positions.append(pos)
        self.save_positions()
        logger.info(
            "[포지션] 등록: %s %d주 @ %.0f원 (등급=%s, 트리거=%s)",
            pos.ticker, pos.shares, pos.entry_price, pos.grade, pos.trigger_type,
        )
        return pos

    def remove_position(self, ticker: str) -> LivePosition | None:
        """포지션 완전 제거"""
        for i, p in enumerate(self.positions):
            if p.ticker == ticker:
                removed = self.positions.pop(i)
                self.save_positions()
                return removed
        return None

    # ──────────────────────────────────────────
    # 현재가 갱신
    # ──────────────────────────────────────────

    def update_prices(self, price_port: CurrentPricePort) -> None:
        """모든 포지션의 현재가를 갱신"""
        for pos in self.positions:
            try:
                data = price_port.fetch_current_price(pos.ticker)
                price = data.get("current_price", 0)
                if price > 0:
                    pos.current_price = float(price)
                    high = data.get("high", 0)
                    if high > pos.highest_price:
                        pos.highest_price = float(high)
                        pos.trailing_stop = pos.highest_price - pos.atr_value * self.atr_stop_mult
            except Exception as e:
                logger.debug("[포지션] %s 현재가 갱신 실패: %s", pos.ticker, e)
        self.save_positions()

    # ──────────────────────────────────────────
    # 청산 조건 판정 (backtest_engine 로직 재활용)
    # ──────────────────────────────────────────

    def check_exit_conditions(self) -> list[tuple[LivePosition, ExitReason, int]]:
        """
        모든 포지션의 청산 조건 체크.

        Returns:
            [(position, exit_reason, sell_quantity), ...]
        """
        exits = []
        today = datetime.now().strftime("%Y-%m-%d")

        for pos in list(self.positions):
            price = pos.current_price
            if price <= 0:
                continue

            # 0. 최대 보유일 체크
            try:
                hold_days = (
                    datetime.strptime(today, "%Y-%m-%d")
                    - datetime.strptime(pos.entry_date, "%Y-%m-%d")
                ).days
            except Exception:
                hold_days = 0

            if hold_days >= pos.max_hold_days:
                exits.append((pos, ExitReason.MAX_HOLD, pos.shares))
                continue

            # 1. 절대 손절선
            if price <= pos.stop_loss:
                exits.append((pos, ExitReason.STOP_LOSS, pos.shares))
                continue

            # 2. 퍼센트 손절
            pct_loss = price / pos.entry_price - 1
            if pct_loss <= -pos.stop_loss_pct:
                exits.append((pos, ExitReason.PCT_STOP, pos.shares))
                continue

            # 3. 4단계 부분청산
            if pos.partial_exits_done < len(self.partial_exit_r):
                risk_per_share = pos.entry_price - pos.stop_loss
                if risk_per_share > 0:
                    next_r = self.partial_exit_r[pos.partial_exits_done]
                    target_r_price = pos.entry_price + risk_per_share * next_r
                    if price >= target_r_price:
                        exit_shares = max(1, int(pos.initial_shares * self.partial_exit_pct))
                        exit_shares = min(exit_shares, pos.shares)
                        if exit_shares > 0:
                            reason_map = {
                                2: ExitReason.PARTIAL_2R,
                                4: ExitReason.PARTIAL_4R,
                                8: ExitReason.PARTIAL_8R,
                                10: ExitReason.PARTIAL_10R,
                            }
                            exits.append((pos, reason_map.get(next_r, ExitReason.TRAILING_STOP), exit_shares))
                            # 부분청산 후 수량/카운터 갱신은 실제 체결 후 처리
                            continue

            # 4. 트레일링 스탑 (부분청산 시작 후)
            if pos.partial_exits_done > 0 and price <= pos.trailing_stop:
                exits.append((pos, ExitReason.TRAILING_STOP, pos.shares))
                continue

        return exits

    def check_exit_conditions_adaptive(
        self, data_dict: dict[str, pd.DataFrame],
    ) -> list[tuple[LivePosition, ExitReason, int, dict]]:
        """
        v4.1 적응형 청산 판정 (DataFrame 기반).

        각 포지션에 대해 조정 건강도 + 일일 보유 점수를 평가하여
        동적으로 손절/보유/청산을 결정.

        Args:
            data_dict: {ticker: DataFrame} — 기술적 지표 포함

        Returns:
            [(position, exit_reason, sell_quantity, metadata), ...]
            metadata: {"health": PullbackHealth, "hold": HoldScore, ...}
        """
        if not self.adaptive_exit_enabled:
            # 적응형 비활성 → 기존 로직 폴백
            basic = self.check_exit_conditions()
            return [(pos, reason, qty, {}) for pos, reason, qty in basic]

        exits = []
        today = datetime.now().strftime("%Y-%m-%d")

        for pos in list(self.positions):
            price = pos.current_price
            if price <= 0:
                continue

            df = data_dict.get(pos.ticker)
            if df is None or df.empty:
                # DataFrame 없으면 기존 로직
                basic_exits = self._check_basic_exits(pos, today)
                exits.extend(basic_exits)
                continue

            idx = len(df) - 1

            # 보유일 계산
            try:
                hold_days = (
                    datetime.strptime(today, "%Y-%m-%d")
                    - datetime.strptime(pos.entry_date, "%Y-%m-%d")
                ).days
            except Exception:
                hold_days = 0

            # ── 1. 일일 보유 점수 ──
            hold_result = None
            effective_max_hold = pos.max_hold_days
            if self.hold_scorer and hold_days >= 2:
                hold_result = self.hold_scorer.score(
                    df, idx, pos.entry_price, pos.highest_price,
                    trigger_type=pos.trigger_type,
                    news_grade=pos.news_grade,
                    hold_days=hold_days,
                )
                effective_max_hold += hold_result.hold_days_adjustment

                if hold_result.action == "exit":
                    exits.append((pos, ExitReason.PCT_STOP, pos.shares, {
                        "hold_score": hold_result,
                        "reason": "hold_score_exit",
                    }))
                    continue

            # ── 2. 최대 보유일 ──
            if hold_days >= effective_max_hold:
                exits.append((pos, ExitReason.MAX_HOLD, pos.shares, {
                    "hold_score": hold_result,
                }))
                continue

            # ── 3. 절대 손절 ──
            if price <= pos.stop_loss:
                exits.append((pos, ExitReason.STOP_LOSS, pos.shares, {}))
                continue

            # ── 4. 적응형 퍼센트 손절 ──
            pct_loss = price / pos.entry_price - 1
            if self.adaptive_exit:
                health = self.adaptive_exit.evaluate_pullback(
                    df, idx, pos.entry_price, pos.highest_price, pos.trigger_type,
                )

                if health.classification == "critical":
                    exits.append((pos, ExitReason.PCT_STOP, pos.shares, {
                        "health": health, "reason": "adaptive_critical",
                    }))
                    continue
                elif health.classification == "dangerous":
                    if pct_loss <= -health.adjusted_stop_pct:
                        exits.append((pos, ExitReason.PCT_STOP, pos.shares, {
                            "health": health, "reason": "adaptive_dangerous",
                        }))
                        continue
                elif health.classification == "healthy":
                    if health.adjusted_stop_price > 0 and price <= health.adjusted_stop_price:
                        exits.append((pos, ExitReason.PCT_STOP, pos.shares, {
                            "health": health, "reason": "adaptive_healthy_stop",
                        }))
                        continue
                    # 건강한 조정이면 기존 고정 손절 무시 (MA20 기반으로 대체)
                else:
                    # caution → 기존 고정 손절
                    if pct_loss <= -pos.stop_loss_pct:
                        exits.append((pos, ExitReason.PCT_STOP, pos.shares, {
                            "health": health, "reason": "pct_stop_caution",
                        }))
                        continue

            # ── 5. 부분청산 ──
            if pos.partial_exits_done < len(self.partial_exit_r):
                risk_per_share = pos.entry_price - pos.stop_loss
                if risk_per_share > 0:
                    next_r = self.partial_exit_r[pos.partial_exits_done]
                    target_r_price = pos.entry_price + risk_per_share * next_r
                    if price >= target_r_price:
                        exit_shares = max(1, int(pos.initial_shares * self.partial_exit_pct))
                        exit_shares = min(exit_shares, pos.shares)
                        reason_map = {
                            2: ExitReason.PARTIAL_2R, 4: ExitReason.PARTIAL_4R,
                            8: ExitReason.PARTIAL_8R, 10: ExitReason.PARTIAL_10R,
                        }
                        exits.append((pos, reason_map.get(next_r, ExitReason.TRAILING_STOP), exit_shares, {
                            "hold_score": hold_result,
                        }))
                        continue

            # ── 6. 트레일링 스탑 ──
            if pos.partial_exits_done > 0 and price <= pos.trailing_stop:
                exits.append((pos, ExitReason.TRAILING_STOP, pos.shares, {}))
                continue

        return exits

    def _check_basic_exits(
        self, pos: LivePosition, today: str,
    ) -> list[tuple[LivePosition, ExitReason, int, dict]]:
        """DataFrame 없을 때 기본 청산 체크"""
        exits = []
        price = pos.current_price
        try:
            hold_days = (
                datetime.strptime(today, "%Y-%m-%d")
                - datetime.strptime(pos.entry_date, "%Y-%m-%d")
            ).days
        except Exception:
            hold_days = 0

        if hold_days >= pos.max_hold_days:
            exits.append((pos, ExitReason.MAX_HOLD, pos.shares, {}))
        elif price <= pos.stop_loss:
            exits.append((pos, ExitReason.STOP_LOSS, pos.shares, {}))
        elif (price / pos.entry_price - 1) <= -pos.stop_loss_pct:
            exits.append((pos, ExitReason.PCT_STOP, pos.shares, {}))
        return exits

    def apply_partial_exit(self, pos: LivePosition, sold_shares: int, reason: ExitReason) -> None:
        """부분청산 체결 후 포지션 업데이트"""
        pos.shares -= sold_shares
        if reason.value.startswith("partial_"):
            pos.partial_exits_done += 1

        if pos.shares <= 0:
            self.remove_position(pos.ticker)
        else:
            self.save_positions()

        # 체결 이력 기록
        self._log_trade(pos, sold_shares, reason)

    # ──────────────────────────────────────────
    # 요약/조회
    # ──────────────────────────────────────────

    def get_summary(self) -> dict:
        """전체 포지션 요약"""
        total_investment = sum(p.investment for p in self.positions)
        total_eval = sum(p.current_price * p.shares for p in self.positions)
        total_pnl = total_eval - total_investment

        return {
            "count": len(self.positions),
            "total_investment": int(total_investment),
            "total_eval": int(total_eval),
            "total_pnl": int(total_pnl),
            "total_pnl_pct": round(total_pnl / total_investment * 100, 2) if total_investment > 0 else 0.0,
            "positions": [
                {
                    "ticker": p.ticker,
                    "name": p.name,
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "pnl_pct": round(p.unrealized_pnl_pct, 2),
                    "grade": p.grade,
                    "trigger": p.trigger_type,
                }
                for p in self.positions
            ],
        }

    def sync_with_broker(self, holdings: list[dict]) -> None:
        """한투 실잔고와 동기화 (불일치 감지)"""
        broker_map = {h["ticker"]: h for h in holdings}
        for pos in self.positions:
            if pos.ticker in broker_map:
                broker_qty = broker_map[pos.ticker]["quantity"]
                if broker_qty != pos.shares:
                    logger.warning(
                        "[동기화] %s 수량 불일치: 트래커=%d, 실잔고=%d",
                        pos.ticker, pos.shares, broker_qty,
                    )
                    pos.shares = broker_qty
            else:
                logger.warning(
                    "[동기화] %s 트래커에만 존재 (실잔고 없음) — 제거",
                    pos.ticker,
                )
        # 트래커에 없는 종목 경고
        tracked = {p.ticker for p in self.positions}
        for ticker in broker_map:
            if ticker not in tracked:
                logger.warning(
                    "[동기화] %s 실잔고에만 존재 (트래커 미등록)",
                    ticker,
                )
        self.save_positions()

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _log_trade(self, pos: LivePosition, shares: int, reason: ExitReason) -> None:
        """체결 이력을 trades_history.json에 기록"""
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if TRADES_FILE.exists():
            try:
                history = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
            except Exception:
                history = []

        trade = {
            "ticker": pos.ticker,
            "name": pos.name,
            "entry_date": pos.entry_date,
            "exit_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "entry_price": pos.entry_price,
            "exit_price": pos.current_price,
            "shares": shares,
            "pnl": (pos.current_price - pos.entry_price) * shares,
            "pnl_pct": round(pos.unrealized_pnl_pct, 2),
            "exit_reason": reason.value,
            "grade": pos.grade,
            "trigger_type": pos.trigger_type,
        }
        history.append(trade)
        TRADES_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
