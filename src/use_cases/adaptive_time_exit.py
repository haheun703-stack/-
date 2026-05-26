"""시간 기반 매도 — H8 (D+3 익절) + H9 (D+5 데드라인). 5/26 PDCA flexible-pullback-buy.

배경 (5/14 백테스트, 메모리):
- 3단계 시그널: D+1 +2.02% / D+3 +4.00% / **D+5 -0.35% (음수 전환)**
- 즉 D+3가 평균 peak, D+5에서 손실 시작
- 현행 자동 매도 시스템: MVP-2.5 trailing (+7%) / MVP-2.6 손절 (-5%) / MVP-1 천장 -3%
- **시간 차원 매도가 누락** — 백테스트 D+3 peak를 코드로 활용 못 함

룰:
- H8 D+3 시간 익절: 매수일 + 3 거래일 경과 + 현재 수익률 > 0% → 즉시 매도
  (현재 수익률 ≤ 0%이면 D+5까지 대기 — 손실 확정 회피)
- H9 D+5 데드라인: 매수일 + 5 거래일 경과 → 손익 무관 시장가 매도 (음수 진입 회피)

이 모듈은 adaptive_buy_queue.execute_auto_buy() 후 매 사이클 (run_adaptive_cycle 30분)에서 호출.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# 환경변수 — 백테스트 후 조정 가능
D3_PROFIT_THRESHOLD_PCT = float(os.getenv("ADAPTIVE_D3_PROFIT_PCT", "0.0"))  # D+3에서 +X% 이상이면 익절
D5_DEADLINE_DAYS = int(os.getenv("ADAPTIVE_D5_DEADLINE_DAYS", "5"))
D3_EXIT_DAYS = int(os.getenv("ADAPTIVE_D3_EXIT_DAYS", "3"))
TIME_EXIT_ENABLED = os.getenv("ADAPTIVE_TIME_EXIT_ENABLED", "0") == "1"


@dataclass
class TimeExitSignal:
    """시간 기반 매도 신호."""
    ticker: str
    triggered: bool                  # 매도 신호 발생 여부
    exit_type: str                   # 'D3_PROFIT' / 'D5_DEADLINE' / 'WAIT' / 'PARSE_ERROR'
    trade_days_elapsed: int          # 매수 후 경과 거래일
    entry_price: int
    current_price: int
    pnl_pct: float                   # 현재 손익률 %
    qty: int                         # 매도 예정 수량
    reason: str                      # 사유 텍스트


def _count_trading_days(start: date, end: date) -> int:
    """start ~ end 사이 거래일 수 (KRX). end는 포함, start는 미포함.

    예: start=금요일(D+0), end=다음주 월요일 → 1 거래일 (D+1)
       start=금요일, end=다음주 수요일 → 3 거래일 (D+3)

    src/trading_calendar.is_kr_trading_day 활용 — 공휴일/주말 자동 제외.
    """
    try:
        from src.trading_calendar import is_kr_trading_day
    except ImportError:
        # fallback: 단순 주말만 제외 (공휴일 미반영)
        def is_kr_trading_day(d):
            return d.weekday() < 5

    if end <= start:
        return 0

    count = 0
    cur = start + timedelta(days=1)
    while cur <= end:
        if is_kr_trading_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count


def evaluate_time_exit(
    ticker: str,
    triggered_at_iso: str,
    entry_price: int,
    current_price: int,
    qty: int,
    now: Optional[datetime] = None,
    d3_threshold_pct: float = D3_PROFIT_THRESHOLD_PCT,
    d3_days: int = D3_EXIT_DAYS,
    d5_days: int = D5_DEADLINE_DAYS,
) -> TimeExitSignal:
    """시간 매도 평가.

    Args:
        ticker: 종목코드
        triggered_at_iso: 매수 시각 ISO 문자열 (e.g. "2026-05-26T10:30:05")
        entry_price: 매수 단가
        current_price: 현재가
        qty: 보유 수량
        now: 평가 시각 (None이면 datetime.now())
        d3_threshold_pct: D+3 익절 임계 (기본 0.0% — 수익이면 익절)
        d3_days: D+3 거래일 수 (기본 3)
        d5_days: D+5 거래일 수 (기본 5)
    """
    if not triggered_at_iso:
        return TimeExitSignal(
            ticker=ticker, triggered=False, exit_type="PARSE_ERROR",
            trade_days_elapsed=0, entry_price=entry_price,
            current_price=current_price, pnl_pct=0.0, qty=qty,
            reason="triggered_at 없음",
        )

    try:
        trig_dt = datetime.fromisoformat(triggered_at_iso)
    except (ValueError, TypeError) as e:
        return TimeExitSignal(
            ticker=ticker, triggered=False, exit_type="PARSE_ERROR",
            trade_days_elapsed=0, entry_price=entry_price,
            current_price=current_price, pnl_pct=0.0, qty=qty,
            reason=f"triggered_at 파싱 실패: {e}",
        )

    now = now or datetime.now()
    elapsed = _count_trading_days(trig_dt.date(), now.date())

    pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0

    # H9 D+5 데드라인 — 최우선 (손익 무관 강제 매도)
    if elapsed >= d5_days:
        return TimeExitSignal(
            ticker=ticker, triggered=True, exit_type="D5_DEADLINE",
            trade_days_elapsed=elapsed, entry_price=entry_price,
            current_price=current_price, pnl_pct=pnl_pct, qty=qty,
            reason=f"D+{elapsed} 데드라인 (백테스트 D+5 -0.35% 음수 회피)",
        )

    # H8 D+3 익절 — 수익 시만
    if elapsed >= d3_days and pnl_pct > d3_threshold_pct:
        return TimeExitSignal(
            ticker=ticker, triggered=True, exit_type="D3_PROFIT",
            trade_days_elapsed=elapsed, entry_price=entry_price,
            current_price=current_price, pnl_pct=pnl_pct, qty=qty,
            reason=f"D+{elapsed} 익절 (수익 {pnl_pct:+.2f}% > {d3_threshold_pct:.1f}%, 백테스트 D+3 +4.00% peak)",
        )

    return TimeExitSignal(
        ticker=ticker, triggered=False, exit_type="WAIT",
        trade_days_elapsed=elapsed, entry_price=entry_price,
        current_price=current_price, pnl_pct=pnl_pct, qty=qty,
        reason=f"D+{elapsed} 대기 (D+3 익절 임계 {d3_threshold_pct:.1f}% 미달 또는 손실)",
    )


def scan_queue_for_time_exits(queue_state: dict, broker, now: Optional[datetime] = None) -> list[TimeExitSignal]:
    """전체 큐 스캔 — TRIGGERED 상태 단계들의 시간 매도 평가.

    Args:
        queue_state: adaptive_buy_queue.json 내용 dict
        broker: 현재가 조회용 (broker.fetch_price)
        now: 평가 시각

    Returns:
        매도 신호 발생한 TimeExitSignal 리스트.
    """
    signals: list[TimeExitSignal] = []
    if not TIME_EXIT_ENABLED:
        logger.debug("[H8/H9] ADAPTIVE_TIME_EXIT_ENABLED=0 — 비활성")
        return signals

    queues = queue_state.get("queues", {})
    for ticker, entry in queues.items():
        for stage in entry.get("stages", []):
            if stage.get("status") != "TRIGGERED":
                continue
            actual_price = int(stage.get("actual_price", 0))
            actual_qty = int(stage.get("actual_qty", 0))
            triggered_at = stage.get("triggered_at")
            if actual_price <= 0 or actual_qty <= 0 or not triggered_at:
                continue

            # 현재가 조회
            try:
                res = broker.fetch_price(ticker)
                out = res.get("output", {}) if res else {}
                cur = int(out.get("stck_prpr", 0))
            except Exception as e:
                logger.warning("[H8/H9] %s fetch_price 실패: %s", ticker, e)
                continue
            if cur <= 0:
                continue

            sig = evaluate_time_exit(
                ticker=ticker,
                triggered_at_iso=triggered_at,
                entry_price=actual_price,
                current_price=cur,
                qty=actual_qty,
                now=now,
            )
            if sig.triggered:
                signals.append(sig)
                logger.info(
                    "[H8/H9] %s 시간 매도: %s (D+%d, %+.2f%%) %s",
                    ticker, sig.exit_type, sig.trade_days_elapsed,
                    sig.pnl_pct, sig.reason,
                )
    return signals


def execute_time_exit(broker, sig: TimeExitSignal) -> dict:
    """시간 매도 실행.

    5/26 변경: 시장가 → 지정가 우선 (시장가 슬리피지 회피, 사용자 지시).
    - D+3 익절: 지정가 -0.3% (체결 우선)
    - D+5 데드라인: 지정가 -0.5% (강제 매도이지만 슬리피지 최소화)
    - 환경변수 ADAPTIVE_SELL_USE_LIMIT=0이면 시장가 fallback.

    Returns:
        {"success": bool, "order_id": str, "error": str, "limit_price": int}
    """
    if not sig.triggered:
        return {"success": False, "error": "trigger=False"}

    use_limit = os.getenv("ADAPTIVE_SELL_USE_LIMIT", "1") == "1"
    # D+5 강제는 더 큰 slippage 허용 (체결 우선)
    if sig.exit_type == "D5_DEADLINE":
        sell_slippage_pct = float(os.getenv("ADAPTIVE_SELL_LIMIT_SLIPPAGE_D5_PCT", "0.5"))
    else:
        sell_slippage_pct = float(os.getenv("ADAPTIVE_SELL_LIMIT_SLIPPAGE_PCT", "0.3"))

    try:
        if use_limit and hasattr(broker, "sell_limit") and sig.current_price > 0:
            limit_price = int(sig.current_price * (1 - sell_slippage_pct / 100))
            order = broker.sell_limit(sig.ticker, limit_price, sig.qty)
            logger.info(
                "[H8/H9] 지정가 매도 %s %s %d주 @ %d (현재 %d, slippage -%.1f%%)",
                sig.ticker, sig.exit_type, sig.qty, limit_price, sig.current_price, sell_slippage_pct,
            )
            return {
                "success": True,
                "order_id": getattr(order, "order_id", ""),
                "qty": sig.qty,
                "exit_type": sig.exit_type,
                "limit_price": limit_price,
                "pnl_pct": sig.pnl_pct,
            }
        elif hasattr(broker, "sell_market"):
            order = broker.sell_market(sig.ticker, sig.qty)
            return {
                "success": True,
                "order_id": getattr(order, "order_id", ""),
                "qty": sig.qty,
                "exit_type": sig.exit_type,
                "pnl_pct": sig.pnl_pct,
            }
        else:
            # fallback: 지정가 매도
            limit_price = int(sig.current_price * (1 - sell_slippage_pct / 100))
            order = broker.sell_limit(sig.ticker, limit_price, sig.qty)
            return {
                "success": True,
                "order_id": getattr(order, "order_id", ""),
                "qty": sig.qty,
                "exit_type": sig.exit_type,
                "limit_price": limit_price,
            }
    except Exception as e:
        logger.error("[H8/H9] %s 시간 매도 실행 실패: %s", sig.ticker, e)
        return {"success": False, "error": str(e)}
