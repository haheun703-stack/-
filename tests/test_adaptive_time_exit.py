"""H8 D+3 익절 + H9 D+5 데드라인 단위 테스트 (5/26)."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.use_cases.adaptive_time_exit import (
    evaluate_time_exit,
    TimeExitSignal,
    _count_trading_days,
    scan_queue_for_time_exits,
    execute_time_exit,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# ─────────────────────────────────────────
# _count_trading_days
# ─────────────────────────────────────────
def test_trading_days_simple_weekday():
    """월요일 → 수요일 = 2 거래일."""
    from datetime import date
    # 2026-05-25 = 월, 2026-05-27 = 수
    n = _count_trading_days(date(2026, 5, 25), date(2026, 5, 27))
    assert n == 2


def test_trading_days_weekend_excluded():
    """금요일 → 다음 화요일 = 1 거래일 (토/일 + 5/25 부처님오신날 휴장 제외)."""
    from datetime import date
    # 2026-05-22 = 금, 2026-05-26 = 화 (5/25 월은 부처님오신날 휴장)
    n = _count_trading_days(date(2026, 5, 22), date(2026, 5, 26))
    assert n == 1  # 5/26만 거래일


def test_trading_days_same_day():
    """같은 날 → 0 거래일."""
    from datetime import date
    n = _count_trading_days(date(2026, 5, 26), date(2026, 5, 26))
    assert n == 0


# ─────────────────────────────────────────
# evaluate_time_exit
# ─────────────────────────────────────────
def test_d3_profit_exit_triggered():
    """D+3 도달 + 수익 → D3_PROFIT 매도."""
    trig = datetime(2026, 5, 21, 10, 0)  # 목
    now = datetime(2026, 5, 27, 15, 0)   # 다음주 수 (D+4 거래일)
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso=_iso(trig),
        entry_price=10000,
        current_price=10500,  # +5%
        qty=1,
        now=now,
    )
    assert sig.triggered
    assert sig.exit_type == "D3_PROFIT"
    assert sig.pnl_pct == 5.0
    assert sig.trade_days_elapsed >= 3


def test_d3_loss_no_exit():
    """D+3 도달 but 손실 → 대기 (D+5 까지)."""
    trig = datetime(2026, 5, 21, 10, 0)
    now = datetime(2026, 5, 26, 15, 0)  # D+3 거래일
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso=_iso(trig),
        entry_price=10000,
        current_price=9800,  # -2% 손실
        qty=1,
        now=now,
    )
    assert sig.triggered is False
    assert sig.exit_type == "WAIT"
    assert sig.pnl_pct == -2.0


def test_d5_deadline_force_exit():
    """D+5 도달 → 손익 무관 강제 매도. 5/25 부처님오신날 휴장 우회."""
    trig = datetime(2026, 5, 19, 10, 0)  # 화
    now = datetime(2026, 5, 28, 15, 0)   # 다음주 목 (5/25 휴장 + 주말 제외 시 D+5)
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso=_iso(trig),
        entry_price=10000,
        current_price=9500,
        qty=1,
        now=now,
    )
    assert sig.triggered
    assert sig.exit_type == "D5_DEADLINE"
    assert sig.trade_days_elapsed >= 5


def test_d5_deadline_with_profit():
    """D+5 + 수익 → 강제 매도 (D3 익절 우선이지만 D5는 무조건)."""
    trig = datetime(2026, 5, 19, 10, 0)
    now = datetime(2026, 5, 28, 15, 0)   # D+5 도달 (5/25 휴장 우회)
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso=_iso(trig),
        entry_price=10000,
        current_price=10800,
        qty=1,
        now=now,
    )
    assert sig.triggered
    assert sig.exit_type == "D5_DEADLINE"  # D+5 우선


def test_d1_no_exit():
    """D+1 → 매도 없음."""
    trig = datetime(2026, 5, 25, 10, 0)  # 월
    now = datetime(2026, 5, 26, 15, 0)   # 화 (D+1)
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso=_iso(trig),
        entry_price=10000,
        current_price=10500,
        qty=1,
        now=now,
    )
    assert sig.triggered is False
    assert sig.exit_type == "WAIT"
    assert sig.trade_days_elapsed == 1


def test_invalid_triggered_at():
    """파싱 실패 → PARSE_ERROR (matlab 없음)."""
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso="invalid_date",
        entry_price=10000,
        current_price=10500,
        qty=1,
    )
    assert sig.triggered is False
    assert sig.exit_type == "PARSE_ERROR"


def test_empty_triggered_at():
    """빈 문자열 → PARSE_ERROR."""
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso="",
        entry_price=10000,
        current_price=10500,
        qty=1,
    )
    assert sig.exit_type == "PARSE_ERROR"


def test_custom_d3_threshold():
    """D+3 익절 임계 +2%로 변경 시 +1% 수익은 매도 X."""
    trig = datetime(2026, 5, 21, 10, 0)
    now = datetime(2026, 5, 26, 15, 0)
    sig = evaluate_time_exit(
        ticker="067310",
        triggered_at_iso=_iso(trig),
        entry_price=10000,
        current_price=10100,  # +1% — 임계 +2% 미달
        qty=1,
        now=now,
        d3_threshold_pct=2.0,
    )
    assert sig.triggered is False
    assert sig.exit_type == "WAIT"


# ─────────────────────────────────────────
# scan_queue_for_time_exits
# ─────────────────────────────────────────
def test_scan_disabled_returns_empty(monkeypatch):
    """ADAPTIVE_TIME_EXIT_ENABLED=0 (기본) → 빈 리스트."""
    monkeypatch.setattr("src.use_cases.adaptive_time_exit.TIME_EXIT_ENABLED", False)
    queue = {"queues": {"067310": {"stages": []}}}
    broker = MagicMock()
    sigs = scan_queue_for_time_exits(queue, broker)
    assert sigs == []


def test_scan_only_triggered_stages(monkeypatch):
    """TRIGGERED 상태만 검사 (PENDING/FILLED/FAILED는 스킵)."""
    monkeypatch.setattr("src.use_cases.adaptive_time_exit.TIME_EXIT_ENABLED", True)
    trig = datetime(2026, 5, 19, 10, 0)
    queue = {
        "queues": {
            "067310": {
                "stages": [
                    {  # PENDING — 스킵
                        "status": "PENDING", "actual_price": 0, "actual_qty": 0,
                        "triggered_at": None,
                    },
                    {  # TRIGGERED + D+5 — 매도
                        "status": "TRIGGERED",
                        "actual_price": 10000, "actual_qty": 1,
                        "triggered_at": _iso(trig),
                    },
                ]
            }
        }
    }
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_prpr": "9500"}}
    now = datetime(2026, 5, 27, 15, 0)
    sigs = scan_queue_for_time_exits(queue, broker, now=now)
    assert len(sigs) == 1
    assert sigs[0].exit_type == "D5_DEADLINE"


def test_scan_fetch_price_failure_skipped(monkeypatch):
    """현재가 조회 실패 → 해당 stage 스킵."""
    monkeypatch.setattr("src.use_cases.adaptive_time_exit.TIME_EXIT_ENABLED", True)
    queue = {
        "queues": {
            "067310": {
                "stages": [{
                    "status": "TRIGGERED",
                    "actual_price": 10000, "actual_qty": 1,
                    "triggered_at": _iso(datetime(2026, 5, 19, 10, 0)),
                }]
            }
        }
    }
    broker = MagicMock()
    broker.fetch_price = MagicMock(side_effect=Exception("API timeout"))
    sigs = scan_queue_for_time_exits(queue, broker)
    assert sigs == []


# ─────────────────────────────────────────
# execute_time_exit
# ─────────────────────────────────────────
def test_execute_sell_market_success():
    """sell_market 성공 → success=True."""
    sig = TimeExitSignal(
        ticker="067310", triggered=True, exit_type="D5_DEADLINE",
        trade_days_elapsed=5, entry_price=10000, current_price=9500,
        pnl_pct=-5.0, qty=1, reason="test",
    )
    broker = MagicMock()
    mock_order = MagicMock()
    mock_order.order_id = "ORDER123"
    broker.sell_market = lambda t, q: mock_order
    r = execute_time_exit(broker, sig)
    assert r["success"] is True
    assert r["order_id"] == "ORDER123"


def test_execute_fallback_sell_limit():
    """sell_market 없으면 sell_limit fallback."""
    sig = TimeExitSignal(
        ticker="067310", triggered=True, exit_type="D3_PROFIT",
        trade_days_elapsed=3, entry_price=10000, current_price=10500,
        pnl_pct=5.0, qty=1, reason="test",
    )
    broker = MagicMock(spec=["sell_limit"])  # sell_market 없음
    mock_order = MagicMock()
    mock_order.order_id = "ORDER456"
    broker.sell_limit = lambda t, p, q: mock_order
    r = execute_time_exit(broker, sig)
    assert r["success"] is True
    assert r["limit_price"] == 10395  # 10500 * 0.99


def test_execute_not_triggered():
    """trigger=False → 실패."""
    sig = TimeExitSignal(
        ticker="067310", triggered=False, exit_type="WAIT",
        trade_days_elapsed=1, entry_price=10000, current_price=10100,
        pnl_pct=1.0, qty=1, reason="test",
    )
    broker = MagicMock()
    r = execute_time_exit(broker, sig)
    assert r["success"] is False
