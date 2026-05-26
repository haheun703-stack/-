"""동시호가 강도 게이트 단위 테스트 (5/26)."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.use_cases.opening_call_gate import (
    check_opening_call_gate,
    _is_opening_call_time,
    OpeningCallGate,
)


def _book(bid_total, ask_total):
    return {"total_bid_vol": bid_total, "total_ask_vol": ask_total, "asks": [], "bids": []}


def test_out_of_hours_pass():
    """동시호가 시간 외 (09:30) → PASS (OUT_OF_HOURS)."""
    now = datetime(2026, 5, 27, 9, 30)
    r = check_opening_call_gate(_book(1000, 1000), now=now)
    assert r.allow
    assert r.reason == "OUT_OF_HOURS"


def test_strong_open_passes():
    """동시호가 + 매수세 우세 (ratio 2.0) → STRONG_OPEN."""
    now = datetime(2026, 5, 27, 8, 45)
    r = check_opening_call_gate(_book(2000, 1000), now=now)
    assert r.allow
    assert r.reason == "STRONG_OPEN"
    assert r.is_strong
    assert r.ratio == 2.0


def test_weak_open_blocks():
    """동시호가 + 매도세 우세 (ratio 0.5) → WEAK_OPEN 차단."""
    now = datetime(2026, 5, 27, 8, 45)
    r = check_opening_call_gate(_book(500, 1000), now=now)
    assert r.allow is False
    assert r.reason == "WEAK_OPEN"


def test_balanced_passes():
    """동시호가 + 균형 (ratio 1.0) → BALANCED."""
    now = datetime(2026, 5, 27, 8, 45)
    r = check_opening_call_gate(_book(1000, 1000), now=now)
    assert r.allow
    assert r.reason == "BALANCED"


def test_low_volume_fail_open():
    """잔량 < 1000주 → fail-open (장 초반 데이터 부족)."""
    now = datetime(2026, 5, 27, 8, 45)
    r = check_opening_call_gate(_book(100, 200), now=now)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_empty_orderbook_fail_open():
    """빈 orderbook → fail-open."""
    now = datetime(2026, 5, 27, 8, 45)
    r = check_opening_call_gate({}, now=now)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_none_orderbook_fail_open():
    """None → fail-open."""
    now = datetime(2026, 5, 27, 8, 45)
    r = check_opening_call_gate(None, now=now)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_boundary_8_30_inclusive():
    """8:30 정확히 → 동시호가 시간 (포함)."""
    now = datetime(2026, 5, 27, 8, 30)
    assert _is_opening_call_time(now)


def test_boundary_9_00_exclusive():
    """9:00 정확히 → 동시호가 시간 외 (제외)."""
    now = datetime(2026, 5, 27, 9, 0)
    assert not _is_opening_call_time(now)


def test_real_world_sanil_5_26():
    """5/26 산일전기 시초가 -3% — 동시호가 강도 약세 시뮬레이션."""
    # 동시호가 매수 5000 / 매도 12000 = ratio 0.42 < 0.67 → 차단
    now = datetime(2026, 5, 26, 8, 50)
    r = check_opening_call_gate(_book(5000, 12000), now=now)
    assert r.allow is False
    assert r.reason == "WEAK_OPEN"
