"""4수급 실시간 게이트 단위 테스트 (5/26)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.use_cases.supply_flow_gate import check_supply_flow_gate, SupplyFlowGate


def _make_adapter(foreign, inst, indiv=0):
    a = MagicMock()
    a.fetch_investor_flow = lambda t: {
        "foreign_net_buy": foreign, "inst_net_buy": inst,
        "individual_net_buy": indiv,
    }
    return a


def test_dual_buy_strong():
    """외인 + 기관 모두 양수 → DUAL_BUY ★"""
    a = _make_adapter(foreign=10000, inst=5000)
    r = check_supply_flow_gate(a, "067310")
    assert r.allow
    assert r.reason == "DUAL_BUY"
    assert r.is_dual_buy
    assert r.foreign_net == 10000
    assert r.inst_net == 5000


def test_net_buy_foreign_only():
    """외인만 매수 + 기관 0 → NET_BUY (합계 양수)."""
    a = _make_adapter(foreign=10000, inst=0)
    r = check_supply_flow_gate(a, "067310")
    assert r.allow
    assert r.reason == "NET_BUY"
    assert not r.is_dual_buy


def test_dual_sell_blocks():
    """외인 + 기관 동반 -1000 초과 → DUAL_SELL 차단."""
    a = _make_adapter(foreign=-2000, inst=-500)
    # 합계 -2500 < -1000 → 차단
    r = check_supply_flow_gate(a, "067310")
    assert r.allow is False
    assert r.reason == "DUAL_SELL"


def test_dual_sell_small_passes():
    """동반 매도 but 합계 -1000 미달 → WEAK_SELL 통과."""
    a = _make_adapter(foreign=-300, inst=-200)
    # 합계 -500 > -1000
    r = check_supply_flow_gate(a, "067310")
    assert r.allow
    assert r.reason == "WEAK_SELL"


def test_weak_sell_one_side():
    """외인 - / 기관 + (합계 음수 but DUAL_SELL 미달) → WEAK_SELL."""
    a = _make_adapter(foreign=-500, inst=100)
    # 합계 -400 (WEAK_SELL, DUAL_SELL X — 기관이 양수)
    r = check_supply_flow_gate(a, "067310")
    assert r.allow
    assert r.reason == "WEAK_SELL"


def test_none_adapter_fail_open():
    """intraday_adapter=None → fail-open."""
    r = check_supply_flow_gate(None, "067310")
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_fetch_exception_fail_open():
    """fetch_investor_flow 예외 → fail-open."""
    a = MagicMock()
    a.fetch_investor_flow = MagicMock(side_effect=Exception("API down"))
    r = check_supply_flow_gate(a, "067310")
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_empty_flow_fail_open():
    """빈 dict → fail-open."""
    a = MagicMock()
    a.fetch_investor_flow = lambda t: {}
    r = check_supply_flow_gate(a, "067310")
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_custom_threshold():
    """엄격 임계 -200으로 → -500 합계도 차단."""
    a = _make_adapter(foreign=-300, inst=-200)
    r = check_supply_flow_gate(a, "067310", dual_sell_threshold=-200)
    # 합계 -500 < -200 + 둘 다 음수 → DUAL_SELL
    assert r.allow is False


def test_real_world_isc_5_26():
    """5/26 ISC +16.9% 시나리오: 외인 +200만 + 기관 +30만."""
    a = _make_adapter(foreign=2000000, inst=300000, indiv=-1500000)
    r = check_supply_flow_gate(a, "095340")
    assert r.allow
    assert r.reason == "DUAL_BUY"
    assert r.is_dual_buy
