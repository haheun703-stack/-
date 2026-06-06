from __future__ import annotations

import inspect

from src.etf.regime_monitor import REGIME_BEAR, REGIME_BULL
from src.use_cases import regime_router_v1 as rr


def _report(
    regime: str,
    close: float = 150.0,
    ma60: float = 100.0,
    kospi_warn: bool = False,
    days: int = 3,
) -> dict:
    return {
        "ticker": "TEST",
        "name": "테스트",
        "rows": 120,
        "last_date": "2026-06-05",
        "current_regime": regime,
        "current_close": close,
        "current_ma60": ma60,
        "days_in_current_regime": days,
        "current_observations": {
            "vol_cluster_warn": False,
            "kospi_warn": kospi_warn,
        },
        "observation_gate_status": {"foreign_net": "NOISE / gate 제외"},
    }


def test_r4_bull_allows_shadow_smart_entry_but_no_paper_open() -> None:
    route = rr.route_from_report("005930", _report(REGIME_BULL, close=130, ma60=100))

    assert route["hard_gate_regime"] == rr.REGIME_R4
    assert route["hard_gate_status"] == "HARD_GATE"
    assert route["allow_new_entries"] is True
    assert route["allow_hypothesis_c"] is True
    assert route["smart_entry_observation"] == "ALLOWED_SHADOW"
    assert route["paper_open_allowed"] is False
    assert route["sell_automation"] == "BLOCKED"


def test_r1_bear_blocks_new_entries_and_hypothesis_c() -> None:
    route = rr.route_from_report("005930", _report(REGIME_BEAR, close=95, ma60=100))

    assert route["hard_gate_regime"] == rr.REGIME_R1
    assert route["allow_new_entries"] is False
    assert route["allow_hypothesis_c"] is False
    assert route["smart_entry_observation"] == "SHADOW_ONLY"
    assert route["sell_automation"] == "BLOCKED"


def test_r0_and_r5_are_shadow_labels_only() -> None:
    route = rr.route_from_report("005930", _report(REGIME_BULL, close=170, ma60=100, kospi_warn=True))

    labels = {label["regime"]: label for label in route["shadow_labels"]}
    assert rr.REGIME_R0 in labels
    assert rr.REGIME_R5 in labels
    assert labels[rr.REGIME_R0]["engine_switch_authority"] is False
    assert labels[rr.REGIME_R5]["engine_switch_authority"] is False
    assert route["allow_new_entries"] is True


def test_build_route_document_global_safety() -> None:
    doc = rr.build_route_document({"005930": _report(REGIME_BULL), "000660": _report(REGIME_BEAR)})

    assert doc["as_of_date"] == "2026-06-05"
    assert doc["global_safety"]["real_order"] is False
    assert doc["global_safety"]["scheduler_changed"] is False
    assert doc["global_safety"]["sajang_changed"] is False
    assert doc["routes"]["000660"]["allow_new_entries"] is False


def test_router_source_has_no_order_symbols() -> None:
    src = inspect.getsource(rr)
    for forbidden in ("KisOrderAdapter", "send_order", "place_order", "buy_limit", "sell_market"):
        assert forbidden not in src


def test_hysteresis_bear_first_day_keeps_bull_policy() -> None:
    # BEAR 전환 첫날(days=1): 미확정 → 직전 BULL 정책 유지(1일 휩쏘 방지)
    route = rr.route_from_report("005930", _report(REGIME_BEAR, close=95, ma60=100, days=1))

    assert route["c60_regime_raw"] == REGIME_BEAR
    assert route["effective_regime"] == REGIME_BULL
    assert route["regime_confirmed"] is False
    assert route["in_hysteresis_window"] is True
    assert route["hysteresis_days"] == 2
    assert route["hard_gate_regime"] == rr.REGIME_R4
    assert route["allow_new_entries"] is True


def test_hysteresis_bear_second_day_confirms_r1() -> None:
    # BEAR 2거래일째(days=2): 확정 → R1 발동
    route = rr.route_from_report("005930", _report(REGIME_BEAR, close=95, ma60=100, days=2))

    assert route["effective_regime"] == REGIME_BEAR
    assert route["regime_confirmed"] is True
    assert route["in_hysteresis_window"] is False
    assert route["hard_gate_regime"] == rr.REGIME_R1
    assert route["allow_new_entries"] is False


def test_hysteresis_bull_first_day_keeps_bear_policy() -> None:
    # BULL 전환 첫날(days=1): 직전 BEAR 정책 유지(약세장 1일 반등에 안 속음)
    route = rr.route_from_report("005930", _report(REGIME_BULL, close=130, ma60=100, days=1))

    assert route["effective_regime"] == REGIME_BEAR
    assert route["regime_confirmed"] is False
    assert route["hard_gate_regime"] == rr.REGIME_R1
    assert route["allow_new_entries"] is False
    assert route["allow_hypothesis_c"] is False
