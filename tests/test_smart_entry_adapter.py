from __future__ import annotations

import inspect

from src.use_cases import engine_policy_map as epm
from src.use_cases import smart_entry_adapter as sea


def _rows(tier: str, n: int) -> list[dict]:
    return [
        {
            "ticker": f"{tier}-{i}",
            "name": f"{tier}{i}",
            "tier": tier,
            "ref_close": 1000,
            "stop_loss": 920,
            "target": 1200,
            "floor_label": "바닥다지기후보",
            "drop_context": "normal",
            "supply_state": "foreign_accumulation",
            "observation": "장중 SmartEntry 관찰 대상",
        }
        for i in range(n)
    ]


def _plan(market: str, smart_mode: str, core: int = 0, watch: int = 0, control: int = 0) -> dict:
    return {
        "as_of_date": "2026-06-05",
        "market_regime": market,
        "engines": {"smart_entry": smart_mode, "sell": "BLOCKED"},
        "tiers": {"CORE": _rows("CORE", core), "WATCH": _rows("WATCH", watch), "CONTROL": _rows("CONTROL", control)},
        "sell_automation": "BLOCKED",
    }


def test_r4_core_watch_to_shadow_open() -> None:
    doc = sea.build_shadow_entries(_plan(epm.MARKET_R4, epm.MODE_ALLOWED_SHADOW, core=1, watch=2))
    assert doc["counts"]["shadow_open"] == 3
    for e in doc["shadow_entries"]:
        assert e["status"] == sea.STATUS_SHADOW_OPEN
        assert e["real_order"] is False
        assert e["paper_mode"] is True
        assert e["grade"] == sea.SHADOW_GRADE


def test_r4_control_is_control_only_not_entry() -> None:
    doc = sea.build_shadow_entries(_plan(epm.MARKET_R4, epm.MODE_ALLOWED_SHADOW, core=1, control=9))
    assert doc["counts"]["control_only"] == 9
    entry_tickers = {e["ticker"] for e in doc["shadow_entries"]}
    control_tickers = {c["ticker"] for c in doc["control_only"]}
    assert entry_tickers.isdisjoint(control_tickers)  # CONTROL은 진입 후보 아님
    for c in doc["control_only"]:
        assert c["status"] == sea.STATUS_CONTROL_ONLY


def test_r1_blocks_core_watch_by_regime() -> None:
    doc = sea.build_shadow_entries(_plan(epm.MARKET_R1, "SHADOW_ONLY", core=1, watch=1))
    assert doc["counts"]["shadow_open"] == 0
    assert doc["counts"]["blocked"] == 2
    for b in doc["blocked"]:
        assert b["status"] == sea.STATUS_BLOCKED_REGIME


def test_data_unavailable_blocks_by_data() -> None:
    doc = sea.build_shadow_entries(_plan(epm.MARKET_DATA_UNAVAILABLE, "SHADOW_ONLY", core=2))
    assert doc["counts"]["shadow_open"] == 0
    for b in doc["blocked"]:
        assert b["status"] == sea.STATUS_BLOCKED_DATA


def test_paper_open_default_false_shadow() -> None:
    doc = sea.build_shadow_entries(_plan(epm.MARKET_R4, epm.MODE_ALLOWED_SHADOW, core=1))
    assert doc["paper_open"] is False
    assert doc["safety"]["paper_open_default"] is False
    assert doc["shadow_entries"][0]["status"] == sea.STATUS_SHADOW_OPEN


def test_run_path_cannot_enable_paper_open() -> None:
    # run 경로 시그니처에 paper_open 인자 없음 = PAPER_OPEN 강제 차단
    sig = inspect.signature(sea.run_smart_entry_adapter)
    assert "paper_open" not in sig.parameters


def test_sell_automation_blocked() -> None:
    doc = sea.build_shadow_entries(_plan(epm.MARKET_R4, epm.MODE_ALLOWED_SHADOW, core=1))
    assert doc["safety"]["sell_automation"] == "BLOCKED"
    assert doc["safety"]["real_order"] is False
    assert doc["safety"]["order_adapter"] == "None"


def test_source_has_no_order_or_engine_symbols() -> None:
    src = inspect.getsource(sea)
    forbidden = (
        "KisOrderAdapter",
        "kis_order_adapter",
        "SmartEntryEngine",
        "kis_intraday_adapter",
        "order_intents_gate",
        "buy_limit",
        "sell_market",
        "place_order",
        "send_order",
        "run_full_session",
        "place_initial_orders",
        "update_orders",
        "cancel_all_unfilled",
    )
    for f in forbidden:
        assert f not in src
