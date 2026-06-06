from __future__ import annotations

import inspect

import pandas as pd

from src.use_cases import engine_policy_map as epm
from src.use_cases import exit_signal_observer as eso


def _ohlcv(rows: list[tuple]) -> pd.DataFrame:
    """rows: [(date, open, high, low, close), ...]"""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1000 for _ in rows],
        },
        index=idx,
    )


def _entry(ticker: str = "005930", tier: str = "CORE", price: int = 1000) -> dict:
    return {
        "ticker": ticker,
        "name": ticker,
        "tier": tier,
        "virtual_entry_price": price,
        "entry_date": "2026-06-05",
        "source_type": "smart_entry_adapter",
    }


def test_shadow_open_candidate_produces_observation() -> None:
    # 진입 1000 → 고가 1120(+12%), 저가 940(-6%)
    ohlcv = _ohlcv([
        ("2026-06-05", 1000, 1050, 990, 1010),
        ("2026-06-08", 1010, 1120, 940, 1080),
    ])
    obs = eso.build_exit_observation(_entry(), ohlcv, epm.MARKET_R4)
    assert obs["mfe_pct"] == 12.0
    assert obs["mae_pct"] == -6.0
    assert obs["hold_status"] == eso.HOLD_OBSERVING
    assert obs["real_order"] is False
    assert obs["sell_automation"] == "BLOCKED"
    assert obs["order_intent_created"] is False
    assert "take_profit" in {s.get("kind") for s in obs["exit_signals_triggered"] if s["type"] == "target"}
    assert "loss_cut" in {s.get("kind") for s in obs["exit_signals_triggered"] if s["type"] == "stop"}


def test_empty_candidates_blank_report() -> None:
    doc = eso.build_observer_document([], epm.MARKET_R4, {})
    assert doc["counts"]["observed"] == 0
    assert doc["observations"] == []
    md = eso.build_observer_markdown(doc)
    assert md.strip()  # 빈 후보여도 MD 생성


def test_r1_regime_risk_exit_label_no_sell() -> None:
    ohlcv = _ohlcv([("2026-06-05", 1000, 1010, 950, 980), ("2026-06-08", 980, 990, 900, 920)])
    obs = eso.build_exit_observation(_entry(), ohlcv, epm.MARKET_R1)
    assert obs["hold_status"] == eso.HOLD_RISK_EXIT  # 관찰상 EXIT, 실매도 아님
    assert obs["real_order"] is False
    assert obs["sell_automation"] == "BLOCKED"


def test_data_unavailable_blocks_calc_safe_report() -> None:
    obs = eso.build_exit_observation(_entry(), None, epm.MARKET_DATA_UNAVAILABLE)
    assert obs["hold_status"] == eso.HOLD_BLOCKED
    assert obs["blocked_reason"] == "DATA_UNAVAILABLE"
    assert obs["mfe_pct"] is None
    assert obs["real_order"] is False


def test_no_ohlcv_holds_blocked() -> None:
    obs = eso.build_exit_observation(_entry(), None, epm.MARKET_R4)
    assert obs["hold_status"] == eso.HOLD_BLOCKED
    assert obs["blocked_reason"] == "no_ohlcv"


def test_order_intent_never_created() -> None:
    ohlcv = _ohlcv([("2026-06-05", 1000, 1100, 980, 1050), ("2026-06-08", 1050, 1080, 1000, 1030)])
    doc = eso.build_observer_document([_entry()], epm.MARKET_R4, {"005930": ohlcv})
    assert doc["safety"]["order_intent_created"] is False
    assert doc["safety"]["position_modified"] is False
    for o in doc["observations"]:
        assert o["order_intent_created"] is False


def test_safety_block_real_order_and_position() -> None:
    doc = eso.build_observer_document([_entry()], epm.MARKET_R4, {})
    s = doc["safety"]
    assert s["real_order"] is False
    assert s["sell_automation"] == "BLOCKED"
    assert s["position_modified"] is False
    assert s["scheduler_changed"] is False
    assert s["sajang_changed"] is False


def test_source_has_no_sell_or_order_symbols() -> None:
    src = inspect.getsource(eso)
    forbidden = (
        "smart_sell",
        "SmartSellExecutor",
        "sell_brain",
        "SellBrainAgent",
        "owner_rule",
        "evaluate_owner_rule",
        "sell_market",
        "sell_limit",
        "order_intents_gate",
        "assert_order_intent_exists",
        "KisOrderAdapter",
        "place_order",
        "send_order",
        "create_market_sell_order",
        "create_limit_sell_order",
    )
    for f in forbidden:
        assert f not in src
