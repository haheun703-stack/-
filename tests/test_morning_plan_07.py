from __future__ import annotations

import inspect

from src.use_cases import engine_policy_map as epm
from src.use_cases import morning_plan_07 as mp


def _policy(market: str) -> dict:
    return {
        "version": "engine_policy_map_v1",
        "as_of_date": "2026-06-05",
        "market_regime": market,
        "market_regime_rule": "테스트 규칙",
        "engines": dict(epm.ENGINE_POLICY[market]),
        "hysteresis": {"applied": True, "days": 2, "pending_tickers": []},
        "per_ticker": {
            "005930": {
                "data_available": True,
                "hard_gate_regime": "R4_NORMAL_BULL",
                "c60_regime_raw": "BULL",
                "in_hysteresis_window": False,
            }
        },
        "shadow_advisories": [],
        "paper_open_allowed": False,
        "sell_automation": "BLOCKED",
    }


def _pick(ticker: str, tier: str) -> dict:
    return {
        "ticker": ticker,
        "name": ticker,
        "close": 1000,
        "stop_loss": 920,
        "target_price": 1200,
        "_tier": tier,
        "_floor_label": "바닥다지기후보",
        "_drop_context": "resilient_pullback",
        "_supply_state": "foreign_accumulation",
    }


def _meta() -> dict:
    return {"as_of_date": "2026-06-05", "source": "pullback", "total": 10, "enter_count": 3, "avoid_count": 7}


def test_data_unavailable_blocks_entries() -> None:
    plan = mp.build_plan_document(_policy(epm.MARKET_DATA_UNAVAILABLE), [], [], _meta())
    assert plan["market_regime"] == epm.MARKET_DATA_UNAVAILABLE
    assert plan["engines"]["hypothesis_c"] == "BLOCKED"
    assert plan["paper_open_allowed"] is False
    md = mp.build_plan_markdown(plan)
    assert "신규진입 차단" in md


def test_r1_smart_entry_shadow_only_even_with_candidates() -> None:
    picks = [_pick("005930", "CORE"), _pick("000660", "WATCH")]
    plan = mp.build_plan_document(_policy(epm.MARKET_R1), picks, [], _meta())
    assert plan["engines"]["smart_entry"] == "SHADOW_ONLY"
    rows = plan["tiers"]["CORE"] + plan["tiers"]["WATCH"]
    assert rows  # 후보는 존재하되
    for row in rows:
        assert row["observation"] == mp.OBS_SHADOW  # SmartEntry 실행 안 함


def test_r4_core_watch_are_observation_candidates() -> None:
    picks = [_pick("005930", "CORE"), _pick("000660", "WATCH")]
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), picks, [], _meta())
    assert plan["engines"]["smart_entry"] == "ALLOWED_SHADOW"
    assert len(plan["tiers"]["CORE"]) == 1
    assert len(plan["tiers"]["WATCH"]) == 1
    for row in plan["tiers"]["CORE"] + plan["tiers"]["WATCH"]:
        assert row["observation"] == mp.OBS_SMARTENTRY


def test_control_is_comparison_group() -> None:
    control = [_pick("011200", "CONTROL")]
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), [], control, _meta())
    assert len(plan["tiers"]["CONTROL"]) == 1
    assert plan["tiers"]["CONTROL"][0]["observation"] == mp.OBS_COMPARISON


def test_empty_candidates_still_generates() -> None:
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), [], [], mp._candidate_log_meta([]))
    assert plan["tiers"] == {"CORE": [], "WATCH": [], "CONTROL": []}
    assert plan["candidate_log"]["total"] == 0
    md = mp.build_plan_markdown(plan)
    assert md.strip()  # 후보 0건이어도 MD 생성


def test_paper_open_default_blocked_and_sell_blocked() -> None:
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), [], [], _meta())
    assert plan["paper_open_allowed"] is False
    assert plan["sell_automation"] == "BLOCKED"
    assert plan["safety"]["paper_open_default"] is False
    md = mp.build_plan_markdown(plan)
    assert "PAPER_OPEN 금지" in md
    assert "BLOCKED" in md


def test_no_buy_word_in_markdown() -> None:
    # 지시서 §6: "매수" 표현 금지, "관찰/후보/비교군"으로만
    picks = [_pick("005930", "CORE")]
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), picks, [], _meta())
    md = mp.build_plan_markdown(plan)
    assert "매수" not in md


def test_source_has_no_order_symbols() -> None:
    src = inspect.getsource(mp)
    for forbidden in (
        "KisOrderAdapter",
        "send_order",
        "place_order",
        "buy_limit",
        "sell_market",
        "record_control_pool",
        "paper_smart_entry.main",
    ):
        assert forbidden not in src
