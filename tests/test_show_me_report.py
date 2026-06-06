from __future__ import annotations

import inspect

from src.use_cases import show_me_report as sm


def _review_doc(candidates=None, entries=None, tier_counts=None, market="R4_NORMAL_BULL") -> dict:
    return {
        "observation_date": "2026-06-05",
        "market_regime": market,
        "one_line": "테스트 한 줄",
        "candidate_performance": {
            "basis": "as_of_close",
            "candidate_count": len(candidates or []),
            "tier_counts": tier_counts or {"CORE": 0, "WATCH": 0, "CONTROL": 0},
            "candidates": candidates or [],
            "missed_winner": [],
            "false_positive": [],
        },
        "execution_performance": {
            "basis": "virtual_entry_price",
            "entry_count": len(entries or []),
            "entries": entries or [],
        },
        "exit_observer_summary": {"exit_type_trigger_counts": {"stop": 0, "target": 0, "trend": 0, "time": 0}, "note": "n"},
        "data_warnings": [],
    }


def _route_doc(regime="BULL") -> dict:
    return {
        "routes": {
            "488080": {
                "data_available": True, "name": "반도체레버", "current_close": 30000,
                "current_ma60": 25000, "close_vs_ma60_pct": 20.0, "c60_regime_raw": regime,
                "effective_regime": regime, "hard_gate_regime": "R4_NORMAL_BULL",
                "in_hysteresis_window": False,
            }
        }
    }


def test_empty_data_report_generated() -> None:
    doc = sm.build_show_me_document(_review_doc(), {"routes": {}})
    md = sm.build_show_me_markdown(doc)
    assert md.strip()  # 데이터 0건도 리포트 생성
    assert doc["candidate_flow"]["candidate_count"] == 0


def test_zero_candidates_blank_tables() -> None:
    doc = sm.build_show_me_document(_review_doc(), _route_doc())
    assert doc["candidate_performance"]["candidates"] == []
    assert doc["execution_performance"]["entries"] == []
    md = sm.build_show_me_markdown(doc)
    assert "후보선정 성능 SHOW ME" in md
    assert "실행 성능 SHOW ME" in md


def test_basis_not_mixed() -> None:
    # 후보선정 기준 as_of_close vs 실행 기준 virtual_entry_price 분리 유지
    doc = sm.build_show_me_document(_review_doc(), _route_doc())
    assert doc["candidate_performance"]["basis"] == "as_of_close"
    assert doc["execution_performance"]["basis"] == "virtual_entry_price"
    assert doc["candidate_performance"]["basis"] != doc["execution_performance"]["basis"]


def test_safety_panel_present() -> None:
    doc = sm.build_show_me_document(_review_doc(), _route_doc())
    sp = doc["safety_panel"]
    assert sp["real_order"] is False
    assert sp["scheduler_changed"] is False
    assert sp["sajang_changed"] is False
    assert sp["paper_open_allowed"] is False
    assert sp["sell_automation"] == "BLOCKED"
    assert sp["policy_changed"] is False
    assert sp["order_symbol_grep"] == 0


def test_c60_panel_in_regime_show_me() -> None:
    doc = sm.build_show_me_document(_review_doc(), _route_doc())
    panel = doc["regime_show_me"]
    assert len(panel) == 1
    assert panel[0]["ticker"] == "488080"
    assert panel[0]["effective_regime"] == "BULL"


def test_source_has_no_order_symbols() -> None:
    src = inspect.getsource(sm)
    forbidden = (
        "smart_sell", "sell_brain", "owner_rule", "sell_market", "sell_limit",
        "buy_limit", "order_intents_gate", "KisOrderAdapter", "place_order", "send_order",
    )
    for f in forbidden:
        assert f not in src
