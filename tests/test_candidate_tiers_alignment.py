"""FLOWX Market OS v1 4단계 — candidate_tiers 정합 검증.

plan 경로(morning_plan_07 → build_picks_from_candidate_log)와
ledger 경로(paper_smart_entry.main → record_entries / record_control_pool)가
같은 candidate_log로 **같은 classify_tier(SSOT)** 등급을 매기는지 검증한다.

새 분류기/스키마 없음. monkeypatch로 pse.LEDGER를 tmp 파일에 격리하여
실제 data/paper_ledger.json은 절대 건드리지 않는다(실주문 0 / read-only).
"""

from __future__ import annotations

import json

from scripts import paper_smart_entry as pse


def _candidate(ticker, name, floor_label, drop_context, supply_state, entry=1000):
    return {
        "ticker": ticker,
        "name": name,
        "decision": "진입",
        "date": "2026-06-05",
        "floor_quality": {"label": floor_label, "floor_quality_score": 1},
        "market_context": {"drop_context": drop_context},
        "supply_confirmation": {"supply_state": supply_state},
        "risk_reward": {"entry_price": entry},
    }


def _ledger_3tier() -> dict:
    """CORE / WATCH / CONTROL 3등급을 다 가진 candidate_log."""
    return {
        "candidate_log": [
            {
                "as_of_date": "2026-06-05",
                "source": "alignment_unit",
                "total": 3,
                "enter_count": 3,
                "avoid_count": 0,
                "candidates": [
                    # CORE: 횡보 + 급락회피 + 단독수급
                    _candidate("000001", "CORE종목", "바닥다지기후보", "normal", "foreign_accumulation"),
                    # WATCH: 횡보 + 급락회피, 수급 약함(dual_buying=단독 아님)
                    _candidate("000003", "WATCH종목", "바닥다지기후보", "normal", "dual_buying"),
                    # CONTROL: 개별급락 → 차트4통과·C-rule 탈락
                    _candidate("000002", "CONTROL종목", "관찰(위험)", "stock_specific_drop", "distribution_warning"),
                ],
            }
        ]
    }


def _setup_ledger_file(tmp_path, monkeypatch, ledger: dict):
    """실제 data/paper_ledger.json 대신 tmp 파일로 격리. build_picks가 이 파일을 읽는다."""
    ledger_file = tmp_path / "paper_ledger.json"
    ledger_file.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pse, "LEDGER", ledger_file)
    return ledger_file


def _plan_tier_map(picks, control) -> dict:
    m = {p["ticker"]: p["_tier"] for p in picks}
    m.update({c["ticker"]: c["_tier"] for c in control})
    return m


def _ledger_tier_map(ledger: dict) -> dict:
    m = {}
    for r in ledger.get("shadow_observations", []):
        m[r["ticker"]] = r["tier"]
    for r in ledger.get("shadow_control", []):
        m[r["ticker"]] = r["tier"]
    return m


def test_plan_and_ledger_tiers_match(tmp_path, monkeypatch) -> None:
    """plan(build_picks)과 ledger(record_entries+record_control_pool)가 같은 등급을 매긴다."""
    _setup_ledger_file(tmp_path, monkeypatch, _ledger_3tier())

    # plan 측: 실제 build_picks_from_candidate_log() (파일 읽기)
    picks, control, ledger = pse.build_picks_from_candidate_log()
    plan_tier = _plan_tier_map(picks, control)
    assert plan_tier == {"000001": "CORE", "000003": "WATCH", "000002": "CONTROL"}

    # ledger 측: CORE/WATCH → shadow_observations, CONTROL → shadow_control
    entry_tickers = [p["ticker"] for p in picks]  # CORE + WATCH (CONTROL 제외)
    report = {"details": [{"ticker": t, "decision": "buy"} for t in entry_tickers]}
    pse.record_entries(report, ledger, paper_open=False)
    pse.record_control_pool(control, ledger)

    # 정합: plan 등급 == ledger 등급 (전 ticker)
    assert _ledger_tier_map(ledger) == plan_tier


def test_core_watch_boundary_consistent(tmp_path, monkeypatch) -> None:
    """CORE vs WATCH 경계(단독수급 유무)가 양 경로에서 동일하게 갈린다."""
    _setup_ledger_file(tmp_path, monkeypatch, _ledger_3tier())

    picks, control, ledger = pse.build_picks_from_candidate_log()
    plan_tier = _plan_tier_map(picks, control)
    assert plan_tier["000001"] == "CORE"
    assert plan_tier["000003"] == "WATCH"

    report = {"details": [
        {"ticker": "000001", "decision": "buy"},
        {"ticker": "000003", "decision": "buy"},
    ]}
    pse.record_entries(report, ledger, paper_open=False)
    ledger_tier = _ledger_tier_map(ledger)
    assert ledger_tier["000001"] == "CORE"
    assert ledger_tier["000003"] == "WATCH"


def test_empty_candidates_no_tier_mismatch(tmp_path, monkeypatch) -> None:
    """후보 0건이어도 plan/ledger 양쪽 등급이 비고 예외 없음."""
    _setup_ledger_file(tmp_path, monkeypatch, {"candidate_log": []})

    picks, control, ledger = pse.build_picks_from_candidate_log()
    assert picks == []
    assert control == []

    pse.record_entries({"details": []}, ledger, paper_open=False)
    pse.record_control_pool(control, ledger)
    assert _ledger_tier_map(ledger) == _plan_tier_map(picks, control) == {}


def test_shadow_open_and_paper_open_same_tier(tmp_path, monkeypatch) -> None:
    """SHADOW_OPEN(기본)과 PAPER_OPEN(--paper-open)은 status/key만 다르고 tier는 같다.

    5단계에서 헷갈리면 안 되는 SHADOW↔PAPER 경계가 등급 분류에는 영향을 주지 않음을 못박는다.
    """
    # SHADOW_OPEN
    _setup_ledger_file(tmp_path, monkeypatch, _ledger_3tier())
    picks_s, _, ledger_s = pse.build_picks_from_candidate_log()
    report_s = {"details": [{"ticker": p["ticker"], "decision": "buy"} for p in picks_s]}
    pse.record_entries(report_s, ledger_s, paper_open=False)
    shadow_tier = {r["ticker"]: r["tier"] for r in ledger_s["shadow_observations"]}
    assert all(r["status"] == "SHADOW_OPEN" for r in ledger_s["shadow_observations"])

    # PAPER_OPEN (별도 tmp 파일)
    ledger_file2 = tmp_path / "ledger_paper.json"
    ledger_file2.write_text(json.dumps(_ledger_3tier(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pse, "LEDGER", ledger_file2)
    picks_p, _, ledger_p = pse.build_picks_from_candidate_log()
    report_p = {"details": [{"ticker": p["ticker"], "decision": "buy"} for p in picks_p]}
    pse.record_entries(report_p, ledger_p, paper_open=True)
    paper_tier = {r["ticker"]: r["tier"] for r in ledger_p["paper_trades"]}
    assert all(r["status"] == "PAPER_OPEN" for r in ledger_p["paper_trades"])

    assert shadow_tier == paper_tier  # tier 분류는 SHADOW/PAPER 무관하게 동일
