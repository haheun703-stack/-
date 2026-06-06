from __future__ import annotations

import inspect

from src.etf.regime_monitor import REGIME_BEAR, REGIME_BULL
from src.use_cases import engine_policy_map as epm
from src.use_cases import regime_router_v1 as rr


def _report(ticker: str, regime: str, days: int = 3, overheat: bool = False, warn: bool = False) -> dict:
    if regime is None:
        return {"ticker": ticker, "rows": 0, "error": "no data"}
    close = (170 if overheat else 130) if regime == REGIME_BULL else 95
    return {
        "ticker": ticker,
        "name": ticker,
        "rows": 120,
        "last_date": "2026-06-05",
        "current_regime": regime,
        "current_close": close,
        "current_ma60": 100,
        "days_in_current_regime": days,
        "current_observations": {"vol_cluster_warn": warn, "kospi_warn": warn},
    }


def _route_doc(specs: dict) -> dict:
    """specs: {ticker: (regime, days)} 또는 {ticker: regime}. route document 생성."""
    reports = {}
    for ticker, spec in specs.items():
        if isinstance(spec, tuple):
            regime, days = spec
        else:
            regime, days = spec, 3
        reports[ticker] = _report(ticker, regime, days=days)
    return rr.build_route_document(reports)


def test_all_bull_gives_r4_paper_only_hypothesis_c() -> None:
    doc = _route_doc({"488080": REGIME_BULL, "005930": REGIME_BULL, "000660": REGIME_BULL})
    policy = epm.build_policy_document(doc)

    assert policy["market_regime"] == epm.MARKET_R4
    assert policy["engines"]["hypothesis_c"] == epm.MODE_PAPER_ONLY
    assert policy["engines"]["smart_entry"] == epm.MODE_ALLOWED_SHADOW
    assert policy["engines"]["bc_rotation"] == epm.MODE_SHADOW_ONLY
    assert policy["engines"]["event_dart"] == epm.MODE_SHADOW_ONLY
    assert policy["engines"]["sell"] == epm.MODE_BLOCKED
    assert policy["paper_open_allowed"] is False
    assert policy["sell_automation"] == "BLOCKED"


def test_one_bear_forces_r1_all_shadow_or_blocked() -> None:
    # 보수적 통합: 하나라도 BEAR면 R1, 모든 신규진입 엔진 차단/shadow
    doc = _route_doc({"488080": REGIME_BULL, "005930": REGIME_BEAR, "000660": REGIME_BULL})
    policy = epm.build_policy_document(doc)

    assert policy["market_regime"] == epm.MARKET_R1
    assert policy["engines"]["hypothesis_c"] == epm.MODE_BLOCKED
    assert policy["engines"]["smart_entry"] == epm.MODE_SHADOW_ONLY
    assert policy["engines"]["bc_rotation"] == epm.MODE_SHADOW_ONLY
    assert policy["engines"]["sell"] == epm.MODE_BLOCKED


def test_data_unavailable_blocks_entries() -> None:
    doc = _route_doc({"488080": None, "005930": REGIME_BULL, "000660": REGIME_BULL})
    policy = epm.build_policy_document(doc)

    assert policy["market_regime"] == epm.MARKET_DATA_UNAVAILABLE
    assert policy["engines"]["hypothesis_c"] == epm.MODE_BLOCKED
    assert policy["engines"]["smart_entry"] == epm.MODE_SHADOW_ONLY


def test_hysteresis_one_day_bear_stays_r4() -> None:
    # 488080 BEAR 전환 첫날(days=1) → router effective=BULL 유지 → 시장 R4
    doc = _route_doc({"488080": (REGIME_BEAR, 1), "005930": REGIME_BULL, "000660": REGIME_BULL})
    policy = epm.build_policy_document(doc)

    assert policy["market_regime"] == epm.MARKET_R4
    assert policy["hysteresis"]["applied"] is True
    assert policy["hysteresis"]["days"] == 2
    assert "488080" in policy["hysteresis"]["pending_tickers"]


def test_hysteresis_second_day_bear_forces_r1() -> None:
    # 488080 BEAR 2거래일째(days=2) → effective=BEAR 확정 → 시장 R1
    doc = _route_doc({"488080": (REGIME_BEAR, 2), "005930": REGIME_BULL, "000660": REGIME_BULL})
    policy = epm.build_policy_document(doc)

    assert policy["market_regime"] == epm.MARKET_R1
    assert "488080" not in policy["hysteresis"]["pending_tickers"]


def test_r0_r5_shadow_advisories_have_no_authority() -> None:
    # BULL 과열(R5) + 관측경고(R0) 라벨이 있어도 엔진 권한 0, 국면 안 바뀜
    reports = {
        "005930": _report("005930", REGIME_BULL, days=5, overheat=True, warn=True),
        "488080": _report("488080", REGIME_BULL, days=5),
        "000660": _report("000660", REGIME_BULL, days=5),
    }
    doc = rr.build_route_document(reports)
    policy = epm.build_policy_document(doc)

    assert policy["market_regime"] == epm.MARKET_R4
    assert policy["shadow_advisory_authority"] is False
    assert len(policy["shadow_advisories"]) >= 1
    for adv in policy["shadow_advisories"]:
        assert adv.get("engine_switch_authority") is False


def test_policy_source_has_no_order_symbols() -> None:
    src = inspect.getsource(epm)
    for forbidden in ("KisOrderAdapter", "send_order", "place_order", "buy_limit", "sell_market"):
        assert forbidden not in src
