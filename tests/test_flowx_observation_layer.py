"""관측 레이어 종단 흐름 + 안전 불변식 검증.

지시서 핵심: 가격축/반기 시가 라벨은 plan→smart_entry payload→daily_review→show_me로
전파되지만, C60 단독 hard gate / 엔진 정책 / 안전선은 라벨과 무관하게 불변이어야 한다.
"""

from __future__ import annotations

import inspect

import pandas as pd

from src.use_cases import daily_review as dr
from src.use_cases import engine_policy_map as epm
from src.use_cases import morning_plan_07 as mp
from src.use_cases import show_me_report as sm
from src.use_cases import smart_entry_adapter as sea


def _policy(market: str) -> dict:
    return {
        "version": "engine_policy_map_v1",
        "as_of_date": "2026-06-05",
        "market_regime": market,
        "market_regime_rule": "rule",
        "engines": dict(epm.ENGINE_POLICY[market]),
        "hysteresis": {"applied": True, "days": 2, "pending_tickers": []},
        "per_ticker": {},
        "shadow_advisories": [],
        "paper_open_allowed": False,
        "sell_automation": "BLOCKED",
    }


def _pick(ticker: str, tier: str) -> dict:
    return {
        "ticker": ticker, "name": ticker, "close": 1000, "stop_loss": 920,
        "target_price": 1200, "_tier": tier, "_floor_label": "바닥다지기후보",
        "_drop_context": "resilient_pullback", "_supply_state": "foreign_accumulation",
    }


def _labels(ticker: str) -> dict:
    return {
        ticker: {
            "price_axis": {
                "weekly_open_state": "ABOVE", "half_year_open_state": "ABOVE",
                "monthly_open_broken": True, "monthly_open": 80000,
            },
            "candle_turn": {"label": "EUM_YANG_REVERSAL"},
            "annual_overheat": {"overheat_grade": "OVERHEAT_500", "return_1y_pct": 520.0},
            "ipo_reversion": {"data_available": False, "ipo_reversion_state": None},
            "half_year_leader": {
                "data_available": True, "ticker": ticker, "name": ticker,
                "half_year_leader_grade": "HY_LEADER_CORE", "half_year_leader_score": 90,
                "sector": "AI반도체", "above_half_year_open": True,
                "distance_from_half_year_open_pct": 16.4,
            },
        }
    }


def _meta() -> dict:
    return {"as_of_date": "2026-06-05", "source": "pullback", "total": 1, "enter_count": 1, "avoid_count": 0}


def test_labels_flow_into_plan_candidates() -> None:
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), [_pick("005930", "CORE")], [], _meta(), shadow_labels=_labels("005930"))
    core = plan["tiers"]["CORE"][0]
    assert core["shadow_labels"]["half_year_leader"]["half_year_leader_grade"] == "HY_LEADER_CORE"
    md = mp.build_plan_markdown(plan)
    assert "HY_LEADER_CORE" in md
    assert "매수" not in md  # 라벨 붙어도 매수 표현 금지 유지


def test_labels_passthrough_to_smart_entry_payload() -> None:
    plan = mp.build_plan_document(_policy(epm.MARKET_R4), [_pick("005930", "CORE")], [], _meta(), shadow_labels=_labels("005930"))
    doc = sea.build_shadow_entries(plan)
    e = doc["shadow_entries"][0]
    assert e["shadow_labels"]["annual_overheat"]["overheat_grade"] == "OVERHEAT_500"
    assert e["real_order"] is False  # 라벨은 진입 조건에 영향 없음


def test_hard_gate_unchanged_by_labels() -> None:
    # 라벨 유무와 무관하게 엔진 정책(2단계 SSOT)·국면은 그대로
    no_label = mp.build_plan_document(_policy(epm.MARKET_R4), [_pick("005930", "CORE")], [], _meta())
    with_label = mp.build_plan_document(_policy(epm.MARKET_R4), [_pick("005930", "CORE")], [], _meta(), shadow_labels=_labels("005930"))
    assert no_label["engines"] == with_label["engines"]
    assert no_label["market_regime"] == with_label["market_regime"]
    assert with_label["safety"]["real_order"] is False
    assert with_label["paper_open_allowed"] is False
    # R1이면 라벨이 아무리 강해도 SmartEntry는 SHADOW_ONLY(차단) 유지
    r1 = mp.build_plan_document(_policy(epm.MARKET_R1), [_pick("005930", "CORE")], [], _meta(), shadow_labels=_labels("005930"))
    assert r1["engines"]["smart_entry"] == "SHADOW_ONLY"
    assert r1["tiers"]["CORE"][0]["observation"] == mp.OBS_SHADOW


def test_daily_review_label_performance_and_show_me_panels() -> None:
    # daily_review 후보(라벨 부착) → 라벨 성과 + show_me 패널
    def _ohlcv(series: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2026-06-05", periods=len(series), freq="D")
        return pd.DataFrame(
            {"open": series, "high": [s * 1.03 for s in series], "low": [s * 0.97 for s in series],
             "close": series, "volume": [1000] * len(series)}, index=idx
        )

    cand = {"ticker": "005930", "name": "삼성", "tier": "CORE", "shadow_labels": _labels("005930")["005930"]}
    ohlcv = {"005930": _ohlcv([1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100])}
    cr = dr.build_candidate_review([cand], ohlcv)
    lp = cr["label_performance"]
    assert lp["half_year_open"]["ABOVE"]["count"] == 1
    assert lp["annual_overheat"]["OVERHEAT_500"]["mean_d10"] == 10.0
    assert lp["candle_turn"]["EUM_YANG_REVERSAL"]["count"] == 1

    review_doc = dr.build_review_document("2026-06-05", "R4_NORMAL_BULL", cr, dr.build_execution_review([]), dr.build_exit_summary([]))
    show = sm.build_show_me_document(review_doc, {"routes": {}})
    panels = show["shadow_label_panels"]
    assert panels["half_year_leader_top"][0]["grade"] == "HY_LEADER_CORE"
    assert panels["weekly_open_compare"]["above"] == ["005930"]
    assert len(panels["monthly_open_broken"]) == 1
    assert len(panels["annual_overheat_500"]) == 1
    md = sm.build_show_me_markdown(show)
    assert "반기 주도주 TOP 20" in md
    assert show["safety_panel"]["real_order"] is False
    assert show["safety_panel"]["policy_changed"] is False


def test_new_modules_have_no_order_or_sell_symbols() -> None:
    from src.use_cases import half_year_leader_scanner as hls
    from src.use_cases import price_axis_regime as par

    forbidden = (
        "smart_sell", "sell_brain", "owner_rule", "sell_market", "sell_limit",
        "buy_limit", "order_intents_gate", "KisOrderAdapter", "place_order",
        "send_order", "run_adaptive_cycle", "PAPER_OPEN",
    )
    for mod in (par, hls):
        src = inspect.getsource(mod)
        for f in forbidden:
            assert f not in src, f"{mod.__name__}에 금지 심볼 {f}"
