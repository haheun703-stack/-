from __future__ import annotations

import inspect

import pandas as pd

from src.use_cases import daily_review as dr


def _ohlcv(series: list[float]) -> pd.DataFrame:
    """series[0]=as_of 종가, 이후 D+1, D+2, ... 종가."""
    idx = pd.date_range("2026-06-05", periods=len(series), freq="D")
    return pd.DataFrame(
        {
            "open": series,
            "high": [s * 1.02 for s in series],
            "low": [s * 0.98 for s in series],
            "close": series,
            "volume": [1000] * len(series),
        },
        index=idx,
    )


def _cand(ticker: str, tier: str) -> dict:
    return {"ticker": ticker, "name": ticker, "tier": tier}


def _observation(ticker: str, tier: str, entry: int, d10: float) -> dict:
    return {
        "ticker": ticker,
        "name": ticker,
        "tier": tier,
        "virtual_entry_price": entry,
        "mfe_pct": 12.0,
        "mae_pct": -4.0,
        "best_exit_candidate": {"type": "target", "level_pct": 10.0},
        "worst_exit_candidate": {"type": "stop", "level_pct": -3.0},
        "exit_signals_triggered": [
            {"type": "time", "horizon": "D+1", "return_pct": 1.0},
            {"type": "time", "horizon": "D+10", "return_pct": d10},
            {"type": "target", "level_pct": 5.0, "kind": "take_profit"},
        ],
        "hold_status": "HOLD_OBSERVING",
    }


def test_empty_candidates_review_generated() -> None:
    doc = dr.build_review_document("2026-06-05", "R4_NORMAL_BULL",
                                   dr.build_candidate_review([], {}),
                                   dr.build_execution_review([]),
                                   dr.build_exit_summary([]))
    assert doc["candidate_performance"]["candidate_count"] == 0
    assert dr.build_review_markdown(doc).strip()  # 후보 0건도 MD 생성


def test_empty_entries_review_generated() -> None:
    er = dr.build_execution_review([])
    assert er["entry_count"] == 0
    assert er["entries"] == []


def test_candidate_and_execution_use_different_basis() -> None:
    # 같은 종목: as_of 종가 1000(ohlcv) vs virtual_entry_price 950(observation)
    # 11행: D+10 종가 1100 → A raw_fwd D+10 = +10% (base 1000)
    ohlcv = _ohlcv([1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100])
    cr = dr.build_candidate_review([_cand("005930", "CORE")], {"005930": ohlcv})
    a_d10 = cr["candidates"][0]["raw_fwd_pct"]["D+10"]
    assert cr["candidates"][0]["as_of_close"] == 1000
    assert a_d10 == 10.0  # (1100-1000)/1000

    # B: virtual_entry 950 기준, observation의 time D+10 return_pct는 950 기준으로 별도 계산된 값
    er = dr.build_execution_review([_observation("005930", "CORE", entry=950, d10=15.79)])
    b_d10 = er["entries"][0]["pnl_pct"]["D+10"]
    assert er["entries"][0]["entry_price"] == 950
    assert b_d10 == 15.79
    assert cr["basis"] == "as_of_close"
    assert er["basis"] == "virtual_entry_price"
    assert a_d10 != b_d10  # 기준가가 다르므로 성과가 분리됨


def test_missed_winner_computed() -> None:
    # CONTROL 후보가 D+10 +10% → missed_winner
    ohlcv = _ohlcv([1000] + [1000] * 9 + [1100])
    cr = dr.build_candidate_review([_cand("011200", "CONTROL")], {"011200": ohlcv})
    assert len(cr["missed_winner"]) == 1
    assert cr["missed_winner"][0]["ticker"] == "011200"


def test_false_positive_computed() -> None:
    # CORE 후보가 D+10 -5% → false_positive
    ohlcv = _ohlcv([1000] + [1000] * 9 + [950])
    cr = dr.build_candidate_review([_cand("005930", "CORE")], {"005930": ohlcv})
    assert len(cr["false_positive"]) == 1
    assert cr["false_positive"][0]["d10_pct"] == -5.0


def test_exit_observer_connection() -> None:
    er = dr.build_execution_review([_observation("005930", "CORE", entry=1000, d10=5.0)])
    e = er["entries"][0]
    assert e["best_exit_candidate"] == {"type": "target", "level_pct": 10.0}
    assert e["worst_exit_candidate"] == {"type": "stop", "level_pct": -3.0}
    assert e["exit_signal_triggered"]  # 6단계 산출 연결
    summary = dr.build_exit_summary([_observation("005930", "CORE", entry=1000, d10=5.0)])
    assert summary["exit_type_trigger_counts"]["time"] == 2
    assert summary["exit_type_trigger_counts"]["target"] == 1


def test_safety_no_execution() -> None:
    doc = dr.build_review_document("2026-06-05", "R4_NORMAL_BULL",
                                   dr.build_candidate_review([], {}),
                                   dr.build_execution_review([]),
                                   dr.build_exit_summary([]))
    s = doc["safety"]
    assert s["real_order"] is False
    assert s["sell_automation"] == "BLOCKED"
    assert s["order_intent_created"] is False
    assert s["policy_changed"] is False
    assert s["scheduler_changed"] is False
    assert s["sajang_changed"] is False


def test_source_has_no_order_symbols() -> None:
    src = inspect.getsource(dr)
    forbidden = (
        "smart_sell", "sell_brain", "owner_rule", "sell_market", "sell_limit",
        "buy_limit", "order_intents_gate", "KisOrderAdapter", "place_order",
        "send_order", "create_market_sell_order", "create_limit_sell_order",
    )
    for f in forbidden:
        assert f not in src
