from __future__ import annotations

from src.use_cases.semiconductor_self_valuation import (
    ScenarioAssumption,
    StockInputs,
    build_stock_valuation,
    calculate_weighted_etf_scenarios,
)


def _stock(
    ticker: str = "000660",
    q1_op: float = 100.0,
    prev_q1_op: float = 20.0,
    pbr: float = 12.0,
) -> StockInputs:
    return StockInputs(
        ticker=ticker,
        name=ticker,
        current_price=100.0,
        market_cap=1_000.0,
        shares=10.0,
        bps=10.0,
        trailing_per=8.0,
        trailing_pbr=pbr,
        q1_revenue=200.0,
        q1_op=q1_op,
        q1_net=80.0,
        prev_q1_revenue=100.0,
        prev_q1_op=prev_q1_op,
        equity=120.0,
        ttm_op_2025q3=300.0,
    )


def test_build_stock_valuation_flags_extreme_pbr_and_base_downside():
    valuation = build_stock_valuation(
        _stock(),
        [
            ScenarioAssumption("bear", op_factor=1.0, per=1.0, pbr=1.0),
            ScenarioAssumption("base", op_factor=1.0, per=1.0, pbr=1.0),
            ScenarioAssumption("bull", op_factor=1.0, per=1.0, pbr=1.0),
        ],
    )

    assert "EXTREME_PBR" in valuation.flags
    assert "BASE_DOWNSIDE" in valuation.flags
    assert valuation.q1_op_yoy == 4.0


def test_q1_op_shock_and_ttm_floor_are_applied():
    valuation = build_stock_valuation(
        _stock(ticker="042700", q1_op=10.0, prev_q1_op=100.0, pbr=30.0),
        [
            ScenarioAssumption("bear", op_factor=3.0, per=25.0, pbr=6.0),
            ScenarioAssumption("base", op_factor=4.0, per=35.0, pbr=10.0, ttm_floor_factor=0.7),
            ScenarioAssumption("bull", op_factor=6.0, per=50.0, pbr=15.0, ttm_floor_factor=1.2),
        ],
    )

    base = next(item for item in valuation.scenarios if item.scenario == "base")
    bull = next(item for item in valuation.scenarios if item.scenario == "bull")

    assert "Q1_OP_SHOCK" in valuation.flags
    assert base.valuation_note == "ttm_floor"
    assert base.op26 == 210.0
    assert bull.op26 == 360.0


def test_calculate_weighted_etf_scenarios_uses_exposure_weights():
    sk = build_stock_valuation(
        _stock("000660"),
        [ScenarioAssumption("base", op_factor=3.0, per=2.0, pbr=2.0)],
    )
    samsung = build_stock_valuation(
        _stock("005930"),
        [ScenarioAssumption("base", op_factor=2.0, per=1.0, pbr=1.0)],
    )

    report = calculate_weighted_etf_scenarios(
        {"000660": sk, "005930": samsung},
        {"000660": 0.4, "005930": 0.3},
        leverage=2.0,
    )

    assert report["base"]["covered_exposure"] == 0.7
    assert report["base"]["rough_leveraged_effect"] == report["base"]["underlying_effect"] * 2
