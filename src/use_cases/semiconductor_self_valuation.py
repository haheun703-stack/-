"""Self valuation helpers for the 2026 semiconductor cycle check.

This module is pure calculation code. It does not import broker, order,
scheduler, or network adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ScenarioName = Literal["bear", "base", "bull"]


@dataclass(frozen=True)
class StockInputs:
    ticker: str
    name: str
    current_price: float
    market_cap: float
    shares: float
    bps: float
    trailing_per: float
    trailing_pbr: float
    q1_revenue: float
    q1_op: float
    q1_net: float
    prev_q1_revenue: float
    prev_q1_op: float
    equity: float
    ttm_op_2025q3: float


@dataclass(frozen=True)
class ScenarioAssumption:
    name: ScenarioName
    op_factor: float
    per: float
    pbr: float
    earn_weight: float = 0.6
    pbr_weight: float = 0.4
    ttm_floor_factor: float | None = None


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ScenarioName
    op26: float
    net26: float
    net_conversion: float
    earn_price: float
    pbr_price: float
    target_price: float
    upside_pct: float
    target_market_cap: float
    per: float
    pbr: float
    valuation_note: str


@dataclass(frozen=True)
class StockValuation:
    ticker: str
    name: str
    current_price: float
    market_cap: float
    trailing_per: float
    trailing_pbr: float
    q1_revenue: float
    q1_op: float
    q1_net: float
    q1_revenue_yoy: float | None
    q1_op_yoy: float | None
    equity: float
    q1_annualized_op: float
    ttm_op_2025q3: float
    scenarios: list[ScenarioResult]
    flags: list[str]


def pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return current / previous - 1.0


def bounded_net_conversion(q1_net: float, q1_op: float) -> float:
    if q1_op <= 0:
        return 0.7
    return max(0.65, min(0.85, q1_net / q1_op))


def calculate_scenario(
    stock: StockInputs,
    assumption: ScenarioAssumption,
) -> ScenarioResult:
    net_conversion = bounded_net_conversion(stock.q1_net, stock.q1_op)
    direct_op = stock.q1_op * assumption.op_factor
    note = "q1_factor"

    if assumption.ttm_floor_factor is not None:
        ttm_floor = stock.ttm_op_2025q3 * assumption.ttm_floor_factor
        if ttm_floor > direct_op:
            direct_op = ttm_floor
            note = "ttm_floor"

    net26 = direct_op * net_conversion
    earn_market_cap = net26 * assumption.per
    pbr_market_cap = stock.equity * assumption.pbr
    blended_market_cap = (
        earn_market_cap * assumption.earn_weight
        + pbr_market_cap * assumption.pbr_weight
    )
    target_price = blended_market_cap / stock.shares if stock.shares > 0 else 0.0
    upside_pct = (
        target_price / stock.current_price - 1.0
        if stock.current_price > 0
        else 0.0
    )

    return ScenarioResult(
        scenario=assumption.name,
        op26=direct_op,
        net26=net26,
        net_conversion=net_conversion,
        earn_price=earn_market_cap / stock.shares if stock.shares > 0 else 0.0,
        pbr_price=pbr_market_cap / stock.shares if stock.shares > 0 else 0.0,
        target_price=target_price,
        upside_pct=upside_pct,
        target_market_cap=blended_market_cap,
        per=assumption.per,
        pbr=assumption.pbr,
        valuation_note=note,
    )


def build_stock_valuation(
    stock: StockInputs,
    assumptions: list[ScenarioAssumption],
) -> StockValuation:
    scenarios = [calculate_scenario(stock, assumption) for assumption in assumptions]
    flags = []
    q1_op_yoy = pct_change(stock.q1_op, stock.prev_q1_op)
    q1_revenue_yoy = pct_change(stock.q1_revenue, stock.prev_q1_revenue)

    if stock.trailing_pbr >= 10:
        flags.append("EXTREME_PBR")
    elif stock.trailing_pbr >= 5:
        flags.append("HIGH_PBR")
    if q1_op_yoy is not None and q1_op_yoy < -0.5:
        flags.append("Q1_OP_SHOCK")
    if scenarios:
        base = next((s for s in scenarios if s.scenario == "base"), scenarios[0])
        if base.upside_pct < 0:
            flags.append("BASE_DOWNSIDE")
        if max(s.upside_pct for s in scenarios) < 0:
            flags.append("NO_UPSIDE_EVEN_BULL")

    return StockValuation(
        ticker=stock.ticker,
        name=stock.name,
        current_price=stock.current_price,
        market_cap=stock.market_cap,
        trailing_per=stock.trailing_per,
        trailing_pbr=stock.trailing_pbr,
        q1_revenue=stock.q1_revenue,
        q1_op=stock.q1_op,
        q1_net=stock.q1_net,
        q1_revenue_yoy=q1_revenue_yoy,
        q1_op_yoy=q1_op_yoy,
        equity=stock.equity,
        q1_annualized_op=stock.q1_op * 4,
        ttm_op_2025q3=stock.ttm_op_2025q3,
        scenarios=scenarios,
        flags=flags,
    )


def calculate_weighted_etf_scenarios(
    valuations: dict[str, StockValuation],
    exposure: dict[str, float],
    leverage: float = 2.0,
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for scenario in ("bear", "base", "bull"):
        underlying_effect = 0.0
        contributors = {}
        for ticker, weight in exposure.items():
            valuation = valuations.get(ticker)
            if valuation is None:
                continue
            scenario_result = next(
                (item for item in valuation.scenarios if item.scenario == scenario),
                None,
            )
            if scenario_result is None:
                continue
            contribution = weight * scenario_result.upside_pct
            contributors[ticker] = contribution
            underlying_effect += contribution

        output[scenario] = {
            "covered_exposure": sum(exposure.values()),
            "underlying_effect": underlying_effect,
            "rough_leveraged_effect": underlying_effect * leverage,
            "contributors": contributors,
        }
    return output


def stock_valuation_to_dict(valuation: StockValuation) -> dict:
    payload = asdict(valuation)
    payload["scenarios"] = [asdict(item) for item in valuation.scenarios]
    return payload
