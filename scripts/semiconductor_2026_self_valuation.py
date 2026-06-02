#!/usr/bin/env python3
"""2026 semiconductor self valuation.

Inputs:
- KRX/pykrx market cap, PER, PBR, ETF PDF.
- OpenDART 2026 Q1 financial statements.
- Local 2025Q3 financial cache for TTM fallback.

This script is analytics only. It does not import or call order adapters and
does not touch scheduler/systemd.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.dart_adapter import DartAdapter  # noqa: E402
from src.use_cases.semiconductor_self_valuation import (  # noqa: E402
    ScenarioAssumption,
    StockInputs,
    build_stock_valuation,
    calculate_weighted_etf_scenarios,
    stock_valuation_to_dict,
)

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEMI2026] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("semiconductor_2026")

TARGET_STOCKS = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "042700": "Hanmi Semiconductor",
}

ASSUMPTIONS = {
    "005930": [
        ScenarioAssumption("bear", op_factor=2.8, per=8.0, pbr=2.2),
        ScenarioAssumption("base", op_factor=3.4, per=10.0, pbr=3.0),
        ScenarioAssumption("bull", op_factor=4.0, per=12.0, pbr=4.0),
    ],
    "000660": [
        ScenarioAssumption("bear", op_factor=3.0, per=8.0, pbr=4.0),
        ScenarioAssumption("base", op_factor=3.6, per=10.0, pbr=7.0),
        ScenarioAssumption("bull", op_factor=4.2, per=12.0, pbr=10.0),
    ],
    "042700": [
        ScenarioAssumption("bear", op_factor=3.0, per=25.0, pbr=6.0),
        ScenarioAssumption("base", op_factor=4.0, per=35.0, pbr=10.0, ttm_floor_factor=0.7),
        ScenarioAssumption("bull", op_factor=6.0, per=50.0, pbr=15.0, ttm_floor_factor=1.2),
    ],
}

FINANCIAL_CACHE_PATH = PROJECT_ROOT / "data" / "v2_migration" / "financial_quarterly.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "reports" / "semiconductor_2026"


def parse_amount(value) -> float:
    return float(str(value).replace(",", ""))


def load_financial_cache() -> dict:
    with open(FINANCIAL_CACHE_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def ttm_op_2025q3(cache: dict, ticker: str) -> float:
    bs = cache["bs_data"][ticker]
    return (
        bs["2025Q3"]["op_income_cum"]
        + bs["2024Q4"]["op_income_cum"]
        - bs["2024Q3"]["op_income_cum"]
    )


def extract_dart_cfs_by_position(dart_df, ticker: str) -> dict:
    cfs = dart_df[(dart_df.stock_code == ticker) & (dart_df.fs_div == "CFS")].reset_index(drop=True)
    if len(cfs) < 13:
        raise ValueError(f"{ticker} CFS rows are insufficient: {len(cfs)}")

    # OpenDART fnlttMultiAcnt standard CFS order for this response:
    # 8 equity, 9 revenue, 10 operating income, 12 net income.
    return {
        "equity": parse_amount(cfs.loc[8, "thstrm_amount"]),
        "q1_revenue": parse_amount(cfs.loc[9, "thstrm_amount"]),
        "q1_op": parse_amount(cfs.loc[10, "thstrm_amount"]),
        "q1_net": parse_amount(cfs.loc[12, "thstrm_amount"]),
        "prev_q1_revenue": parse_amount(cfs.loc[9, "frmtrm_amount"]),
        "prev_q1_op": parse_amount(cfs.loc[10, "frmtrm_amount"]),
    }


def fetch_market_inputs(tickers: list[str], date_str: str):
    from pykrx import stock

    cap = stock.get_market_cap_by_ticker(date_str, market="ALL").loc[tickers]
    fundamental = stock.get_market_fundamental_by_ticker(date_str, market="ALL").loc[tickers]
    return cap, fundamental


def build_stock_inputs(date_str: str) -> dict[str, StockInputs]:
    tickers = list(TARGET_STOCKS)
    cap, fundamental = fetch_market_inputs(tickers, date_str)
    dart = DartAdapter()
    dart_df = dart.fetch_multi_financials(tickers, 2026, "11013")
    if dart_df is None or dart_df.empty:
        raise RuntimeError("OpenDART 2026Q1 response is empty")

    cache = load_financial_cache()
    output: dict[str, StockInputs] = {}

    for ticker in tickers:
        dart_items = extract_dart_cfs_by_position(dart_df, ticker)
        cap_row = cap.loc[ticker]
        fund_row = fundamental.loc[ticker]
        output[ticker] = StockInputs(
            ticker=ticker,
            name=TARGET_STOCKS[ticker],
            current_price=float(cap_row.iloc[0]),
            market_cap=float(cap_row.iloc[1]),
            shares=float(cap_row.iloc[4]),
            bps=float(fund_row.iloc[0]),
            trailing_per=float(fund_row.iloc[1]),
            trailing_pbr=float(fund_row.iloc[2]),
            q1_revenue=dart_items["q1_revenue"],
            q1_op=dart_items["q1_op"],
            q1_net=dart_items["q1_net"],
            prev_q1_revenue=dart_items["prev_q1_revenue"],
            prev_q1_op=dart_items["prev_q1_op"],
            equity=dart_items["equity"],
            ttm_op_2025q3=ttm_op_2025q3(cache, ticker),
        )
    return output


def fetch_488080_lookthrough_exposure(date_str: str) -> dict[str, float]:
    from pykrx import stock

    levered_pdf = stock.get_etf_portfolio_deposit_file("488080", date_str)
    base_pdf = stock.get_etf_portfolio_deposit_file("396500", date_str)
    positive = levered_pdf[levered_pdf.iloc[:, 1].astype(float) > 0].copy()
    positive = positive[positive.index != "010010"]
    total_amount = positive.iloc[:, 1].astype(float).sum()
    if total_amount <= 0:
        return {}

    exposure = {ticker: 0.0 for ticker in TARGET_STOCKS}
    for ticker in TARGET_STOCKS:
        if ticker in positive.index:
            exposure[ticker] += float(positive.loc[ticker].iloc[1]) / total_amount

    # Main futures rows observed in KRX PDF for 488080.
    futures_map = {
        "A50660": "000660",
        "A11660": "005930",
        "A0Z660": "042700",
    }
    for futures_code, ticker in futures_map.items():
        if futures_code in positive.index:
            exposure[ticker] += float(positive.loc[futures_code].iloc[1]) / total_amount

    if "396500" in positive.index:
        base_weight = float(positive.loc["396500"].iloc[1]) / total_amount
        for ticker in TARGET_STOCKS:
            if ticker in base_pdf.index:
                exposure[ticker] += base_weight * float(base_pdf.loc[ticker].iloc[4]) / 100.0

    return exposure


def won(value: float) -> str:
    return f"{value:,.0f}"


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%"


def trillion(value: float) -> str:
    return f"{value / 1e12:,.2f}T"


def render_markdown(report: dict) -> str:
    lines = [
        "# 2026 Semiconductor Self Valuation v2",
        "",
        f"- Date: {report['date']}",
        "- Sources: OpenDART 2026Q1, KRX/pykrx market cap/fundamental, KRX ETF PDF",
        "- Scope: Samsung Electronics, SK hynix, Hanmi Semiconductor, 488080 look-through",
        "- Trading safety: real orders 0, HOLD, scheduler/systemd untouched",
        "",
        "## Input Snapshot",
        "",
        "| Ticker | Name | Current | Market Cap | PBR | Q1 OP | Q1 OP YoY | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for ticker, stock in report["stocks"].items():
        lines.append(
            "| {ticker} | {name} | {price} | {mcap} | {pbr:.2f} | {q1op} | {yoy} | {flags} |".format(
                ticker=ticker,
                name=stock["name"],
                price=won(stock["current_price"]),
                mcap=trillion(stock["market_cap"]),
                pbr=stock["trailing_pbr"],
                q1op=trillion(stock["q1_op"]),
                yoy=pct(stock["q1_op_yoy"]),
                flags=", ".join(stock["flags"]) or "-",
            )
        )

    lines += [
        "",
        "## Scenario Targets",
        "",
        "| Ticker | Scenario | OP26 | PER/PBR | Target | Upside | Note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for ticker, stock in report["stocks"].items():
        for scenario in stock["scenarios"]:
            lines.append(
                "| {ticker} | {scenario} | {op26} | {per:.1f}/{pbr:.1f} | {target} | {upside} | {note} |".format(
                    ticker=ticker,
                    scenario=scenario["scenario"],
                    op26=trillion(scenario["op26"]),
                    per=scenario["per"],
                    pbr=scenario["pbr"],
                    target=won(scenario["target_price"]),
                    upside=pct(scenario["upside_pct"]),
                    note=scenario["valuation_note"],
                )
            )

    lines += [
        "",
        "## 488080 Look-Through",
        "",
        "| Ticker | Exposure |",
        "|---|---:|",
    ]
    for ticker, value in report["etf_488080"]["lookthrough_exposure"].items():
        lines.append(f"| {ticker} | {value * 100:.2f}% |")

    lines += [
        "",
        "| Scenario | Covered Exposure | Underlying Effect | Rough 2x Effect |",
        "|---|---:|---:|---:|",
    ]
    for scenario, value in report["etf_488080"]["scenario_effect"].items():
        lines.append(
            "| {scenario} | {covered:.2f}% | {underlying} | {levered} |".format(
                scenario=scenario,
                covered=value["covered_exposure"] * 100,
                underlying=pct(value["underlying_effect"]),
                levered=pct(value["rough_leveraged_effect"]),
            )
        )

    lines += [
        "",
        "## One-Line Read",
        "",
        report["one_line_read"],
        "",
        "## Caveats",
        "",
        "- This model uses explicit internal assumptions, not analyst target prices.",
        "- HBM allocation rumors are not treated as confirmed revenue backlog.",
        "- 488080 effect is a one-period rough look-through, not a path-dependent leveraged ETF backtest.",
    ]
    return "\n".join(lines) + "\n"


def build_report(date_str: str) -> dict:
    inputs = build_stock_inputs(date_str)
    valuations = {
        ticker: build_stock_valuation(stock_input, ASSUMPTIONS[ticker])
        for ticker, stock_input in inputs.items()
    }
    exposure = fetch_488080_lookthrough_exposure(date_str)
    etf_scenarios = calculate_weighted_etf_scenarios(valuations, exposure, leverage=2.0)

    report = {
        "date": date_str,
        "status": "SELF_VALUATION_V2",
        "stocks": {
            ticker: stock_valuation_to_dict(valuation)
            for ticker, valuation in valuations.items()
        },
        "etf_488080": {
            "lookthrough_exposure": exposure,
            "scenario_effect": etf_scenarios,
        },
        "assumption_note": "Info-bot confirmed events are reflected through scenario factors; unconfirmed HBM allocation rumors are not treated as hard backlog.",
        "one_line_read": "Strong earnings momentum is real, but PBR-adjusted safety margin is thin; 488080 still needs C60-style risk control.",
        "order_count": 0,
        "live_trading_state": "HOLD",
        "safety_note": "real orders 0 / HOLD maintained / scheduler and systemd untouched",
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2026 semiconductor self valuation v2")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.date)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"semiconductor_2026_self_valuation_{args.date}.json"
    md_path = args.output_dir / f"semiconductor_2026_self_valuation_{args.date}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    logger.info("JSON saved: %s", json_path)
    logger.info("Markdown saved: %s", md_path)
    logger.info("Safety proof: real orders 0 / HOLD / scheduler untouched")

    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
