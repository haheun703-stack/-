"""US leveraged ETF C60/split-buy SHOW ME report.

Outputs:
- data/reports/us_leverage/us_leverage_show_me.json
- docs/02-design/assets/us_leverage_backtest.png
- docs/02-design/us-leverage-c60-split-6_3.md

Read-only analytics. No real orders, no scheduler/systemd changes.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import FinanceDataReader as fdr
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.etf.us_leverage_backtest import (  # noqa: E402
    REPORT_DIR,
    US_LEVERAGE_ASSETS,
    WINDOWS,
    normalize_close,
    prepare_us_pair,
    run_strategy,
    summarize_window,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_PATH = PROJECT_ROOT / "docs" / "02-design" / "us-leverage-c60-split-6_3.md"
ASSET_DIR = PROJECT_ROOT / "docs" / "02-design" / "assets"
PNG_PATH = ASSET_DIR / "us_leverage_backtest.png"
JSON_PATH = REPORT_DIR / "us_leverage_show_me.json"

STRATEGY_LABEL = {
    "A_BUYHOLD": "A buyhold",
    "B_LUMP_C60": "B lump+C60",
    "C_SPLIT_DIP_C60": "C split+C60",
}
STRATEGY_COLOR = {
    "A_BUYHOLD": "#8c8c8c",
    "B_LUMP_C60": "#1f77b4",
    "C_SPLIT_DIP_C60": "#d62728",
}


def fetch_close(ticker: str, start: str = "2018-01-01", end: str = "2026-06-02") -> pd.Series:
    df = fdr.DataReader(ticker, start, end)
    close = normalize_close(df)
    if close.empty:
        raise RuntimeError(f"no data for {ticker}")
    return close


def build_asset_payload(ticker: str, meta: dict) -> dict:
    benchmark = fetch_close(meta["benchmark"])
    leverage = fetch_close(ticker)
    prepared_by_signal = {
        "self_c60": prepare_us_pair(leverage, leverage),
        "benchmark_c60": prepare_us_pair(benchmark, leverage),
    }
    primary = prepared_by_signal["self_c60"]
    windows = {basis: {} for basis in prepared_by_signal}
    curves = {}
    for basis, prepared in prepared_by_signal.items():
        for window_name, (start, end) in WINDOWS.items():
            windows[basis][window_name] = summarize_window(prepared, start, end)
    for window_name, (start, end) in WINDOWS.items():
        curves[window_name] = {
            strategy: run_strategy(primary, strategy, start, end)
            for strategy in ("A_BUYHOLD", "B_LUMP_C60", "C_SPLIT_DIP_C60")
        }
    return {
        "ticker": ticker,
        "name": meta["name"],
        "benchmark": meta["benchmark"],
        "leverage": meta["leverage"],
        "primary_signal_basis": "self_c60",
        "sensitivity_signal_basis": "benchmark_c60",
        "data_start": primary.index[0].strftime("%Y-%m-%d"),
        "data_end": primary.index[-1].strftime("%Y-%m-%d"),
        "windows": windows,
        "_curves": curves,
    }


def save_json(payload: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload, default=lambda value: value.__dict__))
    for asset in serializable["assets"].values():
        asset.pop("_curves", None)
    JSON_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def fig_show_me(payload: dict) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    assets = list(payload["assets"].keys())
    fig, axes = plt.subplots(len(assets), 4, figsize=(18, 4.2 * len(assets)))
    if len(assets) == 1:
        axes = [axes]

    for row_idx, ticker in enumerate(assets):
        asset = payload["assets"][ticker]
        for col_idx, window_name in enumerate(("ai_rally_2023_2026", "bear_2022")):
            curves = asset["_curves"][window_name]
            ax_eq = axes[row_idx][col_idx * 2]
            ax_dd = axes[row_idx][col_idx * 2 + 1]
            for strategy, result in curves.items():
                dates = pd.to_datetime(result.dates)
                ax_eq.plot(
                    dates,
                    result.equity,
                    color=STRATEGY_COLOR[strategy],
                    label=f"{STRATEGY_LABEL[strategy]} {pct(result.final_return)}",
                    lw=1.7 if strategy != "A_BUYHOLD" else 1.1,
                    alpha=0.9 if strategy != "A_BUYHOLD" else 0.65,
                )
                ax_dd.plot(
                    dates,
                    [v * 100 for v in result.drawdown],
                    color=STRATEGY_COLOR[strategy],
                    label=f"{STRATEGY_LABEL[strategy]} MDD {pct(result.mdd)}",
                    lw=1.5,
                    alpha=0.85,
                )
            ax_eq.set_yscale("log")
            ax_eq.set_title(f"{ticker} {window_name} equity")
            ax_eq.grid(alpha=0.25)
            ax_eq.legend(fontsize=7)
            ax_dd.set_title(f"{ticker} {window_name} drawdown")
            ax_dd.grid(alpha=0.25)
            ax_dd.legend(fontsize=7)

    fig.suptitle("US Leveraged ETF Backtest: buyhold vs lump+C60 vs split+C60", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(PNG_PATH, dpi=115)
    plt.close(fig)


def save_markdown(payload: dict) -> None:
    lines = [
        "# US Leverage C60 Split-Buy SHOW ME",
        "",
        "- Scope: SOXL, TQQQ, QLD, USD",
        "- Price basis: actual leveraged ETF close",
        "- Primary signal basis: leveraged ETF self close > MA60, applied from next observed trading day",
        "- Sensitivity: benchmark ETF close > MA60 is also stored because benchmark signals can lag leveraged products in crashes.",
        "- Safety: real orders 0 / HOLD / read-only / scheduler untouched",
        "",
        f"![US leverage backtest](assets/{PNG_PATH.name})",
        "",
        "## Summary",
        "",
        "| Ticker | Benchmark | Signal | Window | A buyhold | B lump+C60 | C split+C60 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for ticker, asset in payload["assets"].items():
        for basis in ("self_c60", "benchmark_c60"):
            for window_name, summary in asset["windows"][basis].items():
                cells = []
                for strategy in ("A_BUYHOLD", "B_LUMP_C60", "C_SPLIT_DIP_C60"):
                    row = summary[strategy]
                    cells.append(f"{pct(row['final_return'])} / {pct(row['mdd'])}")
                lines.append(
                    f"| {ticker} | {asset['benchmark']} | {basis} | {window_name} | "
                    f"{cells[0]} | {cells[1]} | {cells[2]} |"
                )
    lines += [
        "",
        "## Read",
        "",
        "- Bull regimes reward buyhold, especially 3x products.",
        "- 2022 stress shows why unhedged 2x/3x can destroy capital.",
        "- For actual US leveraged ETFs, self C60 is the primary backtest basis.",
        "- Benchmark C60 is useful as a sensitivity check, but it can exit late in leverage crashes.",
        "- C split-buy is not a V-shaped crash shield. It is a trend-bear survival rule.",
        "- Macro early-warning signals should be validated separately before becoming hard gates.",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = {
        "status": "SHOW_ME_READ_ONLY",
        "windows": WINDOWS,
        "assets": {},
        "safety": {
            "order_count": 0,
            "live_trading_state": "HOLD",
            "scheduler_touched": False,
        },
    }
    for ticker, meta in US_LEVERAGE_ASSETS.items():
        payload["assets"][ticker] = build_asset_payload(ticker, meta)
    save_json(payload)
    fig_show_me(payload)
    save_markdown(payload)
    print(json.dumps(json.loads(JSON_PATH.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    print(f"[US-LEV] JSON: {JSON_PATH}")
    print(f"[US-LEV] PNG: {PNG_PATH}")
    print(f"[US-LEV] MD: {DOC_PATH}")
    print("[US-LEV] Safety proof: real orders 0 / HOLD / scheduler untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
