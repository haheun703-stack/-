"""US leveraged ETF C60 and split-buy backtest helpers.

Read-only analytics:
- Uses actual leveraged ETF closes for equity curves.
- Uses the unlevered benchmark ETF close/MA60 for C60 regime decisions.
- Applies after-close signals from the next observed trading day.
- Never imports or calls broker/order/scheduler paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "us_leverage"

COST = 0.001
MA_PERIOD = 60
TRANCHE_C = (0.30, 0.20, 0.20, 0.30)
DIP_TRIGGERS = (None, -0.05, -0.10, -0.15)

US_LEVERAGE_ASSETS: dict[str, dict] = {
    "SOXL": {"name": "Direxion Daily Semiconductor Bull 3X", "benchmark": "SOXX", "leverage": 3.0},
    "TQQQ": {"name": "ProShares UltraPro QQQ 3X", "benchmark": "QQQ", "leverage": 3.0},
    "QLD": {"name": "ProShares Ultra QQQ 2X", "benchmark": "QQQ", "leverage": 2.0},
    "USD": {"name": "ProShares Ultra Semiconductors 2X", "benchmark": "SOXX", "leverage": 2.0},
}

WINDOWS: dict[str, tuple[str, str]] = {
    "ai_rally_2023_2026": ("2023-01-03", "2026-06-02"),
    "bear_2022": ("2022-01-03", "2022-12-30"),
}


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    final_return: float
    mdd: float
    trades: int
    end_invested_frac: float
    worst_drawdown_date: str | None
    dates: list[str]
    equity: list[float]
    drawdown: list[float]


def normalize_close(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if column not in df.columns:
        return pd.Series(dtype=float)
    out = pd.to_numeric(df[column], errors="coerce")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().dropna()
    out = out[out > 0]
    return out.astype(float)


def prepare_us_pair(
    benchmark_close: pd.Series,
    leverage_close: pd.Series,
    ma_period: int = MA_PERIOD,
) -> pd.DataFrame:
    benchmark = pd.to_numeric(benchmark_close, errors="coerce").dropna()
    leverage = pd.to_numeric(leverage_close, errors="coerce").dropna()
    benchmark.index = pd.to_datetime(benchmark.index)
    leverage.index = pd.to_datetime(leverage.index)

    df = pd.DataFrame(
        {
            "benchmark_close": benchmark,
            "leverage_close": leverage,
        }
    ).dropna()
    df = df.sort_index()
    if df.empty:
        return df

    df["benchmark_ma60"] = df["benchmark_close"].rolling(ma_period).mean()
    signal = (df["benchmark_close"] > df["benchmark_ma60"]).astype(bool)
    df["bull_signal_next_day"] = signal.shift(1, fill_value=False).astype(bool)
    df["leverage_return"] = df["leverage_close"].pct_change().fillna(0.0)
    return df.dropna(subset=["benchmark_ma60"])


def _drawdown(curve: list[float]) -> tuple[list[float], float, str | None]:
    peak = curve[0]
    worst = 0.0
    worst_idx = None
    drawdowns: list[float] = []
    for idx, value in enumerate(curve):
        peak = max(peak, value)
        dd = value / peak - 1.0 if peak > 0 else 0.0
        drawdowns.append(dd)
        if dd < worst:
            worst = dd
            worst_idx = idx
    return drawdowns, worst, str(worst_idx) if worst_idx is not None else None


def run_strategy(
    prepared: pd.DataFrame,
    strategy: str,
    start: str,
    end: str,
    cost: float = COST,
) -> BacktestResult:
    window = prepared.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()
    if window.empty:
        return BacktestResult(strategy, 0.0, 0.0, 0, 0.0, None, [], [], [])

    shares = 0.0
    cash = 1.0
    tranches = 0
    peak_price: float | None = None
    trades = 0
    invested_nominal = 0.0
    equity: list[float] = []
    dates: list[str] = []

    def buy(frac: float, price: float) -> None:
        nonlocal shares, cash, trades, invested_nominal
        amount = min(frac, cash)
        if amount <= 1e-12:
            return
        shares += amount * (1.0 - cost) / price
        cash -= amount
        invested_nominal += amount
        trades += 1

    def sell_all(price: float) -> None:
        nonlocal shares, cash, tranches, peak_price, trades, invested_nominal
        if shares <= 1e-12:
            return
        cash += shares * price * (1.0 - cost)
        shares = 0.0
        tranches = 0
        peak_price = None
        invested_nominal = 0.0
        trades += 1

    for idx, (dt, row) in enumerate(window.iterrows()):
        price = float(row["leverage_close"])
        bull = bool(row["bull_signal_next_day"])

        if strategy == "A_BUYHOLD":
            if idx == 0:
                buy(1.0, price)

        elif strategy == "B_LUMP_C60":
            if bull and shares <= 1e-12:
                buy(1.0, price)
                peak_price = price
            elif not bull and shares > 1e-12:
                sell_all(price)

        elif strategy == "C_SPLIT_DIP_C60":
            if bull:
                if tranches == 0:
                    buy(TRANCHE_C[0], price)
                    tranches = 1
                    peak_price = price
                else:
                    peak_price = max(float(peak_price or price), price)
                    while tranches < len(TRANCHE_C):
                        trigger = DIP_TRIGGERS[tranches]
                        if trigger is None or price > peak_price * (1.0 + trigger):
                            break
                        buy(TRANCHE_C[tranches], price)
                        tranches += 1
            elif shares > 1e-12:
                sell_all(price)

        else:
            raise ValueError(f"unknown strategy: {strategy}")

        equity.append(cash + shares * price)
        dates.append(pd.Timestamp(dt).strftime("%Y-%m-%d"))

    drawdowns, mdd, worst_idx = _drawdown(equity)
    worst_date = dates[int(worst_idx)] if worst_idx is not None else None
    return BacktestResult(
        strategy=strategy,
        final_return=round(equity[-1] - 1.0, 6),
        mdd=round(mdd, 6),
        trades=trades,
        end_invested_frac=round(min(invested_nominal, 1.0), 6),
        worst_drawdown_date=worst_date,
        dates=dates,
        equity=[round(v, 6) for v in equity],
        drawdown=[round(v, 6) for v in drawdowns],
    )


def summarize_window(prepared: pd.DataFrame, start: str, end: str) -> dict:
    results = {
        strategy: run_strategy(prepared, strategy, start, end)
        for strategy in ("A_BUYHOLD", "B_LUMP_C60", "C_SPLIT_DIP_C60")
    }
    return {
        key: {
            "final_return": value.final_return,
            "mdd": value.mdd,
            "trades": value.trades,
            "end_invested_frac": value.end_invested_frac,
            "worst_drawdown_date": value.worst_drawdown_date,
        }
        for key, value in results.items()
    }
