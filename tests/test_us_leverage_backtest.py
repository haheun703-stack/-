from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etf.us_leverage_backtest import prepare_us_pair, run_strategy


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="B"))


def test_us_leverage_c60_uses_previous_close_signal():
    benchmark = _series([100.0] * 59 + [101.0, 80.0, 120.0, 121.0])
    leverage = _series([100.0] * 59 + [102.0, 60.0, 140.0, 142.0])
    prepared = prepare_us_pair(benchmark, leverage)

    result = run_strategy(prepared, "B_LUMP_C60", "2026-01-01", "2026-12-31", cost=0.0)
    buyhold = run_strategy(prepared, "A_BUYHOLD", "2026-01-01", "2026-12-31", cost=0.0)

    assert result.trades >= 1
    assert result.final_return != buyhold.final_return


def test_us_split_strategy_never_invests_more_than_full_cash():
    benchmark = _series([100.0] * 59 + [101.0, 103.0, 98.0, 92.0, 87.0, 110.0])
    leverage = _series([100.0] * 59 + [102.0, 106.0, 90.0, 80.0, 70.0, 120.0])
    prepared = prepare_us_pair(benchmark, leverage)

    result = run_strategy(prepared, "C_SPLIT_DIP_C60", "2026-01-01", "2026-12-31", cost=0.0)

    assert result.trades > 0
    assert 0.0 <= result.end_invested_frac <= 1.0


def test_us_leverage_modules_do_not_reference_order_paths():
    root = Path(__file__).resolve().parent.parent
    source = (root / "src" / "etf" / "us_leverage_backtest.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "research" / "us_leverage_show_me.py").read_text(encoding="utf-8")
    combined = f"{source}\n{script}".lower()

    banned = [
        "mojito",
        "kis_order",
        "buy_limit",
        "sell_limit",
        "systemctl",
        "scheduler.service",
    ]
    for token in banned:
        assert token not in combined
