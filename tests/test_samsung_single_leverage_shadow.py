from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etf.samsung_single_leverage_shadow import (
    build_samsung_single_leverage_report,
    build_samsung_single_leverage_shadow_ledger,
    prepare_shadow_prices,
)


def _prices(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"close": values}, index=idx)


def test_prepare_shadow_prices_builds_synthetic_2x_when_product_missing():
    values = [100.0] * 59 + [101.0, 110.0, 99.0, 108.9]
    df = prepare_shadow_prices(_prices(values), leverage_prices=None, multiplier=2.0)

    assert not df.empty
    assert "leverage_close" in df.columns
    assert df["leverage_close"].iloc[0] > 0


def test_samsung_shadow_tracks_c60_and_sajang_rules_independently():
    values = [100.0] * 59 + [101.0]
    values += [110.0, 112.0, 80.0, 105.0, 108.0, 111.0, 115.0, 120.0, 125.0]

    rows = build_samsung_single_leverage_shadow_ledger(_prices(values))
    c60_signals = [row.c60_signal for row in rows]
    sajang_signals = [row.sajang_signal for row in rows]

    assert "EXIT" in c60_signals
    assert "EXIT" in sajang_signals
    assert rows[-1].leverage_ticker == "0193W0"
    assert rows[-1].underlying_buyhold_equity_curve > 0


def test_samsung_shadow_report_is_shadow_only():
    values = [100.0] * 59 + [101.0, 110.0, 112.0, 80.0, 120.0, 121.0]
    rows = build_samsung_single_leverage_shadow_ledger(_prices(values))
    reference = {"ticker": "488080", "latest_c60_position_state": "HOLD"}
    report = build_samsung_single_leverage_report(rows, c60_488080_reference=reference)

    assert report["status"] == "SHADOW_ONLY"
    assert report["order_count"] == 0
    assert report["live_trading_state"] == "HOLD"
    assert report["signal_ticker"] == "005930"
    assert report["c60_488080_reference"] == reference
    assert "488080_c60" in report["comparison_basis"]


def test_samsung_shadow_modules_do_not_reference_order_paths():
    root = Path(__file__).resolve().parent.parent
    source = (root / "src" / "etf" / "samsung_single_leverage_shadow.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "samsung_single_leverage_shadow.py").read_text(encoding="utf-8")
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
