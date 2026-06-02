from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.etf.samsung_single_leverage_shadow import (
    SK_HYNIX_TICKER,
    build_common_period_comparison,
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


def test_single_stock_shadow_accepts_sk_hynix_signal_ticker():
    values = [100.0] * 59 + [101.0, 110.0, 112.0, 80.0, 120.0, 121.0]
    rows = build_samsung_single_leverage_shadow_ledger(
        _prices(values),
        signal_ticker=SK_HYNIX_TICKER,
        leverage_ticker="0193T0",
    )

    assert rows[-1].signal_ticker == "000660"
    assert rows[-1].leverage_ticker == "0193T0"


def test_samsung_shadow_report_is_shadow_only():
    values = [100.0] * 59 + [101.0, 110.0, 112.0, 80.0, 120.0, 121.0]
    rows = build_samsung_single_leverage_shadow_ledger(_prices(values))
    reference = {"ticker": "488080", "latest_c60_position_state": "HOLD"}
    fair_compare = {"status": "FAIR_COMMON_PERIOD"}
    report = build_samsung_single_leverage_report(
        rows,
        c60_488080_reference=reference,
        common_period_comparison=fair_compare,
    )

    assert report["status"] == "SHADOW_ONLY"
    assert report["order_count"] == 0
    assert report["live_trading_state"] == "HOLD"
    assert report["signal_ticker"] == "005930"
    assert report["c60_488080_reference_full_available_period"] == reference
    assert report["common_period_fair_comparison"] == fair_compare
    assert "common_period" in report["comparison_basis"]


def test_common_period_comparison_normalizes_shared_dates_only():
    values = [100.0] * 59 + [101.0, 110.0, 112.0, 80.0, 120.0, 121.0, 130.0]
    samsung_rows = build_samsung_single_leverage_shadow_ledger(_prices(values))
    semi_rows = [
        SimpleNamespace(
            date=row.date,
            c60_equity_curve=1.0 + idx * 0.01,
            buyhold_equity_curve=1.0 + idx * 0.02,
            c60_position_state="HOLD",
        )
        for idx, row in enumerate(samsung_rows[1:])
    ]

    comparison = build_common_period_comparison(samsung_rows, semi_rows, single_label="sk_hynix")

    assert comparison["status"] == "FAIR_COMMON_PERIOD"
    assert comparison["trading_days"] < len(samsung_rows)
    assert "sk_hynix_c60" in comparison["metrics"]
    assert "etf_488080_c60" in comparison["metrics"]
    assert not comparison["winner_by_mdd_defense_leveraged_only"].endswith("_underlying_buyhold")


def test_samsung_shadow_modules_do_not_reference_order_paths():
    root = Path(__file__).resolve().parent.parent
    source = (root / "src" / "etf" / "samsung_single_leverage_shadow.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "samsung_single_leverage_shadow.py").read_text(encoding="utf-8")
    sk_script = (root / "scripts" / "sk_hynix_single_leverage_shadow.py").read_text(encoding="utf-8")
    combined = f"{source}\n{script}\n{sk_script}".lower()

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
