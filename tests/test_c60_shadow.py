from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etf.c60_shadow import build_c60_report, build_c60_shadow_ledger


def _prices(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"close": values}, index=idx)


def test_c60_shadow_records_exit_cash_and_reenter_next_observation_state():
    values = [100.0] * 60
    values += [110.0, 112.0, 80.0, 82.0, 83.0, 120.0, 121.0]

    rows = build_c60_shadow_ledger(_prices(values))

    signals = [row.signal for row in rows]
    assert "EXIT" in signals
    assert "CASH" in signals
    assert "REENTER" in signals

    reenter_idx = signals.index("REENTER")
    assert rows[reenter_idx].c60_position_state == "CASH"
    assert rows[reenter_idx + 1].c60_position_state == "HOLD"
    assert rows[-1].days_in_cash > 0
    assert rows[-1].whipsaw_count >= 1


def test_c60_report_is_shadow_only_with_zero_orders():
    values = [100.0] * 60 + [110.0, 112.0, 80.0, 120.0, 121.0]
    rows = build_c60_shadow_ledger(_prices(values))
    report = build_c60_report(rows)

    assert report["status"] == "SHADOW_ONLY"
    assert report["order_count"] == 0
    assert report["live_trading_state"] == "HOLD"
    assert "실주문 0건" in report["safety_note"]


def test_c60_shadow_modules_do_not_reference_order_paths():
    root = Path(__file__).resolve().parent.parent
    source = (root / "src" / "etf" / "c60_shadow.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "c60_shadow_forward.py").read_text(encoding="utf-8")
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

