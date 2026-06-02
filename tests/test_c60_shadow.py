from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etf.c60_shadow import (
    C60LedgerRow,
    build_accelerated_c60_validation,
    build_c60_report,
    build_c60_shadow_ledger,
)


def _prices(values: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="B")
    data = {"close": values}
    if volumes is not None:
        data["volume"] = volumes
    return pd.DataFrame(data, index=idx)


def test_c60_shadow_records_exit_cash_and_reenter_next_observation_state():
    values = [100.0] * 59 + [101.0]
    values += [110.0, 112.0, 80.0, 82.0, 83.0, 120.0, 121.0, 125.0, 130.0, 136.0, 140.0]
    volumes = [1000.0] * 60
    volumes += [1000.0, 1000.0, 1000.0, 900.0, 800.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0]

    rows = build_c60_shadow_ledger(_prices(values, volumes))

    signals = [row.signal for row in rows]
    assert "EXIT" in signals
    assert "CASH" in signals
    assert "REENTER" in signals

    reenter_idx = signals.index("REENTER")
    assert rows[reenter_idx].c60_position_state == "CASH"
    assert rows[reenter_idx + 1].c60_position_state == "HOLD"
    assert rows[-1].days_in_cash > 0
    assert rows[-1].whipsaw_count >= 1


def test_c60_shadow_does_not_reenter_on_ma60_cross_only():
    values = [100.0] * 59 + [101.0]
    values += [110.0, 112.0, 80.0, 105.0, 106.0, 107.0]
    volumes = [1000.0] * 60
    volumes += [1000.0, 1000.0, 1000.0, 800.0, 700.0, 600.0]

    rows = build_c60_shadow_ledger(_prices(values, volumes))
    signals = [row.signal for row in rows]

    assert "EXIT" in signals
    assert "REENTER" not in signals
    assert rows[-1].c60_position_state == "CASH"


def test_c60_report_is_shadow_only_with_zero_orders():
    values = [100.0] * 59 + [101.0, 110.0, 112.0, 80.0, 120.0, 121.0]
    volumes = [1000.0] * 60 + [1000.0, 1000.0, 900.0, 1500.0, 1600.0]
    rows = build_c60_shadow_ledger(_prices(values, volumes))
    report = build_c60_report(rows)

    assert report["status"] == "SHADOW_ONLY"
    assert report["order_count"] == 0
    assert report["live_trading_state"] == "HOLD"
    assert "실주문 0건" in report["safety_note"]
    assert report["missed_upside_amount"] > 0


def test_accelerated_c60_validation_is_shadow_only():
    values = [100.0] * 59 + [101.0]
    values += [110.0, 112.0, 80.0, 82.0, 83.0, 120.0, 121.0, 125.0, 130.0, 136.0, 140.0]
    volumes = [1000.0] * 60
    volumes += [1000.0, 1000.0, 1000.0, 900.0, 800.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0]
    rows = build_c60_shadow_ledger(_prices(values, volumes))

    report = build_accelerated_c60_validation(rows, windows=(3, 5))

    assert report["status"] == "ACCELERATED_SHADOW_REPLAY"
    assert report["order_count"] == 0
    assert report["live_trading_state"] == "HOLD"
    assert set(report["windows"]) == {"3", "5"}
    assert report["windows"]["3"]["segment_count"] > 0


def test_accelerated_c60_mdd_edge_positive_when_c60_defends_drawdown():
    rows = [
        C60LedgerRow(
            date="2026-01-01",
            ticker="488080",
            close=100.0,
            ma60=90.0,
            signal="HOLD",
            c60_position_state="HOLD",
            c60_equity_curve=1.0,
            buyhold_equity_curve=1.0,
            drawdown_c60=0.0,
            drawdown_buyhold=0.0,
            delta_vs_buyhold=0.0,
            days_in_cash=0,
            whipsaw_count=0,
            missed_upside_after_exit=0.0,
            avoided_drawdown_after_exit=0.0,
        ),
        C60LedgerRow(
            date="2026-01-02",
            ticker="488080",
            close=80.0,
            ma60=90.0,
            signal="EXIT",
            c60_position_state="HOLD",
            c60_equity_curve=0.95,
            buyhold_equity_curve=0.7,
            drawdown_c60=-0.05,
            drawdown_buyhold=-0.3,
            delta_vs_buyhold=0.25,
            days_in_cash=0,
            whipsaw_count=0,
            missed_upside_after_exit=0.0,
            avoided_drawdown_after_exit=0.0,
        ),
        C60LedgerRow(
            date="2026-01-03",
            ticker="488080",
            close=90.0,
            ma60=90.0,
            signal="CASH",
            c60_position_state="CASH",
            c60_equity_curve=0.95,
            buyhold_equity_curve=0.8,
            drawdown_c60=-0.05,
            drawdown_buyhold=-0.2,
            delta_vs_buyhold=0.15,
            days_in_cash=1,
            whipsaw_count=0,
            missed_upside_after_exit=0.0,
            avoided_drawdown_after_exit=0.0,
        ),
    ]

    report = build_accelerated_c60_validation(rows, windows=(3,))

    assert report["windows"]["3"]["latest_segment"]["mdd_edge"] > 0
    assert report["windows"]["3"]["c60_mdd_better_rate"] == 1.0


def test_c60_shadow_modules_do_not_reference_order_paths():
    root = Path(__file__).resolve().parent.parent
    source = (root / "src" / "etf" / "c60_shadow.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "c60_shadow_forward.py").read_text(encoding="utf-8")
    replay_script = (root / "scripts" / "c60_shadow_accelerated_replay.py").read_text(encoding="utf-8")
    combined = f"{source}\n{script}\n{replay_script}".lower()

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
