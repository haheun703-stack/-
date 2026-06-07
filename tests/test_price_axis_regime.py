from __future__ import annotations

import inspect

import pandas as pd

from src.use_cases import price_axis_regime as par


def _ohlcv(rows: list[tuple]) -> pd.DataFrame:
    """rows: [(date, open, high, low, close), ...] → 정규화된 OHLCV."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1000 for _ in rows],
        },
        index=idx,
    )


# ── price_axis ──────────────────────────────────────────────
def test_price_axis_above_all_anchors() -> None:
    # 2026-06: H1. 연초/반기 시가 = 1/2 첫 거래일 시가 70000, 현재 종가 81500(위)
    df = _ohlcv([
        ("2026-01-02", 70000, 71000, 69000, 70500),
        ("2026-06-01", 80000, 80500, 79000, 80200),  # 6월 첫날(월/주 시가 80000)
        ("2026-06-05", 81000, 82000, 80500, 81500),
    ])
    ax = par.compute_price_axis(df)
    assert ax["data_available"] is True
    assert ax["year_open"] == 70000
    assert ax["half_year_open"] == 70000
    assert ax["half_year_open_state"] == "ABOVE"
    assert ax["monthly_open"] == 80000
    assert ax["monthly_open_state"] == "ABOVE"
    assert ax["monthly_open_broken"] is False
    assert "HALF_YEAR_OPEN_ABOVE" in ax["labels"]
    assert "YEAR_OPEN_ABOVE" in ax["labels"]
    # 반기 시가 70000 대비 81500 → +16.43%
    assert ax["distance_from_half_year_open_pct"] == round((81500 - 70000) / 70000 * 100, 2)


def test_price_axis_weekly_monthly_broken_below() -> None:
    # 같은 주: 6/1(월) 시가 80000 → 6/5 종가 78000(주봉/월봉 시가 아래)
    df = _ohlcv([
        ("2026-06-01", 80000, 80500, 79000, 80000),
        ("2026-06-05", 79500, 79800, 77500, 78000),
    ])
    ax = par.compute_price_axis(df)
    assert ax["weekly_open_state"] == "BELOW"
    assert ax["weekly_open_broken"] is True
    assert ax["monthly_open_broken"] is True
    assert "WEEKLY_OPEN_BELOW" in ax["labels"]


def test_price_axis_empty_or_no_open() -> None:
    assert par.compute_price_axis(pd.DataFrame())["data_available"] is False


# ── candle_turn ─────────────────────────────────────────────
def test_candle_turn_eum_yang_reversal() -> None:
    df = _ohlcv([
        ("2026-06-04", 10000, 10050, 9500, 9600),   # 전일 음봉
        ("2026-06-05", 9700, 10400, 9650, 10300),   # 금일 양봉
    ])
    t = par.compute_candle_turn(df)
    assert t["candle_turn_type"] == "EUM_YANG"
    assert t["label"] == "EUM_YANG_REVERSAL"


def test_candle_turn_yang_eum_warning() -> None:
    df = _ohlcv([
        ("2026-06-04", 10000, 10500, 9950, 10400),  # 전일 양봉
        ("2026-06-05", 10350, 10380, 9800, 9900),   # 금일 음봉
    ])
    t = par.compute_candle_turn(df)
    assert t["candle_turn_type"] == "YANG_EUM"
    assert t["label"] == "YANG_EUM_WARNING"


def test_candle_turn_no_turn() -> None:
    df = _ohlcv([
        ("2026-06-04", 10000, 10500, 9950, 10400),  # 양봉
        ("2026-06-05", 10450, 10900, 10400, 10800),  # 양봉 → 전환 아님
    ])
    assert par.compute_candle_turn(df)["candle_turn_type"] == "NO_TURN"


# ── annual_overheat ─────────────────────────────────────────
def test_annual_overheat_grade_500() -> None:
    df = _ohlcv([
        ("2025-06-05", 10000, 10100, 9900, 10000),
        ("2026-06-05", 62000, 63000, 61000, 62000),  # +520%
    ])
    o = par.compute_annual_overheat(df)
    assert o["return_1y_pct"] == 520.0
    assert o["annual_overheat_warning"] is True
    assert o["overheat_grade"] == "OVERHEAT_500"


def test_annual_overheat_not_overheated() -> None:
    df = _ohlcv([
        ("2025-06-05", 10000, 10100, 9900, 10000),
        ("2026-06-05", 11000, 11100, 10900, 11000),  # +10%
    ])
    o = par.compute_annual_overheat(df)
    assert o["annual_overheat_warning"] is False
    assert o["overheat_grade"] is None


# ── ipo_reversion ───────────────────────────────────────────
def test_ipo_reversion_core_deep_drawdown() -> None:
    r = par.build_ipo_reversion("2025-03-12", 50000, 24000, quality_tag="GROUP_AFFILIATE")
    assert r["ipo_reversion_state"] == "IPO_REVERSION_CORE"  # -52%
    assert r["drawdown_from_listing_open_pct"] == -52.0


def test_ipo_reversion_watch_and_avoid() -> None:
    watch = par.build_ipo_reversion("2025-03-12", 50000, 32000)  # -36%
    assert watch["ipo_reversion_state"] == "IPO_REVERSION_WATCH"
    avoid = par.build_ipo_reversion("2025-03-12", 50000, 24000, managed_risk=True)
    assert avoid["ipo_reversion_state"] == "IPO_REVERSION_AVOID"


def test_ipo_reversion_no_meta_returns_empty() -> None:
    assert par.build_ipo_reversion(None, None, 10000)["data_available"] is False


# ── bundle ──────────────────────────────────────────────────
def test_build_bundle_full() -> None:
    df = _ohlcv([
        ("2025-06-05", 10000, 10100, 9900, 10000),
        ("2026-06-01", 80000, 80500, 79000, 80000),
        ("2026-06-04", 80000, 80500, 79000, 79000),
        ("2026-06-05", 79200, 82000, 79000, 81500),
    ])
    b = par.build_price_axis_labels(df, ipo_meta={"listing_date": "2025-01-02", "listing_open": 120000})
    assert b["data_available"] is True
    assert b["price_axis"]["data_available"] is True
    assert b["candle_turn"]["candle_turn_type"] == "EUM_YANG"  # 6/4 음봉 → 6/5 양봉
    assert b["annual_overheat"]["overheat_grade"] == "OVERHEAT_500"  # 10000→81500 +715%? grade 500/1000
    assert b["ipo_reversion"]["data_available"] is True


def test_build_bundle_empty_safe() -> None:
    b = par.build_price_axis_labels(pd.DataFrame())
    assert b["data_available"] is False
    assert b["price_axis"]["data_available"] is False


# ── 안전선: 주문/매도/스케줄러 심볼 0 ─────────────────────────
def test_source_has_no_order_or_sell_symbols() -> None:
    src = inspect.getsource(par)
    forbidden = (
        "smart_sell", "SmartSellExecutor", "sell_brain", "owner_rule",
        "sell_market", "sell_limit", "order_intents_gate", "KisOrderAdapter",
        "place_order", "send_order", "create_market", "create_limit",
        "PAPER_OPEN", "scheduler", "run_adaptive_cycle",
    )
    for f in forbidden:
        assert f not in src, f"금지 심볼 발견: {f}"
