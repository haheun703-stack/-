"""2층 포트 골격 빌더 테스트 — 순수 함수만(I/O 없음).

설계 상수 / 드로다운 레벨 분류 / 데이터계약 §2·§3 행 변환을 박제.
골격(관측) 단계 — 매매로직 무관·실데이터는 unfreeze 후.
"""
from __future__ import annotations

from src.use_cases.two_layer_portfolio import (
    CORE_PCT,
    DD_ALERT_THRESHOLD,
    SATELLITE_DETAIL,
    SATELLITE_PCT,
    build_drawdown_alert_row,
    build_two_layer_row,
    classify_dd_level,
)

DATE = "2026-06-16"
SNAP = "2026-06-16T16:40:00+09:00"


# ── 설계 상수 ──────────────────────────────────────────────

def test_design_constants():
    assert CORE_PCT == 82.0
    assert SATELLITE_PCT == 18.0
    assert CORE_PCT + SATELLITE_PCT == 100.0
    assert DD_ALERT_THRESHOLD == -15.0


def test_satellite_detail_is_us_leverage_trio():
    tickers = {s["ticker"] for s in SATELLITE_DETAIL}
    assert tickers == {"SOXL", "TQQQ", "NVDL"}


# ── classify_dd_level ─────────────────────────────────────

def test_level_normal_when_above_threshold():
    assert classify_dd_level(-10.0) == "normal"
    assert classify_dd_level(0.0) == "normal"


def test_level_alert_at_and_below_threshold():
    assert classify_dd_level(-15.0) == "alert"   # 경계 포함
    assert classify_dd_level(-22.0) == "alert"


def test_level_none_is_normal():
    # 실데이터 전(None)은 평소로 간주
    assert classify_dd_level(None) == "normal"


# ── build_two_layer_row (데이터계약 §2) ───────────────────

TWO_LAYER_COLS = {
    "date", "core_pct", "satellite_pct", "cum_return", "mdd", "current_dd",
    "satellite_detail", "snapshot_time",
}


def test_two_layer_row_columns_and_skeleton():
    row = build_two_layer_row(DATE, SNAP)
    assert set(row.keys()) == TWO_LAYER_COLS
    assert row["core_pct"] == 82.0 and row["satellite_pct"] == 18.0
    # 실데이터는 골격 단계에서 None(unfreeze 후 채움)
    assert row["cum_return"] is None and row["mdd"] is None and row["current_dd"] is None
    assert len(row["satellite_detail"]) == 3


def test_two_layer_row_accepts_real_data():
    row = build_two_layer_row(DATE, SNAP, cum_return=12.3, mdd=-8.9, current_dd=-3.0)
    assert row["cum_return"] == 12.3
    assert row["current_dd"] == -3.0


# ── build_drawdown_alert_row (데이터계약 §3) ──────────────

ALERT_COLS = {
    "date", "current_dd", "level", "verdict", "history_analog", "crisis_signals",
    "foreign_outflow", "port_exposure", "recommended_actions", "snapshot_time",
}


def test_alert_row_columns_and_normal_default():
    row = build_drawdown_alert_row(DATE, SNAP)
    assert set(row.keys()) == ALERT_COLS
    assert row["level"] == "normal"
    assert row["verdict"] is None
    # 평소엔 JSONB None 허용
    assert row["history_analog"] is None and row["crisis_signals"] is None


def test_alert_row_goes_alert_on_deep_dd():
    row = build_drawdown_alert_row(DATE, SNAP, current_dd=-18.0)
    assert row["level"] == "alert"
    assert row["verdict"] == "판정대기"


def test_alert_row_normal_above_threshold():
    row = build_drawdown_alert_row(DATE, SNAP, current_dd=-12.0)
    assert row["level"] == "normal"
    assert row["verdict"] is None
