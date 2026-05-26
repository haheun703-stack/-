"""ATR 동적 손익절 단위 테스트 — H7 (5/26 작성)."""
from __future__ import annotations

import pytest

from src.use_cases.atr_dynamic_stop import (
    calc_atr_dynamic_stop,
    StopTarget,
    REGIME_MULTIPLIERS,
)


def test_bull_regime_normal():
    """BULL: stop = entry - ATR×2.0, target = entry + ATR×3.0."""
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=2000, regime="BULL")
    assert r.source == "ATR"
    assert r.stop_price == 96000   # 100000 - 2000*2
    assert r.target_price == 106000  # 100000 + 2000*3
    assert r.atr_value == 2000


def test_neutral_regime():
    """NEUTRAL: ATR×1.5/2.5."""
    r = calc_atr_dynamic_stop(entry_price=50000, atr_value=1000, regime="NEUTRAL")
    assert r.source == "ATR"
    assert r.stop_price == 48500   # 50000 - 1000*1.5
    assert r.target_price == 52500  # 50000 + 1000*2.5


def test_bearish_regime():
    """BEARISH: ATR×1.0/2.0 (빠른 익절)."""
    r = calc_atr_dynamic_stop(entry_price=30000, atr_value=600, regime="BEARISH")
    assert r.source == "ATR"
    assert r.stop_price == 29400   # 30000 - 600*1
    assert r.target_price == 31200  # 30000 + 600*2


def test_fallback_no_atr():
    """ATR=None → -5% fallback."""
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=None)
    assert r.source == "FALLBACK"
    assert r.stop_pct == -5.0
    assert r.stop_price == 95000
    assert r.target_pct == 7.0
    assert r.target_price == 107000


def test_fallback_zero_atr():
    """ATR=0 → fallback."""
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=0)
    assert r.source == "FALLBACK"
    assert "0" in r.reason or "미수신" in r.reason


def test_atr_outlier_fallback():
    """ATR > entry × 10% → 이상치 fallback."""
    # entry=100,000 / ATR=15,000 (15%) → fallback
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=15000, regime="BULL")
    assert r.source == "FALLBACK"
    assert r.stop_pct == -5.0  # -5% fallback
    assert "이상치" in r.reason


def test_atr_borderline_cap():
    """ATR = entry × 10% 경계 = 통과 (>는 fallback)."""
    # 100,000 × 10% = 10,000 정확히
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=10000, regime="NEUTRAL")
    # ATR_CAP_PCT = 10.0, > 비교라 10.0은 통과 (정상 ATR 경로)
    assert r.source == "ATR"


def test_invalid_entry():
    """entry=0 → invalid."""
    r = calc_atr_dynamic_stop(entry_price=0, atr_value=2000)
    assert r.source == "FALLBACK"
    assert "invalid" in r.reason


def test_unknown_regime_defaults_neutral():
    """알 수 없는 regime → NEUTRAL 기본."""
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=1000, regime="UNKNOWN")
    # NEUTRAL과 동일: stop 1.5, target 2.5
    assert r.stop_price == 98500
    assert r.target_price == 102500


def test_real_world_diaiti():
    """디아이티 시나리오: entry 25,245 / ATR 800 / NEUTRAL."""
    r = calc_atr_dynamic_stop(entry_price=25245, atr_value=800, regime="NEUTRAL")
    assert r.source == "ATR"
    assert r.stop_price == 24045   # 25245 - 1200
    assert r.target_price == 27245  # 25245 + 2000
    # 손절 -4.75% (-5% 고정보다 약간 좁음 — 저변동주 보호)
    assert -5.5 < r.stop_pct < -4.5


def test_real_world_high_volatility():
    """고변동주: entry 100,000 / ATR 3,500 (3.5%) / BULL."""
    r = calc_atr_dynamic_stop(entry_price=100000, atr_value=3500, regime="BULL")
    assert r.source == "ATR"
    assert r.stop_price == 93000   # 100000 - 7000 (-7%, -5% 고정보다 넓음)
    assert r.target_price == 110500  # 100000 + 10500 (+10.5%)


def test_stoptarget_dataclass():
    """StopTarget dataclass 필드 검증."""
    r = calc_atr_dynamic_stop(entry_price=10000, atr_value=200, regime="BULL")
    assert isinstance(r, StopTarget)
    assert hasattr(r, "stop_price")
    assert hasattr(r, "target_price")
    assert hasattr(r, "source")
    assert hasattr(r, "reason")
    assert hasattr(r, "regime")


def test_regime_multipliers_have_3():
    """REGIME_MULTIPLIERS에 3개 레짐 모두 정의."""
    assert "BULL" in REGIME_MULTIPLIERS
    assert "NEUTRAL" in REGIME_MULTIPLIERS
    assert "BEARISH" in REGIME_MULTIPLIERS
