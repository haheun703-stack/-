"""v8.0 Trigger Engine 단위 테스트"""
import pandas as pd
import pytest

from src.v8_triggers import TriggerEngine


@pytest.fixture
def config():
    return {
        'v8_hybrid': {
            'triggers': {
                'volume_rsi': {
                    'vol_multiplier': 1.5,
                    'rsi_threshold': 45,
                }
            }
        }
    }


@pytest.fixture
def engine(config):
    return TriggerEngine(config)


def make_row(**kwargs):
    defaults = {
        'trix': 0.05,
        'trix_signal': 0.03,
        'trix_prev': 0.02,
        'trix_signal_prev': 0.04,
        'volume': 1500000,
        'volume_ma20': 800000,
        'rsi_14': 48,
        'rsi_prev': 43,
        'ema_curvature': 0.003,
        'obv_trend_5d': 0.02,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestT1TrixGolden:
    def test_golden_cross(self, engine):
        """TRIX > Signal 상향교차 → 발동"""
        row = make_row(trix=0.05, trix_signal=0.03,
                       trix_prev=0.02, trix_signal_prev=0.04)
        result = engine.trigger_trix_golden(row)
        assert result.fired == True
        assert result.trigger_name == "T1_TRIX_Golden"

    def test_no_cross(self, engine):
        """TRIX가 계속 Signal 위에 있으면 미발동 (교차 아님)"""
        row = make_row(trix=0.05, trix_signal=0.03,
                       trix_prev=0.04, trix_signal_prev=0.02)
        result = engine.trigger_trix_golden(row)
        assert result.fired == False


class TestT2VolumeRSI:
    def test_volume_rsi_breakout(self, engine):
        """거래량 폭증 + RSI 45 상향돌파 → 발동"""
        row = make_row(volume=1500000, volume_ma20=800000,
                       rsi_14=48, rsi_prev=43)
        result = engine.trigger_volume_rsi(row)
        assert result.fired == True
        assert result.trigger_name == "T2_Volume_RSI"

    def test_volume_only_no_rsi(self, engine):
        """v8.1 완화 정책: 거래량 충족 → 발동 (OR 조건)."""
        row = make_row(volume=1500000, volume_ma20=800000,
                       rsi_14=48, rsi_prev=46)
        result = engine.trigger_volume_rsi(row)
        # v8.1: vol_surge (vol > 20MA * 0.3) OR rsi_in_range (30~80) → fired
        assert result.fired == True

    def test_rsi_only_no_volume(self, engine):
        """v8.1 완화 정책: RSI 30~80 in range → 발동 (OR 조건)."""
        row = make_row(volume=800000, volume_ma20=800000,
                       rsi_14=48, rsi_prev=43)
        result = engine.trigger_volume_rsi(row)
        # v8.1: rsi_in_range 48 ∈ [30,80] → fired
        assert result.fired == True

    def test_below_threshold_no_fire(self, engine):
        """v8.1: vol surge 미달 + RSI 범위 밖 → 미발동."""
        # vol 100 < 800 * 0.3 = 240 (vol_surge False)
        # RSI 25 < 30 (rsi_in_range False)
        row = make_row(volume=100000, volume_ma20=800000,
                       rsi_14=25, rsi_prev=24)
        result = engine.trigger_volume_rsi(row)
        assert result.fired == False


class TestT3CurvatureOBV:
    def test_curvature_obv_positive(self, engine):
        """곡률 양전환 + OBV 상승 → 발동"""
        row = make_row(ema_curvature=0.003, obv_trend_5d=0.02)
        result = engine.trigger_curvature_obv(row)
        assert result.fired == True
        assert result.trigger_name == "T3_Curvature_OBV"

    def test_negative_curvature(self, engine):
        """곡률 음수 → 미발동"""
        row = make_row(ema_curvature=-0.002, obv_trend_5d=0.02)
        result = engine.trigger_curvature_obv(row)
        assert result.fired == False

    def test_negative_obv(self, engine):
        """OBV 하락 → 미발동"""
        row = make_row(ema_curvature=0.003, obv_trend_5d=-0.01)
        result = engine.trigger_curvature_obv(row)
        assert result.fired == False


class TestCheckAll:
    def test_or_condition(self, engine):
        """하나 이상 트리거 발동 시 리스트 반환"""
        row = make_row()
        results = engine.check_all(row)
        assert len(results) >= 1

    def test_no_triggers(self, engine):
        """v8.1: 모든 트리거 미발동 — 더 보수적인 값 (vol << 0.3x MA + RSI < 30)."""
        row = make_row(
            trix=0.02, trix_signal=0.03,
            trix_prev=0.02, trix_signal_prev=0.03,
            volume=100000, volume_ma20=800000,  # vol 100 < 800*0.3 = 240
            rsi_14=25, rsi_prev=24,  # RSI 25 < 30 (out of range)
            ema_curvature=-0.001, obv_trend_5d=-0.01,
        )
        results = engine.check_all(row)
        assert len(results) == 0
