"""v8.1 Holding Horizon 분류기 테스트"""
import pandas as pd
import pytest

from src.holding_horizon import HoldingHorizonClassifier


@pytest.fixture
def classifier():
    return HoldingHorizonClassifier()


def make_row(**kwargs):
    defaults = {
        'close': 50000,
        'sma_60': 48000,
        'sma_120': 45000,
        'half_life': 10,
        'adx_14': 30,
        'ema_curvature': 0.008,
        'ema_curvature_prev': -0.002,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestShortHorizon:
    def test_fast_reversion_strong_trend(self, classifier):
        """빠른 회귀 + 강한 추세 → 단기"""
        row = make_row(half_life=8, adx_14=30, ema_curvature=0.008)
        result = classifier.classify(row)
        assert result.horizon == "SHORT"
        assert result.horizon_days == 10

    def test_label(self, classifier):
        row = make_row(half_life=8, adx_14=30, ema_curvature=0.008)
        result = classifier.classify(row)
        assert "단기" in classifier.horizon_label(result.horizon)


class TestMediumHorizon:
    def test_moderate_factors(self, classifier):
        """보통 회귀 + 보통 추세 → 중기"""
        row = make_row(half_life=20, adx_14=22, ema_curvature=0.002)
        result = classifier.classify(row)
        assert result.horizon == "MEDIUM"
        assert result.horizon_days == 21


class TestLongHorizon:
    def test_slow_reversion_weak_trend(self, classifier):
        """느린 회귀 + 약한 추세 + 미약한 곡률 → 장기"""
        row = make_row(half_life=50, adx_14=12, ema_curvature=0.0001)
        result = classifier.classify(row)
        assert result.horizon == "LONG"
        assert result.horizon_days == 45

    def test_full_alignment_boosts_long(self, classifier):
        """완전 정배열 → 장기 보강"""
        row = make_row(
            half_life=35, adx_14=15, ema_curvature=0.001,
            close=50000, sma_60=48000, sma_120=45000,
        )
        result = classifier.classify(row)
        assert result.horizon == "LONG"


class TestConfidence:
    def test_confidence_range(self, classifier):
        """신뢰도는 0~1 사이"""
        row = make_row()
        result = classifier.classify(row)
        assert 0.0 <= result.confidence <= 1.0

    def test_has_factors(self, classifier):
        """판단 근거가 항상 포함됨"""
        row = make_row()
        result = classifier.classify(row)
        assert 'half_life' in result.factors
        assert 'adx' in result.factors
        assert 'curvature' in result.factors
        assert 'ma_alignment' in result.factors
