"""v8.0 Gate Engine 단위 테스트"""
import pytest
import pandas as pd
from src.v8_gates import GateEngine, GateResult


@pytest.fixture
def config():
    return {
        'v8_hybrid': {
            'gates': {
                'trend': {'ma_fast': 60, 'ma_slow': 120, 'adx_min': 18},
                'pullback': {'min_atr_pullback': 0.8, 'max_atr_pullback': 4.0, 'high_lookback_days': 20},
                'overheat': {'max_52w_ratio': 0.92},
            }
        }
    }


@pytest.fixture
def engine(config):
    return GateEngine(config)


def make_row(**kwargs):
    defaults = {
        'close': 50000,
        'sma_60': 52000,
        'sma_120': 48000,
        'adx_14': 25,
        'high_20': 55000,
        'atr_14': 2000,
        'high_252': 65000,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestG1Trend:
    def test_uptrend_passes(self, engine):
        """MA60 > MA120, ADX > 18, Close > MA120 → 통과"""
        row = make_row(sma_60=52000, sma_120=48000, adx_14=25, close=50000)
        result = engine.gate_trend(row)
        assert result.passed == True
        assert result.gate_name == "G1_Trend"

    def test_downtrend_fails(self, engine):
        """MA60 < MA120 → 역배열 → 실패"""
        row = make_row(sma_60=46000, sma_120=48000)
        result = engine.gate_trend(row)
        assert result.passed == False

    def test_weak_adx_fails(self, engine):
        """ADX < 18 → 추세 부재 → 실패"""
        row = make_row(adx_14=12)
        result = engine.gate_trend(row)
        assert result.passed == False

    def test_below_slow_ma_fails(self, engine):
        """Close < MA120 → 실패"""
        row = make_row(close=45000, sma_120=48000)
        result = engine.gate_trend(row)
        assert result.passed == False


class TestG2Pullback:
    def test_healthy_pullback_passes(self, engine):
        """0.8 ATR ≤ pullback ≤ 4.0 ATR → 통과"""
        # pullback = (55000 - 50000) / 2000 = 2.5 ATR
        row = make_row(high_20=55000, close=50000, atr_14=2000)
        result = engine.gate_pullback(row)
        assert result.passed == True

    def test_shallow_pullback_fails(self, engine):
        """pullback < 0.8 ATR → 조정 부족"""
        # pullback = (51000 - 50000) / 2000 = 0.5 ATR
        row = make_row(high_20=51000, close=50000, atr_14=2000)
        result = engine.gate_pullback(row)
        assert result.passed == False

    def test_deep_pullback_fails(self, engine):
        """pullback > 4.0 ATR → 과도한 하락"""
        # pullback = (60000 - 50000) / 2000 = 5.0 ATR
        row = make_row(high_20=60000, close=50000, atr_14=2000)
        result = engine.gate_pullback(row)
        assert result.passed == False


class TestG3Overheat:
    def test_cooled_stock_passes(self, engine):
        """52주 고점 대비 < 92% → 통과"""
        # ratio = 50000 / 65000 = 76.9%
        row = make_row(close=50000, high_252=65000)
        result = engine.gate_overheat(row)
        assert result.passed == True

    def test_near_high_fails(self, engine):
        """52주 고점 대비 ≥ 92% → 실패"""
        # ratio = 60000 / 65000 = 92.3%
        row = make_row(close=60000, high_252=65000)
        result = engine.gate_overheat(row)
        assert result.passed == False


class TestRunAllGates:
    def test_all_pass(self, engine):
        """모든 게이트 통과"""
        row = make_row()
        passed, results = engine.run_all_gates(row)
        assert passed == True
        assert len(results) == 3

    def test_early_exit_on_trend_fail(self, engine):
        """G1 실패 시 G2, G3는 평가하지 않음"""
        row = make_row(sma_60=46000, sma_120=48000)
        passed, results = engine.run_all_gates(row)
        assert passed == False
        assert len(results) == 1
        assert results[0].gate_name == "G1_Trend"

    def test_early_exit_on_pullback_fail(self, engine):
        """G1 통과, G2 실패 시 G3는 평가하지 않음"""
        row = make_row(high_20=51000, close=50000, atr_14=2000)
        passed, results = engine.run_all_gates(row)
        assert passed == False
        assert len(results) == 2
