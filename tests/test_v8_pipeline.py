"""v8.0 Pipeline 통합 테스트"""
import pandas as pd
import pytest

from src.v8_pipeline import QuantumPipelineV8


@pytest.fixture
def config():
    return {
        'v8_hybrid': {
            'enabled': True,
            'gates': {
                'trend': {'ma_fast': 60, 'ma_slow': 120, 'adx_min': 18},
                'pullback': {'min_atr_pullback': 0.8, 'max_atr_pullback': 4.0, 'high_lookback_days': 20},
                'overheat': {'max_52w_ratio': 0.92},
            },
            'scoring': {
                'weights': {
                    'energy_depletion': 0.30,
                    'valuation': 0.20,
                    'ou_reversion': 0.20,
                    'momentum_decel': 0.15,
                    'smart_money': 0.15,
                },
                'energy': {
                    'rsi_optimal_center': 45,
                    'rsi_optimal_range': [38, 52],
                    'vol_ratio_threshold': 0.7,
                    'bb_position_optimal': [0.05, 0.30],
                },
                'valuation': {
                    'per_discount_threshold': 0.8,
                    'pbr_percentile_good': 30,
                    'forward_per_discount': 0.85,
                },
                'ou': {
                    'z_score_optimal': [-2.5, -1.0],
                    'z_score_weak': [-1.0, -0.5],
                    'half_life_fast': [3, 15],
                    'half_life_medium': [15, 30],
                    'theta_strong': 0.05,
                },
                'smart_money': {
                    'drs_safe_threshold': 0.3,
                    'drs_neutral_threshold': 0.5,
                },
                'grade_cutoffs': {'A': 0.80, 'B': 0.65, 'C': 0.50},
            },
            'triggers': {
                'volume_rsi': {'vol_multiplier': 1.5, 'rsi_threshold': 45},
            },
            'position': {
                'A_grade_pct': 0.20,
                'B_grade_pct': 0.10,
                'stop_loss_atr': 2.0,
                'target_atr_A': 5.0,
                'target_atr_B': 3.5,
                'min_rr_ratio': 1.5,
            },
        }
    }


@pytest.fixture
def pipeline(config):
    return QuantumPipelineV8(config)


def make_ideal_row():
    """이상적 초점 시나리오"""
    return pd.Series({
        'close': 50000,
        'sma_60': 52000,
        'sma_120': 48000,
        'adx_14': 25,
        'high_20': 55000,
        'atr_14': 2000,
        'high_252': 65000,
        'rsi_14': 42,
        'volume_ma5': 500000,
        'volume_ma20': 800000,
        'bb_upper': 55000,
        'bb_lower': 45000,
        'per': 8.5,
        'sector_per': 12.0,
        'pbr_percentile_3y': 25,
        'forward_per': 7.0,
        'ou_z': -1.5,
        'half_life': 10,
        'ou_theta': 0.06,
        'linreg_slope_20': -0.3,
        'linreg_slope_5': -0.05,
        'ema_curvature': 0.002,
        'ema_curvature_prev': -0.001,
        'macd_histogram': -0.1,
        'macd_histogram_prev': -0.3,
        'price_trend_5d': -0.02,
        'obv_trend_5d': 0.03,
        'institutional_net_buy_5d': 50000,
        'distribution_risk_score': 0.2,
        'trix': 0.05,
        'trix_signal': 0.03,
        'trix_prev': 0.02,
        'trix_signal_prev': 0.04,
        'volume': 1500000,
        'rsi_prev': 39,
    })


def make_bad_row():
    """나쁜 종목 시나리오 (게이트에서 걸려야 함)"""
    return pd.Series({
        'close': 59000,
        'sma_60': 47000,
        'sma_120': 50000,  # 역배열
        'adx_14': 12,
        'high_20': 60000,
        'atr_14': 2000,
        'high_252': 60000,
    })


class TestIdealFocus:
    def test_buy_signal(self, pipeline):
        """이상적 초점 종목 → BUY 시그널"""
        result = pipeline.scan_single(make_ideal_row(), ticker='TEST001', date='2024-06-15')
        assert result["signal"] is True
        assert result["v8_action"] == "BUY"
        assert result["grade"] in ("A", "B")

    def test_has_required_keys(self, pipeline):
        """backtest_engine 호환 키 존재"""
        result = pipeline.scan_single(make_ideal_row(), ticker='TEST001', date='2024-06-15')
        required_keys = [
            "ticker", "date", "signal", "zone_score", "grade",
            "trigger_type", "entry_price", "stop_loss", "target_price",
            "risk_reward_ratio", "atr_value", "position_ratio",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_rr_ratio_positive(self, pipeline):
        """손익비 > 0"""
        result = pipeline.scan_single(make_ideal_row(), ticker='TEST001', date='2024-06-15')
        assert result["risk_reward_ratio"] > 0

    def test_stop_loss_below_entry(self, pipeline):
        """손절가 < 진입가"""
        result = pipeline.scan_single(make_ideal_row(), ticker='TEST001', date='2024-06-15')
        assert result["stop_loss"] < result["entry_price"]


class TestBadStock:
    def test_gate_fail(self, pipeline):
        """나쁜 종목 → 게이트 실패 → SKIP"""
        result = pipeline.scan_single(make_bad_row(), ticker='BAD001', date='2024-06-15')
        assert result["signal"] is False
        assert result["v8_action"] == "SKIP"

    def test_no_score_for_gate_fail(self, pipeline):
        """게이트 실패 시 스코어링 하지 않음"""
        result = pipeline.scan_single(make_bad_row(), ticker='BAD001', date='2024-06-15')
        assert result["zone_score"] == 0.0


class TestNoTrigger:
    def test_watch_when_no_trigger(self, pipeline):
        """B등급+이지만 트리거 미발동 → WATCH"""
        row = make_ideal_row()
        # 모든 트리거를 끔
        row['trix'] = 0.02
        row['trix_signal'] = 0.03
        row['trix_prev'] = 0.02
        row['trix_signal_prev'] = 0.03
        row['volume'] = 500000  # 거래량 부족
        row['rsi_prev'] = 46  # RSI 이미 45 위
        row['ema_curvature'] = -0.001  # 곡률 음수
        row['obv_trend_5d'] = -0.01  # OBV 하락

        result = pipeline.scan_single(row, ticker='WATCH001', date='2024-06-15')
        assert result["signal"] is False
        # B등급 이상이지만 트리거 없으면 WATCH
        if result["grade"] in ("A", "B"):
            assert result["v8_action"] == "WATCH"
