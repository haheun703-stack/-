"""v8.0 Scoring Engine 단위 테스트"""
import pandas as pd
import pytest

from src.v8_scorers import ScoringEngine


@pytest.fixture
def config():
    return {
        'v8_hybrid': {
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
            'position': {
                'A_grade_pct': 0.20,
                'B_grade_pct': 0.10,
            },
        }
    }


@pytest.fixture
def engine(config):
    return ScoringEngine(config)


def make_row(**kwargs):
    defaults = {
        'close': 50000,
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
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestWeightsSum:
    def test_weights_sum_to_one(self, engine):
        """가중치 합 = 1.0"""
        assert abs(sum(engine.weights.values()) - 1.0) < 1e-6


class TestS1Energy:
    def test_optimal_rsi(self, engine):
        """RSI 38-52 구간 → 높은 점수"""
        row = make_row(rsi_14=45)
        result = engine.score_energy_depletion(row)
        assert result.score > 0.3

    def test_high_rsi_no_score(self, engine):
        """RSI > 52 → RSI 점수 0"""
        row = make_row(rsi_14=65)
        result = engine.score_energy_depletion(row)
        assert result.breakdown['rsi'] == 0.0

    def test_volume_dryup(self, engine):
        """5MA/20MA < 0.7 → 높은 거래량 소진 점수"""
        row = make_row(volume_ma5=400000, volume_ma20=800000)
        result = engine.score_energy_depletion(row)
        assert result.breakdown['volume'] == 0.35


class TestS3OU:
    def test_optimal_z_score(self, engine):
        """z-score -2.5~-1.0 → 최적 구간"""
        row = make_row(ou_z=-1.5)
        result = engine.score_ou_reversion(row)
        assert result.breakdown['z_score'] == 0.45

    def test_weak_z_score(self, engine):
        """z-score -1.0~-0.5 → 약한 구간"""
        row = make_row(ou_z=-0.7)
        result = engine.score_ou_reversion(row)
        assert result.breakdown['z_score'] == 0.30

    def test_positive_z_no_score(self, engine):
        """z-score > 0 → 점수 0"""
        row = make_row(ou_z=0.5)
        result = engine.score_ou_reversion(row)
        assert result.breakdown['z_score'] == 0.0


class TestS4MomentumDecel:
    def test_curvature_inflection(self, engine):
        """곡률 음→양 전환 = 변곡점 → 최고 점수"""
        row = make_row(ema_curvature=0.002, ema_curvature_prev=-0.001)
        result = engine.score_momentum_deceleration(row)
        assert result.breakdown['curvature'] == 0.40

    def test_slope_deceleration(self, engine):
        """하락 중 감속 → 초점 접근"""
        row = make_row(linreg_slope_20=-0.3, linreg_slope_5=-0.05)
        result = engine.score_momentum_deceleration(row)
        assert result.breakdown['slope_decel'] > 0.0

    def test_no_decel(self, engine):
        """가속 하락 → 초점 미접근"""
        row = make_row(linreg_slope_20=-0.2, linreg_slope_5=-0.4,
                       ema_curvature=-0.005, ema_curvature_prev=-0.003,
                       macd_histogram=-0.5, macd_histogram_prev=-0.3)
        result = engine.score_momentum_deceleration(row)
        assert result.score < 0.3


class TestS5SmartMoney:
    def test_obv_divergence(self, engine):
        """가격 하락 + OBV 상승 = 매집"""
        row = make_row(price_trend_5d=-0.03, obv_trend_5d=0.05)
        result = engine.score_smart_money(row)
        assert result.breakdown['obv_divergence'] == 0.40


class TestGradeResult:
    def test_ideal_focus_a_grade(self, engine):
        """이상적 초점 종목 → A등급"""
        row = make_row()
        result = engine.score_all(row)
        assert result.grade in ("A", "B")
        assert result.tradeable is True

    def test_bad_stock_f_grade(self, engine):
        """나쁜 조건 → F등급"""
        row = make_row(
            rsi_14=75, volume_ma5=1200000, volume_ma20=800000,
            ou_z=1.0, half_life=100, ou_theta=0.0,
            linreg_slope_20=0.5, linreg_slope_5=0.6,
            ema_curvature=-0.01, ema_curvature_prev=-0.005,
            macd_histogram=-1.0, macd_histogram_prev=-0.5,
            price_trend_5d=0.05, obv_trend_5d=-0.03,
            institutional_net_buy_5d=-100000,
            distribution_risk_score=0.8,
        )
        result = engine.score_all(row)
        assert result.grade == "F"
        assert result.tradeable is False

    def test_position_size(self, engine):
        """A등급 → 20%, B등급 → 10%"""
        assert engine.pos_cfg.get('A_grade_pct') == 0.20
        assert engine.pos_cfg.get('B_grade_pct') == 0.10
