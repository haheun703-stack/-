"""
v6.0 Martin Momentum Engine 단위 테스트

테스트 항목:
  1. EMA2 계산: fast - slow
  2. Dead Zone: |ema2_norm| < epsilon → 무시
  3. Sigmoid 활성화: 연속적 강도 0~1
  4. 역전 Sigmoid: 하이브리드 비중 제약
  5. 최적 보유기간: 1.7 × (N_fast + N_slow)
  6. 변동성 정규화 비중: clip(target/realized, min, max)
  7. 종합 evaluate: signal_type + confidence
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.martin_momentum import MartinMomentumEngine
from src.entities.momentum_models import MartinMomentumResult


# ── 헬퍼 ──

def make_df(n=100, trend=0.001):
    """테스트용 DataFrame (close, ema_8, ema_24, daily_sigma 포함)"""
    np.random.seed(42)
    close = 10000 + np.cumsum(np.random.randn(n) * 100 + trend * 10000)
    close = close.clip(min=1000)
    df = pd.DataFrame({"close": close})
    df["ema_8"] = df["close"].ewm(span=8).mean()
    df["ema_24"] = df["close"].ewm(span=24).mean()
    df["ret1"] = df["close"].pct_change()
    df["daily_sigma"] = df["ret1"].rolling(20).std()
    return df


# ── EMA2 테스트 ──

class TestEMA2:
    def test_basic_calc(self):
        """EMA2 = ema_8 - ema_24"""
        engine = MartinMomentumEngine()
        df = make_df(100, trend=0.002)
        idx = 99
        ema2 = engine.calc_ema2(df, idx)
        expected = float(df["ema_8"].iloc[idx] - df["ema_24"].iloc[idx])
        assert abs(ema2 - expected) < 0.01

    def test_missing_columns(self):
        """EMA 컬럼 없으면 0.0"""
        engine = MartinMomentumEngine()
        df = pd.DataFrame({"close": [100, 101, 102]})
        assert engine.calc_ema2(df, 2) == 0.0

    def test_normalized(self):
        """EMA2 정규화 = ema2 / close * 100"""
        engine = MartinMomentumEngine()
        assert engine.calc_ema2_normalized(100, 10000) == 1.0
        assert engine.calc_ema2_normalized(0, 10000) == 0.0
        assert engine.calc_ema2_normalized(100, 0) == 0.0


# ── Dead Zone 테스트 ──

class TestDeadZone:
    def test_inside_dead_zone(self):
        """작은 신호 → Dead Zone"""
        engine = MartinMomentumEngine(epsilon=0.6)
        assert engine.check_dead_zone(0.3) is True
        assert engine.check_dead_zone(-0.5) is True

    def test_outside_dead_zone(self):
        """큰 신호 → 통과"""
        engine = MartinMomentumEngine(epsilon=0.6)
        assert engine.check_dead_zone(0.7) is False
        assert engine.check_dead_zone(-1.0) is False

    def test_boundary(self):
        """경계값: epsilon과 동일 → 통과 (< 연산)"""
        engine = MartinMomentumEngine(epsilon=0.6)
        assert engine.check_dead_zone(0.6) is False


# ── Sigmoid 테스트 ──

class TestSigmoid:
    def test_zero_input(self):
        """0 입력 → 0"""
        engine = MartinMomentumEngine(sigmoid_k=5.0)
        result = engine.sigmoid_activation(0.0)
        assert abs(result) < 0.01

    def test_large_positive(self):
        """큰 양수 → ~1.0"""
        engine = MartinMomentumEngine(sigmoid_k=5.0)
        result = engine.sigmoid_activation(3.0)
        assert result > 0.99

    def test_large_negative(self):
        """큰 음수 → ~1.0 (abs 적용)"""
        engine = MartinMomentumEngine(sigmoid_k=5.0)
        result = engine.sigmoid_activation(-3.0)
        assert result > 0.99

    def test_moderate_value(self):
        """중간 값 → 0~1 사이"""
        engine = MartinMomentumEngine(sigmoid_k=5.0)
        result = engine.sigmoid_activation(0.5)
        assert 0.0 < result < 1.0

    def test_symmetry(self):
        """sigmoid(x) == sigmoid(-x) (abs 적용이므로)"""
        engine = MartinMomentumEngine(sigmoid_k=5.0)
        pos = engine.sigmoid_activation(1.0)
        neg = engine.sigmoid_activation(-1.0)
        assert abs(pos - neg) < 0.001


# ── 역전 Sigmoid 테스트 ──

class TestReversalSigmoid:
    def test_cap_constraint(self):
        """역전 강도 ≤ 추세 강도 × reversal_cap_ratio"""
        engine = MartinMomentumEngine(reversal_cap_ratio=0.68)
        for x in [0.5, 1.0, 1.5, 2.0]:
            trend = engine.sigmoid_activation(x)
            reversal = engine.reversal_sigmoid(x)
            assert reversal <= trend * 0.68 + 0.001


# ── 최적 보유기간 테스트 ──

class TestOptimalHold:
    def test_default(self):
        """기본값: 1.7 × (8+24) = 54.4 → 54"""
        engine = MartinMomentumEngine(n_fast=8, n_slow=24)
        assert engine.calc_optimal_hold() == 54

    def test_custom(self):
        """커스텀: 1.7 × (5+15) = 34"""
        engine = MartinMomentumEngine(n_fast=5, n_slow=15)
        assert engine.calc_optimal_hold() == 34


# ── 변동성 정규화 테스트 ──

class TestVolWeight:
    def test_high_volatility_reduces(self):
        """높은 변동성 → 비중 축소"""
        engine = MartinMomentumEngine(target_sigma=0.02, max_vol_weight=2.0, min_vol_weight=0.3)
        df = make_df(100, trend=0.01)
        # daily_sigma가 target보다 클 때
        df["daily_sigma"] = 0.04  # 4% 일간 변동성 (목표의 2배)
        weight = engine.calc_vol_weight(df, 99)
        assert weight == 0.5  # 0.02 / 0.04 = 0.5

    def test_low_volatility_increases(self):
        """낮은 변동성 → 비중 확대"""
        engine = MartinMomentumEngine(target_sigma=0.02, max_vol_weight=2.0)
        df = make_df(100)
        df["daily_sigma"] = 0.005  # 0.5%
        weight = engine.calc_vol_weight(df, 99)
        assert weight == 2.0  # 0.02/0.005 = 4.0 → clipped to 2.0

    def test_no_sigma_column(self):
        """daily_sigma 없으면 1.0"""
        engine = MartinMomentumEngine()
        df = pd.DataFrame({"close": [100, 101, 102]})
        assert engine.calc_vol_weight(df, 2) == 1.0

    def test_zero_sigma(self):
        """sigma=0 → 1.0 (안전 처리)"""
        engine = MartinMomentumEngine()
        df = make_df(10)
        df["daily_sigma"] = 0.0
        assert engine.calc_vol_weight(df, 5) == 1.0


# ── 종합 evaluate 테스트 ──

class TestEvaluate:
    def test_returns_dataclass(self):
        """evaluate()는 MartinMomentumResult 반환"""
        engine = MartinMomentumEngine()
        df = make_df(100)
        result = engine.evaluate(df, 99)
        assert isinstance(result, MartinMomentumResult)

    def test_dead_zone_signal(self):
        """Dead Zone 내 → confidence=0, signal_type=dead_zone"""
        engine = MartinMomentumEngine(epsilon=100.0)  # 매우 넓은 Dead Zone
        df = make_df(100)
        result = engine.evaluate(df, 99)
        assert bool(result.in_dead_zone) is True
        assert result.signal_type == "dead_zone"
        assert result.confidence == 0.0

    def test_trend_signal(self):
        """강한 상승추세 → trend 신호"""
        engine = MartinMomentumEngine(epsilon=0.01)
        df = make_df(100, trend=0.01)
        # 강제로 ema_8 > ema_24 격차 크게
        df["ema_8"] = df["close"] + 500
        df["ema_24"] = df["close"] - 500
        result = engine.evaluate(df, 99)
        assert result.ema2_value > 0
        assert result.signal_type in ("trend", "neutral")

    def test_optimal_hold_in_result(self):
        """결과에 최적 보유기간 포함"""
        engine = MartinMomentumEngine(n_fast=8, n_slow=24)
        df = make_df(100)
        result = engine.evaluate(df, 99)
        assert result.optimal_hold_days == 54
