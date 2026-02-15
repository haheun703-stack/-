"""
v6.0 Martin Momentum Engine

Martin(2023) 논문 6대 핵심 포인트 구현:
1. EMA2 필터: fast EMA - slow EMA (EMA1보다 우수)
2. Dead Zone: |ema2_norm| < ε → 약한 신호 무시 (SR 손실 없음)
3. Sigmoid 활성화: 연속적 신호 강도 (0~1)
4. 역전 Sigmoid: 추세 반전 신호 (하이브리드 비중 제약)
5. 변동성 정규화: position ∝ 1/σ
6. 최적 보유기간: ≈ 1.7 × (N_fast + N_slow)
"""

import logging
import math

import numpy as np
import pandas as pd

from .entities.momentum_models import MartinMomentumResult

logger = logging.getLogger(__name__)


class MartinMomentumEngine:
    """Martin(2023) 모멘텀 전략 엔진"""

    def __init__(
        self,
        n_fast: int = 8,
        n_slow: int = 24,
        epsilon: float = 0.6,
        sigmoid_k: float = 5.0,
        min_confidence: float = 0.3,
        target_sigma: float = 0.02,
        max_vol_weight: float = 2.0,
        min_vol_weight: float = 0.3,
        reversal_cap_ratio: float = 0.68,
    ):
        self.n_fast = n_fast
        self.n_slow = n_slow
        self.epsilon = epsilon
        self.sigmoid_k = sigmoid_k
        self.min_confidence = min_confidence
        self.target_sigma = target_sigma
        self.max_vol_weight = max_vol_weight
        self.min_vol_weight = min_vol_weight
        self.reversal_cap_ratio = reversal_cap_ratio  # 역전 비중 < 추세 비중 × 68%

    def calc_ema2(self, df: pd.DataFrame, idx: int) -> float:
        """EMA2 = EMA(close, n_fast) - EMA(close, n_slow)"""
        if idx < 0 or idx >= len(df):
            return 0.0

        if "ema_8" not in df.columns or "ema_24" not in df.columns:
            return 0.0

        fast_val = df["ema_8"].iloc[idx]
        slow_val = df["ema_24"].iloc[idx]

        if pd.isna(fast_val) or pd.isna(slow_val):
            return 0.0

        return float(fast_val - slow_val)

    def calc_ema2_normalized(self, ema2_value: float, close: float) -> float:
        """EMA2 정규화 = EMA2 / close * 100"""
        if close <= 0:
            return 0.0
        return ema2_value / close * 100

    def check_dead_zone(self, ema2_norm: float) -> bool:
        """|ema2_norm| < epsilon → True (Dead Zone, 신호 무시)"""
        return abs(ema2_norm) < self.epsilon

    def sigmoid_activation(self, ema2_norm: float) -> float:
        """
        Sigmoid 활성화: 2*sigmoid(k*x) - 1 → [-1, 1] → abs → [0, 1]

        추세 방향의 신호 강도를 연속적으로 측정.
        Dead Zone 밖에서만 유의미한 값 반환.
        """
        try:
            raw = 2.0 / (1.0 + math.exp(-self.sigmoid_k * ema2_norm)) - 1.0
        except OverflowError:
            raw = 1.0 if ema2_norm > 0 else -1.0
        return abs(raw)

    def reversal_sigmoid(self, ema2_norm: float) -> float:
        """
        역전 Sigmoid: 신호 방향이 반대일 때의 강도.
        하이브리드 비중 제약: 역전 비중 < 추세 비중 × reversal_cap_ratio
        """
        trend = self.sigmoid_activation(ema2_norm)
        # 역전 강도 = 1 - trend (추세가 약할수록 역전 가능성 ↑)
        reversal = 1.0 - trend
        # 역전 비중 상한: 추세 비중 × cap_ratio
        max_reversal = trend * self.reversal_cap_ratio
        return min(reversal, max_reversal)

    def calc_optimal_hold(self) -> int:
        """최적 보유기간 = int(1.7 * (N_fast + N_slow))"""
        return int(1.7 * (self.n_fast + self.n_slow))

    def calc_vol_weight(self, df: pd.DataFrame, idx: int) -> float:
        """
        변동성 정규화 비중 = clip(target_sigma / realized_sigma_20d, min, max)

        Martin 논문: position ∝ 1/σ → 변동성 높으면 비중 축소
        """
        if idx < 0 or idx >= len(df) or "daily_sigma" not in df.columns:
            return 1.0

        sigma_val = df["daily_sigma"].iloc[idx]
        if pd.isna(sigma_val) or sigma_val <= 0:
            return 1.0

        weight = self.target_sigma / sigma_val
        return float(np.clip(weight, self.min_vol_weight, self.max_vol_weight))

    def evaluate(self, df: pd.DataFrame, idx: int) -> MartinMomentumResult:
        """
        종합 Martin 모멘텀 평가.

        Returns:
            MartinMomentumResult: 모든 Martin 지표를 포함한 결과
        """
        if idx < 0 or idx >= len(df) or "close" not in df.columns:
            return MartinMomentumResult()

        close = df["close"].iloc[idx]
        if pd.isna(close) or close <= 0:
            return MartinMomentumResult()

        # 1. EMA2 계산
        ema2_val = self.calc_ema2(df, idx)
        ema2_norm = self.calc_ema2_normalized(ema2_val, close)

        # 2. Dead Zone 체크
        in_dead = self.check_dead_zone(ema2_norm)

        # 3. 활성화 함수
        trend_str = self.sigmoid_activation(ema2_norm)
        reversal_str = self.reversal_sigmoid(ema2_norm)

        # 4. 신호 유형 결정
        if in_dead:
            signal_type = "dead_zone"
        elif ema2_norm > 0 and trend_str > 0.3:
            signal_type = "trend"
        elif ema2_norm < 0 and reversal_str > 0.2:
            signal_type = "reversal"
        else:
            signal_type = "neutral"

        # 5. 최적 보유기간
        optimal_hold = self.calc_optimal_hold()

        # 6. 변동성 정규화 비중
        vol_weight = self.calc_vol_weight(df, idx)

        # 7. 종합 신뢰도
        if in_dead:
            confidence = 0.0
        elif signal_type == "trend":
            confidence = trend_str * 0.7 + min(vol_weight, 1.0) * 0.3
        elif signal_type == "reversal":
            confidence = reversal_str * 0.5
        else:
            confidence = trend_str * 0.3

        confidence = float(np.clip(confidence, 0.0, 1.0))

        return MartinMomentumResult(
            ema2_value=round(ema2_val, 4),
            ema2_normalized=round(ema2_norm, 4),
            in_dead_zone=in_dead,
            trend_strength=round(trend_str, 4),
            reversal_strength=round(reversal_str, 4),
            signal_type=signal_type,
            optimal_hold_days=optimal_hold,
            vol_normalized_weight=round(vol_weight, 4),
            confidence=round(confidence, 4),
        )
