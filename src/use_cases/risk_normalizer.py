"""
v6.0 리스크 예산 정규화

WaveLSFormer 논문 기반:
- position = raw * clip(target_vol / realized_vol, min_scale, max_scale)
- 검증셋에서 s_val 한 번 계산 → 고정
- 변동성 높으면 비중 축소, 낮으면 비중 확대

의존성: 없음 (순수 유틸리티)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskBudgetNormalizer:
    """WaveLSFormer 리스크 예산 정규화"""

    def __init__(
        self,
        target_daily_vol: float = 0.02,
        lookback: int = 60,
        max_scale: float = 2.0,
        min_scale: float = 0.3,
    ):
        self.target_daily_vol = target_daily_vol
        self.lookback = lookback
        self.max_scale = max_scale
        self.min_scale = min_scale

    def normalize_position(
        self,
        df: pd.DataFrame,
        idx: int,
        raw_shares: int,
        entry_price: float,
    ) -> int:
        """
        리스크 예산 정규화.

        position = raw * clip(target_vol / realized_vol, min_scale, max_scale)
        """
        if raw_shares <= 0 or entry_price <= 0:
            return raw_shares

        realized_vol = self._calc_realized_vol(df, idx)
        if realized_vol <= 0:
            return raw_shares

        scale = self.target_daily_vol / realized_vol
        scale = float(np.clip(scale, self.min_scale, self.max_scale))

        normalized = int(raw_shares * scale)
        return max(normalized, 1)

    def calc_scale_factor(self, df: pd.DataFrame, idx: int) -> float:
        """스케일 팩터만 반환 (외부에서 사용)"""
        realized_vol = self._calc_realized_vol(df, idx)
        if realized_vol <= 0:
            return 1.0

        scale = self.target_daily_vol / realized_vol
        return float(np.clip(scale, self.min_scale, self.max_scale))

    def _calc_realized_vol(self, df: pd.DataFrame, idx: int) -> float:
        """실현 변동성 (일간 수익률의 표준편차)"""
        if "ret1" not in df.columns:
            return 0.0

        start = max(0, idx - self.lookback)
        returns = df["ret1"].iloc[start:idx + 1].dropna()

        if len(returns) < 10:
            return 0.0

        return float(returns.std())
