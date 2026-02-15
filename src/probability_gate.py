"""
v4.5 Rolling Probability Gate

Adaptive Regime Sector Strategy 기반:
  - Rolling 200일 윈도우에서 "성공" 비율을 실시간 계산
  - 성공 정의: 진입 후 12일간 평균 수익률 >= 1%
  - 확률 >= 65% 이면 게이트 통과 (시장이 최근 수익 환경)

Look-ahead bias 방지:
  - 200일 윈도우는 (idx-212) ~ (idx-13) 범위 사용
  - 마지막 12일은 미래 수익률 계산이 필요하므로 제외
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProbabilityGate:
    """Rolling 확률 기반 시장 환경 필터"""

    def __init__(
        self,
        rolling_window: int = 200,
        threshold: float = 0.65,
        lookahead_bars: int = 12,
        min_avg_return: float = 0.01,
    ):
        self.rolling_window = rolling_window
        self.threshold = threshold
        self.lookahead_bars = lookahead_bars
        self.min_avg_return = min_avg_return

    def compute_rolling_probability(self, df: pd.DataFrame) -> pd.Series:
        """
        전체 DataFrame에 대해 rolling 확률 시리즈를 계산.

        각 bar i에 대해:
          윈도우 = [i - rolling_window - lookahead_bars + 1, i - lookahead_bars]
          각 bar j in 윈도우:
            성공 = mean(close[j+1:j+lookahead_bars+1]) / close[j] - 1 >= min_avg_return
          확률 = 성공 수 / 윈도우 크기

        Returns:
            pd.Series: 각 bar의 rolling 성공 확률 (0.0 ~ 1.0)
        """
        close = df["close"].values
        n = len(close)
        probs = np.full(n, np.nan)

        # 사전 계산: 각 bar j의 향후 lookahead_bars 일간 평균 수익률
        future_returns = np.full(n, np.nan)
        for j in range(n - self.lookahead_bars):
            future_avg = np.mean(close[j + 1: j + self.lookahead_bars + 1])
            future_returns[j] = future_avg / close[j] - 1

        # rolling window 계산
        min_idx = self.rolling_window + self.lookahead_bars
        for i in range(min_idx, n):
            # 윈도우: look-ahead bias 없는 범위
            window_end = i - self.lookahead_bars  # 이 bar까지는 결과가 확정됨
            window_start = window_end - self.rolling_window + 1

            if window_start < 0:
                continue

            window_returns = future_returns[window_start: window_end + 1]
            valid = ~np.isnan(window_returns)

            if valid.sum() < self.rolling_window * 0.5:
                continue

            successes = np.sum(window_returns[valid] >= self.min_avg_return)
            probs[i] = successes / valid.sum()

        return pd.Series(probs, index=df.index, name="prob_success")

    def check_gate(self, df: pd.DataFrame, idx: int) -> tuple[bool, float]:
        """
        특정 인덱스에서 확률 게이트 통과 여부 확인.

        Returns:
            (passed, probability): 통과 여부 + 현재 확률
        """
        prob_col = "prob_success"

        if prob_col in df.columns:
            prob = df[prob_col].iloc[idx]
        else:
            # 컬럼이 없으면 직접 계산 (성능 주의)
            prob_series = self.compute_rolling_probability(df)
            prob = prob_series.iloc[idx]

        if pd.isna(prob):
            return True, 0.0  # 데이터 부족 시 통과 (보수적 비활성화)

        return prob >= self.threshold, round(float(prob), 3)

    def compute_and_attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """확률 시리즈를 계산하여 DataFrame에 컬럼 추가."""
        df = df.copy()
        df["prob_success"] = self.compute_rolling_probability(df)
        return df
