"""
HMM 기반 3-상태 레짐 감지

논문 기반:
  - 동적 적응 전략 논문: MS-AR 4-레짐 (Wyckoff)
  - 참조 코드: GaussianHMM 3-state (Advance/Distribution/Accumulation)
  - OUbv 논문: phi(t) 모멘텀/평균회귀 전환

3-state 레짐:
  Advance     = 상승 추세 (ret1 최대)
  Distribution = 하락/분배 (ret1 최소)
  Accumulation = 횡보/축적 (나머지) → 진입 허용
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn/sklearn 미설치. 레짐 감지 비활성화됩니다.")


class RegimeDetector:
    """HMM 3-state 레짐 감지기"""

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_state: int = 7):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None
        self._scaler = None

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        특징(ret1, ATR_pct, vol_z, smart_z)으로 HMM 학습 후 상태 확률 반환.

        Returns:
            DataFrame with columns: P_Advance, P_Distrib, P_Accum
        """
        if not HMM_AVAILABLE:
            return self._fallback(df)

        feature_cols = []
        for col in ["ret1", "ATR_pct", "vol_z", "smart_z"]:
            if col in df.columns:
                feature_cols.append(col)

        if len(feature_cols) < 2:
            return self._fallback(df)

        feats = df[feature_cols].dropna()
        if len(feats) < 100:
            return self._fallback(df)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(feats.values)

        try:
            self._model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self._model.fit(X)
            proba = self._model.predict_proba(X)
        except Exception as e:
            logger.warning(f"HMM 학습 실패: {e}")
            return self._fallback(df)

        proba_df = pd.DataFrame(
            proba, index=feats.index, columns=[f"S{i}" for i in range(self.n_states)]
        )

        # 상태 라벨링: ret1 기준
        state_means = pd.DataFrame(
            self._model.means_, columns=feature_cols
        )
        if "ret1" in state_means.columns:
            advance_state = int(state_means["ret1"].idxmax())
            dist_state = int(state_means["ret1"].idxmin())
        else:
            advance_state, dist_state = 0, 1

        acc_state = ({0, 1, 2} - {advance_state, dist_state}).pop()

        result = pd.DataFrame(index=df.index, dtype=float)
        result["P_Advance"] = np.nan
        result["P_Distrib"] = np.nan
        result["P_Accum"] = np.nan

        result.loc[proba_df.index, "P_Advance"] = proba_df.iloc[:, advance_state].values
        result.loc[proba_df.index, "P_Distrib"] = proba_df.iloc[:, dist_state].values
        result.loc[proba_df.index, "P_Accum"] = proba_df.iloc[:, acc_state].values

        # NaN 채우기 (초기 구간)
        result = result.ffill().fillna(1 / self.n_states)

        return result

    def _fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """HMM 불가 시 균등 확률 반환"""
        result = pd.DataFrame(index=df.index, dtype=float)
        result["P_Advance"] = 1 / 3
        result["P_Distrib"] = 1 / 3
        result["P_Accum"] = 1 / 3
        return result

    @staticmethod
    def check_regime_gate(
        row: pd.Series,
        p_accum_entry: float = 0.40,
        extreme_vol_direction: str | None = None,
    ) -> tuple[bool, str]:
        """
        레짐 게이트 체크.

        v6.0: 극한 변동성 방향에 따른 통과/차단
        - capitulation → 통과 허용 (반전 기회)
        - bullish_breakout → 통과 허용
        - bearish_breakdown / ambiguous → 차단

        Returns:
            (passed, block_reason)
        """
        # v6.0: 극한 변동성이 감지된 경우 방향 기반 판단
        if extreme_vol_direction:
            if extreme_vol_direction in ("bearish_breakdown", "ambiguous"):
                return False, f"extreme_vol_{extreme_vol_direction}"
            # capitulation 또는 bullish_breakout → 레짐 무시하고 통과
            if extreme_vol_direction in ("capitulation", "bullish_breakout"):
                return True, ""

        p_accum = row.get("P_Accum", np.nan)
        if pd.isna(p_accum):
            return True, ""  # 데이터 없으면 통과

        if p_accum < p_accum_entry:
            return False, "low_accum"

        return True, ""
