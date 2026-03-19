"""Value Composite Score — 기존 S2 확장 (STEP 4-3)

기존 S2: PER할인(0.6) + EPS revision(0.4) = 0~1.0
신규 V1: EBITDA/EV → Z-Score (0~1.0)
신규 V2: FCF Yield → Z-Score (0~1.0)

통합:
  V_composite = S2_existing×0.40 + V1_EBITDA_EV×0.35 + V2_FCF_Yield×0.25

레짐별 가중치 조정:
  BULL:    S2(0.50) V1(0.25) V2(0.25) — 모멘텀 밸류에이션 중시
  CAUTION: S2(0.40) V1(0.35) V2(0.25) — 기본값
  BEAR:    S2(0.30) V1(0.35) V2(0.35) — 현금흐름 중시
  CRISIS:  S2(0.20) V1(0.35) V2(0.45) — FCF 생존력 중시
"""

from __future__ import annotations

import logging

from src.alpha.factors.value_ebitda_ev import (
    ValueEbitdaEV,
    ValueFCFYield,
)

logger = logging.getLogger(__name__)

# 레짐별 V 가중치
_V_WEIGHTS = {
    "BULL":    {"S2": 0.50, "V1": 0.25, "V2": 0.25},
    "CAUTION": {"S2": 0.40, "V1": 0.35, "V2": 0.25},
    "BEAR":    {"S2": 0.30, "V1": 0.35, "V2": 0.35},
    "CRISIS":  {"S2": 0.20, "V1": 0.35, "V2": 0.45},
}


class ValueComposite:
    """기존 S2 + 신규 V1/V2 통합 밸류 스코어"""

    def __init__(
        self,
        financial_data: dict | None = None,
        market_cap_data: dict | None = None,
    ):
        self._v1 = ValueEbitdaEV(financial_data, market_cap_data)
        self._v2 = ValueFCFYield(financial_data, market_cap_data)

        # 유니버스 Z-Score 캐시
        self._scores_cache: dict[str, dict[str, float]] | None = None

    def _ensure_scores(self):
        """유니버스 Z-Score 계산 (1회만)."""
        if self._scores_cache is not None:
            return

        self._scores_cache = {
            "V1": self._v1.score_universe(),
            "V2": self._v2.score_universe(),
        }

    def score(
        self,
        ticker: str,
        regime: str = "CAUTION",
        s2_score: float | None = None,
    ) -> float:
        """단일 종목 통합 Value 스코어.

        Args:
            ticker: 종목코드
            regime: BULL/CAUTION/BEAR/CRISIS
            s2_score: 기존 S2 스코어 (0~1.0). None이면 V1/V2만으로 계산.

        Returns:
            0.0~1.0 통합 V 스코어
        """
        self._ensure_scores()

        weights = _V_WEIGHTS.get(regime, _V_WEIGHTS["CAUTION"])

        total = 0.0
        weight_sum = 0.0

        # 기존 S2
        if s2_score is not None:
            total += s2_score * weights["S2"]
            weight_sum += weights["S2"]

        # V1 EBITDA/EV
        v1_scores = self._scores_cache["V1"]
        if ticker in v1_scores:
            total += v1_scores[ticker] * weights["V1"]
            weight_sum += weights["V1"]

        # V2 FCF Yield
        v2_scores = self._scores_cache["V2"]
        if ticker in v2_scores:
            total += v2_scores[ticker] * weights["V2"]
            weight_sum += weights["V2"]

        if weight_sum < 0.3:
            return 0.5  # 데이터 부족 → 중립

        return round(total / weight_sum, 4)

    def score_universe(self, regime: str = "CAUTION") -> dict[str, float]:
        """전체 유니버스 V1+V2 스코어 (S2 없이)."""
        self._ensure_scores()

        all_tickers = set()
        for key in ["V1", "V2"]:
            all_tickers |= set(self._scores_cache[key].keys())

        result: dict[str, float] = {}
        for ticker in all_tickers:
            result[ticker] = self.score(ticker, regime)

        return result

    def score_breakdown(self, ticker: str, regime: str = "CAUTION") -> dict:
        """개별 종목 V 스코어 분해."""
        self._ensure_scores()

        weights = _V_WEIGHTS.get(regime, _V_WEIGHTS["CAUTION"])

        breakdown = {}
        for key in ["V1", "V2"]:
            factor_scores = self._scores_cache[key]
            z_score = factor_scores.get(ticker, None)
            breakdown[key] = {
                "z_score": z_score,
                "weight": weights[key],
                "weighted": round(z_score * weights[key], 4) if z_score else None,
            }

        breakdown["total"] = self.score(ticker, regime)
        breakdown["regime"] = regime

        return breakdown
