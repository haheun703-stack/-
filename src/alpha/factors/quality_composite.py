"""Quality Composite Score — Q1~Q4 통합 (STEP 3-6)

Z(Q1)×0.35 + Z(Q2)×0.25 + Z(Q3)×0.25 + Z(Q4)×0.15

레짐별 가중치 조정:
  BULL:    Q1(0.40) Q2(0.15) Q3(0.15) Q4(0.30) — 수익성+배당
  CAUTION: Q1(0.35) Q2(0.25) Q3(0.25) Q4(0.15) — 기본값
  BEAR:    Q1(0.25) Q2(0.30) Q3(0.35) Q4(0.10) — 방어적 품질
  CRISIS:  Q1(0.20) Q2(0.35) Q3(0.35) Q4(0.10) — 생존 중심
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.alpha.factors.quality_roe import (
    QualityAccruals,
    QualityDebt,
    QualityDividend,
    QualityROE,
)

logger = logging.getLogger(__name__)

# 레짐별 Q 가중치
_Q_WEIGHTS = {
    "BULL":    {"Q1": 0.40, "Q2": 0.15, "Q3": 0.15, "Q4": 0.30},
    "CAUTION": {"Q1": 0.35, "Q2": 0.25, "Q3": 0.25, "Q4": 0.15},
    "BEAR":    {"Q1": 0.25, "Q2": 0.30, "Q3": 0.35, "Q4": 0.10},
    "CRISIS":  {"Q1": 0.20, "Q2": 0.35, "Q3": 0.35, "Q4": 0.10},
}


class QualityComposite:
    """Q1~Q4 통합 퀄리티 스코어"""

    def __init__(self, quality_data: dict | None = None):
        self._q1 = QualityROE(quality_data)
        self._q2 = QualityDebt(quality_data)
        self._q3 = QualityAccruals(quality_data)
        self._q4 = QualityDividend(quality_data)

        # 유니버스 Z-Score 캐시
        self._scores_cache: dict[str, dict[str, float]] | None = None

    def _ensure_scores(self):
        """유니버스 Z-Score 계산 (1회만)."""
        if self._scores_cache is not None:
            return

        self._scores_cache = {
            "Q1": self._q1.score_universe(),
            "Q2": self._q2.score_universe(),
            "Q3": self._q3.score_universe(),
            "Q4": self._q4.score_universe(),
        }

    def score(self, ticker: str, regime: str = "CAUTION") -> float:
        """단일 종목 통합 Quality 스코어.

        Args:
            ticker: 종목코드
            regime: BULL/CAUTION/BEAR/CRISIS

        Returns:
            0.0~1.0 통합 Q 스코어
        """
        self._ensure_scores()

        weights = _Q_WEIGHTS.get(regime, _Q_WEIGHTS["CAUTION"])

        total = 0.0
        weight_sum = 0.0

        for key in ["Q1", "Q2", "Q3", "Q4"]:
            factor_scores = self._scores_cache[key]
            if ticker in factor_scores:
                total += factor_scores[ticker] * weights[key]
                weight_sum += weights[key]

        if weight_sum < 0.5:
            return 0.5  # 데이터 부족 → 중립

        return round(total / weight_sum, 4)

    def score_universe(self, regime: str = "CAUTION") -> dict[str, float]:
        """전체 유니버스 통합 Q 스코어."""
        self._ensure_scores()

        # Q1~Q4 모두 스코어가 있는 종목 집합
        all_tickers = set()
        for key in ["Q1", "Q2", "Q3", "Q4"]:
            all_tickers |= set(self._scores_cache[key].keys())

        result: dict[str, float] = {}
        for ticker in all_tickers:
            result[ticker] = self.score(ticker, regime)

        return result

    def score_breakdown(self, ticker: str, regime: str = "CAUTION") -> dict:
        """개별 종목 Q 스코어 분해."""
        self._ensure_scores()

        weights = _Q_WEIGHTS.get(regime, _Q_WEIGHTS["CAUTION"])

        breakdown = {}
        for key in ["Q1", "Q2", "Q3", "Q4"]:
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
