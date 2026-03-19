"""
Alpha Engine V2 — 레짐 조건부 가중합 스코어러

STEP 1 독립 팩터 백테스트 결과 기반:
  S5(수급) Sharpe 0.96 >> S4(모멘텀) 0.53 > S3(OU) 0.47 > S1(에너지) 0.29 > S2(밸류) 0.16

레짐별 가중치:
  BULL:    공격축(S5+S4) 강화 — 모멘텀+수급이 이끄는 시장
  CAUTION: Sharpe 비율 기본값
  BEAR:    방어축(S1+S3) 강화 — 에너지소진+평균회귀 중심
  CRISIS:  매수 금지 (가중치 전부 0)

사용법:
  from src.alpha.factors.regime_weighted_scorer import RegimeWeightedScorer
  scorer = RegimeWeightedScorer(config)
  result = scorer.score(row, regime_level)
"""

from __future__ import annotations

import logging

import pandas as pd

from src.alpha.models import AlphaRegimeLevel
from src.v8_scorers import GradeResult, ScoreResult, ScoringEngine

logger = logging.getLogger(__name__)

# 기존 5축 기본 가중치 (V1, 변경 없음)
_V1_WEIGHTS = {
    "S1": 0.30,
    "S2": 0.20,
    "S3": 0.20,
    "S4": 0.15,
    "S5": 0.15,
}

# Sharpe 비율 기반 기본값 (settings.yaml 없을 때 폴백)
_DEFAULT_V2_WEIGHTS = {
    "BULL":    {"S1": 0.12, "S2": 0.08, "S3": 0.15, "S4": 0.25, "S5": 0.40},
    "CAUTION": {"S1": 0.12, "S2": 0.07, "S3": 0.19, "S4": 0.22, "S5": 0.40},
    "BEAR":    {"S1": 0.25, "S2": 0.10, "S3": 0.30, "S4": 0.10, "S5": 0.25},
    "CRISIS":  {"S1": 0.00, "S2": 0.00, "S3": 0.00, "S4": 0.00, "S5": 0.00},
}

_SCORER_KEYS = ["S1", "S2", "S3", "S4", "S5"]


class RegimeWeightedScorer:
    """레짐별 동적 가중합으로 기존 5축 점수를 재합산"""

    def __init__(self, config: dict):
        self._config = config
        self._scorer = ScoringEngine(config)

        # settings.yaml에서 V2 가중치 로딩
        v2_cfg = config.get("alpha_v2", {})
        self._weights = {}

        scorer_weights = v2_cfg.get("scorer_weights", {})
        for regime in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
            regime_lower = regime.lower()
            if regime_lower in scorer_weights:
                self._weights[regime] = scorer_weights[regime_lower]
            else:
                self._weights[regime] = _DEFAULT_V2_WEIGHTS[regime]

        # 등급 커트라인 (기존과 동일)
        v8_cfg = config.get("v8_hybrid", {})
        scoring_cfg = v8_cfg.get("scoring", {})
        self._cutoffs = scoring_cfg.get("grade_cutoffs", {
            "A": 0.65, "B": 0.50, "C": 0.35
        })

        logger.debug(
            "V2 RegimeWeightedScorer 초기화: BULL=%s, CAUTION=%s",
            self._weights["BULL"], self._weights["CAUTION"],
        )

    def score(
        self,
        row: pd.Series,
        regime_level: AlphaRegimeLevel,
    ) -> GradeResult:
        """
        레짐별 가중합으로 5축 점수를 재합산하여 등급 결정.

        Args:
            row: parquet DataFrame의 한 행
            regime_level: 현재 시장 레짐

        Returns:
            GradeResult (기존과 동일 인터페이스)
        """
        # 1. 기존 5축 스코어 개별 계산 (raw score 0.0~1.0)
        raw_scores = [
            self._scorer.score_energy_depletion(row),
            self._scorer.score_valuation(row),
            self._scorer.score_ou_reversion(row),
            self._scorer.score_momentum_deceleration(row),
            self._scorer.score_smart_money(row),
        ]

        # 2. 레짐별 가중치 적용
        weights = self._weights.get(
            regime_level.value, _DEFAULT_V2_WEIGHTS["CAUTION"]
        )

        v2_scores = []
        for i, key in enumerate(_SCORER_KEYS):
            w = weights.get(key, 0.0)
            v2_scores.append(ScoreResult(
                name=raw_scores[i].name,
                score=raw_scores[i].score,
                weight=w,
                breakdown=raw_scores[i].breakdown,
            ))

        # 3. 가중합
        total = sum(s.weighted for s in v2_scores)

        # 4. 등급 결정
        grade = self._determine_grade(total)

        # 5. 포지션 비중
        pos_cfg = self._config.get("v8_hybrid", {}).get("position", {})
        pos_pct = 0.0
        tradeable = False
        if grade == "A":
            pos_pct = pos_cfg.get("A_grade_pct", 0.20)
            tradeable = True
        elif grade == "B":
            pos_pct = pos_cfg.get("B_grade_pct", 0.10)
            tradeable = True

        return GradeResult(
            total_score=total,
            grade=grade,
            scores=v2_scores,
            position_size_pct=pos_pct,
            tradeable=tradeable,
        )

    def _determine_grade(self, total: float) -> str:
        if total >= self._cutoffs.get("A", 0.65):
            return "A"
        elif total >= self._cutoffs.get("B", 0.50):
            return "B"
        elif total >= self._cutoffs.get("C", 0.35):
            return "C"
        else:
            return "F"

    def get_weights(self, regime_level: AlphaRegimeLevel) -> dict:
        """현재 레짐의 가중치 반환 (디버그용)"""
        return self._weights.get(
            regime_level.value, _DEFAULT_V2_WEIGHTS["CAUTION"]
        )
