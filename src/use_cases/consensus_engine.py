"""
v5.0 ConsensusVerifier — Sci-CoE 기하학적 합의 판정 엔진

Sci-CoE 논문의 3축 기하학적 보상을 퀀텀전략 파이프라인에 적용:
  축 ①: Consistency (합의성) — 레이어 통과율 / tau 정규화
  축 ②: Reliability (신뢰성) — 앵커 DB 중심 거리 또는 confidence 평균
  축 ③: Diversity (다양성) — 10지표 신호 각도 분산

최종 점수: C * R * D^(1/3)

클린 아키텍처: entities만 import.
"""

import logging
import math

logger = logging.getLogger(__name__)

from src.entities.consensus_models import (
    AnchorDatabase,
    ConsensusResult,
)


class ConsensusVerifier:
    """3축 기하학적 합의 판정기"""

    def __init__(
        self,
        tau: float = 0.8,
        min_voters: int = 4,
        grade_thresholds: dict | None = None,
        anchor_db: AnchorDatabase | None = None,
    ):
        self.tau = tau
        self.min_voters = min_voters
        self.thresholds = grade_thresholds or {
            "strong": 0.6,
            "moderate": 0.4,
            "weak": 0.2,
        }
        self.anchor_db = anchor_db

    def verify(
        self,
        votes: list,
        geo_indicators: dict | None = None,
    ) -> ConsensusResult:
        """
        3축 합의 판정 수행.

        Args:
            votes: LayerVote 리스트
            geo_indicators: 10지표 결과 dict (Diversity 계산용)

        Returns:
            ConsensusResult
        """
        if not votes:
            return ConsensusResult()

        try:
            passed = [v for v in votes if v.passed]
            total = len(votes)

            consistency = self.calc_consistency(votes)
            reliability = self.calc_reliability(votes)
            diversity = self.calc_diversity(votes, geo_indicators)
            geometric_reward = self.calc_geometric_reward(
                consistency, reliability, diversity
            )
            grade = self._determine_grade(geometric_reward)

            return ConsensusResult(
                consistency=round(consistency, 4),
                reliability=round(reliability, 4),
                diversity=round(diversity, 4),
                geometric_reward=round(geometric_reward, 4),
                consensus_grade=grade,
                votes=votes,
                passed_voters=len(passed),
                total_voters=total,
            )
        except Exception as e:
            logger.error("Consensus verification failed: %s", e)
            return ConsensusResult()

    def calc_consistency(self, votes: list) -> float:
        """
        축 ①: 합의성 — 레이어 통과율 / tau 정규화.

        통과율 = passed_count / total_count
        정규화 = min(통과율 / tau, 1.0)
        """
        if not votes or self.tau <= 0:
            return 0.0

        passed_count = sum(1 for v in votes if v.passed)
        total = len(votes)
        pass_rate = passed_count / total

        return min(pass_rate / self.tau, 1.0)

    def calc_reliability(self, votes: list) -> float:
        """
        축 ②: 신뢰성 — 앵커 DB 있으면 중심 거리, 없으면 confidence 평균.

        앵커 DB 모드:
            features 벡터와 success_centroid 간 유클리드 거리 계산.
            r = 1 - d / (max_d + epsilon)

        Fallback 모드:
            통과 레이어들의 confidence 평균.
        """
        if self.anchor_db and self.anchor_db.success_centroid:
            return self._anchor_reliability(votes)

        # Fallback: confidence 평균
        passed_votes = [v for v in votes if v.passed]
        if not passed_votes:
            return 0.0

        return sum(v.confidence for v in passed_votes) / len(passed_votes)

    def _anchor_reliability(self, votes: list) -> float:
        """앵커 DB 기반 신뢰성 계산"""
        centroid = self.anchor_db.success_centroid
        if not centroid:
            return 0.5

        # 현재 투표의 특성 벡터 구축
        current_features = {}
        for v in votes:
            current_features[v.layer_name] = v.confidence

        # 유클리드 거리 계산
        keys = set(centroid.keys()) & set(current_features.keys())
        if not keys:
            return 0.5

        dist_sq = sum(
            (current_features[k] - centroid[k]) ** 2 for k in keys
        )
        distance = math.sqrt(dist_sq)

        # 정규화: max 거리 = sqrt(len(keys)) (모든 차원에서 1.0 차이)
        max_distance = math.sqrt(len(keys))
        epsilon = 1e-8

        return max(0.0, 1.0 - distance / (max_distance + epsilon))

    def calc_diversity(
        self,
        votes: list,
        geo_indicators: dict | None = None,
    ) -> float:
        """
        축 ③: 다양성 — 지표 신호 각도 분산 + 레이어 카테고리 분산.

        Sci-CoE의 PCA → 극좌표 → 각도 분산 개념을 간소화 적용:
          1) 10지표 신호 다양성: BUY/SELL/HOLD 분포의 엔트로피
          2) 레이어 카테고리 분산: 통과 레이어가 다양한 카테고리에서 나오는지

        최종 = 0.6 * 지표 다양성 + 0.4 * 레이어 카테고리 다양성
        """
        indicator_div = self._indicator_diversity(geo_indicators)
        layer_div = self._layer_category_diversity(votes)

        return 0.6 * indicator_div + 0.4 * layer_div

    def _indicator_diversity(self, geo_indicators: dict | None) -> float:
        """10지표 신호 다양성 (정보 엔트로피 기반)"""
        if not geo_indicators:
            return 0.5  # 데이터 없으면 중립

        signals = []
        for name, ind_data in geo_indicators.items():
            if isinstance(ind_data, dict):
                signals.append(ind_data.get("signal", "HOLD"))
            else:
                signals.append("HOLD")

        if not signals:
            return 0.5

        # 신호 분포 계산
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for s in signals:
            counts[s] = counts.get(s, 0) + 1

        total = len(signals)
        if total == 0:
            return 0.5

        # Shannon 엔트로피 (최대 = log2(3) ≈ 1.585)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(3)  # 3가지 신호의 최대 엔트로피
        return min(entropy / max_entropy, 1.0)

    def _layer_category_diversity(self, votes: list) -> float:
        """레이어 카테고리 다양성"""
        if not votes:
            return 0.0

        # 레이어를 카테고리별로 분류
        # v6.0: 극한 변동성, Martin, RTTP 카테고리 추가
        categories = {
            "news": ["L-1_rttp"],
            "position": ["L0_grade"],
            "regime": ["L1_regime", "L1_extreme_vol"],
            "filter": ["L2_ou", "L4.5_probability"],
            "momentum": ["L3_momentum", "L3_martin", "L4_smart_money"],
            "trigger": ["L6_trigger"],
            "risk": ["L5_risk"],
            "geometric": ["L7_geometric"],
        }

        # 각 카테고리에서 통과한 레이어가 있는지 확인
        passed_names = {v.layer_name for v in votes if v.passed}
        categories_hit = 0

        for cat_layers in categories.values():
            if any(layer in passed_names for layer in cat_layers):
                categories_hit += 1

        total_categories = len(categories)
        return categories_hit / total_categories if total_categories > 0 else 0.0

    def calc_geometric_reward(
        self, consistency: float, reliability: float, diversity: float
    ) -> float:
        """
        최종 기하학적 보상: C * R * D^(1/3)

        Sci-CoE의 가중합 대신 곱셈 형태를 사용하여
        한 축이라도 0이면 전체가 0이 되는 균형 특성 반영.
        D에 1/3 지수를 적용하여 다양성의 영향을 부드럽게 조절.
        """
        if consistency <= 0 or reliability <= 0:
            return 0.0

        d_term = diversity ** (1.0 / 3.0) if diversity > 0 else 0.0
        return consistency * reliability * d_term

    def _determine_grade(self, geometric_reward: float) -> str:
        """등급 판정"""
        if geometric_reward >= self.thresholds["strong"]:
            return "strong"
        elif geometric_reward >= self.thresholds["moderate"]:
            return "moderate"
        elif geometric_reward >= self.thresholds["weak"]:
            return "weak"
        else:
            return "reject"
