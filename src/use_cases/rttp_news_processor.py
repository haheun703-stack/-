"""
v6.0 RTTP 뉴스 프로세서

RTTP 논문 핵심 구현:
1. 소스 권위 가중 점수 계산
2. 인게이지먼트 깊이 5단계 계산
3. Recall@K 모니터링 및 재조정 트리거
4. 기존 NewsGateResult에 RTTP 점수 가산

의존성: entities만 (클린 아키텍처 준수)
"""

import logging

import numpy as np
import pandas as pd

from src.entities.rttp_models import (
    DEFAULT_ENGAGEMENT_WEIGHTS,
    DEFAULT_SOURCE_WEIGHTS,
    RecallTracker,
    RttpEnhancement,
    SourceTier,
)
from src.entities.news_models import NewsGateResult, NewsItem

logger = logging.getLogger(__name__)


class RttpNewsProcessor:
    """RTTP 뉴스 강화 프로세서"""

    def __init__(
        self,
        source_weights: dict | None = None,
        engagement_weights: dict | None = None,
        recall_threshold: float = 0.5,
        recall_window: int = 100,
    ):
        self.source_weights = source_weights or DEFAULT_SOURCE_WEIGHTS
        self.engagement_weights = engagement_weights or DEFAULT_ENGAGEMENT_WEIGHTS
        self.recall_tracker = RecallTracker(
            window_size=recall_window,
        )
        self.recall_threshold = recall_threshold

    def calc_source_weighted_score(self, news_items: list[NewsItem]) -> float:
        """
        소스 권위 가중 평균 점수.

        각 뉴스의 소스를 SourceTier에 매핑하고 가중 평균 계산.
        DART > Bloomberg > 증권사 > 포털 > 커뮤니티
        """
        if not news_items:
            return 0.0

        total_weight = 0.0
        total_score = 0.0

        for item in news_items:
            tier = self._classify_source(item.source)
            weight = self.source_weights.get(tier, 0.15)
            impact = item.impact_score / 10.0 if item.impact_score > 0 else 0.5
            total_weight += weight
            total_score += weight * impact

        if total_weight == 0:
            return 0.0

        return round(total_score / total_weight, 4)

    def calc_engagement_depth(
        self,
        df: pd.DataFrame,
        idx: int,
        news_items: list[NewsItem],
    ) -> float:
        """
        5단계 인게이지먼트 깊이 계산.

        L5(기관순매수=5.0) > L4(거래량=3.5) > L3(리포트=2.0) > L2(검색=1.0) > L1(SNS=0.3)
        실제 시장 데이터와 뉴스 메타데이터를 조합하여 계산.
        """
        if idx < 0 or idx >= len(df):
            return 0.0

        row = df.iloc[idx]
        depth_score = 0.0
        max_possible = sum(self.engagement_weights.values())

        # L5: 기관 순매수 감지
        inst_streak = row.get("inst_net_streak", 0)
        if not pd.isna(inst_streak) and inst_streak >= 3:
            depth_score += self.engagement_weights.get("L5_inst", 5.0)

        # L4: 거래량 급증 감지
        vol_surge = row.get("volume_surge_ratio", np.nan)
        if not pd.isna(vol_surge) and vol_surge >= 2.0:
            depth_score += self.engagement_weights.get("L4_volume", 3.5)

        # L3: 증권사 리포트 (소스에서 판별)
        has_brokerage = any(
            self._classify_source(n.source) == SourceTier.BROKERAGE.value
            for n in news_items
        )
        if has_brokerage:
            depth_score += self.engagement_weights.get("L3_report", 2.0)

        # L2: 검색량 증가 (뉴스 건수로 대리)
        if len(news_items) >= 3:
            depth_score += self.engagement_weights.get("L2_search", 1.0)

        # L1: SNS/커뮤니티 (소스 기반)
        has_community = any(
            self._classify_source(n.source) == SourceTier.COMMUNITY.value
            for n in news_items
        )
        if has_community:
            depth_score += self.engagement_weights.get("L1_sns", 0.3)

        # 정규화: 0~1
        normalized = depth_score / max_possible if max_possible > 0 else 0.0
        return round(normalized, 4)

    def check_recall_trigger(self) -> bool:
        """Recall@K < threshold → 재조정 필요"""
        return self.recall_tracker.needs_recalibration

    def enhance_gate_result(
        self,
        gate_result: NewsGateResult,
        df: pd.DataFrame,
        idx: int,
    ) -> RttpEnhancement:
        """
        기존 A/B/C 결과에 RTTP 점수를 가산.

        Returns:
            RttpEnhancement: 추가 RTTP 점수 정보
        """
        news_items = gate_result.news_items or []

        source_score = self.calc_source_weighted_score(news_items)
        engagement = self.calc_engagement_depth(df, idx, news_items)

        # 소스 최고 등급
        best_tier = SourceTier.COMMUNITY.value
        if news_items:
            tiers = [self._classify_source(n.source) for n in news_items]
            tier_values = {t: self.source_weights.get(t, 0.0) for t in tiers}
            if tier_values:
                best_tier = max(tier_values, key=tier_values.get)

        # RTTP 종합 부스트: 소스 가중 점수와 인게이지먼트 조합
        rttp_boost = 0.0
        if source_score >= 0.7:
            rttp_boost += 0.05
        elif source_score >= 0.5:
            rttp_boost += 0.03
        if engagement >= 0.5:
            rttp_boost += 0.03
        elif engagement >= 0.3:
            rttp_boost += 0.01

        return RttpEnhancement(
            source_weighted_score=source_score,
            engagement_depth=engagement,
            rttp_boost=round(rttp_boost, 3),
            source_tier=best_tier,
        )

    def _classify_source(self, source: str) -> str:
        """뉴스 소스 문자열 → SourceTier 값"""
        source_lower = source.lower() if source else ""

        if any(kw in source_lower for kw in ["dart", "공시", "거래소", "금감원"]):
            return SourceTier.DART.value
        if any(kw in source_lower for kw in ["bloomberg", "reuters", "ap"]):
            return SourceTier.BLOOMBERG.value
        if any(kw in source_lower for kw in ["증권", "리포트", "애널리스트", "투자의견"]):
            return SourceTier.BROKERAGE.value
        if any(kw in source_lower for kw in ["조선", "중앙", "한경", "매경", "서울경제"]):
            return SourceTier.NEWSPAPER.value
        if any(kw in source_lower for kw in ["네이버", "다음", "구글"]):
            return SourceTier.PORTAL.value

        return SourceTier.COMMUNITY.value
