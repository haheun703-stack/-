"""
v6.0 RTTP 뉴스 신호 엔터티 모델

RTTP 논문 (Hui et al., WWW '26) 기반:
- 소스 권위 가중치 (DART > Bloomberg > 증권사 > 포털 > 커뮤니티)
- 인게이지먼트 깊이 5단계 (L5 기관 > L4 거래량 > L3 리포트 > L2 검색 > L1 SNS)
- LLM 합성 수혜주 (1차/2차/3차 + 역수혜주)
- Recall@K 모니터링
"""

from dataclasses import dataclass, field
from enum import Enum


class SourceTier(Enum):
    """뉴스 소스 등급"""
    DART = "DART"              # 공시 시스템 (권위 1.0)
    BLOOMBERG = "BLOOMBERG"    # 해외 통신사 (0.95)
    BROKERAGE = "BROKERAGE"    # 증권사 리포트 (0.80)
    NEWSPAPER = "NEWSPAPER"    # 주요 신문사 (0.70)
    PORTAL = "PORTAL"          # 포털 뉴스 (0.50)
    COMMUNITY = "COMMUNITY"    # 커뮤니티/SNS (0.15)


# 소스별 권위 가중치 기본값
DEFAULT_SOURCE_WEIGHTS = {
    SourceTier.DART.value: 1.0,
    SourceTier.BLOOMBERG.value: 0.95,
    SourceTier.BROKERAGE.value: 0.80,
    SourceTier.NEWSPAPER.value: 0.70,
    SourceTier.PORTAL.value: 0.50,
    SourceTier.COMMUNITY.value: 0.15,
}

# 인게이지먼트 단계별 가중치 기본값
DEFAULT_ENGAGEMENT_WEIGHTS = {
    "L5_inst": 5.0,       # 기관 순매수 (가장 높은 신뢰)
    "L4_volume": 3.5,     # 거래량 급증
    "L3_report": 2.0,     # 증권사 리포트
    "L2_search": 1.0,     # 검색량 증가
    "L1_sns": 0.3,        # SNS/커뮤니티
}


@dataclass
class BeneficiarySignal:
    """LLM 합성 수혜주 신호"""
    ticker: str = ""
    name: str = ""
    order: int = 1               # 1차/2차/3차 수혜주
    reason: str = ""
    confidence: float = 0.0      # 0~1
    is_inverse: bool = False     # True = 역수혜주 (하락 예상)


@dataclass
class RecallTracker:
    """Recall@K 모니터링 (예측 정확도 추적)"""
    total_signals: int = 0
    correct_signals: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    needs_recalibration: bool = False
    window_size: int = 100

    def update(self, was_correct: bool) -> None:
        self.total_signals += 1
        if was_correct:
            self.correct_signals += 1
        if self.total_signals >= self.window_size:
            self.recall_at_5 = self.correct_signals / self.total_signals
            self.needs_recalibration = self.recall_at_5 < 0.5


@dataclass
class RttpEnhancement:
    """RTTP 뉴스 강화 결과 — NewsGateResult에 추가되는 정보"""
    source_weighted_score: float = 0.0     # 소스 권위 가중 평균
    engagement_depth: float = 0.0          # 인게이지먼트 깊이 점수
    beneficiaries: list = field(default_factory=list)  # BeneficiarySignal 리스트
    rttp_boost: float = 0.0                # RTTP 종합 부스트 점수
    source_tier: str = "COMMUNITY"         # 가장 높은 소스 등급
