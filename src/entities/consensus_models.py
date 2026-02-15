"""
v5.0 Sci-CoE 합의 기반 엔티티

Sci-CoE 논문(2026.02.13)의 핵심 개념을 퀀텀전략에 적용:
  - LayerVote: 각 파이프라인 레이어의 투표 결과
  - ConsensusResult: 3축 기하학적 합의 판정 결과
  - AnchorCase/AnchorDatabase: 앵커 학습용 사례 저장

클린 아키텍처: 외부 의존 없는 순수 엔티티.
"""

from dataclasses import dataclass, field


@dataclass
class LayerVote:
    """파이프라인 레이어의 투표 결과"""
    layer_name: str          # "L0_grade", "L1_regime" 등
    passed: bool             # 통과 여부
    confidence: float = 0.0  # 0~1 신뢰도
    features: dict = field(default_factory=dict)  # 추가 특성 벡터


@dataclass
class ConsensusResult:
    """3축 기하학적 합의 판정 결과"""
    consistency: float = 0.0      # 합의성 (레이어 통과율)
    reliability: float = 0.0      # 신뢰성 (앵커 중심 거리)
    diversity: float = 0.0        # 다양성 (지표 각도 분산)
    geometric_reward: float = 0.0  # C * R * D^(1/3) 최종 점수
    consensus_grade: str = "reject"  # strong/moderate/weak/reject
    votes: list = field(default_factory=list)
    passed_voters: int = 0
    total_voters: int = 0


@dataclass
class AnchorCase:
    """확인된 매매 사례 (앵커 학습용)"""
    ticker: str
    date: str
    outcome: str          # "success" / "failure"
    pnl_pct: float
    trigger_type: str
    grade: str
    features: dict = field(default_factory=dict)  # 레이어별 confidence 등


@dataclass
class AnchorDatabase:
    """앵커 사례 데이터베이스"""
    cases: list = field(default_factory=list)           # List[AnchorCase]
    success_centroid: dict = field(default_factory=dict)  # 성공 사례 중심점
    failure_centroid: dict = field(default_factory=dict)  # 실패 사례 중심점
