"""
v6.0 Martin Momentum 엔터티 모델

Martin(2023) 논문 기반:
- EMA2 필터 (fast EMA - slow EMA)
- Dead Zone (약한 신호 무시)
- Sigmoid 활성화 함수
- 변동성 정규화 포지션 (1/σ)
- 최적 보유기간 공식
"""

from dataclasses import dataclass


@dataclass
class MartinMomentumResult:
    """Martin 모멘텀 평가 결과"""

    ema2_value: float = 0.0          # fast EMA - slow EMA (원시)
    ema2_normalized: float = 0.0     # EMA2 / close * 100 (정규화)
    in_dead_zone: bool = False       # |EMA2_norm| < epsilon → 신호 무시
    trend_strength: float = 0.0      # sigmoid 활성화 (0~1) — 추세 강도
    reversal_strength: float = 0.0   # 역전 sigmoid (0~1) — 역전 강도
    signal_type: str = "neutral"     # trend / reversal / dead_zone / neutral
    optimal_hold_days: int = 10      # 1.7 * (N_fast + N_slow)
    vol_normalized_weight: float = 1.0  # target_sigma / realized_sigma (포지션 비중)
    confidence: float = 0.0          # 종합 신뢰도 (0~1)
