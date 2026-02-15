"""
v6.0 극한 변동성 엔터티 모델

Alparslan & Kim(2021) 논문 기반:
- 극한 변동성 탐지: ATR ratio > 3.0, vol ratio > 5.0, daily range > 10%
- Capitulation 프로파일: 연속 하락 + 거래량 클라이맥스 + RSI 극단
- 방향 판별: bullish_breakout / bearish_breakdown / capitulation / ambiguous
"""

from dataclasses import dataclass


@dataclass
class ExtremeVolatilityResult:
    """극한 변동성 탐지 결과"""
    is_extreme: bool = False
    atr_ratio: float = 0.0           # 현재 ATR / 60일 평균 ATR
    vol_ratio: float = 0.0           # 현재 거래량 / 60일 평균 거래량
    daily_range_pct: float = 0.0     # (고가 - 저가) / 종가 * 100
    direction: str = "neutral"       # bullish_breakout / bearish_breakdown / capitulation / ambiguous
    is_capitulation: bool = False
    confidence: float = 0.0          # 0~1


@dataclass
class CapitulationProfile:
    """투매(Capitulation) 프로파일"""
    consecutive_down_days: int = 0   # 연속 하락 일수
    volume_climax: bool = False      # 거래량 클라이맥스 (3x 이상)
    rsi_extreme_low: bool = False    # RSI < 20
    reversal_candle: bool = False    # 양봉 반전 캔들
    score: float = 0.0               # 종합 점수 (0~100)
