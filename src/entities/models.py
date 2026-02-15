"""핵심 비즈니스 엔티티 - 외부 의존 없는 순수 데이터 객체"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

# ─── 기본 열거형 ────────────────────────────────────────────

class Market(Enum):
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"


class Direction(Enum):
    UP = "상승"
    DOWN = "하락"
    SIDEWAYS = "횡보"


class ConditionType(Enum):
    HOLD = "유지"       # ✅ 유지해도 되는 조건
    ACTION = "대응"     # 🚨 반드시 대응해야 할 조건


class AlertType(Enum):
    FOREIGN_BIG_BUY = "외국인 대량 매수"
    FOREIGN_BIG_SELL = "외국인 대량 매도"
    INST_BIG_BUY = "기관 대량 매수"
    INST_BIG_SELL = "기관 대량 매도"
    VOLUME_SPIKE = "거래량 급증"
    PRICE_BREAKOUT = "가격 돌파"
    TREND_REVERSAL = "추세 전환"


class SignalStrength(Enum):
    STRONG = "강함"
    MODERATE = "보통"
    WEAK = "약함"


class MovingAverageAlignment(Enum):
    BULLISH = "정배열"      # 5 > 20 > 60 > 120
    BEARISH = "역배열"      # 5 < 20 < 60 < 120
    MIXED = "혼조"


# ─── 핵심 엔티티 ────────────────────────────────────────────

@dataclass(frozen=True)
class Stock:
    """종목 기본 정보"""
    ticker: str          # 종목코드 (예: "005930")
    name: str            # 종목명 (예: "삼성전자")
    market: Market       # 시장 (KOSPI/KOSDAQ)
    sector: str          # 업종 (예: "반도체")


@dataclass
class OHLCV:
    """일봉 데이터 한 줄"""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TechnicalIndicators:
    """기술적 지표 값"""
    rsi: float | None = None                    # RSI (0~100)
    macd: float | None = None                   # MACD 값
    macd_signal: float | None = None            # MACD 시그널
    macd_histogram: float | None = None         # MACD 히스토그램
    bollinger_upper: float | None = None        # 볼린저 상단
    bollinger_middle: float | None = None       # 볼린저 중단
    bollinger_lower: float | None = None        # 볼린저 하단
    stochastic_k: float | None = None           # 스토캐스틱 %K
    stochastic_d: float | None = None           # 스토캐스틱 %D
    ma5: float | None = None                    # 이동평균 5일
    ma20: float | None = None                   # 이동평균 20일
    ma60: float | None = None                   # 이동평균 60일
    ma120: float | None = None                  # 이동평균 120일


@dataclass
class ChartData:
    """차트 데이터 (OHLCV + 기술적 지표)"""
    candles: list[OHLCV] = field(default_factory=list)
    indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)

    @property
    def latest(self) -> OHLCV | None:
        return self.candles[-1] if self.candles else None

    @property
    def latest_close(self) -> float | None:
        return self.latest.close if self.latest else None


@dataclass
class TechnicalPattern:
    """기술적 분석 결과"""
    candle_pattern: str                              # 캔들 패턴 (예: "망치형", "도지")
    ma_alignment: MovingAverageAlignment             # 이평선 배열
    rsi_signal: str                                  # RSI 해석 (예: "과매수 구간")
    macd_signal: str                                 # MACD 해석 (예: "골든크로스 임박")
    bollinger_signal: str                            # 볼린저 해석
    stochastic_signal: str                           # 스토캐스틱 해석
    overall_trend: Direction                         # 종합 추세
    strength: SignalStrength                         # 신호 강도
    key_points: list[str] = field(default_factory=list)  # 핵심 분석 포인트


@dataclass
class SupplyDemandZone:
    """매물대 (지지/저항 구간)"""
    price_low: float                # 구간 하한
    price_high: float               # 구간 상한
    zone_type: str                  # "지지" 또는 "저항"
    strength: SignalStrength        # 강도
    volume_ratio: float             # 거래량 비율 (평균 대비)
    description: str = ""           # 설명


@dataclass
class VolumeAnalysis:
    """거래량 분석 결과"""
    avg_volume_ratio: float                          # 평균 거래량 대비 비율
    volume_trend: Direction                          # 거래량 추세
    accumulation_signal: str                         # 매집/분산 신호
    zones: list[SupplyDemandZone] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)


@dataclass
class FlowPrediction:
    """내일 흐름 예측"""
    direction: Direction            # 예상 방향
    confidence: float               # 확신도 (0.0 ~ 1.0)
    price_low: float                # 예상 가격 하한
    price_high: float               # 예상 가격 상한
    key_factors: list[str] = field(default_factory=list)   # 핵심 요인
    summary: str = ""               # 요약


@dataclass
class Condition:
    """유지/대응 조건"""
    condition_type: ConditionType   # 유지 or 대응
    title: str                      # 조건 제목
    description: str                # 상세 설명
    trigger_price: float | None     # 트리거 가격 (해당 시)
    priority: int                   # 우선순위 (1이 가장 높음)
    confidence: float               # 확신도 (0.0 ~ 1.0)


@dataclass
class AnalysisReport:
    """최종 분석 리포트 (모든 분석 결과 종합)"""
    stock: Stock
    chart_data: ChartData
    technical_pattern: TechnicalPattern
    volume_analysis: VolumeAnalysis
    flow_prediction: FlowPrediction
    conditions: list[Condition] = field(default_factory=list)
    investor_flow: InvestorFlow | None = None
    score: AnalysisScore | None = None
    analyzed_at: datetime = field(default_factory=datetime.now)

    @property
    def hold_conditions(self) -> list[Condition]:
        """✅ 유지 조건만 필터"""
        return [c for c in self.conditions if c.condition_type == ConditionType.HOLD]

    @property
    def action_conditions(self) -> list[Condition]:
        """🚨 대응 조건만 필터"""
        return [c for c in self.conditions if c.condition_type == ConditionType.ACTION]


# ─── 수급 모니터링 엔티티 ─────────────────────────────────────

@dataclass
class InvestorFlow:
    """투자자별 수급 데이터 (외국인/기관/개인)"""
    ticker: str
    date: date
    foreign_buy: int = 0        # 외국인 매수량
    foreign_sell: int = 0       # 외국인 매도량
    foreign_net: int = 0        # 외국인 순매수
    inst_buy: int = 0           # 기관 매수량
    inst_sell: int = 0          # 기관 매도량
    inst_net: int = 0           # 기관 순매수
    individual_net: int = 0     # 개인 순매수
    total_shares: int = 0       # 총 발행주식수
    foreign_holding_qty: int = 0  # 외국인 보유주식수
    foreign_holding_ratio: float | None = None  # 외국인 보유 비율 (%)


@dataclass
class FlowAlert:
    """수급 이상 감지 알림"""
    alert_type: AlertType
    ticker: str
    message: str
    severity: SignalStrength     # 강도
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)


# ─── AI 스코어링 엔티티 ───────────────────────────────────────

@dataclass
class ScoreDetail:
    """점수 세부항목"""
    name: str           # 항목명 (예: "추세 분석")
    score: float        # 획득 점수
    max_score: float    # 만점
    comment: str = ""   # 코멘트


@dataclass
class ScoreCategory:
    """분석 카테고리별 점수"""
    name: str                           # 카테고리명 (예: "기술적 분석")
    score: float                        # 획득 점수
    max_score: float                    # 만점 (예: 35)
    details: list[ScoreDetail] = field(default_factory=list)


@dataclass
class AnalysisScore:
    """100점 종합 분석 점수"""
    categories: list[ScoreCategory] = field(default_factory=list)
    summary: str = ""                   # 종합 평가 요약
    recommendation: str = ""            # 투자 권고

    @property
    def total_score(self) -> float:
        return sum(c.score for c in self.categories)

    @property
    def max_score(self) -> float:
        return sum(c.max_score for c in self.categories)

    @property
    def grade(self) -> str:
        s = self.total_score
        if s >= 80:
            return "A (적극 매수)"
        elif s >= 65:
            return "B (매수 고려)"
        elif s >= 50:
            return "C (중립/관망)"
        elif s >= 35:
            return "D (매도 고려)"
        else:
            return "F (적극 매도)"
