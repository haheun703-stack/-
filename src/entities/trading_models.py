"""v4.0 라이브 트레이딩 엔티티 — 주문, 포지션, 안전장치, 일일성과, 적응형 청산"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ─── 열거형 ────────────────────────────────────────────

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    PCT_STOP = "pct_stop"
    PARTIAL_2R = "partial_2R"
    PARTIAL_4R = "partial_4R"
    PARTIAL_8R = "partial_8R"
    PARTIAL_10R = "partial_10R"
    TRAILING_STOP = "trailing_stop"
    MAX_HOLD = "max_hold_days"
    TREND_EXIT = "trend_exit"
    MANUAL = "manual"
    EMERGENCY = "emergency"


# ─── 주문 ────────────────────────────────────────────

@dataclass
class Order:
    """한투 API 주문 결과"""
    order_id: str = ""          # 한투 주문번호 (ODNO)
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    price: int = 0              # 지정가 (시장가는 0)
    quantity: int = 0
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    org_no: str = ""            # 한투 조직코드 (정정/취소용 KRX_FWDG_ORD_ORGNO)
    message: str = ""           # API 응답 메시지


# ─── 라이브 포지션 ────────────────────────────────────

@dataclass
class LivePosition:
    """실전 매매 보유 포지션"""
    ticker: str = ""
    name: str = ""
    entry_date: str = ""
    entry_price: float = 0.0
    shares: int = 0
    current_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    atr_value: float = 0.0
    grade: str = "C"
    trigger_type: str = "confirm"   # impulse / confirm / breakout
    stop_loss_pct: float = 0.05
    highest_price: float = 0.0
    trailing_stop: float = 0.0
    partial_exits_done: int = 0     # 0~4 (4단계 부분청산)
    initial_shares: int = 0
    news_grade: str = ""            # v3.1 뉴스 등급
    max_hold_days: int = 10

    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익 (원)"""
        if self.entry_price <= 0 or self.shares <= 0:
            return 0.0
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """미실현 수익률 (%)"""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price / self.entry_price - 1) * 100

    @property
    def investment(self) -> float:
        """투입 금액"""
        return self.entry_price * self.shares

    def to_dict(self) -> dict:
        """JSON 직렬화용"""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "atr_value": self.atr_value,
            "grade": self.grade,
            "trigger_type": self.trigger_type,
            "stop_loss_pct": self.stop_loss_pct,
            "highest_price": self.highest_price,
            "trailing_stop": self.trailing_stop,
            "partial_exits_done": self.partial_exits_done,
            "initial_shares": self.initial_shares,
            "news_grade": self.news_grade,
            "max_hold_days": self.max_hold_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LivePosition:
        """JSON 역직렬화"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── 안전장치 상태 ────────────────────────────────────

@dataclass
class SafetyState:
    """안전장치 현재 상태"""
    stop_signal_active: bool = False
    daily_loss_pct: float = 0.0
    max_daily_loss_pct: float = -0.03       # -3%
    total_loss_pct: float = 0.0
    max_total_loss_pct: float = -0.10       # -10%
    emergency_triggered: bool = False
    is_holiday: bool = False

    @property
    def can_trade(self) -> bool:
        """매매 가능 여부"""
        if self.stop_signal_active:
            return False
        if self.emergency_triggered:
            return False
        if self.is_holiday:
            return False
        if self.daily_loss_pct <= self.max_daily_loss_pct:
            return False
        return True

    @property
    def must_liquidate(self) -> bool:
        """긴급 전량 청산 필요 여부"""
        return self.total_loss_pct <= self.max_total_loss_pct


# ─── 일일 성과 ────────────────────────────────────────

@dataclass
class DailyPerformance:
    """일일 매매 성과"""
    date: str = ""
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_executed: int = 0
    win_trades: int = 0
    loss_trades: int = 0

    @property
    def daily_return_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.ending_balance / self.starting_balance - 1) * 100

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "trades_executed": self.trades_executed,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "daily_return_pct": self.daily_return_pct,
        }


# ─── v4.1 적응형 청산 엔티티 ────────────────────────────

class PullbackClassification(Enum):
    """조정 분류"""
    HEALTHY = "healthy"       # 건강한 조정 → 손절 완화
    CAUTION = "caution"       # 주의 → 현행 유지
    DANGEROUS = "dangerous"   # 위험 → 즉시 청산
    CRITICAL = "critical"     # 긴급 → 시장가 청산


class HoldAction(Enum):
    """일일 보유 판단"""
    STRONG_HOLD = "strong_hold"   # 강력 보유 → 손절 완화 + 보유일 연장
    HOLD = "hold"                 # 보유 → 현행 유지
    TIGHTEN = "tighten"           # 경계 → 손절 강화
    EXIT = "exit"                 # 청산 → 다음 봉 매도


@dataclass
class PullbackHealth:
    """조정 건강도 판정 결과"""
    health_score: float = 50.0     # 0~100 (높을수록 건강)
    classification: str = "caution"  # PullbackClassification.value
    # ── 개별 지표 ──
    ma20_support: bool = False      # 종가 > MA20 (또는 MA20 근처 지지)
    ma60_support: bool = False      # 종가 > MA60
    adx_value: float = 0.0         # ADX 값
    adx_trending: bool = False     # ADX >= 25 (추세 유지)
    di_positive: bool = False      # +DI > -DI (방향성 유지)
    obv_intact: bool = False       # OBV 기울기 >= 0 (매집 유지)
    volume_declining: bool = False  # 조정 중 거래량 감소 (건강한 신호)
    inst_not_selling: bool = False  # 기관 순매도 아님
    foreign_not_selling: bool = False  # 외국인 순매도 아님
    rsi_value: float = 50.0        # RSI 값
    rsi_not_oversold: bool = True  # RSI > 30
    # ── 조정 컨텍스트 ──
    pullback_pct: float = 0.0     # 고점 대비 하락률
    pullback_days: int = 0        # 조정 지속일
    reasons: list = field(default_factory=list)
    # ── 적응형 손절 ──
    adjusted_stop_pct: float = 0.03  # 조정된 손절 퍼센트
    adjusted_stop_price: float = 0.0  # 조정된 손절가

    def to_dict(self) -> dict:
        return {
            "health_score": self.health_score,
            "classification": self.classification,
            "ma20_support": self.ma20_support,
            "ma60_support": self.ma60_support,
            "adx_value": self.adx_value,
            "adx_trending": self.adx_trending,
            "di_positive": self.di_positive,
            "obv_intact": self.obv_intact,
            "volume_declining": self.volume_declining,
            "inst_not_selling": self.inst_not_selling,
            "foreign_not_selling": self.foreign_not_selling,
            "pullback_pct": self.pullback_pct,
            "pullback_days": self.pullback_days,
            "reasons": self.reasons,
            "adjusted_stop_pct": self.adjusted_stop_pct,
        }


@dataclass
class HoldScore:
    """일일 보유 점수 (매일 포지션 건강 평가)"""
    total_score: float = 50.0       # 0~100
    action: str = "hold"            # HoldAction.value
    # ── 세부 스코어 ──
    technical_score: float = 0.0    # 0~40 (MA정렬, ADX, RSI, MACD)
    supply_demand_score: float = 0.0  # 0~30 (OBV, 거래량, 수급)
    news_issue_score: float = 0.0   # 0~15 (뉴스, 살아있는 이슈)
    position_health_score: float = 0.0  # 0~15 (PnL 모멘텀, 조정 건강도)
    # ── 판단 근거 ──
    reasons: list = field(default_factory=list)
    # ── 적응형 파라미터 조정 ──
    stop_adjustment: float = 1.0    # 손절 배율 (0.5=강화, 1.0=유지, 2.0=완화)
    hold_days_adjustment: int = 0   # 보유일 조정 (-5 ~ +10)
    trailing_tightness: float = 1.0  # 트레일링 배율 (0.5=타이트, 2.0=넓게)

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "action": self.action,
            "technical_score": self.technical_score,
            "supply_demand_score": self.supply_demand_score,
            "news_issue_score": self.news_issue_score,
            "position_health_score": self.position_health_score,
            "reasons": self.reasons,
            "stop_adjustment": self.stop_adjustment,
            "hold_days_adjustment": self.hold_days_adjustment,
            "trailing_tightness": self.trailing_tightness,
        }
