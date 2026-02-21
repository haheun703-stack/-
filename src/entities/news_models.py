"""v3.1 뉴스 도메인 엔티티 — 외부 의존 없는 순수 데이터 객체"""

from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# Enums
# ============================================================

class NewsGrade(Enum):
    """뉴스 등급 (v3.1 News Gate)"""
    A = "A"  # 확정 공시 → 파라미터 조정 후 즉시 필터
    B = "B"  # 신뢰도 높은 루머 → 관찰 리스트
    C = "C"  # 추측/테마 → 무시


class NewsCategory(Enum):
    """뉴스 카테고리"""
    MA_ACQUISITION = "ma_acquisition"          # M&A/매각
    EARNINGS_SURPRISE = "earnings_surprise"    # 실적 서프라이즈
    CONTRACT_WIN = "contract_win"              # 대규모 수주
    POLICY_CHANGE = "policy_change"            # 정책 변경
    SHAREHOLDER_ACTION = "shareholder_action"  # 지분 변동
    GENERAL_POSITIVE = "general_positive"      # 일반 호재
    GENERAL_NEGATIVE = "general_negative"      # 일반 악재
    THEME = "theme"                            # 테마/루머


class EventDrivenAction(Enum):
    """이벤트 드리븐 행동 지침"""
    ENTER = "enter"          # 즉시 진입 가능
    WATCHLIST = "watchlist"  # 관찰 리스트 등록
    IGNORE = "ignore"        # 무시


# ============================================================
# 뉴스 관련 데이터클래스
# ============================================================

@dataclass
class NewsItem:
    """개별 뉴스 항목"""
    title: str
    summary: str
    category: str = "theme"       # NewsCategory.value
    source: str = ""
    date: str = ""
    impact_score: int = 0         # 1-10
    sentiment: str = "neutral"    # positive / negative / neutral
    is_confirmed: bool = False    # DART 공시 확인 여부
    has_specific_amount: bool = False   # 구체적 금액 언급
    has_definitive_language: bool = False  # 확정적 표현 ("선정","체결")
    cross_verified: bool = False  # 복수 언론 교차 확인
    # v6.0 RTTP 추가 필드
    source_authority: float = 0.0      # 소스 권위 가중치 (0~1)
    engagement_level: int = 0          # 인게이지먼트 단계 (1~5)
    beneficiary_order: int = 0         # 수혜주 차수 (0=해당없음, 1=1차, 2=2차, 3=3차)


@dataclass
class NewsGateResult:
    """L-1 News Gate 판정 결과"""
    grade: NewsGrade = NewsGrade.C
    action: EventDrivenAction = EventDrivenAction.IGNORE
    ticker: str = ""
    news_items: list = field(default_factory=list)
    param_overrides: dict = field(default_factory=dict)
    watchlist_days: int = 0
    timestamp: str = ""
    reason: str = ""
    # v3.2: 뉴스 스코어 부스트 (Zone Score 가산 + Trend Cont 연동)
    score_boost: float = 0.0
    living_issues: list = field(default_factory=list)   # LivingIssue 리스트
    earnings_estimate: "EarningsEstimate | None" = None
    # v6.0 RTTP 추가 필드
    source_weighted_score: float = 0.0   # 소스 권위 가중 평균
    engagement_depth: float = 0.0        # 인게이지먼트 깊이 (0~1)
    beneficiaries: list = field(default_factory=list)  # BeneficiarySignal 리스트


# ============================================================
# v3.2 살아있는 이슈 + 실적 예상
# ============================================================

@dataclass
class LivingIssue:
    """현재 진행 중인 이슈 (과거에 시작됐지만 아직 유효한 재료)"""
    title: str                        # 이슈 제목
    category: str = "theme"           # NewsCategory.value
    start_date: str = ""              # 이슈 시작일
    status: str = "active"            # active / resolved / fading
    impact_score: int = 0             # 1~10 (현재 시점 영향력)
    sentiment: str = "positive"       # positive / negative / neutral
    description: str = ""             # 상세 설명
    expected_resolution: str = ""     # 예상 해소 시점/조건
    source_count: int = 0             # 관련 뉴스 건수


@dataclass
class EarningsEstimate:
    """실적 예상 정보"""
    next_earnings_date: str = ""      # 다음 실적 발표일 (예: "2026-03-15")
    days_until_earnings: int = -1     # 실적 발표까지 남은 일수 (-1=미정)
    consensus_revenue: float = 0.0    # 매출 컨센서스 (억원)
    consensus_op: float = 0.0         # 영업이익 컨센서스 (억원)
    consensus_eps: float = 0.0        # EPS 컨센서스 (원)
    surprise_direction: str = "neutral"  # beat / miss / in_line / neutral
    yoy_growth_pct: float = 0.0       # 전년 동기 대비 증감률 (%)
    analyst_count: int = 0            # 추정 애널리스트 수
    pre_earnings_accumulation: bool = False  # 실적 전 매집 패턴 감지
    description: str = ""             # 요약 설명


# ============================================================
# 이벤트 드리븐 포지션
# ============================================================

@dataclass
class EventPosition:
    """이벤트 드리븐 포지션 관리 데이터"""
    ticker: str = ""
    news_grade: str = "C"         # NewsGrade.value
    entry_price: float = 0.0
    target_1: float = 0.0         # 반익 목표
    target_2: float = 0.0         # 전량 청산 목표
    stop_loss: float = 0.0
    position_pct: float = 0.0     # 포지션 비중 (%)
    max_hold_days: int = 3
    days_held: int = 0
    entry_date: str = ""
    gap_up_pct: float = 0.0


# ============================================================
# 스마트머니 매집 관련
# ============================================================

@dataclass
class AccumulationSignal:
    """매집 단계 감지 결과"""
    phase: str = "none"           # none/phase1/phase2/phase3/dumping
    phase_name: str = "매집 미감지"
    confidence: float = 0.0       # 0~100
    reasons: list = field(default_factory=list)
    inst_streak: int = 0          # 기관 연속 순매수 일수
    foreign_streak: int = 0       # 외국인 연속 순매수 일수
    flow_pattern: str = "unknown"  # both_buying/inst_only/foreign_only/both_selling
    score_modifier: int = 0       # SmartZ 점수 가감
    action: str = "neutral"       # watch/prepare/buy_ready/danger


@dataclass
class DivergenceSignal:
    """OBV/주가 다이버전스 결과"""
    type: str = "none"            # bullish / bearish / none
    strength: str = "weak"        # strong / moderate / weak
    confidence: float = 0.0
    price_trend: str = "unknown"  # falling / rising / flat
    obv_trend: str = "unknown"    # falling / rising / flat
    lookback_days: int = 30
    reason: str = ""


# ============================================================
# v3.2 시장 시그널 (골든크로스, 스마트머니 등)
# ============================================================

# ============================================================
# v11.0 테마 스캐너 (RSS + Grok 하이브리드)
# ============================================================

@dataclass
class ThemeStock:
    """테마 관련 종목"""
    ticker: str
    name: str
    order: int = 1              # 1=1차, 2=2차, 3=3차 수혜
    source: str = "dictionary"  # "dictionary" | "grok_expanded"
    pipeline_status: str = ""   # "PASS" | "FAIL_G2" | "NOT_IN_UNIVERSE" 등
    current_rsi: float = 0.0    # 참고용 기술지표
    ma20_dist_pct: float = 0.0  # MA20 대비 괴리율 (%)


@dataclass
class ThemeAlert:
    """RSS 테마 감지 알림"""
    theme_name: str              # "페로브스카이트"
    matched_keyword: str         # 매칭된 키워드
    news_title: str              # RSS 기사 제목
    news_url: str = ""           # 기사 URL
    news_source: str = ""        # RSS 피드명
    published: str = ""          # 발행 시각
    related_stocks: list = field(default_factory=list)  # ThemeStock 리스트
    grok_expanded: bool = False  # Grok 확장 완료 여부
    timestamp: str = ""


class SignalImportance(Enum):
    """시그널 중요도"""
    CRITICAL = "critical"    # 즉시 확인 필요
    HIGH = "high"            # 중요
    MEDIUM = "medium"        # 참고
    LOW = "low"              # 정보


@dataclass
class MarketSignal:
    """시장 시그널 (알림용)"""
    signal_type: str              # golden_cross_imminent, smart_money_optimal, etc.
    title: str                    # "골든크로스 임박!"
    description: str              # 상세 설명
    importance: str = "medium"    # SignalImportance.value
    confidence: float = 0.0       # 0~100
    data: dict = field(default_factory=dict)  # 지원 데이터
