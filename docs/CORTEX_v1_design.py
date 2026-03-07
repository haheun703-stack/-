"""
╔══════════════════════════════════════════════════════════════════╗
║  JARVIS CORTEX v1.0 — Context-Oriented Real-Time EXecution     ║
║  자율판단 트레이딩 엔진                                           ║
║                                                                  ║
║  NIGHTWATCH(매크로) + v10.3(종목선별) 사이를 연결하는 "두뇌"       ║
║                                                                  ║
║  구조:                                                           ║
║  ┌─────────────────────────────────────────────────────┐        ║
║  │  Layer 1: 시장 체제 인식 (Market Regime Detector)     │        ║
║  │  Layer 2: 충격 분류기 (Shock Classifier)              │        ║
║  │  Layer 3: 기회 스코어링 (Opportunity Scorer)          │        ║
║  │  Layer 4: 자율 실행 엔진 (Autonomous Executor)        │        ║
║  └─────────────────────────────────────────────────────┘        ║
║                                                                  ║
║  연동: NIGHTWATCH(17:00) → CORTEX(장중5분) → v10.3(매매)        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import json
import requests
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# ============================================================
#  설정 (CONFIG)
# ============================================================

class Config:
    """CORTEX 전체 설정 — 여기서 모든 파라미터 조정"""
    
    # --- DB ---
    DB_PATH = "jarvis_control_tower.db"          # Control Tower DB
    
    # --- 텔레그램 ---
    TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
    
    # --- 체제 판단 임계값 (Layer 1) ---
    REGIME_PANIC_THRESHOLD = -0.07       # 5일 수익률 -7% 이하 → PANIC
    REGIME_SHOCK_THRESHOLD = -0.03       # 당일 -3% 이하 → SHOCK
    REGIME_VIX_HIGH = 25                 # VIX 25 이상 → 경계
    REGIME_RECOVERY_VOL_MULT = 1.0       # 거래량 20일평균 대비 배수
    
    # --- 충격 분류 임계값 (Layer 2) ---
    GDELT_SPIKE_THRESHOLD = 2.0          # GDELT 평소 대비 2배 → 지정학
    OIL_SPIKE_THRESHOLD = 0.05           # 유가 5% 급등 → 에너지 충격
    RATE_SPIKE_THRESHOLD = 0.10          # 금리 10bp 급등 → 금리 충격
    
    # --- 기회 스코어링 (Layer 3) ---
    SCORE_DRAWDOWN_WEIGHT = 30           # 낙폭 점수 비중
    SCORE_FUNDAMENTAL_WEIGHT = 25        # 펀더멘탈 점수 비중
    SCORE_UNRELATED_WEIGHT = 25          # 충격 무관 점수 비중
    SCORE_RECOVERY_WEIGHT = 20           # 회복 신호 점수 비중
    SCORE_MIN_THRESHOLD = 60             # 최소 매수 기준 점수
    
    DRAWDOWN_MIN = -0.10                 # 최소 -10% 낙폭부터 관심
    RSI_STRONG_FLOOR = 55               # 크래시 전 RSI 이 이상이면 "강한 종목"
    
    # --- 실행 엔진 (Layer 4) ---
    CORTEX_CAPITAL_RATIO = 0.15          # 전체 자본 중 CORTEX 슬롯 비율 (15%)
    MAX_SINGLE_POSITION = 0.03           # 종목당 최대 비중 (3%)
    MAX_POSITIONS = 10                   # CORTEX 최대 동시 보유 종목 수
    STOP_LOSS = -0.05                    # 손절 라인 -5% (하드코딩)
    TAKE_PROFIT = 0.08                   # 익절 라인 +8%
    
    # --- 실행 모드 ---
    # "REPORT"   = 텔레그램 보고만 (Phase 1)
    # "APPROVE"  = 텔레그램 보고 + 승인 시 매수 (Phase 1.5)  
    # "AUTO"     = 자율 매수 + 사후 보고 (Phase 2)
    EXECUTION_MODE = "REPORT"
    
    # --- 기준 기간 ---
    REFERENCE_START = "2024-12-01"       # 기준 기간 시작
    REFERENCE_END = "2025-02-27"         # 기준 기간 끝 (충격 전)
    
    # --- 섹터 매핑 (JARVIS_SECTOR_MAP 기반) ---
    SECTOR_MAP = {
        "반도체": ["삼성전자", "SK하이닉스", "한미반도체", "리노공업", "이오테크닉스"],
        "소프트웨어": ["NAVER", "카카오", "더존비즈온", "한글과컴퓨터"],
        "바이오": ["삼성바이오로직스", "셀트리온", "유한양행", "알테오젠"],
        "전력에너지": ["HD현대일렉트릭", "LS일렉트릭", "효성중공업"],
        "방산": ["한화에어로스페이스", "현대로템", "LIG넥스원", "KAI", "풍산"],
        "정유": ["SK이노베이션", "S-Oil", "GS칼텍스"],
        "우주항공": ["한화시스템", "쎄트렉아이", "AP위성"],
        "엔터": ["하이브", "SM", "JYP", "YG"],
        "증권": ["키움증권", "미래에셋증권", "삼성증권"],
    }
    
    # 충격 유형별 직접 피해 섹터
    SHOCK_AFFECTED_SECTORS = {
        "GEOPOLITICAL": ["방산", "정유"],       # 수혜 섹터 (매수X, 이미 올랐음)
        "RATE":         ["바이오", "소프트웨어"],  # 금리에 민감
        "EARNINGS":     [],                      # 개별 종목 문제
        "LIQUIDITY":    [],                      # 전체 하락
    }


# ============================================================
#  Layer 1: 시장 체제 인식 (Market Regime Detector)
# ============================================================

class Regime(Enum):
    """시장 체제 — 봇이 가장 먼저 파악해야 할 것"""
    NORMAL = "NORMAL"                    # 평시: v10.3 기본 로직
    CAUTION = "CAUTION"                  # 주의: 신규매수 축소
    SHOCK = "SHOCK"                      # 충격: 신규매수 중단
    PANIC = "PANIC"                      # 패닉: 현금 비중 확대
    RECOVERY_EARLY = "RECOVERY_EARLY"    # 회복 초기: ⭐ 줍줍 타이밍
    RECOVERY_CONFIRMED = "RECOVERY_CONFIRMED"  # 회복 확인: 공격적 매수


@dataclass
class RegimeState:
    """현재 체제 상태 + 전환 히스토리"""
    current: Regime = Regime.NORMAL
    previous: Regime = Regime.NORMAL
    changed_at: str = ""
    kospi_5d_return: float = 0.0
    kospi_today_return: float = 0.0
    vix: float = 0.0
    consecutive_down_days: int = 0
    panic_low: Optional[float] = None    # 패닉 구간 최저점 (회복 판단용)
    
    # 체제별 봇 행동 규칙
    RULES = {
        Regime.NORMAL:             {"new_buy": True,  "capital_use": 1.0,  "cortex_active": False},
        Regime.CAUTION:            {"new_buy": True,  "capital_use": 0.7,  "cortex_active": False},
        Regime.SHOCK:              {"new_buy": False, "capital_use": 0.5,  "cortex_active": False},
        Regime.PANIC:              {"new_buy": False, "capital_use": 0.3,  "cortex_active": False},
        Regime.RECOVERY_EARLY:     {"new_buy": True,  "capital_use": 0.5,  "cortex_active": True},
        Regime.RECOVERY_CONFIRMED: {"new_buy": True,  "capital_use": 0.8,  "cortex_active": True},
    }
    
    def get_rules(self) -> dict:
        return self.RULES[self.current]


class RegimeDetector:
    """
    시장 체제를 판단하는 엔진
    
    판단 흐름:
    KOSPI 5일 수익률, 당일 등락, VIX, 거래량, 연속 하락일 → 체제 결정
    
    전환 규칙:
    NORMAL → CAUTION: KOSPI -2% 단일일 or VIX > 20
    CAUTION → SHOCK: KOSPI -3% 단일일 and VIX > 25
    SHOCK → PANIC: KOSPI 5일 -7% 이하 or 3일 연속 -1.5%
    PANIC → RECOVERY_EARLY: 첫 양봉 + 거래량 증가 + 저점 이탈 없음
    RECOVERY_EARLY → RECOVERY_CONFIRMED: KOSPI > 5일선 + 외국인 순매수
    RECOVERY_CONFIRMED → NORMAL: 20일선 탈환 + VIX < 20
    
    ※ 상향 전환은 한 단계씩만 (안전장치)
    ※ 하향 전환은 건너뛰기 가능 (급락 시 NORMAL→PANIC 직행 가능)
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.state = RegimeState()
        self.logger = logging.getLogger("CORTEX.Regime")
    
    def detect(self, market_data: dict) -> RegimeState:
        """
        시장 데이터를 받아 체제를 판단한다.
        
        Args:
            market_data: {
                "kospi_close": float,           # 오늘 KOSPI 종가
                "kospi_prev_close": float,      # 어제 KOSPI 종가
                "kospi_5d_ago_close": float,    # 5일 전 KOSPI 종가
                "kospi_20ma": float,            # KOSPI 20일 이평선
                "kospi_5ma": float,             # KOSPI 5일 이평선
                "kospi_volume": float,          # 오늘 거래량
                "kospi_vol_20avg": float,       # 20일 평균 거래량
                "vix": float,                   # VIX 지수
                "foreign_net_buy": float,       # 외국인 순매수 (양수=매수)
                "consecutive_down": int,        # 연속 하락일 수
                "is_bullish_candle": bool,      # 오늘 양봉 여부
            }
        
        Returns:
            RegimeState: 현재 체제 상태
        """
        
        # 기본 지표 계산
        today_return = (
            (market_data["kospi_close"] - market_data["kospi_prev_close"]) 
            / market_data["kospi_prev_close"]
        )
        five_day_return = (
            (market_data["kospi_close"] - market_data["kospi_5d_ago_close"]) 
            / market_data["kospi_5d_ago_close"]
        )
        vol_ratio = (
            market_data["kospi_volume"] / market_data["kospi_vol_20avg"]
            if market_data["kospi_vol_20avg"] > 0 else 1.0
        )
        
        self.state.kospi_5d_return = five_day_return
        self.state.kospi_today_return = today_return
        self.state.vix = market_data["vix"]
        self.state.consecutive_down_days = market_data["consecutive_down"]
        
        prev_regime = self.state.current
        new_regime = self._determine_regime(
            today_return, five_day_return, market_data, vol_ratio
        )
        
        # 체제 전환 처리
        if new_regime != prev_regime:
            self.state.previous = prev_regime
            self.state.current = new_regime
            self.state.changed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.logger.info(
                f"🔄 체제 전환: {prev_regime.value} → {new_regime.value}"
            )
            
            # PANIC 진입 시 저점 기록 시작
            if new_regime == Regime.PANIC:
                self.state.panic_low = market_data["kospi_close"]
            
        # PANIC 중 저점 갱신
        if self.state.current == Regime.PANIC:
            if self.state.panic_low is None or market_data["kospi_close"] < self.state.panic_low:
                self.state.panic_low = market_data["kospi_close"]
        
        return self.state
    
    def _determine_regime(
        self, today_ret: float, five_day_ret: float, 
        md: dict, vol_ratio: float
    ) -> Regime:
        """체제 결정 로직 — 하향은 스킵 가능, 상향은 단계적"""
        
        current = self.state.current
        
        # ── 하향 전환 (위험 감지: 스킵 가능) ──
        
        # PANIC 조건: 어디서든 즉시 전환
        if (five_day_ret <= self.config.REGIME_PANIC_THRESHOLD or
            (md["consecutive_down"] >= 3 and today_ret < -0.015)):
            return Regime.PANIC
        
        # SHOCK 조건
        if (today_ret <= self.config.REGIME_SHOCK_THRESHOLD and
            md["vix"] >= self.config.REGIME_VIX_HIGH):
            return Regime.SHOCK
        
        # CAUTION 조건
        if current == Regime.NORMAL:
            if today_ret < -0.02 or md["vix"] > 20:
                return Regime.CAUTION
        
        # ── 상향 전환 (회복 감지: 한 단계씩만) ──
        
        if current == Regime.PANIC:
            # PANIC → RECOVERY_EARLY
            if (md["is_bullish_candle"] and 
                vol_ratio >= self.config.REGIME_RECOVERY_VOL_MULT and
                self.state.panic_low is not None and
                md["kospi_close"] > self.state.panic_low):
                return Regime.RECOVERY_EARLY
            return Regime.PANIC  # 조건 불충분 시 유지
        
        if current == Regime.RECOVERY_EARLY:
            # RECOVERY_EARLY → RECOVERY_CONFIRMED
            if (md["kospi_close"] > md["kospi_5ma"] and
                md["foreign_net_buy"] > 0):
                return Regime.RECOVERY_CONFIRMED
            # 다시 빠지면 PANIC 복귀
            if today_ret <= self.config.REGIME_SHOCK_THRESHOLD:
                return Regime.PANIC
            return Regime.RECOVERY_EARLY
        
        if current == Regime.RECOVERY_CONFIRMED:
            # RECOVERY_CONFIRMED → NORMAL
            if (md["kospi_close"] > md["kospi_20ma"] and
                md["vix"] < 20):
                return Regime.NORMAL
            return Regime.RECOVERY_CONFIRMED
        
        if current == Regime.SHOCK:
            # SHOCK → CAUTION (완화)
            if today_ret > 0 and md["vix"] < self.config.REGIME_VIX_HIGH:
                return Regime.CAUTION
            return Regime.SHOCK
        
        if current == Regime.CAUTION:
            # CAUTION → NORMAL
            if today_ret > 0 and md["vix"] < 20:
                return Regime.NORMAL
            return Regime.CAUTION
        
        return current


# ============================================================
#  Layer 2: 충격 분류기 (Shock Classifier)
# ============================================================

class ShockType(Enum):
    """충격의 원인 — 원인에 따라 기회 섹터가 달라진다"""
    NONE = "NONE"
    GEOPOLITICAL = "GEOPOLITICAL"   # 지정학 (전쟁, 제재)
    RATE = "RATE"                   # 금리 (Fed, 한은)
    EARNINGS = "EARNINGS"           # 실적 (어닝 시즌)
    LIQUIDITY = "LIQUIDITY"         # 유동성 (마진콜, 신용경색)
    COMPOUND = "COMPOUND"           # 복합 (2개 이상 동시)


@dataclass
class ShockAnalysis:
    """충격 분석 결과"""
    shock_type: ShockType = ShockType.NONE
    confidence: float = 0.0              # 0~1 판단 신뢰도
    description: str = ""
    affected_sectors: list = field(default_factory=list)   # 직접 영향 섹터
    opportunity_sectors: list = field(default_factory=list) # 기회 섹터 (과도하게 빠진)
    signals: dict = field(default_factory=dict)             # 개별 시그널 값


class ShockClassifier:
    """
    "왜 떨어졌는지"를 판단한다.
    
    NIGHTWATCH 데이터를 활용:
    - GDELT (지정학 이벤트)
    - 유가 변동
    - 금리 변동 (US10Y, KR3Y)
    - VIX 레벨
    - 방산/정유 주가 (수혜주 움직임으로 역추론)
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.logger = logging.getLogger("CORTEX.Shock")
    
    def classify(self, nightwatch_data: dict) -> ShockAnalysis:
        """
        NIGHTWATCH에서 받은 데이터로 충격 유형을 분류한다.
        
        Args:
            nightwatch_data: {
                "gdelt_conflict_index": float,    # GDELT 분쟁 지수 (평소 대비 배수)
                "oil_change_pct": float,           # 유가 변동률
                "us10y_change_bp": float,          # 미국 10년물 금리 변동 (bp)
                "kr3y_change_bp": float,           # 한국 3년물 금리 변동 (bp)
                "vix": float,                      # VIX
                "defense_sector_return": float,    # 방산 섹터 수익률
                "refinery_sector_return": float,   # 정유 섹터 수익률
                "credit_spread_change": float,     # 신용 스프레드 변동
                "margin_call_volume": float,       # 반대매매 규모 (추정)
                "earnings_surprise_count": int,    # 어닝 서프라이즈/쇼크 종목 수
            }
        
        Returns:
            ShockAnalysis
        """
        
        signals = {}
        scores = {
            "GEOPOLITICAL": 0,
            "RATE": 0,
            "EARNINGS": 0,
            "LIQUIDITY": 0,
        }
        
        nd = nightwatch_data
        
        # ── 지정학 시그널 ──
        gdelt = nd.get("gdelt_conflict_index", 0)
        oil = nd.get("oil_change_pct", 0)
        defense_ret = nd.get("defense_sector_return", 0)
        
        if gdelt >= self.config.GDELT_SPIKE_THRESHOLD:
            scores["GEOPOLITICAL"] += 35
            signals["gdelt_spike"] = gdelt
        if oil >= self.config.OIL_SPIKE_THRESHOLD:
            scores["GEOPOLITICAL"] += 25
            signals["oil_spike"] = oil
        if defense_ret > 0.03:  # 방산 +3% 이상 = 지정학 수혜 확인
            scores["GEOPOLITICAL"] += 20
            signals["defense_surge"] = defense_ret
        
        # ── 금리 시그널 ──
        us10y = abs(nd.get("us10y_change_bp", 0))
        kr3y = abs(nd.get("kr3y_change_bp", 0))
        
        if us10y >= self.config.RATE_SPIKE_THRESHOLD:
            scores["RATE"] += 40
            signals["us10y_spike"] = us10y
        if kr3y >= self.config.RATE_SPIKE_THRESHOLD * 0.7:
            scores["RATE"] += 25
            signals["kr3y_spike"] = kr3y
        
        # ── 실적 시그널 ──
        earnings_count = nd.get("earnings_surprise_count", 0)
        if earnings_count >= 5:
            scores["EARNINGS"] += 50
            signals["earnings_shock_count"] = earnings_count
        
        # ── 유동성 시그널 ──
        credit_spread = nd.get("credit_spread_change", 0)
        margin_call = nd.get("margin_call_volume", 0)
        
        if credit_spread > 0.5:
            scores["LIQUIDITY"] += 35
            signals["credit_spread_widen"] = credit_spread
        if margin_call > 0:
            scores["LIQUIDITY"] += 30
            signals["margin_call"] = margin_call
        
        # ── 최종 판정 ──
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        # 복합 충격 체크: 2개 이상 40점 넘으면
        high_scores = [k for k, v in scores.items() if v >= 40]
        
        if max_score < 30:
            shock_type = ShockType.NONE
            confidence = 0.0
        elif len(high_scores) >= 2:
            shock_type = ShockType.COMPOUND
            confidence = min(max_score / 100, 0.95)
        else:
            shock_type = ShockType[max_type]
            confidence = min(max_score / 100, 0.95)
        
        # 기회 섹터 결정
        all_sectors = list(self.config.SECTOR_MAP.keys())
        affected = self.config.SHOCK_AFFECTED_SECTORS.get(max_type, [])
        opportunity = [s for s in all_sectors if s not in affected]
        
        description = self._build_description(shock_type, signals)
        
        result = ShockAnalysis(
            shock_type=shock_type,
            confidence=confidence,
            description=description,
            affected_sectors=affected,
            opportunity_sectors=opportunity,
            signals=signals,
        )
        
        self.logger.info(f"⚡ 충격 분류: {shock_type.value} (신뢰도: {confidence:.0%})")
        return result
    
    def _build_description(self, shock_type: ShockType, signals: dict) -> str:
        """사람이 읽기 쉬운 충격 요약"""
        
        if shock_type == ShockType.NONE:
            return "특이 충격 없음 — 일반적 변동"
        
        parts = []
        if shock_type == ShockType.GEOPOLITICAL:
            parts.append("🌍 지정학 충격 감지")
            if "gdelt_spike" in signals:
                parts.append(f"GDELT 분쟁지수 {signals['gdelt_spike']:.1f}배")
            if "oil_spike" in signals:
                parts.append(f"유가 {signals['oil_spike']:+.1%}")
            if "defense_surge" in signals:
                parts.append(f"방산섹터 {signals['defense_surge']:+.1%}")
        
        elif shock_type == ShockType.RATE:
            parts.append("📈 금리 충격 감지")
            if "us10y_spike" in signals:
                parts.append(f"US10Y {signals['us10y_spike']:+.0f}bp")
        
        elif shock_type == ShockType.LIQUIDITY:
            parts.append("💧 유동성 충격 감지")
            if "margin_call" in signals:
                parts.append("반대매매 급증")
        
        elif shock_type == ShockType.COMPOUND:
            parts.append("🔴 복합 충격 감지 — 다중 리스크 동시 발생")
        
        return " | ".join(parts)


# ============================================================
#  Layer 3: 기회 스코어링 (Opportunity Scorer)
# ============================================================

@dataclass
class StockOpportunity:
    """개별 종목 기회 분석 결과"""
    code: str                     # 종목코드
    name: str                     # 종목명
    sector: str                   # 섹터
    
    # 가격 데이터
    reference_high: float = 0.0   # 기준기간(12월~2/27) 고점
    current_price: float = 0.0    # 현재가
    drawdown_pct: float = 0.0     # 고점 대비 낙폭 (%)
    
    # 크래시 전 상태
    rsi_before_crash: float = 0.0    # 크래시 직전 RSI
    ma20_slope_before: float = 0.0   # 크래시 전 20일선 기울기
    was_uptrend: bool = False        # 상승추세였는지
    
    # 현재 회복 신호
    is_first_bullish: bool = False   # 크래시 후 첫 양봉
    volume_surge: bool = False       # 거래량 증가
    above_crash_low: bool = True     # 저점 이탈 안함
    
    # 점수
    drawdown_score: float = 0.0
    fundamental_score: float = 0.0
    unrelated_score: float = 0.0
    recovery_score: float = 0.0
    total_score: float = 0.0
    
    # 실행 정보
    recommended_action: str = ""     # "BUY" / "WATCH" / "SKIP"
    suggested_size_pct: float = 0.0  # 추천 비중 (%)
    stop_loss_price: float = 0.0     # 손절가
    take_profit_price: float = 0.0   # 익절가


class OpportunityScorer:
    """
    "어떤 종목이 진짜 기회인지" 점수를 매긴다.
    
    핵심 원칙: "강한 종목의 이유 없는 하락" = 최고의 기회
    → RSI 높았는데 + 충격과 무관한데 + 많이 빠졌으면 = 고점수
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.logger = logging.getLogger("CORTEX.Scorer")
    
    def score_universe(
        self, 
        stocks: list[dict], 
        shock: ShockAnalysis,
        regime: RegimeState
    ) -> list[StockOpportunity]:
        """
        전체 유니버스를 스코어링한다.
        
        Args:
            stocks: 종목 데이터 리스트. 각 항목:
                {
                    "code": str,
                    "name": str,
                    "sector": str,
                    "closes": list[float],        # 기준기간 종가 배열
                    "current_close": float,        # 현재 종가
                    "prev_close": float,           # 전일 종가
                    "rsi_before": float,           # 크래시 직전 RSI(14)
                    "ma20_slope": float,           # 20일선 기울기 (크래시 전)
                    "ma5_before": float,           # 크래시 전 5일선
                    "ma20_before": float,          # 크래시 전 20일선
                    "volume_today": float,         # 오늘 거래량
                    "vol_20avg": float,            # 20일 평균 거래량
                    "crash_low": float,            # 크래시 기간 최저가
                }
            shock: Layer 2의 충격 분석 결과
            regime: Layer 1의 체제 상태
        
        Returns:
            기회 점수 내림차순 정렬된 StockOpportunity 리스트
        """
        
        opportunities = []
        
        for stock in stocks:
            opp = self._score_single(stock, shock)
            if opp.total_score >= self.config.SCORE_MIN_THRESHOLD:
                opp.recommended_action = "BUY"
                opp = self._set_execution_params(opp, regime)
            elif opp.total_score >= 40:
                opp.recommended_action = "WATCH"
            else:
                opp.recommended_action = "SKIP"
            
            opportunities.append(opp)
        
        # 점수 내림차순 정렬
        opportunities.sort(key=lambda x: x.total_score, reverse=True)
        
        buy_count = sum(1 for o in opportunities if o.recommended_action == "BUY")
        self.logger.info(
            f"📊 스코어링 완료: {len(stocks)}종목 → BUY {buy_count}개"
        )
        
        return opportunities
    
    def _score_single(self, stock: dict, shock: ShockAnalysis) -> StockOpportunity:
        """개별 종목 점수 계산"""
        
        opp = StockOpportunity(
            code=stock["code"],
            name=stock["name"],
            sector=stock.get("sector", "기타"),
        )
        
        # ── 1. 낙폭 점수 (많이 빠질수록 높은 점수) ──
        ref_high = max(stock["closes"]) if stock["closes"] else stock["current_close"]
        opp.reference_high = ref_high
        opp.current_price = stock["current_close"]
        
        if ref_high > 0:
            opp.drawdown_pct = (stock["current_close"] - ref_high) / ref_high
        
        if opp.drawdown_pct <= self.config.DRAWDOWN_MIN:
            # -10% ~ -50% 구간을 0~100 스케일로 변환
            raw = min(abs(opp.drawdown_pct), 0.50)
            opp.drawdown_score = (raw - 0.10) / 0.40 * self.config.SCORE_DRAWDOWN_WEIGHT
        
        # ── 2. 펀더멘탈 건재 점수 ──
        opp.rsi_before_crash = stock.get("rsi_before", 50)
        opp.ma20_slope_before = stock.get("ma20_slope", 0)
        
        # RSI > 55 (크래시 전 강했음)
        if opp.rsi_before_crash >= self.config.RSI_STRONG_FLOOR:
            opp.fundamental_score += self.config.SCORE_FUNDAMENTAL_WEIGHT * 0.5
        
        # 상승추세였음 (5일선 > 20일선)
        if stock.get("ma5_before", 0) > stock.get("ma20_before", 0):
            opp.was_uptrend = True
            opp.fundamental_score += self.config.SCORE_FUNDAMENTAL_WEIGHT * 0.3
        
        # 20일선 기울기 양수 (우상향)
        if opp.ma20_slope_before > 0:
            opp.fundamental_score += self.config.SCORE_FUNDAMENTAL_WEIGHT * 0.2
        
        # ── 3. 충격 무관 점수 ──
        if opp.sector not in shock.affected_sectors:
            opp.unrelated_score = self.config.SCORE_UNRELATED_WEIGHT
        elif opp.sector in shock.opportunity_sectors:
            opp.unrelated_score = self.config.SCORE_UNRELATED_WEIGHT * 0.5
        
        # ── 4. 회복 신호 점수 ──
        # 첫 양봉
        if stock["current_close"] > stock.get("prev_close", stock["current_close"]):
            opp.is_first_bullish = True
            opp.recovery_score += self.config.SCORE_RECOVERY_WEIGHT * 0.4
        
        # 거래량 증가
        vol_avg = stock.get("vol_20avg", 1)
        if vol_avg > 0 and stock.get("volume_today", 0) > vol_avg * 1.3:
            opp.volume_surge = True
            opp.recovery_score += self.config.SCORE_RECOVERY_WEIGHT * 0.3
        
        # 저점 이탈 안함
        crash_low = stock.get("crash_low", 0)
        if crash_low > 0 and stock["current_close"] > crash_low:
            opp.above_crash_low = True
            opp.recovery_score += self.config.SCORE_RECOVERY_WEIGHT * 0.3
        
        # ── 합산 ──
        opp.total_score = (
            opp.drawdown_score + 
            opp.fundamental_score + 
            opp.unrelated_score + 
            opp.recovery_score
        )
        
        return opp
    
    def _set_execution_params(
        self, opp: StockOpportunity, regime: RegimeState
    ) -> StockOpportunity:
        """매수 실행 파라미터 설정 (포지션 크기, 손절/익절)"""
        
        # 점수 기반 비중 결정: 60~100점 → 1~3%
        score_ratio = min((opp.total_score - 60) / 40, 1.0)
        opp.suggested_size_pct = (
            self.config.MAX_SINGLE_POSITION * (0.3 + 0.7 * score_ratio)
        )
        
        # 체제에 따른 비중 조정
        capital_use = regime.get_rules()["capital_use"]
        opp.suggested_size_pct *= capital_use
        
        # 손절/익절가
        opp.stop_loss_price = opp.current_price * (1 + self.config.STOP_LOSS)
        opp.take_profit_price = opp.current_price * (1 + self.config.TAKE_PROFIT)
        
        return opp


# ============================================================
#  Layer 4: 자율 실행 엔진 (Autonomous Executor)
# ============================================================

@dataclass
class ExecutionOrder:
    """매매 주문"""
    stock_code: str
    stock_name: str
    action: str              # "BUY" / "SELL"
    price: float
    quantity: int
    reason: str
    stop_loss: float
    take_profit: float
    cortex_score: float
    timestamp: str = ""


class AutonomousExecutor:
    """
    CORTEX의 최종 실행 레이어
    
    3가지 모드:
    - REPORT: 텔레그램 보고만 (네가 직접 매매)
    - APPROVE: 텔레그램 보고 + 승인 시 매수
    - AUTO: 자율 매수 + 사후 보고
    
    안전장치:
    1. 손절 -5% 하드코딩 (어떤 모드든 무조건)
    2. CORTEX 슬롯 자본 15% 상한
    3. 종목당 3% 상한
    4. 최대 10종목 동시 보유
    5. PANIC 체제에서는 절대 매수 안함
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.logger = logging.getLogger("CORTEX.Executor")
        self.positions: list[dict] = []       # 현재 CORTEX 보유 종목
        self.order_history: list[dict] = []   # 주문 이력
    
    def execute(
        self,
        opportunities: list[StockOpportunity],
        regime: RegimeState,
        shock: ShockAnalysis,
        total_capital: float,
        available_cash: float,
    ) -> list[ExecutionOrder]:
        """
        기회 목록을 받아 실행 결정을 한다.
        
        안전장치 체크 순서:
        1. 체제 확인 (PANIC이면 매수 불가)
        2. CORTEX 슬롯 잔여 확인
        3. 현금 잔고 확인 (★미수금 방지★)
        4. 종목별 비중 확인
        5. 최대 보유 종목 수 확인
        """
        
        orders: list[ExecutionOrder] = []
        
        # ── 안전장치 1: 체제 확인 ──
        rules = regime.get_rules()
        if not rules["cortex_active"]:
            self.logger.info(
                f"⏸️ CORTEX 비활성 체제: {regime.current.value} — 매수 건너뜀"
            )
            return orders
        
        if not rules["new_buy"]:
            self.logger.info("🚫 신규매수 불가 체제")
            return orders
        
        # ── 안전장치 2: CORTEX 슬롯 자본 확인 ──
        cortex_budget = total_capital * self.config.CORTEX_CAPITAL_RATIO
        cortex_used = sum(p.get("market_value", 0) for p in self.positions)
        cortex_remaining = cortex_budget - cortex_used
        
        if cortex_remaining <= 0:
            self.logger.info("💰 CORTEX 슬롯 자본 소진")
            return orders
        
        # ── 안전장치 3: 현금 잔고 확인 (★미수금 방지★) ──
        if available_cash <= 0:
            self.logger.warning("🚨 현금 잔고 부족 — 미수금 방지를 위해 매수 중단")
            return orders
        
        # ── 안전장치 4: 최대 보유 종목 수 ──
        current_count = len(self.positions)
        remaining_slots = self.config.MAX_POSITIONS - current_count
        
        if remaining_slots <= 0:
            self.logger.info("📦 최대 보유 종목 수 도달")
            return orders
        
        # ── BUY 종목만 필터 → 상위 N개 ──
        buy_targets = [
            o for o in opportunities 
            if o.recommended_action == "BUY" and o.total_score >= self.config.SCORE_MIN_THRESHOLD
        ]
        buy_targets = buy_targets[:remaining_slots]
        
        for target in buy_targets:
            # 종목별 투자금 계산
            invest_amount = min(
                total_capital * target.suggested_size_pct,  # 비중 기반
                cortex_remaining,                           # CORTEX 슬롯 잔여
                available_cash,                             # 현금 잔고
            )
            
            if invest_amount < 100000:  # 최소 10만원
                continue
            
            quantity = int(invest_amount / target.current_price)
            if quantity <= 0:
                continue
            
            order = ExecutionOrder(
                stock_code=target.code,
                stock_name=target.name,
                action="BUY",
                price=target.current_price,
                quantity=quantity,
                reason=(
                    f"CORTEX 점수 {target.total_score:.0f} | "
                    f"낙폭 {target.drawdown_pct:.1%} | "
                    f"RSI전 {target.rsi_before_crash:.0f} | "
                    f"충격유형: {shock.shock_type.value}"
                ),
                stop_loss=target.stop_loss_price,
                take_profit=target.take_profit_price,
                cortex_score=target.total_score,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            
            orders.append(order)
            cortex_remaining -= quantity * target.current_price
            available_cash -= quantity * target.current_price
            
            if available_cash <= 0 or cortex_remaining <= 0:
                break
        
        # ── 실행 모드에 따른 처리 ──
        if self.config.EXECUTION_MODE == "REPORT":
            self._send_report(orders, regime, shock)
        elif self.config.EXECUTION_MODE == "APPROVE":
            self._send_approval_request(orders, regime, shock)
        elif self.config.EXECUTION_MODE == "AUTO":
            self._auto_execute(orders)
            self._send_execution_report(orders, regime, shock)
        
        return orders
    
    def check_stop_loss(self, current_prices: dict) -> list[ExecutionOrder]:
        """
        보유 종목 손절 체크 — 장중 5분마다 실행
        
        ★ 이 함수는 모드 무관하게 항상 자동 실행 ★
        손절은 하드코딩이다. 판단의 여지가 없다.
        """
        
        sell_orders = []
        
        for pos in self.positions:
            code = pos["code"]
            if code not in current_prices:
                continue
            
            current = current_prices[code]
            entry = pos["entry_price"]
            pnl_pct = (current - entry) / entry
            
            # 손절
            if pnl_pct <= self.config.STOP_LOSS:
                order = ExecutionOrder(
                    stock_code=code,
                    stock_name=pos["name"],
                    action="SELL",
                    price=current,
                    quantity=pos["quantity"],
                    reason=f"🔴 CORTEX 손절: {pnl_pct:.1%} (한도: {self.config.STOP_LOSS:.0%})",
                    stop_loss=0,
                    take_profit=0,
                    cortex_score=0,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                sell_orders.append(order)
                self.logger.warning(f"🔴 손절 실행: {pos['name']} {pnl_pct:.1%}")
            
            # 익절
            elif pnl_pct >= self.config.TAKE_PROFIT:
                order = ExecutionOrder(
                    stock_code=code,
                    stock_name=pos["name"],
                    action="SELL",
                    price=current,
                    quantity=pos["quantity"],
                    reason=f"🟢 CORTEX 익절: {pnl_pct:.1%}",
                    stop_loss=0,
                    take_profit=0,
                    cortex_score=0,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                sell_orders.append(order)
                self.logger.info(f"🟢 익절 실행: {pos['name']} {pnl_pct:.1%}")
        
        if sell_orders:
            self._send_stoploss_report(sell_orders)
        
        return sell_orders
    
    # ── 텔레그램 메시지 ──
    
    def _send_report(
        self, orders: list[ExecutionOrder], 
        regime: RegimeState, shock: ShockAnalysis
    ):
        """Phase 1: 보고만"""
        
        if not orders:
            return
        
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━",
            "🧠 CORTEX 줍줍 리포트",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📊 체제: {regime.current.value}",
            f"⚡ 충격: {shock.shock_type.value} ({shock.confidence:.0%})",
            f"📝 {shock.description}",
            f"",
            f"🎯 매수 후보 {len(orders)}종목:",
            f"",
        ]
        
        for i, order in enumerate(orders, 1):
            lines.extend([
                f"{i}. {order.stock_name} ({order.stock_code})",
                f"   점수: {order.cortex_score:.0f} | 현재가: {order.price:,.0f}원",
                f"   수량: {order.quantity}주 | 손절: {order.stop_loss:,.0f}원",
                f"   사유: {order.reason}",
                f"",
            ])
        
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━",
            '💬 "매수 실행" → 전체 매수',
            '💬 "1,3,5" → 선택 매수',
            '💬 "대기" → 다음 시그널까지 홀드',
            "━━━━━━━━━━━━━━━━━━━━━",
        ])
        
        self._telegram_send("\n".join(lines))
    
    def _send_approval_request(
        self, orders: list[ExecutionOrder],
        regime: RegimeState, shock: ShockAnalysis
    ):
        """Phase 1.5: 보고 + 승인 대기"""
        self._send_report(orders, regime, shock)
        # 승인 응답 처리는 텔레그램 봇 webhook에서 처리
    
    def _send_execution_report(
        self, orders: list[ExecutionOrder],
        regime: RegimeState, shock: ShockAnalysis
    ):
        """Phase 2: 자율 실행 후 사후 보고"""
        
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━",
            "🤖 CORTEX 자율 매수 완료",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📊 체제: {regime.current.value}",
            f"⚡ 충격: {shock.shock_type.value}",
            f"",
        ]
        
        total_invested = 0
        for i, order in enumerate(orders, 1):
            amount = order.price * order.quantity
            total_invested += amount
            lines.append(
                f"{i}. {order.stock_name} {order.price:,.0f}원 x {order.quantity}주 "
                f"= {amount:,.0f}원 (점수: {order.cortex_score:.0f})"
            )
        
        lines.extend([
            f"",
            f"💰 총 투자: {total_invested:,.0f}원",
            f"🛡️ 손절: 전종목 {self.config.STOP_LOSS:.0%} 자동 설정",
            "━━━━━━━━━━━━━━━━━━━━━",
        ])
        
        self._telegram_send("\n".join(lines))
    
    def _send_stoploss_report(self, orders: list[ExecutionOrder]):
        """손절/익절 실행 보고"""
        
        lines = ["🛡️ CORTEX 자동 청산 실행", ""]
        for order in orders:
            lines.append(
                f"{'🔴' if 'SELL' else '🟢'} {order.stock_name} "
                f"{order.quantity}주 @ {order.price:,.0f}원 | {order.reason}"
            )
        
        self._telegram_send("\n".join(lines))
    
    def _auto_execute(self, orders: list[ExecutionOrder]):
        """
        실제 KIS API를 통한 주문 실행
        ★ 여기에 기존 JARVIS의 KIS API 주문 함수를 연결 ★
        """
        for order in orders:
            self.logger.info(
                f"📤 주문 전송: {order.action} {order.stock_name} "
                f"{order.quantity}주 @ {order.price:,.0f}원"
            )
            # TODO: KIS API 연동
            # kis_api.place_order(
            #     code=order.stock_code,
            #     qty=order.quantity,
            #     price=order.price,
            #     order_type="BUY" if order.action == "BUY" else "SELL"
            # )
    
    def _telegram_send(self, message: str):
        """텔레그램 메시지 발송"""
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": self.config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            self.logger.error(f"텔레그램 발송 실패: {e}")


# ============================================================
#  CORTEX 메인 엔진 — 모든 레이어 통합
# ============================================================

class CORTEX:
    """
    JARVIS CORTEX — 모든 레이어를 하나로 통합한 메인 엔진
    
    사용법:
        cortex = CORTEX()
        result = cortex.run(market_data, nightwatch_data, stocks, capital_info)
    
    실행 주기:
        - 장중: 5분마다 run() 호출
        - 장마감 후: NIGHTWATCH 결과 반영하여 다음날 준비
    
    ┌──────────────────────────────────────────────┐
    │  CORTEX.run() 실행 흐름                       │
    │                                                │
    │  market_data ──→ [Layer 1] 체제 판단           │
    │                      ↓                         │
    │  nightwatch  ──→ [Layer 2] 충격 분류           │
    │                      ↓                         │
    │  stocks      ──→ [Layer 3] 기회 스코어링       │
    │                      ↓                         │
    │  capital     ──→ [Layer 4] 실행 (매수/보고)    │
    │                      ↓                         │
    │               텔레그램 알림 + DB 기록           │
    └──────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.regime_detector = RegimeDetector(config)
        self.shock_classifier = ShockClassifier(config)
        self.opportunity_scorer = OpportunityScorer(config)
        self.executor = AutonomousExecutor(config)
        self.logger = logging.getLogger("CORTEX")
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    
    def run(
        self,
        market_data: dict,
        nightwatch_data: dict,
        stocks: list[dict],
        total_capital: float,
        available_cash: float,
    ) -> dict:
        """
        CORTEX 메인 실행 함수
        
        Args:
            market_data: Layer 1용 시장 데이터 (RegimeDetector.detect 참조)
            nightwatch_data: Layer 2용 NIGHTWATCH 데이터 (ShockClassifier.classify 참조)
            stocks: Layer 3용 종목 데이터 리스트 (OpportunityScorer.score_universe 참조)
            total_capital: 전체 투자 자본
            available_cash: 현재 사용 가능한 현금 잔고
        
        Returns:
            {
                "regime": RegimeState,
                "shock": ShockAnalysis,
                "opportunities": list[StockOpportunity],
                "orders": list[ExecutionOrder],
                "summary": str,
            }
        """
        
        self.logger.info("=" * 50)
        self.logger.info("🧠 CORTEX 실행 시작")
        self.logger.info("=" * 50)
        
        # ── Layer 1: 체제 판단 ──
        self.logger.info("── Layer 1: 시장 체제 인식 ──")
        regime = self.regime_detector.detect(market_data)
        self.logger.info(
            f"   체제: {regime.current.value} | "
            f"KOSPI 당일: {regime.kospi_today_return:+.2%} | "
            f"5일: {regime.kospi_5d_return:+.2%} | "
            f"VIX: {regime.vix:.1f}"
        )
        
        # ── Layer 2: 충격 분류 ──
        self.logger.info("── Layer 2: 충격 분류 ──")
        shock = self.shock_classifier.classify(nightwatch_data)
        self.logger.info(
            f"   충격: {shock.shock_type.value} | "
            f"신뢰도: {shock.confidence:.0%} | "
            f"{shock.description}"
        )
        
        # ── Layer 3: 기회 스코어링 ──
        self.logger.info("── Layer 3: 기회 스코어링 ──")
        opportunities = self.opportunity_scorer.score_universe(
            stocks, shock, regime
        )
        
        buy_list = [o for o in opportunities if o.recommended_action == "BUY"]
        watch_list = [o for o in opportunities if o.recommended_action == "WATCH"]
        self.logger.info(
            f"   BUY: {len(buy_list)}종목 | WATCH: {len(watch_list)}종목"
        )
        
        if buy_list:
            self.logger.info("   TOP 5:")
            for o in buy_list[:5]:
                self.logger.info(
                    f"   - {o.name}: 점수 {o.total_score:.0f} | "
                    f"낙폭 {o.drawdown_pct:.1%} | RSI전 {o.rsi_before_crash:.0f}"
                )
        
        # ── Layer 4: 실행 ──
        self.logger.info("── Layer 4: 실행 엔진 ──")
        self.logger.info(f"   모드: {self.config.EXECUTION_MODE}")
        
        orders = self.executor.execute(
            opportunities=opportunities,
            regime=regime,
            shock=shock,
            total_capital=total_capital,
            available_cash=available_cash,
        )
        
        self.logger.info(f"   주문: {len(orders)}건")
        
        # ── 결과 요약 ──
        summary = self._build_summary(regime, shock, buy_list, orders)
        self.logger.info("=" * 50)
        self.logger.info("🧠 CORTEX 실행 완료")
        self.logger.info("=" * 50)
        
        # ── DB 기록 ──
        self._save_to_db(regime, shock, opportunities, orders)
        
        return {
            "regime": regime,
            "shock": shock,
            "opportunities": opportunities,
            "orders": orders,
            "summary": summary,
        }
    
    def run_stoploss_check(self, current_prices: dict) -> list[ExecutionOrder]:
        """
        장중 손절 체크 — 5분마다 호출
        ★ run()과 별도로 돌아야 함 ★
        """
        return self.executor.check_stop_loss(current_prices)
    
    def _build_summary(
        self, regime: RegimeState, shock: ShockAnalysis,
        buy_list: list, orders: list
    ) -> str:
        """콘솔 + DB용 요약"""
        return (
            f"체제={regime.current.value} | "
            f"충격={shock.shock_type.value}({shock.confidence:.0%}) | "
            f"후보={len(buy_list)} | "
            f"주문={len(orders)} | "
            f"모드={self.config.EXECUTION_MODE}"
        )
    
    def _save_to_db(
        self, regime, shock, opportunities, orders
    ):
        """Control Tower DB에 CORTEX 실행 결과 저장"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            # CORTEX 로그 테이블 (없으면 생성)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cortex_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    regime TEXT,
                    shock_type TEXT,
                    shock_confidence REAL,
                    buy_candidates INTEGER,
                    orders_placed INTEGER,
                    execution_mode TEXT,
                    summary TEXT,
                    details TEXT
                )
            """)
            
            # CORTEX 주문 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cortex_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    stock_code TEXT,
                    stock_name TEXT,
                    action TEXT,
                    price REAL,
                    quantity INTEGER,
                    cortex_score REAL,
                    reason TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'PENDING'
                )
            """)
            
            # 로그 저장
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            buy_list = [o for o in opportunities if o.recommended_action == "BUY"]
            
            cursor.execute("""
                INSERT INTO cortex_log 
                (timestamp, regime, shock_type, shock_confidence, 
                 buy_candidates, orders_placed, execution_mode, summary, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                regime.current.value,
                shock.shock_type.value,
                shock.confidence,
                len(buy_list),
                len(orders),
                self.config.EXECUTION_MODE,
                self._build_summary(regime, shock, buy_list, orders),
                json.dumps([asdict(o) for o in buy_list[:20]], ensure_ascii=False),
            ))
            
            # 주문 저장
            for order in orders:
                cursor.execute("""
                    INSERT INTO cortex_orders
                    (timestamp, stock_code, stock_name, action, price,
                     quantity, cortex_score, reason, stop_loss, take_profit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.timestamp,
                    order.stock_code,
                    order.stock_name,
                    order.action,
                    order.price,
                    order.quantity,
                    order.cortex_score,
                    order.reason,
                    order.stop_loss,
                    order.take_profit,
                ))
            
            conn.commit()
            conn.close()
            self.logger.info("💾 DB 저장 완료")
            
        except Exception as e:
            self.logger.error(f"DB 저장 실패: {e}")


# ============================================================
#  사용 예시 (퐝가의 현재 상황에 맞춘 데모)
# ============================================================

if __name__ == "__main__":
    
    print("""
    ╔══════════════════════════════════════════════════╗
    ║  CORTEX v1.0 데모 — 이란전쟁 후 줍줍 시뮬레이션  ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    # ── 1. CORTEX 초기화 ──
    cortex = CORTEX()
    
    # ── 2. 현재 시장 상태 입력 (3/6 이후 상황 가정) ──
    market_data = {
        "kospi_close": 2450,
        "kospi_prev_close": 2380,         # 전일 대비 양봉
        "kospi_5d_ago_close": 2650,       # 5일 전 대비 -7.5%
        "kospi_20ma": 2600,
        "kospi_5ma": 2420,
        "kospi_volume": 15_000_000_000,   # 거래량 높음
        "kospi_vol_20avg": 10_000_000_000,
        "vix": 28,
        "foreign_net_buy": -500_000_000,  # 외국인 아직 매도 중
        "consecutive_down": 1,             # 어제 하락, 오늘 반등
        "is_bullish_candle": True,         # 오늘 양봉
    }
    
    # ── 3. NIGHTWATCH 데이터 ──
    nightwatch_data = {
        "gdelt_conflict_index": 3.5,       # 평소 3.5배 → 지정학 충격
        "oil_change_pct": 0.12,            # 유가 +12%
        "us10y_change_bp": 0.03,           # 금리 거의 변동 없음
        "kr3y_change_bp": 0.02,
        "vix": 28,
        "defense_sector_return": 0.08,     # 방산 +8%
        "refinery_sector_return": 0.05,    # 정유 +5%
        "credit_spread_change": 0.2,
        "margin_call_volume": 50_000_000,
        "earnings_surprise_count": 1,
    }
    
    # ── 4. 종목 데이터 (v10.3 유니버스 중 일부 예시) ──
    stocks = [
        {
            "code": "005930", "name": "삼성전자", "sector": "반도체",
            "closes": [72000, 73000, 74000, 75000, 73000, 74500, 75500, 76000],
            "current_close": 63000, "prev_close": 61000,
            "rsi_before": 62, "ma20_slope": 0.002,
            "ma5_before": 74500, "ma20_before": 73000,
            "volume_today": 25_000_000, "vol_20avg": 18_000_000,
            "crash_low": 59000,
        },
        {
            "code": "000660", "name": "SK하이닉스", "sector": "반도체",
            "closes": [180000, 185000, 190000, 195000, 192000, 188000, 193000],
            "current_close": 155000, "prev_close": 148000,
            "rsi_before": 58, "ma20_slope": 0.001,
            "ma5_before": 191000, "ma20_before": 188000,
            "volume_today": 8_000_000, "vol_20avg": 5_500_000,
            "crash_low": 145000,
        },
        {
            "code": "035420", "name": "NAVER", "sector": "소프트웨어",
            "closes": [220000, 225000, 230000, 228000, 232000, 235000],
            "current_close": 198000, "prev_close": 195000,
            "rsi_before": 60, "ma20_slope": 0.003,
            "ma5_before": 231000, "ma20_before": 226000,
            "volume_today": 3_000_000, "vol_20avg": 2_200_000,
            "crash_low": 190000,
        },
        {
            "code": "207940", "name": "삼성바이오로직스", "sector": "바이오",
            "closes": [800000, 810000, 820000, 815000, 825000],
            "current_close": 720000, "prev_close": 710000,
            "rsi_before": 55, "ma20_slope": 0.001,
            "ma5_before": 818000, "ma20_before": 810000,
            "volume_today": 200_000, "vol_20avg": 180_000,
            "crash_low": 700000,
        },
        {
            "code": "012330", "name": "현대모비스", "sector": "기타",
            "closes": [250000, 255000, 260000, 258000, 262000],
            "current_close": 230000, "prev_close": 225000,
            "rsi_before": 57, "ma20_slope": 0.002,
            "ma5_before": 259000, "ma20_before": 255000,
            "volume_today": 400_000, "vol_20avg": 350_000,
            "crash_low": 220000,
        },
    ]
    
    # ── 5. 실행 ──
    result = cortex.run(
        market_data=market_data,
        nightwatch_data=nightwatch_data,
        stocks=stocks,
        total_capital=100_000_000,     # 총 자본 1억
        available_cash=30_000_000,     # 현금 잔고 3천만
    )
    
    # ── 6. 결과 출력 ──
    print("\n" + "=" * 60)
    print("📋 CORTEX 실행 결과 요약")
    print("=" * 60)
    print(f"체제:   {result['regime'].current.value}")
    print(f"충격:   {result['shock'].shock_type.value} ({result['shock'].confidence:.0%})")
    print(f"설명:   {result['shock'].description}")
    print(f"기회섹터: {', '.join(result['shock'].opportunity_sectors)}")
    print()
    
    print("📊 종목별 스코어:")
    for opp in result["opportunities"]:
        emoji = {"BUY": "🟢", "WATCH": "🟡", "SKIP": "⚪"}.get(opp.recommended_action, "")
        print(
            f"  {emoji} {opp.name:12s} | "
            f"점수: {opp.total_score:5.1f} | "
            f"낙폭: {opp.drawdown_pct:7.1%} | "
            f"RSI전: {opp.rsi_before_crash:4.0f} | "
            f"판정: {opp.recommended_action}"
        )
        if opp.recommended_action == "BUY":
            print(
                f"       → 비중 {opp.suggested_size_pct:.1%} | "
                f"손절 {opp.stop_loss_price:,.0f} | "
                f"익절 {opp.take_profit_price:,.0f}"
            )
    
    print()
    print(f"📤 주문: {len(result['orders'])}건")
    for order in result["orders"]:
        print(
            f"  {order.stock_name} | {order.quantity}주 x {order.price:,.0f}원 | "
            f"{order.reason}"
        )
    
    print()
    print("💡 다음 단계:")
    print("  1. Config에서 TELEGRAM_BOT_TOKEN, CHAT_ID 설정")
    print("  2. EXECUTION_MODE를 'REPORT'로 시작 (텔레그램 보고만)")
    print("  3. 실제 KIS API 데이터로 market_data, stocks 연동")
    print("  4. 신뢰 구축 후 'APPROVE' → 'AUTO'로 단계적 전환")
