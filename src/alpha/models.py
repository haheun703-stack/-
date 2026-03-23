"""
Alpha Engine 데이터 모델 — 레짐, VETO, 청산 규칙 정의

ALPHA_ENGINE_SPECIFICATION.docx 기반:
- L1 REGIME: 4등급 (BULL/CAUTION/BEAR/CRISIS)
- L4 RISK: VETO 8조건 + X1-X5 청산 규칙
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ──────────────────────────────────────────────
# L1: Regime Level
# ──────────────────────────────────────────────

class AlphaRegimeLevel(Enum):
    """시장 레짐 4등급"""
    BULL = "BULL"         # 공격 가능
    CAUTION = "CAUTION"   # 신중
    BEAR = "BEAR"         # 방어적
    CRISIS = "CRISIS"     # 생존 모드


@dataclass
class RegimeParams:
    """레짐별 운영 파라미터"""
    max_positions: int        # 최대 동시 보유 종목 수
    position_pct: float       # 종목당 최대 비중 (0.0~1.0)
    stop_mult: float          # ATR 손절 배수
    cash_min_pct: float       # 최소 현금 비중 (0.0~1.0)

    # 레짐별 기본값
    _DEFAULTS: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._DEFAULTS = {
            "BULL":    {"max_positions": 5, "position_pct": 0.20, "stop_mult": 2.0, "cash_min_pct": 0.25},
            "CAUTION": {"max_positions": 3, "position_pct": 0.15, "stop_mult": 1.5, "cash_min_pct": 0.30},
            "BEAR":    {"max_positions": 2, "position_pct": 0.10, "stop_mult": 1.5, "cash_min_pct": 0.40},
            "CRISIS":  {"max_positions": 0, "position_pct": 0.00, "stop_mult": 1.0, "cash_min_pct": 0.65},
        }

    @classmethod
    def from_regime(cls, regime: AlphaRegimeLevel, config: dict) -> RegimeParams:
        """settings.yaml alpha_engine.regime.params에서 로딩"""
        defaults = {
            "BULL":    {"max_positions": 5, "position_pct": 0.20, "stop_mult": 2.0, "cash_min_pct": 0.25},
            "CAUTION": {"max_positions": 3, "position_pct": 0.15, "stop_mult": 1.5, "cash_min_pct": 0.30},
            "BEAR":    {"max_positions": 2, "position_pct": 0.10, "stop_mult": 1.5, "cash_min_pct": 0.40},
            "CRISIS":  {"max_positions": 0, "position_pct": 0.00, "stop_mult": 1.0, "cash_min_pct": 0.65},
        }
        params_cfg = config.get("alpha_engine", {}).get("regime", {}).get("params", {})
        d = params_cfg.get(regime.value, defaults.get(regime.value, defaults["CAUTION"]))
        return cls(
            max_positions=d["max_positions"],
            position_pct=d["position_pct"],
            stop_mult=d["stop_mult"],
            cash_min_pct=d["cash_min_pct"],
        )


# ──────────────────────────────────────────────
# L4: VETO System
# ──────────────────────────────────────────────

class VetoReason(Enum):
    """매수 거부(VETO) 사유"""
    V1_CRISIS = "CRISIS 레짐 — 매수 금지"
    V2_CASH_MIN = "최소 현금 비중 미달"
    V3_MAX_POS = "최대 동시 보유 초과"
    V4_MDD_CIRCUIT = "MDD 서킷브레이커 발동"
    V5_DAILY_LOSS = "일일 손실 한도 초과"
    V6_WEEKLY_LOSS = "주간 손실 한도 초과"
    V7_POS_SIZE = "종목당 비중 한도 초과"
    V8_STOP_SIGNAL = "STOP.signal 활성"


@dataclass
class VetoDecision:
    """VETO 판정 결과"""
    vetoed: bool
    reasons: list[VetoReason] = field(default_factory=list)
    details: str = ""

    @classmethod
    def pass_through(cls) -> VetoDecision:
        """VETO 없음 (통과)"""
        return cls(vetoed=False)


# ──────────────────────────────────────────────
# L4: Exit Rules (X1-X5 + Portfolio)
# ──────────────────────────────────────────────

class ExitRuleType(Enum):
    """청산 규칙 종류"""
    X1_HARD_STOP = "Hard Stop (ATR×2)"
    X2_FLOW_EXIT = "수급 이탈 (3일 연속 매도)"
    X3_TRAILING = "트레일링 스탑 (ATR×2.5)"
    X4_TIME_EXIT = "시간 청산 (10일+수익<2%)"
    X5_TARGET = "목표 도달 (ATR×4, 50% 부분청산)"
    X6_SCENARIO_EXIT = "시나리오 무효화 (유예 종료)"
    XP_DAILY = "일일 포트폴리오 손실 -3%"
    XP_WEEKLY = "주간 포트폴리오 손실 -5%"
    XP_MDD = "MDD -15% 서킷브레이커"


@dataclass
class ExitSignal:
    """청산 시그널"""
    triggered: bool
    rule: ExitRuleType
    exit_price: float = 0.0
    partial_pct: float = 1.0   # 1.0=전량, 0.5=부분
    details: str = ""
