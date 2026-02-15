"""Macro Analyst 에이전트 엔티티 — 매크로 환경 및 섹터 분석

시장 레짐, 섹터 로테이션, 시장 폭 등 매크로 수준 분석에 필요한
핵심 비즈니스 객체. 외부 의존 없는 순수 데이터 클래스로 구성.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MarketRegime(Enum):
    """시장 레짐 열거형"""

    BULL = "bull"  # 강세장
    RECOVERY = "recovery"  # 회복기
    SIDEWAYS = "sideways"  # 횡보
    CORRECTION = "correction"  # 조정기
    BEAR = "bear"  # 약세장
    CRISIS = "crisis"  # 위기


class SectorMomentum(Enum):
    """섹터 모멘텀 열거형"""

    LEADING = "leading"  # 선도 섹터
    IMPROVING = "improving"  # 개선 중
    WEAKENING = "weakening"  # 약화 중
    LAGGING = "lagging"  # 후행 섹터


@dataclass
class MacroRegimeAnalysis:
    """매크로 레짐 분석"""

    regime: str = "sideways"
    regime_confidence: float = 0.5
    regime_duration_days: int = 0
    transition_probability: float = 0.0  # 레짐 전환 확률
    transition_direction: str = ""  # 전환 방향 예상
    key_factors: list = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SectorRotation:
    """섹터 로테이션 분석"""

    leading_sectors: list = field(default_factory=list)  # [{"name": "반도체", "momentum": "leading", "score": 85}]
    lagging_sectors: list = field(default_factory=list)
    rotation_signal: str = "none"  # none / early_cycle / mid_cycle / late_cycle / defensive
    recommended_sectors: list = field(default_factory=list)
    avoid_sectors: list = field(default_factory=list)
    reasoning: str = ""


@dataclass
class MarketBreadth:
    """시장 폭 분석"""

    advance_decline_ratio: float = 1.0
    new_highs: int = 0
    new_lows: int = 0
    pct_above_ma20: float = 0.5
    pct_above_ma60: float = 0.5
    pct_above_ma200: float = 0.5
    breadth_thrust: bool = False  # 폭발적 상승 폭
    breadth_signal: str = "neutral"  # bullish / neutral / bearish / divergence
    reasoning: str = ""


@dataclass
class MacroDashboard:
    """매크로 대시보드 (종합)"""

    timestamp: str = ""
    regime: MacroRegimeAnalysis = field(default_factory=MacroRegimeAnalysis)
    sector_rotation: SectorRotation = field(default_factory=SectorRotation)
    market_breadth: MarketBreadth = field(default_factory=MarketBreadth)
    # 매크로 지표
    kospi_trend: str = "neutral"  # bullish / neutral / bearish
    usd_krw_impact: str = "neutral"  # positive / neutral / negative
    bond_yield_signal: str = "neutral"
    global_risk_appetite: str = "neutral"  # risk_on / neutral / risk_off
    overall_stance: str = "neutral"  # aggressive / moderate / defensive / cash
    recommendations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "regime": self.regime.to_dict(),
            "sector_rotation": self.sector_rotation.__dict__,
            "market_breadth": self.market_breadth.__dict__,
            "kospi_trend": self.kospi_trend,
            "usd_krw_impact": self.usd_krw_impact,
            "bond_yield_signal": self.bond_yield_signal,
            "global_risk_appetite": self.global_risk_appetite,
            "overall_stance": self.overall_stance,
            "recommendations": self.recommendations,
        }
