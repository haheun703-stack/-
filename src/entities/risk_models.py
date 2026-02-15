"""Risk Sentinel 에이전트 엔티티 — 꼬리 리스크 감지 및 스트레스 테스트"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ThreatLevel(Enum):
    """위협 수준 열거형"""

    GREEN = "green"  # 정상
    YELLOW = "yellow"  # 주의
    ORANGE = "orange"  # 경계
    RED = "red"  # 위험
    BLACK = "black"  # 블랙스완


@dataclass
class TailRiskAlert:
    """꼬리 리스크 경보"""

    threat_level: str = "green"
    vkospi_level: float = 0.0
    vkospi_change_pct: float = 0.0
    correlation_spike: bool = False
    avg_correlation: float = 0.0
    market_breadth: float = 0.5  # 0=극단하락, 1=극단상승
    foreign_flow_signal: str = "neutral"  # massive_sell / sell / neutral / buy
    circuit_breaker_risk: bool = False
    alerts: list = field(default_factory=list)
    recommended_action: str = "hold"  # hold / reduce / hedge / emergency_exit
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CorrelationRegime:
    """상관관계 레짐 분석"""

    avg_pairwise_corr: float = 0.0
    max_pairwise_corr: float = 0.0
    regime: str = "normal"  # normal / elevated / crisis
    most_correlated_pair: tuple = ("", "")
    diversification_ratio: float = 1.0
    warnings: list = field(default_factory=list)


@dataclass
class StressTestResult:
    """스트레스 테스트 결과"""

    scenario_name: str = ""
    scenario_description: str = ""
    portfolio_impact_pct: float = 0.0
    worst_position: str = ""
    worst_position_impact_pct: float = 0.0
    survival: bool = True  # max_drawdown 내 생존 여부
    recommendations: list = field(default_factory=list)


@dataclass
class RiskDashboard:
    """리스크 대시보드 (종합)"""

    timestamp: str = ""
    tail_risk: TailRiskAlert = field(default_factory=TailRiskAlert)
    correlation: CorrelationRegime = field(default_factory=CorrelationRegime)
    stress_tests: list = field(default_factory=list)  # [StressTestResult]
    portfolio_var_95: float = 0.0  # 95% 일간 VaR
    portfolio_cvar_95: float = 0.0  # 95% CVaR (Expected Shortfall)
    overall_threat: str = "green"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tail_risk": self.tail_risk.to_dict(),
            "correlation": self.correlation.__dict__,
            "stress_tests": [s.__dict__ for s in self.stress_tests],
            "portfolio_var_95": self.portfolio_var_95,
            "portfolio_cvar_95": self.portfolio_cvar_95,
            "overall_threat": self.overall_threat,
        }
