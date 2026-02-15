"""CFO 에이전트 엔티티 — 포트폴리오 리스크 관리 및 자본 배분

C-suite 레벨 의사결정에 필요한 핵심 비즈니스 객체.
외부 의존 없는 순수 데이터 클래스로 구성.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """포트폴리오 리스크 수준"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EMERGENCY = "emergency"


@dataclass
class PortfolioRiskBudget:
    """포트폴리오 리스크 예산"""
    total_capital: float = 0.0
    available_cash: float = 0.0
    invested_amount: float = 0.0
    max_single_position_pct: float = 0.20
    max_portfolio_risk_pct: float = 0.06
    max_daily_loss_pct: float = 0.03
    current_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    risk_level: str = "moderate"
    utilization_pct: float = 0.0  # invested / total

    @property
    def remaining_risk_budget(self) -> float:
        """남은 리스크 예산 (최대 포트폴리오 리스크 - 현재 낙폭)"""
        return max(self.max_portfolio_risk_pct - abs(self.current_drawdown_pct), 0)


@dataclass
class CapitalAllocation:
    """자본 배분 결정"""
    ticker: str = ""
    recommended_size_pct: float = 0.0
    recommended_amount: float = 0.0
    kelly_fraction: float = 0.0
    risk_adjusted_size: float = 0.0
    max_allowed: float = 0.0
    reasoning: str = ""
    # 방어 조건
    correlation_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    concentration_penalty: float = 0.0


@dataclass
class PortfolioHealthCheck:
    """포트폴리오 건강 진단"""
    overall_score: float = 0.0  # 0~100
    risk_level: str = "moderate"
    positions_count: int = 0
    sector_concentration: float = 0.0
    top_holding_pct: float = 0.0
    estimated_var_95: float = 0.0  # 95% VaR
    max_correlated_exposure: float = 0.0
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "positions_count": self.positions_count,
            "sector_concentration": self.sector_concentration,
            "top_holding_pct": self.top_holding_pct,
            "estimated_var_95": self.estimated_var_95,
            "max_correlated_exposure": self.max_correlated_exposure,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


@dataclass
class DrawdownAnalysis:
    """낙폭 분석"""
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    drawdown_duration_days: int = 0
    recovery_estimate_days: int = 0
    action: str = "normal"  # normal / reduce / halt / emergency
    reasoning: str = ""
