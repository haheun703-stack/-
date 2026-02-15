"""CFO 서브에이전트 — 포트폴리오 리스크 관리 및 자본 배분

C-suite 레벨 에이전트: 개별 종목이 아닌 포트폴리오 전체를 관리.
Kelly Criterion, 상관관계 패널티, 낙폭 관리를 Claude에게 위임.
"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.cfo_models import (
    CapitalAllocation,
    DrawdownAnalysis,
    PortfolioHealthCheck,
    PortfolioRiskBudget,
)
from src.use_cases.ports import CFOPort

SYSTEM_PROMPT = """당신은 한국 주식시장 전문 CFO(최고재무관리자)입니다.
포트폴리오 전체의 리스크를 관리하고 자본 배분을 결정합니다.

핵심 원칙:
1. Kelly Criterion 기반 포지션 사이징 (Quarter-Kelly 적용)
2. 상관관계 높은 종목 집중 방지 (업종 분산)
3. 낙폭 관리 — MDD 15% 초과 시 포지션 축소
4. 단일 종목 최대 20%, 포트폴리오 리스크 최대 6%
5. 현금 비중 30% 이상 유지 (기회 대비)

반드시 아래 JSON 형식으로 응답하세요."""

ALLOCATE_PROMPT = SYSTEM_PROMPT + """
자본 배분 결정을 해주세요:
```json
{
  "recommended_size_pct": 0.15,
  "recommended_amount": 4500000,
  "kelly_fraction": 0.12,
  "risk_adjusted_size": 0.10,
  "max_allowed": 0.20,
  "reasoning": "배분 근거 설명",
  "correlation_penalty": 0.02,
  "drawdown_penalty": 0.0,
  "concentration_penalty": 0.03
}
```"""

HEALTH_PROMPT = SYSTEM_PROMPT + """
포트폴리오 건강 진단을 해주세요:
```json
{
  "overall_score": 75,
  "risk_level": "moderate",
  "sector_concentration": 0.35,
  "top_holding_pct": 0.25,
  "estimated_var_95": -0.025,
  "max_correlated_exposure": 0.40,
  "warnings": ["경고1", "경고2"],
  "recommendations": ["권고1", "권고2"]
}
```"""

DRAWDOWN_PROMPT = SYSTEM_PROMPT + """
낙폭 분석 및 대응 방안을 제시해주세요:
```json
{
  "current_drawdown_pct": -0.08,
  "max_drawdown_pct": -0.12,
  "drawdown_duration_days": 15,
  "recovery_estimate_days": 30,
  "action": "reduce",
  "reasoning": "대응 근거 설명"
}
```
action 옵션: "normal" (유지), "reduce" (비중 축소), "halt" (신규 매수 중단), "emergency" (전량 청산)"""


def _format_allocation_input(signal: dict, portfolio: dict, risk_budget: dict) -> str:
    """자본 배분 요청을 프롬프트용 텍스트로 변환"""
    lines = ["[신규 진입 시그널]"]
    lines.append(f"종목: {signal.get('ticker', '?')}")
    lines.append(f"등급: {signal.get('grade', 'F')}")
    lines.append(f"Zone Score: {signal.get('zone_score', 0):.3f}")
    lines.append(f"트리거: {signal.get('trigger_type', 'none')}")
    lines.append(f"손익비: {signal.get('risk_reward_ratio', 0):.1f}")
    lines.append(f"ATR: {signal.get('atr_value', 0):.0f}")
    lines.append(f"진입가: {signal.get('entry_price', 0):,.0f}")
    lines.append(f"손절가: {signal.get('stop_loss', 0):,.0f}")

    lines.append("\n[현재 포트폴리오]")
    positions = portfolio.get("positions", [])
    lines.append(f"보유 종목 수: {len(positions)}")
    for pos in positions:
        lines.append(
            f"  {pos.get('ticker', '?')}: {pos.get('weight_pct', 0):.1f}% "
            f"(수익률: {pos.get('pnl_pct', 0):+.1f}%)"
        )

    lines.append(f"\n총 자본: {portfolio.get('total_capital', 0):,.0f}원")
    lines.append(f"현금: {portfolio.get('cash', 0):,.0f}원")
    lines.append(f"투자금: {portfolio.get('invested', 0):,.0f}원")

    lines.append("\n[리스크 예산]")
    lines.append(f"포트폴리오 리스크: {risk_budget.get('portfolio_risk_pct', 0):.1%}")
    lines.append(f"현재 낙폭: {risk_budget.get('current_drawdown', 0):.1%}")
    lines.append(f"최대 허용 낙폭: {risk_budget.get('max_drawdown', -0.15):.0%}")
    lines.append(f"일일 손실: {risk_budget.get('daily_loss', 0):.1%}")

    return "\n".join(lines)


def _format_health_input(portfolio: dict, market_context: dict) -> str:
    """포트폴리오 건강 진단 요청을 프롬프트용 텍스트로 변환"""
    lines = ["[포트폴리오 현황]"]
    lines.append(f"총 자본: {portfolio.get('total_capital', 0):,.0f}원")
    lines.append(f"현금 비중: {portfolio.get('cash_pct', 0):.0%}")

    positions = portfolio.get("positions", [])
    lines.append(f"보유 종목 수: {len(positions)}")

    sectors: dict[str, float] = {}
    for pos in positions:
        sec = pos.get("sector", "기타")
        sectors[sec] = sectors.get(sec, 0) + pos.get("weight_pct", 0)

    if sectors:
        lines.append("\n[업종별 비중]")
        for sec, pct in sorted(sectors.items(), key=lambda x: -x[1]):
            lines.append(f"  {sec}: {pct:.1f}%")

    lines.append("\n[보유 종목 상세]")
    for pos in positions:
        lines.append(
            f"  {pos.get('ticker', '?')} ({pos.get('name', '')}): "
            f"비중 {pos.get('weight_pct', 0):.1f}%, "
            f"수익률 {pos.get('pnl_pct', 0):+.1f}%, "
            f"보유일 {pos.get('hold_days', 0)}일"
        )

    lines.append("\n[시장 컨텍스트]")
    lines.append(
        f"KOSPI: {market_context.get('kospi', 0):,.0f} "
        f"({market_context.get('kospi_change_pct', 0):+.1f}%)"
    )
    lines.append(f"시장 레짐: {market_context.get('regime', '알 수 없음')}")

    return "\n".join(lines)


def _format_drawdown_input(equity_curve: list, current_positions: list) -> str:
    """낙폭 분석 요청을 프롬프트용 텍스트로 변환"""
    lines = ["[자산 곡선 (최근 60일)]"]
    for point in equity_curve[-60:]:
        lines.append(f"  {point.get('date', '')}: {point.get('equity', 0):,.0f}원")

    if equity_curve:
        peak = max(p.get("equity", 0) for p in equity_curve)
        current = equity_curve[-1].get("equity", 0)
        dd = (current / peak - 1) if peak > 0 else 0
        lines.append(f"\n고점: {peak:,.0f}원")
        lines.append(f"현재: {current:,.0f}원")
        lines.append(f"낙폭: {dd:.1%}")

    lines.append("\n[현재 포지션]")
    for pos in current_positions:
        lines.append(
            f"  {pos.get('ticker', '?')}: "
            f"수익률 {pos.get('pnl_pct', 0):+.1f}%, "
            f"비중 {pos.get('weight_pct', 0):.1f}%"
        )

    return "\n".join(lines)


class CFOAgent(BaseAgent, CFOPort):
    """CFO 에이전트 — 포트폴리오 리스크 관리 및 자본 배분"""

    async def allocate_capital(
        self, signal: dict, portfolio: dict, risk_budget: dict
    ) -> CapitalAllocation:
        """신규 진입 시 자본 배분 결정"""
        user_prompt = _format_allocation_input(signal, portfolio, risk_budget)
        data = await self._ask_claude_json(ALLOCATE_PROMPT, user_prompt)

        return CapitalAllocation(
            ticker=signal.get("ticker", ""),
            recommended_size_pct=data.get("recommended_size_pct", 0.0),
            recommended_amount=data.get("recommended_amount", 0),
            kelly_fraction=data.get("kelly_fraction", 0.0),
            risk_adjusted_size=data.get("risk_adjusted_size", 0.0),
            max_allowed=data.get("max_allowed", 0.20),
            reasoning=data.get("reasoning", ""),
            correlation_penalty=data.get("correlation_penalty", 0.0),
            drawdown_penalty=data.get("drawdown_penalty", 0.0),
            concentration_penalty=data.get("concentration_penalty", 0.0),
        )

    async def health_check(
        self, portfolio: dict, market_context: dict
    ) -> PortfolioHealthCheck:
        """포트폴리오 건강 진단"""
        user_prompt = _format_health_input(portfolio, market_context)
        data = await self._ask_claude_json(HEALTH_PROMPT, user_prompt)

        return PortfolioHealthCheck(
            overall_score=data.get("overall_score", 50),
            risk_level=data.get("risk_level", "moderate"),
            positions_count=len(portfolio.get("positions", [])),
            sector_concentration=data.get("sector_concentration", 0.0),
            top_holding_pct=data.get("top_holding_pct", 0.0),
            estimated_var_95=data.get("estimated_var_95", 0.0),
            max_correlated_exposure=data.get("max_correlated_exposure", 0.0),
            warnings=data.get("warnings", []),
            recommendations=data.get("recommendations", []),
        )

    async def drawdown_analysis(
        self, equity_curve: list, current_positions: list
    ) -> DrawdownAnalysis:
        """낙폭 분석 및 대응 방안"""
        user_prompt = _format_drawdown_input(equity_curve, current_positions)
        data = await self._ask_claude_json(DRAWDOWN_PROMPT, user_prompt)

        return DrawdownAnalysis(
            current_drawdown_pct=data.get("current_drawdown_pct", 0.0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            drawdown_duration_days=data.get("drawdown_duration_days", 0),
            recovery_estimate_days=data.get("recovery_estimate_days", 0),
            action=data.get("action", "normal"),
            reasoning=data.get("reasoning", ""),
        )
