"""Risk Sentinel 서브에이전트 — 꼬리 리스크 감시 및 스트레스 테스트

VKOSPI 모니터링, 상관관계 급등, 외국인 대량 매도, 서킷브레이커 위험,
시장 폭(breadth) 등을 종합 분석하여 포트폴리오 리스크 경보를 생성한다.
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.risk_models import (
    CorrelationRegime,
    StressTestResult,
    TailRiskAlert,
)
from src.use_cases.ports import RiskSentinelPort

# ─── 시스템 프롬프트 ──────────────────────────────────────────

SYSTEM_PROMPT = """당신은 한국 주식시장 리스크 관리 전문가(Risk Sentinel)입니다.
포트폴리오의 꼬리 리스크(tail risk)를 감시하고, 상관관계 급등·블랙스완 이벤트를
조기에 탐지하여 경보를 발령합니다.

핵심 감시 영역:
1. VKOSPI(변동성지수): 20 이상 주의, 30 이상 경계, 40 이상 위험
2. 상관관계 급등: 평균 쌍별 상관 > 0.7 이면 분산 효과 소멸 경고
3. 외국인 대량 매도: 5일 연속 순매도 또는 단일일 -3000억 이상
4. 서킷브레이커 위험: KOSPI -8% 이상 급락 시 발동 가능성
5. 시장 폭(Market Breadth): 상승종목비율 20% 미만이면 극단 하락

위협 수준 기준:
- green: 모든 지표 정상
- yellow: 1개 지표 경고
- orange: 2개 이상 지표 경고
- red: 3개 이상 또는 VKOSPI 40 이상
- black: 서킷브레이커 발동 수준

반드시 아래 JSON 형식으로만 응답하세요."""

# ─── 꼬리 리스크 스캔 프롬프트 ──────────────────────────────────

TAIL_RISK_JSON_SPEC = """
응답 형식:
```json
{
  "threat_level": "green|yellow|orange|red|black",
  "vkospi_level": 18.5,
  "vkospi_change_pct": 2.3,
  "correlation_spike": false,
  "avg_correlation": 0.35,
  "market_breadth": 0.55,
  "foreign_flow_signal": "neutral|sell|massive_sell|buy",
  "circuit_breaker_risk": false,
  "alerts": ["경고 메시지1", "경고 메시지2"],
  "recommended_action": "hold|reduce|hedge|emergency_exit",
  "reasoning": "종합 판단 근거 서술"
}
```"""

# ─── 상관관계 분석 프롬프트 ──────────────────────────────────

CORRELATION_JSON_SPEC = """
응답 형식:
```json
{
  "avg_pairwise_corr": 0.35,
  "max_pairwise_corr": 0.82,
  "regime": "normal|elevated|crisis",
  "most_correlated_pair": ["종목A", "종목B"],
  "diversification_ratio": 1.45,
  "warnings": ["경고 메시지1"]
}
```"""

# ─── 스트레스 테스트 프롬프트 ──────────────────────────────────

STRESS_TEST_JSON_SPEC = """
각 시나리오에 대해 아래 형식으로 응답하세요:
```json
[
  {
    "scenario_name": "시나리오 이름",
    "scenario_description": "시나리오 상세 설명",
    "portfolio_impact_pct": -8.5,
    "worst_position": "종목코드",
    "worst_position_impact_pct": -15.2,
    "survival": true,
    "recommendations": ["대응 조치1", "대응 조치2"]
  }
]
```"""

# ─── 기본 스트레스 시나리오 ──────────────────────────────────

DEFAULT_SCENARIOS = [
    {
        "name": "KOSPI -5% 급락",
        "description": "2020년 3월 COVID 수준의 시장 급락. KOSPI가 하루 만에 5% 하락하고, "
        "VKOSPI 40 돌파, 외국인 대량 매도 동반.",
    },
    {
        "name": "반도체 섹터 -15%",
        "description": "반도체 섹터 쇼크. 삼성전자·SK하이닉스 등 반도체 대형주가 -15% 급락하고 "
        "관련 중소형주까지 연쇄 하락.",
    },
    {
        "name": "외국인 5일 연속 순매도",
        "description": "외국인 투자자가 5거래일 연속 KOSPI·KOSDAQ 순매도. "
        "누적 순매도 금액 1조 5천억 원 이상.",
    },
    {
        "name": "금리 50bp 인상",
        "description": "한국은행 기준금리 50bp 인상 충격. 성장주·부채비율 높은 종목 급락, "
        "은행·보험주 상대적 상승.",
    },
]


# ─── 포매팅 함수 ──────────────────────────────────────────────


def _format_market_data(market_data: dict, portfolio: dict) -> str:
    """시장 데이터와 포트폴리오를 꼬리 리스크 분석용 텍스트로 변환"""
    lines = []

    # 시장 지표
    lines.append("=== 시장 현황 ===")
    lines.append(f"KOSPI: {market_data.get('kospi', 'N/A')}")
    lines.append(f"KOSDAQ: {market_data.get('kosdaq', 'N/A')}")
    lines.append(f"VKOSPI: {market_data.get('vkospi', 'N/A')}")
    lines.append(f"VKOSPI 전일대비(%): {market_data.get('vkospi_change_pct', 'N/A')}")
    lines.append(f"원/달러 환율: {market_data.get('usd_krw', 'N/A')}")
    lines.append(f"시장 상승종목 비율: {market_data.get('market_breadth', 'N/A')}")

    # 외국인 수급
    foreign = market_data.get("foreign_flow", {})
    lines.append("\n=== 외국인 수급 ===")
    lines.append(f"오늘 순매수(억): {foreign.get('today', 'N/A')}")
    lines.append(f"5일 누적 순매수(억): {foreign.get('5d_cumulative', 'N/A')}")
    lines.append(f"연속 매도 일수: {foreign.get('consecutive_sell_days', 0)}")

    # 포트폴리오 현황
    lines.append("\n=== 포트폴리오 현황 ===")
    lines.append(f"총 평가금액: {portfolio.get('total_value', 'N/A')}")
    lines.append(f"현금 비중(%): {portfolio.get('cash_ratio', 'N/A')}")

    holdings = portfolio.get("holdings", [])
    if holdings:
        lines.append("\n[보유 종목]")
        for h in holdings:
            lines.append(
                f"  {h.get('name', '?')}({h.get('ticker', '?')}) | "
                f"비중: {h.get('weight_pct', '?')}% | "
                f"수익률: {h.get('return_pct', '?')}% | "
                f"섹터: {h.get('sector', '?')}"
            )

    return "\n".join(lines)


def _format_returns_matrix(returns_matrix: dict) -> str:
    """수익률 매트릭스를 상관관계 분석용 텍스트로 변환"""
    lines = ["=== 포트폴리오 수익률 매트릭스 ==="]

    tickers = list(returns_matrix.keys())
    lines.append(f"종목 수: {len(tickers)}")
    lines.append(f"종목: {', '.join(tickers)}")

    lines.append("\n[일간 수익률 (최근 데이터)]")
    for ticker, returns in returns_matrix.items():
        if isinstance(returns, list):
            recent = returns[-20:] if len(returns) >= 20 else returns
            formatted = [f"{r:+.2f}%" for r in recent]
            lines.append(f"  {ticker}: {', '.join(formatted)}")

    return "\n".join(lines)


def _format_stress_scenarios(portfolio: dict, scenarios: list) -> str:
    """포트폴리오와 스트레스 시나리오를 분석용 텍스트로 변환"""
    lines = ["=== 스트레스 테스트 대상 포트폴리오 ==="]
    lines.append(f"총 평가금액: {portfolio.get('total_value', 'N/A')}")
    lines.append(f"최대 허용 낙폭: {portfolio.get('max_drawdown_pct', -20)}%")

    holdings = portfolio.get("holdings", [])
    if holdings:
        lines.append("\n[보유 종목]")
        for h in holdings:
            lines.append(
                f"  {h.get('name', '?')}({h.get('ticker', '?')}) | "
                f"비중: {h.get('weight_pct', '?')}% | "
                f"섹터: {h.get('sector', '?')} | "
                f"베타: {h.get('beta', 1.0)}"
            )

    lines.append("\n=== 스트레스 시나리오 ===")
    for i, s in enumerate(scenarios, 1):
        lines.append(f"\n[시나리오 {i}] {s.get('name', '?')}")
        lines.append(f"  설명: {s.get('description', '?')}")

    return "\n".join(lines)


# ─── 에이전트 구현 ──────────────────────────────────────────


class RiskSentinelAgent(BaseAgent, RiskSentinelPort):
    """리스크 감시 에이전트 — 꼬리 리스크 탐지 및 스트레스 테스트 수행

    RiskSentinelPort를 구현하며, Claude API를 통해
    시장 상황을 종합 판단하고 리스크 경보를 생성한다.
    """

    async def scan_tail_risk(self, market_data: dict, portfolio: dict) -> dict:
        """꼬리 리스크 스캔 — 시장 데이터와 포트폴리오를 분석하여 경보 생성

        Args:
            market_data: 시장 지표 (KOSPI, VKOSPI, 외국인 수급 등)
            portfolio: 포트폴리오 현황 (보유 종목, 비중 등)

        Returns:
            TailRiskAlert 딕셔너리
        """
        user_prompt = _format_market_data(market_data, portfolio) + "\n" + TAIL_RISK_JSON_SPEC
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        alert = TailRiskAlert(
            threat_level=data.get("threat_level", "green"),
            vkospi_level=float(data.get("vkospi_level", 0)),
            vkospi_change_pct=float(data.get("vkospi_change_pct", 0)),
            correlation_spike=bool(data.get("correlation_spike", False)),
            avg_correlation=float(data.get("avg_correlation", 0)),
            market_breadth=float(data.get("market_breadth", 0.5)),
            foreign_flow_signal=data.get("foreign_flow_signal", "neutral"),
            circuit_breaker_risk=bool(data.get("circuit_breaker_risk", False)),
            alerts=data.get("alerts", []),
            recommended_action=data.get("recommended_action", "hold"),
            reasoning=data.get("reasoning", ""),
        )
        return alert.to_dict()

    async def analyze_correlation(self, returns_matrix: dict) -> dict:
        """포트폴리오 상관관계 분석 — 수익률 매트릭스에서 레짐 판별

        Args:
            returns_matrix: {종목코드: [일간수익률]} 형태의 딕셔너리

        Returns:
            CorrelationRegime 딕셔너리
        """
        user_prompt = _format_returns_matrix(returns_matrix) + "\n" + CORRELATION_JSON_SPEC
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        pair_raw = data.get("most_correlated_pair", ["", ""])
        if isinstance(pair_raw, (list, tuple)) and len(pair_raw) >= 2:
            pair = (str(pair_raw[0]), str(pair_raw[1]))
        else:
            pair = ("", "")

        regime = CorrelationRegime(
            avg_pairwise_corr=float(data.get("avg_pairwise_corr", 0)),
            max_pairwise_corr=float(data.get("max_pairwise_corr", 0)),
            regime=data.get("regime", "normal"),
            most_correlated_pair=pair,
            diversification_ratio=float(data.get("diversification_ratio", 1.0)),
            warnings=data.get("warnings", []),
        )
        return regime.__dict__

    async def stress_test(self, portfolio: dict, scenarios: list) -> list:
        """스트레스 테스트 실행 — 시나리오별 포트폴리오 영향 분석

        시나리오가 주어지지 않으면 기본 4대 시나리오를 사용한다:
        1. KOSPI -5% 급락 (2020.03 COVID 수준)
        2. 반도체 섹터 -15% (섹터 쇼크)
        3. 외국인 5일 연속 순매도 (외국인 이탈)
        4. 금리 50bp 인상 (통화긴축)

        Args:
            portfolio: 포트폴리오 현황
            scenarios: 스트레스 시나리오 목록 (비어 있으면 기본값 사용)

        Returns:
            [StressTestResult 딕셔너리] 리스트
        """
        if not scenarios:
            scenarios = DEFAULT_SCENARIOS

        user_prompt = _format_stress_scenarios(portfolio, scenarios) + "\n" + STRESS_TEST_JSON_SPEC
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        # Claude가 리스트를 반환하지 않은 경우 방어
        if isinstance(data, dict):
            data = [data]

        results = []
        for item in data:
            result = StressTestResult(
                scenario_name=item.get("scenario_name", ""),
                scenario_description=item.get("scenario_description", ""),
                portfolio_impact_pct=float(item.get("portfolio_impact_pct", 0)),
                worst_position=item.get("worst_position", ""),
                worst_position_impact_pct=float(item.get("worst_position_impact_pct", 0)),
                survival=bool(item.get("survival", True)),
                recommendations=item.get("recommendations", []),
            )
            results.append(result.__dict__)

        return results
