"""AI 스코어링 어댑터 - Claude API로 100점 종합 분석"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.models import (
    AnalysisScore,
    ChartData,
    InvestorFlow,
    ScoreCategory,
    ScoreDetail,
    Stock,
)
from src.use_cases.ports import AIAnalysisPort

SYSTEM_PROMPT = """당신은 주식 종합 분석 전문가입니다.
주어진 데이터를 기반으로 100점 만점 스코어링을 수행합니다.

4개 카테고리로 채점하세요:
1. 기술적 분석 (35점 만점): 추세 분석(15) + 지표 신호(10) + 패턴 인식(10)
2. 수급 분석 (30점 만점): 외국인 동향(12) + 기관 동향(10) + 수급 추세(8)
3. 매매기법 (20점 만점): 진입 타이밍(10) + 리스크 관리(10)
4. 이슈/모멘텀 (15점 만점): 시장 환경(8) + 종목 모멘텀(7)

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "categories": [
    {
      "name": "기술적 분석",
      "score": 25,
      "max_score": 35,
      "details": [
        {"name": "추세 분석", "score": 12, "max_score": 15, "comment": "상승 추세 유지"},
        {"name": "지표 신호", "score": 7, "max_score": 10, "comment": "RSI 중립"},
        {"name": "패턴 인식", "score": 6, "max_score": 10, "comment": "망치형 출현"}
      ]
    },
    {
      "name": "수급 분석",
      "score": 20,
      "max_score": 30,
      "details": [
        {"name": "외국인 동향", "score": 8, "max_score": 12, "comment": "순매수 전환"},
        {"name": "기관 동향", "score": 7, "max_score": 10, "comment": "소폭 매도"},
        {"name": "수급 추세", "score": 5, "max_score": 8, "comment": "개선 중"}
      ]
    },
    {
      "name": "매매기법",
      "score": 14,
      "max_score": 20,
      "details": [
        {"name": "진입 타이밍", "score": 7, "max_score": 10, "comment": "지지선 근처"},
        {"name": "리스크 관리", "score": 7, "max_score": 10, "comment": "손절라인 명확"}
      ]
    },
    {
      "name": "이슈/모멘텀",
      "score": 10,
      "max_score": 15,
      "details": [
        {"name": "시장 환경", "score": 5, "max_score": 8, "comment": "시장 보합세"},
        {"name": "종목 모멘텀", "score": 5, "max_score": 7, "comment": "실적 개선 기대"}
      ]
    }
  ],
  "summary": "종합 평가 요약 (2~3문장)",
  "recommendation": "투자 권고 (매수/관망/매도 등)"
}
```"""


def _format_scoring_input(
    stock: Stock,
    chart_data: ChartData,
    investor_flow: InvestorFlow | None,
) -> str:
    """스코어링 입력 데이터를 프롬프트용 텍스트로 변환"""
    latest = chart_data.latest
    ind = chart_data.indicators
    price_info = f"현재가: {latest.close:,.0f}" if latest else "현재가: 없음"

    lines = [
        f"종목: {stock.name} ({stock.ticker})",
        f"시장: {stock.market.value} / 업종: {stock.sector}",
        price_info,
        "",
        "[기술 지표]",
        f"- RSI: {f'{ind.rsi:.1f}' if ind.rsi else 'N/A'}",
        f"- MACD: {f'{ind.macd:.2f}' if ind.macd else 'N/A'}",
        f"- MACD Signal: {f'{ind.macd_signal:.2f}' if ind.macd_signal else 'N/A'}",
        f"- 볼린저 상단: {f'{ind.bollinger_upper:,.0f}' if ind.bollinger_upper else 'N/A'}",
        f"- 볼린저 하단: {f'{ind.bollinger_lower:,.0f}' if ind.bollinger_lower else 'N/A'}",
        f"- 스토캐스틱 K: {f'{ind.stochastic_k:.1f}' if ind.stochastic_k else 'N/A'}",
        f"- MA5: {f'{ind.ma5:,.0f}' if ind.ma5 else 'N/A'}",
        f"- MA20: {f'{ind.ma20:,.0f}' if ind.ma20 else 'N/A'}",
        f"- MA60: {f'{ind.ma60:,.0f}' if ind.ma60 else 'N/A'}",
        f"- MA120: {f'{ind.ma120:,.0f}' if ind.ma120 else 'N/A'}",
    ]

    # 최근 5일 캔들
    if chart_data.candles:
        lines.extend(["", "[최근 5일 가격]"])
        for c in chart_data.candles[-5:]:
            lines.append(
                f"- {c.date}: 시가 {c.open:,.0f} / 고가 {c.high:,.0f} / "
                f"저가 {c.low:,.0f} / 종가 {c.close:,.0f} / 거래량 {c.volume:,}"
            )

    # 수급 데이터
    if investor_flow:
        lines.extend([
            "",
            "[수급 데이터]",
            f"- 외국인 순매수: {investor_flow.foreign_net:,}",
            f"- 기관 순매수: {investor_flow.inst_net:,}",
            f"- 개인 순매수: {investor_flow.individual_net:,}",
        ])
        if investor_flow.foreign_holding_ratio is not None:
            lines.append(f"- 외국인 보유비율: {investor_flow.foreign_holding_ratio:.1f}%")
    else:
        lines.extend(["", "[수급 데이터]", "- 수급 데이터 없음 (네이버 금융 미조회)"])

    return "\n".join(lines)


class ClaudeScoringAdapter(BaseAgent, AIAnalysisPort):
    """Claude API를 이용한 100점 스코어링 어댑터"""

    async def analyze(
        self,
        stock: Stock,
        chart_data: ChartData,
        investor_flow: InvestorFlow | None = None,
    ) -> AnalysisScore:
        user_prompt = _format_scoring_input(stock, chart_data, investor_flow)
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        categories = []
        for cat in data.get("categories", []):
            details = [
                ScoreDetail(
                    name=d.get("name", ""),
                    score=float(d.get("score", 0)),
                    max_score=float(d.get("max_score", 0)),
                    comment=d.get("comment", ""),
                )
                for d in cat.get("details", [])
            ]
            categories.append(ScoreCategory(
                name=cat.get("name", ""),
                score=float(cat.get("score", 0)),
                max_score=float(cat.get("max_score", 0)),
                details=details,
            ))

        return AnalysisScore(
            categories=categories,
            summary=data.get("summary", ""),
            recommendation=data.get("recommendation", ""),
        )
