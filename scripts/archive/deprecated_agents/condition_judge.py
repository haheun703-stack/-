"""조건판단 서브에이전트 - 유지/대응 조건 생성"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.models import (
    ChartData,
    Condition,
    ConditionType,
    FlowPrediction,
    Stock,
    TechnicalPattern,
    VolumeAnalysis,
)
from src.use_cases.ports import ConditionJudgePort

SYSTEM_PROMPT = """당신은 주식 포지션 유지/대응 조건 판단 전문가입니다.
기술적 분석, 거래량 분석, 흐름 예측 결과를 종합하여
투자자가 따라야 할 유지조건과 대응조건을 생성합니다.

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "conditions": [
    {
      "type": "유지|대응",
      "title": "조건 제목",
      "description": "상세 설명",
      "trigger_price": 50000,
      "priority": 1,
      "confidence": 0.85
    }
  ]
}
```

유지조건: 현재 포지션을 유지해도 되는 근거 (지지대 유지, 정배열, RSI 정상범위 등)
대응조건: 즉시 행동이 필요한 상황 (지지대 이탈→손절, 목표가→익절, 급락→긴급매도 등)"""


def _format_judge_input(
    stock: Stock,
    chart_data: ChartData,
    pattern: TechnicalPattern,
    volume: VolumeAnalysis,
    prediction: FlowPrediction,
) -> str:
    """판단 입력 데이터를 프롬프트용 텍스트로 변환"""
    latest = chart_data.latest
    price_info = f"현재가: {latest.close:,.0f}" if latest else "현재가: 없음"

    lines = [
        f"종목: {stock.name} ({stock.ticker})",
        price_info,
        "",
        "[기술적 분석]",
        f"- 추세: {pattern.overall_trend.value} / 강도: {pattern.strength.value}",
        f"- 이평선: {pattern.ma_alignment.value}",
        f"- RSI: {pattern.rsi_signal}",
        f"- MACD: {pattern.macd_signal}",
        "",
        "[거래량 분석]",
        f"- 거래량 비율: {volume.avg_volume_ratio:.1f}배",
        f"- 매집/분산: {volume.accumulation_signal}",
    ]

    if volume.zones:
        for z in volume.zones:
            lines.append(f"- {z.zone_type}: {z.price_low:,.0f}~{z.price_high:,.0f}")

    lines.extend([
        "",
        "[흐름 예측]",
        f"- 방향: {prediction.direction.value} (확신도: {prediction.confidence:.0%})",
        f"- 예상 범위: {prediction.price_low:,.0f} ~ {prediction.price_high:,.0f}",
        f"- 요약: {prediction.summary}",
    ])

    return "\n".join(lines)


class ConditionJudgeAgent(BaseAgent, ConditionJudgePort):
    """⚖️ 조건판단 에이전트 - ConditionJudgePort 구현"""

    async def judge(
        self,
        stock: Stock,
        chart_data: ChartData,
        pattern: TechnicalPattern,
        volume: VolumeAnalysis,
        prediction: FlowPrediction,
    ) -> list[Condition]:
        user_prompt = _format_judge_input(stock, chart_data, pattern, volume, prediction)
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        return [
            Condition(
                condition_type=ConditionType.HOLD if c.get("type") == "유지" else ConditionType.ACTION,
                title=c.get("title", ""),
                description=c.get("description", ""),
                trigger_price=c.get("trigger_price"),
                priority=c.get("priority", 1),
                confidence=c.get("confidence", 0.5),
            )
            for c in data.get("conditions", [])
        ]
