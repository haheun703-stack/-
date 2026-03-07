"""흐름예측 서브에이전트 - 내일 흐름 종합 예측"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.models import (
    ChartData,
    Direction,
    FlowPrediction,
    Stock,
    TechnicalPattern,
    VolumeAnalysis,
)
from src.use_cases.ports import FlowPredictionPort

SYSTEM_PROMPT = """당신은 한국 주식시장 흐름 예측 전문가입니다.
기술적 분석과 거래량 분석 결과를 종합하여 내일 흐름을 예측합니다.

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "direction": "상승|하락|횡보",
  "confidence": 0.75,
  "price_low": 50000,
  "price_high": 53000,
  "key_factors": ["요인1", "요인2"],
  "summary": "종합 예측 요약"
}
```"""


def _format_prediction_input(
    stock: Stock,
    chart_data: ChartData,
    pattern: TechnicalPattern,
    volume: VolumeAnalysis,
) -> str:
    """예측 입력 데이터를 프롬프트용 텍스트로 변환"""
    latest = chart_data.latest
    price_info = f"현재가: {latest.close:,.0f}" if latest else "현재가: 없음"

    lines = [
        f"종목: {stock.name} ({stock.ticker}) / {stock.market.value}",
        price_info,
        "",
        "[기술적 분석 결과]",
        f"- 캔들 패턴: {pattern.candle_pattern}",
        f"- 이평선 배열: {pattern.ma_alignment.value}",
        f"- RSI: {pattern.rsi_signal}",
        f"- MACD: {pattern.macd_signal}",
        f"- 볼린저: {pattern.bollinger_signal}",
        f"- 종합 추세: {pattern.overall_trend.value} (강도: {pattern.strength.value})",
        f"- 핵심 포인트: {', '.join(pattern.key_points)}",
        "",
        "[거래량 분석 결과]",
        f"- 평균 거래량 대비: {volume.avg_volume_ratio:.1f}배",
        f"- 거래량 추세: {volume.volume_trend.value}",
        f"- 매집/분산: {volume.accumulation_signal}",
    ]

    if volume.zones:
        lines.append("- 매물대:")
        for z in volume.zones:
            lines.append(f"  {z.zone_type} {z.price_low:,.0f}~{z.price_high:,.0f} (강도: {z.strength.value})")

    lines.append(f"- 핵심 포인트: {', '.join(volume.key_points)}")

    return "\n".join(lines)


_DIRECTION_MAP = {"상승": Direction.UP, "하락": Direction.DOWN}


class FlowPredictionAgent(BaseAgent, FlowPredictionPort):
    """🔮 흐름예측 에이전트 - FlowPredictionPort 구현"""

    async def predict(
        self,
        stock: Stock,
        chart_data: ChartData,
        pattern: TechnicalPattern,
        volume: VolumeAnalysis,
    ) -> FlowPrediction:
        user_prompt = _format_prediction_input(stock, chart_data, pattern, volume)
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        return FlowPrediction(
            direction=_DIRECTION_MAP.get(data.get("direction", "횡보"), Direction.SIDEWAYS),
            confidence=data.get("confidence", 0.5),
            price_low=data.get("price_low", 0),
            price_high=data.get("price_high", 0),
            key_factors=data.get("key_factors", []),
            summary=data.get("summary", ""),
        )
