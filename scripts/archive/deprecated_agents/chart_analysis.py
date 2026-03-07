"""차트분석 서브에이전트 - 기술적 패턴/지표 분석"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.models import (
    ChartData,
    Direction,
    MovingAverageAlignment,
    SignalStrength,
    Stock,
    TechnicalPattern,
)
from src.use_cases.ports import ChartAnalysisPort

SYSTEM_PROMPT = """당신은 한국 주식시장 기술적 분석 전문가입니다.
주어진 차트 데이터를 분석하여 기술적 패턴과 지표를 해석합니다.

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "candle_pattern": "캔들 패턴 이름",
  "ma_alignment": "정배열|역배열|혼조",
  "rsi_signal": "RSI 해석",
  "macd_signal": "MACD 해석",
  "bollinger_signal": "볼린저밴드 해석",
  "stochastic_signal": "스토캐스틱 해석",
  "overall_trend": "상승|하락|횡보",
  "strength": "강함|보통|약함",
  "key_points": ["핵심 포인트1", "핵심 포인트2"]
}
```"""


def _format_chart_data(stock: Stock, chart_data: ChartData) -> str:
    """차트 데이터를 프롬프트용 텍스트로 변환"""
    lines = [f"종목: {stock.name} ({stock.ticker}) / {stock.market.value} / {stock.sector}"]

    # 최근 20일 캔들
    recent = chart_data.candles[-20:] if len(chart_data.candles) >= 20 else chart_data.candles
    lines.append("\n[최근 일봉 데이터]")
    for c in recent:
        lines.append(f"{c.date} | 시:{c.open:,.0f} 고:{c.high:,.0f} 저:{c.low:,.0f} 종:{c.close:,.0f} 거래량:{c.volume:,}")

    # 기술적 지표
    ind = chart_data.indicators
    lines.append("\n[기술적 지표 (최근)]")
    lines.append(f"RSI: {ind.rsi}, MACD: {ind.macd} / Signal: {ind.macd_signal} / Hist: {ind.macd_histogram}")
    lines.append(f"볼린저: 상{ind.bollinger_upper} / 중{ind.bollinger_middle} / 하{ind.bollinger_lower}")
    lines.append(f"스토캐스틱: %K={ind.stochastic_k} / %D={ind.stochastic_d}")
    lines.append(f"이평선: 5일={ind.ma5} / 20일={ind.ma20} / 60일={ind.ma60} / 120일={ind.ma120}")

    return "\n".join(lines)


_ALIGNMENT_MAP = {"정배열": MovingAverageAlignment.BULLISH, "역배열": MovingAverageAlignment.BEARISH}
_DIRECTION_MAP = {"상승": Direction.UP, "하락": Direction.DOWN}
_STRENGTH_MAP = {"강함": SignalStrength.STRONG, "약함": SignalStrength.WEAK}


class ChartAnalysisAgent(BaseAgent, ChartAnalysisPort):
    """📊 차트분석 에이전트 - ChartAnalysisPort 구현"""

    async def analyze(self, stock: Stock, chart_data: ChartData) -> TechnicalPattern:
        user_prompt = _format_chart_data(stock, chart_data)
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        return TechnicalPattern(
            candle_pattern=data.get("candle_pattern", ""),
            ma_alignment=_ALIGNMENT_MAP.get(data.get("ma_alignment", "혼조"), MovingAverageAlignment.MIXED),
            rsi_signal=data.get("rsi_signal", ""),
            macd_signal=data.get("macd_signal", ""),
            bollinger_signal=data.get("bollinger_signal", ""),
            stochastic_signal=data.get("stochastic_signal", ""),
            overall_trend=_DIRECTION_MAP.get(data["overall_trend"], Direction.SIDEWAYS),
            strength=_STRENGTH_MAP.get(data["strength"], SignalStrength.MODERATE),
            key_points=data.get("key_points", []),
        )
