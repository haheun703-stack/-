"""거래량분석 서브에이전트 - 거래량 추세 및 매물대 분석"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.models import (
    ChartData,
    Direction,
    SignalStrength,
    Stock,
    SupplyDemandZone,
    VolumeAnalysis,
)
from src.use_cases.ports import VolumeAnalysisPort

SYSTEM_PROMPT = """당신은 한국 주식시장 거래량 및 매물대 분석 전문가입니다.
주어진 차트 데이터에서 거래량 추세와 매물대를 분석합니다.

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "avg_volume_ratio": 1.5,
  "volume_trend": "상승|하락|횡보",
  "accumulation_signal": "매집 의심|분산 의심|중립",
  "zones": [
    {
      "price_low": 50000,
      "price_high": 52000,
      "zone_type": "지지|저항",
      "strength": "강함|보통|약함",
      "volume_ratio": 2.1,
      "description": "설명"
    }
  ],
  "key_points": ["핵심 포인트1", "핵심 포인트2"]
}
```"""


def _format_volume_data(stock: Stock, chart_data: ChartData) -> str:
    """거래량 데이터를 프롬프트용 텍스트로 변환"""
    lines = [f"종목: {stock.name} ({stock.ticker})"]

    recent = chart_data.candles[-60:] if len(chart_data.candles) >= 60 else chart_data.candles
    lines.append("\n[일봉 데이터 (거래량 중심)]")
    for c in recent:
        lines.append(f"{c.date} | 종:{c.close:,.0f} 고:{c.high:,.0f} 저:{c.low:,.0f} 거래량:{c.volume:,}")

    if recent:
        avg_vol = sum(c.volume for c in recent) / len(recent)
        lines.append(f"\n평균 거래량: {avg_vol:,.0f}")

    return "\n".join(lines)


_DIRECTION_MAP = {"상승": Direction.UP, "하락": Direction.DOWN}
_STRENGTH_MAP = {"강함": SignalStrength.STRONG, "약함": SignalStrength.WEAK}


class VolumeAnalysisAgent(BaseAgent, VolumeAnalysisPort):
    """📈 거래량분석 에이전트 - VolumeAnalysisPort 구현"""

    async def analyze(self, stock: Stock, chart_data: ChartData) -> VolumeAnalysis:
        user_prompt = _format_volume_data(stock, chart_data)
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        zones = [
            SupplyDemandZone(
                price_low=z.get("price_low", 0),
                price_high=z.get("price_high", 0),
                zone_type=z.get("zone_type", "지지"),
                strength=_STRENGTH_MAP.get(z.get("strength", "보통"), SignalStrength.MODERATE),
                volume_ratio=z.get("volume_ratio", 1.0),
                description=z.get("description", ""),
            )
            for z in data.get("zones", [])
        ]

        return VolumeAnalysis(
            avg_volume_ratio=data.get("avg_volume_ratio", 1.0),
            volume_trend=_DIRECTION_MAP.get(data.get("volume_trend", "횡보"), Direction.SIDEWAYS),
            accumulation_signal=data.get("accumulation_signal", "중립"),
            zones=zones,
            key_points=data.get("key_points", []),
        )
