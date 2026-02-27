"""AI 두뇌 에이전트 — Claude API 기반 정성적 뉴스 판단 엔진

매일 50~70개 뉴스를 종합 분석하여 종목별 BUY/WATCH/AVOID 판단 생성.
기존 Bot(기술적 지표)과 독립적으로 정성적 판단을 제공합니다.

환각 방지:
  - 84종목 유니버스 ticker→name 매핑을 프롬프트에 주입
  - _ask_claude_json() → JSON 스키마 강제
  - 반환된 ticker를 유니버스 대조 검증
"""

from __future__ import annotations

import logging

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 한국 증시 전문 애널리스트입니다. 매일 수집된 뉴스를 종합 분석하여
향후 3~7일 내 투자 판단을 내립니다.

## 판단 기준
- **BUY**: 향후 3~7일 내 5%+ 상승 기대. 구체적 촉매(공시/수주/정책)가 있어야 함.
- **WATCH**: 재료는 있으나 아직 미확인이거나 타이밍 불분명. 관찰 필요.
- **AVOID**: 명확한 악재 감지. 유상증자, 실적 부진, 소송, 관리종목 등.

## 신뢰도 기준
- 확정 공시(DART) > 복수 언론 보도 > 단일 언론 > 루머/테마
- "이미 반영된" 재료(과거 1주 내 10%+ 상승)는 BUY에서 제외
- 신뢰도(confidence): 0.5(낮음) ~ 1.0(확실)

## 반드시 지킬 규칙
1. 아래 제공된 유니버스 목록에 있는 종목만 판단하세요.
2. 유니버스에 없는 종목은 절대 언급하지 마세요.
3. ticker는 반드시 6자리 숫자 문자열로 작성하세요 (예: "005930").
4. 최대 15개 종목까지만 판단하세요.
5. 근거가 불충분하면 WATCH로 분류하세요. 추측으로 BUY 하지 마세요.

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{
  "market_sentiment": "bullish 또는 bearish 또는 neutral",
  "key_themes": ["핵심 테마1", "핵심 테마2", "핵심 테마3"],
  "stock_judgments": [
    {
      "ticker": "6자리 숫자",
      "name": "종목명",
      "action": "BUY 또는 WATCH 또는 AVOID",
      "confidence": 0.5~1.0,
      "reasoning": "판단 근거 1~2문장",
      "catalysts": ["촉매1", "촉매2"],
      "risks": ["리스크1"],
      "urgency": "high 또는 medium 또는 low",
      "expected_impact_pct": 5~20
    }
  ],
  "sector_outlook": {
    "섹터명": {"direction": "positive 또는 negative 또는 neutral", "reason": "근거"}
  }
}
```
"""


class NewsBrainAgent(BaseAgent):
    """뉴스 기반 정성적 종목 판단 AI 두뇌"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        super().__init__(model=model)

    async def analyze_daily_news(
        self,
        news_items: list[dict],
        universe: dict[str, str],
    ) -> dict:
        """50~70개 뉴스를 종합 분석하여 종목별 판단 생성.

        Args:
            news_items: 뉴스 리스트 [{title, summary, source, category, ...}]
            universe: ticker→name 매핑 (예: {"005930": "삼성전자"})

        Returns:
            AI 판단 결과 dict
        """
        # 유니버스 목록을 프롬프트에 삽입
        universe_text = "\n".join(
            f"  {t}: {n}" for t, n in sorted(universe.items())
        )

        news_text = ""
        for i, item in enumerate(news_items, 1):
            title = item.get("title", "")
            summary = item.get("summary", "")
            source = item.get("source", "")
            category = item.get("category", "")
            impact = item.get("impact", "")
            news_text += f"[{i}] [{category}] {title}\n"
            if summary:
                news_text += f"    요약: {summary}\n"
            if source:
                news_text += f"    출처: {source}"
                if impact:
                    news_text += f" | 영향: {impact}"
                news_text += "\n"

        user_prompt = f"""\
## 투자 유니버스 ({len(universe)}종목)
{universe_text}

## 오늘의 뉴스 ({len(news_items)}건)
{news_text}

위 뉴스를 종합 분석하여 투자 판단을 JSON으로 응답하세요.
유니버스에 있는 종목만 판단하고, 근거가 명확한 종목만 BUY로 분류하세요.
"""

        logger.info(
            "AI 두뇌 분석 시작: %d건 뉴스, %d종목 유니버스",
            len(news_items),
            len(universe),
        )

        try:
            result = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error("AI 두뇌 분석 실패: %s", e)
            return {
                "market_sentiment": "neutral",
                "key_themes": [],
                "stock_judgments": [],
                "sector_outlook": {},
                "error": str(e),
            }

        # 유니버스 검증: 유니버스 밖 종목 제거
        if "stock_judgments" in result:
            verified = []
            for j in result["stock_judgments"]:
                t = j.get("ticker", "")
                if t in universe:
                    verified.append(j)
                else:
                    logger.warning("유니버스 밖 종목 제거: %s (%s)", t, j.get("name", ""))
            result["stock_judgments"] = verified

        logger.info(
            "AI 두뇌 분석 완료: 센티먼트=%s, 판단종목=%d",
            result.get("market_sentiment", "?"),
            len(result.get("stock_judgments", [])),
        )
        return result
