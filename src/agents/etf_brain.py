"""ETF AI 두뇌 — Claude API 기반 ETF 주문 필터링 에이전트

룰베이스 오케스트레이터가 생성한 ETF 주문에 대해
AI가 PASS/KILL/HOLD 판단을 내린다.

핵심 원칙:
  - AI는 "사라"고 절대 안 함. "사지 마라"만 할 수 있음.
  - 공격은 룰이, 방어 보강만 AI가.
  - 3가지 개입: KILL(매수 보류), HOLD(교체 연기), WARNING(이상 징후 알림)
"""

from __future__ import annotations

import logging

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 한국 ETF 트레이딩 리스크 관리 전문가입니다.
룰베이스 로테이션 엔진이 생성한 ETF 매수/매도 주문을 검토하여
위험 요소를 필터링합니다.

## 당신의 역할
- 룰이 만든 주문을 검토하는 "방어 필터"입니다.
- 새로운 매수를 제안하지 마세요. 기존 주문을 PASS/KILL/HOLD만 판단하세요.
- 확실한 위험이 아니면 PASS하세요. 의심만으로 KILL하지 마세요.

## 판단 기준
- **PASS**: 위험 없음. 룰대로 실행.
- **KILL**: 명확한 위험 감지 → 매수 보류.
  - 해당 섹터에 구체적 악재 (정책 변경, 대규모 매도, 지정학 리스크)
  - 모멘텀 1위가 이미 과열 (RSI 80+, 단기 급등 후 차익실현 예상)
  - 횡보장에서 모멘텀 랭킹이 매주 바뀌는 불안정 상황
- **HOLD**: 교체 연기 (기존 보유 유지).
  - 모멘텀은 바뀌었지만 기존 섹터 추세가 여전히 강함
  - 교체 비용(수수료) 대비 기대이익 불분명
- **WARNING**: 주문과 별개로 주의할 이상 징후 경고.

## 레버리지 특별 규칙
- 레버리지 ETF 매수는 기본적으로 높은 위험 → 더 엄격하게 검토
- VIX 급등, 지정학 긴장, 주말 리스크 → KILL 우선

## 매도 주문에 대해
- SELL 주문은 기본 PASS (방어적 청산은 항상 허용)
- 단, 급매도 시 시장 충격이 우려되면 WARNING

## 반드시 지킬 규칙
1. 모든 주문에 대해 반드시 verdict를 내리세요.
2. 근거 없이 KILL하지 마세요. 구체적 이유를 1~2문장으로 작성.
3. 50% 이상의 주문은 PASS여야 합니다 (과도한 필터링 금지).
4. 뉴스에 없는 내용을 추측하지 마세요.

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{
  "market_assessment": "현재 시장 상황 1줄 요약",
  "risk_level": "low 또는 medium 또는 high",
  "order_verdicts": [
    {
      "code": "ETF 코드",
      "name": "ETF 이름",
      "action": "원래 주문 (BUY/SELL)",
      "verdict": "PASS 또는 KILL 또는 HOLD",
      "reason": "판단 근거 1~2문장",
      "confidence": 0.5~1.0
    }
  ],
  "warnings": [
    "주문과 별개로 주의할 사항 (있을 때만)"
  ],
  "sector_risk_notes": {
    "섹터명": "해당 섹터 리스크 요약 (위험 섹터만)"
  }
}
```
"""


class ETFBrainAgent(BaseAgent):
    """ETF 주문 필터링 AI 두뇌"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        super().__init__(model=model)

    async def filter_orders(
        self,
        order_queue: list[dict],
        regime: str,
        allocation: dict,
        news_context: str,
        market_context: str,
    ) -> dict:
        """오케스트레이터 주문에 대해 AI 필터링 판단.

        Args:
            order_queue: 오케스트레이터가 생성한 주문 리스트
            regime: 현재 KOSPI 레짐
            allocation: 비중 매트릭스
            news_context: 관련 뉴스 요약 텍스트
            market_context: 시장 상태 요약 (KOSPI, US, 레짐 등)

        Returns:
            AI 필터링 결과 dict
        """
        if not order_queue:
            logger.info("주문 큐 비어있음 — AI 필터 스킵")
            return {
                "market_assessment": "주문 없음",
                "risk_level": "low",
                "order_verdicts": [],
                "warnings": [],
                "sector_risk_notes": {},
            }

        # 주문 큐 텍스트
        orders_text = ""
        for i, order in enumerate(order_queue, 1):
            orders_text += (
                f"[{i}] {order.get('name', '?')} ({order.get('code', '?')})\n"
                f"    축: {order.get('axis', '?')} | "
                f"동작: {order.get('action', '?')} | "
                f"비중: {order.get('target_weight_pct', 0)}%\n"
                f"    근거: {order.get('reason', '')}\n"
            )

        user_prompt = f"""\
## 시장 상태
{market_context}

## 현재 레짐: {regime}
## 비중 배분: 섹터 {allocation.get('sector', 0)}% | 레버리지 {allocation.get('leverage', 0)}% | 지수 {allocation.get('index', 0)}% | 현금 {allocation.get('cash', 0)}%

## 룰베이스 엔진이 생성한 주문 ({len(order_queue)}건)
{orders_text}

## 오늘의 뉴스/맥락
{news_context if news_context else "(뉴스 데이터 없음 — 기본 PASS 권장)"}

위 주문을 검토하여 각각에 대해 PASS/KILL/HOLD 판단을 JSON으로 응답하세요.
확실한 위험이 아니면 PASS하세요. 과도한 필터링은 성과를 해칩니다.
"""

        logger.info(
            "ETF AI 필터 시작: %d건 주문, 레짐=%s",
            len(order_queue), regime,
        )

        try:
            result = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error("ETF AI 필터 실패: %s — 전체 PASS 처리", e)
            return {
                "market_assessment": f"AI 분석 실패: {e}",
                "risk_level": "unknown",
                "order_verdicts": [
                    {
                        "code": o.get("code", ""),
                        "name": o.get("name", ""),
                        "action": o.get("action", ""),
                        "verdict": "PASS",
                        "reason": "AI 분석 실패 — 안전 PASS",
                        "confidence": 0,
                    }
                    for o in order_queue
                ],
                "warnings": [f"AI 필터 오류: {e}"],
                "sector_risk_notes": {},
                "error": str(e),
            }

        logger.info(
            "ETF AI 필터 완료: 리스크=%s, KILL=%d건",
            result.get("risk_level", "?"),
            sum(1 for v in result.get("order_verdicts", [])
                if v.get("verdict") == "KILL"),
        )
        return result
