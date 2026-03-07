"""GameAnalyst 서브에이전트 — 6D 게임 설계 분석

주식 시장을 '게임'으로 보고 설계자/역할/함정을 분석.
기존 1D~5D 분석에 6D 메타 레이어를 추가하여 투자 의사결정을 강화.
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.game_models import GameDesignAnalysis
from src.use_cases.ports import GameAnalystPort

SYSTEM_PROMPT = """당신은 한국 주식시장의 6D 게임 설계 분석 전문가입니다.
주식 투자를 '게임'의 관점으로 분석합니다.

핵심 분석 프레임워크:
1. 설계자 (Designer): 이 게임을 설계하고 운영하는 주체는 누구인가?
   - 대주주/경영진, 정부 정책, 시장 사이클, 기관 투자자, IB 등
2. 우리 역할 (Our Role): 이 게임에서 개인 투자자인 우리는 어떤 역할인가?
   - 동반자, 역발상 투자자, 이벤트 대기자, 추격자, 피해자 등
3. 엣지 (Edge): 우리가 가진 비대칭적 정보 우위는 무엇인가?
   - 시간 우위, 정보 우위, 가치 괴리, 이벤트 선점 등
4. 함정 (Trap): 이 게임에 숨겨진 함정은 무엇인가?
   - 유동성 함정, 정보 비대칭, 모멘텀 소멸, 가치 함정 등
   - 함정 위험도를 0~100%로 정량화

분석 원칙:
- 군중이 모이는 곳에 함정이 있다
- 설계자의 이해관계와 우리의 이해관계가 정렬되는가
- "이미 알려진 정보"는 엣지가 아니다
- 시간이 내 편인가, 적의 편인가

반드시 아래 JSON 형식으로 응답하세요.
```json
{
  "designer": "설계자 분석 (누가 이 게임을 만들었나)",
  "our_role": "우리의 역할 (이 게임에서 우리는 누구인가)",
  "edge": "우리의 엣지 (비대칭 정보 우위)",
  "trap_description": "함정 설명 (구체적으로)",
  "trap_risk_pct": 30,
  "game_score": 65.0,
  "reasoning": "종합 판단 근거"
}
```

game_score 기준:
- 80+ : 게임이 우리에게 극도로 유리 (설계자 이해 정렬 + 낮은 함정)
- 65~79 : 게임 참여 가치 있음 (적절한 엣지 존재)
- 50~64 : 중립 (특별한 엣지 없음)
- 50 미만 : 게임이 불리 (함정 높음, 후발 추격, 정보 열위)"""


def _format_game_input(stock_context: dict) -> str:
    """6D 게임 분석 요청을 프롬프트용 텍스트로 변환"""
    lines = ["[종목 기본 정보]"]
    lines.append(f"종목: {stock_context.get('ticker', '?')} {stock_context.get('name', '')}")
    lines.append(f"현재가: {stock_context.get('current_price', 0):,.0f}원")
    lines.append(f"수익률: {stock_context.get('return_pct', 0):+.1f}%")
    lines.append(f"보유일: {stock_context.get('hold_days', 0)}일")

    lines.append("\n[가격 위치 (1D)]")
    lines.append(f"52주 고점 대비: {stock_context.get('pct_52w_high', 0):.1f}%")

    lines.append("\n[밸류에이션 (2D)]")
    per = stock_context.get("per", 0)
    if per > 0:
        lines.append(f"PER: {per:.1f}배")
    target = stock_context.get("target_price", 0)
    if target > 0:
        gap = (target / stock_context.get("current_price", 1) - 1) * 100
        lines.append(f"목표가: {target:,.0f}원 (괴리 {gap:+.1f}%)")

    lines.append("\n[기술적 분석 (3D)]")
    lines.append(f"멀티팩터 점수: {stock_context.get('multifactor_score', 0):.1f}")

    lines.append("\n[타이밍 (4D)]")
    lines.append(f"나선시계: {stock_context.get('clock_hour', 0):.1f}시")
    lines.append(f"타이밍 점수: {stock_context.get('timing_score', 0):.1f}")

    lines.append("\n[메타게임 (5D)]")
    lines.append(f"군중 관심도: {stock_context.get('crowd_pct', 50):.0f}%")
    lines.append(f"무관심 점수: {stock_context.get('neglect_score', 50):.1f}")

    lines.append("\n[업종/섹터 정보]")
    lines.append(f"업종: {stock_context.get('sector', '기타')}")

    catalyst = stock_context.get("catalyst", "")
    if catalyst:
        lines.append(f"\n[알려진 카탈리스트]\n{catalyst}")

    news = stock_context.get("recent_news", "")
    if news:
        lines.append(f"\n[최근 뉴스]\n{news}")

    return "\n".join(lines)


class GameAnalystAgent(BaseAgent, GameAnalystPort):
    """6D 게임 설계 분석 에이전트"""

    async def analyze_game(self, stock_context: dict) -> GameDesignAnalysis:
        """종목의 6D 게임 구조 분석"""
        user_prompt = _format_game_input(stock_context)
        data = await self._ask_claude_json(SYSTEM_PROMPT, user_prompt)

        return GameDesignAnalysis(
            ticker=stock_context.get("ticker", ""),
            name=stock_context.get("name", ""),
            designer=data.get("designer", ""),
            our_role=data.get("our_role", ""),
            edge=data.get("edge", ""),
            trap_description=data.get("trap_description", ""),
            trap_risk_pct=int(data.get("trap_risk_pct", 50)),
            game_score=float(data.get("game_score", 50.0)),
            reasoning=data.get("reasoning", ""),
        )

    async def batch_analyze(self, stock_contexts: list[dict]) -> list[GameDesignAnalysis]:
        """여러 종목 6D 게임 분석 (순차 호출)"""
        results = []
        for ctx in stock_contexts:
            try:
                result = await self.analyze_game(ctx)
                results.append(result)
            except Exception:
                # 실패 시 기본값
                results.append(GameDesignAnalysis(
                    ticker=ctx.get("ticker", ""),
                    name=ctx.get("name", ""),
                    designer="분석 실패",
                    our_role="알 수 없음",
                    edge="알 수 없음",
                    trap_description="분석 실패로 판단 불가",
                    trap_risk_pct=50,
                    game_score=50.0,
                    reasoning="Claude API 호출 실패",
                ))
        return results
