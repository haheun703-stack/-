"""v3 Agent 2E — 최종 포트폴리오 결정 (Portfolio Brain)

Phase 4 DeepAnalyst 통과 종목들을 받아서:
  - 기존 포트폴리오(positions.json)와 중복/집중 체크
  - CFO의 리스크 예산 참조하여 비중 배분
  - 최종 매수 종목 + 비중 결정

입력:
  - Phase 4 통과 종목 (conviction >= 5)
  - positions.json (현재 보유)
  - ai_strategic_analysis.json (레짐, 현금 비중 권고)

출력:
  - ai_v3_picks.json (최종)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PORTFOLIO_BRAIN = """\
당신은 한국 주식시장 포트폴리오 의사결정자입니다.
Deep Analyst가 선별한 후보 종목들과 현재 포트폴리오 상태를 분석하여,
오늘 실제 매수할 종목과 비중을 최종 결정합니다.

## 의사결정 프레임워크

1. **포트폴리오 맥락 분석**
   - 현재 보유 종목 수 vs 최대 슬롯 (KOSPI 레짐 기반)
   - 섹터 집중도: 동일 섹터 3종목 이상 → 신규 진입 억제
   - 현금 비중: 최소 20% 유지 (방어 레짐은 50%)
   - 기존 보유 종목과 중복 체크

2. **비중 배분 원칙** (Quarter-Kelly 기반)
   - conviction 9~10: 최대 20% (풀 사이즈)
   - conviction 7~8: 10~15% (표준 사이즈)
   - conviction 5~6: 5~10% (축소 사이즈)
   - 동일 섹터 기존 보유 → 50% 축소
   - 공격 레짐: 현금 20% → 최대 80% 투자
   - 방어 레짐: 현금 50% → 최대 50% 투자

3. **우선순위 결정**
   - conviction 높은 순 → thesis_alignment 강한 순
   - 동일 conviction이면 리스크/리워드 비율 우선
   - 최대 신규매수: max_new_buys 이하 (Phase 1 결정)

4. **제외 조건**
   - 현재 보유 종목은 재매수 불가
   - 동일 섹터 3종목 이상 보유 → 해당 섹터 신규 진입 불가
   - 현금 비중이 권고 이하 → 매수 중단

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요.

```json
{
  "decision_date": "YYYY-MM-DD",
  "regime": "공격|중립|방어|회피",
  "available_slots": 3,
  "current_holdings": 2,
  "cash_pct_before": 60.0,
  "cash_pct_after": 40.0,
  "buys": [
    {
      "ticker": "종목코드",
      "name": "종목명",
      "conviction": 8,
      "size_pct": 15.0,
      "strategy": "momentum",
      "thesis_alignment": "strong",
      "entry_price": 50000,
      "stop_loss_pct": -3.0,
      "target_pct": 12.0,
      "reasoning": "배분 근거 1문장"
    }
  ],
  "skipped": [
    {
      "ticker": "종목코드",
      "name": "종목명",
      "skip_reason": "섹터 집중도 초과"
    }
  ],
  "portfolio_warnings": ["경고1"],
  "reasoning": "전체 포트폴리오 판단 근거 2~3문장"
}
```
"""


class PortfolioBrainAgent(BaseAgent):
    """v3 Agent 2E — 최종 포트폴리오 결정 (Opus)"""

    def __init__(self, model: str | None = None):
        if model is None:
            model = self._load_model_from_settings()
        super().__init__(model=model)

    @staticmethod
    def _load_model_from_settings() -> str:
        """settings.yaml에서 strategic_model 로드 (Opus)"""
        try:
            import yaml
            settings_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_brain_v3", {}).get("strategic_model", "claude-sonnet-4-5-20250929")
        except Exception:
            return "claude-sonnet-4-5-20250929"

    async def _ask_claude(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
        """Opus는 스트리밍 필수 — BaseAgent 오버라이드"""
        if "opus" in self.model:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                return await stream.get_final_text()
        return await super()._ask_claude(system_prompt, user_prompt, max_tokens)

    async def decide(
        self,
        deep_picks: list[dict],
        positions: list[dict],
        strategic_result: dict,
    ) -> dict:
        """최종 포트폴리오 결정.

        Args:
            deep_picks: Phase 4 통과 종목 리스트
            positions: 현재 보유 포지션 (positions.json)
            strategic_result: Phase 1 결과 (레짐, max_new_buys 등)

        Returns:
            ai_v3_picks.json 최종 형식
        """
        picks_text = self._format_picks(deep_picks)
        portfolio_text = self._format_portfolio(positions)
        regime_text = self._format_regime(strategic_result)

        today = datetime.now().strftime("%Y-%m-%d")
        user_prompt = f"""\
## 결정 날짜: {today}

{regime_text}

{portfolio_text}

{picks_text}

위 정보를 분석하여 오늘 실제 매수할 종목과 비중을 JSON으로 결정하세요.
현금 비중과 섹터 집중도를 반드시 체크하세요.
"""

        logger.info("v3 Portfolio Brain 결정 시작 (%d 후보)", len(deep_picks))

        try:
            result = await self._ask_claude_json(SYSTEM_PORTFOLIO_BRAIN, user_prompt)
        except Exception as e:
            logger.error("v3 Portfolio Brain 실패: %s", e)
            return self._fallback_result(str(e), strategic_result)

        result.setdefault("decision_date", today)
        result = self._validate_result(result, strategic_result)

        logger.info(
            "v3 Portfolio Brain 완료: %d종목 매수, %d종목 스킵",
            len(result.get("buys", [])),
            len(result.get("skipped", [])),
        )
        return result

    @staticmethod
    def _format_picks(picks: list[dict]) -> str:
        """Deep Analyst 통과 종목 → 텍스트"""
        if not picks:
            return "[Phase 4 통과 종목] 없음"

        lines = [f"[Phase 4 통과 종목 ({len(picks)}개)]"]
        for p in picks:
            lines.append(
                f"  {p.get('name', '?')} ({p.get('ticker', '?')}): "
                f"conviction={p.get('conviction', 0)}/10, "
                f"strategy={p.get('strategy', '?')}, "
                f"thesis={p.get('thesis_alignment', '?')}"
            )
            if p.get("entry_price"):
                lines.append(
                    f"    진입가={p.get('entry_price', 0):,}, "
                    f"손절={p.get('stop_loss_pct', 0)}%, "
                    f"목표={p.get('target_pct', 0)}%"
                )
            if p.get("reasoning"):
                lines.append(f"    근거: {p['reasoning'][:80]}")
            if p.get("risks"):
                lines.append(f"    리스크: {', '.join(p['risks'][:2])}")
        return "\n".join(lines)

    @staticmethod
    def _format_portfolio(positions: list[dict]) -> str:
        """현재 포트폴리오 → 텍스트"""
        if not positions:
            return "[현재 포트폴리오] 보유 종목 없음 (현금 100%)"

        lines = [f"[현재 포트폴리오 ({len(positions)}종목)]"]
        sectors = {}
        for p in positions:
            name = p.get("name", "?")
            ticker = p.get("ticker", "?")
            entry = p.get("entry_price", 0)
            current = p.get("current_price", entry)
            pnl = ((current / entry) - 1) * 100 if entry > 0 else 0
            sector = p.get("sector", "미분류")

            lines.append(f"  {name}({ticker}): 진입={entry:,} 현재={current:,} ({pnl:+.1f}%)")
            sectors[sector] = sectors.get(sector, 0) + 1

        if sectors:
            lines.append(f"\n  섹터 분포: {sectors}")

        return "\n".join(lines)

    @staticmethod
    def _format_regime(data: dict) -> str:
        """레짐 + 리스크 → 텍스트"""
        lines = ["[레짐 및 제약 조건]"]
        lines.append(f"  레짐: {data.get('regime', '?')} (신뢰도: {data.get('regime_confidence', 0):.0%})")
        lines.append(f"  최대 신규매수: {data.get('max_new_buys', 0)}종목")
        lines.append(f"  현금 비중 권고: {data.get('cash_reserve_suggestion', 20)}%")

        risks = data.get("risk_factors", [])
        if risks:
            lines.append(f"  리스크: {', '.join(risks[:3])}")

        return "\n".join(lines)

    @staticmethod
    def _validate_result(result: dict, strategic: dict) -> dict:
        """결과 검증 및 보정"""
        result.setdefault("buys", [])
        result.setdefault("skipped", [])
        result.setdefault("portfolio_warnings", [])
        result.setdefault("reasoning", "")

        # max_new_buys 초과 방지
        max_buys = strategic.get("max_new_buys", 3)
        if len(result["buys"]) > max_buys:
            overflow = result["buys"][max_buys:]
            result["buys"] = result["buys"][:max_buys]
            for o in overflow:
                result["skipped"].append({
                    "ticker": o.get("ticker", ""),
                    "name": o.get("name", ""),
                    "skip_reason": f"max_new_buys({max_buys}) 초과",
                })

        # size_pct 클램핑
        for b in result["buys"]:
            b["size_pct"] = max(3.0, min(20.0, b.get("size_pct", 10.0)))
            b["conviction"] = max(1, min(10, b.get("conviction", 5)))

        return result

    @staticmethod
    def _fallback_result(error_msg: str, strategic: dict) -> dict:
        """AI 장애 시 안전한 빈 결과"""
        return {
            "decision_date": datetime.now().strftime("%Y-%m-%d"),
            "regime": strategic.get("regime", "방어"),
            "available_slots": 0,
            "current_holdings": 0,
            "cash_pct_before": 100.0,
            "cash_pct_after": 100.0,
            "buys": [],
            "skipped": [],
            "portfolio_warnings": [f"AI 장애: {error_msg}"],
            "reasoning": "AI 장애로 매수 중단",
            "error": error_msg,
        }
