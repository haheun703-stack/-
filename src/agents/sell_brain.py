"""v3 Agent 4B/4C — 장중 AI 매도 판단 (Sell Brain)

장중 3회 (10:00, 12:00, 14:00) + 프리클로즈(14:30) 호출하여
보유 종목별 매도/유지 판단.

입력:
  - positions.json (현재 보유)
  - ai_strategic_analysis.json (레짐, thesis 변화)
  - ai_sector_focus.json (섹터 경고)
  - 실시간 가격/수급 데이터

출력:
  - ai_sell_cache.json (35분 TTL)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_SELL_BRAIN = """\
당신은 한국 주식시장 매도 판단 전문가입니다.
보유 종목의 현재 상태를 분석하여 매도/유지 판단을 내립니다.

## 매도 판단 프레임워크

1. **Thesis 변화 감지** (최우선)
   - 매수 당시 산업 thesis가 유효한가?
   - thesis 소멸/약화 → 즉시 매도 권고
   - thesis 강화 → 목표가 상향 고려

2. **기술적 이탈**
   - MA20 이탈 → 경고 (1일 관찰)
   - MA60 이탈 → 강한 매도 시그널
   - SAR 반전 (상승→하락) → 매도 시그널
   - ADX < 15 (추세 소멸) → 관망 or 매도

3. **수급 이탈**
   - 외인+기관 동반 매도 3일 연속 → 경고
   - 거래량 급증 + 음봉 → 분배 가능성
   - 섹터 전체 매도세 → 강한 매도 시그널

4. **릴레이 완료 감지**
   - 릴레이 타겟 종목 → 소스 섹터 하락 전환 시 매도
   - 릴레이 매수 후 3~5일 경과 → 모멘텀 약화 시 매도

5. **오버나이트 리스크** (프리클로즈 전용)
   - 미국 선물 하락, VIX 급등 → 비중 축소 권고
   - 어닝 시즌 전날 → 리스크 경고

## 판단 등급
- HOLD: 유지 (변화 없음)
- WATCH: 관찰 (내일까지 대기)
- PARTIAL_SELL: 50% 부분 매도
- SELL_NOW: 즉시 전량 매도

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요.

```json
{
  "check_time": "HH:MM",
  "check_type": "intraday|preclose",
  "positions": [
    {
      "ticker": "종목코드",
      "name": "종목명",
      "action": "HOLD|WATCH|PARTIAL_SELL|SELL_NOW",
      "urgency": "immediate|end_of_day|tomorrow",
      "thesis_status": "intact|weakened|broken",
      "reasoning": "판단 근거 1~2문장",
      "risks": ["리스크1"]
    }
  ],
  "portfolio_action": "maintain|reduce|defensive",
  "overall_reasoning": "전체 판단 1문장"
}
```
"""


class SellBrainAgent(BaseAgent):
    """v3 Agent 4B/4C — 장중 AI 매도 판단 (Sonnet)"""

    def __init__(self, model: str | None = None):
        if model is None:
            model = self._load_model_from_settings()
        super().__init__(model=model)

    @staticmethod
    def _load_model_from_settings() -> str:
        """settings.yaml에서 tactical_model 로드"""
        try:
            import yaml
            settings_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_brain_v3", {}).get("tactical_model", "claude-sonnet-4-5-20250929")
        except Exception:
            return "claude-sonnet-4-5-20250929"

    async def check_sell(
        self,
        positions: list[dict],
        strategic_result: dict,
        sector_focus: dict | None = None,
    ) -> dict:
        """장중 AI 매도 체크 (10:00, 12:00, 14:00).

        Args:
            positions: 현재 보유 포지션
            strategic_result: Phase 1 결과 (thesis 참조)
            sector_focus: Phase 2 결과 (섹터 경고 참조)

        Returns:
            {positions: [{ticker, action, reasoning, ...}]}
        """
        if not positions:
            logger.info("보유 종목 없음 — 매도 체크 스킵")
            return self._empty_result("intraday")

        position_text = self._format_positions(positions)
        context_text = self._format_context(strategic_result, sector_focus)

        now = datetime.now().strftime("%H:%M")
        user_prompt = f"""\
## 장중 매도 체크: {now}

{context_text}

{position_text}

각 보유 종목의 매도/유지 판단을 JSON으로 응답하세요.
thesis 변화와 기술적 이탈에 주의하세요.
"""

        logger.info("v3 Sell Brain 장중 체크 시작 (%d종목)", len(positions))

        try:
            result = await self._ask_claude_json(SYSTEM_SELL_BRAIN, user_prompt)
        except Exception as e:
            logger.error("v3 Sell Brain 실패: %s", e)
            return self._fallback_result(str(e), positions)

        result.setdefault("check_time", now)
        result.setdefault("check_type", "intraday")

        sell_count = sum(
            1 for p in result.get("positions", [])
            if p.get("action") in ("SELL_NOW", "PARTIAL_SELL")
        )
        logger.info(
            "v3 Sell Brain 완료: %d종목 중 %d매도 시그널",
            len(positions), sell_count,
        )
        return result

    async def preclose_check(
        self,
        positions: list[dict],
        strategic_result: dict,
        overnight_signal: dict | None = None,
    ) -> dict:
        """프리클로즈 AI 매도 (14:30) — 오버나이트 리스크 집중 분석.

        Args:
            positions: 현재 보유 포지션
            strategic_result: Phase 1 결과
            overnight_signal: US overnight 시그널 (있으면)

        Returns:
            {positions: [{ticker, action, ...}]}
        """
        if not positions:
            return self._empty_result("preclose")

        position_text = self._format_positions(positions)

        overnight_text = ""
        if overnight_signal:
            grade = overnight_signal.get("grade", "?")
            score = overnight_signal.get("score", 0)
            overnight_text = f"\n[US Overnight] 등급={grade}, 점수={score:.1f}"
            vix = overnight_signal.get("vix_close", 0)
            if vix:
                overnight_text += f", VIX={vix:.1f}"

        user_prompt = f"""\
## 프리클로즈 매도 체크: 14:30
{overnight_text}

[레짐: {strategic_result.get('regime', '?')}]
[리스크: {', '.join(strategic_result.get('risk_factors', [])[:3])}]

{position_text}

오버나이트 리스크를 고려하여 매도/유지 판단을 JSON으로 응답하세요.
미국 시장 하락 가능성이 높으면 비중 축소를 권고하세요.
"""

        logger.info("v3 Sell Brain 프리클로즈 체크 시작 (%d종목)", len(positions))

        try:
            result = await self._ask_claude_json(SYSTEM_SELL_BRAIN, user_prompt)
        except Exception as e:
            logger.error("v3 Sell Brain 프리클로즈 실패: %s", e)
            return self._fallback_result(str(e), positions)

        result.setdefault("check_time", "14:30")
        result.setdefault("check_type", "preclose")
        return result

    @staticmethod
    def _format_positions(positions: list[dict]) -> str:
        """보유 종목 → 텍스트"""
        lines = [f"[보유 종목 ({len(positions)}개)]"]
        for p in positions:
            name = p.get("name", "?")
            ticker = p.get("ticker", "?")
            entry = p.get("entry_price", 0)
            current = p.get("current_price", entry)
            pnl = ((current / entry) - 1) * 100 if entry > 0 else 0
            hold_days = p.get("hold_days", "?")
            grade = p.get("grade", "?")
            trigger = p.get("trigger_type", "?")
            stop = p.get("stop_loss", 0)

            lines.append(f"  {name}({ticker}): 등급={grade}, 트리거={trigger}")
            lines.append(f"    진입={entry:,} → 현재={current:,} ({pnl:+.1f}%)")
            lines.append(f"    보유 {hold_days}일, 손절={stop:,}")

            # 기술적 상태 (있으면)
            rsi = p.get("rsi")
            adx = p.get("adx")
            sar = p.get("sar_trend")
            if rsi is not None:
                lines.append(f"    RSI={rsi:.1f}, ADX={adx or '?'}, SAR={'↑' if sar == 1 else '↓' if sar == -1 else '?'}")

        return "\n".join(lines)

    @staticmethod
    def _format_context(strategic: dict, sector_focus: dict | None) -> str:
        """전략 맥락 → 텍스트"""
        lines = [f"[전략 맥락]"]
        lines.append(f"  레짐: {strategic.get('regime', '?')}")

        # 산업 thesis 요약
        theses = strategic.get("industry_thesis", [])
        if theses:
            lines.append(f"  산업 Thesis:")
            for t in theses[:3]:
                lines.append(f"    - {t.get('sector', '?')}: {t.get('thesis', '?')[:40]}")

        # 섹터 경고
        if sector_focus:
            warnings = sector_focus.get("sector_warnings", [])
            if warnings:
                lines.append(f"  섹터 경고:")
                for w in warnings[:3]:
                    lines.append(f"    - [{w.get('severity')}] {w.get('sector')}: {w.get('warning', '')[:40]}")

        return "\n".join(lines)

    @staticmethod
    def _empty_result(check_type: str) -> dict:
        return {
            "check_time": datetime.now().strftime("%H:%M"),
            "check_type": check_type,
            "positions": [],
            "portfolio_action": "maintain",
            "overall_reasoning": "보유 종목 없음",
        }

    @staticmethod
    def _fallback_result(error_msg: str, positions: list[dict]) -> dict:
        """AI 장애 시 전종목 HOLD"""
        return {
            "check_time": datetime.now().strftime("%H:%M"),
            "check_type": "intraday",
            "positions": [
                {
                    "ticker": p.get("ticker", ""),
                    "name": p.get("name", ""),
                    "action": "HOLD",
                    "urgency": "tomorrow",
                    "thesis_status": "intact",
                    "reasoning": f"AI 장애로 유지: {error_msg}",
                    "risks": [],
                }
                for p in positions
            ],
            "portfolio_action": "maintain",
            "overall_reasoning": f"AI 장애 — 전종목 유지: {error_msg}",
            "error": error_msg,
        }
