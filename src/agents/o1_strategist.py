"""GPT o1 거시/미시 분석 에이전트 — Phase 0 Deep Thinking

o1의 deep reasoning으로 거시적+미시적 분석을 수행.
결과를 Phase 1 StrategicBrain의 컨텍스트로 주입.

입력:
  - overnight_signal.json
  - market_intelligence.json (Perplexity)
  - market_news.json (RSS)
  - sector_momentum.json
  - regime_macro_signal.json

출력:
  - data/o1_deep_analysis.json
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_json(name: str) -> dict | list:
    path = DATA_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


O1_DEEP_PROMPT = """\
당신은 세계적 수준의 거시경제+산업 구조 분석가입니다.
아래 데이터를 깊이 사고하여 한국 주식시장의 거시적/미시적 분석을 수행하세요.

## 거시적 분석 (Macro)
1. 글로벌 유동성 흐름: 미국 금리, 달러 인덱스, VIX의 조합이 시사하는 바
2. 미국-한국 커플링/디커플링: EWY vs SPY 괴리도, 원인 분석
3. 섹터 로테이션 싸이클: 현재 위치 (early/mid/late cycle)
4. 지정학 리스크 프리미엄: 구체적 수치로 정량화

## 미시적 분석 (Micro)
1. 각 주요 섹터의 실적 사이클 위치 (반도체, 2차전지, 조선, 방산, 바이오)
2. 수주잔고/ASP/출하량 등 선행지표 변화
3. 밸류에이션 멀티플 재평가 가능성
4. 수급 구조 변화 (외인/기관 포지셔닝 방향)

## Deep Reasoning 요구사항
- 표면적 데이터 너머의 인과관계를 추론하세요
- "왜?"를 3단계 이상 파고드세요
- 컨센서스와 다른 독창적 관점이 있다면 제시하세요
- 확신도를 정량화하세요

## 출력 형식 (JSON)
{
  "analysis_date": "YYYY-MM-DD",
  "macro_analysis": {
    "global_liquidity": "유동성 상황 요약",
    "coupling_status": "동조|독립|역행",
    "cycle_position": "early|mid|late",
    "geopolitical_risk_score": 1~10,
    "key_macro_insights": ["인사이트1", "인사이트2"],
    "macro_regime": "RISK_ON|RISK_OFF|TRANSITION",
    "confidence": 0.0~1.0
  },
  "micro_analysis": [
    {
      "sector": "섹터명",
      "earnings_cycle": "trough|recovery|expansion|peak|contraction",
      "leading_indicators": "선행지표 변화 요약",
      "valuation_rerating": "상향|유지|하향",
      "supply_demand_shift": "매집|중립|분배",
      "conviction": 1~10,
      "contrarian_view": "컨센서스와 다른 관점 (있으면)"
    }
  ],
  "cross_sector_dynamics": "섹터간 상호작용/릴레이 분석",
  "risk_scenarios": [
    {"scenario": "리스크 시나리오", "probability": 0.0~1.0, "impact": "high|medium|low"}
  ],
  "actionable_summary": "Phase 1 StrategicBrain에 전달할 핵심 인사이트 3줄"
}
"""


class O1StrategistAgent:
    """GPT o1 기반 Deep Thinking 거시/미시 분석 에이전트."""

    def __init__(self, model: str | None = None):
        if model is None:
            model = self._load_model_from_settings()
        self.model = model
        self._client = None

    @staticmethod
    def _load_model_from_settings() -> str:
        try:
            import yaml
            settings_path = PROJECT_ROOT / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_upgrade", {}).get("o1_model", "o1")
        except Exception:
            return "o1"

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        return self._client

    async def deep_analyze(self, context: dict | None = None) -> dict:
        """거시+미시 Deep Thinking 분석.

        Args:
            context: 외부 데이터 (None이면 자동 로드)

        Returns:
            o1_deep_analysis.json 형식
        """
        if context is None:
            context = self._gather_context()

        today = datetime.now().strftime("%Y-%m-%d")
        context_text = self._format_context(context)

        user_prompt = f"""{O1_DEEP_PROMPT}

## 분석 날짜: {today}

## 입력 데이터:

{context_text}

위 데이터를 깊이 사고하여 거시적+미시적 분석 결과를 JSON으로 응답하세요.
"""

        logger.info("o1 Deep Thinking 분석 시작 (모델: %s)", self.model)

        try:
            timeout_sec = self._load_timeout()
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        # o1은 system 미지원 → developer role 사용
                        {
                            "role": "developer",
                            "content": (
                                "You are a world-class macro/micro market analyst. "
                                "Respond in Korean. Output JSON only."
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=8000,
                    # o1은 temperature, top_p 미지원 → 생략
                ),
                timeout=timeout_sec,
            )
            result = json.loads(response.choices[0].message.content)
        except asyncio.TimeoutError:
            logger.error("o1 타임아웃 (%ds)", timeout_sec)
            return self._fallback_result("타임아웃")
        except Exception as e:
            logger.error("o1 분석 실패: %s", e)
            return self._fallback_result(str(e))

        result.setdefault("analysis_date", today)
        macro = result.get("macro_analysis", {})
        logger.info(
            "o1 Deep Thinking 완료: macro_regime=%s, confidence=%.0f%%",
            macro.get("macro_regime", "?"),
            macro.get("confidence", 0) * 100,
        )
        return result

    def _gather_context(self) -> dict:
        """분석에 필요한 데이터 자동 수집."""
        return {
            "overnight": _load_json("us_market/overnight_signal.json"),
            "intelligence": _load_json("market_intelligence.json"),
            "news": _load_json("market_news.json"),
            "sector_momentum": _load_json("sector_rotation/sector_momentum.json"),
            "regime_macro": _load_json("regime_macro_signal.json"),
        }

    @staticmethod
    def _format_context(ctx: dict) -> str:
        """컨텍스트를 텍스트로 포매팅 (토큰 효율적)."""
        sections = []

        # overnight signal
        ov = ctx.get("overnight", {})
        if ov:
            sections.append(
                f"[US Overnight] 등급: {ov.get('grade', 'N/A')}, "
                f"L1점수: {ov.get('l1_score', 0):.1f}"
            )
            changes = ov.get("changes", {})
            if changes:
                chg_lines = [f"  {k}: {v:+.2f}%" for k, v in changes.items()
                             if isinstance(v, (int, float))]
                sections.extend(chg_lines[:8])

            rules = ov.get("triggered_rules", [])
            if rules:
                sections.append(f"  특수 룰: {', '.join(rules)}")

        # Perplexity intelligence
        intel = ctx.get("intelligence", {})
        if intel:
            sections.append(
                f"[Perplexity Intel] {intel.get('us_market_summary', '')[:300]}"
            )
            for ev in intel.get("key_events", [])[:5]:
                sections.append(
                    f"  이벤트: {ev.get('event', '')} — {ev.get('detail', '')[:100]}"
                )
            sections.append(
                f"  내일전망: {intel.get('kr_open_forecast', 'N/A')} "
                f"({intel.get('kr_forecast_reason', '')[:100]})"
            )

        # sector momentum
        sm = ctx.get("sector_momentum", {})
        if sm:
            rankings = sm.get("rankings", sm.get("sectors", []))[:10]
            if rankings:
                sections.append("[섹터 모멘텀 순위]")
                for i, r in enumerate(rankings, 1):
                    if isinstance(r, dict):
                        sections.append(
                            f"  {i}. {r.get('name', r.get('sector', '?'))}: "
                            f"{r.get('momentum_score', 0):.1f}"
                        )

        # regime macro
        rm = ctx.get("regime_macro", {})
        if rm:
            sections.append(
                f"[매크로 레짐] regime={rm.get('regime', 'N/A')}, "
                f"kospi={rm.get('kospi_regime', 'N/A')}"
            )

        # RSS news (주요 뉴스만)
        news = ctx.get("news", {})
        if news:
            articles = news.get("articles", [])
            high = [a for a in articles if a.get("impact") == "high"][:5]
            if high:
                sections.append("[핵심 뉴스]")
                for a in high:
                    sections.append(f"  - {a.get('title', '')[:80]}")

        return "\n".join(sections) if sections else "데이터 없음"

    @staticmethod
    def _load_timeout() -> int:
        try:
            import yaml
            settings_path = PROJECT_ROOT / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_upgrade", {}).get("o1_timeout_sec", 120)
        except Exception:
            return 120

    @staticmethod
    def _fallback_result(error_msg: str) -> dict:
        """o1 장애 시 빈 결과 (Phase 1은 기존대로 독립 동작)."""
        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "error": error_msg,
            "macro_analysis": {},
            "micro_analysis": [],
            "cross_sector_dynamics": "",
            "risk_scenarios": [],
            "actionable_summary": "o1 분석 실패 — Phase 1 독립 판단으로 전환",
        }
