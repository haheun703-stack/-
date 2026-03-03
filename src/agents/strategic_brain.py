"""v3 Agent 2A — 산업 구조 해석 전략가 (Strategic Brain)

v3의 핵심 두뇌. 기존 news_brain + macro_analyst를 결합한 상위 레이어.
"왜 오르는가"를 해석하고 "무엇이 더 오를 것인가"를 판단합니다.

판단 우선순위:
  1순위: 산업 구조 분석 (수요/공급/CAPA/글로벌 파급)
  2순위: 밸류에이션 판단 (PER/PEG/글로벌 Peer 비교)
  3순위: 거시 환경 (금리/환율/지정학/정책)
  4순위: 기술적 확인 (차트/수급은 마지막 확인 수단)

입력:
  - overnight_signal.json (미장 시그널)
  - ai_brain_judgment.json (news_brain 결과)
  - sector_momentum.json (섹터 ETF 모멘텀)
  - relay_pattern_db.json (릴레이 패턴 DB)
  - positions.json (현재 포트폴리오)

출력:
  - ai_strategic_analysis.json (레짐 + 산업 thesis + 섹터 우선순위)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ─── v3 전략 두뇌 시스템 프롬프트 (Section 12 기반) ─────────────

SYSTEM_STRATEGIC_ANALYSIS = """\
당신은 한국 주식시장 전문 전략가입니다.
"왜 오르는가"를 해석하고 "무엇이 더 오를 것인가"를 판단합니다.

## 판단 우선순위 (반드시 이 순서대로)

1순위: 산업 구조 분석 (수요/공급/CAPA/글로벌 파급)
2순위: 밸류에이션 판단 (PER/PEG/글로벌 Peer 비교)
3순위: 거시 환경 (금리/환율/지정학/정책)
4순위: 기술적 확인 (차트/수급은 마지막 확인 수단)

★ 중요: 백테스팅 데이터는 "이 경로가 과거에도 작동했는가"를
   확인하는 보조 도구입니다. 판단의 주체가 아닙니다.

## 산업 구조 분석 체크리스트

각 주요 섹터에 대해 반드시 답하세요:
  Q1. 수요 vs 공급: 현재 수급 밸런스는? 공급 부족인가 과잉인가?
  Q2. CAPA: 주요 기업의 생산능력 확장 계획은? 병목은 어디인가?
  Q3. ASP (평균 판매가): 상승 추세인가 하락 추세인가?
  Q4. 글로벌 파급: 미국/유럽/중국의 같은 섹터 동향은?
  Q5. 이벤트: 전쟁, 관세, 규제, 정책 변화의 영향은?
  Q6. PER 리레이팅: 이익성장률 대비 현재 밸류에이션은 적정한가?

## 릴레이 경로 확인

아래 relay_pattern_db의 패턴을 참조하여:
  - 오늘 발화한 섹터 → 후행 섹터 경로가 과거에 검증되었는가?
  - 검증됨(win_rate >= 70%) → confidence 유지 또는 +1
  - 미검증(신규 패턴) → "신규 패턴, 주의" 태그 부여
  - 과거 반대 결과 → confidence -1, 비중 축소 권고

## 레짐 분류 기준

- **공격**: 글로벌 우호 + 산업 구조 양호 + 수급 긍정 → 최대 3종목 매수
- **중립**: 혼조세 또는 일부 약세 요인 → 1~2종목 선별 매수
- **방어**: 다수 약세 요인 → 신규 매수 축소, 현금 확대
- **회피**: 위기 징후 + 글로벌 급락 → 매수 중단, 현금 100%

## 피드백 반영

{feedback_section}

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{
  "analysis_date": "YYYY-MM-DD",
  "regime": "공격|중립|방어|회피",
  "regime_confidence": 0.5~1.0,
  "regime_reasoning": "레짐 판단 근거 2~3문장",
  "industry_thesis": [
    {
      "sector": "섹터명",
      "thesis": "왜 이 섹터가 유망한가 1~2문장",
      "demand_supply": "공급부족|균형|공급과잉",
      "asp_trend": "상승|횡보|하락",
      "global_alignment": "동조|독립|역행",
      "confidence": 1~10,
      "catalysts": ["촉매1", "촉매2"],
      "risks": ["리스크1"],
      "relay_source": "릴레이 소스 섹터 (해당 시)",
      "relay_verified": true
    }
  ],
  "sector_priority": {
    "attack": ["공격 매수할 섹터"],
    "watch": ["관찰할 섹터"],
    "avoid": ["회피할 섹터"]
  },
  "max_new_buys": 0~3,
  "risk_factors": ["리스크1", "리스크2"],
  "cash_reserve_suggestion": 15~100,
  "global_summary": "글로벌 시장 요약 1~2문장",
  "relay_alerts": [
    {
      "pattern": "소스→타겟",
      "status": "발화|대기|완료",
      "action": "매수 준비|주의|무시"
    }
  ]
}
```
"""


# ─── 데이터 포매팅 함수 ─────────────────────────────────────────


def _format_overnight_signal(data: dict) -> str:
    """US overnight signal → 텍스트"""
    if not data:
        return "[미장 시그널] 데이터 없음 (stale)"

    lines = ["[미장 시그널]"]
    grade = data.get("grade", "N/A")
    score = data.get("l1_score", 0)
    lines.append(f"등급: {grade} (L1 점수: {score:.1f})")

    # 개별 지수 변화
    changes = data.get("changes", {})
    if changes:
        for ticker, chg in changes.items():
            if isinstance(chg, (int, float)):
                lines.append(f"  {ticker}: {chg:+.2f}%")

    # 특수 룰
    rules = data.get("triggered_rules", [])
    if rules:
        lines.append(f"특수 룰 발동: {', '.join(rules)}")

    # 섹터 Kill
    kills = data.get("sector_kills", [])
    if kills:
        lines.append(f"섹터 Kill: {', '.join(kills)}")

    return "\n".join(lines)


def _format_news_judgment(data: dict) -> str:
    """AI Brain 뉴스 판단 결과 → 텍스트"""
    if not data:
        return "[뉴스 AI 판단] 데이터 없음"

    lines = ["[뉴스 AI 판단]"]
    sentiment = data.get("market_sentiment", "neutral")
    themes = data.get("key_themes", [])
    lines.append(f"시장 센티먼트: {sentiment}")
    if themes:
        lines.append(f"핵심 테마: {', '.join(themes[:5])}")

    # 섹터 아웃룩
    sector_outlook = data.get("sector_outlook", {})
    if sector_outlook:
        lines.append("\n[섹터 전망]")
        for sector, outlook in sector_outlook.items():
            if isinstance(outlook, dict):
                direction = outlook.get("direction", "neutral")
                reason = outlook.get("reason", "")
                lines.append(f"  {sector}: {direction} — {reason}")
            else:
                lines.append(f"  {sector}: {outlook}")

    # BUY/AVOID 판단 요약
    judgments = data.get("stock_judgments", [])
    if judgments:
        buys = [j for j in judgments if j.get("action") == "BUY"]
        avoids = [j for j in judgments if j.get("action") == "AVOID"]
        if buys:
            buy_names = [f"{j.get('name', '?')}({j.get('confidence', 0):.1f})" for j in buys[:5]]
            lines.append(f"\nAI BUY: {', '.join(buy_names)}")
        if avoids:
            avoid_names = [j.get("name", "?") for j in avoids[:5]]
            lines.append(f"AI AVOID: {', '.join(avoid_names)}")

    return "\n".join(lines)


def _format_sector_momentum(data: dict) -> str:
    """섹터 모멘텀 JSON → 텍스트"""
    if not data:
        return "[섹터 모멘텀] 데이터 없음"

    lines = ["[섹터 모멘텀 순위]"]

    rankings = data.get("rankings", data.get("sectors", []))
    if isinstance(rankings, list):
        for i, item in enumerate(rankings[:10], 1):
            if isinstance(item, dict):
                name = item.get("name", item.get("sector", "?"))
                score = item.get("momentum_score", item.get("score", 0))
                ret_20d = item.get("return_20d", item.get("change_20d_pct", 0))
                lines.append(f"  {i}. {name}: 점수 {score:.1f}, 20일 수익 {ret_20d:+.1f}%")
            elif isinstance(item, str):
                lines.append(f"  {i}. {item}")

    return "\n".join(lines)


def _format_relay_patterns(data: dict) -> str:
    """릴레이 패턴 DB → 텍스트"""
    if not data:
        return "[릴레이 패턴] 데이터 없음"

    lines = ["[릴레이 패턴 DB]"]

    patterns = data.get("patterns", [])
    active = [p for p in patterns if p.get("status") == "active"]
    for p in active:
        src = p.get("source_sector", "?")
        tgt = p.get("target_sector", "?")
        wr = p.get("win_rate", 0)
        conf = p.get("confidence", "?")
        lag = p.get("avg_lag_days", 0)
        lines.append(f"  {src} → {tgt}: 승률 {wr:.1f}%, 신뢰도 {conf}, 지연 {lag}일")

    triggers = data.get("global_triggers", [])
    if triggers:
        lines.append("\n[글로벌 트리거]")
        for t in triggers:
            src = t.get("source", "?")
            thresh = t.get("threshold", "?")
            target = t.get("target_sector", "?")
            acc = t.get("historical_accuracy", "?")
            lines.append(f"  {src} {thresh} → {target} (적중률 {acc})")

    return "\n".join(lines)


def _format_portfolio(data: dict | list) -> str:
    """현재 포트폴리오 → 텍스트"""
    if not data:
        return "[현재 포트폴리오] 보유 없음"

    lines = ["[현재 포트폴리오]"]

    positions = data if isinstance(data, list) else data.get("positions", [])
    if not positions:
        lines.append("보유 종목 없음 (현금 100%)")
        return "\n".join(lines)

    for pos in positions:
        name = pos.get("name", pos.get("stock_name", "?"))
        code = pos.get("code", pos.get("ticker", "?"))
        pnl = pos.get("pnl_pct", pos.get("unrealized_pnl_pct", 0))
        weight = pos.get("weight_pct", pos.get("allocation_pct", 0))
        sector = pos.get("sector", "?")
        lines.append(
            f"  {name}({code}): {pnl:+.1f}%, 비중 {weight:.0f}%, 섹터: {sector}"
        )

    return "\n".join(lines)


# ─── 에이전트 구현 ─────────────────────────────────────────────

class StrategicBrainAgent(BaseAgent):
    """v3 Agent 2A — 거시 해석 + 산업 구조 분석 (Opus)

    기존 news_brain + macro_analyst의 상위 레이어.
    개별 뉴스 판단이 아닌 '산업 구조 해석'에 초점.
    """

    def __init__(self, model: str | None = None):
        # settings.yaml에서 모델 읽기, 기본값은 Opus
        if model is None:
            model = self._load_model_from_settings()
        super().__init__(model=model)

    async def _ask_claude(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
        """Opus는 스트리밍 필수 — BaseAgent 오버라이드"""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            return await stream.get_final_text()

    @staticmethod
    def _load_model_from_settings() -> str:
        """settings.yaml에서 strategic_model 로드"""
        try:
            import yaml
            settings_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_brain_v3", {}).get("strategic_model", "claude-opus-4-20250514")
        except Exception:
            return "claude-opus-4-20250514"

    async def analyze(self, context: dict) -> dict:
        """5개 소스를 종합 분석하여 전략적 판단 생성.

        Args:
            context: {
                "global_market": overnight_signal.json 데이터,
                "news": ai_brain_judgment.json 데이터,
                "sector_flow": sector_momentum.json 데이터,
                "relay_patterns": relay_pattern_db.json 데이터,
                "portfolio": positions.json 데이터,
                "feedback": (선택) 지난주 피드백 텍스트,
            }

        Returns:
            ai_strategic_analysis.json 형식의 dict
        """
        # 5개 소스 포매팅
        overnight_text = _format_overnight_signal(context.get("global_market", {}))
        news_text = _format_news_judgment(context.get("news", {}))
        sector_text = _format_sector_momentum(context.get("sector_flow", {}))
        relay_text = _format_relay_patterns(context.get("relay_patterns", {}))
        portfolio_text = _format_portfolio(context.get("portfolio", {}))

        # 피드백 섹션 (있으면 주입, 없으면 빈 문자열)
        feedback = context.get("feedback", "")
        feedback_section = ""
        if feedback:
            feedback_section = f"""지난주 AI 판단 피드백:
{feedback}
위 피드백을 참고하여 이번 판단에 반영하세요."""
        else:
            feedback_section = "아직 피드백 데이터가 없습니다. 초기 판단을 수행하세요."

        # 시스템 프롬프트에 피드백 주입
        system_prompt = SYSTEM_STRATEGIC_ANALYSIS.replace(
            "{feedback_section}", feedback_section
        )

        # o1 Deep Thinking 분석 결과 (Phase 0, 있으면 주입)
        o1_text = ""
        o1_data = context.get("o1_deep_analysis", {})
        if o1_data and not o1_data.get("error"):
            o1_text = self._format_o1_analysis(o1_data)

        # 유저 프롬프트 조합
        today = datetime.now().strftime("%Y-%m-%d")
        user_prompt = f"""\
## 분석 날짜: {today}

{o1_text}

{overnight_text}

{news_text}

{sector_text}

{relay_text}

{portfolio_text}

위 소스를 종합 분석하여 오늘의 전략 판단을 JSON으로 응답하세요.
산업 구조(수요/공급/CAPA)를 1순위로, 거시 환경을 2순위로 판단하세요.
릴레이 패턴 DB를 참조하여 발화 섹터의 후행 경로를 확인하세요.
"""

        logger.info(
            "v3 Strategic Brain 분석 시작 (모델: %s)", self.model
        )

        try:
            result = await self._ask_claude_json(system_prompt, user_prompt)
        except Exception as e:
            logger.error("v3 Strategic Brain 분석 실패: %s", e)
            return self._fallback_result(str(e))

        # 날짜 보정
        result.setdefault("analysis_date", today)

        # 유효성 검증
        result = self._validate_result(result)

        logger.info(
            "v3 Strategic Brain 분석 완료: 레짐=%s, thesis=%d개, 최대매수=%d",
            result.get("regime", "?"),
            len(result.get("industry_thesis", [])),
            result.get("max_new_buys", 0),
        )

        return result

    @staticmethod
    def _validate_result(result: dict) -> dict:
        """결과 유효성 검증 + 기본값 보정"""
        # regime 검증
        valid_regimes = {"공격", "중립", "방어", "회피"}
        if result.get("regime") not in valid_regimes:
            logger.warning(
                "유효하지 않은 레짐 '%s' → '중립'으로 보정",
                result.get("regime"),
            )
            result["regime"] = "중립"

        # max_new_buys 클램핑
        max_buys = result.get("max_new_buys", 2)
        result["max_new_buys"] = max(0, min(3, max_buys))

        # cash_reserve_suggestion 클램핑
        cash = result.get("cash_reserve_suggestion", 20)
        result["cash_reserve_suggestion"] = max(10, min(100, cash))

        # industry_thesis confidence 클램핑
        for thesis in result.get("industry_thesis", []):
            conf = thesis.get("confidence", 5)
            thesis["confidence"] = max(1, min(10, conf))

        # 필수 필드 기본값
        result.setdefault("regime_confidence", 0.5)
        result.setdefault("regime_reasoning", "")
        result.setdefault("industry_thesis", [])
        result.setdefault("sector_priority", {"attack": [], "watch": [], "avoid": []})
        result.setdefault("risk_factors", [])
        result.setdefault("relay_alerts", [])
        result.setdefault("global_summary", "")

        return result

    @staticmethod
    def _format_o1_analysis(data: dict) -> str:
        """o1 Deep Thinking 결과를 프롬프트 텍스트로 변환."""
        lines = ["[o1 Deep Thinking 거시/미시 분석]"]

        macro = data.get("macro_analysis", {})
        if macro:
            lines.append(f"  매크로 레짐: {macro.get('macro_regime', 'N/A')}")
            lines.append(f"  글로벌 유동성: {macro.get('global_liquidity', 'N/A')}")
            lines.append(f"  동조 상태: {macro.get('coupling_status', 'N/A')}")
            lines.append(f"  사이클 위치: {macro.get('cycle_position', 'N/A')}")
            lines.append(f"  지정학 리스크: {macro.get('geopolitical_risk_score', 'N/A')}/10")
            lines.append(f"  신뢰도: {macro.get('confidence', 0):.0%}")
            insights = macro.get("key_macro_insights", [])
            if insights:
                lines.append("  핵심 인사이트:")
                for ins in insights[:3]:
                    lines.append(f"    - {ins}")

        micros = data.get("micro_analysis", [])
        if micros:
            lines.append("\n  [섹터별 미시 분석]")
            for m in micros[:5]:
                lines.append(
                    f"  {m.get('sector', '?')}: "
                    f"실적={m.get('earnings_cycle', '?')}, "
                    f"밸류={m.get('valuation_rerating', '?')}, "
                    f"수급={m.get('supply_demand_shift', '?')}, "
                    f"확신={m.get('conviction', 0)}/10"
                )
                contrarian = m.get("contrarian_view", "")
                if contrarian:
                    lines.append(f"    → 역발상: {contrarian}")

        cross = data.get("cross_sector_dynamics", "")
        if cross:
            lines.append(f"\n  섹터간 역학: {cross[:200]}")

        risks = data.get("risk_scenarios", [])
        if risks:
            lines.append("\n  [리스크 시나리오]")
            for r in risks[:3]:
                lines.append(
                    f"    - {r.get('scenario', '?')} "
                    f"(확률 {r.get('probability', 0):.0%}, "
                    f"영향 {r.get('impact', '?')})"
                )

        summary = data.get("actionable_summary", "")
        if summary:
            lines.append(f"\n  ★ 핵심 요약: {summary}")

        return "\n".join(lines)

    @staticmethod
    def _fallback_result(error_msg: str) -> dict:
        """AI 장애 시 안전한 방어 모드 결과"""
        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "regime": "방어",
            "regime_confidence": 0.3,
            "regime_reasoning": f"AI 분석 실패 — 방어 모드 자동 전환. 오류: {error_msg}",
            "industry_thesis": [],
            "sector_priority": {"attack": [], "watch": [], "avoid": []},
            "max_new_buys": 0,
            "risk_factors": ["AI 분석 장애 — 신규 매수 중단"],
            "cash_reserve_suggestion": 50,
            "global_summary": "AI 분석 실패로 판단 불가",
            "relay_alerts": [],
            "error": error_msg,
        }
