"""GPT-4o 뉴스 촉매 분석 에이전트 — 매도 판단의 "내일 전망" 담당

역할 분담 모델:
  - GPT-4o: 뉴스 촉매 분석 + 내일/모레 전망 (이 파일)
  - Claude Sonnet: 기술적 분석 + 포트폴리오 맥락 (sell_brain.py)

보유 종목별로 "이 촉매가 내일도 살아있는가?"를 판단한다.

입력:
  - positions (현재 보유종목)
  - 5개 뉴스 소스 JSON 재활용 (ai_filter.py 패턴)

출력:
  - gpt_catalyst_analysis.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYSTEM_GPT_CATALYST = """\
당신은 한국 주식시장 뉴스 촉매 분석 전문가입니다.
보유 종목의 뉴스 촉매를 분석하여 "내일도 유효한가?"를 판단합니다.

## 핵심 질문 (종목별 반드시 답할 것)
1. 현재 뉴스 촉매가 무엇인가? (구체적 이벤트/공시/정책)
2. 이 촉매의 수명은? (1일 이벤트 vs 구조적 변화)
3. 내일(D+1) 이 촉매가 추가 상승/하락을 이끌 가능성은?
4. 모레(D+2) 이후 전망은?
5. 촉매 소멸/반전 시그널이 있는가?

## 판단 기준
- CATALYST_ALIVE: 촉매 생존. 내일 추가 상승(또는 방어) 가능성 50% 이상
- CATALYST_FADING: 촉매 약화 중. 1-2일 내 소멸 예상
- CATALYST_DEAD: 촉매 소멸 또는 이미 가격에 반영 완료
- CATALYST_NEW: 기존에 없던 새 촉매 발견 (긍정 또는 부정)

## 핵심 원칙
- "뉴스에 의한 움직임"과 "기술적 반등"을 구분하라
- 지정학 이벤트(전쟁, 제재)는 수일간 지속되는 경우가 많다
- 실적 발표/공시는 당일 반영 후 빠르게 소멸되는 경우가 많다
- 구조적 변화(HBM 수퍼사이클, 글로벌 군비 확장)는 수개월 지속

## 출력 형식
반드시 아래 JSON 형식으로 응답하세요.

{
  "analysis_time": "HH:MM",
  "market_catalyst_summary": "전체 시장 촉매 1줄 요약",
  "positions": [
    {
      "ticker": "종목코드",
      "name": "종목명",
      "catalyst_status": "CATALYST_ALIVE|CATALYST_FADING|CATALYST_DEAD|CATALYST_NEW",
      "catalyst_strength": 0.0~1.0,
      "primary_catalyst": "핵심 촉매 설명",
      "catalyst_lifespan": "예상 수명 (예: 3-5일, 1주+, 구조적)",
      "tomorrow_direction": "UP|DOWN|SIDEWAYS|UNCERTAIN",
      "tomorrow_confidence": 0.0~1.0,
      "day_after_direction": "UP|DOWN|SIDEWAYS|UNCERTAIN",
      "news_summary": "관련 뉴스 핵심 요약 1-2문장",
      "catalyst_risks": ["소멸 리스크1", "반전 리스크2"],
      "new_catalysts": ["새로 발견된 촉매 (있으면)"]
    }
  ]
}
"""


def _load_json(name: str) -> dict | list:
    path = DATA_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("JSON 로드 실패 %s: %s", name, e)
            return {}
    return {}


def build_sell_news_context(positions: list[dict], max_items: int = 40) -> str:
    """보유 종목 관련 뉴스 맥락 수집.

    5개 소스 재활용 (ai_filter.py build_news_context() 패턴):
      1. market_intelligence.json (Perplexity)
      2. market_news.json (RSS)
      3. ai_brain_judgment.json (종목별 BUY/WATCH/AVOID)
      4. dart_disclosures.json (공시)
      5. us_market/overnight_signal.json
    """
    # 보유 종목명/코드 세트 (관련 뉴스 우선 필터링용)
    held_names = {p.get("name", "") for p in positions}
    held_tickers = {p.get("ticker", "") for p in positions}

    priority_lines = []  # 보유 종목 관련
    general_lines = []   # 일반 시장

    # 1. Perplexity 인텔리전스 — 가장 유용
    intel = _load_json("market_intelligence.json")
    if intel:
        us_summary = intel.get("us_market_summary", "")
        if us_summary:
            general_lines.append(f"[미국장] {us_summary[:200]}")

        kr_forecast = intel.get("kr_open_forecast", "")
        kr_reason = intel.get("kr_forecast_reason", "")
        if kr_forecast:
            general_lines.append(f"[내일전망] {kr_forecast}: {kr_reason[:150]}")

        for ev in intel.get("key_events", [])[:5]:
            line = f"[이벤트] {ev.get('event', '')} — {ev.get('detail', '')[:100]}"
            # 보유 종목 섹터와 관련?
            affected = ev.get("kr_sectors_affected", [])
            if any(name in str(affected) for name in held_names):
                priority_lines.append(line)
            else:
                general_lines.append(line)

        for sec in intel.get("sector_impacts", [])[:5]:
            line = f"[섹터:{sec.get('sector', '')}] {sec.get('impact', '')} {sec.get('reason', '')[:100]}"
            top_stocks = sec.get("top_stocks", [])
            if any(name in str(top_stocks) for name in held_names):
                priority_lines.append(line)
            else:
                general_lines.append(line)

        # beneficiary / risk stocks
        for bene in intel.get("beneficiary_stocks", []):
            if any(name in str(bene) for name in held_names):
                priority_lines.append(f"[수혜종목] {bene}")
        for risk in intel.get("risk_stocks", []):
            if any(name in str(risk) for name in held_names):
                priority_lines.append(f"[리스크종목] {risk}")

    # 2. RSS 뉴스 — high/medium만
    news = _load_json("market_news.json")
    for a in (news.get("articles", []) if isinstance(news, dict) else []):
        if a.get("impact") not in ("high", "medium"):
            continue
        title = a.get("title", "")
        line = f"[뉴스] {title}"
        if any(name in title for name in held_names):
            priority_lines.append(line)
        else:
            general_lines.append(line)

    # 3. AI 두뇌 — 종목별 촉매 직접 참조
    ai_brain = _load_json("ai_brain_judgment.json")
    if ai_brain:
        sentiment = ai_brain.get("market_sentiment", "")
        if sentiment:
            general_lines.append(f"[AI센티먼트] {sentiment}")

        for sj in ai_brain.get("stock_judgments", []):
            ticker = sj.get("ticker", "")
            name = sj.get("name", "")
            if ticker in held_tickers or name in held_names:
                action = sj.get("action", "")
                catalysts = sj.get("catalysts", [])
                risks = sj.get("risks", [])
                priority_lines.append(
                    f"[AI종목:{name}] {action} | 촉매: {', '.join(catalysts[:3])} | 리스크: {', '.join(risks[:2])}"
                )

        for sec, info in ai_brain.get("sector_outlook", {}).items():
            if isinstance(info, dict):
                general_lines.append(
                    f"[섹터전망:{sec}] {info.get('direction', '')} — {info.get('reason', '')[:80]}"
                )

    # 4. DART 공시 — tier1만
    dart = _load_json("dart_disclosures.json")
    for d in (dart.get("tier1", []) if isinstance(dart, dict) else []):
        corp = d.get("corp_name", "")
        line = f"[DART] {corp} — {d.get('report_nm', '')}"
        if corp in held_names:
            priority_lines.append(line)
        else:
            general_lines.append(line)

    # 5. US Overnight
    us = _load_json(str(Path("us_market") / "overnight_signal.json"))
    if us:
        general_lines.append(
            f"[US야간] 등급:{us.get('grade', 'N/A')} "
            f"VIX:{us.get('vix', {}).get('level', 'N/A')} "
            f"유가:{us.get('commodities', {}).get('oil', {}).get('ret_1d', 0):+.1f}%"
        )

    # 보유종목 관련 뉴스를 앞에 배치
    all_lines = priority_lines + general_lines
    if len(all_lines) > max_items:
        # 우선순위(보유종목) 뉴스는 최대한 보존
        all_lines = priority_lines[:max_items // 2] + general_lines[: max_items - len(priority_lines[:max_items // 2])]

    return "\n".join(all_lines) if all_lines else "뉴스 데이터 없음"


class GPTCatalystAgent:
    """GPT-4o 기반 뉴스 촉매 분석 — 매도 판단의 '내일 전망' 담당."""

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
            return cfg.get("dual_sell_system", {}).get("gpt_catalyst", {}).get("model", "gpt-4o")
        except Exception:
            return "gpt-4o"

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        return self._client

    async def _ask_gpt_json(self, system: str, user: str, max_tokens: int = 4000) -> dict:
        """GPT API 호출 + JSON 응답."""
        try:
            import yaml
            settings_path = PROJECT_ROOT / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            gpt_cfg = cfg.get("dual_sell_system", {}).get("gpt_catalyst", {})
            temperature = gpt_cfg.get("temperature", 0.3)
            timeout_sec = gpt_cfg.get("timeout_sec", 30)
        except Exception:
            temperature = 0.3
            timeout_sec = 30

        import asyncio
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout_sec,
        )
        return json.loads(response.choices[0].message.content)

    async def analyze_catalysts(
        self,
        positions: list[dict],
        news_context: str | None = None,
    ) -> dict:
        """보유종목별 뉴스 촉매 분석.

        Args:
            positions: 보유 포지션 리스트
            news_context: 뉴스 맥락 텍스트 (None이면 자동 수집)

        Returns:
            {analysis_time, market_catalyst_summary, positions: [...]}
        """
        if not positions:
            return self._empty_result()

        if news_context is None:
            news_context = build_sell_news_context(positions)

        # 보유 종목 요약
        pos_lines = []
        for p in positions:
            name = p.get("name", "?")
            ticker = p.get("ticker", "?")
            pnl = p.get("pnl_pct", 0)
            entry = p.get("entry_price", 0)
            current = p.get("current_price", entry)
            grade = p.get("grade", "?")
            pos_lines.append(
                f"  {name}({ticker}): 등급={grade}, 손익={pnl:+.1f}%, "
                f"진입={entry:,} → 현재={current:,}"
            )

        now = datetime.now().strftime("%H:%M")
        user_prompt = f"""\
## 보유 종목 뉴스 촉매 분석: {now}

[보유 종목 ({len(positions)}개)]
{chr(10).join(pos_lines)}

[뉴스 맥락]
{news_context}

위 뉴스와 보유 종목을 대조하여, 각 종목별 뉴스 촉매 상태를 분석하세요.
특히 "내일(D+1) 이 촉매가 추가 상승/하락을 이끌 가능성"에 집중하세요.
"""

        logger.info("GPT 촉매 분석 시작 (%d종목, 뉴스 %d줄)", len(positions), len(news_context.split("\n")))

        try:
            result = await self._ask_gpt_json(SYSTEM_GPT_CATALYST, user_prompt)
        except Exception as e:
            logger.error("GPT 촉매 분석 실패: %s", e)
            return self._fallback_result(str(e), positions)

        result.setdefault("analysis_time", now)
        logger.info(
            "GPT 촉매 분석 완료: %d종목",
            len(result.get("positions", [])),
        )
        return result

    @staticmethod
    def _empty_result() -> dict:
        return {
            "analysis_time": datetime.now().strftime("%H:%M"),
            "market_catalyst_summary": "보유 종목 없음",
            "positions": [],
        }

    @staticmethod
    def _fallback_result(error_msg: str, positions: list[dict]) -> dict:
        """GPT 장애 시 전종목 CATALYST_DEAD (보수적 — Claude 단독 판단에 위임)."""
        return {
            "analysis_time": datetime.now().strftime("%H:%M"),
            "market_catalyst_summary": f"GPT 장애: {error_msg}",
            "positions": [
                {
                    "ticker": p.get("ticker", ""),
                    "name": p.get("name", ""),
                    "catalyst_status": "CATALYST_DEAD",
                    "catalyst_strength": 0.0,
                    "primary_catalyst": "GPT 장애 — 촉매 판단 불가",
                    "catalyst_lifespan": "N/A",
                    "tomorrow_direction": "UNCERTAIN",
                    "tomorrow_confidence": 0.0,
                    "day_after_direction": "UNCERTAIN",
                    "news_summary": "",
                    "catalyst_risks": [],
                    "new_catalysts": [],
                }
                for p in positions
            ],
            "error": error_msg,
        }
