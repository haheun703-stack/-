"""Perplexity 교차검증 에이전트 — Phase 6 팩트체크

v3 Brain Phase 5 결과(ai_v3_picks.json)의 주요 주장을
Perplexity 실시간 웹검색으로 교차검증.

검증 대상:
  1. 종목별 촉매(catalysts)가 실제로 존재하는가?
  2. 리스크(risks)가 실제인가, AI 환각인가?
  3. 산업 thesis의 수요/공급 주장이 사실인가?

입력:
  - ai_v3_picks.json (Phase 5 결과)
  - ai_strategic_analysis.json (Phase 1 결과)

출력:
  - data/perplexity_verification.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityVerifier:
    """Perplexity 기반 AI 판단 교차검증."""

    def __init__(
        self,
        model: str | None = None,
        max_verifications: int = 5,
        max_thesis_checks: int = 3,
    ):
        if model is None:
            model = self._load_model_from_settings()
        self.model = model
        self.max_verifications = max_verifications
        self.max_thesis_checks = max_thesis_checks

    @staticmethod
    def _load_model_from_settings() -> str:
        try:
            import yaml
            settings_path = PROJECT_ROOT / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_upgrade", {}).get("perplexity_model", "sonar")
        except Exception:
            return "sonar"

    def _query(self, prompt: str) -> dict | None:
        """Perplexity API 호출 (동기)."""
        if not PERPLEXITY_API_KEY:
            logger.error("PERPLEXITY_API_KEY 미설정")
            return None

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "당신은 한국 주식시장 팩트체크 전문가입니다. "
                        "AI가 분석한 주장을 실시간 웹검색으로 교차검증합니다. "
                        "반드시 JSON 형식으로만 응답하세요."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 3000,
        }

        try:
            resp = requests.post(
                PERPLEXITY_URL, json=payload, headers=headers, timeout=60
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # JSON 파싱 (코드블록 제거)
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            content = content.strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("Perplexity JSON 파싱 실패: %s", e)
            return None
        except Exception as e:
            logger.error("Perplexity API 오류: %s", e)
            return None

    def verify_picks(self, picks: dict, strategic: dict) -> dict:
        """Phase 5 결과를 교차검증.

        Args:
            picks: ai_v3_picks.json
            strategic: ai_strategic_analysis.json

        Returns:
            perplexity_verification.json 형식
        """
        today = datetime.now().strftime("%Y-%m-%d")
        verifications = []
        thesis_verifications = []

        # 1. 종목별 촉매/리스크 검증
        buys = picks.get("buys", [])[:self.max_verifications]
        for buy in buys:
            logger.info("종목 검증: %s(%s)", buy.get("name", "?"), buy.get("ticker", "?"))
            result = self._verify_stock(buy)
            if result:
                verifications.append(result)

        # 2. 산업 thesis 검증
        theses = strategic.get("industry_thesis", [])[:self.max_thesis_checks]
        for thesis in theses:
            logger.info("Thesis 검증: %s", thesis.get("sector", "?"))
            result = self._verify_thesis(thesis)
            if result:
                thesis_verifications.append(result)

        # 3. 종합 신뢰도
        stock_scores = [v.get("confidence_score", 0.5) for v in verifications]
        thesis_scores = [v.get("confidence_score", 0.5) for v in thesis_verifications]
        all_scores = stock_scores + thesis_scores
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.5

        output = {
            "verification_date": today,
            "verified_at": datetime.now().strftime("%H:%M"),
            "overall_confidence": round(overall, 2),
            "stock_verifications": verifications,
            "thesis_verifications": thesis_verifications,
            "warnings": self._collect_warnings(verifications, thesis_verifications),
            "hallucination_flags": [
                v for v in verifications
                if v.get("confidence_score", 1.0) < 0.3
            ],
        }

        logger.info(
            "교차검증 완료: 종합 신뢰도 %.0f%%, 종목 %d건, thesis %d건",
            overall * 100, len(verifications), len(thesis_verifications),
        )
        return output

    def _verify_stock(self, buy: dict) -> dict | None:
        """단일 종목 촉매/리스크 검증."""
        name = buy.get("name", "?")
        ticker = buy.get("ticker", "?")
        catalysts = buy.get("catalysts", [])
        risks = buy.get("risks", [])
        reasoning = buy.get("reasoning", "")

        today = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""오늘 날짜: {today}

AI가 다음 종목에 대해 매수 판단을 내렸습니다. 주장의 사실 여부를 웹검색으로 검증하세요.

종목: {name} ({ticker})
AI 판단 근거: {reasoning}
주장된 촉매: {', '.join(catalysts) if catalysts else '없음'}
주장된 리스크: {', '.join(risks) if risks else '없음'}

검증 결과를 다음 JSON으로 응답:
{{
  "ticker": "{ticker}",
  "name": "{name}",
  "catalyst_verification": [
    {{"claim": "AI 주장", "verified": true, "evidence": "검증 근거", "source": "출처"}}
  ],
  "risk_verification": [
    {{"claim": "AI 주장 리스크", "verified": true, "actual_status": "실제 상황"}}
  ],
  "additional_findings": ["AI가 놓친 중요 사실"],
  "confidence_score": 0.7,
  "verdict": "CONFIRMED"
}}

verdict 옵션: CONFIRMED | PARTIALLY_CONFIRMED | UNVERIFIED | HALLUCINATION_DETECTED"""

        return self._query(prompt)

    def _verify_thesis(self, thesis: dict) -> dict | None:
        """산업 thesis 검증."""
        sector = thesis.get("sector", "?")
        thesis_text = thesis.get("thesis", "?")
        demand_supply = thesis.get("demand_supply", "?")
        asp_trend = thesis.get("asp_trend", "?")

        today = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""오늘 날짜: {today}

AI가 다음 산업 분석을 제시했습니다. 웹검색으로 사실 여부를 검증하세요.

섹터: {sector}
AI Thesis: {thesis_text}
수요/공급: {demand_supply}
ASP 추세: {asp_trend}

검증 결과를 다음 JSON으로 응답:
{{
  "sector": "{sector}",
  "thesis_verified": true,
  "demand_supply_check": "AI 주장과 실제 비교",
  "asp_check": "ASP 실제 추세 확인",
  "recent_data_points": ["최신 데이터 포인트"],
  "confidence_score": 0.8,
  "verdict": "CONFIRMED"
}}

verdict 옵션: CONFIRMED | PARTIALLY_CONFIRMED | UNVERIFIED | OUTDATED"""

        return self._query(prompt)

    @staticmethod
    def _collect_warnings(stock_v: list, thesis_v: list) -> list[str]:
        """검증 결과에서 경고 수집."""
        warnings = []
        for v in stock_v:
            if v.get("verdict") == "HALLUCINATION_DETECTED":
                warnings.append(
                    f"환각 감지: {v.get('name', '?')} — AI 촉매 주장이 사실과 다름"
                )
            elif v.get("confidence_score", 1.0) < 0.4:
                warnings.append(
                    f"검증 실패: {v.get('name', '?')} — 촉매/리스크 확인 불가"
                )
        for v in thesis_v:
            if v.get("verdict") == "OUTDATED":
                warnings.append(
                    f"오래된 정보: {v.get('sector', '?')} — thesis 근거가 최신이 아님"
                )
        return warnings
