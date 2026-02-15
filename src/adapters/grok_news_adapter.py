"""
Grok Responses API 뉴스 어댑터

xAI Grok-4-1-fast 모델을 사용하여 뉴스를 검색하고
NewsSearchPort 인터페이스를 구현한다.

tools: web_search + x_search (서버사이드 자동 실행)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta

import requests

from src.use_cases.ports import NewsSearchPort

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1/responses"
MODEL = "grok-4-1-fast"


class GrokNewsAdapter(NewsSearchPort):
    """Grok Responses API 뉴스 검색 어댑터"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY", "")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ──────────────────────────────────────────
    # Port 인터페이스 구현
    # ──────────────────────────────────────────

    async def search_stock_news(self, stock_name: str, market: str = "korean") -> dict | None:
        """종목별 최신 뉴스 검색"""
        if not self.api_key:
            logger.warning("XAI_API_KEY 미설정 — 뉴스 검색 건너뜀")
            return None

        if market == "korean":
            prompt = self._korean_stock_prompt(stock_name)
        else:
            prompt = self._us_stock_prompt(stock_name)

        return self._call_api(
            system="너는 전문 주식 뉴스 분석가다. "
                   "웹과 X(트위터)를 검색해서 주식 관련 최신 뉴스를 수집하고, "
                   "호재/악재/이슈/변동으로 정확히 분류한다. "
                   "반드시 요청된 JSON 형식으로만 응답한다.",
            user=prompt,
        )

    async def search_market_overview(self) -> dict | None:
        """전체 시장 동향 요약"""
        if not self.api_key:
            return None

        prompt = (
            "오늘의 한국 및 미국 주식시장 전체 동향을 분석해줘.\n\n"
            "반드시 아래 JSON 형식으로만 응답해:\n"
            '{"date":"오늘날짜","korean_market":{"kospi":"상황","kosdaq":"상황",'
            '"hot_sectors":[],"key_news":[]},"us_market":{"sp500":"상황",'
            '"nasdaq":"상황","dow":"상황","hot_sectors":[],"key_news":[]},'
            '"global_factors":[],"overall_outlook":"전망"}'
        )
        return self._call_api(
            system="너는 글로벌 주식시장 전문 애널리스트다.",
            user=prompt,
        )

    async def search_x_sentiment(self, stock_name: str, days: int = 3) -> dict | None:
        """X(트위터) 여론/감성 분석"""
        if not self.api_key:
            return None

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        prompt = (
            f"X(트위터)에서 '{stock_name}' 관련 최근 게시물을 검색하고 여론을 분석해줘.\n"
            f"기간: {from_date} ~ {to_date}\n\n"
            "JSON 형식으로 응답:\n"
            '{"stock":"종목명","period":"기간","sentiment_score":-100~100,'
            '"positive_count":0,"negative_count":0,"neutral_count":0,'
            '"hot_topics":[],"notable_posts":[],"summary":"요약"}'
        )
        return self._call_api(
            system="너는 소셜 미디어 감성 분석 전문가다.",
            user=prompt,
            tools_config=[{"type": "x_search"}],
        )

    # ──────────────────────────────────────────
    # v3.2: 살아있는 이슈 + 실적 예상 통합 검색
    # ──────────────────────────────────────────

    async def search_deep_analysis(self, stock_name: str, ticker: str = "") -> dict | None:
        """
        종목 심층 분석: 최신 뉴스 + 살아있는 이슈 + 실적 예상 통합.
        기존 search_stock_news보다 넓은 범위를 검색.
        """
        if not self.api_key:
            logger.warning("XAI_API_KEY 미설정 — 심층 분석 건너뜀")
            return None

        prompt = self._deep_analysis_prompt(stock_name, ticker)
        return self._call_api(
            system=(
                "너는 한국 주식시장 전문 리서치 애널리스트다. "
                "웹과 X(트위터)를 검색해서 종목의 최신 뉴스뿐 아니라, "
                "아직 해소되지 않은 과거 이슈(M&A 진행 중, 정책 수혜 지속 등), "
                "실적 전망(컨센서스, 서프라이즈 가능성), 기관/외국인 수급 동향을 "
                "종합적으로 분석한다. 반드시 요청된 JSON 형식으로만 응답한다."
            ),
            user=prompt,
        )

    @staticmethod
    def _deep_analysis_prompt(stock_name: str, ticker: str = "") -> str:
        ticker_info = f" (종목코드: {ticker})" if ticker else ""
        return (
            f"다음 종목에 대해 심층 분석해줘: {stock_name}{ticker_info}\n\n"
            "## 분석 범위\n"
            "1. **최신 뉴스** (최근 72시간): 호재/악재/이슈/변동\n"
            "2. **살아있는 이슈** (과거에 시작됐지만 아직 유효한 재료):\n"
            "   - 진행 중인 M&A, 인수합병, 매각\n"
            "   - 정부 정책 수혜/규제 (아직 시행 전이거나 효과 지속 중)\n"
            "   - 대규모 수주/계약 진행 중\n"
            "   - 사업 구조조정/신사업 진출\n"
            "   - 주주환원 정책 (배당 확대, 자사주 매입)\n"
            "   - 해외 진출/파트너십\n"
            "3. **실적 전망**:\n"
            "   - 다음 실적 발표 예상 시기\n"
            "   - 매출/영업이익 컨센서스 (증권사 추정)\n"
            "   - 전년 동기 대비 성장률 예상\n"
            "   - 어닝 서프라이즈 가능성 (beat/miss/in_line)\n"
            "4. **기관/외국인 동향**: 최근 순매수/순매도 추이\n\n"
            "반드시 아래 JSON 형식으로만 응답해. 다른 텍스트 없이 순수 JSON만:\n\n"
            "{\n"
            f'  "stock": "{stock_name}",\n'
            '  "timestamp": "ISO8601",\n'
            '  "latest_news": [\n'
            '    {"title": "제목", "summary": "요약", "category": "ma_acquisition|earnings_surprise|contract_win|policy_change|shareholder_action|general_positive|general_negative|theme",\n'
            '     "impact_score": 1, "sentiment": "positive|negative|neutral", "source": "출처", "date": "날짜",\n'
            '     "is_confirmed": false, "has_specific_amount": false, "cross_verified": false}\n'
            '  ],\n'
            '  "living_issues": [\n'
            '    {"title": "이슈 제목", "category": "카테고리", "start_date": "시작일",\n'
            '     "status": "active|fading|resolved", "impact_score": 1,\n'
            '     "sentiment": "positive|negative|neutral",\n'
            '     "description": "상세 설명", "expected_resolution": "예상 해소 시점",\n'
            '     "source_count": 0}\n'
            '  ],\n'
            '  "earnings_estimate": {\n'
            '    "next_earnings_date": "YYYY-MM-DD 또는 빈문자열",\n'
            '    "days_until_earnings": -1,\n'
            '    "consensus_revenue_억": 0,\n'
            '    "consensus_op_억": 0,\n'
            '    "consensus_eps_원": 0,\n'
            '    "surprise_direction": "beat|miss|in_line|neutral",\n'
            '    "yoy_growth_pct": 0,\n'
            '    "analyst_count": 0,\n'
            '    "description": "실적 전망 요약"\n'
            '  },\n'
            '  "institutional_flow": {\n'
            '    "foreign_trend": "순매수|순매도|중립",\n'
            '    "institutional_trend": "순매수|순매도|중립",\n'
            '    "notable_activity": "특이 수급 활동 설명"\n'
            '  },\n'
            '  "overall_sentiment": "긍정|부정|중립",\n'
            '  "key_takeaway": "핵심 한줄 요약"\n'
            "}\n\n"
            "모르는 항목은 빈 값으로, 수치는 0으로 채워."
        )

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _call_api(
        self,
        system: str,
        user: str,
        tools_config: list | None = None,
    ) -> dict | None:
        """Grok Responses API 호출"""
        payload = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": tools_config or [{"type": "web_search"}, {"type": "x_search"}],
        }
        try:
            resp = requests.post(
                XAI_BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())
        except requests.exceptions.RequestException as e:
            logger.error("Grok API 호출 실패: %s", e)
            return None

    @staticmethod
    def _parse_response(response_data: dict) -> dict | None:
        """Grok Responses API 응답에서 JSON 추출"""
        output = response_data.get("output", [])
        for item in output:
            if item.get("type") == "message":
                content_blocks = item.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "output_text":
                        text = block.get("text", "")
                        return _extract_json(text)
        return None

    @staticmethod
    def _korean_stock_prompt(stock_name: str) -> str:
        return (
            f"다음 종목에 대한 최신 뉴스를 검색하고 분석해줘: {stock_name}\n\n"
            "반드시 아래 형식의 JSON으로만 응답해. 다른 텍스트 없이 순수 JSON만 출력해.\n\n"
            '{"stock":"' + stock_name + '","timestamp":"ISO8601","news":['
            '{"title":"제목","summary":"요약","category":"호재|악재|이슈|변동",'
            '"impact_score":1,"source":"출처","date":"날짜"}],'
            '"overall_sentiment":"긍정|부정|중립","key_takeaway":"핵심 한줄"}\n\n'
            "최근 24~72시간 이내 뉴스를 우선 수집하되, 중요한 뉴스는 1주일 이내도 포함."
        )

    @staticmethod
    def _us_stock_prompt(stock_name: str) -> str:
        return (
            f"Search for the latest news about: {stock_name}\n\n"
            "Respond ONLY in JSON format:\n"
            '{"stock":"' + stock_name + '","timestamp":"ISO8601","news":['
            '{"title":"headline","summary":"2-3 sentences in Korean",'
            '"category":"호재|악재|이슈|변동","impact_score":1,'
            '"source":"source","date":"date"}],'
            '"overall_sentiment":"긍정|부정|중립","key_takeaway":"핵심 요약"}'
        )


def _extract_json(text: str) -> dict | None:
    """텍스트에서 JSON 객체 추출 (코드블록 포함 처리)"""
    # ```json ... ``` 블록 추출
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)

    # { ... } 추출
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
