"""OpenAI gpt-4o-mini 공시 분류기 — DART 공시 → 촉매/소음 판정

역할 분담:
- ChatGPT(OpenAI) = DART 공시 담당
- Grok(xAI) = 뉴스 담당
- 겹치지 않게 분리

설계 4원칙:
1. JSON 강제 (response_format=json_object)
2. Confidence 0.7 미만 무시
3. LLM이 최종 매매 결정하지 않음 (점수 가중치만)
4. Fail-safe (API 장애 시 기존 로직 유지)
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# 기본 응답 (API 실패 / confidence 미달 시)
DEFAULT_RESULT = {
    "catalyst_type": "none",
    "catalyst_category": "",
    "confidence": 0.0,
}

SYSTEM_PROMPT = """\
너는 한국 주식시장 전문 공시 분석가다.
DART 전자공시 목록을 보고, 해당 종목에 대해 '선행 촉매'인지 '후행 소음'인지 판단한다.

선행 촉매 (catalyst) — 아직 주가에 미반영될 가능성이 높은 이벤트:
- earnings_surprise: 실적 서프라이즈 (영업이익 QoQ +40% 이상, 컨센서스 상회)
- new_order: 대규모 수주 공시 (매출 대비 10% 이상 신규 계약)
- buyback: 경영진/회사 자사주 매입
- short_delist: 공매도 과열 지정 해제
- ma_event: M&A, 경영권 이벤트, 최대주주 변경
- turnaround: 적자→흑자 전환 공시

후행 소음 (noise) — 이미 주가에 반영되었거나 방향성 없는 정보:
- 정기 보고서 제출 (사업보고서, 분기보고서 등 — 실적 서프라이즈 없는 경우)
- 지분 변동 단순 보고
- 임원 변경 단순 보고
- 주주총회 소집 통지

없음 (none) — 공시가 없거나 판단 불가

반드시 아래 JSON 형식으로만 응답해:
{
  "catalyst_type": "catalyst" | "noise" | "none",
  "catalyst_category": "earnings_surprise" | "new_order" | "buyback" | "short_delist" | "ma_event" | "turnaround" | "",
  "confidence": 0.0 ~ 1.0,
  "reason": "판단 근거 한줄 요약"
}"""


def classify_disclosures(
    ticker: str,
    name: str,
    filings: list[dict],
    api_key: str | None = None,
) -> dict:
    """DART 공시 목록을 gpt-4o-mini로 분류.

    Args:
        ticker: 종목코드
        name: 종목명
        filings: dart_adapter.fetch_recent_disclosures() 결과
        api_key: OpenAI API 키 (없으면 환경변수)

    Returns:
        {"catalyst_type": ..., "catalyst_category": ..., "confidence": ..., "reason": ...}
    """
    if not filings:
        return DEFAULT_RESULT.copy()

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        logger.warning("OPENAI_API_KEY 미설정 — 공시 분류 건너뜀")
        return DEFAULT_RESULT.copy()

    # 공시 목록 → 텍스트 포맷
    filing_text = "\n".join(
        f"- [{f['date']}] {f['title']}" + (f" ({f['remark']})" if f.get("remark") else "")
        for f in filings[:15]  # 최대 15건
    )

    user_prompt = f"""종목: {name} ({ticker})

최근 30일 DART 공시 목록:
{filing_text}

위 공시 중 가장 의미 있는 선행 촉매가 있는지 판단해줘."""

    try:
        import openai

        client = openai.OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
            timeout=15,
        )

        result = json.loads(response.choices[0].message.content)

        # Confidence 0.7 미만 → 무시
        if result.get("confidence", 0) < 0.7:
            logger.debug(
                f"공시 분류 confidence 미달: {ticker} {result.get('confidence', 0):.2f}"
            )
            return DEFAULT_RESULT.copy()

        # 필수 필드 검증
        if result.get("catalyst_type") not in ("catalyst", "noise", "none"):
            return DEFAULT_RESULT.copy()

        return {
            "catalyst_type": result["catalyst_type"],
            "catalyst_category": result.get("catalyst_category", ""),
            "confidence": round(result.get("confidence", 0), 2),
            "reason": result.get("reason", ""),
        }

    except Exception as e:
        # Fail-safe: API 장애 시 기존 로직 유지
        logger.warning(f"OpenAI 공시 분류 실패 ({ticker}): {e}")
        return DEFAULT_RESULT.copy()


def classify_batch(
    candidates: list[dict],
    dart_adapter,
    api_key: str | None = None,
) -> dict[str, dict]:
    """전체 후보 종목에 대해 DART 공시 조회 + 분류 일괄 실행.

    Args:
        candidates: scan_all() 결과 리스트
        dart_adapter: DartAdapter 인스턴스
        api_key: OpenAI API 키

    Returns:
        {ticker: {"catalyst_type": ..., ...}}
    """
    results = {}

    if not dart_adapter or not dart_adapter.is_available:
        logger.info("DART API 미설정 — 공시 분류 건너뜀")
        return results

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        logger.info("OPENAI_API_KEY 미설정 — 공시 분류 건너뜀")
        return results

    for sig in candidates:
        ticker = sig["ticker"]
        name = sig.get("name", ticker)

        # DART에서 최근 30일 공시 목록 조회
        filings = dart_adapter.fetch_recent_disclosures(ticker, days=30)
        if not filings:
            results[ticker] = DEFAULT_RESULT.copy()
            continue

        # gpt-4o-mini 분류
        result = classify_disclosures(ticker, name, filings, api_key=key)
        results[ticker] = result

        cat = result["catalyst_type"]
        conf = result["confidence"]
        reason = result.get("reason", "")[:40]
        logger.info(f"  공시분류: {name}({ticker}) → {cat} (conf={conf:.2f}) {reason}")

    return results
