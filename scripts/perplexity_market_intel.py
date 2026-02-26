"""
Perplexity 시장 인텔리전스 — 실시간 미국장/글로벌 이벤트 → 한국 섹터/종목 파급 분석

Perplexity sonar 모델의 실시간 웹검색 능력을 활용:
  1. 미국장 주요 이벤트 요약 (왜 올랐는지/빠졌는지)
  2. 글로벌 매크로 이벤트 (FOMC, 관세, 지정학)
  3. 한국 섹터별 파급효과 분석
  4. 수혜/피해 종목 매핑
  5. 긴급도 판단 (BREAKING / IMPORTANT / NORMAL)

출력: data/market_intelligence.json
연동: US Overnight Signal 보조, 내일 추천 보정, 텔레그램 알림

Usage:
    python scripts/perplexity_market_intel.py [--send]
    --send: 텔레그램으로 핵심 요약 전송
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "market_intelligence.json"

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# 한국 섹터 → 대표 종목 매핑 (파급효과 분석용)
KR_SECTOR_STOCKS = {
    "반도체": ["삼성전자", "SK하이닉스", "한미반도체", "리노공업"],
    "2차전지": ["LG에너지솔루션", "삼성SDI", "에코프로비엠", "포스코퓨처엠"],
    "바이오": ["삼성바이오로직스", "셀트리온", "알테오젠", "HLB"],
    "자동차": ["현대차", "기아", "현대모비스", "HL만도"],
    "조선": ["HD한국조선해양", "삼성중공업", "한화오션"],
    "방산": ["한화에어로스페이스", "LIG넥스원", "현대로템", "한화시스템"],
    "IT/소프트웨어": ["네이버", "카카오", "삼성SDS", "NHN"],
    "금융": ["KB금융", "신한지주", "하나금융", "삼성화재"],
    "철강": ["POSCO홀딩스", "현대제철", "고려아연"],
    "화학/에너지": ["LG화학", "롯데케미칼", "S-Oil", "SK이노베이션"],
    "전력기기": ["HD현대일렉트릭", "LS ELECTRIC", "효성중공업"],
    "로봇/AI": ["레인보우로보틱스", "두산로보틱스", "엔젤로보틱스"],
    "원전": ["두산에너빌리티", "한전기술", "비에이치아이"],
}


def query_perplexity(prompt: str, model: str = "sonar") -> dict | None:
    """Perplexity API 호출."""
    if not PERPLEXITY_API_KEY:
        logger.error("PERPLEXITY_API_KEY가 .env에 없습니다")
        return None

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 글로벌 금융시장 전문 애널리스트입니다. "
                    "미국 시장과 글로벌 이벤트가 한국 주식시장에 미치는 영향을 분석합니다. "
                    "반드시 JSON 형식으로만 응답하세요. 마크다운이나 설명 텍스트 없이 순수 JSON만 출력하세요."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    try:
        resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # JSON 파싱 (코드블록 제거)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if content.startswith("json"):
            content = content[4:].strip()

        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("JSON 파싱 실패: %s — 원본: %s", e, content[:200])
        return {"raw_response": content, "parse_error": True}
    except Exception as e:
        logger.error("Perplexity API 오류: %s", e)
        return None


def analyze_us_market() -> dict | None:
    """Q1: 미국장 주요 이벤트 + 한국 영향 분석."""
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""오늘 날짜: {today}

최근 미국 주식시장과 글로벌 금융시장의 주요 이벤트를 분석하여 다음 JSON 구조로 응답하세요:

{{
  "us_market_summary": "미국장 1~2문장 요약",
  "key_events": [
    {{
      "event": "이벤트 제목",
      "category": "실적|매크로|지정학|정책|섹터|기술",
      "impact": "positive|negative|neutral",
      "urgency": "BREAKING|IMPORTANT|NORMAL",
      "detail": "구체적 내용 1~2문장",
      "kr_sectors_affected": ["영향받는 한국 섹터"],
      "kr_impact_direction": "수혜|피해|중립",
      "kr_impact_score": -5에서 5사이 정수
    }}
  ],
  "us_market_mood": "RISK_ON|RISK_OFF|MIXED|NEUTRAL",
  "kr_open_forecast": "상승|하락|보합",
  "kr_forecast_reason": "한국장 영향 예측 이유 1문장"
}}

key_events는 중요도순 최대 7개, kr_sectors_affected는 다음 중에서 선택:
반도체, 2차전지, 바이오, 자동차, 조선, 방산, IT/소프트웨어, 금융, 철강, 화학/에너지, 전력기기, 로봇/AI, 원전"""

    return query_perplexity(prompt)


def analyze_sector_impact(events: list[dict]) -> dict | None:
    """Q2: 이벤트별 구체적 한국 종목 파급효과 분석."""
    if not events:
        return None

    # 영향력 있는 이벤트만 추출
    significant = [e for e in events if e.get("kr_impact_score", 0) != 0]
    if not significant:
        return None

    events_text = json.dumps(significant[:5], ensure_ascii=False)
    sectors_text = json.dumps(KR_SECTOR_STOCKS, ensure_ascii=False)

    prompt = f"""다음 글로벌 이벤트들의 한국 주식시장 파급효과를 종목 수준까지 분석하세요.

이벤트: {events_text}

한국 섹터별 대표종목: {sectors_text}

다음 JSON 구조로 응답:
{{
  "sector_impacts": [
    {{
      "sector": "섹터명",
      "impact": "수혜|피해|중립",
      "score": -5에서 5사이 정수,
      "reason": "영향 이유 1문장",
      "top_stocks": ["수혜/피해 대표종목 최대 3개"],
      "trade_action": "매수관심|관망|비중축소"
    }}
  ],
  "hot_themes": ["현재 시장에서 주목할 테마 최대 3개"],
  "risk_factors": ["주의할 리스크 최대 3개"]
}}"""

    return query_perplexity(prompt)


def build_intelligence_output(
    us_analysis: dict, sector_analysis: dict | None,
) -> dict:
    """최종 인텔리전스 JSON 구성."""
    now = datetime.now()

    events = us_analysis.get("key_events", [])

    # 섹터 부스트 맵 (scan_tomorrow_picks 연동용)
    sector_boost = {}
    if sector_analysis:
        for si in sector_analysis.get("sector_impacts", []):
            sector = si.get("sector", "")
            score = si.get("score", 0)
            if sector and score != 0:
                sector_boost[sector] = score

    # 수혜/피해 종목 리스트
    beneficiary_stocks = []
    risk_stocks = []
    if sector_analysis:
        for si in sector_analysis.get("sector_impacts", []):
            stocks = si.get("top_stocks", [])
            if si.get("impact") == "수혜":
                beneficiary_stocks.extend(stocks)
            elif si.get("impact") == "피해":
                risk_stocks.extend(stocks)

    # 긴급 이벤트 확인
    breaking = [e for e in events if e.get("urgency") == "BREAKING"]
    important = [e for e in events if e.get("urgency") == "IMPORTANT"]

    output = {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "us_market_summary": us_analysis.get("us_market_summary", ""),
        "us_market_mood": us_analysis.get("us_market_mood", "NEUTRAL"),
        "kr_open_forecast": us_analysis.get("kr_open_forecast", "보합"),
        "kr_forecast_reason": us_analysis.get("kr_forecast_reason", ""),
        "key_events": events,
        "event_stats": {
            "total": len(events),
            "breaking": len(breaking),
            "important": len(important),
            "positive": len([e for e in events if e.get("impact") == "positive"]),
            "negative": len([e for e in events if e.get("impact") == "negative"]),
        },
        "sector_impacts": sector_analysis.get("sector_impacts", []) if sector_analysis else [],
        "sector_boost": sector_boost,
        "hot_themes": sector_analysis.get("hot_themes", []) if sector_analysis else [],
        "risk_factors": sector_analysis.get("risk_factors", []) if sector_analysis else [],
        "beneficiary_stocks": list(set(beneficiary_stocks)),
        "risk_stocks": list(set(risk_stocks)),
    }

    return output


def build_telegram_message(intel: dict) -> str:
    """텔레그램 알림 메시지 생성."""
    mood_icons = {
        "RISK_ON": "🟢", "RISK_OFF": "🔴", "MIXED": "🟡", "NEUTRAL": "⚪",
    }
    forecast_icons = {"상승": "📈", "하락": "📉", "보합": "➡️"}

    mood = intel.get("us_market_mood", "NEUTRAL")
    forecast = intel.get("kr_open_forecast", "보합")

    lines = [
        f"🌐 시장 인텔리전스 ({intel['date']})",
        f"{'─' * 30}",
        f"{mood_icons.get(mood, '⚪')} 미국장: {intel.get('us_market_summary', '')}",
        f"{forecast_icons.get(forecast, '➡️')} 한국장 전망: {forecast} — {intel.get('kr_forecast_reason', '')}",
    ]

    # 주요 이벤트
    events = intel.get("key_events", [])
    if events:
        lines.append(f"\n📋 주요 이벤트 ({len(events)}건)")
        for e in events[:5]:
            urgency = "🚨" if e.get("urgency") == "BREAKING" else "⚡" if e.get("urgency") == "IMPORTANT" else "📌"
            impact = "🟢" if e.get("impact") == "positive" else "🔴" if e.get("impact") == "negative" else "⚪"
            sectors = ", ".join(e.get("kr_sectors_affected", [])[:3])
            lines.append(f"  {urgency}{impact} {e.get('event', '')}")
            if sectors:
                lines.append(f"     → KR: {sectors} ({e.get('kr_impact_direction', '')})")

    # 수혜/피해 종목
    bene = intel.get("beneficiary_stocks", [])
    risk = intel.get("risk_stocks", [])
    if bene:
        lines.append(f"\n✅ 수혜: {', '.join(bene[:8])}")
    if risk:
        lines.append(f"⚠️ 주의: {', '.join(risk[:5])}")

    # 핫 테마
    themes = intel.get("hot_themes", [])
    if themes:
        lines.append(f"\n🔥 핫테마: {' | '.join(themes)}")

    return "\n".join(lines)


def send_telegram(message: str):
    """텔레그램 전송."""
    try:
        from src.telegram_sender import send_message
        send_message(message)
        logger.info("텔레그램 전송 완료")
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="텔레그램 전송")
    args = parser.parse_args()

    if not PERPLEXITY_API_KEY:
        logger.error("PERPLEXITY_API_KEY 미설정. .env 확인 필요")
        return

    print("=" * 50)
    print("🌐 Perplexity 시장 인텔리전스 분석 시작")
    print("=" * 50)

    # Q1: 미국장 + 글로벌 이벤트
    print("\n[Q1] 미국장 주요 이벤트 분석 중...")
    us_analysis = analyze_us_market()
    if not us_analysis:
        logger.error("미국장 분석 실패")
        return
    if us_analysis.get("parse_error"):
        logger.warning("JSON 파싱 실패, 원본 저장")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"error": "parse_error", "raw": us_analysis.get("raw_response", "")}, f, ensure_ascii=False, indent=2)
        return

    mood = us_analysis.get("us_market_mood", "?")
    forecast = us_analysis.get("kr_open_forecast", "?")
    events = us_analysis.get("key_events", [])
    print(f"  미국장 분위기: {mood}")
    print(f"  한국장 전망: {forecast}")
    print(f"  주요 이벤트: {len(events)}건")

    for e in events[:5]:
        urgency = {"BREAKING": "🚨", "IMPORTANT": "⚡"}.get(e.get("urgency", ""), "📌")
        impact = {"positive": "+", "negative": "-"}.get(e.get("impact", ""), "=")
        print(f"    {urgency}[{impact}] {e.get('event', '')} → {','.join(e.get('kr_sectors_affected', []))}")

    # Q2: 섹터/종목 파급효과
    print("\n[Q2] 한국 섹터/종목 파급효과 분석 중...")
    sector_analysis = analyze_sector_impact(events)
    if sector_analysis and not sector_analysis.get("parse_error"):
        impacts = sector_analysis.get("sector_impacts", [])
        print(f"  섹터 영향: {len(impacts)}개")
        for si in impacts[:8]:
            icon = "🟢" if si.get("impact") == "수혜" else "🔴" if si.get("impact") == "피해" else "⚪"
            print(f"    {icon} {si.get('sector', '')}: {si.get('reason', '')} → {', '.join(si.get('top_stocks', []))}")
    else:
        sector_analysis = None

    # 최종 출력 생성
    intel = build_intelligence_output(us_analysis, sector_analysis)

    # JSON 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(intel, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUTPUT_PATH}")

    # 통계
    stats = intel["event_stats"]
    print(f"\n{'─' * 50}")
    print(f"  이벤트: {stats['total']}건 (🚨{stats['breaking']} ⚡{stats['important']})")
    print(f"  긍정: {stats['positive']} | 부정: {stats['negative']}")
    print(f"  수혜종목: {len(intel['beneficiary_stocks'])}개 | 주의종목: {len(intel['risk_stocks'])}개")
    if intel.get("hot_themes"):
        print(f"  핫테마: {' | '.join(intel['hot_themes'])}")
    print(f"{'─' * 50}")

    # 텔레그램 전송
    if args.send:
        msg = build_telegram_message(intel)
        send_telegram(msg)
    else:
        print("\n[미리보기] 텔레그램 메시지:")
        print(build_telegram_message(intel))


if __name__ == "__main__":
    main()
