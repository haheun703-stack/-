"""시나리오 뉴스 엔진 — 뉴스 수집 + 시나리오 평가 + 원가갭 통합 보고

매일 BAT-A(06:10) 실행:
1. Finnhub 글로벌 뉴스 수집 (최대 100건)
2. 시나리오별 키워드 매칭 + 필터링
3. Alpha Vantage 감성분석 (상위 10건, 선택)
4. 원자재 가격 → 원가 갭 로드 (commodity_prices.json)
5. 시나리오 조건 평가 → active_scenarios.json 업데이트
6. US ETF → 한국 종목 전파 체인 매핑
7. 텔레그램 모닝 다이제스트 전송
8. data/news/daily_digest.json 저장

사용:
    python -u -X utf8 scripts/news_scenario_engine.py              # 전체 실행
    python -u -X utf8 scripts/news_scenario_engine.py --no-send    # 텔레그램 제외
    python -u -X utf8 scripts/news_scenario_engine.py --news-only  # 뉴스 수집만
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NEWS_DIR = DATA_DIR / "news"
SCENARIOS_DIR = DATA_DIR / "scenarios"
COMMODITY_PATH = DATA_DIR / "commodity_prices.json"

# ────────────────────────────────────────────
# 시나리오 키워드 (영문 + 한글)
# scenario_chains.json에서 로드하되, 추가 키워드도 포함
# ────────────────────────────────────────────
SCENARIO_KEYWORDS = {
    "WAR_MIDDLE_EAST": {
        "en": ["iran", "middle east", "war", "missile", "hormuz", "israel", "airstrike",
               "oil spike", "crude surge", "geopolit", "conflict escalat"],
        "kr": ["이란", "전쟁", "중동", "미사일", "호르무즈", "이스라엘", "공습"],
        "weight": 1.5,  # 전쟁 뉴스는 가중치 높게
    },
    "OIL_SPIKE": {
        "en": ["oil price", "crude oil", "opec", "oil surge", "brent", "wti",
               "energy crisis", "oil supply", "petroleum"],
        "kr": ["유가", "원유", "OPEC", "감산", "에너지위기"],
        "weight": 1.2,
    },
    "FED_RATE_CUT": {
        "en": ["rate cut", "fed pivot", "dovish", "interest rate", "federal reserve",
               "monetary policy", "fomc", "powell"],
        "kr": ["금리인하", "피벗", "비둘기", "연준", "FOMC", "통화정책"],
        "weight": 1.0,
    },
    "SEMICONDUCTOR_CYCLE_UP": {
        "en": ["semiconductor", "hbm", "ai chip", "tsmc", "nvidia", "memory",
               "foundry", "chip", "wafer", "dram", "nand"],
        "kr": ["반도체", "HBM", "AI칩", "파운드리", "메모리"],
        "weight": 1.0,
    },
    "SPACEX_IPO": {
        "en": ["spacex", "ipo", "starlink", "rocket lab", "rklb", "space industry",
               "satellite", "launch", "starship", "falcon"],
        "kr": ["스페이스X", "우주", "위성", "발사체", "로켓랩", "스타링크"],
        "weight": 1.3,
    },
    "COMMODITY_SUPERCYCLE": {
        "en": ["copper price", "uranium", "rare earth", "critical mineral", "tio2",
               "titanium", "mining", "commodity supercycle", "supply shortage",
               "metal price", "lithium", "cobalt"],
        "kr": ["구리", "우라늄", "희토류", "원자재", "슈퍼사이클", "광산", "리튬"],
        "weight": 1.2,
    },
    "AI_POWER_DEMAND": {
        "en": ["data center", "power demand", "electricity", "nuclear energy",
               "smr", "utility", "grid", "ai energy", "power shortage"],
        "kr": ["데이터센터", "전력", "원전", "SMR", "전력망", "전력부족"],
        "weight": 1.1,
    },
    "CHINA_STIMULUS": {
        "en": ["china stimulus", "china economy", "yuan", "pboc", "infrastructure",
               "chinese demand", "china gdp"],
        "kr": ["중국", "부양", "인프라", "위안화", "양회"],
        "weight": 1.0,
    },
}

# 시나리오별 US ETF + KR 종목 매핑
SCENARIO_PROPAGATION = {
    "WAR_MIDDLE_EAST": {
        "us_etf": ["XLE", "XOP", "ERX", "GLD", "ITA"],
        "kr_stocks": ["012450:한화에어로스페이스", "079550:LIG넥스원", "064350:현대로템",
                      "010950:S-Oil"],
    },
    "OIL_SPIKE": {
        "us_etf": ["XLE", "XOP", "ERX", "USO"],
        "kr_stocks": ["010950:S-Oil", "267250:HD현대", "078930:GS"],
    },
    "FED_RATE_CUT": {
        "us_etf": ["QQQ", "TQQQ", "XLF", "FINX", "FAS"],
        "kr_stocks": ["005930:삼성전자", "000660:SK하이닉스", "105560:KB금융"],
    },
    "SEMICONDUCTOR_CYCLE_UP": {
        "us_etf": ["SOXX", "SMH", "SOXL"],
        "kr_stocks": ["005930:삼성전자", "000660:SK하이닉스", "042700:한미반도체",
                      "036930:주성엔지니어링"],
    },
    "SPACEX_IPO": {
        "us_etf": ["UFO", "ARKX", "ITA"],
        "us_stocks": ["RKLB:Rocket Lab", "ASTS:AST SpaceMobile"],
        "kr_stocks": ["012450:한화에어로스페이스", "347700:스피어", "047810:한국항공우주"],
    },
    "COMMODITY_SUPERCYCLE": {
        "us_etf": ["COPX", "COPJ", "URNM", "URA", "REMX", "SETM"],
        "us_stocks": ["TROX:Tronox", "CC:Chemours", "CCJ:Cameco", "FCX:Freeport"],
        "kr_stocks": ["006260:LS", "103140:풍산", "034020:두산에너빌리티"],
    },
    "AI_POWER_DEMAND": {
        "us_etf": ["XLU", "UTSL", "URA", "URNM"],
        "kr_stocks": ["034020:두산에너빌리티", "267260:HD현대일렉트릭"],
    },
    "CHINA_STIMULUS": {
        "us_etf": ["FXI", "KWEB"],
        "kr_stocks": ["005490:POSCO홀딩스", "090430:아모레퍼시픽"],
    },
}


# ────────────────────────────────────────────
# 1단계: 뉴스 수집 (Finnhub)
# ────────────────────────────────────────────

def collect_finnhub_news(api_key: str) -> list[dict]:
    """Finnhub에서 글로벌 뉴스 수집."""
    import requests

    articles = []

    # 일반 뉴스
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": api_key},
            timeout=15,
        )
        if r.status_code == 200:
            for item in r.json():
                articles.append({
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", "")[:200],
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "datetime": item.get("datetime", 0),
                    "category": item.get("category", ""),
                    "type": "general",
                })
    except Exception as e:
        logger.warning(f"Finnhub general news 실패: {e}")

    # 시나리오 핵심 종목 뉴스
    key_symbols = ["RKLB", "CCJ", "TROX", "CC", "FCX", "XLE", "COPX"]
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    for symbol in key_symbols:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": symbol, "from": week_ago, "to": today, "token": api_key},
                timeout=10,
            )
            if r.status_code == 200:
                for item in r.json()[:5]:  # 종목당 최대 5건
                    articles.append({
                        "headline": item.get("headline", ""),
                        "summary": item.get("summary", "")[:200],
                        "source": item.get("source", ""),
                        "url": item.get("url", ""),
                        "datetime": item.get("datetime", 0),
                        "symbol": symbol,
                        "type": "company",
                    })
            time.sleep(0.5)  # rate limit
        except Exception as e:
            logger.warning(f"Finnhub {symbol} news 실패: {e}")

    logger.info(f"Finnhub 뉴스 수집: {len(articles)}건")
    return articles


# ────────────────────────────────────────────
# 2단계: 키워드 필터 + 시나리오 매칭
# ────────────────────────────────────────────

def match_scenarios(articles: list[dict]) -> dict:
    """뉴스 헤드라인+요약에서 시나리오 키워드 매칭."""
    scenario_hits = {}  # scenario_id -> [matched articles]

    for article in articles:
        text = (article.get("headline", "") + " " + article.get("summary", "")).lower()

        for scenario_id, kw_config in SCENARIO_KEYWORDS.items():
            matched_keywords = []

            # 영문 키워드
            for kw in kw_config.get("en", []):
                if kw.lower() in text:
                    matched_keywords.append(kw)

            # 한글 키워드
            for kw in kw_config.get("kr", []):
                if kw in text:
                    matched_keywords.append(kw)

            if matched_keywords:
                if scenario_id not in scenario_hits:
                    scenario_hits[scenario_id] = []
                scenario_hits[scenario_id].append({
                    "headline": article["headline"],
                    "source": article.get("source", ""),
                    "matched_keywords": matched_keywords,
                    "weight": kw_config.get("weight", 1.0),
                    "symbol": article.get("symbol", ""),
                    "url": article.get("url", ""),
                })

    logger.info(f"시나리오 매칭: {len(scenario_hits)}개 시나리오에 뉴스 히트")
    for sid, hits in scenario_hits.items():
        logger.info(f"  {sid}: {len(hits)}건")

    return scenario_hits


# ────────────────────────────────────────────
# 3단계: Alpha Vantage 감성분석 (선택)
# ────────────────────────────────────────────

def enrich_sentiment(api_key: str, scenario_hits: dict) -> dict:
    """Alpha Vantage News Sentiment로 상위 뉴스 감성 보강."""
    import requests

    if not api_key:
        return scenario_hits

    # 시나리오별 상위 키워드로 감성 검색
    topic_map = {
        "WAR_MIDDLE_EAST": "energy_transportation",
        "OIL_SPIKE": "energy_transportation",
        "FED_RATE_CUT": "finance",
        "SEMICONDUCTOR_CYCLE_UP": "technology",
        "SPACEX_IPO": "technology",
        "COMMODITY_SUPERCYCLE": "mining",
        "AI_POWER_DEMAND": "technology",
    }

    # 활성 시나리오 중 상위 2개만 감성분석 (API 절약)
    active_scenarios = sorted(
        scenario_hits.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:2]

    for scenario_id, hits in active_scenarios:
        topic = topic_map.get(scenario_id, "")
        if not topic:
            continue

        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "topics": topic,
                    "apikey": api_key,
                },
                timeout=15,
            )
            data = r.json()
            feed = data.get("feed", [])

            # 감성 점수 높은 순 상위 5개
            sentiment_articles = []
            for a in feed[:10]:
                score = float(a.get("overall_sentiment_score", 0))
                sentiment_articles.append({
                    "title": a.get("title", "")[:80],
                    "sentiment_score": score,
                    "sentiment_label": a.get("overall_sentiment_label", ""),
                    "source": a.get("source", ""),
                })

            if sentiment_articles:
                # 시나리오 평균 감성 점수 계산
                avg_sentiment = sum(a["sentiment_score"] for a in sentiment_articles) / len(sentiment_articles)
                for hit in hits:
                    hit["avg_sentiment"] = round(avg_sentiment, 3)

            logger.info(f"  {scenario_id} 감성분석: {len(sentiment_articles)}건, 평균={round(avg_sentiment, 3) if sentiment_articles else 'N/A'}")
            time.sleep(13)  # rate limit

        except Exception as e:
            logger.warning(f"AV sentiment {scenario_id} 실패: {e}")

    return scenario_hits


# ────────────────────────────────────────────
# 4단계: 시나리오 점수 계산 + 상태 업데이트
# ────────────────────────────────────────────

def evaluate_scenarios(scenario_hits: dict, commodity_data: dict) -> dict:
    """시나리오별 점수 계산 + 활성 상태 판별."""
    today = datetime.now().strftime("%Y-%m-%d")

    # 기존 active_scenarios.json 로드
    active_path = SCENARIOS_DIR / "active_scenarios.json"
    if active_path.exists():
        with open(active_path, encoding="utf-8") as f:
            active = json.load(f)
    else:
        active = {"scenarios": {}}

    existing = active.get("scenarios", {})
    commodities = commodity_data.get("commodities", {})

    for scenario_id, kw_config in SCENARIO_KEYWORDS.items():
        hits = scenario_hits.get(scenario_id, [])
        reasons = []
        score = 0

        # 뉴스 키워드 점수
        n_hits = len(hits)
        if n_hits >= 5:
            score += 30
            reasons.append(f"키워드 {n_hits}건 (강)")
        elif n_hits >= 2:
            score += 15
            reasons.append(f"키워드 {n_hits}건 (약)")
        elif n_hits >= 1:
            score += 5
            reasons.append(f"키워드 {n_hits}건 (미약)")

        # 감성분석 점수
        if hits and hits[0].get("avg_sentiment") is not None:
            avg_s = hits[0]["avg_sentiment"]
            if avg_s > 0.3:
                score += 15
                reasons.append(f"감성 +{avg_s:.2f} (긍정)")
            elif avg_s < -0.3:
                score += 20  # 부정 뉴스도 시나리오 활성화 근거
                reasons.append(f"감성 {avg_s:.2f} (부정 = 위기)")

        # 시장 시그널 점수 (원자재 데이터 기반)
        if scenario_id == "WAR_MIDDLE_EAST":
            gold = commodities.get("gold", {})
            vix = commodities.get("vix", {})
            if gold.get("change_pct") and float(gold.get("change_pct", 0)) >= 1.5:
                score += 15
                reasons.append(f"Gold 1d +{gold['change_pct']}% (>=1.5%)")
            if vix.get("price") and float(vix["price"]) >= 25:
                score += 10
                reasons.append(f"VIX {vix['price']} >= 25")

        elif scenario_id == "OIL_SPIKE":
            wti = commodities.get("wti", {})
            if wti.get("price") and float(wti["price"]) >= 95:
                score += 20
                reasons.append(f"WTI ${wti['price']} >= $95")
            elif wti.get("price") and float(wti["price"]) >= 85:
                score += 10
                reasons.append(f"WTI ${wti['price']} >= $85")

        elif scenario_id == "COMMODITY_SUPERCYCLE":
            copper = commodities.get("copper", {})
            gap = copper.get("cost_gap", {})
            if gap.get("gap_pct") and gap["gap_pct"] >= 80:
                score += 15
                reasons.append(f"구리 갭 {gap['gap_pct']}% (과열)")
            # TiO2 매수 구간 체크
            tio2 = commodities.get("tio2", {})
            tio2_gap = tio2.get("cost_gap", {})
            if tio2_gap.get("zone") == "buy":
                score += 20
                reasons.append(f"TiO2 갭 {tio2_gap['gap_pct']}% (매수구간!)")

        elif scenario_id == "AI_POWER_DEMAND":
            uranium = commodities.get("uranium", {})
            u_gap = uranium.get("cost_gap", {})
            if u_gap.get("zone") in ("buy", "watch"):
                score += 15
                reasons.append(f"우라늄 갭 {u_gap.get('gap_pct')}% ({u_gap.get('zone')})")

        # 가중치 적용
        weight = kw_config.get("weight", 1.0)
        score = int(score * weight)
        score = min(score, 100)

        # 페이즈 결정
        prev = existing.get(scenario_id, {})
        prev_phase = prev.get("current_phase", 0)
        if score >= 60:
            phase = max(prev_phase, 3)
        elif score >= 40:
            phase = max(prev_phase, 2)
        elif score >= 20:
            phase = max(prev_phase, 1)
        else:
            phase = 0

        # 활성 시나리오만 저장 (score > 0)
        if score > 0 or scenario_id in existing:
            first_detected = prev.get("first_detected", today)
            days_active = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(first_detected, "%Y-%m-%d")).days

            existing[scenario_id] = {
                "first_detected": first_detected,
                "current_phase": phase,
                "days_active": days_active,
                "score": score,
                "reasons": reasons,
                "news_count": n_hits,
                "last_evaluated": today,
                "last_phase_change": today if phase != prev_phase else prev.get("last_phase_change", today),
            }

    # 점수 0인 오래된 시나리오 정리
    to_remove = [sid for sid, s in existing.items() if s.get("score", 0) == 0 and s.get("days_active", 0) > 14]
    for sid in to_remove:
        del existing[sid]

    active["updated"] = datetime.now().isoformat()
    active["scenarios"] = existing

    # 저장
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"시나리오 상태 업데이트: {len(existing)}개 활성")

    return active


# ────────────────────────────────────────────
# 5단계: 매수 신호 감지
# ────────────────────────────────────────────

def detect_buy_signals(active_scenarios: dict, commodity_data: dict) -> list[dict]:
    """원가 갭 매수 구간 + 활성 시나리오 → 매수 신호."""
    signals = []
    commodities = commodity_data.get("commodities", {})
    scenarios = active_scenarios.get("scenarios", {})

    for commodity_name, data in commodities.items():
        gap = data.get("cost_gap", {})
        if not gap or gap.get("zone") not in ("buy", "watch"):
            continue

        # 이 원자재와 관련된 활성 시나리오 찾기
        related_scenarios = []
        for sid, sdata in scenarios.items():
            if sdata.get("score", 0) < 20:
                continue
            prop = SCENARIO_PROPAGATION.get(sid, {})
            # 시나리오의 관련 종목에 이 원자재가 연결되는지 체크
            all_related = str(prop)
            if commodity_name in all_related.lower() or any(
                kw in data.get("name", "").lower()
                for kw in ["구리", "copper", "우라늄", "uranium", "tio2", "티타늄", "oil", "원유"]
                if kw in all_related.lower() or kw in str(SCENARIO_KEYWORDS.get(sid, {})).lower()
            ):
                related_scenarios.append(sid)

        if gap["zone"] == "buy":
            signal_strength = "strong"
        elif related_scenarios:
            signal_strength = "moderate"
        else:
            continue

        signals.append({
            "commodity": commodity_name,
            "name": data.get("name", commodity_name),
            "price": data.get("price"),
            "gap_pct": gap["gap_pct"],
            "zone": gap["zone"],
            "signal_strength": signal_strength,
            "related_scenarios": related_scenarios,
            "related_etf": [],
            "related_kr": [],
        })

        # 전파 체인 추가
        for sid in related_scenarios:
            prop = SCENARIO_PROPAGATION.get(sid, {})
            signals[-1]["related_etf"].extend(prop.get("us_etf", []))
            signals[-1]["related_kr"].extend(prop.get("kr_stocks", []))

        # 중복 제거
        signals[-1]["related_etf"] = list(set(signals[-1]["related_etf"]))
        signals[-1]["related_kr"] = list(set(signals[-1]["related_kr"]))

    logger.info(f"매수 신호 감지: {len(signals)}건")
    return signals


# ────────────────────────────────────────────
# 6단계: 텔레그램 보고
# ────────────────────────────────────────────

def build_telegram_report(
    scenario_hits: dict,
    active_scenarios: dict,
    commodity_data: dict,
    buy_signals: list[dict],
) -> str:
    """텔레그램 모닝 다이제스트 생성."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"\U0001f4f0 [\uc2dc\ub098\ub9ac\uc624 \uc5d4\uc9c4] {now}")
    lines.append("")

    # ── 원가 갭 요약 ──
    commodities = commodity_data.get("commodities", {})
    gap_items = []
    for name, data in commodities.items():
        gap = data.get("cost_gap", {})
        if gap.get("gap_pct") is not None:
            gap_items.append((name, data, gap))
    gap_items.sort(key=lambda x: x[2]["gap_pct"])

    if gap_items:
        lines.append("\u2501\u2501 \uc6d0\uac00 \uac2d \ub300\uc2dc\ubcf4\ub4dc \u2501\u2501")
        for name, data, gap in gap_items:
            emoji = gap.get("emoji", "")
            label = data.get("name", name)[:8]
            price = data.get("price", 0)
            cost = gap.get("production_cost", 0)
            gap_pct = gap["gap_pct"]
            zone_kr = {"buy": "\ub9e4\uc218\uad6c\uac04", "watch": "\uad00\ucc30", "hold": "\ubcf4\ub958", "overheated": "\uacfc\uc5f4"}.get(gap.get("zone", ""), "")
            lines.append(f"{emoji} {label} ${price} (\uc6d0\uac00${cost}) \uac2d{gap_pct}% {zone_kr}")
        lines.append("")

    # ── 활성 시나리오 ──
    scenarios = active_scenarios.get("scenarios", {})
    active_list = sorted(scenarios.items(), key=lambda x: x[1].get("score", 0), reverse=True)

    if active_list:
        active_count = sum(1 for _, s in active_list if s.get("score", 0) >= 20)
        lines.append(f"\u2501\u2501 \ud65c\uc131 \uc2dc\ub098\ub9ac\uc624 ({active_count}/{len(SCENARIO_KEYWORDS)}) \u2501\u2501")
        for sid, sdata in active_list:
            score = sdata.get("score", 0)
            if score < 10:
                continue
            phase = sdata.get("current_phase", 0)
            days = sdata.get("days_active", 0)
            news_count = sdata.get("news_count", 0)

            if score >= 60:
                s_emoji = "\U0001f534"  # 🔴
            elif score >= 40:
                s_emoji = "\U0001f7e1"  # 🟡
            else:
                s_emoji = "\u26aa"  # ⚪

            name_map = {
                "WAR_MIDDLE_EAST": "\uc911\ub3d9\uc804\uc7c1",
                "OIL_SPIKE": "\uc720\uac00\uae09\ub4f1",
                "FED_RATE_CUT": "Fed\uae08\ub9ac\uc778\ud558",
                "SEMICONDUCTOR_CYCLE_UP": "\ubc18\ub3c4\uccb4\uc0ac\uc774\ud074",
                "SPACEX_IPO": "SpaceX-IPO",
                "COMMODITY_SUPERCYCLE": "\uc6d0\uc790\uc7ac\uc288\ud37c\uc0ac\uc774\ud074",
                "AI_POWER_DEMAND": "AI\uc804\ub825\uc218\uc694",
                "CHINA_STIMULUS": "\uc911\uad6d\ubd80\uc591",
            }
            name = name_map.get(sid, sid)
            lines.append(f"{s_emoji} {name} [{score}\uc810] P{phase} D+{days}")

            # 핵심 이유 1개
            reasons = sdata.get("reasons", [])
            if reasons:
                lines.append(f"   {reasons[0]}")

            # 전파 체인
            prop = SCENARIO_PROPAGATION.get(sid, {})
            etfs = prop.get("us_etf", [])[:3]
            kr = prop.get("kr_stocks", [])[:2]
            if etfs:
                lines.append(f"   \u2192 US: {', '.join(etfs)}")
            if kr:
                kr_names = [s.split(":")[1] if ":" in s else s for s in kr]
                lines.append(f"   \u2192 KR: {', '.join(kr_names)}")

        lines.append("")

    # ── 주목 뉴스 TOP 5 ──
    all_hits = []
    for sid, hits in scenario_hits.items():
        for h in hits:
            h["scenario"] = sid
            all_hits.append(h)

    # 가중치 기반 정렬
    all_hits.sort(key=lambda x: x.get("weight", 1.0) * len(x.get("matched_keywords", [])), reverse=True)

    if all_hits:
        lines.append(f"\u2501\u2501 \uc8fc\ubaa9 \ub274\uc2a4 ({len(all_hits)}\uac74 \uc911 TOP 5) \u2501\u2501")
        seen_headlines = set()
        count = 0
        for hit in all_hits:
            headline = hit["headline"][:60]
            if headline in seen_headlines:
                continue
            seen_headlines.add(headline)
            count += 1
            if count > 5:
                break
            sid = hit["scenario"]
            kws = ", ".join(hit["matched_keywords"][:2])
            lines.append(f"{count}. {headline}")
            lines.append(f"   #{sid} | \ud0a4\uc6cc\ub4dc: {kws}")
        lines.append("")

    # ── 매수 신호 ──
    if buy_signals:
        lines.append("\u2501\u2501 \ub9e4\uc218 \uc2e0\ud638 \u2501\u2501")
        for sig in buy_signals:
            strength = "\U0001f7e2 \uac15\ub825" if sig["signal_strength"] == "strong" else "\U0001f7e1 \ubcf4\ud1b5"
            lines.append(f"{strength} {sig['name']} \uac2d{sig['gap_pct']}%")
            if sig["related_etf"]:
                lines.append(f"  US ETF: {', '.join(sig['related_etf'][:4])}")
            if sig["related_kr"]:
                kr_names = [s.split(":")[1] if ":" in s else s for s in sig["related_kr"][:3]]
                lines.append(f"  KR: {', '.join(kr_names)}")
    else:
        lines.append("\u2501\u2501 \ub9e4\uc218 \uc2e0\ud638: \uc5c6\uc74c \u2501\u2501")

    return "\n".join(lines)


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="\uc2dc\ub098\ub9ac\uc624 \ub274\uc2a4 \uc5d4\uc9c4")
    parser.add_argument("--no-send", action="store_true", help="\ud154\ub808\uadf8\ub7a8 \uc804\uc1a1 \uc81c\uc678")
    parser.add_argument("--news-only", action="store_true", help="\ub274\uc2a4 \uc218\uc9d1\ub9cc")
    parser.add_argument("--no-sentiment", action="store_true", help="AV \uac10\uc131\ubd84\uc11d \uc81c\uc678 (API \uc808\uc57d)")
    args = parser.parse_args()

    fh_key = os.environ.get("FINNHUB_API_KEY", "")
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip().rstrip(".")

    if not fh_key:
        logger.error("FINNHUB_API_KEY \uc5c6\uc74c")
        return

    logger.info("=== \uc2dc\ub098\ub9ac\uc624 \ub274\uc2a4 \uc5d4\uc9c4 \uc2dc\uc791 ===")

    # 1. 뉴스 수집
    articles = collect_finnhub_news(fh_key)

    # 2. 시나리오 매칭
    scenario_hits = match_scenarios(articles)

    if args.news_only:
        # 뉴스만 저장하고 종료
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        digest_path = NEWS_DIR / f"news_{datetime.now().strftime('%Y%m%d')}.json"
        digest_path.write_text(json.dumps({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_articles": len(articles),
            "scenario_hits": {k: len(v) for k, v in scenario_hits.items()},
            "top_headlines": [a["headline"] for a in articles[:20]],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"\ub274\uc2a4\ub9cc \uc800\uc7a5: {digest_path}")
        return

    # 3. 감성분석 (선택)
    if not args.no_sentiment and av_key:
        scenario_hits = enrich_sentiment(av_key, scenario_hits)

    # 4. 원자재 데이터 로드
    commodity_data = {}
    if COMMODITY_PATH.exists():
        with open(COMMODITY_PATH, encoding="utf-8") as f:
            commodity_data = json.load(f)
        logger.info(f"\uc6d0\uc790\uc7ac \ub370\uc774\ud130 \ub85c\ub4dc: {len(commodity_data.get('commodities', {}))}종")
    else:
        logger.warning("commodity_prices.json \uc5c6\uc74c \u2014 \uc6d0\uac00 \uac2d \ubd84\uc11d \ubd88\uac00")

    # 5. 시나리오 평가 + 상태 업데이트
    active_scenarios = evaluate_scenarios(scenario_hits, commodity_data)

    # 6. 매수 신호 감지
    buy_signals = detect_buy_signals(active_scenarios, commodity_data)

    # 7. 다이제스트 저장
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    digest = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "total_articles": len(articles),
        "scenario_hits": {k: len(v) for k, v in scenario_hits.items()},
        "active_scenarios": active_scenarios.get("scenarios", {}),
        "buy_signals": buy_signals,
        "top_news": [],
    }

    # 상위 뉴스 20건 저장
    all_hits_flat = []
    for sid, hits in scenario_hits.items():
        for h in hits:
            h["scenario"] = sid
            all_hits_flat.append(h)
    all_hits_flat.sort(key=lambda x: x.get("weight", 1) * len(x.get("matched_keywords", [])), reverse=True)
    digest["top_news"] = all_hits_flat[:20]

    digest_path = NEWS_DIR / "daily_digest.json"
    digest_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"\ub2e4\uc774\uc81c\uc2a4\ud2b8 \uc800\uc7a5: {digest_path}")

    # 날짜별 히스토리도 저장
    history_path = NEWS_DIR / f"digest_{datetime.now().strftime('%Y%m%d')}.json"
    history_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. 텔레그램 보고
    report = build_telegram_report(scenario_hits, active_scenarios, commodity_data, buy_signals)
    print("\n" + report)

    if not args.no_send:
        try:
            from src.telegram_sender import send_message
            send_message(report)
            logger.info("\ud154\ub808\uadf8\ub7a8 \uc804\uc1a1 \uc644\ub8cc")
        except Exception as e:
            logger.error(f"\ud154\ub808\uadf8\ub7a8 \uc804\uc1a1 \uc2e4\ud328: {e}")

    logger.info("=== \uc2dc\ub098\ub9ac\uc624 \ub274\uc2a4 \uc5d4\uc9c4 \uc644\ub8cc ===")
    return digest


if __name__ == "__main__":
    main()
