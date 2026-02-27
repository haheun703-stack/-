"""장전 증권사 리포트 스캔 — 네이버증권 리서치 + Perplexity 테마 예측

07:00 BAT-B에서 호출. 당일 발행 리포트에서 목표가 상향/투자의견 변경 감지,
Perplexity에 장전 급등 테마 질문, 후보풀 종목 뉴스 모니터링.

출력: data/morning_reports.json
연동: scan_tomorrow_picks.py에서 report_bonus (전략 G) 반영

Usage:
    python scripts/crawl_morning_reports.py              # 기본 (미리보기)
    python scripts/crawl_morning_reports.py --send        # 텔레그램 발송
    python scripts/crawl_morning_reports.py --no-perplexity  # Perplexity 스킵
    python scripts/crawl_morning_reports.py --no-news     # 뉴스 스캔 스킵
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "morning_reports.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}

NAVER_RESEARCH_URL = "https://finance.naver.com/research/company_list.naver"

# ── 리포트 투자의견 분류 ──
POSITIVE_OPINIONS = {"매수", "BUY", "Strong Buy", "Trading BUY", "Outperform", "비중확대"}
NEGATIVE_OPINIONS = {"매도", "SELL", "Underperform", "비중축소", "중립"}


# ═══════════════════════════════════════════════════════════
#  1. 증권사 리포트 크롤링
# ═══════════════════════════════════════════════════════════

def _fetch_detail_opinion(detail_url: str) -> tuple[str, int]:
    """상세 페이지에서 목표가 + 투자의견 추출.

    Returns:
        (opinion, target_price) — 예: ("Buy", 90000)
    """
    try:
        resp = requests.get(detail_url, headers=HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 상세 페이지의 type_1 테이블에서 "목표가...|투자의견..." 형태 파싱
        table = soup.select_one("table.type_1")
        if not table:
            return "", 0

        text = table.get_text(" ", strip=True)
        # 목표가 추출: "목표가90,000" or "목표가 90,000"
        tp_match = re.search(r"목표가\s*([0-9,]+)", text)
        target_price = int(tp_match.group(1).replace(",", "")) if tp_match else 0
        # 투자의견 추출: "투자의견Buy" or "투자의견 매수"
        op_match = re.search(
            r"투자의견\s*(\w+)",
            text,
        )
        opinion = op_match.group(1) if op_match else ""
        return opinion, target_price

    except Exception:
        return "", 0


def fetch_naver_research(max_pages: int = 3) -> list[dict]:
    """네이버증권 리서치에서 당일+전일 증권사 리포트 수집.

    실제 HTML 구조 (table.type_1):
      컬럼: 종목명 | 제목 | 증권사 | 첨부 | 작성일 | 조회수
      인덱스: td[0]  td[1]  td[2]   td[3]  td[4]   td[5]

    목표가/투자의견은 목록에 없음 → 상세 페이지(company_read)에서 추출.

    Returns:
        [{"stock_name", "stock_code", "title", "broker",
          "target_price", "opinion", "date", "url"}, ...]
    """
    today_str = datetime.now().strftime("%y.%m.%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%y.%m.%d")
    valid_dates = {today_str, yesterday_str}

    reports: list[dict] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                NAVER_RESEARCH_URL,
                params={"page": page},
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            resp.encoding = "euc-kr"
            soup = BeautifulSoup(resp.text, "html.parser")

            table = soup.select_one("table.type_1")
            if not table:
                logger.warning("테이블 미발견 (page %d)", page)
                break

            rows = table.select("tr")
            found_old = False
            for row in rows:
                tds = row.select("td")
                if len(tds) < 6:
                    continue

                # 작성일은 td[4] (인덱스 4)
                date_text = tds[4].get_text(strip=True)
                if date_text not in valid_dates:
                    if date_text and date_text < yesterday_str:
                        found_old = True
                    continue

                # td[0]: 종목명 + 코드
                stock_link = tds[0].select_one("a")
                if not stock_link:
                    continue
                stock_name = stock_link.get_text(strip=True)
                href = stock_link.get("href", "")
                code_match = re.search(r"code=(\d{6})", href)
                stock_code = code_match.group(1) if code_match else ""

                # td[1]: 리포트 제목 + 상세 URL (nid)
                title_link = tds[1].select_one("a")
                title = title_link.get_text(strip=True) if title_link else ""
                detail_url = ""
                if title_link:
                    href2 = title_link.get("href", "")
                    if href2 and not href2.startswith("http"):
                        detail_url = "https://finance.naver.com/research/" + href2
                    else:
                        detail_url = href2

                # td[2]: 증권사
                broker = tds[2].get_text(strip=True)

                # 중복 제거
                key = f"{stock_code}_{broker}_{date_text}"
                if key in seen:
                    continue
                seen.add(key)

                reports.append({
                    "stock_name": stock_name,
                    "stock_code": stock_code,
                    "title": title,
                    "broker": broker,
                    "target_price": 0,
                    "opinion": "",
                    "date": date_text,
                    "url": detail_url,
                })

            if found_old:
                break

            time.sleep(1.0)

        except Exception as e:
            logger.warning("리서치 크롤링 실패 (page %d): %s", page, e)
            break

    # 2단계: 각 리포트 상세 페이지에서 목표가/투자의견 추출
    if reports:
        logger.info("상세 페이지 %d건 조회 중...", len(reports))
        for i, r in enumerate(reports):
            if r["url"]:
                opinion, target = _fetch_detail_opinion(r["url"])
                r["opinion"] = opinion
                r["target_price"] = target
                if (i + 1) % 5 == 0:
                    logger.info("  %d/%d 완료", i + 1, len(reports))
                time.sleep(0.5)

    logger.info("증권사 리포트 %d건 수집", len(reports))
    return reports


def classify_reports(reports: list[dict]) -> list[dict]:
    """리포트를 영향도 기준으로 분류하고 boost 점수 산정."""
    for r in reports:
        opinion = r.get("opinion", "")
        title = r.get("title", "")

        # 제목 키워드 감지
        is_upgrade = bool(re.search(
            r"상향|업그레이드|긍정|서프라이즈|목표가.*인상|실적.*호전|수주.*확대|성장.*가속",
            title,
        ))
        is_downgrade = bool(re.search(
            r"하향|다운그레이드|부정|실적.*부진|둔화|리스크.*확대|목표가.*인하",
            title,
        ))

        is_buy = any(op in opinion for op in POSITIVE_OPINIONS)
        is_sell = any(op in opinion for op in NEGATIVE_OPINIONS)

        if is_buy and is_upgrade:
            boost, tag = 8, "목표가상향+매수"
        elif is_buy:
            boost, tag = 5, "매수"
        elif is_sell or is_downgrade:
            boost, tag = -5, "비중축소"
        else:
            boost, tag = 0, "중립"

        r["boost"] = boost
        r["tag"] = tag

    return reports


# ═══════════════════════════════════════════════════════════
#  2. Perplexity 장전 테마 예측
# ═══════════════════════════════════════════════════════════

def query_perplexity_morning_theme() -> dict | None:
    """Perplexity에 '오늘 한국장 급등 예상 테마/종목' 질문 (1회 호출)."""
    try:
        from scripts.perplexity_market_intel import query_perplexity
    except ImportError:
        logger.warning("perplexity_market_intel import 실패")
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""오늘 날짜: {today}

오늘 한국 주식시장에서 주목해야 할 급등 가능 테마와 종목을 분석하세요.
다음 JSON 구조로만 응답하세요:

{{
  "hot_themes": [
    {{
      "theme": "테마명",
      "catalyst": "촉매 이유 1문장",
      "urgency": "HIGH|MEDIUM|LOW",
      "stocks": ["종목명1", "종목명2", "종목명3"]
    }}
  ],
  "breaking_catalysts": [
    {{
      "stock_name": "종목명",
      "catalyst": "촉매 내용 (수주/FDA/M&A/실적/리포트 등)",
      "impact": "positive|negative",
      "source": "뉴스 출처"
    }}
  ],
  "sector_outlook": {{
    "bullish": ["상승 예상 섹터 최대 3개"],
    "bearish": ["하락 주의 섹터 최대 2개"]
  }}
}}

hot_themes는 중요도순 최대 5개.
breaking_catalysts는 개별 종목 촉매 뉴스 최대 10개.
한국 장전 시점의 최신 정보를 기반으로 분석하세요.
증권사 리포트 발행, 수주/계약 공시, FDA 승인, 정책 수혜 등 구체적 촉매만 포함하세요."""

    result = query_perplexity(prompt)
    if result and result.get("parse_error"):
        logger.warning("Perplexity JSON 파싱 실패")
        return None
    return result


# ═══════════════════════════════════════════════════════════
#  3. 후보풀 종목 뉴스 체크
# ═══════════════════════════════════════════════════════════

def _build_name_map() -> dict[str, str]:
    """ticker → 종목명 매핑 (tomorrow_picks.json에서)."""
    try:
        with open(DATA_DIR / "tomorrow_picks.json", "r", encoding="utf-8") as f:
            picks = json.load(f)
        return {p["ticker"]: p["name"] for p in picks.get("picks", []) if p.get("name")}
    except Exception:
        return {}


def scan_candidate_news(tickers: list[str], name_map: dict[str, str]) -> list[dict]:
    """후보풀 종목 중 24h 내 뉴스 터진 종목 감지.

    최대 30종목만 검색 (rate limit).
    """
    try:
        from scripts.crawl_market_news import crawl_stock_news
    except ImportError:
        logger.warning("crawl_market_news import 실패")
        return []

    stock_names = [name_map[t] for t in tickers[:30] if t in name_map]
    if not stock_names:
        return []

    news = crawl_stock_news(stock_names, days=1)
    # high/medium만 필터
    return [n for n in news if n.get("impact") in ("high", "medium")]


# ═══════════════════════════════════════════════════════════
#  4. 최종 출력 + 텔레그램
# ═══════════════════════════════════════════════════════════

def build_morning_output(
    reports: list[dict],
    perplexity_result: dict | None,
    candidate_news: list[dict],
) -> dict:
    """morning_reports.json 최종 구성."""
    now = datetime.now()

    # report_boost_map: ticker → {boost, tag, broker, target_price, title}
    report_boost_map: dict[str, dict] = {}
    for r in reports:
        code = r.get("stock_code", "")
        if code and r.get("boost", 0) != 0:
            existing = report_boost_map.get(code, {})
            if abs(r["boost"]) > abs(existing.get("boost", 0)):
                report_boost_map[code] = {
                    "boost": r["boost"],
                    "tag": r["tag"],
                    "broker": r["broker"],
                    "target_price": r["target_price"],
                    "title": r["title"][:50],
                }

    # news_boost_map: 종목명 → {boost, reason}
    news_boost_map: dict[str, dict] = {}
    for n in candidate_news:
        sname = n.get("stock_name", "")
        if not sname:
            continue
        if n.get("impact") == "high":
            news_boost_map[sname] = {"boost": 5, "reason": n.get("title", "")[:40]}
        elif sname not in news_boost_map:
            news_boost_map[sname] = {"boost": 3, "reason": n.get("title", "")[:40]}

    return {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "reports": reports,
        "report_boost_map": report_boost_map,
        "perplexity_themes": perplexity_result or {},
        "candidate_news": candidate_news,
        "news_boost_map": news_boost_map,
        "stats": {
            "total_reports": len(reports),
            "positive_reports": len([r for r in reports if r.get("boost", 0) > 0]),
            "negative_reports": len([r for r in reports if r.get("boost", 0) < 0]),
            "news_alerts": len(candidate_news),
            "themes_count": len((perplexity_result or {}).get("hot_themes", [])),
        },
    }


def build_telegram_alert(output: dict) -> str:
    """장전 리포트 스캔 텔레그램 메시지."""
    date = output["date"]
    stats = output["stats"]
    reports = output.get("reports", [])

    lines = [
        f"📋 장전 리포트 스캔 ({date})",
        "━" * 28,
    ]

    # 매수 리포트
    positive = [r for r in reports if r.get("boost", 0) > 0]
    if positive:
        lines.append(f"\n✅ 매수 리포트 ({len(positive)}건)")
        for r in positive[:8]:
            tp = f" 목표:{r['target_price']:,}" if r.get("target_price") else ""
            lines.append(f"  📌 {r['stock_name']}({r['stock_code']}) — {r['broker']}")
            lines.append(f"     {r['tag']}{tp}")

    # 주의 리포트
    negative = [r for r in reports if r.get("boost", 0) < 0]
    if negative:
        lines.append(f"\n⚠️ 주의 리포트 ({len(negative)}건)")
        for r in negative[:3]:
            lines.append(f"  🔻 {r['stock_name']}({r['stock_code']}) — {r['broker']} [{r['tag']}]")

    # Perplexity 테마
    pplx = output.get("perplexity_themes", {})
    themes = pplx.get("hot_themes", [])
    if themes:
        lines.append(f"\n🔥 오늘 주목 테마")
        for t in themes[:4]:
            icon = {"HIGH": "🚨", "MEDIUM": "⚡", "LOW": "📌"}.get(t.get("urgency", ""), "📌")
            stocks_str = ", ".join(t.get("stocks", [])[:3])
            lines.append(f"  {icon} {t.get('theme', '')}")
            if t.get("catalyst"):
                lines.append(f"     {t['catalyst']}")
            if stocks_str:
                lines.append(f"     → {stocks_str}")

    # 개별 촉매
    catalysts = pplx.get("breaking_catalysts", [])
    if catalysts:
        lines.append(f"\n💥 개별 촉매 ({len(catalysts)}건)")
        for c in catalysts[:5]:
            icon = "🟢" if c.get("impact") == "positive" else "🔴"
            lines.append(f"  {icon} {c.get('stock_name', '')} — {c.get('catalyst', '')[:40]}")

    # 후보풀 뉴스
    news = output.get("candidate_news", [])
    if news:
        lines.append(f"\n📰 후보풀 뉴스 ({len(news)}건)")
        for n in news[:5]:
            lines.append(f"  • [{n.get('stock_name', '')}] {n.get('title', '')[:35]}")

    lines.append(
        f"\n📊 리포트 {stats['total_reports']}건 | "
        f"매수 {stats['positive_reports']} | "
        f"주의 {stats['negative_reports']} | "
        f"테마 {stats['themes_count']}"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="장전 증권사 리포트 스캔")
    parser.add_argument("--send", action="store_true", help="텔레그램 전송")
    parser.add_argument("--no-perplexity", action="store_true", help="Perplexity 스킵")
    parser.add_argument("--no-news", action="store_true", help="뉴스 스캔 스킵")
    args = parser.parse_args()

    print("=" * 55)
    print("📋 장전 증권사 리포트 스캔 시작")
    print("=" * 55)

    # 1. 증권사 리포트
    print("\n[1/3] 네이버 리서치 크롤링...")
    reports = fetch_naver_research(max_pages=3)
    reports = classify_reports(reports)
    pos = len([r for r in reports if r.get("boost", 0) > 0])
    neg = len([r for r in reports if r.get("boost", 0) < 0])
    print(f"  → {len(reports)}건 (매수 {pos} | 주의 {neg})")

    # 2. Perplexity 장전 테마
    pplx_result = None
    if not args.no_perplexity:
        print("\n[2/3] Perplexity 장전 테마 예측...")
        pplx_result = query_perplexity_morning_theme()
        if pplx_result:
            themes = pplx_result.get("hot_themes", [])
            cats = pplx_result.get("breaking_catalysts", [])
            print(f"  → 테마 {len(themes)}개 | 촉매 {len(cats)}건")
        else:
            print("  → Perplexity 실패 — 스킵")
    else:
        print("\n[2/3] Perplexity 스킵 (--no-perplexity)")

    # 3. 후보풀 뉴스
    candidate_news: list[dict] = []
    if not args.no_news:
        print("\n[3/3] 후보풀 뉴스 스캔...")
        name_map = _build_name_map()
        # 관찰 등급 이상 종목만
        try:
            with open(DATA_DIR / "tomorrow_picks.json", "r", encoding="utf-8") as f:
                picks = json.load(f)
            watchable = [
                p["ticker"]
                for p in picks.get("picks", [])
                if p.get("grade") in ("적극매수", "매수", "관심매수", "관찰")
            ]
        except Exception:
            watchable = []
        if watchable:
            candidate_news = scan_candidate_news(watchable, name_map)
            print(f"  → 뉴스 {len(candidate_news)}건 감지")
        else:
            print("  → 후보풀 없음 — 스킵")
    else:
        print("\n[3/3] 뉴스 스킵 (--no-news)")

    # 최종 출력
    output = build_morning_output(reports, pplx_result, candidate_news)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUTPUT_PATH}")

    # 텔레그램
    msg = build_telegram_alert(output)
    if args.send:
        try:
            from src.telegram_sender import send_message
            ok = send_message(msg)
            print(f"[텔레그램] 전송 {'성공' if ok else '실패'}")
        except Exception as e:
            print(f"[텔레그램] 전송 실패: {e}")
    else:
        print("\n[미리보기]")
        print(msg)

    print(f"\n📊 최종: 리포트 {output['stats']['total_reports']}건 | "
          f"매수 {output['stats']['positive_reports']} | "
          f"테마 {output['stats']['themes_count']}")


if __name__ == "__main__":
    main()
