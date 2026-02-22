"""글로벌 시장 뉴스 크롤링 → data/market_news.json

네이버 금융 시장뉴스 RSS에서 최근 7일 뉴스를 수집하고,
키워드 기반으로 시장 영향도(impact)를 자동 판정한다.

사용법:
  python scripts/crawl_market_news.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "market_news.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ── RSS 피드 소스 ──
RSS_FEEDS = [
    ("https://news.google.com/rss/search?q=%ED%95%9C%EA%B5%AD+%EC%A6%9D%EC%8B%9C+%EC%A3%BC%EC%8B%9D&hl=ko&gl=KR&ceid=KR:ko", "구글뉴스"),
    ("https://news.google.com/rss/search?q=%EB%82%98%EC%8A%A4%EB%8B%A5+%EB%AF%B8%EA%B5%AD+%EC%A6%9D%EC%8B%9C&hl=ko&gl=KR&ceid=KR:ko", "구글뉴스"),
    ("https://news.google.com/rss/search?q=%ED%99%98%EC%9C%A8+%EA%B8%88%EB%A6%AC+%EA%B2%BD%EC%A0%9C&hl=ko&gl=KR&ceid=KR:ko", "구글뉴스"),
]

# ── 시장 영향도 키워드 ──
HIGH_IMPACT_KEYWORDS = [
    "금리", "기준금리", "인하", "인상", "관세", "무역전쟁", "전쟁", "침공",
    "환율급등", "환율급락", "원달러", "반도체규제", "수출규제", "제재",
    "VIX", "폭락", "급락", "서킷브레이커", "블랙먼데이", "패닉",
    "디폴트", "부도", "위기", "붕괴", "트럼프", "관세폭탄",
    "FOMC", "연준", "파월", "양적긴축", "QT", "QE",
]

MEDIUM_IMPACT_KEYWORDS = [
    "실적", "어닝", "GDP", "고용", "실업", "원유", "유가",
    "인플레이션", "CPI", "PPI", "PCE", "ISM", "PMI",
    "코스피", "코스닥", "나스닥", "S&P", "다우",
    "반도체", "AI", "삼성전자", "SK하이닉스", "NVIDIA",
    "바이오", "2차전지", "방산", "조선", "건설",
    "외국인", "기관", "순매수", "순매도",
]


def classify_impact(title: str) -> str:
    """뉴스 제목에서 시장 영향도를 판정."""
    for kw in HIGH_IMPACT_KEYWORDS:
        if kw.lower() in title.lower():
            return "high"
    for kw in MEDIUM_IMPACT_KEYWORDS:
        if kw.lower() in title.lower():
            return "medium"
    return "low"


def parse_rss_date(date_str: str) -> str | None:
    """RSS 날짜를 YYYY-MM-DD로 변환."""
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def crawl_rss_feeds() -> list[dict]:
    """RSS 피드에서 뉴스를 수집."""
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    articles = []
    seen_titles = set()

    for url, source_name in RSS_FEEDS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "xml")

            items = soup.find_all("item")
            for item in items:
                title = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")

                if not title or not pub_date:
                    continue

                title_text = title.get_text(strip=True)
                # 제목에서 소스 분리 (구글뉴스 형식: "제목 - 소스")
                parts = title_text.rsplit(" - ", 1)
                clean_title = parts[0].strip()
                news_source = parts[1].strip() if len(parts) == 2 else source_name

                date_str = parse_rss_date(pub_date.get_text(strip=True))
                if not date_str or date_str < cutoff:
                    continue

                # 중복 제거
                title_key = re.sub(r'\s+', '', clean_title)[:30]
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                articles.append({
                    "date": date_str,
                    "title": clean_title,
                    "source": news_source,
                    "impact": classify_impact(clean_title),
                    "url": link.get_text(strip=True) if link else "",
                })

            logger.info(f"  {source_name}: {len(items)}건 파싱")

        except Exception as e:
            logger.warning(f"RSS 크롤 실패 ({source_name}): {e}")

    # 날짜 역순 정렬
    articles.sort(key=lambda x: x["date"], reverse=True)
    return articles


def main():
    logger.info("글로벌 시장 뉴스 크롤링 시작")

    articles = crawl_rss_feeds()

    high_count = sum(1 for a in articles if a["impact"] == "high")
    med_count = sum(1 for a in articles if a["impact"] == "medium")

    output = {
        "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "article_count": len(articles),
        "high_impact": high_count,
        "medium_impact": med_count,
        "articles": articles,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"뉴스 수집 완료: {len(articles)}건 (HIGH:{high_count} MED:{med_count})")
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
