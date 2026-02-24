"""글로벌 시장 뉴스 + 섹터/종목 뉴스 크롤링 → data/market_news.json

Google News RSS에서 시장/섹터/종목 뉴스를 수집하고,
키워드 기반으로 시장 영향도(impact)를 자동 판정한다.

기능:
  1. 매크로 뉴스: 한국증시, 미국증시, 환율/금리
  2. 섹터 뉴스: 반도체, 바이오, 2차전지, 방산, 조선, AI 등
  3. 종목 뉴스 검색: 급등주 등 특정 종목명으로 동적 검색

사용법:
  python scripts/crawl_market_news.py              # 전체 크롤링
  python scripts/crawl_market_news.py --stocks "현대바이오,온코닉"  # 종목 검색
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

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
sys.path.insert(0, str(PROJECT_ROOT))
OUT_PATH = PROJECT_ROOT / "data" / "market_news.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _google_rss(query: str) -> str:
    """Google News RSS URL 생성."""
    return f"https://news.google.com/rss/search?q={quote(query)}&hl=ko&gl=KR&ceid=KR:ko"


# ── RSS 피드 소스: 매크로 + 섹터 ──
RSS_FEEDS = [
    # --- 매크로 (기존) ---
    (_google_rss("한국 증시 주식"), "매크로"),
    (_google_rss("나스닥 미국 증시"), "매크로"),
    (_google_rss("환율 금리 경제"), "매크로"),
    # --- 섹터별 뉴스 (신규) ---
    (_google_rss("반도체 HBM 파운드리 주식"), "반도체"),
    (_google_rss("바이오 신약 임상 FDA 승인"), "바이오"),
    (_google_rss("2차전지 배터리 양극재 음극재"), "2차전지"),
    (_google_rss("방산 방위산업 K방산 수출"), "방산"),
    (_google_rss("조선 수주 LNG선 친환경"), "조선"),
    (_google_rss("AI 인공지능 데이터센터 GPU"), "AI"),
    (_google_rss("로봇 휴머노이드 자동화"), "로봇"),
    (_google_rss("원전 SMR 원자력 에너지"), "원전"),
    (_google_rss("자율주행 전기차 모빌리티"), "자율주행"),
    (_google_rss("항암제 면역항암 CAR-T 신약승인"), "항암"),
    (_google_rss("전력기기 변압기 초고압 전력망"), "전력기기"),
    (_google_rss("양자컴퓨터 양자암호 양자기술"), "양자"),
    (_google_rss("우주항공 위성 발사체 누리호"), "우주항공"),
    (_google_rss("수소경제 수소차 연료전지"), "수소"),
    (_google_rss("페로브스카이트 태양전지"), "페로브스카이트"),
    (_google_rss("공매도 재개 금지 대차"), "공매도"),
]

# ── 시장 영향도 키워드 ──
HIGH_IMPACT_KEYWORDS = [
    "금리", "기준금리", "인하", "인상", "관세", "무역전쟁", "전쟁", "침공",
    "환율급등", "환율급락", "원달러", "반도체규제", "수출규제", "제재",
    "VIX", "폭락", "급락", "서킷브레이커", "블랙먼데이", "패닉",
    "디폴트", "부도", "위기", "붕괴", "트럼프", "관세폭탄",
    "FOMC", "연준", "파월", "양적긴축", "QT", "QE",
    "FDA승인", "임상3상", "블록딜", "대규모수주", "인수합병",
    "상한가", "하한가", "거래정지", "상장폐지",
]

MEDIUM_IMPACT_KEYWORDS = [
    "실적", "어닝", "GDP", "고용", "실업", "원유", "유가",
    "인플레이션", "CPI", "PPI", "PCE", "ISM", "PMI",
    "코스피", "코스닥", "나스닥", "S&P", "다우",
    "반도체", "AI", "삼성전자", "SK하이닉스", "NVIDIA",
    "바이오", "2차전지", "방산", "조선", "건설",
    "외국인", "기관", "순매수", "순매도",
    "HBM", "파운드리", "신약", "임상", "배터리",
    "LNG", "수주", "로봇", "휴머노이드", "원전", "SMR",
    "수소", "전기차", "자율주행", "양자", "위성",
    "페로브스카이트", "항암", "변압기", "전력",
    "공매도", "밸류업", "자사주", "배당",
    "테마주", "급등", "신고가", "돌파",
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


def crawl_stock_news(stock_names: list[str], days: int = 2) -> list[dict]:
    """특정 종목명으로 뉴스 검색 (급등주 자동 뉴스용).

    scan_force_hybrid.py 등에서 import하여 사용.
    종목명 리스트를 받아 Google News RSS로 검색한다.

    Args:
        stock_names: 검색할 종목명 리스트 (예: ["현대바이오", "온코닉테라퓨틱스"])
        days: 최근 N일 뉴스만 수집 (기본 2일)

    Returns:
        뉴스 기사 리스트 [{stock_name, title, source, date, impact, url}, ...]
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    articles = []
    seen_titles = set()

    for name in stock_names:
        query = f"{name} 주식"
        url = _google_rss(query)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "xml")

            items = soup.find_all("item")
            found = 0
            for item in items:
                title_tag = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")
                if not title_tag or not pub_date:
                    continue

                title_text = title_tag.get_text(strip=True)
                parts = title_text.rsplit(" - ", 1)
                clean_title = parts[0].strip()
                news_source = parts[1].strip() if len(parts) == 2 else "구글뉴스"

                date_str = parse_rss_date(pub_date.get_text(strip=True))
                if not date_str or date_str < cutoff:
                    continue

                title_key = re.sub(r'\s+', '', clean_title)[:30]
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                articles.append({
                    "stock_name": name,
                    "date": date_str,
                    "title": clean_title,
                    "source": news_source,
                    "impact": classify_impact(clean_title),
                    "url": link.get_text(strip=True) if link else "",
                })
                found += 1

            logger.info(f"  종목뉴스 [{name}]: {found}건 수집")

        except Exception as e:
            logger.warning(f"종목뉴스 크롤 실패 ({name}): {e}")

    articles.sort(key=lambda x: x["date"], reverse=True)
    return articles


def main():
    import argparse
    parser = argparse.ArgumentParser(description="시장+섹터+종목 뉴스 크롤링")
    parser.add_argument("--stocks", type=str, default="",
                        help="종목명 콤마 구분 (예: '현대바이오,온코닉테라퓨틱스')")
    args = parser.parse_args()

    logger.info("글로벌 시장+섹터 뉴스 크롤링 시작")

    articles = crawl_rss_feeds()

    # 종목 뉴스 추가 검색
    stock_news = []
    if args.stocks:
        names = [n.strip() for n in args.stocks.split(",") if n.strip()]
        if names:
            logger.info(f"종목 뉴스 추가 검색: {names}")
            stock_news = crawl_stock_news(names, days=3)

    high_count = sum(1 for a in articles if a["impact"] == "high")
    med_count = sum(1 for a in articles if a["impact"] == "medium")

    output = {
        "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "article_count": len(articles),
        "high_impact": high_count,
        "medium_impact": med_count,
        "articles": articles,
    }

    # 종목 뉴스가 있으면 별도 섹션으로 저장
    if stock_news:
        output["stock_news"] = stock_news
        output["stock_news_count"] = len(stock_news)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"뉴스 수집 완료: 시장{len(articles)}건 (HIGH:{high_count} MED:{med_count})")
    if stock_news:
        logger.info(f"종목뉴스: {len(stock_news)}건")
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
