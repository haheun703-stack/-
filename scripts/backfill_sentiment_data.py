"""
Phase 4: L5 네이버 종목토론실 센티먼트 크롤링

비관 키워드 빈도로 패닉 센티먼트 측정.
제약: 과거 데이터 없음 → 백테스트에는 중립(0.5) 사용.
실운용 시: 매일 장마감 후 크롤링하여 데이터 축적.

사용법:
  python scripts/backfill_sentiment_data.py --dry-run   # 005930만
  python scripts/backfill_sentiment_data.py              # 전체
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
SENTIMENT_DIR = Path("data/sentiment")
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 비관 키워드
PESSIMISM_KEYWORDS = [
    "손절", "물타기", "망했다", "폭락", "끝", "탈출", "손실",
    "개미", "작전", "눈물", "떡락", "깡통", "반토막", "무너",
    "탈출하세요", "도망", "피눈물", "후회",
]


def fetch_discussion_sentiment(ticker: str, pages: int = 5) -> dict:
    """네이버 종목토론실 비관도 측정

    Returns:
        {
            "pessimism_ratio": float (0~1),
            "total_posts": int,
            "pessimism_count": int,
        }
    """
    total_posts = 0
    pessimism_count = 0

    for page in range(1, pages + 1):
        url = (
            f"https://finance.naver.com/item/board.naver"
            f"?code={ticker}&page={page}"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = "euc-kr"
            soup = BeautifulSoup(resp.text, "html.parser")

            # 게시글 제목 추출
            titles = soup.select("td.title a")
            for a_tag in titles:
                title_text = a_tag.get_text(strip=True)
                if not title_text:
                    continue
                total_posts += 1
                for kw in PESSIMISM_KEYWORDS:
                    if kw in title_text:
                        pessimism_count += 1
                        break  # 1게시글 1카운트

        except Exception as e:
            logger.debug(f"  [{ticker}] page {page} 크롤링 실패: {e}")

        time.sleep(0.5)

    if total_posts == 0:
        return {"pessimism_ratio": 0.5, "total_posts": 0, "pessimism_count": 0}

    return {
        "pessimism_ratio": pessimism_count / total_posts,
        "total_posts": total_posts,
        "pessimism_count": pessimism_count,
    }


def main():
    parser = argparse.ArgumentParser(description="센티먼트 데이터 수집")
    parser.add_argument("--dry-run", action="store_true", help="005930만 테스트")
    args = parser.parse_args()

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    tickers = [p.stem for p in parquet_files]

    if args.dry_run:
        tickers = ["005930"]
        logger.info("=== DRY RUN: 005930만 테스트 ===")

    logger.info(f"총 {len(tickers)}종목 센티먼트 수집")

    results = {}
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] {ticker} 토론실 크롤링...")
        sentiment = fetch_discussion_sentiment(ticker, pages=3)
        results[ticker] = sentiment

        logger.info(
            f"  -> 비관도: {sentiment['pessimism_ratio']:.2%} "
            f"({sentiment['pessimism_count']}/{sentiment['total_posts']})"
        )
        time.sleep(1.0)  # 네이버 차단 방지

    # 결과를 단일 parquet로 저장
    rows = []
    for ticker, sent in results.items():
        rows.append({
            "ticker": ticker,
            "pessimism_ratio": sent["pessimism_ratio"],
            "total_posts": sent["total_posts"],
            "pessimism_count": sent["pessimism_count"],
        })

    if rows:
        df = pd.DataFrame(rows)
        path = SENTIMENT_DIR / "latest_sentiment.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"저장 완료: {path} ({len(df)}종목)")

    # 각 종목 parquet에 센티먼트 컬럼 추가 (최신 1개 값만 — 과거는 중립)
    for ticker in tickers:
        ppath = RAW_DIR / f"{ticker}.parquet"
        if not ppath.exists():
            continue

        df = pd.read_parquet(ppath)
        if "sentiment_pessimism" not in df.columns:
            df["sentiment_pessimism"] = 0.5  # 기본 중립

        # 최근 날짜에만 실제 값 기록
        if ticker in results and not df.empty:
            last_date = df.index[-1]
            df.loc[last_date, "sentiment_pessimism"] = results[ticker]["pessimism_ratio"]

        df.to_parquet(ppath)

    logger.info("=== 센티먼트 Backfill 완료 ===")


if __name__ == "__main__":
    main()
