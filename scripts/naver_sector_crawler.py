"""네이버증권 79개 업종 크롤러 — 전종목 섹터 매핑 생성.

네이버증권의 GICS 기반 79개 업종 분류를 크롤링하여,
전종목(KOSPI+KOSDAQ) 섹터 매핑 CSV를 생성한다.

출력:
  data/sector_rotation/naver_sector_map.csv
    ticker, name, sector, sector_no, market, market_cap

특징:
  - 우선주 포함 (삼성전자우 → 반도체와반도체장비)
  - 79개 세부 업종 (손해보험/생명보험, 반도체/디스플레이 구분)
  - 전종목 ~2,500+ 커버

사용법:
  python scripts/naver_sector_crawler.py              # 전체 크롤링
  python scripts/naver_sector_crawler.py --dry-run    # 3개 업종만 테스트
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ─────────────────────────────────────────────
# 1단계: 업종 목록 크롤링
# ─────────────────────────────────────────────

def fetch_sector_list() -> list[dict]:
    """네이버증권 업종 목록 페이지에서 79개 업종 번호+이름 추출."""
    url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    content = resp.content.decode("euc-kr", errors="replace")

    # type=upjong&no=278 → 반도체와반도체장비
    matches = re.findall(
        r'type=upjong&no=(\d+)[^>]*>([^<]+)<', content
    )

    sectors = []
    seen = set()
    for no, name in matches:
        name = name.strip()
        if no not in seen:
            sectors.append({"sector_no": int(no), "sector": name})
            seen.add(no)

    logger.info("업종 목록: %d개", len(sectors))
    return sectors


# ─────────────────────────────────────────────
# 2단계: 업종별 종목 크롤링 (페이지네이션)
# ─────────────────────────────────────────────

def fetch_sector_stocks(sector_no: int, sector_name: str) -> list[dict]:
    """단일 업종의 전체 종목 목록 크롤링 (멀티페이지)."""
    stocks = []
    page = 1

    while True:
        url = (
            f"https://finance.naver.com/sise/sise_group_detail.naver"
            f"?type=upjong&no={sector_no}&page={page}"
        )
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        content = resp.content.decode("euc-kr", errors="replace")

        # 종목 코드 + 종목명 추출
        # <a href="/item/main.naver?code=005930">삼성전자</a>
        items = re.findall(
            r'/item/main\.naver\?code=(\d{6})"[^>]*>\s*([^<]+?)\s*<',
            content,
        )

        if not items:
            break

        # 중복 체크 (같은 페이지 반복 방지)
        new_codes = set()
        for code, name in items:
            if code not in {s["ticker"] for s in stocks}:
                stocks.append({
                    "ticker": code,
                    "name": name.strip(),
                    "sector": sector_name,
                    "sector_no": sector_no,
                })
                new_codes.add(code)

        if not new_codes:
            break

        # 다음 페이지 존재 확인
        if f"page={page + 1}" not in content:
            break

        page += 1
        time.sleep(0.3)

    return stocks


# ─────────────────────────────────────────────
# 3단계: 시가총액 + 시장구분 보강 (pykrx)
# ─────────────────────────────────────────────

def enrich_with_market_info(df: pd.DataFrame) -> pd.DataFrame:
    """pykrx로 시장구분(KOSPI/KOSDAQ) + 시가총액 추가."""
    try:
        from pykrx import stock

        # 최근 거래일
        date = stock.get_nearest_business_day_in_a_week()

        # KOSPI
        kospi = stock.get_market_sector_classifications(date, "KOSPI")
        kospi["market"] = "KOSPI"

        # KOSDAQ
        kosdaq = stock.get_market_sector_classifications(date, "KOSDAQ")
        kosdaq["market"] = "KOSDAQ"

        krx = pd.concat([kospi, kosdaq])
        krx = krx[["market", "시가총액"]].rename(columns={"시가총액": "market_cap"})

        df = df.merge(
            krx, left_on="ticker", right_index=True, how="left",
        )

        # 시가총액 없는 종목 (상장폐지 등)
        df["market"] = df["market"].fillna("UNKNOWN")
        df["market_cap"] = df["market_cap"].fillna(0).astype(int)

    except Exception as e:
        logger.warning("pykrx 보강 실패 (무시): %s", e)
        df["market"] = "UNKNOWN"
        df["market_cap"] = 0

    return df


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def run_crawler(dry_run: bool = False) -> pd.DataFrame:
    """전체 크롤링 실행."""
    sectors = fetch_sector_list()

    if dry_run:
        sectors = sectors[:3]
        logger.info("DRY-RUN: 상위 %d개 업종만 테스트", len(sectors))

    all_stocks = []

    for i, sec in enumerate(sectors):
        stocks = fetch_sector_stocks(sec["sector_no"], sec["sector"])
        all_stocks.extend(stocks)
        logger.info(
            "[%d/%d] %s (no=%d): %d종목",
            i + 1, len(sectors), sec["sector"], sec["sector_no"], len(stocks),
        )
        time.sleep(0.5)  # 네이버 부하 방지

    df = pd.DataFrame(all_stocks)

    # 중복 제거 (동일 종목이 2개 업종에 나올 수 있음 — 거의 없지만 안전장치)
    before = len(df)
    df.drop_duplicates(subset="ticker", keep="first", inplace=True)
    if len(df) < before:
        logger.info("중복 제거: %d → %d", before, len(df))

    # pykrx로 시장구분 + 시가총액 보강
    logger.info("시장구분/시가총액 보강 중...")
    df = enrich_with_market_info(df)

    # 시가총액 순 정렬
    df.sort_values("market_cap", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "naver_sector_map.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("저장 → %s (%d종목, %d업종)", out_path, len(df), df["sector"].nunique())

    # 요약 출력
    print(f"\n{'=' * 60}")
    print(f"  네이버증권 업종 매핑 완료")
    print(f"{'=' * 60}")
    print(f"  총 종목: {len(df)}")
    print(f"  업종 수: {df['sector'].nunique()}")
    print(f"  KOSPI: {len(df[df['market'] == 'KOSPI'])}")
    print(f"  KOSDAQ: {len(df[df['market'] == 'KOSDAQ'])}")
    print(f"  UNKNOWN: {len(df[df['market'] == 'UNKNOWN'])}")

    print(f"\n  업종별 종목 수:")
    sector_counts = df["sector"].value_counts()
    for sector, cnt in sector_counts.items():
        print(f"    {sector:<20} {cnt:>4}종목")

    return df


def main():
    parser = argparse.ArgumentParser(description="네이버증권 79개 업종 크롤러")
    parser.add_argument("--dry-run", action="store_true",
                        help="3개 업종만 테스트")
    args = parser.parse_args()

    run_crawler(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
