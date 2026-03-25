"""KOSPI 투자자별 순매수 수집 — 네이버 금융 스크래핑.

증분 업데이트 방식으로 data/kospi_investor_flow.csv를 갱신한다.
기존 CSV가 있으면 마지막 날짜 이후만 수집, 없으면 최근 60영업일 수집.

출력 컬럼: Date, foreign_net, inst_net, retail_net (억원 단위)

Usage:
    python scripts/collect_investor_flow.py          # 증분 업데이트
    python scripts/collect_investor_flow.py --full    # 전체 재수집 (60영업일)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "data" / "kospi_investor_flow.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

NAVER_URL = "https://finance.naver.com/sise/investorDealTrendDay.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://finance.naver.com/sise/sise_deal.naver",
}
MAX_RETRIES = 3
RETRY_DELAY = 5


def _parse_amount(text: str) -> float:
    """쉼표/공백 제거 후 float 변환."""
    text = text.replace(",", "").replace(" ", "").replace("\xa0", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _fetch_page(page_num: int, bizdate: str) -> list[dict]:
    """네이버 금융 단일 페이지 스크래핑. 실패 시 3회 재시도."""
    params = {"bizdate": bizdate, "sosok": "01", "page": page_num}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                NAVER_URL, headers=HEADERS, params=params, timeout=10
            )
            resp.encoding = "euc-kr"
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.error("페이지 %d 요청 실패 (%d회): %s", page_num, attempt, e)
                return []
            time.sleep(RETRY_DELAY)

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "type_1"})
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        date_text = cells[0].get_text(strip=True)
        if not date_text or "." not in date_text:
            continue

        try:
            dt = datetime.strptime(f"20{date_text}", "%Y.%m.%d")
        except ValueError:
            continue

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "foreign_net": _parse_amount(cells[2].get_text(strip=True)),
            "inst_net": _parse_amount(cells[3].get_text(strip=True)),
            "retail_net": _parse_amount(cells[1].get_text(strip=True)),
        })

    return rows


def collect(full: bool = False) -> pd.DataFrame:
    """투자자수급 데이터 수집.

    Args:
        full: True면 전체 재수집(60영업일), False면 증분 업데이트.

    Returns:
        갱신된 전체 DataFrame.
    """
    # 기존 CSV 로드
    existing = pd.DataFrame()
    last_date = None
    if CSV_PATH.exists() and not full:
        existing = pd.read_csv(CSV_PATH)
        if len(existing) > 0:
            last_date = existing["Date"].max()
            logger.info("기존 CSV: %d행, 마지막 날짜: %s", len(existing), last_date)

    # 수집 범위 결정
    if full or last_date is None:
        target_days = 90  # 달력일 기준 (≈60영업일)
        logger.info("전체 수집 모드: 최근 %d일", target_days)
    else:
        last_dt = datetime.strptime(last_date, "%Y-%m-%d")
        gap_days = (datetime.now() - last_dt).days
        if gap_days <= 0:
            logger.info("이미 최신입니다. 수집 불필요.")
            return existing
        target_days = gap_days + 5  # 여유분
        logger.info("증분 수집: %s 이후 %d일", last_date, gap_days)

    cutoff = datetime.now() - timedelta(days=target_days)
    bizdate = datetime.today().strftime("%Y%m%d")
    max_pages = target_days // 10 + 20

    # 페이지별 수집
    all_rows = []
    for page_num in range(1, max_pages + 1):
        rows = _fetch_page(page_num, bizdate)
        if not rows:
            break

        all_rows.extend(rows)
        oldest = rows[-1]["Date"]

        if page_num % 10 == 0:
            logger.info("  ... %d페이지, %d행 (최고일: %s)", page_num, len(all_rows), oldest)

        # cutoff 도달
        if datetime.strptime(oldest, "%Y-%m-%d") < cutoff:
            break

        time.sleep(0.15)

    if not all_rows:
        logger.warning("수집 결과 0건! 네이버 금융 응답 확인 필요.")
        return existing

    new_df = pd.DataFrame(all_rows)
    logger.info("신규 수집: %d행 (%s ~ %s)",
                len(new_df), new_df["Date"].min(), new_df["Date"].max())

    # 병합 (증분 모드)
    if not existing.empty and not full:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
    else:
        combined = new_df

    # 정렬 + 저장
    combined = combined.sort_values("Date").reset_index(drop=True)
    combined.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("저장: %d행 → %s (마지막: %s)",
                len(combined), CSV_PATH.name, combined["Date"].iloc[-1])

    # stale 체크
    last = datetime.strptime(combined["Date"].iloc[-1], "%Y-%m-%d")
    gap = (datetime.now() - last).days
    if gap > 5:  # 3영업일 ≈ 5달력일
        logger.warning("CSV 마지막 날짜가 %d일 전입니다. 수집 결과 확인 필요.", gap)

    return combined


def main():
    parser = argparse.ArgumentParser(description="KOSPI 투자자수급 수집")
    parser.add_argument("--full", action="store_true", help="전체 재수집 (60영업일)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  KOSPI 투자자별 순매수 수집 — 네이버 금융")
    logger.info("=" * 60)

    df = collect(full=args.full)

    if df.empty:
        logger.error("수집 실패!")
        sys.exit(1)

    logger.info("완료: %d행, 마지막 %s", len(df), df["Date"].iloc[-1])


if __name__ == "__main__":
    main()
