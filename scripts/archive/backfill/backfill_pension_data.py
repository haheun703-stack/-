"""
Phase 5: L6 연기금 순매수 상위 데이터 backfill

pykrx get_market_net_purchases_of_equities_by_ticker()로
일별 연기금 순매수 TOP20 종목을 구하고 종목 parquet에 플래그 추가.

추가 컬럼:
  - pension_top_buyer: 해당일 연기금 순매수 TOP20 진입 여부 (0/1)
  - pension_net_5d: 연기금 5일 누적 순매수 (원)

사용법:
  python scripts/backfill_pension_data.py --dry-run
  python scripts/backfill_pension_data.py
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx
except ImportError:
    logger.error("pykrx 미설치: pip install pykrx")
    sys.exit(1)


RAW_DIR = Path("data/raw")
BACKFILL_START = "20190101"
BACKFILL_END = "20260214"
TOP_N = 20


def get_business_days(start: str, end: str) -> list[str]:
    """영업일 목록 생성 (pykrx 호출 없이 pandas)"""
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y%m%d") for d in dates]


def fetch_pension_top_by_date(date: str) -> dict[str, int]:
    """특정일 연기금 순매수 상위 종목 dict {ticker: net_amount}"""
    try:
        df = krx.get_market_net_purchases_of_equities_by_ticker(
            date, date, "KOSPI"
        )
        if df is None or df.empty:
            return {}

        # 연기금등 컬럼 확인
        pension_col = None
        for col_name in ["연기금등", "연기금"]:
            if col_name in df.columns:
                pension_col = col_name
                break
        if pension_col is None:
            return {}

        # TOP N 추출
        top = df.nlargest(TOP_N, pension_col)
        result = {}
        for ticker in top.index:
            result[ticker] = int(top.loc[ticker, pension_col])
        return result

    except Exception as e:
        logger.debug(f"  [{date}] 연기금 데이터 조회 실패: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="연기금 순매수 backfill")
    parser.add_argument("--dry-run", action="store_true", help="최근 5일만 테스트")
    args = parser.parse_args()

    # 기존 parquet 종목 목록
    parquet_files = {p.stem: p for p in sorted(RAW_DIR.glob("*.parquet"))}
    logger.info(f"총 {len(parquet_files)}개 parquet 파일")

    # 영업일 목록
    biz_days = get_business_days(BACKFILL_START, BACKFILL_END)
    if args.dry_run:
        biz_days = biz_days[-5:]
        logger.info(f"=== DRY RUN: 최근 {len(biz_days)}일만 테스트 ===")
    else:
        logger.info(f"총 {len(biz_days)} 영업일 처리 예정")

    # {ticker: {date: pension_net}} 수집
    pension_data: dict[str, dict[str, int]] = {}
    pension_top_data: dict[str, set] = {}  # {date_str: set(tickers)}

    for i, date in enumerate(biz_days):
        if i % 50 == 0:
            logger.info(f"  [{i+1}/{len(biz_days)}] {date} 처리 중...")

        top = fetch_pension_top_by_date(date)
        date_key = pd.Timestamp(datetime.strptime(date, "%Y%m%d"))

        if top:
            pension_top_data[date] = set(top.keys())
            for ticker, net_amount in top.items():
                if ticker not in pension_data:
                    pension_data[ticker] = {}
                pension_data[ticker][date_key] = net_amount

        time.sleep(0.5)

    logger.info(f"수집 완료: {len(pension_data)}종목 데이터")

    # parquet 업데이트
    updated_count = 0
    for ticker, ppath in parquet_files.items():
        df = pd.read_parquet(ppath)
        df.index = pd.to_datetime(df.index)

        # pension_top_buyer 초기화
        if "pension_top_buyer" not in df.columns:
            df["pension_top_buyer"] = 0
        if "pension_net" not in df.columns:
            df["pension_net"] = 0

        changed = False

        # 연기금 순매수 데이터 추가
        if ticker in pension_data:
            for date_ts, net_amount in pension_data[ticker].items():
                if date_ts in df.index:
                    df.loc[date_ts, "pension_net"] = net_amount
                    df.loc[date_ts, "pension_top_buyer"] = 1
                    changed = True

        if changed:
            df.to_parquet(ppath)
            updated_count += 1

    logger.info(f"=== 연기금 Backfill 완료: {updated_count}종목 업데이트 ===")


if __name__ == "__main__":
    main()
