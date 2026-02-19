"""
Phase 1: L2 공매도 데이터 backfill

수집 데이터:
  - get_shorting_status_by_date → short_volume, short_ratio
  - get_shorting_balance_by_date → short_balance, lending_balance

4개 컬럼을 기존 parquet에 추가.
공매도 금지 기간(2020-03-16~2021-05-02, 2023-11-06~2025-03-30)은 자연스럽게 0/NaN.

사용법:
  python scripts/backfill_short_data.py --dry-run   # 005930 테스트
  python scripts/backfill_short_data.py              # 전체
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
BACKFILL_START = "20250331"  # 공매도 금지 해제일 (2023-11-06~2025-03-30)
BACKFILL_END = "20260219"

# 6개월 단위 청크 (KRX API 조회 범위 제한)
CHUNK_MONTHS = 6

SHORT_COLS = ["short_volume", "short_ratio", "short_balance", "lending_balance"]


def date_chunks(start_str: str, end_str: str, months: int = 6):
    """날짜 범위를 N개월 단위 청크로 분할"""
    start = datetime.strptime(start_str, "%Y%m%d")
    end = datetime.strptime(end_str, "%Y%m%d")
    chunks = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=months * 30), end)
        chunks.append((cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


def check_needs_backfill(df: pd.DataFrame) -> bool:
    """공매도 컬럼이 없거나 전부 0인지 확인"""
    for col in SHORT_COLS:
        if col not in df.columns:
            return True
    # 컬럼은 있지만 전부 0이면 재수집
    nonzero = 0
    for col in SHORT_COLS:
        nonzero += (df[col] != 0).sum()
    return nonzero == 0


def fetch_short_status(ticker: str) -> pd.DataFrame:
    """공매도 거래현황 (일별): 공매도거래량, 비중 — 6개월 단위 청크 조회"""
    chunks = date_chunks(BACKFILL_START, BACKFILL_END, CHUNK_MONTHS)
    frames = []
    for start, end in chunks:
        try:
            df = krx.get_shorting_status_by_date(start, end, ticker)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            logger.debug(f"  [{ticker}] 공매도 거래현황 {start}~{end}: {e}")
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "date"

    result = pd.DataFrame(index=merged.index)
    if "공매도거래량" in merged.columns:
        result["short_volume"] = merged["공매도거래량"]
    if "비중" in merged.columns:
        result["short_ratio"] = merged["비중"]
    return result


def fetch_short_balance(ticker: str) -> pd.DataFrame:
    """공매도 잔고현황 (일별): 공매도잔고, 대차잔고 — 6개월 단위 청크 조회"""
    chunks = date_chunks(BACKFILL_START, BACKFILL_END, CHUNK_MONTHS)
    frames = []
    for start, end in chunks:
        try:
            df = krx.get_shorting_balance_by_date(start, end, ticker)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            logger.debug(f"  [{ticker}] 공매도 잔고 {start}~{end}: {e}")
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "date"

    result = pd.DataFrame(index=merged.index)
    if "공매도잔고" in merged.columns:
        result["short_balance"] = merged["공매도잔고"]
    for col_name in ["대차잔고", "잔고"]:
        if col_name in merged.columns:
            result["lending_balance"] = merged[col_name]
            break
    return result


def backfill_ticker(ticker: str, parquet_path: Path) -> dict:
    """단일 종목 공매도 backfill"""
    result = {"ticker": ticker, "status": "skip", "updated_cols": 0}

    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)

    if not check_needs_backfill(df):
        result["status"] = "no_need"
        return result

    # 공매도 거래현황 (내부에서 6개월 청크 + sleep 처리)
    status_df = fetch_short_status(ticker)

    # 공매도 잔고현황
    balance_df = fetch_short_balance(ticker)

    if status_df.empty and balance_df.empty:
        result["status"] = "no_data"
        return result

    # 기존 parquet에 컬럼 초기화
    for col in SHORT_COLS:
        if col not in df.columns:
            df[col] = 0.0

    updated = 0

    # 거래현황 병합
    if not status_df.empty:
        common_dates = df.index.intersection(status_df.index)
        for col in ["short_volume", "short_ratio"]:
            if col in status_df.columns:
                for date in common_dates:
                    new_val = status_df.loc[date, col]
                    if pd.notna(new_val) and new_val != 0:
                        df.loc[date, col] = new_val
                        updated += 1

    # 잔고현황 병합
    if not balance_df.empty:
        common_dates = df.index.intersection(balance_df.index)
        for col in ["short_balance", "lending_balance"]:
            if col in balance_df.columns:
                for date in common_dates:
                    new_val = balance_df.loc[date, col]
                    if pd.notna(new_val) and new_val != 0:
                        df.loc[date, col] = new_val
                        updated += 1

    if updated > 0:
        df.to_parquet(parquet_path)
        result["status"] = "updated"
        result["updated_cols"] = updated
    else:
        result["status"] = "all_zero"

    return result


def main():
    parser = argparse.ArgumentParser(description="공매도 데이터 backfill")
    parser.add_argument("--dry-run", action="store_true", help="1종목만 테스트")
    args = parser.parse_args()

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    logger.info(f"총 {len(parquet_files)}개 parquet 파일 발견")

    if args.dry_run:
        test_file = RAW_DIR / "005930.parquet"
        if test_file.exists():
            parquet_files = [test_file]
        else:
            parquet_files = parquet_files[:1]
        logger.info("=== DRY RUN: 1종목만 테스트 ===")

    stats = {"no_need": 0, "updated": 0, "no_data": 0, "all_zero": 0, "error": 0}

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        logger.info(f"[{i+1}/{len(parquet_files)}] {ticker} 처리 중...")

        try:
            result = backfill_ticker(ticker, pf)
            stats[result["status"]] = stats.get(result["status"], 0) + 1

            if result["status"] == "updated":
                logger.info(f"  -> {result['updated_cols']}건 업데이트 완료")
            elif result["status"] == "no_need":
                logger.info(f"  -> backfill 불필요 (이미 데이터 있음)")
            elif result["status"] == "all_zero":
                logger.info(f"  -> API에서도 0 반환")
            elif result["status"] == "no_data":
                logger.info(f"  -> API 데이터 없음")

        except Exception as e:
            logger.error(f"  [{ticker}] 오류: {e}")
            stats["error"] += 1

        # API rate limit (실제 API 호출한 경우만 sleep)
        if result.get("status") in ("updated", "no_data", "all_zero"):
            time.sleep(1.0)

    logger.info("=" * 50)
    logger.info("=== 공매도 Backfill 완료 ===")
    logger.info(f"  업데이트: {stats['updated']}종목")
    logger.info(f"  불필요:   {stats['no_need']}종목")
    logger.info(f"  데이터 없음: {stats['no_data']}종목")
    logger.info(f"  전량 0: {stats['all_zero']}종목")
    logger.info(f"  오류:   {stats['error']}종목")


if __name__ == "__main__":
    main()
