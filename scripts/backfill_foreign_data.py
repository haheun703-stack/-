"""
Phase 1-1: 외국인/기관/개인 투자자 매매동향 backfill (2025+)

문제: 2025-01-01 이후 외국인합계/기관합계/개인 컬럼이 모두 0
원인: 원본 수집 시 pykrx API가 아직 반영 안 된 데이터 or 수집 오류
해결: PyKRX로 2025년 이후 투자자 매매동향을 재수집 → parquet 업데이트

사용법:
  python scripts/backfill_foreign_data.py
  python scripts/backfill_foreign_data.py --dry-run   # 테스트 (1종목만)
"""

import argparse
import logging
import sys
import time
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
BACKFILL_START = "20250101"
BACKFILL_END = "20260214"

# 투자자 매매동향 컬럼 매핑
INVESTOR_COLS = ["기관합계", "외국인합계", "개인"]


def check_needs_backfill(df: pd.DataFrame) -> bool:
    """2025년 이후 외국인합계가 모두 0인지 확인"""
    if "외국인합계" not in df.columns:
        return False

    mask_2025 = df.index >= "2025-01-01"
    if mask_2025.sum() == 0:
        return False

    foreign_2025 = df.loc[mask_2025, "외국인합계"]
    nonzero = (foreign_2025 != 0).sum()
    return nonzero == 0


def fetch_investor_data(ticker: str) -> pd.DataFrame:
    """PyKRX에서 투자자 매매동향 수집"""
    try:
        inv_df = krx.get_market_trading_value_by_date(
            BACKFILL_START, BACKFILL_END, ticker
        )
        if inv_df is None or inv_df.empty:
            return pd.DataFrame()

        inv_df.index.name = "date"

        # 필요한 컬럼만 선택
        available = [c for c in INVESTOR_COLS if c in inv_df.columns]
        if not available:
            return pd.DataFrame()

        return inv_df[available]

    except Exception as e:
        logger.warning(f"  [{ticker}] 투자자 매매동향 수집 실패: {e}")
        return pd.DataFrame()


def backfill_ticker(ticker: str, parquet_path: Path) -> dict:
    """단일 종목 backfill 실행"""
    result = {"ticker": ticker, "status": "skip", "updated_rows": 0}

    # 1. 기존 parquet 읽기
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)

    # 2. backfill 필요 여부 확인
    if not check_needs_backfill(df):
        result["status"] = "no_need"
        return result

    # 3. PyKRX에서 투자자 데이터 수집
    inv_df = fetch_investor_data(ticker)
    if inv_df.empty:
        result["status"] = "no_data"
        return result

    inv_df.index = pd.to_datetime(inv_df.index)

    # 4. 2025년 이후 날짜만 필터
    mask_2025 = df.index >= "2025-01-01"
    dates_to_update = df.index[mask_2025]

    updated = 0
    for date in dates_to_update:
        if date in inv_df.index:
            for col in INVESTOR_COLS:
                if col in inv_df.columns and col in df.columns:
                    new_val = inv_df.loc[date, col]
                    old_val = df.loc[date, col]
                    if old_val == 0 and new_val != 0:
                        df.loc[date, col] = new_val
                        updated += 1

    if updated > 0:
        df.to_parquet(parquet_path)
        result["status"] = "updated"
        result["updated_rows"] = updated
    else:
        result["status"] = "all_zero"  # PyKRX에서도 0 반환

    return result


def main():
    parser = argparse.ArgumentParser(description="외국인 데이터 backfill")
    parser.add_argument("--dry-run", action="store_true", help="1종목만 테스트")
    args = parser.parse_args()

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    logger.info(f"총 {len(parquet_files)}개 parquet 파일 발견")

    if args.dry_run:
        # 삼성전자로 테스트
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
                logger.info(f"  -> {result['updated_rows']}건 업데이트 완료")
            elif result["status"] == "no_need":
                logger.info(f"  -> backfill 불필요 (이미 데이터 있음)")
            elif result["status"] == "all_zero":
                logger.info(f"  -> PyKRX에서도 0 반환 (데이터 미제공)")
            elif result["status"] == "no_data":
                logger.info(f"  -> PyKRX 데이터 없음")

        except Exception as e:
            logger.error(f"  [{ticker}] 오류: {e}")
            stats["error"] += 1

        # API rate limit 방지
        time.sleep(0.8)

    logger.info("=" * 50)
    logger.info("=== Backfill 완료 ===")
    logger.info(f"  업데이트: {stats['updated']}종목")
    logger.info(f"  불필요:   {stats['no_need']}종목")
    logger.info(f"  데이터 없음: {stats['no_data']}종목")
    logger.info(f"  전량 0: {stats['all_zero']}종목")
    logger.info(f"  오류:   {stats['error']}종목")


if __name__ == "__main__":
    main()
