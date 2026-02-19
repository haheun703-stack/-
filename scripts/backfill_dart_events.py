"""
Phase 3: L3 DART 실적 서프라이즈 데이터 backfill

DART API로 분기별 영업이익을 조회하여 턴어라운드/증익 이벤트를 감지.
DART_API_KEY 환경변수 필수 (없으면 자동 스킵).

추가 컬럼:
  - earnings_surprise: 1=턴어라운드(적자→흑자), 0=기타
  - qoq_oi_growth: 분기 영업이익 성장률 (%)

사용법:
  python scripts/backfill_dart_events.py --dry-run   # 005930만
  python scripts/backfill_dart_events.py              # 전체

필수: export DART_API_KEY=your_key
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.dart_adapter import DartAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def main():
    parser = argparse.ArgumentParser(description="DART 실적 backfill")
    parser.add_argument("--dry-run", action="store_true", help="005930만 테스트")
    args = parser.parse_args()

    api_key = os.getenv("DART_API_KEY", "")
    if not api_key:
        logger.warning("DART_API_KEY 미설정 — 실적 서프라이즈 데이터 스킵")
        logger.info("기존 parquet에 earnings_surprise=0, qoq_oi_growth=0 기본값 추가")

        # 기본값 추가만 수행
        for ppath in sorted(RAW_DIR.glob("*.parquet")):
            df = pd.read_parquet(ppath)
            changed = False
            if "earnings_surprise" not in df.columns:
                df["earnings_surprise"] = 0
                changed = True
            if "qoq_oi_growth" not in df.columns:
                df["qoq_oi_growth"] = 0.0
                changed = True
            if changed:
                df.to_parquet(ppath)
        logger.info("기본값 추가 완료")
        return

    adapter = DartAdapter(api_key=api_key)
    if not adapter.is_available:
        logger.error("DART API 사용 불가")
        return

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    tickers = [p.stem for p in parquet_files]

    if args.dry_run:
        tickers = ["005930"]
        parquet_files = [RAW_DIR / "005930.parquet"]
        logger.info("=== DRY RUN: 005930만 테스트 ===")

    logger.info(f"총 {len(tickers)}종목 DART 실적 조회")

    # 분석 연도 (최근 3년)
    years = [2024, 2025, 2026]

    updated = 0
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
        ppath = RAW_DIR / f"{ticker}.parquet"
        df = pd.read_parquet(ppath)
        df.index = pd.to_datetime(df.index)

        if "earnings_surprise" not in df.columns:
            df["earnings_surprise"] = 0
        if "qoq_oi_growth" not in df.columns:
            df["qoq_oi_growth"] = 0.0

        for year in years:
            try:
                turnaround = adapter.get_qoq_turnaround(ticker, year)
                if turnaround["turnaround"]:
                    # 해당 연도의 첫 영업일부터 90일간 플래그
                    year_start = f"{year}-01-01"
                    year_end = f"{year}-04-01"
                    mask = (df.index >= year_start) & (df.index <= year_end)
                    df.loc[mask, "earnings_surprise"] = 1
                    logger.info(f"  {year} 턴어라운드 감지!")

                if turnaround["qoq_oi_growth"] != 0:
                    year_mask = df.index.year == year
                    # 가장 최근 분기에 반영
                    df.loc[year_mask, "qoq_oi_growth"] = turnaround["qoq_oi_growth"]

            except Exception as e:
                logger.debug(f"  [{ticker}] {year}: {e}")

            time.sleep(0.2)

        df.to_parquet(ppath)
        updated += 1

        # API rate limit
        time.sleep(0.5)

        # API 호출 횟수 체크
        if adapter.get_api_calls_count() > 9000:
            logger.warning("DART API 일일 한도 근접 (9000/10000), 중단")
            break

    logger.info(f"=== DART Backfill 완료: {updated}종목 ===")
    logger.info(f"API 호출: {adapter.get_api_calls_count()}건")


if __name__ == "__main__":
    main()
