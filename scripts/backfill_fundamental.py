"""PER/PBR 펀더멘탈 데이터 백필.

pykrx에서 전종목 PER/PBR을 날짜별로 수집하여
processed parquet에 fund_PER, fund_PBR 컬럼을 추가(갱신)한다.

방식: 날짜 단위 전종목 조회 (get_market_fundamental_by_ticker)
  → 한 번 호출로 ~2700종목 PER/PBR 확보
  → 거래일 수만큼 반복 (3년 ≈ 750회)

사용법:
  python scripts/backfill_fundamental.py                   # 전체 백필 (3년)
  python scripts/backfill_fundamental.py --days 30         # 최근 30일만
  python scripts/backfill_fundamental.py --latest-only     # 최신 1일만 (일일 업데이트용)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FUND_CACHE_DIR = PROJECT_ROOT / "data" / "fundamental_cache"
FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_trading_dates(start: str, end: str) -> list[str]:
    """pykrx에서 거래일 목록 조회."""
    from pykrx import stock as pykrx_stock
    dates = pykrx_stock.get_previous_business_days(
        fromdate=start, todate=end
    )
    return [d.strftime("%Y%m%d") for d in dates]


def fetch_fundamental_by_date(date_str: str) -> pd.DataFrame:
    """특정 날짜의 전종목 PER/PBR 조회."""
    from pykrx import stock as pykrx_stock
    df = pykrx_stock.get_market_fundamental_by_ticker(date_str, market="ALL")
    if df.empty:
        return pd.DataFrame()
    df.index.name = "ticker"
    df = df.reset_index()
    df["date"] = pd.to_datetime(date_str)
    return df[["ticker", "date", "BPS", "PER", "PBR", "EPS", "DIV", "DPS"]]


def backfill_all(days: int = 0, latest_only: bool = False) -> dict:
    """PER/PBR 데이터를 날짜별로 수집하여 캐시에 저장."""

    end_date = datetime.now().strftime("%Y%m%d")

    if latest_only:
        # 최신 거래일 1일만
        dates = get_trading_dates(
            (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"), end_date
        )
        dates = dates[-1:]  # 마지막 거래일만
    elif days > 0:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        dates = get_trading_dates(start_date, end_date)
    else:
        # 3년 전체
        start_date = (datetime.now() - timedelta(days=365 * 3 + 30)).strftime("%Y%m%d")
        dates = get_trading_dates(start_date, end_date)

    logger.info("PER/PBR 백필: %d 거래일 (%s ~ %s)", len(dates), dates[0], dates[-1])

    # 이미 캐시된 날짜 확인
    cached_dates = set()
    cache_file = FUND_CACHE_DIR / "fundamental_daily.parquet"
    if cache_file.exists():
        existing = pd.read_parquet(cache_file)
        cached_dates = set(existing["date"].dt.strftime("%Y%m%d").unique())
        logger.info("기존 캐시: %d일", len(cached_dates))
    else:
        existing = pd.DataFrame()

    # 미수집 날짜만 필터
    new_dates = [d for d in dates if d not in cached_dates]
    if not new_dates:
        logger.info("모든 날짜 이미 캐시됨 — 스킵")
        return {"total_dates": len(dates), "new_dates": 0, "cached": len(cached_dates)}

    logger.info("신규 수집 대상: %d일", len(new_dates))

    # 날짜별 수집
    chunks = []
    for i, d in enumerate(new_dates):
        try:
            df = fetch_fundamental_by_date(d)
            if not df.empty:
                chunks.append(df)

            if (i + 1) % 20 == 0:
                logger.info("  [%d/%d] %s 완료", i + 1, len(new_dates), d)

            # pykrx rate limit 방지
            time.sleep(0.3)
        except Exception as e:
            logger.warning("  %s 수집 실패: %s", d, e)
            time.sleep(1)

    if not chunks:
        logger.info("신규 데이터 없음")
        return {"total_dates": len(dates), "new_dates": 0, "cached": len(cached_dates)}

    new_df = pd.concat(chunks, ignore_index=True)
    logger.info("신규 수집: %d행 (%d일)", len(new_df), len(new_dates))

    # 기존 캐시와 병합
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined.to_parquet(cache_file, index=False)
    logger.info("캐시 저장: %d행 → %s", len(combined), cache_file)

    return {
        "total_dates": len(dates),
        "new_dates": len(new_dates),
        "new_rows": len(new_df),
        "total_rows": len(combined),
    }


def merge_to_processed() -> dict:
    """캐시된 PER/PBR을 processed parquet에 병합."""
    cache_file = FUND_CACHE_DIR / "fundamental_daily.parquet"
    if not cache_file.exists():
        logger.error("캐시 파일 없음: %s", cache_file)
        return {"merged": 0, "skip": 0}

    fund_df = pd.read_parquet(cache_file)
    fund_df["ticker"] = fund_df["ticker"].astype(str).str.zfill(6)
    fund_df["date"] = pd.to_datetime(fund_df["date"])

    # ticker별로 그룹화
    fund_groups = {t: g.set_index("date") for t, g in fund_df.groupby("ticker")}

    processed_files = sorted(PROCESSED_DIR.glob("*.parquet"))
    stats = {"merged": 0, "skip": 0, "no_data": 0}

    for pf in processed_files:
        ticker = pf.stem
        if ticker not in fund_groups:
            stats["no_data"] += 1
            continue

        try:
            df = pd.read_parquet(pf)
            fg = fund_groups[ticker]

            # 기존 fund_PER/PBR 컬럼 제거 (0으로 채워진 기존 데이터)
            for col in ["fund_PER", "fund_PBR", "fund_BPS", "fund_EPS",
                        "fund_DIV", "fund_DPS"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # date 인덱스 기준 병합
            df.index = pd.to_datetime(df.index)
            fg_renamed = fg[["PER", "PBR", "BPS", "EPS", "DIV", "DPS"]].rename(
                columns={
                    "PER": "fund_PER",
                    "PBR": "fund_PBR",
                    "BPS": "fund_BPS",
                    "EPS": "fund_EPS",
                    "DIV": "fund_DIV",
                    "DPS": "fund_DPS",
                }
            )

            df = df.join(fg_renamed, how="left")

            # NaN → 0 (지표 계산에서 NaN 에러 방지)
            for col in ["fund_PER", "fund_PBR", "fund_BPS", "fund_EPS",
                        "fund_DIV", "fund_DPS"]:
                df[col] = df[col].fillna(0)

            df.to_parquet(pf)
            stats["merged"] += 1
        except Exception as e:
            logger.error("%s 병합 실패: %s", ticker, e)
            stats["skip"] += 1

    logger.info(
        "processed 병합 완료: %d종목 성공 / %d 데이터없음 / %d 실패",
        stats["merged"], stats["no_data"], stats["skip"],
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="PER/PBR 펀더멘탈 백필")
    parser.add_argument("--days", type=int, default=0,
                        help="최근 N일만 (0=전체 3년)")
    parser.add_argument("--latest-only", action="store_true",
                        help="최신 1거래일만 (일일 업데이트용)")
    parser.add_argument("--merge-only", action="store_true",
                        help="캐시→processed 병합만 (수집 건너뜀)")
    args = parser.parse_args()

    if not args.merge_only:
        result = backfill_all(days=args.days, latest_only=args.latest_only)
        logger.info("수집 결과: %s", result)

    merge_result = merge_to_processed()
    logger.info("병합 결과: %s", merge_result)


if __name__ == "__main__":
    main()
