#!/usr/bin/env python3
"""
data/raw/*.parquet 일일 업데이트
═══════════════════════════════════

장마감 후 실행: 전 종목 parquet에 오늘 OHLCV 추가.
scan_surge_pullback.py 의 전처리 단계로 사용.

Usage:
    python scripts/update_raw_parquet.py              # 오늘 데이터 추가
    python scripts/update_raw_parquet.py --date 20260509  # 특정일

BAT-D 연동:
    python -u -X utf8 scripts/update_raw_parquet.py
    python -u -X utf8 scripts/scan_surge_pullback.py --telegram
"""
from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pykrx_quiet import silence_pykrx_logging

silence_pykrx_logging()  # pykrx 로그인/로깅 노이즈 억제 (진입부 1회)

RAW_DIR = PROJECT_ROOT / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def update_parquets(date_str: str):
    """pykrx로 특정일 개별 종목 OHLCV → 기존 parquet에 append"""
    from pykrx import stock as pykrx_stock

    fmt_date = date_str.replace("-", "")
    target_date = pd.Timestamp(fmt_date)

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)
    logger.info("parquet 업데이트 시작: %s, 대상 %d종목", fmt_date, total)

    updated = 0
    skipped = 0
    failed = 0

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        try:
            # 기존 parquet 로드
            df = pd.read_parquet(pf)
            df.index = pd.to_datetime(df.index)

            # 이미 해당 날짜 있으면 스킵
            if target_date in df.index:
                skipped += 1
                continue

            # pykrx로 당일 OHLCV 가져오기
            new_df = pykrx_stock.get_market_ohlcv_by_date(fmt_date, fmt_date, ticker)
            time.sleep(0.15)  # API 부하 방지 (1182종목 × 0.15s ≈ 3분)

            if new_df is None or len(new_df) == 0:
                skipped += 1
                continue

            # 컬럼 정규화 (기존 parquet 형식에 맞추기)
            col_map = {}
            for c in new_df.columns:
                cl = str(c)
                if "시가" in cl: col_map[c] = "open"
                elif "고가" in cl: col_map[c] = "high"
                elif "저가" in cl: col_map[c] = "low"
                elif "종가" in cl: col_map[c] = "close"
                elif "거래량" in cl: col_map[c] = "volume"
                elif "거래대금" in cl: col_map[c] = "trading_value"
                elif "등락률" in cl: col_map[c] = "price_change"
            if col_map:
                new_df = new_df.rename(columns=col_map)

            # 기존 parquet과 같은 컬럼만 유지
            common_cols = [c for c in df.columns if c in new_df.columns]
            if not common_cols:
                skipped += 1
                continue

            new_df = new_df[common_cols]

            # close가 0이면 거래정지 → 스킵
            if "close" in new_df.columns and (new_df["close"] == 0).all():
                skipped += 1
                continue

            # 기존 데이터에 append
            combined = pd.concat([df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(pf)
            updated += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning("  %s 실패: %s", ticker, str(e)[:50])

        # 진행률 표시 (100종목마다)
        if (i + 1) % 200 == 0:
            logger.info("  진행: %d/%d (업데이트:%d 스킵:%d 실패:%d)",
                       i + 1, total, updated, skipped, failed)

    logger.info("="*50)
    logger.info("업데이트 완료: %s", fmt_date)
    logger.info("  업데이트: %d종목", updated)
    logger.info("  스킵(이미존재/거래없음): %d종목", skipped)
    logger.info("  실패: %d종목", failed)
    logger.info("="*50)

    return updated, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="data/raw parquet 일일 업데이트")
    parser.add_argument("--date", type=str, default=None,
                       help="업데이트 날짜 (YYYYMMDD, 기본=오늘)")
    args = parser.parse_args()

    date = args.date or datetime.now().strftime("%Y%m%d")
    update_parquets(date)


if __name__ == "__main__":
    main()
