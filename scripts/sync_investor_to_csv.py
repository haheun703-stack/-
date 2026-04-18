#!/usr/bin/env python3
"""투자자별 순매수 DB → stock_data_daily CSV 동기화

investor_daily.db에서 외국인/기관/기타법인 순매수(억원)를
stock_data_daily/*.csv의 Foreign_Net/Inst_Net 컬럼에 반영.
기타법인은 Corp_Net 신규 컬럼으로 추가.

Usage:
    python scripts/sync_investor_to_csv.py              # 전체 동기화
    python scripts/sync_investor_to_csv.py --dry-run    # 변경 미적용, 통계만
    python scripts/sync_investor_to_csv.py --ticker 005930  # 특정 종목만
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
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

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"


def load_investor_data(db_path: Path) -> dict[str, pd.DataFrame]:
    """DB에서 투자자별 순매수 데이터 로드 (티커별 그룹핑).

    Returns:
        {ticker: DataFrame(date, foreign_net, inst_net, corp_net)} (억원 단위)
    """
    if not db_path.exists():
        logger.error("DB 없음: %s", db_path)
        return {}

    conn = sqlite3.connect(str(db_path), timeout=30)

    # 핵심 3주체 피벗: 외국인, 기관합계, 기타법인 (net_val = 순매수금액, 원 단위)
    query = """
    SELECT date, ticker,
           SUM(CASE WHEN investor = '외국인' THEN net_val ELSE 0 END) as foreign_net,
           SUM(CASE WHEN investor = '기관합계' THEN net_val ELSE 0 END) as inst_net,
           SUM(CASE WHEN investor = '기타법인' THEN net_val ELSE 0 END) as corp_net
    FROM investor_daily
    WHERE investor IN ('외국인', '기관합계', '기타법인')
    GROUP BY date, ticker
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return {}

    # 원 → 억원 변환
    for col in ["foreign_net", "inst_net", "corp_net"]:
        df[col] = (df[col] / 1e8).round(1)

    # 티커별 그룹핑
    result = {}
    for ticker, group in df.groupby("ticker"):
        group = group.set_index("date").sort_index()
        result[ticker] = group[["foreign_net", "inst_net", "corp_net"]]

    logger.info("DB 로드: %d종목 / %d거래일", len(result), df["date"].nunique())
    return result


def sync_csv(csv_path: Path, investor_df: pd.DataFrame, dry_run: bool = False) -> dict:
    """단일 CSV 파일에 수급 데이터 동기화."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        return {"status": "READ_FAIL", "updated": 0, "error": str(e)}

    if "Date" not in df.columns:
        return {"status": "NO_DATE", "updated": 0}

    # Date 형식 통일 (YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # 컬럼 없으면 추가
    for col in ["Foreign_Net", "Inst_Net", "Corp_Net"]:
        if col not in df.columns:
            df[col] = 0.0

    # DB 날짜 형식 통일
    inv_dates = set(investor_df.index)

    updated = 0
    for i, row in df.iterrows():
        date_str = row["Date"]
        # DB 키는 YYYYMMDD 형식일 수 있으므로 변환
        date_key = date_str.replace("-", "")

        # DB에서 해당 날짜 찾기 (YYYY-MM-DD 또는 YYYYMMDD)
        inv_row = None
        if date_str in inv_dates:
            inv_row = investor_df.loc[date_str]
        elif date_key in inv_dates:
            inv_row = investor_df.loc[date_key]

        if inv_row is not None:
            df.at[i, "Foreign_Net"] = inv_row["foreign_net"]
            df.at[i, "Inst_Net"] = inv_row["inst_net"]
            df.at[i, "Corp_Net"] = inv_row["corp_net"]
            updated += 1

    if updated > 0 and not dry_run:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {"status": "OK", "updated": updated}


def main():
    parser = argparse.ArgumentParser(description="투자자별 순매수 DB→CSV 동기화")
    parser.add_argument("--dry-run", action="store_true", help="변경 미적용")
    parser.add_argument("--ticker", type=str, default=None, help="특정 종목만")
    args = parser.parse_args()

    investor_data = load_investor_data(DB_PATH)
    if not investor_data:
        logger.error("수급 데이터 없음 — 먼저 collect_investor_bulk.py 실행")
        sys.exit(1)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    logger.info("CSV 파일: %d개", len(csv_files))

    total_updated = 0
    total_files = 0
    processed = 0

    for path in csv_files:
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, ticker = parts

        if args.ticker and ticker != args.ticker:
            continue

        if ticker not in investor_data:
            continue

        result = sync_csv(path, investor_data[ticker], dry_run=args.dry_run)
        if result["status"] == "OK" and result["updated"] > 0:
            total_files += 1
            total_updated += result["updated"]

        processed += 1
        if processed % 500 == 0:
            logger.info("  %d종목 처리... (%d파일 업데이트)", processed, total_files)

    action = "시뮬레이션" if args.dry_run else "동기화"
    logger.info("%s 완료: %d파일 / %d행 업데이트", action, total_files, total_updated)


if __name__ == "__main__":
    main()
