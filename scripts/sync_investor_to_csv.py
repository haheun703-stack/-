#!/usr/bin/env python3
"""투자자별 순매수 → stock_data_daily CSV 동기화

수급 데이터 소스 우선순위:
  1) 단타봇 flow CSV (primary) — /home/ubuntu/bodyhunter/scalper-agent/data_store/flow/
  2) investor_daily.db (fallback) — 단타봇 미수집 시 또는 로컬 환경

Usage:
    python scripts/sync_investor_to_csv.py              # 자동감지 (VPS=단타봇, 로컬=DB)
    python scripts/sync_investor_to_csv.py --source scalper  # 단타봇 강제
    python scripts/sync_investor_to_csv.py --source db       # DB 강제
    python scripts/sync_investor_to_csv.py --dry-run    # 변경 미적용, 통계만
    python scripts/sync_investor_to_csv.py --ticker 005930  # 특정 종목만
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
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

# 단타봇 수급 데이터 경로 (같은 VPS)
SCALPER_FLOW_DIR = Path("/home/ubuntu/bodyhunter/scalper-agent/data_store/flow")
SCALPER_META_FILE = SCALPER_FLOW_DIR / "_last_update.json"


def check_scalper_freshness() -> tuple[bool, str]:
    """단타봇 수급 데이터 신선도 확인.

    Returns:
        (is_fresh, message)
    """
    if not SCALPER_FLOW_DIR.exists():
        return False, "단타봇 flow 디렉토리 없음"

    if not SCALPER_META_FILE.exists():
        return False, "_last_update.json 없음"

    try:
        meta = json.loads(SCALPER_META_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"meta 파싱 실패: {e}"

    scalper_date = meta.get("date", "")
    today = datetime.now().strftime("%Y-%m-%d")

    if scalper_date != today:
        return False, f"단타봇 수급 미갱신 (last={scalper_date}, today={today})"

    count = meta.get("investor", 0)
    if count < 100:
        return False, f"수집 종목 부족 ({count}종목)"

    return True, f"OK — {count}종목 수집 완료 ({scalper_date})"


def load_from_scalper(flow_dir: Path) -> dict[str, pd.DataFrame]:
    """단타봇 flow CSV에서 4유형 수급 로드.

    Returns:
        {ticker: DataFrame(index=date, cols=[foreign_net, inst_net, corp_net])} (억원 단위)
    """
    result = {}
    csv_files = list(flow_dir.glob("*_investor.csv"))

    for csv_path in csv_files:
        ticker = csv_path.stem.replace("_investor", "")

        # NXT 코드(영문 포함) 스킵 — 퀀트봇 stock_data_daily는 순수 숫자 6자리만
        if not ticker.isdigit():
            continue

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            continue

        if "외국인_금액" not in df.columns:
            continue

        # 백만원 → 억원 변환 (1억 = 100백만)
        df = df.rename(columns={
            "외국인_금액": "foreign_net",
            "기관_금액": "inst_net",
            "기타법인_금액": "corp_net",
        })

        for col in ["foreign_net", "inst_net", "corp_net"]:
            if col in df.columns:
                df[col] = (df[col] / 100).round(1)
            else:
                df[col] = 0.0

        # date 인덱스 설정
        df["date"] = df["date"].astype(str)
        df = df.set_index("date")[["foreign_net", "inst_net", "corp_net"]]
        result[ticker] = df

    logger.info("단타봇 로드: %d종목 / 평균 %d거래일",
                len(result),
                int(sum(len(v) for v in result.values()) / max(len(result), 1)))
    return result


def load_from_db(db_path: Path) -> dict[str, pd.DataFrame]:
    """DB에서 투자자별 순매수 데이터 로드 (폴백용).

    Returns:
        {ticker: DataFrame(index=date, cols=[foreign_net, inst_net, corp_net])} (억원 단위)
    """
    if not db_path.exists():
        logger.error("DB 없음: %s", db_path)
        return {}

    conn = sqlite3.connect(str(db_path), timeout=30)

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

    result = {}
    for ticker, group in df.groupby("ticker"):
        group = group.set_index("date").sort_index()
        result[ticker] = group[["foreign_net", "inst_net", "corp_net"]]

    logger.info("DB 로드: %d종목 / %d거래일", len(result), df["date"].nunique())
    return result


def sync_csv(csv_path: Path, investor_df: pd.DataFrame, dry_run: bool = False) -> dict:
    """단일 CSV 파일에 수급 데이터 동기화 (벡터화 merge)."""
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

    # investor_df 인덱스를 YYYY-MM-DD로 통일
    inv = investor_df.copy()
    inv.index = pd.to_datetime(inv.index).strftime("%Y-%m-%d")
    inv = inv.rename(columns={
        "foreign_net": "Foreign_Net",
        "inst_net": "Inst_Net",
        "corp_net": "Corp_Net",
    })

    # Date 기준 merge → 벡터화 업데이트 (iterrows 제거)
    matching = df["Date"].isin(inv.index)
    updated = int(matching.sum())

    if updated > 0:
        df_indexed = df.set_index("Date")
        df_indexed.update(inv)
        df = df_indexed.reset_index()

        if not dry_run:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {"status": "OK", "updated": updated}


def main():
    parser = argparse.ArgumentParser(description="투자자별 순매수 → CSV 동기화")
    parser.add_argument("--dry-run", action="store_true", help="변경 미적용")
    parser.add_argument("--ticker", type=str, default=None, help="특정 종목만")
    parser.add_argument("--source", choices=["auto", "scalper", "db"], default="auto",
                        help="수급 소스 (auto=자동감지, scalper=단타봇, db=investor_daily.db)")
    args = parser.parse_args()

    # 소스 결정
    source = args.source
    investor_data = {}

    if source == "auto":
        fresh, msg = check_scalper_freshness()
        if fresh:
            source = "scalper"
            logger.info("소스: 단타봇 flow CSV (%s)", msg)
        else:
            source = "db"
            logger.info("소스: investor_daily.db (단타봇 미사용: %s)", msg)

    if source == "scalper":
        fresh, msg = check_scalper_freshness()
        if not fresh:
            logger.warning("단타봇 데이터 비정상: %s → DB 폴백", msg)
            source = "db"
        else:
            investor_data = load_from_scalper(SCALPER_FLOW_DIR)
            if not investor_data:
                logger.warning("단타봇 로드 실패 → DB 폴백")
                source = "db"

    if source == "db":
        investor_data = load_from_db(DB_PATH)

    if not investor_data:
        logger.error("수급 데이터 없음 — 단타봇 + DB 모두 실패")
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
    logger.info("%s 완료: [%s] %d파일 / %d행 업데이트", action, source, total_files, total_updated)


if __name__ == "__main__":
    main()
