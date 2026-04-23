#!/usr/bin/env python3
"""전종목 투자자별 순매수 일괄 수집 — pykrx 벌크 모드

핵심 6주체(외국인/기관합계/기타법인/개인/연기금/금융투자) + 세부 5주체(투신/보험/은행/사모/기타금융)
KOSPI + KOSDAQ 전종목을 한 번에 수집하여 SQLite에 저장.

Usage:
    python scripts/collect_investor_bulk.py                    # 전일(T-1) 수집
    python scripts/collect_investor_bulk.py --date 20260417    # 특정 날짜
    python scripts/collect_investor_bulk.py --backfill 60      # 최근 60거래일 백필
    python scripts/collect_investor_bulk.py --backfill 260     # 1년 백필
    python scripts/collect_investor_bulk.py --core-only        # 핵심 4주체만

데이터 출처: KRX STAT API (pykrx 경유, KRX_ID/KRX_PW 로그인 필요)
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "investor_flow"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "investor_daily.db"

# ─── pykrx import ───
try:
    from pykrx import stock as krx
    PYKRX_OK = True
    # pykrx 내부 로깅 버그 억제 (logging.info(args, kwargs) TypeError)
    logging.getLogger("pykrx").setLevel(logging.WARNING)
except ImportError:
    PYKRX_OK = False
    logger.error("pykrx 미설치")

# ─── 투자자 유형 ───
CORE_INVESTORS = ["외국인", "기관합계", "기타법인", "개인", "연기금", "금융투자"]
DETAIL_INVESTORS = ["투신", "보험", "은행", "사모", "기타금융"]
ALL_INVESTORS = CORE_INVESTORS + DETAIL_INVESTORS

MARKETS = ["KOSPI", "KOSDAQ"]

# ─── SQLite 스키마 ───
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS investor_daily (
    date         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    name         TEXT NOT NULL,
    investor     TEXT NOT NULL,
    sell_vol     INTEGER NOT NULL DEFAULT 0,
    buy_vol      INTEGER NOT NULL DEFAULT 0,
    net_vol      INTEGER NOT NULL DEFAULT 0,
    sell_val     INTEGER NOT NULL DEFAULT 0,
    buy_val      INTEGER NOT NULL DEFAULT 0,
    net_val      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, ticker, investor)
);

CREATE INDEX IF NOT EXISTS idx_inv_ticker_date
    ON investor_daily(ticker, date);

CREATE INDEX IF NOT EXISTS idx_inv_date
    ON investor_daily(date);

CREATE TABLE IF NOT EXISTS collect_log (
    date         TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    investors    TEXT NOT NULL,
    total_rows   INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'OK',
    elapsed_sec  REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (date)
);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def is_collected(conn: sqlite3.Connection, date: str) -> bool:
    row = conn.execute(
        "SELECT status FROM collect_log WHERE date = ?", (date,)
    ).fetchone()
    return row is not None and row[0] == "OK"


def fetch_one_investor(date: str, market: str, investor: str,
                       max_retries: int = 3) -> pd.DataFrame:
    """단일 투자자 유형의 전종목 순매수 데이터 수집 (재시도 포함)."""
    for attempt in range(max_retries):
        try:
            df = krx.get_market_net_purchases_of_equities_by_ticker(
                date, date, market, investor
            )
            if df is None or df.empty:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s → 2s → 4s
                    logger.info("  %s %s %s 빈 응답 → %d초 후 재시도 (%d/%d)",
                                date, market, investor, wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    continue
                return pd.DataFrame()

            df = df.reset_index()
            # pykrx 반환 컬럼 수 검증 (8개: ticker, name, sell_vol, buy_vol, net_vol, sell_val, buy_val, net_val)
            if len(df.columns) != 8:
                logger.debug("  %s %s %s 컬럼 수 불일치 (%d)", date, market, investor, len(df.columns))
                return pd.DataFrame()
            df.columns = ["ticker", "name", "sell_vol", "buy_vol", "net_vol",
                           "sell_val", "buy_val", "net_val"]
            df["investor"] = investor
            df["date"] = date
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("  %s %s %s 실패 → %d초 후 재시도 (%d/%d): %s",
                               date, market, investor, wait, attempt + 1, max_retries, e)
                time.sleep(wait)
            else:
                logger.warning("  %s %s %s 최종 실패 (%d회 시도): %s",
                               date, market, investor, max_retries, e)
    return pd.DataFrame()


def collect_date(conn: sqlite3.Connection, date: str, investors: list[str],
                  update_log: bool = True) -> dict:
    """특정 날짜의 전종목 투자자별 수급 수집."""
    t0 = time.time()
    all_rows = []

    for market in MARKETS:
        for investor in investors:
            df = fetch_one_investor(date, market, investor)
            if len(df) > 0:
                all_rows.append(df)
            time.sleep(0.5)

    if not all_rows:
        return {"date": date, "rows": 0, "status": "EMPTY", "elapsed": time.time() - t0}

    combined = pd.concat(all_rows, ignore_index=True)

    # SQLite INSERT OR REPLACE
    rows_inserted = 0
    for _, row in combined.iterrows():
        try:
            conn.execute(
                """INSERT OR REPLACE INTO investor_daily
                   (date, ticker, name, investor, sell_vol, buy_vol, net_vol,
                    sell_val, buy_val, net_val)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["date"], row["ticker"], row["name"], row["investor"],
                 int(float(row.get("sell_vol") or 0)), int(float(row.get("buy_vol") or 0)),
                 int(float(row.get("net_vol") or 0)),
                 int(float(row.get("sell_val") or 0)), int(float(row.get("buy_val") or 0)),
                 int(float(row.get("net_val") or 0))),
            )
            rows_inserted += 1
        except Exception as e:
            logger.debug("INSERT 실패 %s/%s: %s", row["ticker"], row["investor"], e)

    elapsed = time.time() - t0

    # 수집 로그 (partial_mode에서는 기존 로그 덮어쓰기 방지)
    if update_log:
        conn.execute(
            """INSERT OR REPLACE INTO collect_log
               (date, collected_at, investors, total_rows, status, elapsed_sec)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (date, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             ",".join(investors), rows_inserted, "OK", round(elapsed, 1)),
        )
    conn.commit()

    return {"date": date, "rows": rows_inserted, "status": "OK", "elapsed": elapsed}


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """거래일 목록 생성 (주말 제외)."""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def main():
    parser = argparse.ArgumentParser(description="전종목 투자자별 순매수 일괄 수집")
    parser.add_argument("--date", type=str, default=None, help="수집 날짜 (YYYYMMDD)")
    parser.add_argument("--backfill", type=int, default=0, help="최근 N거래일 백필")
    parser.add_argument("--core-only", action="store_true", help="핵심 5주체만 수집")
    parser.add_argument("--investors", type=str, default=None,
                        help="특정 투자자만 수집 (쉼표 구분, 예: 연기금,투신)")
    parser.add_argument("--force", action="store_true", help="이미 수집된 날짜도 재수집")
    args = parser.parse_args()

    if not PYKRX_OK:
        logger.error("pykrx 없음 — 종료")
        sys.exit(1)

    # --investors 지정 시 해당 유형만 수집 (is_collected 무시)
    partial_mode = False
    if args.investors:
        investors = [x.strip() for x in args.investors.split(",")]
        partial_mode = True
    elif args.core_only:
        investors = CORE_INVESTORS
    else:
        investors = ALL_INVESTORS
    conn = init_db()

    logger.info("=== 투자자별 순매수 일괄 수집 ===")
    logger.info("투자자: %s (%d유형)", ",".join(investors), len(investors))

    if args.backfill > 0:
        # 백필 모드
        today = datetime.now()
        dates = []
        d = today - timedelta(days=1)
        while len(dates) < args.backfill:
            if d.weekday() < 5:
                dates.append(d.strftime("%Y%m%d"))
            d -= timedelta(days=1)
        dates = sorted(dates)

        logger.info("백필: %d거래일 (%s ~ %s)", len(dates), dates[0], dates[-1])

        ok_count = 0
        skip_count = 0
        fail_count = 0

        for i, dt in enumerate(dates):
            if not partial_mode and not args.force and is_collected(conn, dt):
                skip_count += 1
                continue

            result = collect_date(conn, dt, investors, update_log=not partial_mode)
            status = result["status"]
            if status == "OK":
                ok_count += 1
                logger.info("  [%d/%d] %s: %d행 (%.1f초)",
                            i + 1, len(dates), dt, result["rows"], result["elapsed"])
            else:
                fail_count += 1
                logger.warning("  [%d/%d] %s: %s", i + 1, len(dates), dt, status)

        logger.info("백필 완료: OK %d / SKIP %d / FAIL %d", ok_count, skip_count, fail_count)

    else:
        # 단일 날짜 수집
        if args.date:
            target = args.date
        else:
            # 당일 수집 (BAT-D 16:30 이후 실행 → KRX 15:30 이후 데이터 제공)
            today = datetime.now()
            d = today
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            target = d.strftime("%Y%m%d")

        if not partial_mode and not args.force and is_collected(conn, target):
            logger.info("%s 이미 수집됨 (--force로 재수집 가능)", target)
            conn.close()
            return

        result = collect_date(conn, target, investors, update_log=not partial_mode)
        logger.info("수집: %s / %d행 / %s / %.1f초",
                     result["date"], result["rows"], result["status"], result["elapsed"])

    # DB 통계
    row_count = conn.execute("SELECT COUNT(*) FROM investor_daily").fetchone()[0]
    date_count = conn.execute("SELECT COUNT(DISTINCT date) FROM investor_daily").fetchone()[0]
    logger.info("DB 현황: %d행 / %d거래일", row_count, date_count)

    conn.close()


if __name__ == "__main__":
    main()
