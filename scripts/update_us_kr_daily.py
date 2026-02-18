"""US-KR 학습 루프: 매일 KR 장마감 후 실적 수집 → DB 누적

매일 15:40 KST 실행하여:
1. 오늘의 US 마감 데이터 (전일 미국장)
2. 오늘의 KR 실적 (KOSPI, KOSDAQ, 섹터 ETF)
3. us_kr_history.db에 INSERT → 패턴매칭 정확도 지속 개선

사용법:
    python scripts/update_us_kr_daily.py              # 오늘 데이터 추가
    python scripts/update_us_kr_daily.py --days 5     # 최근 5일 보충

daily_auto_update.bat 또는 daily_scheduler.py에서 호출.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from scripts.backfill_us_kr_history import (
    DB_PATH,
    US_SYMBOLS,
    KR_SYMBOLS,
    _clamp,
    _calc_overnight_score,
)

# 최소 이 날짜 이후의 신규 데이터만 추가
MIN_LOOKBACK_DAYS = 10


def get_last_db_date() -> str | None:
    """DB의 마지막 날짜 조회."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.execute("SELECT MAX(date) FROM us_kr_history")
        row = cursor.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None
    finally:
        conn.close()


def fetch_recent_data(days: int = 10) -> tuple[dict, dict]:
    """최근 N일 US+KR 데이터를 yfinance로 수집."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=days + 5)  # 여유분

    us_data = {}
    for name, symbol in US_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df["change_pct"] = df["Close"].pct_change() * 100
                us_data[name] = df
        except Exception as e:
            logger.warning(f"  US {name}: {e}")
        time.sleep(0.15)

    kr_data = {}
    for name, symbol in KR_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df["change_pct"] = df["Close"].pct_change() * 100
                kr_data[name] = df
        except Exception as e:
            logger.warning(f"  KR {name}: {e}")
        time.sleep(0.15)

    logger.info(f"수집: US {len(us_data)}/{len(US_SYMBOLS)}, KR {len(kr_data)}/{len(KR_SYMBOLS)}")
    return us_data, kr_data


def _get_change(data_dict: dict, key: str, date) -> float | None:
    if key not in data_dict:
        return None
    df = data_dict[key]
    try:
        date_n = pd.Timestamp(date).normalize()
        mask = df.index.normalize() == date_n
        if mask.any():
            val = df.loc[mask, "change_pct"].iloc[0]
            return round(val, 4) if pd.notna(val) else None
    except Exception:
        pass
    return None


def _get_close(data_dict: dict, key: str, date) -> float | None:
    if key not in data_dict:
        return None
    df = data_dict[key]
    try:
        date_n = pd.Timestamp(date).normalize()
        mask = df.index.normalize() == date_n
        if mask.any():
            val = df.loc[mask, "Close"].iloc[0]
            return round(val, 2) if pd.notna(val) else None
    except Exception:
        pass
    return None


def _get_open_gap(data_dict: dict, key: str, date) -> float | None:
    if key not in data_dict:
        return None
    df = data_dict[key]
    try:
        date_n = pd.Timestamp(date).normalize()
        mask = df.index.normalize() == date_n
        if mask.any():
            idx = df.index.get_loc(df.index[mask][0])
            if idx > 0:
                prev_close = df.iloc[idx - 1]["Close"]
                today_open = df.iloc[idx]["Open"]
                if prev_close > 0:
                    return round(((today_open - prev_close) / prev_close) * 100, 4)
    except Exception:
        pass
    return None


def build_new_records(
    us_data: dict,
    kr_data: dict,
    existing_dates: set[str],
) -> list[dict]:
    """DB에 없는 새 날짜 페어의 레코드 생성."""

    kr_kospi = kr_data.get("kr_kospi")
    us_spy = us_data.get("us_sp500")
    if kr_kospi is None or us_spy is None:
        logger.error("KOSPI 또는 SPY 데이터 없음")
        return []

    kr_trading_days = set(kr_kospi.index.normalize())
    us_trading_days = sorted(us_spy.index.normalize())

    us_fields = {
        "us_sp500_chg":   "us_sp500",
        "us_nasdaq_chg":  "us_nasdaq",
        "us_dow_chg":     "us_dow",
        "us_soxx_chg":    "us_soxx",
        "us_oil_chg":     "us_oil",
        "us_dollar_chg":  "us_dollar",
        "us_bond10y_chg": "us_bond10y",
        "us_gold_chg":    "us_gold",
        "us_china_chg":   "us_china",
        "us_tsla_chg":    "us_tsla",
        "us_xbi_chg":     "us_xbi",
        "us_xle_chg":     "us_xle",
        "us_xlf_chg":     "us_xlf",
        "us_ewy_chg":     "us_ewy",
    }

    kr_fields = {
        "kr_kospi_chg":    "kr_kospi",
        "kr_kosdaq_chg":   "kr_kosdaq",
        "kr_semi_chg":     "kr_semi",
        "kr_ev_chg":       "kr_ev",
        "kr_bio_chg":      "kr_bio",
        "kr_bank_chg":     "kr_bank",
        "kr_steel_chg":    "kr_steel",
        "kr_it_chg":       "kr_it",
        "kr_oil_chg":      "kr_energy",
        "kr_domestic_chg": "kr_domestic",
    }

    records = []

    for us_date in us_trading_days:
        for offset in range(1, 6):
            candidate = (us_date + timedelta(days=offset)).normalize()
            if candidate in kr_trading_days:
                kr_date_str = candidate.strftime("%Y-%m-%d")

                # 이미 DB에 있으면 스킵
                if kr_date_str in existing_dates:
                    break

                record = {
                    "date": kr_date_str,
                    "us_date": us_date.strftime("%Y-%m-%d"),
                    "gap_days": offset,
                }

                for field, data_key in us_fields.items():
                    record[field] = _get_change(us_data, data_key, us_date)

                record["us_vix_chg"] = _get_change(us_data, "us_vix", us_date)
                record["us_vix_level"] = _get_close(us_data, "us_vix", us_date)

                for field, data_key in kr_fields.items():
                    record[field] = _get_change(kr_data, data_key, candidate)

                record["kr_kospi_open_gap"] = _get_open_gap(kr_data, "kr_kospi", candidate)
                record["us_overnight_score"] = _calc_overnight_score(record)

                if record["us_sp500_chg"] is not None and record["kr_kospi_chg"] is not None:
                    records.append(record)

                break

    return records


def insert_new_records(records: list[dict]) -> int:
    """새 레코드를 DB에 삽입."""
    if not records:
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    columns = list(records[0].keys())
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    inserted = 0
    for record in records:
        try:
            values = [record.get(col) for col in columns]
            conn.execute(
                f"INSERT OR REPLACE INTO us_kr_history ({columns_str}) VALUES ({placeholders})",
                values,
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"삽입 실패 ({record.get('date')}): {e}")

    conn.commit()
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="US-KR 학습 루프: 일일 DB 누적")
    parser.add_argument("--days", type=int, default=MIN_LOOKBACK_DAYS,
                        help=f"최근 N일 데이터 조회 (기본: {MIN_LOOKBACK_DAYS})")
    args = parser.parse_args()

    if not DB_PATH.exists():
        logger.error(f"DB 없음: {DB_PATH} -> 먼저 backfill_us_kr_history.py 실행")
        return

    # 기존 DB 날짜 목록
    conn = sqlite3.connect(str(DB_PATH))
    existing = set(pd.read_sql("SELECT date FROM us_kr_history", conn)["date"].tolist())
    total_before = len(existing)
    conn.close()

    last_date = get_last_db_date()
    logger.info(f"DB 현재: {total_before}건 (마지막: {last_date})")

    # 최근 데이터 수집
    logger.info(f"최근 {args.days}일 US+KR 데이터 수집...")
    us_data, kr_data = fetch_recent_data(days=args.days)

    # 신규 레코드 생성
    records = build_new_records(us_data, kr_data, existing)
    logger.info(f"신규 레코드: {len(records)}건")

    if records:
        inserted = insert_new_records(records)
        logger.info(f"DB 삽입: {inserted}건 (총 {total_before + inserted}건)")
        for r in records:
            logger.info(f"  + {r['date']} (US:{r['us_date']}) "
                         f"KOSPI:{r.get('kr_kospi_chg', '?')}% "
                         f"Score:{r.get('us_overnight_score', '?')}")
    else:
        logger.info("추가할 신규 데이터 없음 (이미 최신)")


if __name__ == "__main__":
    main()
