"""실시간 신호 SQLite 스냅샷 모듈 (5/26~5/28 3일 학습 모드).

배경 (퐝가님 5/25 결단):
  3일간 1주씩 매매 + 장중 학습 모드 가동.
  매 30분 cron마다 후보 종목 전체 신호값을 SQLite snapshot.
  5/29(목) 결단: 9개 학습 항목(S1~S6 + D1~D3) 임계값 적정성 + 자금 확대 결단.

9개 학습 항목:
  - S1 signal_strength: 체결강도 (KIS API tday_rltv, 0~300)
  - S2 volume_ratio: 5일 평균 대비 배수
  - S3 bullish_ratio: 양봉 비율 (0~1)
  - S4 foreign_inst_buy: 외인+기관 동시 매수 여부 (bool)
  - S5 time_slot: 매수 시간대 ('MORNING' / 'NOON' / 'AFTERNOON')
  - S6 c2_combo: C2 시그널 조합 dict (JSON)
  - D1 peak_drop_pct: 천장 대비 하락 %
  - D2 trailing_drop_pct: Trailing 고점 대비 꺾임 %
  - D3 supply_outflow_days: 수급 이탈 연속 일수

설계 원칙:
  - WAL 모드 (동시 cron + 분석 안전)
  - executemany 일괄 transaction
  - ADAPTIVE_DAILY_LEARNING_MODE=1 일 때만 저장 (no-op switch)
  - 멱등 init_db (재호출 안전)

사용:
  from src.use_cases.signal_snapshot import init_db, snapshot_signals, query_recent
  init_db()
  saved = snapshot_signals(candidates=[{...}, {...}])
  rows = query_recent(ticker="005930", hours=24)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "realtime_signals.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_at TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  current_price INTEGER,
  signal_strength REAL DEFAULT 0,
  volume_ratio REAL DEFAULT 0,
  bullish_ratio REAL DEFAULT 0,
  foreign_inst_buy INTEGER DEFAULT 0,
  time_slot TEXT,
  c2_combo TEXT,
  peak_drop_pct REAL DEFAULT 0,
  trailing_drop_pct REAL DEFAULT 0,
  supply_outflow_days INTEGER DEFAULT 0,
  in_queue INTEGER DEFAULT 0,
  in_holdings INTEGER DEFAULT 0,
  raw_data TEXT
);
"""

_INDEX_TICKER_TIME = (
    "CREATE INDEX IF NOT EXISTS idx_snapshot_ticker_time "
    "ON signal_snapshots(ticker, snapshot_at);"
)
_INDEX_TIME = (
    "CREATE INDEX IF NOT EXISTS idx_snapshot_time "
    "ON signal_snapshots(snapshot_at);"
)


def _get_db_path() -> Path:
    """DB 경로 조회 (테스트에서 monkeypatch 가능)."""
    return DB_PATH


def _connect() -> sqlite3.Connection:
    """SQLite 연결 + WAL 모드 설정.

    timeout=10초로 동시 cron + 분석 락 충돌 방지.
    """
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError as exc:
        logger.warning("WAL 모드 설정 실패: %s", exc)
    return conn


def _is_learning_mode() -> bool:
    """학습 모드 활성 여부 (ADAPTIVE_DAILY_LEARNING_MODE=1)."""
    return os.getenv("ADAPTIVE_DAILY_LEARNING_MODE", "0") == "1"


def init_db() -> None:
    """DB + 테이블 + 인덱스 초기화 (멱등).

    재호출 안전. 이미 존재하면 무시.
    """
    with _connect() as conn:
        conn.execute(_SCHEMA_SQL)
        conn.execute(_INDEX_TICKER_TIME)
        conn.execute(_INDEX_TIME)
        conn.commit()
    logger.info("signal_snapshots DB 초기화 완료: %s", _get_db_path())


def _normalize_candidate(c: dict, snapshot_at: str) -> tuple:
    """후보 dict → INSERT tuple 변환.

    누락 필드는 기본값(0 / None / '') 적용.
    c2_combo는 dict → JSON 직렬화.
    """
    c2_raw = c.get("c2_combo")
    if isinstance(c2_raw, (dict, list)):
        c2_json: Optional[str] = json.dumps(c2_raw, ensure_ascii=False)
    elif isinstance(c2_raw, str):
        c2_json = c2_raw
    else:
        c2_json = None

    raw_raw = c.get("raw_data")
    if isinstance(raw_raw, (dict, list)):
        raw_json: Optional[str] = json.dumps(raw_raw, ensure_ascii=False)
    elif isinstance(raw_raw, str):
        raw_json = raw_raw
    else:
        raw_json = None

    return (
        snapshot_at,
        str(c.get("ticker", "")),
        c.get("name"),
        int(c["current_price"]) if c.get("current_price") is not None else None,
        float(c.get("signal_strength", 0) or 0),
        float(c.get("volume_ratio", 0) or 0),
        float(c.get("bullish_ratio", 0) or 0),
        1 if c.get("foreign_inst_buy") else 0,
        c.get("time_slot"),
        c2_json,
        float(c.get("peak_drop_pct", 0) or 0),
        float(c.get("trailing_drop_pct", 0) or 0),
        int(c.get("supply_outflow_days", 0) or 0),
        1 if c.get("in_queue") else 0,
        1 if c.get("in_holdings") else 0,
        raw_json,
    )


def snapshot_signals(
    candidates: list[dict],
    snapshot_at: Optional[str] = None,
) -> int:
    """N개 종목 신호값 일괄 저장.

    Args:
        candidates: [{ticker, name, current_price, signal_strength, ...}, ...]
        snapshot_at: ISO 포맷 시각 (기본 now)

    Returns:
        저장된 row 수. 학습 모드 OFF 시 0.
    """
    if not _is_learning_mode():
        logger.debug("ADAPTIVE_DAILY_LEARNING_MODE != 1 → snapshot no-op")
        return 0

    if not candidates:
        return 0

    if snapshot_at is None:
        snapshot_at = datetime.now().isoformat(timespec="seconds")

    rows = [_normalize_candidate(c, snapshot_at) for c in candidates]

    insert_sql = """
    INSERT INTO signal_snapshots (
        snapshot_at, ticker, name, current_price,
        signal_strength, volume_ratio, bullish_ratio, foreign_inst_buy,
        time_slot, c2_combo,
        peak_drop_pct, trailing_drop_pct, supply_outflow_days,
        in_queue, in_holdings, raw_data
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with _connect() as conn:
        conn.executemany(insert_sql, rows)
        conn.commit()

    logger.info("snapshot 저장 완료: %d건 @ %s", len(rows), snapshot_at)
    return len(rows)


def query_recent(
    ticker: Optional[str] = None,
    hours: int = 24,
    limit: int = 1000,
) -> list[dict]:
    """최근 N시간 snapshot 조회.

    Args:
        ticker: 특정 종목 필터 (None=전체)
        hours: 최근 몇 시간 (기본 24)
        limit: 최대 row 수 (기본 1000)

    Returns:
        snapshot dict 리스트 (snapshot_at desc 정렬).
    """
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")

    if ticker:
        sql = """
        SELECT * FROM signal_snapshots
        WHERE snapshot_at >= ? AND ticker = ?
        ORDER BY snapshot_at DESC
        LIMIT ?
        """
        params: tuple = (cutoff, ticker, limit)
    else:
        sql = """
        SELECT * FROM signal_snapshots
        WHERE snapshot_at >= ?
        ORDER BY snapshot_at DESC
        LIMIT ?
        """
        params = (cutoff, limit)

    with _connect() as conn:
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

    return rows


def summarize_by_ticker(hours: int = 24) -> dict:
    """종목별 누적 snapshot 통계.

    Args:
        hours: 최근 몇 시간

    Returns:
        {ticker: {count, avg_signal_strength, latest_at, latest_price}}
    """
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")

    sql = """
    SELECT
        ticker,
        COUNT(*) AS cnt,
        AVG(signal_strength) AS avg_sig,
        MAX(snapshot_at) AS latest_at
    FROM signal_snapshots
    WHERE snapshot_at >= ?
    GROUP BY ticker
    """

    summary: dict[str, dict[str, Any]] = {}
    with _connect() as conn:
        cur = conn.execute(sql, (cutoff,))
        agg_rows = cur.fetchall()

        for row in agg_rows:
            ticker = row["ticker"]
            latest_at = row["latest_at"]
            # 최신 가격 별도 조회
            price_cur = conn.execute(
                "SELECT current_price FROM signal_snapshots "
                "WHERE ticker = ? AND snapshot_at = ? LIMIT 1",
                (ticker, latest_at),
            )
            price_row = price_cur.fetchone()
            latest_price = price_row["current_price"] if price_row else None

            summary[ticker] = {
                "count": int(row["cnt"]),
                "avg_signal_strength": float(row["avg_sig"] or 0),
                "latest_at": latest_at,
                "latest_price": latest_price,
            }

    return summary
