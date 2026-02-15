"""
Phase 1: SQLite 장중 데이터 저장소

IntradayStorePort 구현체. 7개 테이블에 장중 수집 데이터를 저장한다.
DB 파일: data/intraday.db

테이블:
  - tick_data: 1분 현재가/체결
  - candle_5min: 5분봉 OHLCV
  - investor_flow: 10분 투자자별 매매동향
  - market_context: 5분 시장 지수/환경
  - sector_price: 10분 업종별 시세
  - news_events: 뉴스/공시 (Phase 5)
  - ai_judgments: AI 판단 (Phase 3)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.use_cases.ports import IntradayStorePort

logger = logging.getLogger(__name__)

DB_PATH = Path("data/intraday.db")


class SqliteIntradayStore(IntradayStorePort):
    """SQLite 기반 장중 데이터 저장소"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        """테이블 생성 (없으면)"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    current_price INTEGER DEFAULT 0,
                    open_price INTEGER DEFAULT 0,
                    high_price INTEGER DEFAULT 0,
                    low_price INTEGER DEFAULT 0,
                    volume INTEGER DEFAULT 0,
                    cum_volume INTEGER DEFAULT 0,
                    change_pct REAL DEFAULT 0.0,
                    bid_price INTEGER DEFAULT 0,
                    ask_price INTEGER DEFAULT 0,
                    strength REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_tick_ticker_ts
                    ON tick_data(ticker, timestamp);

                CREATE TABLE IF NOT EXISTS candle_5min (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open INTEGER DEFAULT 0,
                    high INTEGER DEFAULT 0,
                    low INTEGER DEFAULT 0,
                    close INTEGER DEFAULT 0,
                    volume INTEGER DEFAULT 0,
                    vwap REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_candle_ticker_ts
                    ON candle_5min(ticker, timestamp);

                CREATE TABLE IF NOT EXISTS investor_flow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    foreign_net_buy INTEGER DEFAULT 0,
                    inst_net_buy INTEGER DEFAULT 0,
                    individual_net_buy INTEGER DEFAULT 0,
                    foreign_cum_net INTEGER DEFAULT 0,
                    inst_cum_net INTEGER DEFAULT 0,
                    program_net_buy INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_flow_ticker_ts
                    ON investor_flow(ticker, timestamp);

                CREATE TABLE IF NOT EXISTS market_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    kospi REAL DEFAULT 0.0,
                    kospi_change_pct REAL DEFAULT 0.0,
                    kosdaq REAL DEFAULT 0.0,
                    kosdaq_change_pct REAL DEFAULT 0.0,
                    usd_krw REAL DEFAULT 0.0,
                    us_futures REAL DEFAULT 0.0,
                    vix REAL DEFAULT 0.0,
                    bond_yield_kr_3y REAL DEFAULT 0.0,
                    kospi_volume INTEGER DEFAULT 0,
                    kosdaq_volume INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_market_ts
                    ON market_context(timestamp);

                CREATE TABLE IF NOT EXISTS sector_price (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sector_code TEXT NOT NULL,
                    sector_name TEXT DEFAULT '',
                    index_value REAL DEFAULT 0.0,
                    change_pct REAL DEFAULT 0.0,
                    volume INTEGER DEFAULT 0,
                    advance_count INTEGER DEFAULT 0,
                    decline_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_sector_ts
                    ON sector_price(timestamp, sector_code);

                CREATE TABLE IF NOT EXISTS news_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    ticker TEXT DEFAULT '',
                    source TEXT DEFAULT '',
                    title TEXT DEFAULT '',
                    content TEXT DEFAULT '',
                    sentiment TEXT DEFAULT '',
                    impact_score REAL DEFAULT 0.0,
                    category TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_news_ticker_ts
                    ON news_events(ticker, timestamp);

                CREATE TABLE IF NOT EXISTS ai_judgments (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    alert_level TEXT DEFAULT 'GREEN',
                    action TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    reasoning TEXT DEFAULT '',
                    target_price REAL DEFAULT 0.0,
                    stop_price REAL DEFAULT 0.0,
                    position_advice TEXT DEFAULT '',
                    context_summary TEXT DEFAULT '',
                    model TEXT DEFAULT '',
                    cost_usd REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_judgment_ticker_ts
                    ON ai_judgments(ticker, timestamp);
            """)
            conn.commit()
            logger.info("[SQLite] 장중 DB 초기화 완료: %s", self.db_path)
        finally:
            conn.close()

    # ──────────────────────────────────────────
    # 저장 (IntradayStorePort 구현)
    # ──────────────────────────────────────────

    def save_tick(self, tick: dict) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO tick_data
                    (ticker, timestamp, current_price, open_price, high_price,
                     low_price, volume, cum_volume, change_pct,
                     bid_price, ask_price, strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tick.get("ticker", ""),
                tick.get("timestamp", ""),
                tick.get("current_price", 0),
                tick.get("open_price", 0),
                tick.get("high_price", 0),
                tick.get("low_price", 0),
                tick.get("volume", 0),
                tick.get("cum_volume", 0),
                tick.get("change_pct", 0.0),
                tick.get("bid_price", 0),
                tick.get("ask_price", 0),
                tick.get("strength", 0.0),
            ))
            conn.commit()
        finally:
            conn.close()

    def save_ticks_batch(self, ticks: list[dict]) -> None:
        """배치 저장 (다수 종목 동시 저장)"""
        if not ticks:
            return
        conn = self._get_conn()
        try:
            conn.executemany("""
                INSERT INTO tick_data
                    (ticker, timestamp, current_price, open_price, high_price,
                     low_price, volume, cum_volume, change_pct,
                     bid_price, ask_price, strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    t.get("ticker", ""), t.get("timestamp", ""),
                    t.get("current_price", 0), t.get("open_price", 0),
                    t.get("high_price", 0), t.get("low_price", 0),
                    t.get("volume", 0), t.get("cum_volume", 0),
                    t.get("change_pct", 0.0), t.get("bid_price", 0),
                    t.get("ask_price", 0), t.get("strength", 0.0),
                )
                for t in ticks
            ])
            conn.commit()
            logger.debug("[SQLite] %d건 틱 배치 저장", len(ticks))
        finally:
            conn.close()

    def save_candle(self, candle: dict) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO candle_5min
                    (ticker, timestamp, open, high, low, close, volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle.get("ticker", ""),
                candle.get("timestamp", ""),
                candle.get("open", 0),
                candle.get("high", 0),
                candle.get("low", 0),
                candle.get("close", 0),
                candle.get("volume", 0),
                candle.get("vwap", 0.0),
            ))
            conn.commit()
        finally:
            conn.close()

    def save_investor_flow(self, flow: dict) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO investor_flow
                    (ticker, timestamp, foreign_net_buy, inst_net_buy,
                     individual_net_buy, foreign_cum_net, inst_cum_net,
                     program_net_buy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flow.get("ticker", ""),
                flow.get("timestamp", ""),
                flow.get("foreign_net_buy", 0),
                flow.get("inst_net_buy", 0),
                flow.get("individual_net_buy", 0),
                flow.get("foreign_cum_net", 0),
                flow.get("inst_cum_net", 0),
                flow.get("program_net_buy", 0),
            ))
            conn.commit()
        finally:
            conn.close()

    def save_market_context(self, ctx: dict) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO market_context
                    (timestamp, kospi, kospi_change_pct, kosdaq, kosdaq_change_pct,
                     usd_krw, us_futures, vix, bond_yield_kr_3y,
                     kospi_volume, kosdaq_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ctx.get("timestamp", ""),
                ctx.get("kospi", 0.0),
                ctx.get("kospi_change_pct", 0.0),
                ctx.get("kosdaq", 0.0),
                ctx.get("kosdaq_change_pct", 0.0),
                ctx.get("usd_krw", 0.0),
                ctx.get("us_futures", 0.0),
                ctx.get("vix", 0.0),
                ctx.get("bond_yield_kr_3y", 0.0),
                ctx.get("kospi_volume", 0),
                ctx.get("kosdaq_volume", 0),
            ))
            conn.commit()
        finally:
            conn.close()

    def save_sector_price(self, sector: dict) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO sector_price
                    (timestamp, sector_code, sector_name, index_value,
                     change_pct, volume, advance_count, decline_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sector.get("timestamp", ""),
                sector.get("sector_code", ""),
                sector.get("sector_name", ""),
                sector.get("index_value", 0.0),
                sector.get("change_pct", 0.0),
                sector.get("volume", 0),
                sector.get("advance_count", 0),
                sector.get("decline_count", 0),
            ))
            conn.commit()
        finally:
            conn.close()

    # ──────────────────────────────────────────
    # 조회 (IntradayStorePort 구현)
    # ──────────────────────────────────────────

    def get_recent_ticks(self, ticker: str, minutes: int = 60) -> list[dict]:
        since = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM tick_data
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (ticker, since)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_today_candles(self, ticker: str) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM candle_5min
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (ticker, today)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_latest_market_context(self) -> dict | None:
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT * FROM market_context
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_today_investor_flow(self, ticker: str) -> list[dict]:
        """오늘 투자자 수급 전체 조회"""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM investor_flow
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (ticker, today)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_today_sector_prices(self) -> list[dict]:
        """오늘 업종 시세 최신 스냅샷"""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM sector_price
                WHERE timestamp = (
                    SELECT MAX(timestamp) FROM sector_price
                )
                ORDER BY sector_code
            """).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_tick_count_today(self, ticker: str) -> int:
        """오늘 수집된 틱 수 (모니터링용)"""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM tick_data
                WHERE ticker = ? AND timestamp >= ?
            """, (ticker, today)).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def cleanup_old_data(self, days: int = 30) -> int:
        """N일 이전 데이터 삭제"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        conn = self._get_conn()
        total = 0
        try:
            for table in ["tick_data", "candle_5min", "investor_flow",
                          "market_context", "sector_price"]:
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,)
                )
                total += cursor.rowcount
            conn.commit()
            if total > 0:
                conn.execute("VACUUM")
                logger.info("[SQLite] %d건 오래된 데이터 정리 (%d일 이전)", total, days)
        finally:
            conn.close()
        return total

    def get_db_stats(self) -> dict:
        """DB 통계 (테이블별 행 수)"""
        conn = self._get_conn()
        stats = {}
        try:
            for table in ["tick_data", "candle_5min", "investor_flow",
                          "market_context", "sector_price", "news_events",
                          "ai_judgments"]:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                stats[table] = row["cnt"] if row else 0
        finally:
            conn.close()
        return stats
