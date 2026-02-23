"""
일일 데이터 아카이브 — SQLite 영구 저장

매일 BAT-D 실행 후 호출:
  - 당일 JSON 데이터들을 SQLite DB에 아카이브
  - 과거 날짜별 조회, 주간/월간 집계 가능

테이블:
  daily_summary   — 일별 핵심 지표 (PnL, 승률, 추천 수 등)
  daily_picks     — 일별 추천 종목 상세
  daily_etf       — 일별 ETF 시그널 상세
  weekly_reports  — 주간 집계 보고서
  monthly_reports — 월간 집계 보고서
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "jarvis_archive.db"
DATA_DIR = PROJECT_ROOT / "data"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """테이블 생성 (없으면)"""
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS daily_summary (
        date            TEXT PRIMARY KEY,
        total_picks     INTEGER DEFAULT 0,
        buy_grade       INTEGER DEFAULT 0,
        watch_grade     INTEGER DEFAULT 0,
        avg_score       REAL DEFAULT 0,
        holding_count   INTEGER DEFAULT 0,
        settled_count   INTEGER DEFAULT 0,
        win_rate        REAL DEFAULT 0,
        avg_return      REAL DEFAULT 0,
        profit_factor   REAL DEFAULT 0,
        etf_hot_count   INTEGER DEFAULT 0,
        etf_buy_count   INTEGER DEFAULT 0,
        sector_top      TEXT DEFAULT '',
        market_regime   TEXT DEFAULT '',
        raw_json        TEXT DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS daily_picks (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT NOT NULL,
        ticker          TEXT NOT NULL,
        name            TEXT NOT NULL,
        grade           TEXT DEFAULT '',
        score           REAL DEFAULT 0,
        sources         TEXT DEFAULT '',
        entry_price     INTEGER DEFAULT 0,
        stop_loss       INTEGER DEFAULT 0,
        target_price    INTEGER DEFAULT 0,
        open_price      INTEGER DEFAULT 0,
        close_price     INTEGER DEFAULT 0,
        day1_return     REAL DEFAULT 0,
        status          TEXT DEFAULT 'pending',
        reasons         TEXT DEFAULT '',
        rsi             REAL DEFAULT 0,
        stoch_k         REAL DEFAULT 0,
        UNIQUE(date, ticker)
    );

    CREATE TABLE IF NOT EXISTS daily_etf (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT NOT NULL,
        sector          TEXT NOT NULL,
        etf_code        TEXT NOT NULL,
        etf_name        TEXT NOT NULL,
        close           INTEGER DEFAULT 0,
        score           REAL DEFAULT 0,
        grade           TEXT DEFAULT '',
        ret_5           REAL DEFAULT 0,
        ret_20          REAL DEFAULT 0,
        rsi             REAL DEFAULT 0,
        foreign_5d      REAL DEFAULT 0,
        inst_5d         REAL DEFAULT 0,
        reasons         TEXT DEFAULT '',
        UNIQUE(date, etf_code)
    );

    CREATE TABLE IF NOT EXISTS weekly_reports (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        week_start      TEXT NOT NULL,
        week_end        TEXT NOT NULL,
        total_picks     INTEGER DEFAULT 0,
        settled_count   INTEGER DEFAULT 0,
        win_rate        REAL DEFAULT 0,
        avg_return      REAL DEFAULT 0,
        profit_factor   REAL DEFAULT 0,
        best_pick       TEXT DEFAULT '',
        worst_pick      TEXT DEFAULT '',
        summary_json    TEXT DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(week_start)
    );

    CREATE TABLE IF NOT EXISTS monthly_reports (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        year_month      TEXT NOT NULL,
        total_picks     INTEGER DEFAULT 0,
        settled_count   INTEGER DEFAULT 0,
        win_rate        REAL DEFAULT 0,
        avg_return      REAL DEFAULT 0,
        profit_factor   REAL DEFAULT 0,
        best_pick       TEXT DEFAULT '',
        worst_pick      TEXT DEFAULT '',
        summary_json    TEXT DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(year_month)
    );

    CREATE INDEX IF NOT EXISTS idx_picks_date ON daily_picks(date);
    CREATE INDEX IF NOT EXISTS idx_picks_ticker ON daily_picks(ticker);
    CREATE INDEX IF NOT EXISTS idx_etf_date ON daily_etf(date);
    """)
    conn.commit()
    conn.close()
    logger.info("[아카이브] DB 초기화 완료: %s", DB_PATH)


# ──────────────────────────────────────────
# 일일 아카이브
# ──────────────────────────────────────────

def archive_daily(date_str: str | None = None):
    """당일 JSON → SQLite 아카이브"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    init_db()
    conn = get_conn()

    # 이미 아카이브 됐는지 확인
    existing = conn.execute(
        "SELECT date FROM daily_summary WHERE date = ?", (date_str,)
    ).fetchone()
    if existing:
        logger.info("[아카이브] %s 이미 저장됨 → 업데이트", date_str)

    # ── 추천 종목 (tomorrow_picks.json) ──
    picks_data = _load_json("tomorrow_picks.json")
    picks = picks_data.get("picks", []) if picks_data else []
    buy_grades = [p for p in picks if p.get("grade") in ("강력매수", "매수")]
    watch_grades = [p for p in picks if p.get("grade") == "관심매수"]

    for p in picks:
        if p.get("grade") not in ("강력매수", "매수", "관심매수"):
            continue
        conn.execute("""
            INSERT OR REPLACE INTO daily_picks
            (date, ticker, name, grade, score, sources, entry_price, stop_loss,
             target_price, reasons, rsi, stoch_k)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str, p.get("ticker", ""), p.get("name", ""),
            p.get("grade", ""), p.get("total_score", 0),
            json.dumps(p.get("sources", []), ensure_ascii=False),
            p.get("entry_price", 0) or p.get("close", 0),
            p.get("stop_loss", 0), p.get("target_price", 0),
            json.dumps(p.get("reasons", []), ensure_ascii=False),
            p.get("rsi", 0), p.get("stoch_k", 0),
        ))

    # ── 성과 추적 (picks_history.json) ──
    history = _load_json("picks_history.json")
    summary = history.get("summary", {}) if history else {}
    records = history.get("records", []) if history else []

    # 성과 데이터로 picks 업데이트
    for rec in records:
        if rec.get("open_price") is not None:
            conn.execute("""
                UPDATE daily_picks
                SET open_price = ?, close_price = ?, day1_return = ?, status = ?
                WHERE date = ? AND ticker = ?
            """, (
                rec.get("open_price", 0), rec.get("close_price", 0),
                rec.get("day1_return", 0), rec.get("status", ""),
                rec.get("pick_date", ""), rec.get("ticker", ""),
            ))

    # ── ETF 마스터 (etf_master.json) ──
    etf_data = _load_json("etf_master.json")
    etfs = etf_data.get("etfs", []) if etf_data else []
    etf_hot = [e for e in etfs if e.get("grade") == "HOT매수"]
    etf_buy = [e for e in etfs if e.get("grade") == "매수"]

    for e in etfs:
        conn.execute("""
            INSERT OR REPLACE INTO daily_etf
            (date, sector, etf_code, etf_name, close, score, grade,
             ret_5, ret_20, rsi, foreign_5d, inst_5d, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str, e.get("sector", ""), e.get("etf_code", ""),
            e.get("etf_name", ""), e.get("close", 0),
            e.get("score", 0), e.get("grade", ""),
            e.get("ret_5", 0), e.get("ret_20", 0), e.get("rsi", 0),
            e.get("foreign_5d", 0), e.get("inst_5d", 0),
            json.dumps(e.get("reasons", []), ensure_ascii=False),
        ))

    # ── 섹터 ──
    sector_data = _load_json("sector_rotation/sector_momentum.json")
    sector_top = ""
    if sector_data and isinstance(sector_data, list) and len(sector_data) > 0:
        sector_top = sector_data[0].get("sector", "") if isinstance(sector_data[0], dict) else ""
    elif sector_data and isinstance(sector_data, dict):
        sectors_list = sector_data.get("sectors", [])
        if sectors_list:
            sector_top = sectors_list[0].get("sector", "") if isinstance(sectors_list[0], dict) else ""

    # ── 일일 요약 저장 ──
    avg_score = 0
    scored_picks = [p for p in picks if p.get("total_score", 0) > 0 and p.get("grade") in ("강력매수", "매수", "관심매수")]
    if scored_picks:
        avg_score = round(sum(p["total_score"] for p in scored_picks) / len(scored_picks), 1)

    raw = {
        "picks_count": len(picks),
        "etf_count": len(etfs),
        "summary": summary,
        "generated_at": picks_data.get("generated_at", "") if picks_data else "",
    }

    conn.execute("""
        INSERT OR REPLACE INTO daily_summary
        (date, total_picks, buy_grade, watch_grade, avg_score,
         holding_count, settled_count, win_rate, avg_return, profit_factor,
         etf_hot_count, etf_buy_count, sector_top, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date_str, len(buy_grades) + len(watch_grades),
        len(buy_grades), len(watch_grades), avg_score,
        summary.get("holding", 0), summary.get("total_settled", 0),
        summary.get("win_rate", 0), summary.get("avg_return", 0),
        summary.get("profit_factor", 0),
        len(etf_hot), len(etf_buy), sector_top,
        json.dumps(raw, ensure_ascii=False),
    ))

    conn.commit()
    conn.close()

    total_picks = len(buy_grades) + len(watch_grades)
    logger.info(
        "[아카이브] %s 저장 완료: 추천 %d건, ETF %d건, 섹터TOP=%s",
        date_str, total_picks, len(etfs), sector_top,
    )
    return {"date": date_str, "picks": total_picks, "etfs": len(etfs)}


# ──────────────────────────────────────────
# 주간/월간 집계
# ──────────────────────────────────────────

def generate_weekly_report(week_end: str | None = None):
    """주간 보고서 생성 (최근 5거래일 집계)"""
    init_db()
    conn = get_conn()

    if not week_end:
        week_end = datetime.now().strftime("%Y-%m-%d")

    end_dt = datetime.strptime(week_end, "%Y-%m-%d")
    week_start = (end_dt - timedelta(days=6)).strftime("%Y-%m-%d")

    # 해당 주 picks
    rows = conn.execute("""
        SELECT * FROM daily_picks
        WHERE date BETWEEN ? AND ? AND grade IN ('강력매수', '매수', '관심매수')
        ORDER BY date, score DESC
    """, (week_start, week_end)).fetchall()

    settled = [r for r in rows if r["status"] in ("hit_target", "hit_stop", "expired")]
    wins = [r for r in settled if (r["day1_return"] or 0) > 0]

    total = len(rows)
    settled_cnt = len(settled)
    win_rate = round(len(wins) / settled_cnt * 100, 1) if settled_cnt > 0 else 0
    avg_ret = round(sum(r["day1_return"] or 0 for r in settled) / settled_cnt, 2) if settled_cnt else 0

    # Best/Worst
    if rows:
        best = max(rows, key=lambda r: r["day1_return"] or -999)
        worst = min(rows, key=lambda r: r["day1_return"] or 999)
        best_str = f"{best['name']}({best['ticker']}) {best['day1_return'] or 0:+.1f}%"
        worst_str = f"{worst['name']}({worst['ticker']}) {worst['day1_return'] or 0:+.1f}%"
    else:
        best_str = worst_str = ""

    # 일별 요약
    daily_rows = conn.execute("""
        SELECT * FROM daily_summary WHERE date BETWEEN ? AND ?
        ORDER BY date
    """, (week_start, week_end)).fetchall()

    summary = {
        "daily": [dict(r) for r in daily_rows],
        "picks_by_date": {},
    }
    for r in rows:
        d = r["date"]
        if d not in summary["picks_by_date"]:
            summary["picks_by_date"][d] = []
        summary["picks_by_date"][d].append({
            "ticker": r["ticker"], "name": r["name"],
            "grade": r["grade"], "score": r["score"],
            "day1_return": r["day1_return"], "status": r["status"],
        })

    gain = sum(r["day1_return"] or 0 for r in wins)
    loss = abs(sum(r["day1_return"] or 0 for r in settled if (r["day1_return"] or 0) <= 0))
    pf = round(gain / loss, 2) if loss > 0 else 0

    conn.execute("""
        INSERT OR REPLACE INTO weekly_reports
        (week_start, week_end, total_picks, settled_count, win_rate,
         avg_return, profit_factor, best_pick, worst_pick, summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        week_start, week_end, total, settled_cnt, win_rate,
        avg_ret, pf, best_str, worst_str,
        json.dumps(summary, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()

    logger.info("[주간] %s ~ %s: %d건, 승률 %.1f%%, 평균수익 %+.2f%%",
                week_start, week_end, total, win_rate, avg_ret)
    return {
        "week_start": week_start, "week_end": week_end,
        "total_picks": total, "win_rate": win_rate, "avg_return": avg_ret,
    }


def generate_monthly_report(year_month: str | None = None):
    """월간 보고서 생성"""
    init_db()
    conn = get_conn()

    if not year_month:
        year_month = datetime.now().strftime("%Y-%m")

    rows = conn.execute("""
        SELECT * FROM daily_picks
        WHERE date LIKE ? AND grade IN ('강력매수', '매수', '관심매수')
        ORDER BY date, score DESC
    """, (f"{year_month}%",)).fetchall()

    settled = [r for r in rows if r["status"] in ("hit_target", "hit_stop", "expired")]
    wins = [r for r in settled if (r["day1_return"] or 0) > 0]

    total = len(rows)
    settled_cnt = len(settled)
    win_rate = round(len(wins) / settled_cnt * 100, 1) if settled_cnt > 0 else 0
    avg_ret = round(sum(r["day1_return"] or 0 for r in settled) / settled_cnt, 2) if settled_cnt else 0

    best_str = worst_str = ""
    if rows:
        best = max(rows, key=lambda r: r["day1_return"] or -999)
        worst = min(rows, key=lambda r: r["day1_return"] or 999)
        best_str = f"{best['name']}({best['ticker']}) {best['day1_return'] or 0:+.1f}%"
        worst_str = f"{worst['name']}({worst['ticker']}) {worst['day1_return'] or 0:+.1f}%"

    daily_rows = conn.execute("""
        SELECT * FROM daily_summary WHERE date LIKE ?
        ORDER BY date
    """, (f"{year_month}%",)).fetchall()

    summary = {"daily": [dict(r) for r in daily_rows]}

    gain = sum(r["day1_return"] or 0 for r in wins)
    loss = abs(sum(r["day1_return"] or 0 for r in settled if (r["day1_return"] or 0) <= 0))
    pf = round(gain / loss, 2) if loss > 0 else 0

    conn.execute("""
        INSERT OR REPLACE INTO monthly_reports
        (year_month, total_picks, settled_count, win_rate,
         avg_return, profit_factor, best_pick, worst_pick, summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        year_month, total, settled_cnt, win_rate,
        avg_ret, pf, best_str, worst_str,
        json.dumps(summary, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()

    logger.info("[월간] %s: %d건, 승률 %.1f%%, 평균수익 %+.2f%%",
                year_month, total, win_rate, avg_ret)
    return {
        "year_month": year_month, "total_picks": total,
        "win_rate": win_rate, "avg_return": avg_ret,
    }


# ──────────────────────────────────────────
# 조회 API
# ──────────────────────────────────────────

def get_daily_summary(date_str: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM daily_summary WHERE date = ?", (date_str,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_daily_picks(date_str: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM daily_picks WHERE date = ? ORDER BY score DESC",
        (date_str,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_daily_etfs(date_str: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM daily_etf WHERE date = ? ORDER BY score DESC",
        (date_str,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_date_list(limit: int = 90) -> list[str]:
    """아카이브된 날짜 목록 (최근순)"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT date FROM daily_summary ORDER BY date DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [r["date"] for r in rows]


def get_weekly_reports(limit: int = 12) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM weekly_reports ORDER BY week_end DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_monthly_reports(limit: int = 12) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM monthly_reports ORDER BY year_month DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_performance_chart_data(days: int = 30) -> list[dict]:
    """Chart.js용 일별 성과 데이터"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT date, total_picks, settled_count, win_rate, avg_return,
               profit_factor, etf_hot_count, etf_buy_count, sector_top
        FROM daily_summary
        ORDER BY date DESC LIMIT ?
    """, (days,)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def get_stock_history(ticker: str) -> list[dict]:
    """종목별 추천 이력"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM daily_picks WHERE ticker = ?
        ORDER BY date DESC LIMIT 100
    """, (ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────

def _load_json(filename: str) -> dict | list | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "--weekly":
        generate_weekly_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "--monthly":
        generate_monthly_report()
    else:
        archive_daily()
        print(f"\n아카이브 DB: {DB_PATH}")
        dates = get_date_list(5)
        print(f"저장된 날짜: {dates}")
