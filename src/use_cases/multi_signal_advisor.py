"""다중 시그널 advisory 통합 (6번~10번 작업, 2026-05-18)

배경: 사장님 "맵핑+연결" — 자비스 가진 모든 시스템을 advisory에 통합
- 6번: 섹터발화 (quant_sector_fire) — 14섹터 fire_score
- 7번: 테마수급/모멘텀 (quant_theme_flow, quant_theme_momentum)
- 8번: 상한가 눌림목 (quant_limit_up_pullback)
- 9번: 급락반등 (dashboard_crash_bounce)
- 10번: 피보나치 (quant_fib_scanner)

모든 함수가 동일 패턴:
  - 테이블 존재 확인 → 미존재 시 빈 결과
  - 최근 거래일 SELECT + 폴백
  - format_for_advisory() + format_for_telegram()
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _connect():
    try:
        import psycopg2
        url = os.environ.get("DATABASE_URL")
        if not url:
            return None
        return psycopg2.connect(url, connect_timeout=10)
    except Exception as e:
        logger.warning("DB 연결 실패: %s", e)
        return None


def _table_exists(con, name: str) -> bool:
    cur = con.cursor()
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=%s)",
        (name,),
    )
    return cur.fetchone()[0]


def _fallback_dates(today: str) -> list[str]:
    """오늘 → 어제 → 7일 전까지 폴백 날짜 리스트."""
    base = datetime.strptime(today, "%Y-%m-%d")
    return [(base - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(0, 8)]


# ────────────────────────────────────────────────────────────
# 6번: 섹터발화 (quant_sector_fire)
# ────────────────────────────────────────────────────────────


def fetch_sector_fire(date: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """섹터발화 TOP N (fire_score 정렬)."""
    con = _connect()
    if not con or not _table_exists(con, "quant_sector_fire"):
        if con:
            con.close()
        return []

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        cur = con.cursor()
        for d in _fallback_dates(date):
            cur.execute(
                "SELECT sector, fire_score, fire_grade, flow_score, inflection_score, "
                "       rhythm_score, energy_score, fgn_5d, inst_5d, pension_5d "
                "FROM quant_sector_fire WHERE date=%s "
                "ORDER BY fire_score DESC LIMIT %s",
                (d, top_n),
            )
            rows = cur.fetchall()
            if rows:
                con.close()
                return [
                    {
                        "sector": r[0],
                        "fire_score": float(r[1] or 0),
                        "fire_grade": r[2],
                        "flow_score": float(r[3] or 0),
                        "inflection_score": float(r[4] or 0),
                        "fgn_5d": float(r[7] or 0),
                        "inst_5d": float(r[8] or 0),
                        "pension_5d": float(r[9] or 0),
                        "date": d,
                    }
                    for r in rows
                ]
        con.close()
        return []
    except Exception as e:
        logger.warning("sector_fire SELECT 실패: %s", e)
        if con:
            con.close()
        return []


# ────────────────────────────────────────────────────────────
# 7번: 테마수급/모멘텀 (quant_theme_flow, quant_theme_momentum)
# ────────────────────────────────────────────────────────────


def fetch_theme_momentum(date: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """테마 모멘텀 HOT/COLD."""
    con = _connect()
    if not con or not _table_exists(con, "quant_theme_momentum"):
        if con:
            con.close()
        return []

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        cur = con.cursor()
        # 컬럼 확인
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='quant_theme_momentum'"
        )
        cols = {r[0] for r in cur.fetchall()}

        # 점수 컬럼 추정
        score_col = "momentum_score" if "momentum_score" in cols else (
            "score" if "score" in cols else "rank"
        )
        for d in _fallback_dates(date):
            cur.execute(
                f"SELECT theme, {score_col} FROM quant_theme_momentum "
                f"WHERE date=%s ORDER BY {score_col} DESC NULLS LAST LIMIT %s",
                (d, top_n),
            )
            rows = cur.fetchall()
            if rows:
                con.close()
                return [{"theme": r[0], "score": float(r[1] or 0), "date": d} for r in rows]
        con.close()
        return []
    except Exception as e:
        logger.warning("theme_momentum SELECT 실패: %s", e)
        if con:
            con.close()
        return []


# ────────────────────────────────────────────────────────────
# 9번: 급락반등 (dashboard_crash_bounce)
# ────────────────────────────────────────────────────────────


def fetch_crash_bounce(date: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """급락반등 후보 TOP N."""
    con = _connect()
    if not con or not _table_exists(con, "dashboard_crash_bounce"):
        if con:
            con.close()
        return []

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        cur = con.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='dashboard_crash_bounce'"
        )
        cols = {r[0] for r in cur.fetchall()}

        # 점수/등급 컬럼 추정
        order_col = "score" if "score" in cols else ("grade" if "grade" in cols else "ticker")

        for d in _fallback_dates(date):
            cur.execute(
                f"SELECT ticker, name, COALESCE({order_col}, '') as ord FROM dashboard_crash_bounce "
                f"WHERE date=%s ORDER BY ord DESC LIMIT %s",
                (d, top_n),
            )
            rows = cur.fetchall()
            if rows:
                con.close()
                return [{"ticker": r[0], "name": r[1], "score_or_grade": str(r[2]), "date": d} for r in rows]
        con.close()
        return []
    except Exception as e:
        logger.warning("crash_bounce SELECT 실패: %s", e)
        if con:
            con.close()
        return []


# ────────────────────────────────────────────────────────────
# 10번: 피보나치 (quant_fib_scanner)
# ────────────────────────────────────────────────────────────


def fetch_fibonacci(date: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """피보나치 신호 TOP N."""
    con = _connect()
    if not con or not _table_exists(con, "quant_fib_scanner"):
        if con:
            con.close()
        return []

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        cur = con.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='quant_fib_scanner'"
        )
        cols = {r[0] for r in cur.fetchall()}

        order_col = "score" if "score" in cols else ("ticker")

        for d in _fallback_dates(date):
            cur.execute(
                f"SELECT ticker, name FROM quant_fib_scanner "
                f"WHERE date=%s LIMIT %s",
                (d, top_n),
            )
            rows = cur.fetchall()
            if rows:
                con.close()
                return [{"ticker": r[0], "name": r[1], "date": d} for r in rows]
        con.close()
        return []
    except Exception as e:
        logger.warning("fib_scanner SELECT 실패: %s", e)
        if con:
            con.close()
        return []


# ────────────────────────────────────────────────────────────
# 통합 함수
# ────────────────────────────────────────────────────────────


def fetch_all_advisory_signals(date: Optional[str] = None) -> dict:
    """6번~10번 모든 시그널 한 번에 SELECT."""
    return {
        "sector_fire_top5": fetch_sector_fire(date, top_n=5),
        "theme_momentum_top5": fetch_theme_momentum(date, top_n=5),
        "crash_bounce_top5": fetch_crash_bounce(date, top_n=5),
        "fibonacci_top5": fetch_fibonacci(date, top_n=5),
    }


def format_multi_for_telegram(signals: dict) -> list[str]:
    """텔레그램 라인 리스트."""
    lines = []
    if signals.get("sector_fire_top5"):
        top = signals["sector_fire_top5"][0]
        lines.append(f"🔥 섹터발화 TOP: {top['sector']} {top['fire_grade']}({top['fire_score']:.0f}점)")
    if signals.get("theme_momentum_top5"):
        top = signals["theme_momentum_top5"][0]
        lines.append(f"📈 테마 HOT: {top['theme']} ({top['score']:.0f})")
    if signals.get("crash_bounce_top5"):
        names = [s["name"] for s in signals["crash_bounce_top5"][:3]]
        lines.append(f"🔄 급락반등: {', '.join(names)}")
    if signals.get("fibonacci_top5"):
        names = [s["name"] for s in signals["fibonacci_top5"][:3]]
        lines.append(f"📐 피보나치: {', '.join(names)}")
    return lines
