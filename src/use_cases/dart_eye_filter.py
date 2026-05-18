"""DART EYE 필터 ⑤ — 막내 정보봇 sentiment 활용 (5번 작업, 2026-05-18)

배경: 막내 5/22 → 5/18 단축 완료
- intelligence_disclosures 테이블에 sentiment / sentiment_score 컬럼 추가
- NEGA ≤-60 = 강한 악재 (감자/횡령/소송/배당삭감 등)

형 EYE 필터 ⑤로 격상:
- 진입 시점에 종목의 최근 N일 DART 악재 체크
- NEGA ≤-60 종목 자동 회피

데이터:
- intelligence_disclosures (1336 row, 5/18 검증)
- 컬럼: ticker, sentiment, sentiment_score, severity, disclosed_at, title, ai_summary, tags
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

THRESHOLD_NEGA_SCORE = -60         # NEGA 임계 (이하 = 강한 악재)
LOOKBACK_DAYS = 2                  # 최근 N일 (DART 48h = 거래일 기준 T+1~2)


def has_dart_negative(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> tuple[bool, dict]:
    """최근 N일 DART 악재 체크.

    Args:
        ticker: 6자리 종목코드
        lookback_days: 조회 기간 (기본 2일)

    Returns:
        (회피여부, 상세 dict)
    """
    try:
        import psycopg2
    except ImportError:
        return False, {"filter": "dart_negative", "verdict": "PASS_NO_DB"}

    url = os.environ.get("DATABASE_URL")
    if not url:
        return False, {"filter": "dart_negative", "verdict": "PASS_NO_URL"}

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        con = psycopg2.connect(url, connect_timeout=10)
        cur = con.cursor()

        # 테이블 존재 + sentiment 컬럼 확인
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='intelligence_disclosures' "
            "AND column_name='sentiment_score')"
        )
        col_exists = cur.fetchone()[0]
        if not col_exists:
            con.close()
            return False, {"filter": "dart_negative", "verdict": "PASS_NO_COL"}

        cur.execute(
            """
            SELECT title, ai_summary, sentiment, sentiment_score, severity, disclosed_at, tags
            FROM intelligence_disclosures
            WHERE ticker = %s
              AND date >= %s
              AND sentiment_score IS NOT NULL
            ORDER BY disclosed_at DESC
            LIMIT 5
            """,
            (ticker, cutoff),
        )
        rows = cur.fetchall()
        con.close()

        if not rows:
            return False, {
                "filter": "dart_negative",
                "ticker": ticker,
                "verdict": "PASS_NO_DATA",
                "lookback_days": lookback_days,
            }

        # 가장 악한 sentiment_score
        worst = min(rows, key=lambda r: r[3] if r[3] is not None else 0)
        worst_score = worst[3]

        is_dangerous = worst_score is not None and worst_score <= THRESHOLD_NEGA_SCORE

        return is_dangerous, {
            "filter": "dart_negative",
            "ticker": ticker,
            "n_disclosures": len(rows),
            "worst_score": worst_score,
            "worst_title": worst[0][:60] if worst[0] else "",
            "worst_sentiment": worst[2],
            "worst_severity": worst[4],
            "verdict": "SKIP" if is_dangerous else "PASS",
            "threshold": THRESHOLD_NEGA_SCORE,
        }
    except Exception as e:
        logger.warning("dart_negative 평가 실패 %s: %s", ticker, e)
        return False, {"filter": "dart_negative", "error": str(e), "verdict": "PASS_ERROR"}


def format_for_advisory(ticker: str, result: dict) -> dict:
    """advisory reasoning(JSONB) 변환."""
    return {
        "dart_filter_ticker": ticker,
        "dart_filter_verdict": result.get("verdict"),
        "dart_worst_score": result.get("worst_score"),
        "dart_worst_title": result.get("worst_title"),
    }
