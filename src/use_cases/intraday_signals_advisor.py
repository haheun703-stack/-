"""막내 정보봇 intraday_signals SELECT 통합 (2026-05-18 5번 작업)

배경: 5/18 14:52 막내 약속
- 5/19 (화) intraday_signal_consolidator.py 신설
- 4 소스 통합 (daily_news + trump_rss + truth_social + price_monitor)
- 매 15분 09:00~15:30 자동 적재
- impact_score 0~100, sentiment POSI/NEGA/NEUT
- related_tickers JSONB

형의 역할 (5번 작업):
- 막내 intraday_signals 테이블 SELECT
- impact_score ≥ 70 + 미만료 시그널 추출
- NEGA 관련 종목 → advisory severity=WARN 격상
- 사장님 advisory에 자동 노출 + 종목 차단

가동 일정:
- 5/18 (지금): 함수 골격 + 미래 호환 (테이블 미존재 시 빈 결과)
- 5/19 09:00: 막내 첫 자동 가동 → 형 SELECT 즉시 작동
- 5/20 08:55: 형 매수 결정 마지막 게이트
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_intraday_signals(
    min_impact: int = 70,
    sentiment_filter: Optional[list[str]] = None,
    today_only: bool = True,
) -> list[dict]:
    """막내 intraday_signals 테이블 SELECT.

    Args:
        min_impact: 최소 impact_score (기본 70)
        sentiment_filter: ['NEGA','POSI','NEUT'] 필터 (None이면 전체)
        today_only: 오늘 데이터만 (False면 어제까지 포함)

    Returns:
        [{signal_at, sentiment, impact_score, title, summary,
          related_tickers, related_themes, expire_at}, ...]
    """
    try:
        import psycopg2
    except ImportError:
        return []

    url = os.environ.get("DATABASE_URL")
    if not url:
        return []

    try:
        con = psycopg2.connect(url, connect_timeout=10)
        cur = con.cursor()

        # 테이블 존재 확인
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='intraday_signals')"
        )
        exists = cur.fetchone()[0]
        if not exists:
            con.close()
            logger.info("intraday_signals 테이블 미생성 (막내 5/19 작업 예정)")
            return []

        # SELECT
        params = [min_impact]
        sql = (
            "SELECT signal_at, sentiment, impact_score, title, "
            "       COALESCE(summary, ''), "
            "       COALESCE(related_tickers, '[]'::jsonb), "
            "       COALESCE(related_themes, '[]'::jsonb), "
            "       expire_at "
            "FROM intraday_signals "
            "WHERE impact_score >= %s "
        )
        if today_only:
            sql += "AND signal_at::date = CURRENT_DATE "
        sql += "AND (expire_at IS NULL OR expire_at > NOW()) "
        if sentiment_filter:
            placeholders = ", ".join(["%s"] * len(sentiment_filter))
            sql += f"AND sentiment IN ({placeholders}) "
            params.extend(sentiment_filter)
        sql += "ORDER BY signal_at DESC LIMIT 20"

        cur.execute(sql, params)
        rows = cur.fetchall()
        con.close()

        signals = []
        for r in rows:
            signals.append({
                "signal_at": r[0].isoformat() if r[0] else None,
                "sentiment": r[1],
                "impact_score": int(r[2]),
                "title": r[3],
                "summary": r[4],
                "related_tickers": r[5] if isinstance(r[5], list) else [],
                "related_themes": r[6] if isinstance(r[6], list) else [],
                "expire_at": r[7].isoformat() if r[7] else None,
            })
        return signals
    except Exception as e:
        logger.warning("intraday_signals SELECT 실패: %s", e)
        return []


def get_nega_blocked_tickers(signals: list[dict]) -> set[str]:
    """NEGA 시그널 관련 종목 추출 → 형 매수 차단 리스트.

    Args:
        signals: fetch_intraday_signals() 결과

    Returns:
        차단 종목 set
    """
    blocked = set()
    for sig in signals:
        if sig["sentiment"] == "NEGA" and sig["impact_score"] >= 70:
            for tk in sig.get("related_tickers", []):
                blocked.add(tk)
    return blocked


def format_for_advisory(signals: list[dict]) -> dict:
    """advisory reasoning(JSONB) 변환."""
    blocked = get_nega_blocked_tickers(signals)
    return {
        "intraday_signals_n": len(signals),
        "intraday_signals_nega": sum(1 for s in signals if s["sentiment"] == "NEGA"),
        "intraday_signals_posi": sum(1 for s in signals if s["sentiment"] == "POSI"),
        "blocked_tickers": list(blocked),
        "top_signals": [
            {
                "sentiment": s["sentiment"],
                "impact": s["impact_score"],
                "title": s["title"][:80],
                "tickers": s.get("related_tickers", [])[:5],
            }
            for s in signals[:5]
        ],
    }


def format_for_telegram(signals: list[dict]) -> str:
    """텔레그램 1~2줄."""
    if not signals:
        return ""

    n_nega = sum(1 for s in signals if s["sentiment"] == "NEGA")
    n_posi = sum(1 for s in signals if s["sentiment"] == "POSI")
    blocked = get_nega_blocked_tickers(signals)

    lines = []
    if signals:
        top = signals[0]
        emoji = {"NEGA": "🔴", "POSI": "🟢", "NEUT": "⚪"}.get(top["sentiment"], "⚪")
        lines.append(f"🤖 막내 {emoji} {top['sentiment']} {top['impact_score']}: {top['title'][:50]}")
    if blocked:
        lines.append(f"🚫 NEGA 차단 종목: {len(blocked)}건 ({', '.join(list(blocked)[:5])})")

    return "\n".join(lines)
