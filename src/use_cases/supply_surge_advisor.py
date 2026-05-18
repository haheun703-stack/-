"""scan_supply_surge → advisory 통합 (2026-05-18 신규)

배경: 사장님 짚으신 "맵핑+연결"
- scan_supply_surge 매일 50건+ 적재 (외인/기관 동반 매수 종목)
- 그러나 형 advisory에 통합 X = 자비스 자체 자원 미활용

quant_supply_surge 컬럼:
- date, ticker, name, close, ret_d0
- supply_type: A_쌍끌이(30) / B_기관연기금(25) / C_3주체합류(25)
                D_외인폭발(20) / E_연기금매집(15) / F_금투기타(10)
                X_개인추격(-10, 매도시그널)
- base_score, tech_score, streak_bonus, final_score
- fgn, inst, pension, finance, corp (5 수급 주체)

advisory 통합:
- 최근 거래일 supply_surge SELECT
- final_score TOP 5 → reasoning(JSONB)
- 텔레그램에 1줄 추가
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_recent_supply_surge(date: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """최근 거래일 supply_surge 상위 N건 SELECT.

    Args:
        date: 'YYYY-MM-DD' (None이면 오늘, 데이터 없으면 -1d/-3d 폴백)
        top_n: 상위 N건

    Returns:
        [{ticker, name, supply_type, final_score, ret_d0, fgn, inst, pension}, ...]
    """
    try:
        import psycopg2
    except ImportError:
        return []

    url = os.environ.get("DATABASE_URL")
    if not url:
        return []

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        con = psycopg2.connect(url, connect_timeout=10)
        cur = con.cursor()

        # 1차: 오늘 데이터
        cur.execute(
            """
            SELECT ticker, name, supply_type, final_score, ret_d0,
                   fgn, inst, pension, finance, corp
            FROM quant_supply_surge
            WHERE date = %s AND final_score > 0
            ORDER BY final_score DESC
            LIMIT %s
            """,
            (date, top_n),
        )
        rows = cur.fetchall()

        # 폴백: 어제 데이터 (오늘 BAT-D 전이면 빈 결과)
        if not rows:
            yesterday = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            cur.execute(
                """
                SELECT ticker, name, supply_type, final_score, ret_d0,
                       fgn, inst, pension, finance, corp
                FROM quant_supply_surge
                WHERE date = %s AND final_score > 0
                ORDER BY final_score DESC
                LIMIT %s
                """,
                (yesterday, top_n),
            )
            rows = cur.fetchall()
            date_used = yesterday
        else:
            date_used = date

        # 폴백 2: 최근 7일 중 최신
        if not rows:
            cur.execute(
                """
                SELECT ticker, name, supply_type, final_score, ret_d0,
                       fgn, inst, pension, finance, corp, date
                FROM quant_supply_surge
                WHERE final_score > 0
                ORDER BY date DESC, final_score DESC
                LIMIT %s
                """,
                (top_n,),
            )
            rows = cur.fetchall()
            date_used = rows[0][10] if rows else None

        con.close()

        results = []
        for r in rows:
            results.append({
                "ticker": r[0],
                "name": r[1],
                "supply_type": r[2],
                "final_score": float(r[3] or 0),
                "ret_d0": float(r[4] or 0),
                "fgn": float(r[5] or 0),       # 외인 (억원)
                "inst": float(r[6] or 0),      # 기관
                "pension": float(r[7] or 0),   # 연기금
                "finance": float(r[8] or 0),   # 금융투자
                "corp": float(r[9] or 0),      # 기타법인
                "date_used": date_used,
            })
        return results
    except Exception as e:
        logger.warning("supply_surge SELECT 실패: %s", e)
        return []


def format_supply_surge_for_advisory(surges: list[dict]) -> dict:
    """advisory reasoning(JSONB)에 포함할 dict."""
    if not surges:
        return {"supply_surge": []}

    return {
        "supply_surge_top5": [
            {
                "ticker": s["ticker"],
                "name": s["name"],
                "type": s["supply_type"],
                "score": s["final_score"],
                "ret_d0_pct": round(s["ret_d0"], 2),
                "fgn_eok": round(s["fgn"], 1),
                "inst_eok": round(s["inst"], 1),
                "pension_eok": round(s["pension"], 1),
            }
            for s in surges
        ],
        "supply_surge_date": surges[0].get("date_used") if surges else None,
        "n_strong_supply": sum(1 for s in surges if s["final_score"] >= 50),
    }


def format_supply_surge_for_telegram(surges: list[dict]) -> str:
    """텔레그램 메시지용 1~2줄."""
    if not surges:
        return ""
    top = surges[0]
    return (
        f"💎 수급폭발 TOP: {top['name']}({top['ticker']}) "
        f"[{top['supply_type']} {top['final_score']:.0f}점]"
    )
