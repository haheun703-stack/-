#!/usr/bin/env python3
"""EWY 월별 수익률만 재계산 — iShares 불필요(보유종목 리스트는 Supabase에서 재사용).

배경 (2026-06-14):
  iShares가 보유종목 CSV 엔드포인트에 투자자유형 동의(JS disclaimer) 모달을 추가 →
  `collect_ewy_holdings.py`의 직접 스크랩이 HTML만 받아 5/13 이후 silent fail
  (BAT run_bat.sh G4.6가 파싱 0종목으로 중단, 업로드 안 됨). → 대시보드가 3·4·5월에서 정지.

이 스크립트:
  - 월별 수익률(3·4·5·6월)은 iShares가 아니라 **pykrx(KRX)** 로 계산된다는 점을 이용.
  - quant_ewy_holdings 최신 행의 holdings(monthly_perf.stocks)를 그대로 쓰고,
  - calc_multi_month_returns(num_months=4)로 6월 포함 재계산,
  - 같은 보유종목 + 새 monthly_perf로 오늘 날짜 행 upsert.
  → 보유종목 비중은 마지막 확보분(May 13) 유지(대시보드 '기준 May 13'과 일치, 강제 stale 아님),
    월별 수익률 컬럼만 6월까지 갱신.

실행:
    python scripts/recompute_ewy_monthly.py            # dry (업로드 안 함)
    python scripts/recompute_ewy_monthly.py --upload   # Supabase 반영
    python scripts/recompute_ewy_monthly.py --upload --months 4 --date 2026-06-14
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from scripts.collect_ewy_holdings import upload_to_supabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _month_boundaries(date_str: str, num_months: int) -> list[dict]:
    """원본 calc_multi_month_returns와 동일한 월 경계(당월=MTD)."""
    import calendar

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    out = []
    for i in range(num_months - 1, -1, -1):
        m, y = dt.month - i, dt.year
        while m <= 0:
            m += 12
            y -= 1
        m_end = dt if i == 0 else datetime(y, m, calendar.monthrange(y, m)[1])
        out.append({
            "key": f"{y:04d}-{m:02d}",
            "label": f"{m}월" + ("(MTD)" if i == 0 else ""),
            "start": datetime(y, m, 1),
            "end": m_end,
        })
    return out


def calc_monthly_local(holdings: list[dict], date_str: str, num_months: int) -> dict:
    """★로컬 parquet(load_daily_ohlcv)로 월별 수익률 계산 — iShares·KRX live 둘 다 우회.

    원본 calc_multi_month_returns와 동일한 출력 구조 + 동일 공식((월말−월초)/월초 종가).
    pykrx 'KRX 로그인'(KRX_DATA_PW 만료로 실패) 경로를 타지 않는다.
    """
    import pandas as pd
    from src.etf.samsung_single_leverage_shadow import load_daily_ohlcv

    months_info = _month_boundaries(date_str, num_months)
    stocks_result = []
    ok = skip = 0
    for h in holdings:
        ticker = h["code"]
        entry = {
            "rank": h.get("rank", 0), "code": ticker, "name": h.get("name", ""),
            "weight": h.get("weight", 0), "weight_change": h.get("weight_change", 0),
            "close": 0, "sector": h.get("sector", ""), "returns": {},
        }
        try:
            df = load_daily_ohlcv(ticker, days=400)
            if df is None or len(df) == 0 or "close" not in df.columns:
                skip += 1
                stocks_result.append(entry)
                continue
            for m in months_info:
                mask = (df.index >= pd.Timestamp(m["start"])) & (df.index <= pd.Timestamp(m["end"]))
                m_df = df[mask]
                if m_df.empty:
                    entry["returns"][m["key"]] = None
                    continue
                open_p = float(m_df["close"].iloc[0])
                close_p = float(m_df["close"].iloc[-1])
                entry["returns"][m["key"]] = round((close_p - open_p) / open_p * 100, 2) if open_p else None
            entry["close"] = int(float(df["close"].iloc[-1]))
            ok += 1
        except Exception as e:  # noqa: BLE001
            logger.debug("[EWY-recompute] %s 로컬 조회 실패: %s", ticker, e)
            skip += 1
        stocks_result.append(entry)

    stocks_result.sort(key=lambda x: x["weight"], reverse=True)

    summary = {}
    for m in months_info:
        key = m["key"]
        valid = [s for s in stocks_result if s["returns"].get(key) is not None]
        if valid:
            avg = round(sum(s["returns"][key] for s in valid) / len(valid), 2)
            total_w = sum(s["weight"] for s in valid)
            wavg = round(sum(s["weight"] * s["returns"][key] for s in valid) / total_w, 2) if total_w > 0 else 0.0
            up = len([s for s in valid if s["returns"][key] > 0])
            dn = len([s for s in valid if s["returns"][key] < 0])
            summary[key] = {"avg": avg, "wavg": wavg, "up": up, "dn": dn, "total": len(valid)}
        else:
            summary[key] = {"avg": 0.0, "wavg": 0.0, "up": 0, "dn": 0, "total": 0}

    logger.info("[EWY-recompute] 로컬 parquet 계산 완료: ok %d / skip %d", ok, skip)
    return {
        "months": [{"key": m["key"], "label": m["label"]} for m in months_info],
        "stocks": stocks_result,
        "summary": summary,
    }


def _fetch_latest_row() -> dict | None:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.error("[EWY-recompute] SUPABASE_URL/KEY 미설정")
        return None
    from supabase import create_client
    client = create_client(url, key)
    r = (
        client.table("quant_ewy_holdings")
        .select("date,as_of,total_stocks,top20,changes,new_entries,removed,summary,monthly_perf")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def _holdings_from_row(row: dict) -> list[dict]:
    """monthly_perf.stocks(80종목) → calc_multi_month_returns 입력 형식으로 변환."""
    stocks = (row.get("monthly_perf") or {}).get("stocks") or []
    holdings = []
    for s in stocks:
        code = s.get("code", "")
        if not code:
            continue
        holdings.append({
            "code": code,
            "name": s.get("name", ""),
            "weight": s.get("weight", 0),
            "rank": s.get("rank", 0),
            "sector": s.get("sector", ""),
            "weight_change": s.get("weight_change", 0),
        })
    return holdings


def main() -> int:
    parser = argparse.ArgumentParser(description="EWY 월별 수익률 재계산(iShares 불필요)")
    parser.add_argument("--upload", action="store_true", help="Supabase 반영")
    parser.add_argument("--months", type=int, default=4, help="표시 개월수 (기본 4: 3·4·5·6월)")
    parser.add_argument("--date", type=str, default=None, help="기준일 YYYY-MM-DD (월 경계 산정용, 기본 오늘)")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    row = _fetch_latest_row()
    if not row:
        logger.error("[EWY-recompute] 최신 행을 읽지 못함 — 중단")
        return 1

    holdings = _holdings_from_row(row)
    if not holdings:
        logger.error("[EWY-recompute] 보유종목 0 — monthly_perf.stocks 비어있음, 중단")
        return 1

    logger.info(
        "[EWY-recompute] 소스 행: date=%s as_of=%s 보유 %d종목 → %d개월 재계산(기준 %s)",
        row.get("date"), row.get("as_of"), len(holdings), args.months, date_str,
    )

    monthly = calc_monthly_local(holdings, date_str, num_months=args.months)
    if not monthly or not monthly.get("months"):
        logger.error("[EWY-recompute] 월별 계산 결과 비어있음 — 중단")
        return 1

    labels = [m["label"] for m in monthly["months"]]
    logger.info("[EWY-recompute] 월별 컬럼: %s", " / ".join(labels))
    for m in monthly["months"]:
        s = monthly["summary"].get(m["key"], {})
        logger.info(
            "  %10s: 상승 %2d / 하락 %2d | 단순 %+.2f%% / 비중가중 %+.2f%%",
            m["label"], s.get("up", 0), s.get("dn", 0), s.get("avg", 0), s.get("wavg", 0),
        )

    if not args.upload:
        logger.info("[EWY-recompute] dry — 업로드 안 함(--upload로 반영). 보유종목/as_of 보존, monthly_perf만 갱신.")
        return 0

    ok = upload_to_supabase(
        date_str=date_str,
        as_of=row.get("as_of", ""),          # 보유종목 날짜 보존(May 13)
        total_stocks=row.get("total_stocks", len(holdings)),
        top20=row.get("top20") or [],         # 보유 비중 보존
        changes=row.get("changes") or [],
        new_entries=row.get("new_entries") or [],
        removed=row.get("removed") or [],
        summary=row.get("summary", ""),
        monthly_summary=monthly,              # ★6월 포함 새 monthly_perf
    )
    logger.info("[EWY-recompute] 업로드 %s (date=%s)", "성공" if ok else "실패", date_str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
