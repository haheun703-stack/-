#!/usr/bin/env python
"""자산군별 ETF 수익률 계산기 — 연간/월간 성과 TOP 정렬

자산배분 제안(채권/금/달러/현금) 하위에 표시할 ETF 수익률 데이터 생성.
각 자산군별 대표 ETF의 1년/1개월 수익률을 계산하여 JSON 저장.

Usage:
    python -u -X utf8 scripts/calc_asset_etf_performance.py
    python -u -X utf8 scripts/calc_asset_etf_performance.py --days 400  # 수집 기간
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = DATA_DIR / "asset_etf_performance.json"

# ═══════════════════════════════════════════════
# 자산군별 ETF 정의
# ═══════════════════════════════════════════════

ASSET_CLASS_ETFS = {
    "채권": [
        {"code": "148070", "name": "KODEX 국고채10년"},
        {"code": "305080", "name": "TIGER 미국채10년선물"},
        {"code": "272580", "name": "TIGER 단기채권액티브"},
        {"code": "385560", "name": "KBSTAR 종합채권(AA-이상)액티브"},
    ],
    "금": [
        {"code": "132030", "name": "KODEX 골드선물(H)"},
        {"code": "411060", "name": "ACE KRX금현물"},
    ],
    "달러": [
        {"code": "261240", "name": "KODEX 미국달러선물"},
        {"code": "329750", "name": "TIGER 미국달러단기채권액티브"},
    ],
    "현금성": [
        {"code": "357870", "name": "TIGER CD금리투자KIS(합성)"},
        {"code": "214980", "name": "KODEX 단기채권PLUS"},
    ],
}


# ═══════════════════════════════════════════════
# pykrx 시세 수집
# ═══════════════════════════════════════════════

def _fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame | None:
    """pykrx로 ETF OHLCV 조회 (ETF API → 주식 API 폴백)."""
    try:
        from pykrx import stock as krx
        import io
        import contextlib

        root = logging.getLogger()
        orig = root.level
        root.setLevel(logging.CRITICAL)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                df = krx.get_etf_ohlcv_by_date(start, end, code)
                if df is not None and not df.empty:
                    return df
                return krx.get_market_ohlcv_by_date(start, end, code)
        finally:
            root.setLevel(orig)
    except Exception as e:
        logger.warning("  %s 시세 조회 실패: %s", code, e)
        return None


def _calc_return(close: pd.Series, days: int) -> float | None:
    """종가 시리즈에서 N거래일 수익률(%) 계산."""
    if len(close) < days + 1:
        return None
    cur = float(close.iloc[-1])
    prev = float(close.iloc[-(days + 1)])
    if prev == 0:
        return None
    return round((cur / prev - 1) * 100, 2)


# ═══════════════════════════════════════════════
# 메인 로직
# ═══════════════════════════════════════════════

def calc_performance(days: int = 400) -> dict:
    """전체 자산군 ETF 수익률 계산."""
    from pykrx import stock as krx

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    logger.info("자산 ETF 수익률 계산: %s ~ %s", start_date, end_date)

    all_etfs = []

    for asset_class, etfs in ASSET_CLASS_ETFS.items():
        logger.info("  [%s] %d종목", asset_class, len(etfs))

        for etf in etfs:
            code = etf["code"]
            name = etf["name"]

            df = _fetch_ohlcv(code, start_date, end_date)
            time.sleep(0.3)

            if df is None or df.empty:
                logger.warning("    %s (%s): 데이터 없음", name, code)
                continue

            # 종가 컬럼 찾기 (pykrx ETF vs 주식 API 컬럼명 차이)
            close_col = None
            for col in ["종가", "close", "Close"]:
                if col in df.columns:
                    close_col = col
                    break

            if close_col is None:
                logger.warning("    %s: 종가 컬럼 없음 (columns=%s)", name, list(df.columns))
                continue

            close = df[close_col].dropna()
            if len(close) < 5:
                logger.warning("    %s: 데이터 부족 (%d일)", name, len(close))
                continue

            cur_price = int(close.iloc[-1])
            ret_1m = _calc_return(close, 20)    # 1개월 ≈ 20거래일
            ret_1y = _calc_return(close, 250)   # 1년 ≈ 250거래일

            entry = {
                "asset_class": asset_class,
                "code": code,
                "name": name,
                "close": cur_price,
                "return_1m": ret_1m,
                "return_1y": ret_1y,
                "data_days": len(close),
            }
            all_etfs.append(entry)
            logger.info("    %s: 현재가 %s, 1M %s%%, 1Y %s%%",
                        name, f"{cur_price:,}", ret_1m, ret_1y)

    # 연간/월간 각각 수익률 높은 순 정렬
    yearly = sorted(
        [e for e in all_etfs if e["return_1y"] is not None],
        key=lambda x: x["return_1y"],
        reverse=True,
    )
    monthly = sorted(
        [e for e in all_etfs if e["return_1m"] is not None],
        key=lambda x: x["return_1m"],
        reverse=True,
    )

    result = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "yearly": yearly,
        "monthly": monthly,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="자산군별 ETF 수익률 계산")
    parser.add_argument("--days", type=int, default=400, help="수집 기간 (일)")
    args = parser.parse_args()

    print(f"\n[자산 ETF] 수익률 계산 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    result = calc_performance(args.days)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[자산 ETF] 완료 — 연간 {len(result['yearly'])}종목, 월간 {len(result['monthly'])}종목")
    print(f"  저장: {OUT_PATH}")

    # 상위 3종목 요약
    if result["yearly"]:
        print("\n  [연간 TOP 3]")
        for e in result["yearly"][:3]:
            print(f"    {e['asset_class']} | {e['name']}: {e['return_1y']:+.1f}%")
    if result["monthly"]:
        print("  [월간 TOP 3]")
        for e in result["monthly"][:3]:
            print(f"    {e['asset_class']} | {e['name']}: {e['return_1m']:+.1f}%")


if __name__ == "__main__":
    main()
