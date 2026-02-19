"""섹터 순환매 엔진 — Phase 1-4: 외인/기관 수급 데이터.

섹터 ETF 구성종목 상위 5개의 투자자별 순매수를 합산하여
섹터 수준의 스마트머니 흐름을 파악한다.

지표:
  - foreign_cum: 외국인 N일 누적 순매수(억)
  - inst_cum: 기관 N일 누적 순매수(억)
  - stealth_buying: 주가 하락 + 외국인 순매수 = 스텔스 매집

사용법:
  python scripts/sector_investor_flow.py           # 5일 누적
  python scripts/sector_investor_flow.py --days 10  # 10일 누적
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx
except ImportError:
    logger.error("pykrx 미설치: pip install pykrx")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = DATA_DIR / "etf_daily"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_sector_map() -> dict:
    path = DATA_DIR / "sector_map.json"
    if not path.exists():
        logger.error("sector_map.json 없음")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_etf_universe() -> dict:
    path = DATA_DIR / "etf_universe.json"
    if not path.exists():
        logger.error("etf_universe.json 없음")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 종목별 수급 조회
# ─────────────────────────────────────────────

def fetch_stock_trading(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """종목의 투자자별 순매수 거래대금 조회.

    Returns: DataFrame (기관합계, 외국인합계 등)
    """
    try:
        df = krx.get_market_trading_value_by_date(start, end, ticker)
        time.sleep(0.15)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.debug("%s 수급 조회 실패: %s", ticker, e)
        return None


# ─────────────────────────────────────────────
# 섹터 수급 분석
# ─────────────────────────────────────────────

def analyze_sector_flow(
    sector_map: dict,
    universe: dict,
    cum_days: int = 5,
    top_stocks: int = 5,
) -> list[dict]:
    """섹터별 상위 종목의 수급을 합산하여 섹터 수준 흐름 분석."""

    end_date = krx.get_nearest_business_day_in_a_week(
        datetime.now().strftime("%Y%m%d"), prev=True
    )
    start_date = (
        datetime.strptime(end_date, "%Y%m%d") - timedelta(days=cum_days + 20)
    ).strftime("%Y%m%d")

    logger.info("수급 조회: %s ~ %s (%d일 누적)", start_date, end_date, cum_days)

    results = []
    total_api_calls = 0

    for sector_name, info in sector_map.items():
        etf_code = info["etf_code"]
        stocks = info["stocks"]

        # market 카테고리 스킵
        category = universe.get(sector_name, {}).get("category", "")
        if category == "market":
            continue

        # 비중 상위 N종목만
        top = stocks[:top_stocks]
        if len(top) < 2:
            continue

        sector_foreign_cum = 0
        sector_inst_cum = 0
        sector_foreign_today = 0
        sector_inst_today = 0
        queried = 0

        for s in top:
            ticker = s["code"]
            df = fetch_stock_trading(ticker, start_date, end_date)
            total_api_calls += 1
            if df is None:
                continue

            # 최근 N일만
            tail = df.tail(cum_days)

            if "외국인합계" in df.columns:
                sector_foreign_cum += float(tail["외국인합계"].sum())
                sector_foreign_today += float(df.iloc[-1]["외국인합계"])
            if "기관합계" in df.columns:
                sector_inst_cum += float(tail["기관합계"].sum())
                sector_inst_today += float(df.iloc[-1]["기관합계"])

            queried += 1

        if queried == 0:
            continue

        # ETF 시세로 스텔스 매집 판단
        price_change_5 = np.nan
        etf_daily_path = DAILY_DIR / f"{etf_code}.parquet"
        if etf_daily_path.exists():
            price_df = pd.read_parquet(etf_daily_path)
            price_df.index = pd.to_datetime(price_df.index)
            if len(price_df) >= 6:
                price_change_5 = float(
                    (price_df["close"].iloc[-1] / price_df["close"].iloc[-6] - 1) * 100
                )

        stealth = False
        if not np.isnan(price_change_5) and price_change_5 < 0 and sector_foreign_cum > 0:
            stealth = True

        results.append({
            "sector": sector_name,
            "etf_code": etf_code,
            "category": category,
            "date": end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:],
            "foreign_today_bil": round(sector_foreign_today / 1e8, 1),
            "inst_today_bil": round(sector_inst_today / 1e8, 1),
            "foreign_cum_bil": round(sector_foreign_cum / 1e8, 1),
            "inst_cum_bil": round(sector_inst_cum / 1e8, 1),
            "cum_days": cum_days,
            "top_stocks_count": queried,
            "stealth_buying": stealth,
            "price_change_5": round(price_change_5, 2) if not np.isnan(price_change_5) else None,
        })

        logger.info("  %s: 외인 %+.0f억(%d일), 기관 %+.0f억",
                     sector_name, sector_foreign_cum / 1e8, cum_days, sector_inst_cum / 1e8)

    logger.info("총 API 호출: %d회", total_api_calls)

    # 외인 누적 내림차순 정렬
    results.sort(key=lambda x: x["foreign_cum_bil"], reverse=True)
    return results


# ─────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────

def print_flow_report(results: list[dict], cum_days: int):
    if not results:
        print("수급 데이터 없음")
        return

    date_str = results[0]["date"]

    print(f"\n{'=' * 80}")
    print(f"  섹터 수급 분석 — {date_str} (상위종목 합산, {cum_days}일 누적)")
    print(f"{'=' * 80}")
    print(f"  {'섹터':<10} {'외인당일':>9} {'외인누적':>9} {'기관당일':>9} {'기관누적':>9} {'5일%':>7} {'신호'}")
    print(f"  {'─' * 72}")

    for r in results:
        pc = r.get("price_change_5")
        pc_str = f"{pc:>+7.2f}" if pc is not None else "   N/A"

        flags = []
        if r["stealth_buying"]:
            flags.append("스텔스")
        if r["foreign_cum_bil"] > 0 and r["inst_cum_bil"] > 0:
            flags.append("외+기")
        elif r["foreign_cum_bil"] > 0:
            flags.append("외인↑")
        elif r["inst_cum_bil"] > 0:
            flags.append("기관↑")
        flag_str = " ".join(flags)

        print(
            f"  {r['sector']:<10} {r['foreign_today_bil']:>+9.0f} {r['foreign_cum_bil']:>+9.0f} "
            f"{r['inst_today_bil']:>+9.0f} {r['inst_cum_bil']:>+9.0f} {pc_str}  {flag_str}"
        )

    # 스텔스 매집
    stealth = [r for r in results if r["stealth_buying"]]
    if stealth:
        print(f"\n  ★ 스텔스 매집 ({len(stealth)}개 섹터):")
        for r in stealth:
            print(f"    {r['sector']}: {r['price_change_5']:+.2f}% 하락 + 외인 {r['foreign_cum_bil']:+.0f}억")

    # 스마트머니
    smart = [r for r in results if r["foreign_cum_bil"] > 0 and r["inst_cum_bil"] > 0]
    if smart:
        print(f"\n  ◆ 스마트머니 (외인+기관 동시 매수, {len(smart)}개 섹터):")
        for r in smart:
            print(f"    {r['sector']}: 외인 {r['foreign_cum_bil']:+.0f}억 + 기관 {r['inst_cum_bil']:+.0f}억")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="섹터 수급 분석")
    parser.add_argument("--days", type=int, default=5,
                        help="누적 기간 (일, 기본 5)")
    parser.add_argument("--top-stocks", type=int, default=5,
                        help="섹터당 상위 종목 수 (기본 5)")
    args = parser.parse_args()

    sector_map = load_sector_map()
    universe = load_etf_universe()
    logger.info("섹터: %d개", len(sector_map))

    results = analyze_sector_flow(
        sector_map, universe,
        cum_days=args.days,
        top_stocks=args.top_stocks,
    )
    print_flow_report(results, args.days)

    # JSON 저장
    out_path = DATA_DIR / "investor_flow.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cum_days": args.days, "sectors": results},
                  f, ensure_ascii=False, indent=2)
    logger.info("수급 결과 → %s", out_path)


if __name__ == "__main__":
    main()
