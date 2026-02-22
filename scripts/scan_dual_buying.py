"""외인+기관 동반매수 종목 스캔 → data/dual_buying_watch.json

stock_data_daily/ CSV에서 외인+기관 동시 순매수 종목을 찾아
S/A/B 등급으로 분류하고 핵심필터 WATCH 리스트를 생성한다.

사용법:
  python scripts/scan_dual_buying.py            # 전체 스캔
  python scripts/scan_dual_buying.py --universe  # 유니버스만
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "stock_data_daily"
OUT_PATH = PROJECT_ROOT / "data" / "dual_buying_watch.json"


def scan_csv(csv_path: Path, uni_tickers: set[str]) -> dict | None:
    """단일 CSV에서 동반매수 여부를 판정한다."""
    parts = csv_path.stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None

    name, ticker = parts[0], parts[1]

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if len(df) < 10 or "Foreign_Net" not in df.columns:
        return None

    tail5 = df.tail(5)
    f5 = tail5["Foreign_Net"].sum()
    i5 = tail5["Inst_Net"].sum()

    if f5 <= 0 or i5 <= 0:
        return None

    # 동반매수일 (5일 중 외인>0 & 기관>0인 날)
    dual_days = int(((tail5["Foreign_Net"] > 0) & (tail5["Inst_Net"] > 0)).sum())
    if dual_days < 2:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = int(last["Close"])
    chg = round((last["Close"] - prev["Close"]) / prev["Close"] * 100, 1)
    rsi = round(float(last.get("RSI", 0)), 1)

    # 5일 수익률
    if len(df) >= 6:
        p5 = df.iloc[-6]["Close"]
        ret5 = round((last["Close"] - p5) / p5 * 100, 1)
    else:
        ret5 = 0.0

    # 거래량 비율
    avg_vol = df.tail(20)["Volume"].mean()
    vol_ratio = round(last["Volume"] / avg_vol, 1) if avg_vol > 0 else 0.0

    # 외인/기관 연속 매수일
    f_streak = 0
    for j in range(1, min(21, len(df))):
        if df.iloc[-j]["Foreign_Net"] > 0:
            f_streak += 1
        else:
            break

    i_streak = 0
    for j in range(1, min(21, len(df))):
        if df.iloc[-j]["Inst_Net"] > 0:
            i_streak += 1
        else:
            break

    return {
        "ticker": ticker,
        "name": name,
        "price": price,
        "chg": chg,
        "ret5": ret5,
        "rsi": rsi,
        "foreign_5d": int(f5),
        "inst_5d": int(i5),
        "dual_days": dual_days,
        "f_streak": f_streak,
        "i_streak": i_streak,
        "vol_ratio": vol_ratio,
        "is_universe": ticker in uni_tickers,
    }


def main():
    parser = argparse.ArgumentParser(description="외인+기관 동반매수 WATCH 스캔")
    parser.add_argument("--universe", action="store_true", help="유니버스 종목만")
    args = parser.parse_args()

    # 유니버스 티커
    parquet_dir = PROJECT_ROOT / "data" / "processed"
    uni_tickers = set(pf.stem for pf in parquet_dir.glob("*.parquet"))

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]

    if args.universe:
        csv_files = [
            f for f in csv_files
            if f.stem.rsplit("_", 1)[-1] in uni_tickers
        ]

    logger.info(f"스캔 대상: {len(csv_files)}개 CSV")

    results = []
    for f in csv_files:
        r = scan_csv(f, uni_tickers)
        if r:
            results.append(r)

    # 등급 분류
    results.sort(
        key=lambda x: (x["dual_days"], x["f_streak"] + x["i_streak"], x["foreign_5d"] + x["inst_5d"]),
        reverse=True,
    )

    s_grade = [r for r in results if r["dual_days"] >= 5]
    a_grade = [r for r in results if r["dual_days"] == 4]
    b_grade = [r for r in results if r["dual_days"] == 3][:20]

    # 핵심필터: RSI 55~70 + 동반매수 3일+
    core_watch = [
        r for r in results
        if 55 <= r["rsi"] <= 70 and r["dual_days"] >= 3
    ]
    core_watch.sort(
        key=lambda x: (x["dual_days"], x["f_streak"] + x["i_streak"]),
        reverse=True,
    )
    core_watch = core_watch[:10]

    # 핵심필터 포인트 생성
    for r in core_watch:
        pts = []
        # 수급 하이라이트
        if r["i_streak"] >= 5:
            pts.append(f"기관 {abs(r['inst_5d']):,}주")
        elif r["f_streak"] >= 5:
            pts.append(f"외인 {abs(r['foreign_5d']):,}주")
        elif r["inst_5d"] > r["foreign_5d"]:
            pts.append(f"기관 {abs(r['inst_5d']):,}주")
        else:
            pts.append(f"외인 {abs(r['foreign_5d']):,}주")
        # RSI 판정
        if r["rsi"] <= 60:
            pts.append("RSI 최적")
        elif r["rsi"] <= 65:
            pts.append("RSI 적정")
        else:
            pts.append("RSI 주의")
        # 연속 매수 강조
        if r["f_streak"] >= 5:
            pts.append(f"외인 {r['f_streak']}일 연속")
        if r["i_streak"] >= 5:
            pts.append(f"기관 {r['i_streak']}일 연속")
        # 거래량 폭증
        if r["vol_ratio"] >= 2.0:
            pts.append(f"거래량 {r['vol_ratio']}x")
        r["point"] = ", ".join(pts)

    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "s_grade": s_grade,
        "a_grade": a_grade,
        "b_grade": b_grade,
        "core_watch": core_watch,
        "stats": {
            "total_scanned": len(csv_files),
            "total_found": len(results),
            "s": len(s_grade),
            "a": len(a_grade),
            "b_total": len([r for r in results if r["dual_days"] == 3]),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"동반매수 스캔 완료: {len(results)}종목 발견")
    logger.info(f"  S급(5일): {len(s_grade)}개")
    logger.info(f"  A급(4일): {len(a_grade)}개")
    logger.info(f"  B급(3일): {len(b_grade)}개 (상위 20)")
    logger.info(f"  핵심필터: {len(core_watch)}개")
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
