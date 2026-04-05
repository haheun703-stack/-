"""거래량 진공 구간 백테스트

가설: 가격대별 거래량 프로파일에서 '진공 구간'(거래 거의 없는 가격대)을
      아래에서 위로 돌파하면, 저항 없이 빠르게 상승.

방법:
  1. 최근 60일 OHLCV로 가격대별 거래량 프로파일 구성 (20구간)
  2. 현재가 바로 위에 '진공 구간'(전체 거래량의 2% 미만)이 있는 종목 탐지
  3. 해당 종목의 D+1/D+3/D+5 수익률 측정
  4. 수급(기관/외인) 필터 추가 시 알파 변화 측정

Usage:
    python scripts/backtest_volume_vacuum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def build_volume_profile(df: pd.DataFrame, window: int = 60, n_bins: int = 20):
    """최근 window일의 가격대별 거래량 프로파일.

    Returns: (bins_edges, vol_by_bin, total_vol)
    """
    recent = df.tail(window)
    price_low = recent["low"].min()
    price_high = recent["high"].max()

    if price_high <= price_low:
        return None, None, 0

    bin_edges = np.linspace(price_low, price_high, n_bins + 1)
    vol_by_bin = np.zeros(n_bins)

    for _, row in recent.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        if vol <= 0 or np.isnan(vol):
            continue
        # 캔들이 걸치는 bin에 거래량 비례 배분
        for b in range(n_bins):
            bin_lo, bin_hi = bin_edges[b], bin_edges[b + 1]
            overlap = max(0, min(hi, bin_hi) - max(lo, bin_lo))
            candle_range = hi - lo if hi > lo else 1
            vol_by_bin[b] += vol * (overlap / candle_range)

    total = vol_by_bin.sum()
    return bin_edges, vol_by_bin, total


def find_vacuum_above(close: float, bin_edges, vol_by_bin, total_vol,
                      vacuum_pct: float = 0.02) -> dict | None:
    """현재가 바로 위에 진공 구간이 있는지 탐지.

    Returns: {vacuum_start, vacuum_end, vacuum_gap_pct, vol_ratio} or None
    """
    if total_vol <= 0:
        return None

    n_bins = len(vol_by_bin)

    # 현재가가 속한 bin 찾기
    current_bin = None
    for b in range(n_bins):
        if bin_edges[b] <= close <= bin_edges[b + 1]:
            current_bin = b
            break

    if current_bin is None or current_bin >= n_bins - 2:
        return None  # 최상단이면 위에 진공 없음

    # 현재 bin 위의 연속 진공 구간 탐지
    vacuum_start = None
    vacuum_end = None

    for b in range(current_bin + 1, n_bins):
        bin_ratio = vol_by_bin[b] / total_vol
        if bin_ratio < vacuum_pct:
            if vacuum_start is None:
                vacuum_start = b
            vacuum_end = b
        else:
            if vacuum_start is not None:
                break  # 진공 끝

    if vacuum_start is None:
        return None

    vacuum_gap_pct = (bin_edges[vacuum_end + 1] - bin_edges[vacuum_start]) / close * 100

    # 진공 너비가 1% 미만이면 무시
    if vacuum_gap_pct < 1.0:
        return None

    return {
        "vacuum_start_price": round(bin_edges[vacuum_start], 0),
        "vacuum_end_price": round(bin_edges[vacuum_end + 1], 0),
        "vacuum_gap_pct": round(vacuum_gap_pct, 2),
        "vacuum_bins": vacuum_end - vacuum_start + 1,
    }


def backtest_volume_vacuum():
    """전종목 거래량 진공 구간 백테스트."""

    parquets = sorted(RAW_DIR.glob("*.parquet"))
    print(f"전체 {len(parquets)}종목 스캔")

    all_trades = []

    for pq in parquets:
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if len(df) < 120:
                continue

            required = ["close", "open", "high", "low", "volume"]
            if not all(c in df.columns for c in required):
                continue

            has_supply = "기관합계" in df.columns and "외국인합계" in df.columns

            # 60일 윈도우로 슬라이딩
            for i in range(60, len(df) - 6):
                window = df.iloc[i - 60:i]
                close = df.iloc[i]["close"]

                if close <= 0 or np.isnan(close):
                    continue

                bin_edges, vol_by_bin, total_vol = build_volume_profile(window)
                if bin_edges is None:
                    continue

                vacuum = find_vacuum_above(close, bin_edges, vol_by_bin, total_vol)
                if vacuum is None:
                    continue

                # 수급 정보
                inst_buy = False
                foreign_buy = False
                dual_buy = False
                if has_supply:
                    inst = df.iloc[i].get("기관합계", 0)
                    frgn = df.iloc[i].get("외국인합계", 0)
                    inst_buy = inst > 0 if not np.isnan(inst) else False
                    foreign_buy = frgn > 0 if not np.isnan(frgn) else False
                    dual_buy = inst_buy and foreign_buy

                # D+1~D+5 수익률
                entry = close
                d1 = df.iloc[i + 1]["close"] / entry - 1
                d3 = df.iloc[i + 3]["close"] / entry - 1
                d5 = df.iloc[i + 5]["close"] / entry - 1

                all_trades.append({
                    "ticker": ticker,
                    "date": str(df.index[i].date()),
                    "close": close,
                    "vacuum_gap_pct": vacuum["vacuum_gap_pct"],
                    "vacuum_bins": vacuum["vacuum_bins"],
                    "inst_buy": inst_buy,
                    "foreign_buy": foreign_buy,
                    "dual_buy": dual_buy,
                    "d1": round(d1 * 100, 2),
                    "d3": round(d3 * 100, 2),
                    "d5": round(d5 * 100, 2),
                })

        except Exception:
            continue

    # ── 결과 분석 ──
    if not all_trades:
        print("결과 없음!")
        return

    df = pd.DataFrame(all_trades)
    print(f"\n{'='*70}")
    print(f"  거래량 진공 구간 백테스트 결과")
    print(f"{'='*70}")
    print(f"  전체 트레이드: {len(df):,}건")

    for label, subset in [
        ("전체", df),
        ("진공 2%+ 갭", df[df["vacuum_gap_pct"] >= 2]),
        ("진공 3%+ 갭", df[df["vacuum_gap_pct"] >= 3]),
        ("진공 5%+ 갭", df[df["vacuum_gap_pct"] >= 5]),
        ("진공 + 기관매수", df[df["inst_buy"]]),
        ("진공 + 외인매수", df[df["foreign_buy"]]),
        ("진공 + 쌍끌이", df[df["dual_buy"]]),
        ("진공 3%+ 갭 + 수급", df[(df["vacuum_gap_pct"] >= 3) & (df["inst_buy"] | df["foreign_buy"])]),
        ("진공 3%+ 갭 + 쌍끌이", df[(df["vacuum_gap_pct"] >= 3) & df["dual_buy"]]),
    ]:
        if len(subset) < 10:
            continue
        print(f"\n  --- {label} (n={len(subset):,}) ---")
        for col, day in [("d1", "D+1"), ("d3", "D+3"), ("d5", "D+5")]:
            avg = subset[col].mean()
            wr = (subset[col] > 0).mean() * 100
            wins = subset[col][subset[col] > 0]
            losses = subset[col][subset[col] <= 0]
            pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
            print(f"    {day}: avg={avg:+.2f}%, WR={wr:.1f}%, PF={pf:.2f}")

    # 진공 갭 크기별 분석
    print(f"\n  --- 진공 갭 크기별 D+5 ---")
    for lo, hi in [(1, 2), (2, 3), (3, 5), (5, 10), (10, 100)]:
        subset = df[(df["vacuum_gap_pct"] >= lo) & (df["vacuum_gap_pct"] < hi)]
        if len(subset) >= 10:
            avg = subset["d5"].mean()
            wr = (subset["d5"] > 0).mean() * 100
            print(f"    갭 {lo}~{hi}%: avg={avg:+.2f}%, WR={wr:.1f}%, n={len(subset):,}")


if __name__ == "__main__":
    backtest_volume_vacuum()
