"""국적별 스마트 머니 백테스트

가설: 기관/외국인의 특정 수급 패턴이 발생하면 주가가 따라감.
      개인 투매 + 스마트머니 매수 = 최강 시그널?

패턴:
  1. DUAL_BUY: 기관+외인 동시 매수 (3일 연속)
  2. RETAIL_PANIC: 개인 극단 매도 + 기관/외인 매수
  3. INST_ACCEL: 기관 매수 가속 (3일 연속 증가)
  4. FOREIGN_REVERSAL: 외인 5일 매도 → 매수 전환
  5. SMART_ABSORB: 개인 투매 흡수 (개인 대량 매도, 기관+외인 매수)

Usage:
    python scripts/backtest_smart_money.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def detect_patterns(df: pd.DataFrame, i: int) -> list[str]:
    """i번째 날짜에서 수급 패턴 탐지. 여러 패턴 동시 발화 가능."""

    patterns = []

    inst = df["기관합계"].values
    frgn = df["외국인합계"].values
    indv = df["개인"].values
    close = df["close"].values
    volume = df["volume"].values

    if i < 10:
        return patterns

    # 기본 NaN 체크
    vals = [inst[i], frgn[i], indv[i]]
    if any(np.isnan(v) for v in vals):
        return patterns

    # ── 패턴 1: DUAL_BUY_3d (기관+외인 3일 연속 동시 매수) ──
    dual_3d = True
    for d in range(3):
        idx = i - d
        if idx < 0:
            dual_3d = False
            break
        i_val, f_val = inst[idx], frgn[idx]
        if np.isnan(i_val) or np.isnan(f_val) or i_val <= 0 or f_val <= 0:
            dual_3d = False
            break
    if dual_3d:
        patterns.append("DUAL_BUY_3d")

    # ── 패턴 2: RETAIL_PANIC (개인 최근 20일 중 하위 5% 매도 + 스마트머니 매수) ──
    indv_20 = indv[max(0, i-19):i+1]
    indv_20_clean = indv_20[~np.isnan(indv_20)]
    if len(indv_20_clean) >= 15:
        pct5 = np.percentile(indv_20_clean, 5)
        if indv[i] <= pct5 and indv[i] < 0:
            # 기관 or 외인 매수
            if inst[i] > 0 or frgn[i] > 0:
                patterns.append("RETAIL_PANIC")
            # 둘 다 매수면 더 강한 시그널
            if inst[i] > 0 and frgn[i] > 0:
                patterns.append("RETAIL_PANIC_DUAL")

    # ── 패턴 3: INST_ACCEL (기관 3일 연속 매수 + 매수량 증가) ──
    if i >= 2:
        i0, i1, i2 = inst[i-2], inst[i-1], inst[i]
        if (not any(np.isnan(v) for v in [i0, i1, i2])
            and i0 > 0 and i1 > 0 and i2 > 0
            and i2 > i1 > i0):
            patterns.append("INST_ACCEL")

    # ── 패턴 4: FOREIGN_REVERSAL (외인 5일 매도 → 매수 전환) ──
    if i >= 5:
        prev5 = frgn[i-5:i]
        prev5_clean = prev5[~np.isnan(prev5)]
        if (len(prev5_clean) >= 4
            and sum(1 for x in prev5_clean if x < 0) >= 4
            and not np.isnan(frgn[i]) and frgn[i] > 0):
            patterns.append("FOREIGN_REVERSAL")

    # ── 패턴 5: SMART_ABSORB (개인 대량 매도 + 기관+외인 흡수) ──
    if indv[i] < 0 and inst[i] > 0 and frgn[i] > 0:
        # 개인 매도량이 최근 10일 평균의 2배 이상
        indv_10 = indv[max(0, i-9):i+1]
        indv_10_clean = indv_10[~np.isnan(indv_10)]
        if len(indv_10_clean) >= 5:
            avg_sell = np.mean([x for x in indv_10_clean if x < 0]) if any(x < 0 for x in indv_10_clean) else 0
            if avg_sell < 0 and indv[i] < avg_sell * 2:  # 2배 이상 매도
                patterns.append("SMART_ABSORB")

    # ── 패턴 6: VOLUME_SUPPLY (거래량 급증 + 수급 일치) ──
    vol_10 = volume[max(0, i-9):i]
    vol_10_clean = vol_10[~np.isnan(vol_10)]
    if len(vol_10_clean) >= 5 and not np.isnan(volume[i]):
        avg_vol = np.mean(vol_10_clean)
        if avg_vol > 0 and volume[i] > avg_vol * 2:
            if inst[i] > 0 and frgn[i] > 0:
                patterns.append("VOL_SURGE_DUAL")
            elif inst[i] > 0 or frgn[i] > 0:
                patterns.append("VOL_SURGE_SMART")

    # ── 패턴 7: 역발상 — 개인 몰빵 매수 (위험 신호) ──
    indv_90 = np.percentile(indv_20_clean, 90) if len(indv_20_clean) >= 15 else None
    if indv_90 is not None and indv[i] >= indv_90 and indv[i] > 0:
        if inst[i] < 0 and frgn[i] < 0:
            patterns.append("RETAIL_FOMO")  # 개인 매수 + 스마트머니 매도 = 위험

    # ── 패턴 8: PULLBACK_SUPPLY — 주가 하락인데 스마트머니 매수 ──
    if i >= 3:
        price_chg_3d = (close[i] / close[i-3] - 1) * 100 if close[i-3] > 0 else 0
        if price_chg_3d < -3:  # 3일간 3% 하락
            if inst[i] > 0 and frgn[i] > 0:
                patterns.append("PULLBACK_DUAL")
            elif inst[i] > 0 or frgn[i] > 0:
                patterns.append("PULLBACK_SMART")

    return patterns


def backtest_smart_money():
    """전종목 스마트 머니 패턴 백테스트."""

    parquets = sorted(RAW_DIR.glob("*.parquet"))
    print(f"전체 {len(parquets)}종목 스캔")

    all_trades: dict[str, list] = {}

    for pq_idx, pq in enumerate(parquets):
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if len(df) < 60:
                continue

            required = ["close", "volume", "기관합계", "외국인합계", "개인"]
            if not all(c in df.columns for c in required):
                continue

            # 슬라이딩 스캔
            for i in range(20, len(df) - 6):
                patterns = detect_patterns(df, i)
                if not patterns:
                    continue

                close_i = df.iloc[i]["close"]
                if close_i <= 0 or np.isnan(close_i):
                    continue

                d1 = df.iloc[i + 1]["close"] / close_i - 1
                d3 = df.iloc[i + 3]["close"] / close_i - 1
                d5 = df.iloc[i + 5]["close"] / close_i - 1

                for pat in patterns:
                    if pat not in all_trades:
                        all_trades[pat] = []
                    all_trades[pat].append({
                        "ticker": ticker,
                        "date": str(df.index[i].date()),
                        "d1": round(d1 * 100, 2),
                        "d3": round(d3 * 100, 2),
                        "d5": round(d5 * 100, 2),
                    })

        except Exception:
            continue

        if (pq_idx + 1) % 200 == 0:
            print(f"  ... {pq_idx + 1}/{len(parquets)} 완료")

    # ── 결과 분석 ──
    if not all_trades:
        print("결과 없음!")
        return

    print(f"\n{'='*75}")
    print(f"  국적별 스마트 머니 백테스트 결과")
    print(f"{'='*75}")

    results = []

    for pat in sorted(all_trades.keys()):
        trades = all_trades[pat]
        if len(trades) < 30:
            continue

        df = pd.DataFrame(trades)

        d5_avg = df["d5"].mean()
        d5_wr = (df["d5"] > 0).mean() * 100
        wins = df["d5"][df["d5"] > 0]
        losses = df["d5"][df["d5"] <= 0]
        d5_pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

        d1_avg = df["d1"].mean()
        d3_avg = df["d3"].mean()

        # 알파 등급 판정
        if d5_avg >= 2.0 and d5_wr >= 55 and d5_pf >= 1.8:
            grade = "STRONG_ALPHA"
        elif d5_avg >= 1.0 and d5_pf >= 1.3:
            grade = "WEAK_ALPHA"
        elif d5_avg >= 0.5:
            grade = "MARGINAL"
        else:
            grade = "NO_ALPHA"

        results.append({
            "pattern": pat,
            "n": len(trades),
            "d1_avg": d1_avg,
            "d3_avg": d3_avg,
            "d5_avg": d5_avg,
            "d5_wr": d5_wr,
            "d5_pf": d5_pf,
            "grade": grade,
        })

    # 등급별 정렬
    grade_order = {"STRONG_ALPHA": 0, "WEAK_ALPHA": 1, "MARGINAL": 2, "NO_ALPHA": 3}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["d5_avg"]))

    for r in results:
        print(f"\n  [{r['grade']}] {r['pattern']} (n={r['n']:,})")
        print(f"    D+1: avg={r['d1_avg']:+.2f}%")
        print(f"    D+3: avg={r['d3_avg']:+.2f}%")
        print(f"    D+5: avg={r['d5_avg']:+.2f}%, WR={r['d5_wr']:.1f}%, PF={r['d5_pf']:.2f}")

    # 조합 패턴 (여러 시그널 동시 발화)
    print(f"\n{'='*75}")
    print(f"  시그널 요약 (D+5 기준, 알파 등급)")
    print(f"{'='*75}")
    print(f"  {'패턴':<22s} {'n':>7s} {'D+5 avg':>8s} {'WR':>6s} {'PF':>6s}  등급")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*6} {'-'*6}  {'-'*12}")
    for r in results:
        print(f"  {r['pattern']:<22s} {r['n']:>7,d} {r['d5_avg']:>+7.2f}% {r['d5_wr']:>5.1f}% {r['d5_pf']:>5.2f}  {r['grade']}")


if __name__ == "__main__":
    backtest_smart_money()
