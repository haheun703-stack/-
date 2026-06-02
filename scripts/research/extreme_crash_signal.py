"""① 극단 폭락 신호 검증 — 봇 탈출 룰 근거 (사장님 6/2).

일상 하락 예측은 53%(동전, 오늘 V자가 증거). 단 "전쟁급 극단 폭락"은 다를 것:
  US 극단 신호(VIX 2σ급등 / SPY-2%↓ / SOXX-3%↓) → 다음 KOSPI 급락 적중률?
  + 반도체레버(488080)를 극단신호시 현금 → buy&hold(+1768%) 대비 MDD 회피?
극단 신호 후 KOSPI가 일반일보다 확실히 빠지면 = 탈출 룰 근거. 차이 없으면 = 극단도 못잡음.
★2025.6~2026.5. look-ahead 0(전일 US로 당일 KOSPI). 이 강세장 극단 표본 적음 주의.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pykrx import stock

SWITCH = 0.001


def main() -> int:
    us = pd.read_parquet(PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet").sort_index()
    us["vix_z"] = us.get("vix_zscore", (us["vix_close"] - us["vix_close"].rolling(20).mean()) / us["vix_close"].rolling(20).std())
    sig = pd.DataFrame(index=us.index)
    sig["vix_spike"] = us["vix_z"] >= 2.0
    sig["spy_crash"] = us["spy_ret_1d"] <= -0.02
    sig["soxx_crash"] = us["soxx_ret_1d"] <= -0.03
    sig["extreme"] = sig["vix_spike"] | sig["spy_crash"] | sig["soxx_crash"]

    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    k["ret"] = k["close"].pct_change() * 100
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    kdays = [d for d in k.index if S <= d <= E]

    # KOSPI일 직전 US 신호 매핑
    sdf = sig.reset_index().rename(columns={"index": "usdate", "Date": "usdate"})
    sdf.columns = ["usdate"] + list(sig.columns)
    m = pd.merge_asof(pd.DataFrame({"kdate": kdays}), sdf, left_on="kdate", right_on="usdate", direction="backward")
    m = m.set_index("kdate")
    kret = k.loc[kdays, "ret"]

    print("① 극단 폭락 신호 → KOSPI 반응 (2025.6~2026.5)\n")
    print(f'{"신호":<14}{"신호일수":>7}{"당일KOSPI평균":>12}{"하락비율":>8}{"일반일평균":>10}')
    base = kret.mean(); base_dn = (kret < 0).mean() * 100
    for col, nm in [("extreme", "극단(통합)"), ("vix_spike", "VIX2σ급등"), ("spy_crash", "SPY-2%↓"), ("soxx_crash", "SOXX-3%↓")]:
        days_sig = m[m[col] == True].index
        days_sig = [d for d in days_sig if d in kret.index]
        if not days_sig:
            print(f'{nm:<14}{"0":>7}'); continue
        sigret = kret.loc[days_sig]
        print(f'{nm:<14}{len(days_sig):>7}{sigret.mean():>+11.2f}%{(sigret<0).mean()*100:>7.0f}%{base:>+9.2f}%')
    print(f'  (전체 일반일 평균 {base:+.2f}%, 하락비율 {base_dn:.0f}%)\n')

    # 반도체레버 극단탈출 vs buy&hold
    try:
        lev = stock.get_market_ohlcv("20250601", "20260529", "488080")["종가"].astype(float)
        lev.index = pd.to_datetime(lev.index)
        lev_r = lev.pct_change().fillna(0)
        for mode in ["buyhold", "extreme_exit"]:
            v = 1.0; prev = True; curve = []
            for d in kdays:
                ext = bool(m.loc[d, "extreme"]) if d in m.index and not pd.isna(m.loc[d, "extreme"]) else False
                inpos = (not ext) if mode == "extreme_exit" else True
                r = lev_r.get(d, 0) if inpos else 0
                v *= (1 + r)
                if inpos != prev:
                    v *= (1 - SWITCH)
                curve.append(v); prev = inpos
            eq = pd.Series(curve, index=kdays)
            ret = (eq.iloc[-1] - 1) * 100; mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
            dr = eq.pct_change().dropna(); sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
            print(f'  반도체레버 {mode:<13} 수익 {ret:>+7.0f}% / MDD {mdd:>5.0f}% / 샤프 {sh:.2f}')
    except Exception as e:
        print("  레버 ETF ERR", e)
    print("\n★ 극단신호 후 KOSPI가 일반일보다 확실히 하락(하락비율 70%+) = 탈출룰 근거. 비슷하면 = 극단도 못잡음.")
    print("  반도체레버 extreme_exit가 buy&hold보다 MDD 크게 낮추면서 수익 유지 = 봇 탈출 가치.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
