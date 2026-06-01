"""①결정적 테스트 — 모멘텀 집중 vs 랜덤/대형주묻어두기 (단타봇 작업지시, 6/1).

핵심 질문: 모멘텀 top5(+409%)가 "모멘텀 실력"인가 "그냥 대형주 집중 베타"인가?
  모멘텀 top5 월리밸  vs  랜덤5 월리밸(시드평균)  vs  거래대금상위5/10 고정 묻어두기
모멘텀 ≈ 랜덤 → 모멘텀 허상 / 거래대금상위 고정 > 모멘텀 → 그냥 큰거 묻어두기가 답.
★2025.6~2026.5. look-ahead 0. 거래일 기준. (위험조정 샤프 동시 측정)
"""
from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def main() -> int:
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = sorted(k["date"].tolist())
    asof = pd.Timestamp("2025-05-30")
    avg, store = {}, {}
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if "trading_value" not in df.columns or len(df) < 120:
            continue
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avg[Path(f).stem] = sub["trading_value"].mean(); store[Path(f).stem] = df["close"]
    ranked = [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)]
    uni = ranked[:150]
    C = pd.DataFrame({c: store[c] for c in uni}).reindex(cal).ffill()
    mom = C / C.shift(20) - 1
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    days = [d for d in cal if S <= d <= E]
    reb = set(days[::20])

    def perf(eq_series):
        eq = pd.Series(eq_series, index=days)
        eq = eq / eq.iloc[0]
        ret = (eq.iloc[-1] - 1) * 100
        dr = eq.pct_change().dropna()
        sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        return ret, mdd, sharpe

    def run(selector):
        cash = 1e8; pos = {}; curve = []
        for d in days:
            val = cash + sum(sh * C.at[d, c] for c, sh in pos.items() if not pd.isna(C.at[d, c]))
            curve.append(val)
            if d in reb:
                tgt = selector(d)
                tgt = [c for c in tgt if not pd.isna(C.at[d, c]) and C.at[d, c] > 0]
                if tgt:
                    pos = {}; per = val / len(tgt)
                    for c in tgt:
                        pos[c] = per / C.at[d, c]
                    cash = val - sum(sh * C.at[d, c] for c, sh in pos.items())
        return perf(curve)

    def mom_sel(K):
        return lambda d: list(mom.loc[d].dropna().nlargest(K).index)

    def rand_sel(K, seed):
        rng = random.Random(seed)
        def f(d):
            avail = [c for c in uni if not pd.isna(C.at[d, c]) and not pd.isna(mom.at[d, c])]
            return rng.sample(avail, min(K, len(avail)))
        return f

    def fixed_sel(codes):
        return lambda d: codes

    print("①결정적 — 모멘텀 집중 vs 랜덤 vs 대형주묻어두기 (2025.6~2026.5)\n")
    print(f'{"전략":<26}{"수익":>9}{"MDD":>8}{"샤프":>7}')
    for K in (5, 10):
        r, m, s = run(mom_sel(K))
        print(f'{"모멘텀 top"+str(K):<26}{r:>+8.0f}%{m:>7.0f}%{s:>7.2f}')
    for K in (5, 10):
        rs = [run(rand_sel(K, sd)) for sd in range(5)]
        r = np.mean([x[0] for x in rs]); m = np.mean([x[1] for x in rs]); s = np.mean([x[2] for x in rs])
        print(f'{"랜덤 "+str(K)+"개(시드5평균)":<26}{r:>+8.0f}%{m:>7.0f}%{s:>7.2f}')
    for K in (5, 10):
        r, m, s = run(fixed_sel(ranked[:K]))
        print(f'{"거래대금상위"+str(K)+" 고정보유":<26}{r:>+8.0f}%{m:>7.0f}%{s:>7.2f}')
    # 삼전+하이닉스
    r, m, s = run(fixed_sel(["005930", "000660"]))
    print(f'{"삼전+하이닉스 고정":<26}{r:>+8.0f}%{m:>7.0f}%{s:>7.2f}')
    ks = k.set_index("date")["close"]; ks = ks[(ks.index >= S) & (ks.index <= E)] / ks[(ks.index >= S) & (ks.index <= E)].iloc[0]
    kdr = ks.pct_change().dropna()
    print(f'{"KOSPI":<26}{(ks.iloc[-1]-1)*100:>+8.0f}%{((ks-ks.cummax())/ks.cummax()).min()*100:>7.0f}%{kdr.mean()/kdr.std()*np.sqrt(252):>7.2f}')
    print("\n★ 모멘텀 ≈ 랜덤 → 모멘텀 허상(답=대형주 집중). 거래대금상위 고정 ≥ 모멘텀 → 그냥 큰거 묻어두기.")
    print("  모멘텀 > 랜덤·고정 둘다 → 모멘텀 선택이 진짜 기여(샤프로도 확인).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
