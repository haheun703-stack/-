"""모멘텀 집중 vs 삼전+하이닉스 묻어두기 (사장님 6/1 "대장 2개면 되지").

14번 검증 다 분산(10종목/21섹터)→폭등대장 평준화. 사장님식 집중(최강 1~2종목) 미검증.
대형주 universe 월리밸 top-K(1/2/3/5/10) 모멘텀 집중 vs 삼전+하이닉스 buy&hold.
집중이 하이닉스(+1039%) 같은 최강대장 잡아 묻어두기 따라가나? look-ahead 0.
★2025.6~2026.5(로컬 5/29까지). 거래일 기준 정확 시뮬.
"""
from __future__ import annotations

import glob
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
    uni = [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:150]]
    C = pd.DataFrame({c: store[c] for c in uni}).reindex(cal).ffill()
    mom = C / C.shift(20) - 1
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    days = [d for d in cal if S <= d <= E]
    reb = set(days[::20])  # 월

    def run(K):
        cash = 1e8; pos = {}; peak = 1e8; mdd = 0.0
        for d in days:
            val = cash + sum(sh * C.at[d, c] for c, sh in pos.items() if not pd.isna(C.at[d, c]))
            peak = max(peak, val); mdd = max(mdd, (peak - val) / peak)
            if d in reb:
                m = mom.loc[d].dropna()
                tgt = list(m.nlargest(K).index)
                if tgt:
                    pos = {}; per = val / len(tgt)
                    for c in tgt:
                        px = C.at[d, c]
                        if not pd.isna(px) and px > 0:
                            pos[c] = per / px
                    cash = val - sum(sh * C.at[d, c] for c, sh in pos.items())
        last = days[-1]
        final = cash + sum(sh * C.at[last, c] for c, sh in pos.items() if not pd.isna(C.at[last, c]))
        return (final / 1e8 - 1) * 100, mdd * 100

    # 삼전+하이닉스 buy&hold (동일 기간)
    def bh(codes):
        eqs = []
        for c in codes:
            s = store[c].reindex(cal).ffill()
            s = s[(s.index >= S) & (s.index <= E)]
            eqs.append(s / s.iloc[0])
        port = sum(eqs) / len(eqs)
        mdd = ((port - port.cummax()) / port.cummax()).min() * 100
        return (port.iloc[-1] - 1) * 100, mdd

    print("모멘텀 집중 vs 대장 묻어두기 — 2025.6~2026.5 (대형주150 universe)\n")
    print(f'{"전략":<22}{"수익":>9}{"MDD":>8}')
    for K in (1, 2, 3, 5, 10):
        r, m = run(K)
        print(f'{"모멘텀집중 top"+str(K)+" 월리밸":<22}{r:>+8.0f}%{m:>7.0f}%')
    print()
    r, m = bh(["005930", "000660"])
    print(f'{"삼전+하이닉스 묻어두기":<22}{r:>+8.0f}%{m:>7.0f}%')
    ks = k.set_index("date")["close"]; ks = ks[(ks.index >= S) & (ks.index <= E)]
    print(f'{"KOSPI":<22}{(ks.iloc[-1]/ks.iloc[0]-1)*100:>+8.0f}%')
    print("\n★ 집중 top1~2가 묻어두기 근처면 = 봇이 '지금 최강대장'을 골라줄 가치. 한참 낮으면 = 그냥 대장 묻어두기가 답.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
