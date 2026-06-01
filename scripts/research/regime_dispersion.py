"""국면 판단기 1단계 — "선별이 통하는 장인지" dispersion이 예측하는가. 사장님 6/1.

봇이 스스로 판단하는 근거 검증:
  dispersion(t) = 그날 전체 종목 횡단면 수익률 표준편차 [t시점 관측가능, look-ahead 0]
  → 이게 높은 날, 신호(1개월 모멘텀) 상위10 - 하위10의 forward 수익 격차(spread)가 큰가?
  격차 크면 = 선별 통함(알파ON), 작으면 = 뭘골라도 비슷(인덱스).
검증: dispersion 5분위별 평균 spread. 우상향이면 판단기 성립.
★2025.6~2026.5만. 생존편향 있음(진단용 1차). spread는 미래수익이나 판단은 dispersion(과거)으로.
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

MIN_TV = 1e9
LOOKBACK = 20   # 1개월 모멘텀
FWD = 10        # forward 수익 측정 (격주)
TOPN = 10


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    C, TV = {}, {}
    for pat in ["data/processed/*.parquet", "data/delisted/*.parquet"]:
        for f in glob.glob(str(PROJECT_ROOT / pat)):
            try:
                df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
            except Exception:
                continue
            if len(df) < 60 or "trading_value" not in df.columns:
                continue
            code = Path(f).stem
            if code in C:
                continue
            C[code] = df["close"]; TV[code] = df["trading_value"]
    Cm = pd.DataFrame(C).reindex(cal)
    TVm = pd.DataFrame(TV).reindex(cal).ffill()
    return cal, k.set_index("date")["close"], Cm, TVm


def main() -> int:
    cal, kclose, Cm, TVm = load()
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    ret1 = Cm.pct_change()                       # 일별 수익률
    mom = Cm / Cm.shift(LOOKBACK) - 1            # 1개월 모멘텀 (t시점)
    fwd = Cm.shift(-FWD) / Cm - 1                # forward FWD일 수익 (t→t+FWD)

    rows = []
    idxs = [i for i, d in enumerate(cal) if S <= d <= E and i + FWD < len(cal) and i - LOOKBACK >= 0]
    for i in idxs:
        d = cal[i]
        # dispersion: 그날 횡단면 일별수익 표준편차 (유동성 종목만)
        liq = TVm.loc[d] >= MIN_TV
        r = ret1.loc[d][liq].dropna()
        if len(r) < 50:
            continue
        disp = r.std()
        # 모멘텀 상위/하위 → forward spread
        m = mom.loc[d][liq].dropna()
        fw = fwd.loc[d]
        common = m.index.intersection(fw.dropna().index)
        if len(common) < 50:
            continue
        m = m[common]; f = fw[common]
        top = m.nlargest(TOPN).index; bot = m.nsmallest(TOPN).index
        spread = f[top].mean() - f[bot].mean()
        top_ret = f[top].mean(); allmean = f.mean()
        rows.append((d, disp, spread, top_ret, allmean))

    df = pd.DataFrame(rows, columns=["date", "disp", "spread", "topret", "allmean"]).dropna()
    print(f"국면 판단기 검증 — dispersion이 '선별 통함'을 예측하는가 (2025.6~2026.5, n={len(df)}일)\n")
    df["q"] = pd.qcut(df["disp"], 5, labels=["1(격차최소)", "2", "3", "4", "5(격차최대)"])
    g = df.groupby("q", observed=True).agg(
        평균dispersion=("disp", "mean"),
        상위10_fwd수익=("topret", "mean"),
        전체평균_fwd수익=("allmean", "mean"),
        상위_하위_격차=("spread", "mean"),
        일수=("disp", "size"))
    print(f'{"dispersion분위":<14}{"평균disp":>9}{"상위10수익":>10}{"전체평균":>9}{"상위-하위":>9}{"일수":>6}')
    for q, r in g.iterrows():
        edge = r["상위10_fwd수익"] - r["전체평균_fwd수익"]
        print(f'{str(q):<14}{r["평균dispersion"]*100:>8.2f}%{r["상위10_fwd수익"]*100:>+9.1f}%'
              f'{r["전체평균_fwd수익"]*100:>+8.1f}%{r["상위_하위_격차"]*100:>+8.1f}%{int(r["일수"]):>6}')
    corr = df["disp"].corr(df["spread"])
    print(f'\n  dispersion vs 상위-하위격차 상관: {corr:+.3f}')
    print("★ 분위 올라갈수록(격차 큰 날) 상위-하위 격차/상위초과가 커지고 상관 +면 = 판단기 성립.")
    print("  (격차 큰 날 = 선별 ON, 작은 날 = 인덱스). 상관 0근처/음수면 = dispersion 무력.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
