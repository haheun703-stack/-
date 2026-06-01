"""대형주 모멘텀 변별력 — out-of-sample + 순위상관 IC (단타봇 정공법 6/1).

단타봇 판정 수용: N=4 이진/+81%p 폐기. 다중검정 피해 변별력만 본다.
  universe: 거래대금 상위 200 (대형주+주요 소부장, asof 2025-05-30)
  IC(t) = 그 시점 모멘텀(20일) rank vs forward(10일 수익) rank 스피어만 상관 (전 종목, 정보량↑)
  in-sample(2025.6~11) vs out-of-sample(2025.12~2026.5) 분리:
    in에서 IC 양수 → out에서도 유지되면 = 진짜 변별력. 무너지면 = in-sample 과적합.
비겹침(10일 간격) 샘플. look-ahead 0. ★1일=표본1 함정 회피, 누적 시계열로.
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

LOOKBACK = 20
FWD = 10
MIN_TV = 1e9


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    store = {}
    for pat in ["data/processed/*.parquet", "data/delisted/*.parquet"]:
        for f in glob.glob(str(PROJECT_ROOT / pat)):
            try:
                df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
            except Exception:
                continue
            if len(df) < 60 or "trading_value" not in df.columns:
                continue
            code = Path(f).stem
            if code not in store:
                store[code] = df
    return cal, store


def universe_top(store, asof, n):
    avg = {}
    for c, df in store.items():
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avg[c] = sub["trading_value"].mean()
    return [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:n]]


def ic_series(cal, store, universe):
    C = pd.DataFrame({c: store[c]["close"] for c in universe}).reindex(cal).ffill()
    Craw = pd.DataFrame({c: store[c]["close"] for c in universe}).reindex(cal)
    lastv = {c: store[c].index.max() for c in universe}
    mom = Craw / Craw.shift(LOOKBACK) - 1
    fwd = Craw.shift(-FWD) / Craw - 1
    rows = []
    for i in range(LOOKBACK, len(cal) - FWD, FWD):  # 비겹침
        d = cal[i]
        valid = [c for c in universe if d <= lastv.get(c, d)]
        m = mom.loc[d][valid].dropna()
        f = fwd.loc[d][valid].dropna()
        common = m.index.intersection(f.index)
        if len(common) < 30:
            continue
        ic = m[common].corr(f[common], method="spearman")
        # 보조: 상위20% - 하위20% forward 평균 격차
        q = m[common].quantile([0.2, 0.8])
        hi = f[common][m[common] >= q.iloc[1]].mean()
        lo = f[common][m[common] <= q.iloc[0]].mean()
        rows.append((d, ic, hi - lo))
    return pd.DataFrame(rows, columns=["date", "ic", "hl"]).set_index("date")


def stat(s, lbl):
    ic = s["ic"].dropna()
    if len(ic) < 3:
        print(f"  {lbl}: 표본부족"); return
    t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0
    print(f"  {lbl:<22} 평균IC {ic.mean():+.3f} | IC>0 {int((ic>0).mean()*100)}% | "
          f"t-stat {t:+.2f} | 상위-하위 forward {s['hl'].mean()*100:+.1f}% | n={len(ic)}")


def main() -> int:
    cal, store = load()
    asof = pd.Timestamp("2025-05-30")
    for n in (200, 300):
        uni = universe_top(store, asof, n)
        s = ic_series(cal, store, uni)
        ins = s[(s.index >= "2025-06-01") & (s.index <= "2025-11-30")]
        oos = s[(s.index >= "2025-12-01") & (s.index <= "2026-05-29")]
        print(f"=== 대형주 상위 {n} — 모멘텀 변별력 (스피어만 IC, 비겹침) ===")
        stat(ins, "in-sample(25.6~11)")
        stat(oos, "out-of-sample(25.12~)")
        stat(s[(s.index >= "2025-06-01") & (s.index <= "2026-05-29")], "전체(25.6~26.5)")
        print()
    print("★ in/out 둘 다 IC>0 & |t|>2 & 부호 일관 = 진짜 변별력(과적합 아님).")
    print("  out에서 0근처/음수면 = in-sample 과적합. t<2면 = 표본상 미확정(노이즈와 구분 불가).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
