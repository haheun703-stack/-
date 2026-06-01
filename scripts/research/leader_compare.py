"""둘 다 비교 — A 중소형모멘텀 vs B 대형주추세추종 vs 삼전닉스 vs KOSPI (사장님 6/1).

선검증: 봇 20일모멘텀(u150)은 대장(삼전닉스) 안 잡음(중소형 폭등 잡음).
질문: 대형주로 한정 or 장기룩백이면 대장이 잡혀 +722% 근접하나?
  A 중소형모멘텀:  u150 중 20일mom top5 (현재 봇)
  B1 대형단기:    u30 중 20일mom top5
  B2 대형장기:    u30 중 120일mom top5 (장기추세=대장특성)
  C 대형동일가중:  거래대금 top10 고정 (대장 묻어두기 분산)
  + 삼전닉스 / KOSPI
각 전략 top5의 삼전닉스 포함율도. 월리밸. look-ahead 0. ★2025.6~2026.5.
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
        if "trading_value" not in df.columns or len(df) < 200:
            continue
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avg[Path(f).stem] = sub["trading_value"].mean(); store[Path(f).stem] = df["close"]
    ranked = [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)]
    u150, u30 = ranked[:150], ranked[:30]
    C = pd.DataFrame({c: store[c] for c in u150}).reindex(cal).ffill()
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    days = [d for d in cal if S <= d <= E]
    reb = days[::20]

    def run(selector):
        cash = 1e8; pos = {}; curve = []; snx_hits = []
        for d in days:
            val = cash + sum(sh * C.at[d, c] for c, sh in pos.items() if not pd.isna(C.at[d, c]))
            curve.append(val)
            if d in reb:
                tgt = [c for c in selector(d) if c in C.columns and not pd.isna(C.at[d, c]) and C.at[d, c] > 0]
                if tgt:
                    snx_hits.append(("005930" in tgt) + ("000660" in tgt))
                    pos = {}; per = val / len(tgt)
                    for c in tgt:
                        pos[c] = per / C.at[d, c]
                    cash = val - sum(sh * C.at[d, c] for c, sh in pos.items())
        eq = pd.Series(curve, index=days) / curve[0]
        ret = (eq.iloc[-1] - 1) * 100
        dr = eq.pct_change().dropna(); sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        snx = np.mean(snx_hits) if snx_hits else 0
        return ret, mdd, sh, snx

    def mom_sel(uni, lb, K):
        m = C[uni] / C[uni].shift(lb) - 1
        return lambda d: list(m.loc[d].dropna().nlargest(K).index)

    print("둘 다 비교 — 2025.6~2026.5 (월리밸, 대형주 universe)\n")
    print(f'{"전략":<26}{"수익":>9}{"MDD":>7}{"샤프":>6}{"삼닉포함":>8}')
    rows = [
        ("A 중소형모멘텀 u150·20일", mom_sel(u150, 20, 5)),
        ("B1 대형 u30·20일mom", mom_sel(u30, 20, 5)),
        ("B2 대형 u30·120일mom", mom_sel(u30, 120, 5)),
        ("B3 대형 u30·60일mom", mom_sel(u30, 60, 5)),
        ("C 거래대금top10 고정", lambda d: ranked[:10]),
        ("삼전+하이닉스", lambda d: ["005930", "000660"]),
    ]
    for nm, sel in rows:
        r, m, s, snx = run(sel)
        print(f'{nm:<26}{r:>+8.0f}%{m:>6.0f}%{s:>6.2f}{snx:>7.1f}/2')
    ks = k.set_index("date")["close"]; ks = ks[(ks.index >= S) & (ks.index <= E)]; ks = ks / ks.iloc[0]
    kdr = ks.pct_change().dropna()
    print(f'{"KOSPI":<26}{(ks.iloc[-1]-1)*100:>+8.0f}%{((ks-ks.cummax())/ks.cummax()).min()*100:>6.0f}%{kdr.mean()/kdr.std()*np.sqrt(252):>6.2f}')
    print("\n★ 대형한정/장기룩백이 삼닉포함율↑ + 수익 +722% 근접하면 = '대장 추종' 기준 발견.")
    print("  여전히 삼닉 못잡고 중소형이 최고면 = 봇은 중소형 모멘텀 별개도구로 확정.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
