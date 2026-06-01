"""A 중소형 모멘텀 견고성 — 위상·K 과적합 점검 (단타봇 ③, 6/1).

A(u150·20일mom 집중)가 +409%지만 top3가 우연/과적합인지:
  K(3/5/10/20) × 위상(0~9, 리밸 시작일) × 룩백(20/40/60) → KOSPI 초과 비율·평균.
대부분 KOSPI 초과 유지면 = 견고(봇 코어 확정). 들쭉날쭉이면 = 과적합.
★2025.6~2026.5. look-ahead 0. 월리밸.
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
        if "trading_value" not in df.columns or len(df) < 150:
            continue
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avg[Path(f).stem] = sub["trading_value"].mean(); store[Path(f).stem] = df["close"]
    u150 = [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:150]]
    C = pd.DataFrame({c: store[c] for c in u150}).reindex(cal).ffill()
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    days = [d for d in cal if S <= d <= E]
    ks = k.set_index("date")["close"]; ks = ks[(ks.index >= S) & (ks.index <= E)]
    kos = (ks.iloc[-1] / ks.iloc[0] - 1) * 100

    def run(lb, K, phase):
        mom = C / C.shift(lb) - 1
        reb = set(days[phase::20])
        cash = 1e8; pos = {}
        for d in days:
            val = cash + sum(sh * C.at[d, c] for c, sh in pos.items() if not pd.isna(C.at[d, c]))
            if d in reb:
                tgt = [c for c in mom.loc[d].dropna().nlargest(K).index if C.at[d, c] > 0]
                if tgt:
                    pos = {}; per = val / len(tgt)
                    for c in tgt:
                        pos[c] = per / C.at[d, c]
                    cash = val - sum(sh * C.at[d, c] for c, sh in pos.items())
        last = days[-1]
        return (cash + sum(sh * C.at[last, c] for c, sh in pos.items() if not pd.isna(C.at[last, c]))) / 1e8 * 100 - 100

    print(f"A 중소형 모멘텀 견고성 — KOSPI {kos:+.0f}% 대비 초과(%p)\n")
    print(f'{"룩백":>5}{"K":>4}' + "".join(f"위{p}".rjust(7) for p in range(0, 10, 2)) + f'{"초과율":>8}')
    allwin = 0; alltot = 0
    for lb in (20, 40, 60):
        for K in (3, 5, 10, 20):
            excs = []
            for p in range(0, 10, 2):
                r = run(lb, K, p)
                excs.append(r - kos)
            win = sum(e > 0 for e in excs); allwin += win; alltot += len(excs)
            print(f'{lb:>5}{K:>4}' + "".join(f'{e:>+6.0f}' for e in excs) + f'{win}/{len(excs)}'.rjust(8))
    print(f'\n  전체 KOSPI 초과율: {allwin}/{alltot} ({allwin/alltot*100:.0f}%)')
    print("★ 8할↑ 초과 = A 중소형모멘텀 견고(봇 코어 확정). 반반/음수섞임 = 과적합(top3 우연).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
