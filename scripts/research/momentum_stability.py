"""모멘텀 10종목 — 마지막 관문: 파라미터 안정성 + 레짐 의존성. 사장님 6/1.

10종목/1개월/격주가 적대적 검증(생존편향+비용0.5%) 통과 +73.9%p. 마지막 의심 둘:
  ① 파라미터 안정성: 종목수 7~13 × 리밸 시작 위상 0~9 흔들어도 KOSPI 초과 유지?
     (10종목만 우연한 sweet spot이면 = 과최적화 → 실전 위험)
  ② 레짐 의존성: 강세 구간을 전반(25.6~25.11)/후반(25.12~26.5) 나눠도 일관 초과?
     (한쪽만 폭발이면 = 그 구간 운)
상폐포함+비용0.5% 고정(현실 조건). ★2025.6~2026.5만. look-ahead 0.
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

CAP = 100_000_000
SELL_TAX = 0.0018
MIN_TV = 1e9
SLIP = 0.005      # 현실 비용
DELIQ = 0.30      # 상폐 정리매매 손실
LOOKBACK = 20     # 1개월


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    C, O, TV, lastv = {}, {}, {}, {}
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
            C[code] = df["close"]; O[code] = df["open"]; TV[code] = df["trading_value"]
            lastv[code] = df.index.max()
    Craw = pd.DataFrame(C).reindex(cal)
    Cm = Craw.ffill()
    Om = pd.DataFrame(O).reindex(cal)
    TVm = pd.DataFrame(TV).reindex(cal).ffill()
    return cal, k.set_index("date")["close"], Craw, Cm, Om, TVm, dict(lastv)


def sim(cal, Craw, Cm, Om, TVm, dla, S, E, K, phase):
    cols = list(Cm.columns)
    days = [d for d in cal if S <= d <= E]
    rb = set(days[phase::10])
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP; target = None
    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        for code in list(pos):
            if d > dla.get(code, d):
                last_px = Cm.at[d, code]
                if not pd.isna(last_px):
                    cash += pos[code] * last_px * (1 - DELIQ) * (1 - SLIP - SELL_TAX)
                del pos[code]
        if dp is not None and dp in rb and target is not None:
            for code in list(pos):
                if code not in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        px = Cm.at[d, code]
                    if not pd.isna(px):
                        cash += pos[code] * px * (1 - SLIP - SELL_TAX)
                    del pos[code]
            if target:
                tgt = mv / len(target)
                for code in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        continue
                    cur = pos.get(code, 0) * px
                    if cur < tgt:
                        sh = int(min(tgt - cur, cash) / (px * (1 + SLIP)))
                        if sh > 0:
                            cash -= sh * px * (1 + SLIP); pos[code] = pos.get(code, 0) + sh
                    elif cur > tgt * 1.3:
                        sh = int((cur - tgt) / px)
                        if 0 < sh <= pos.get(code, 0):
                            cash += sh * px * (1 - SLIP - SELL_TAX); pos[code] -= sh
        if d in rb:
            j = i - LOOKBACK
            if j >= 0:
                ret = {}
                for code in cols:
                    if d > dla.get(code, d):
                        continue
                    p0 = Craw.at[cal[j], code]; p1 = Craw.at[d, code]; tv = TVm.at[d, code]
                    if not pd.isna(p0) and not pd.isna(p1) and p0 > 0 and not pd.isna(tv) and tv >= MIN_TV:
                        ret[code] = p1 / p0 - 1
                target = sorted(ret, key=ret.get, reverse=True)[:K]
        held = sum(sh * Cm.at[d, code] for code, sh in pos.items() if not pd.isna(Cm.at[d, code]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def kos_ret(kclose, s, e):
    sub = kclose[(kclose.index >= s) & (kclose.index <= e)]
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def main() -> int:
    cal, kclose, Craw, Cm, Om, TVm, dla = load()
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    kos = kos_ret(kclose, S, E)

    print(f"관문① 파라미터 안정성 (상폐포함+비용0.5%, 전구간) — vs KOSPI {kos:+.1f}%\n")
    print(f'{"종목수":>6}' + "".join(f"위상{p}".rjust(8) for p in [0, 3, 6, 9]) + f'{"평균초과":>9}')
    win = 0; tot = 0
    for K in (7, 8, 9, 10, 11, 12, 13):
        row = []; excs = []
        for p in [0, 3, 6, 9]:
            ret, _ = sim(cal, Craw, Cm, Om, TVm, dla, S, E, K, p)
            exc = ret * 100 - kos; excs.append(exc); row.append(exc)
            tot += 1; win += (exc > 0)
        print(f'{K:>6}' + "".join(f'{r:>+7.0f}%' for r in row) + f'{np.mean(excs):>+8.0f}%p')
    print(f"\n  → KOSPI 초과 비율: {win}/{tot} ({win/tot*100:.0f}%). 8할↑이면 robust, 반반이면 과최적화.\n")

    print("관문② 레짐 의존성 (10종목, 위상0) — 강세 구간 분할")
    for s, e, lbl in [("2025-06-01", "2025-11-30", "전반"), ("2025-12-01", "2026-05-29", "후반"),
                       ("2025-06-01", "2026-05-29", "전체")]:
        ret, mdd = sim(cal, Craw, Cm, Om, TVm, dla, pd.Timestamp(s), pd.Timestamp(e), 10, 0)
        ks = kos_ret(kclose, pd.Timestamp(s), pd.Timestamp(e))
        print(f'  {lbl:<5} 모멘텀 {ret*100:>+7.1f}% / KOSPI {ks:>+7.1f}% = {ret*100-ks:>+7.1f}%p (MDD {mdd*100:.0f}%)')
    print("\n★ 둘 다 통과(안정성 8할↑ AND 전·후반 모두 초과) = 실전 후보. 한쪽 깨지면 = 운/과최적화.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
