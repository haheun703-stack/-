"""모멘텀 로테이션 — 적대적 재검증 (생존편향 + 비용 현실화). 사장님 6/1.

momentum_rotation에서 1개월/격주가 KOSPI(+214%) 초과(+421%) → 흥분 금지, 깐다:
  ① 생존편향: 상폐 190종목(구간 중 상폐 포함)을 풀에 합침. 보유 중 상폐 →
     마지막 종가 × (1-정리매매손실)로 강제청산. 상폐 후 모멘텀 선정 불가.
  ② 비용 현실화: 슬리피지 0.15%(낙관) vs 0.5%(단기 급등주 갭·호가공백 현실).
이긴 조합(1개월 룩백, 격주, K=5/10/20)만. ★2025.6~2026.5만. look-ahead 0.
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


def load(include_delisted):
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    C, O, TV, lastv = {}, {}, {}, {}
    srcs = [("data/processed/*.parquet", False)]
    if include_delisted:
        srcs.append(("data/delisted/*.parquet", True))
    for pat, _dl in srcs:
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
    Cm = Craw.ffill()          # 평가용(상폐 후 ffill되지만 보유로직에서 강제청산)
    Om = pd.DataFrame(O).reindex(cal)
    TVm = pd.DataFrame(TV).reindex(cal).ffill()
    # 상폐 마스크: 종목별 마지막 유효일 이후 = True(거래불가)
    delisted_after = {c: lastv[c] for c in C}
    return cal, k.set_index("date")["close"], Craw, Cm, Om, TVm, delisted_after


def sim(cal, Craw, Cm, Om, TVm, dla, S, E, lookback, K, slip, deliq):
    cols = list(Cm.columns)
    days = [d for d in cal if S <= d <= E]
    rb = set(days[::10])  # 격주
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP
    target = None; n_delist = 0
    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        # 보유 중 상폐 강제청산 (마지막 유효일 지남)
        for code in list(pos):
            if d > dla.get(code, d):
                last_px = Cm.at[d, code]  # ffill된 마지막 종가
                if not pd.isna(last_px):
                    cash += pos[code] * last_px * (1 - deliq) * (1 - slip - SELL_TAX)
                del pos[code]; n_delist += 1
        # 리밸 재조정 (전일이 리밸일)
        if dp is not None and dp in rb and target is not None:
            for code in list(pos):
                if code not in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        px = Cm.at[d, code]
                    if not pd.isna(px):
                        cash += pos[code] * px * (1 - slip - SELL_TAX)
                    del pos[code]
            if target:
                tgt = mv / len(target)
                for code in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        continue
                    cur = pos.get(code, 0) * px
                    if cur < tgt:
                        buy = min(tgt - cur, cash)
                        sh = int(buy / (px * (1 + slip)))
                        if sh > 0:
                            cash -= sh * px * (1 + slip); pos[code] = pos.get(code, 0) + sh
                    elif cur > tgt * 1.3:
                        sh = int((cur - tgt) / px)
                        if 0 < sh <= pos.get(code, 0):
                            cash += sh * px * (1 - slip - SELL_TAX); pos[code] -= sh
        # 리밸일 = 다음 목표 선정 (오늘 종가까지, 상폐 안 된 종목만)
        if d in rb:
            j = i - lookback
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
    return mv / CAP - 1, mdd, n_delist


def main() -> int:
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    print("모멘텀 1개월/격주 — 적대적 재검증 (생존편향 + 비용)\n")
    scenarios = [
        ("생존만+비용0.15%(원본)", False, 0.0015, 0.0),
        ("상폐포함+비용0.15%", True, 0.0015, 0.30),
        ("상폐포함+비용0.5%(현실)", True, 0.005, 0.30),
    ]
    for lbl, incdl, slip, deliq in scenarios:
        cal, kclose, Craw, Cm, Om, TVm, dla = load(incdl)
        ksub = kclose[(kclose.index >= S) & (kclose.index <= E)]
        kos = (ksub.iloc[-1] / ksub.iloc[0] - 1) * 100
        print(f"=== {lbl} (KOSPI {kos:+.1f}%, 종목풀 {len(Cm.columns)}) ===")
        print(f'{"종목수":>6}{"수익":>10}{"vs KOSPI":>10}{"MDD":>8}{"상폐청산":>8}')
        for K in (5, 10, 20):
            ret, mdd, nd = sim(cal, Craw, Cm, Om, TVm, dla, S, E, 20, K, slip, deliq)
            print(f'{K:>6}{ret*100:>+9.1f}%{(ret*100-kos):>+9.1f}%p{mdd*100:>7.1f}%{nd:>8}')
        print()
    print("★ 상폐 포함·비용 현실화 후에도 KOSPI 초과(+) 유지 = 진짜 무기. 무너지면 = 생존편향/비용 착시.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
