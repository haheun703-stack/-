"""대형주+소부장 집중 — 소형 버리고 강점만 (사장님 6/1 장마감).

6/1 실측: 우리 엔진이 대형 주도주(삼성전자·LG전자·NAVER·로보스타·삼성SDI)는 5/29 미리 포착,
소형 테마·ETF는 못 잡음. → universe를 거래대금 상위(대형+주요 소부장)로 좁혀 검증.
전체 1160종목에선 신호=노이즈(어제)였으나, 좁히면 사는가?
  universe: 최근60일 평균 거래대금 상위 N (소형 잡주 자동 배제)
  전략A: 1개월 모멘텀 상위 K
  전략B(오늘 조합): 수급DIV>0 & 거래량≥1.5 & 박스돌파임박 & 추세 만족 → 모멘텀순 K
격주 리밸 × 위상0/3/6/9(안정성) × 상폐포함 × 비용0.5%. ★2025.6~2026.5. KOSPI대비. look-ahead 0.
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
SLIP = 0.005
DELIQ = 0.30
LOOKBACK = 20


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
            if code in store:
                continue
            store[code] = df
    return cal, k.set_index("date")["close"], store


def build_universe(store, cal, top_n, asof):
    """asof(5/29) 기준 최근60일 평균 거래대금 상위 N = 대형주+주요 소부장."""
    avgtv = {}
    for code, df in store.items():
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avgtv[code] = sub["trading_value"].mean()
    return [c for c, _ in sorted(avgtv.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def mats(store, cal, universe):
    C = pd.DataFrame({c: store[c]["close"] for c in universe}).reindex(cal)
    O = pd.DataFrame({c: store[c]["open"] for c in universe}).reindex(cal)
    V = pd.DataFrame({c: store[c]["volume"] for c in universe}).reindex(cal)
    VMA = pd.DataFrame({c: store[c].get("volume_ma20") for c in universe}).reindex(cal).ffill()
    SD = pd.DataFrame({c: store[c].get("supply_divergence") for c in universe}).reindex(cal).ffill()
    H20 = pd.DataFrame({c: store[c].get("high_20") for c in universe}).reindex(cal).ffill()
    lastv = {c: store[c].index.max() for c in universe}
    return C.ffill(), C, O, V, VMA, SD, H20, lastv


def sim(cal, Cm, Craw, Om, Vm, VMAm, SDm, H20m, lastv, S, E, K, phase, strat):
    cols = list(Cm.columns)
    days = [d for d in cal if S <= d <= E]
    rb = set(days[phase::10])
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    ma20 = Cm.rolling(20).mean()
    mom = Craw / Craw.shift(LOOKBACK) - 1
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP; target = None
    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        for code in list(pos):
            if d > lastv.get(code, d):
                px = Cm.at[d, code]
                if not pd.isna(px):
                    cash += pos[code] * px * (1 - DELIQ) * (1 - SLIP - SELL_TAX)
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
                tgt = mv / max(len(target), 1)
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
        if d in rb and i - LOOKBACK >= 0:
            m = mom.loc[d].dropna()
            valid = [c for c in m.index if d <= lastv.get(c, d)]
            m = m[valid]
            if strat == "A":
                target = list(m.nlargest(K).index)
            else:  # B: 조합 신호 충족 → 모멘텀순
                sig = []
                for c in valid:
                    vr = (Vm.at[d, c] / VMAm.at[d, c]) if VMAm.at[d, c] else 0
                    nb = (Cm.at[d, c] / H20m.at[d, c]) if H20m.at[d, c] else 0
                    up = Cm.at[d, c] > ma20.at[d, c] if not pd.isna(ma20.at[d, c]) else False
                    sd = SDm.at[d, c] if not pd.isna(SDm.at[d, c]) else 0
                    if sd > 0 and vr >= 1.5 and nb >= 0.98 and up:
                        sig.append(c)
                sig = sorted(sig, key=lambda c: m.get(c, -9), reverse=True)
                target = sig[:K]
        held = sum(sh * Cm.at[d, code] for code, sh in pos.items() if not pd.isna(Cm.at[d, code]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def kos_ret(kclose, s, e):
    sub = kclose[(kclose.index >= s) & (kclose.index <= e)]
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def main() -> int:
    cal, kclose, store = load()
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    asof = pd.Timestamp("2025-05-30")  # universe 선정 기준(구간 시작 직전, look-ahead 0)
    kos = kos_ret(kclose, S, E)
    for top_n in (200, 300):
        universe = build_universe(store, cal, top_n, asof)
        M = mats(store, cal, universe)
        print(f"=== universe = 거래대금 상위 {top_n} ({len(universe)}종목) — KOSPI {kos:+.1f}% ===")
        for strat, lbl in [("A", "모멘텀"), ("B", "오늘조합(수급DIV+거래량+돌파+추세)")]:
            row = []
            for p in [0, 3, 6, 9]:
                ret, mdd = sim(cal, *M, S, E, 10, p, strat)
                row.append(ret * 100 - kos)
            win = sum(r > 0 for r in row)
            print(f'  {lbl:<32} 위상별 vs KOSPI: ' + " ".join(f'{r:>+5.0f}' for r in row)
                  + f'  | 초과 {win}/4  평균{np.mean(row):+.0f}%p')
        print()
    print("★ 좁힌 universe에서 위상 4개 다 KOSPI 초과(4/4)면 = 대형주 집중이 진짜. 들쭉날쭉이면 = 여전히 노이즈.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
