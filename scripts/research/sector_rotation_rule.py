"""종목 선정 규칙 검증 — 섹터 로테이션 (사장님 6/1 "기준이 나와야").

규칙: ①21섹터 강도=섹터내 평균20일수익 ②강도 상위3섹터 ③섹터내 거래대금상위 2종목
      ④동일가중 격주리밸 ⑤섹터 top3밖 밀리면 교체(개별손절X).
검증: 종목선택 거래대금순 vs 모멘텀순(개별알파 효과?) × 위상 × in/out × KOSPI대비.
★2025.6~2026.5. 상폐포함. look-ahead 0. 비용0.5%.
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
import yaml

CAP = 100_000_000
SLIP, SELL_TAX = 0.005, 0.0018
LOOKBACK = 20
TOPSEC = 3
PERSEC = 2


def extract_codes(v):
    out = []
    if isinstance(v, dict):
        for vv in v.values():
            out += extract_codes(vv)
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                t = x.get("ticker") or x.get("code")
                if t:
                    out.append(str(t))
            elif isinstance(x, str) and x.isdigit():
                out.append(x)
    return out


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    d = yaml.safe_load(open(PROJECT_ROOT / "config" / "sector_fire_map.yaml", encoding="utf-8"))
    sect = {n: [c for c in extract_codes(v) if c.isdigit() and len(c) == 6] for n, v in d["sectors"].items()}
    allc = sorted({c for v in sect.values() for c in v})
    store = {}
    for pat in ["data/processed", "data/delisted"]:
        for code in allc:
            if code in store:
                continue
            f = PROJECT_ROOT / pat / f"{code}.parquet"
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
            except Exception:
                continue
            if len(df) >= 70 and all(col in df.columns for col in ["close", "open", "trading_value"]):
                store[code] = df
    C = pd.DataFrame({c: store[c]["close"] for c in store}).reindex(cal)
    O = pd.DataFrame({c: store[c]["open"] for c in store}).reindex(cal)
    TV = pd.DataFrame({c: store[c]["trading_value"] for c in store}).reindex(cal).ffill()
    lastv = {c: store[c].index.max() for c in store}
    sect = {s: [c for c in cs if c in C.columns] for s, cs in sect.items()}
    sect = {s: cs for s, cs in sect.items() if len(cs) >= 2}
    return cal, k.set_index("date")["close"], C.ffill(), C, O, TV, lastv, sect


def sim(cal, Cm, Craw, Om, TVm, lastv, sect, S, E, phase, pick):
    days = [d for d in cal if S <= d <= E]
    rb = set(days[phase::10])
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    mom = Craw / Craw.shift(LOOKBACK) - 1
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP; target = None
    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        for c in list(pos):
            if d > lastv.get(c, d):
                px = Cm.at[d, c]
                if not pd.isna(px):
                    cash += pos[c] * px * 0.7 * (1 - SLIP - SELL_TAX)
                del pos[c]
        if dp is not None and dp in rb and target is not None:
            for c in list(pos):
                if c not in target:
                    px = Om.at[d, c]
                    if pd.isna(px) or px <= 0:
                        px = Cm.at[d, c]
                    if not pd.isna(px):
                        cash += pos[c] * px * (1 - SLIP - SELL_TAX)
                    del pos[c]
            if target:
                tgt = mv / len(target)
                for c in target:
                    px = Om.at[d, c]
                    if pd.isna(px) or px <= 0:
                        continue
                    cur = pos.get(c, 0) * px
                    if cur < tgt:
                        sh = int(min(tgt - cur, cash) / (px * (1 + SLIP)))
                        if sh > 0:
                            cash -= sh * px * (1 + SLIP); pos[c] = pos.get(c, 0) + sh
        if d in rb and i - LOOKBACK >= 0:
            # 섹터 강도 = 섹터내 평균 모멘텀
            strength = {}
            for s, codes in sect.items():
                cc = [c for c in codes if c in Cm.columns and d <= lastv.get(c, d)]
                mv2 = mom.loc[d][cc].dropna()
                if len(mv2) >= 2:
                    strength[s] = mv2.mean()
            strong = sorted(strength, key=strength.get, reverse=True)[:TOPSEC]
            tgt = []
            for s in strong:
                cc = [c for c in sect[s] if c in Cm.columns and d <= lastv.get(c, d)]
                if pick == "tv":      # 거래대금 상위 (유동성)
                    rank = TVm.loc[d][cc].dropna().nlargest(PERSEC)
                else:                  # 모멘텀 상위 (개별 알파)
                    rank = mom.loc[d][cc].dropna().nlargest(PERSEC)
                tgt += list(rank.index)
            target = tgt
        held = sum(sh * Cm.at[d, c] for c, sh in pos.items() if not pd.isna(Cm.at[d, c]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def kos(kc, s, e):
    sub = kc[(kc.index >= s) & (kc.index <= e)]
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def main() -> int:
    cal, kc, Cm, Craw, Om, TVm, lastv, sect = load()
    print(f"종목 선정 규칙 검증 — 섹터강도 top{TOPSEC} × 종목{PERSEC} 동일가중 격주 ({len(sect)}섹터)\n")
    for pick, lbl in [("tv", "거래대금순(규칙)"), ("mom", "모멘텀순(개별알파)")]:
        print(f"=== 섹터내 종목선택: {lbl} ===")
        for s, e, plabel in [("2025-06-01", "2025-11-30", "전반"),
                             ("2025-12-01", "2026-05-29", "후반"),
                             ("2025-06-01", "2026-05-29", "전체")]:
            S, E = pd.Timestamp(s), pd.Timestamp(e)
            ko = kos(kc, S, E)
            row = [sim(cal, Cm, Craw, Om, TVm, lastv, sect, S, E, p, pick)[0] * 100 - ko for p in [0, 3, 6, 9]]
            win = sum(r > 0 for r in row)
            print(f'  {plabel:<5} KOSPI{ko:>+6.0f}% | 위상별 vs K: ' + " ".join(f'{r:>+5.0f}' for r in row)
                  + f' | 초과 {win}/4 평균{np.mean(row):+.0f}%p')
        print()
    print("★ '거래대금순'이 후반 4/4 초과 + 모멘텀순과 비슷 = 개별알파 불필요(섹터만으로 충분), 규칙 확정.")
    print("  모멘텀순이 훨씬 높으면 = 개별알파도 기여(단 어제 IC0과 모순 → 재검토).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
