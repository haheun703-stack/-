"""넓은 손절 검증 — "호흡 맞는 손절" (단타봇 제안1, 6/1).

작업2: 타이트 -3%트레일+재진입 = 추세에 독(휩쏘, +201%→-66%). 기각.
이제: 섹터규칙 raw(끝까지보유)에 '넓은/구조적 손절'만(재진입X) 얹으면
  전반 붕괴는 막고 후반 추세는 살리나? = 호흡 맞는 손절.
  raw(섹터빠질때만) vs trail-15%(넓은트레일) vs sma20이탈(명분깨짐) vs sma60이탈.
재진입 없음(휩쏘 주범 제거). look-ahead 0. 비용0.5%. ★2025.6~2026.5.
"""
from __future__ import annotations

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
LOOKBACK, TOPSEC, PERSEC = 20, 3, 2


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
            if len(df) >= 70 and all(c in df.columns for c in ["close", "open", "high", "low", "trading_value"]):
                store[code] = df
    keys = ["close", "open", "high", "low", "trading_value"]
    M = {key: pd.DataFrame({c: store[c][key] for c in store}).reindex(cal) for key in keys}
    lastv = {c: store[c].index.max() for c in store}
    sect = {s: [c for c in cs if c in M["close"].columns] for s, cs in sect.items()}
    sect = {s: cs for s, cs in sect.items() if len(cs) >= 2}
    return cal, k.set_index("date")["close"], M["close"].ffill(), M["close"], M["open"], M["high"], M["low"], M["trading_value"].ffill(), lastv, sect


def sim(cal, Cm, Craw, Om, Hm, Lm, TVm, lastv, sect, S, E, mode):
    days = [d for d in cal if S <= d <= E]
    rb = set(days[::10])
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    mom = Craw / Craw.shift(LOOKBACK) - 1
    sma20 = Cm.rolling(20).mean(); sma60 = Cm.rolling(60).mean()
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP; target = None
    trades = []

    def close_at(code, px):
        sh, entry, _ph = pos[code]; del pos[code]; trades.append(px / entry - 1)
        return sh * px * (1 - SLIP - SELL_TAX)

    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        for code in list(pos):
            if d > lastv.get(code, d):
                px = Cm.at[d, code]
                if not pd.isna(px):
                    cash += close_at(code, px * 0.7)
                elif code in pos:
                    del pos[code]
        # 손절 (모드별, 재진입 없음)
        for code in list(pos):
            sh, entry, ph = pos[code]
            hi = Hm.at[d, code]; lo = Lm.at[d, code]; cl = Cm.at[d, code]
            if not pd.isna(hi):
                ph = max(ph, hi); pos[code] = (sh, entry, ph)
            ex = None
            if mode == "trail15" and not pd.isna(lo) and lo <= ph * 0.85:
                ex = ph * 0.85 * (1 - 0.005)
            elif mode == "sma20" and not pd.isna(cl) and not pd.isna(sma20.at[d, code]) and cl < sma20.at[d, code]:
                ex = cl
            elif mode == "sma60" and not pd.isna(cl) and not pd.isna(sma60.at[d, code]) and cl < sma60.at[d, code]:
                ex = cl
            if ex is not None:
                cash += close_at(code, ex)
        # 격주 리밸 (섹터 빠지면 교체)
        if dp is not None and dp in rb and target is not None:
            for code in list(pos):
                if code not in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        px = Cm.at[d, code]
                    if not pd.isna(px):
                        cash += close_at(code, px)
            if target:
                tgt = mv / len(target)
                for code in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0 or code in pos:
                        continue
                    sh = int(min(tgt, cash) / (px * (1 + SLIP)))
                    if sh > 0:
                        cash -= sh * px * (1 + SLIP); pos[code] = (sh, px, px)
        if d in rb and i - LOOKBACK >= 0:
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
                tgt += list(TVm.loc[d][cc].dropna().nlargest(PERSEC).index)
            target = tgt
        held = sum(sh * Cm.at[d, c] for c, (sh, _e, _p) in pos.items() if not pd.isna(Cm.at[d, c]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    ret = mv / CAP - 1
    tr = np.array(trades) if trades else np.array([0.0])
    w = tr[tr > 0]; l = tr[tr <= 0]
    pf = (w.mean() / abs(l.mean())) if len(l) and l.mean() != 0 else float("inf")
    return ret, mdd, pf, len(trades)


def kos(kc, s, e):
    sub = kc[(kc.index >= s) & (kc.index <= e)]
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def main() -> int:
    cal, kc, Cm, Craw, Om, Hm, Lm, TVm, lastv, sect = load()
    print(f"넓은 손절 검증 — 섹터규칙 raw + 호흡맞는 손절(재진입X) ({len(sect)}섹터)\n")
    for s, e, lbl in [("2025-06-01", "2025-11-30", "전반"), ("2025-12-01", "2026-05-29", "후반"),
                     ("2025-06-01", "2026-05-29", "전체")]:
        S, E = pd.Timestamp(s), pd.Timestamp(e); ko = kos(kc, S, E)
        print(f"=== {lbl} (KOSPI {ko:+.0f}%) ===")
        print(f'{"손절모드":<10}{"수익":>8}{"vsK":>8}{"MDD":>7}{"손익비":>7}{"거래":>6}')
        for mode, nm in [("raw", "없음(raw)"), ("trail15", "트레일-15%"), ("sma60", "60일선이탈"), ("sma20", "20일선이탈")]:
            r, m, pf, n = sim(cal, Cm, Craw, Om, Hm, Lm, TVm, lastv, sect, S, E, mode)
            print(f'{nm:<10}{r*100:>+7.0f}%{r*100-ko:>+7.0f}%{m*100:>6.0f}%{pf:>7.2f}{n:>6}')
        print()
    print("★ 넓은 손절(트레일-15%/sma)이 전반 MDD를 raw보다 낮추면서 후반·전체 수익 거의 유지 = 호흡맞는 손절 성공.")
    print("  raw와 비슷하거나 나쁘면 = 섹터추세엔 손절 자체가 불필요(끝까지 보유가 답).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
