"""하이브리드 첫 실증 — 섹터규칙 + L4(손절+재진입) (단타봇 작업2, 6/1).

섹터규칙(섹터강도 top3 → 거래대금 top2 동일가중 격주)은 후반+94%p/전반-45%p(시변).
L4 얹기: 종목 고점-3% 트레일 손절(슬리피지) + 손절 후 섹터 top3 유지 시 회복 재진입.
측정(단타봇 판정기준 = IC 아닌 손익비·MDD):
  ① 전반 -45%p 국면: 무손절 MDD vs +L4 MDD (손절이 생존가능 만드나)
  ② 후반 +94%p: 재진입이 유지하나
  ③ raw 격주보유 vs +L4 전체기간 수익·손익비
look-ahead 0(전일까지 선정). 비용 레그당 0.5%. 상폐포함. ★2025.6~2026.5.
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
SLIP, SELL_TAX, STOP_SLIP = 0.005, 0.0018, 0.01
LOOKBACK, TOPSEC, PERSEC, STOP = 20, 3, 2, 0.03


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
    Cm = M["close"].ffill()
    lastv = {c: store[c].index.max() for c in store}
    sect = {s: [c for c in cs if c in Cm.columns] for s, cs in sect.items()}
    sect = {s: cs for s, cs in sect.items() if len(cs) >= 2}
    return cal, k.set_index("date")["close"], Cm, M["close"], M["open"], M["high"], M["low"], M["trading_value"].ffill(), lastv, sect


def sim(cal, Cm, Craw, Om, Hm, Lm, TVm, lastv, sect, S, E, use_l4):
    days = [d for d in cal if S <= d <= E]
    rb = set(days[::10])
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    mom = Craw / Craw.shift(LOOKBACK) - 1
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP; target = None
    trades = []  # 청산된 거래 수익률 (손익비용)

    def close_pos(code, px, d):
        sh, entry, _ph = pos[code]
        cash_add = sh * px * (1 - SLIP - SELL_TAX)
        trades.append(px / entry - 1)
        del pos[code]
        return cash_add

    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        # 상폐 강제청산
        for code in list(pos):
            if d > lastv.get(code, d):
                px = Cm.at[d, code]
                if not pd.isna(px):
                    cash += close_pos(code, px * 0.7, d)
                elif code in pos:
                    del pos[code]
        # L4 손절 (고점 -3% 트레일)
        if use_l4:
            for code in list(pos):
                sh, entry, ph = pos[code]
                hi = Hm.at[d, code]; lo = Lm.at[d, code]
                if not pd.isna(hi):
                    ph = max(ph, hi); pos[code] = (sh, entry, ph)
                stop = ph * (1 - STOP)
                if not pd.isna(lo) and lo <= stop:
                    cash += close_pos(code, stop * (1 - STOP_SLIP), d)
        # 격주 리밸 (전일이 리밸일)
        if dp is not None and dp in rb and target is not None:
            for code in list(pos):
                if code not in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        px = Cm.at[d, code]
                    if not pd.isna(px):
                        cash += close_pos(code, px, d)
            if target:
                tgt = mv / len(target)
                for code in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        continue
                    cur = pos.get(code, (0, 0, 0))[0] * px
                    if cur < tgt:
                        sh = int(min(tgt - cur, cash) / (px * (1 + SLIP)))
                        if sh > 0:
                            cash -= sh * px * (1 + SLIP)
                            old = pos.get(code, (0, px, px))
                            pos[code] = (old[0] + sh, px if old[0] == 0 else old[1], max(old[2], px))
        # L4 재진입: 보유 안 했지만 target(섹터 top3)이고 회복(close>ma5)이면 재진입
        if use_l4 and target and i % 1 == 0:
            ma5 = Craw[ [c for c in target if c in Craw.columns] ].rolling(5).mean()
            for code in target:
                if code in pos or code not in Cm.columns or d > lastv.get(code, d):
                    continue
                cl = Cm.at[d, code]
                m5 = ma5.at[d, code] if code in ma5.columns else np.nan
                if not pd.isna(cl) and not pd.isna(m5) and cl > m5 and cash > mv / (TOPSEC * PERSEC) * 0.5:
                    tgt_amt = mv / (TOPSEC * PERSEC)
                    sh = int(min(tgt_amt, cash) / (cl * (1 + SLIP)))
                    if sh > 0:
                        cash -= sh * cl * (1 + SLIP); pos[code] = (sh, cl, cl)
        # 리밸일 = 섹터 선정
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
    wins = tr[tr > 0]; losses = tr[tr <= 0]
    payoff = (wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else float("inf")
    return ret, mdd, payoff, len(trades), (wins.mean() * 100 if len(wins) else 0), (losses.mean() * 100 if len(losses) else 0)


def kos(kc, s, e):
    sub = kc[(kc.index >= s) & (kc.index <= e)]
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def main() -> int:
    cal, kc, Cm, Craw, Om, Hm, Lm, TVm, lastv, sect = load()
    print(f"하이브리드 실증 — 섹터규칙 + L4(손절-3%+재진입) ({len(sect)}섹터)\n")
    for s, e, lbl in [("2025-06-01", "2025-11-30", "전반(시변약세)"),
                     ("2025-12-01", "2026-05-29", "후반(강세)"),
                     ("2025-06-01", "2026-05-29", "전체")]:
        S, E = pd.Timestamp(s), pd.Timestamp(e); ko = kos(kc, S, E)
        print(f"=== {lbl} (KOSPI {ko:+.0f}%) ===")
        print(f'{"":8}{"수익":>8}{"vsK":>8}{"MDD":>8}{"손익비":>8}{"거래":>6}{"평균익/손":>12}')
        for use_l4, nm in [(False, "raw"), (True, "+L4")]:
            r, m, pf, n, aw, al = sim(cal, Cm, Craw, Om, Hm, Lm, TVm, lastv, sect, S, E, use_l4)
            print(f'{nm:8}{r*100:>+7.0f}%{r*100-ko:>+7.0f}%{m*100:>7.0f}%{pf:>8.2f}{n:>6}  {aw:>+5.1f}/{al:>+5.1f}')
        print()
    print("★ 판정(단타봇): ①전반 +L4 MDD가 raw보다 확 낮아 생존가능 ②후반 +L4가 +94%p 유지 ③전체 +L4 손익비 양수+MDD통제 = 하이브리드 paper후보.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
