"""섹터/그룹 로테이션 변별력 — 개별종목(IC≈0) vs 섹터단위 (사장님 6/1).

까먹었던 우리 엔진 부활: sector_fire_map(21섹터) + group_structure(7그룹).
가설: 개별종목 모멘텀은 노이즈에 묻혀 IC≈0였지만, 섹터로 묶으면 테마/수급이
  상쇄 안 되고 살아난다(오늘 로봇섹터 통째 +30% 증거).
검증: 매 시점 섹터 평균 모멘텀(강도) rank vs 섹터 forward 수익 rank = 섹터 IC.
  in-sample(25.6~11) vs out-of-sample(25.12~26.5). 스피어만, 비겹침.
  + 강한 섹터 top-K 종목 보유 vs KOSPI 포트.
★2025.6~2026.5. look-ahead 0.
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

LOOKBACK = 20
FWD = 10
CAP = 100_000_000
BUY_SLIP, SELL_SLIP, SELL_TAX = 0.0015, 0.0015, 0.0018


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


def load_map(fn, key):
    d = yaml.safe_load(open(PROJECT_ROOT / "config" / f"{fn}.yaml", encoding="utf-8"))
    grp = d.get(key, d)
    m = {}
    for name, v in grp.items():
        codes = [c for c in extract_codes(v) if c.isdigit() and len(c) == 6]
        if codes:
            m[name] = codes
    return m


def load_closes(cal, codes):
    C = {}
    for code in codes:
        f = PROJECT_ROOT / "data" / "processed" / f"{code}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) >= 60:
            C[code] = df["close"]
    return pd.DataFrame(C).reindex(cal).ffill()


def sector_ic(cal, Cm, sectors):
    Craw = Cm.copy()
    mom = Craw / Craw.shift(LOOKBACK) - 1
    fwd = Craw.shift(-FWD) / Craw - 1
    rows = []
    for i in range(LOOKBACK, len(cal) - FWD, FWD):
        d = cal[i]
        smom, sfwd = {}, {}
        for s, codes in sectors.items():
            cc = [c for c in codes if c in Cm.columns]
            if len(cc) < 2:
                continue
            mv = mom.loc[d][cc].dropna()
            fv = fwd.loc[d][cc].dropna()
            if len(mv) >= 2 and len(fv) >= 2:
                smom[s] = mv.mean(); sfwd[s] = fv.mean()
        common = set(smom) & set(sfwd)
        if len(common) < 5:
            continue
        a = pd.Series({s: smom[s] for s in common})
        b = pd.Series({s: sfwd[s] for s in common})
        rows.append((d, a.corr(b, method="spearman")))
    return pd.DataFrame(rows, columns=["date", "ic"]).set_index("date")


def stat(s, lbl):
    ic = s["ic"].dropna()
    if len(ic) < 3:
        print(f"  {lbl:<22} 표본부족({len(ic)})"); return
    t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0
    print(f"  {lbl:<22} 평균 섹터IC {ic.mean():+.3f} | IC>0 {int((ic>0).mean()*100)}% | t {t:+.2f} | n={len(ic)}")


def rotation_port(cal, Cm, sectors, S, E, topk_sec=3, per_sec=2):
    """강한 섹터 top-K의 각 종목(모멘텀순 per_sec) 보유, 격주 리밸."""
    Craw = Cm.copy()
    mom = Craw / Craw.shift(LOOKBACK) - 1
    days = [d for d in cal if S <= d <= E]
    rb = set(days[::10])
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0; mv = CAP; target = None
    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        if dp in rb and target is not None:
            for c in list(pos):
                if c not in target:
                    px = Cm.at[d, c]
                    if not pd.isna(px):
                        cash += pos[c] * px * (1 - SELL_SLIP - SELL_TAX)
                    del pos[c]
            if target:
                tgt = mv / len(target)
                for c in target:
                    px = Cm.at[d, c]
                    if pd.isna(px) or px <= 0:
                        continue
                    cur = pos.get(c, 0) * px
                    if cur < tgt:
                        sh = int(min(tgt - cur, cash) / (px * (1 + BUY_SLIP)))
                        if sh > 0:
                            cash -= sh * px * (1 + BUY_SLIP); pos[c] = pos.get(c, 0) + sh
        if d in rb and i - LOOKBACK >= 0:
            smom = {}
            for s, codes in sectors.items():
                cc = [c for c in codes if c in Cm.columns]
                mv2 = mom.loc[d][cc].dropna()
                if len(mv2) >= 2:
                    smom[s] = mv2.mean()
            strong = sorted(smom, key=smom.get, reverse=True)[:topk_sec]
            tgt = []
            for s in strong:
                cc = [c for c in sectors[s] if c in Cm.columns]
                mv2 = mom.loc[d][cc].dropna()
                tgt += list(mv2.nlargest(per_sec).index)
            target = tgt
        held = sum(sh * Cm.at[d, c] for c, sh in pos.items() if not pd.isna(Cm.at[d, c]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def main() -> int:
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    kclose = k.set_index("date")["close"]
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    kos = (kclose[(kclose.index >= S) & (kclose.index <= E)].iloc[-1] /
           kclose[(kclose.index >= S) & (kclose.index <= E)].iloc[0] - 1) * 100

    for fn, key, lbl in [("sector_fire_map", "sectors", "섹터(21)"), ("group_structure", "groups", "그룹주(7)")]:
        m = load_map(fn, key)
        allc = sorted({c for v in m.values() for c in v})
        Cm = load_closes(cal, allc)
        avail = {s: [c for c in cs if c in Cm.columns] for s, cs in m.items()}
        avail = {s: cs for s, cs in avail.items() if len(cs) >= 2}
        print(f"=== {lbl} 로테이션 변별력 — {len(avail)}개 그룹, {len(Cm.columns)}종목 (KOSPI {kos:+.1f}%) ===")
        ics = sector_ic(cal, Cm, avail)
        stat(ics[(ics.index >= "2025-06-01") & (ics.index <= "2025-11-30")], "in-sample(25.6~11)")
        stat(ics[(ics.index >= "2025-12-01") & (ics.index <= "2026-05-29")], "out-of-sample(25.12~)")
        stat(ics[(ics.index >= "2025-06-01") & (ics.index <= "2026-05-29")], "전체")
        ret, mdd = rotation_port(cal, Cm, avail, S, E, 3, 2)
        print(f"  강한섹터 top3×2종목 포트: {ret*100:+.1f}% (vs KOSPI {ret*100-kos:+.1f}%p, MDD {mdd*100:.0f}%)\n")
    print("★ 개별종목 IC≈0(-0.03)과 달리 섹터IC가 +이고 t>2면 = 로테이션이 진짜 무기.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
