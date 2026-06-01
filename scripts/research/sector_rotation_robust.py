"""섹터 로테이션 견고성 — 룩백·forward·위상 흔들어 IC 양수 일관성 (사장님 6/1 계속).

섹터(21) out IC +0.149(t1.79<2) = 유의 경계 미달. 단타봇 교훈: 견고성 먼저.
파라미터 18조합(룩백 10/20/40 × forward 5/10/20 × 위상 0/5)에서 섹터 IC 부호 일관성:
  대부분 양수 = 진짜 신호(파라미터 무관). 들쭉날쭉/음수 섞임 = 노이즈.
+ 전체 평균 IC와 out-of-sample 평균 IC로 종합 t.
★2025.6~2026.5. look-ahead 0. 자체 21섹터(sector_fire_map).
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
    sect = {}
    for name, v in d["sectors"].items():
        codes = [c for c in extract_codes(v) if c.isdigit() and len(c) == 6]
        if codes:
            sect[name] = codes
    allc = sorted({c for v in sect.values() for c in v})
    C = {}
    for code in allc:
        f = PROJECT_ROOT / "data" / "processed" / f"{code}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) >= 70:
            C[code] = df["close"]
    Cm = pd.DataFrame(C).reindex(cal).ffill()
    sect = {s: [c for c in cs if c in Cm.columns] for s, cs in sect.items()}
    sect = {s: cs for s, cs in sect.items() if len(cs) >= 2}
    return cal, Cm, sect


def ic_run(cal, Cm, sect, lb, fw, phase):
    mom = Cm / Cm.shift(lb) - 1
    fwd = Cm.shift(-fw) / Cm - 1
    rows = []
    start = max(lb, 1)
    for i in range(start + phase, len(cal) - fw, max(fw, 1)):
        d = cal[i]
        sm, sf = {}, {}
        for s, codes in sect.items():
            mv = mom.loc[d][codes].dropna(); fv = fwd.loc[d][codes].dropna()
            if len(mv) >= 2 and len(fv) >= 2:
                sm[s] = mv.mean(); sf[s] = fv.mean()
        common = set(sm) & set(sf)
        if len(common) < 5:
            continue
        a = pd.Series({s: sm[s] for s in common}); b = pd.Series({s: sf[s] for s in common})
        rows.append((d, a.corr(b, method="spearman")))
    return pd.DataFrame(rows, columns=["date", "ic"]).set_index("date")


def main() -> int:
    cal, Cm, sect = load()
    print(f"섹터 로테이션 견고성 — {len(sect)}섹터 {len(Cm.columns)}종목, 18조합\n")
    print(f'{"룩백":>5}{"fwd":>5}{"위상":>5}{"전체IC":>9}{"전체t":>7}{"outIC":>8}{"양수%":>7}')
    all_full, all_out, pos_cnt, tot = [], [], 0, 0
    for lb in (10, 20, 40):
        for fw in (5, 10, 20):
            for ph in (0, 5):
                s = ic_run(cal, Cm, sect, lb, fw, ph)
                full = s[(s.index >= "2025-06-01") & (s.index <= "2026-05-29")]["ic"].dropna()
                out = s[(s.index >= "2025-12-01") & (s.index <= "2026-05-29")]["ic"].dropna()
                if len(full) < 3:
                    continue
                t = full.mean() / (full.std() / np.sqrt(len(full))) if full.std() > 0 else 0
                posr = (full > 0).mean() * 100
                all_full.append(full.mean()); all_out.append(out.mean() if len(out) else np.nan)
                pos_cnt += (full.mean() > 0); tot += 1
                print(f'{lb:>5}{fw:>5}{ph:>5}{full.mean():>+8.3f}{t:>+7.2f}'
                      f'{(out.mean() if len(out) else 0):>+8.3f}{posr:>6.0f}%')
    print(f'\n  18조합 중 전체IC 양수: {pos_cnt}/{tot} ({pos_cnt/max(tot,1)*100:.0f}%)')
    print(f'  평균 전체IC {np.nanmean(all_full):+.3f} | 평균 outIC {np.nanmean(all_out):+.3f}')
    print("★ 양수 8할↑ + 평균IC>0 = 파라미터 무관 견고(진짜). 반반/음수섞임 = 노이즈(t1.79는 우연).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
