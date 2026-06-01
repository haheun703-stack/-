"""섹터 로테이션 t 확정 — 일별 IC + Newey-West (사장님 6/1 본론).

비겹침 12시점은 t<2(표본부족). 표본 최대 확대:
  매 거래일 21섹터 모멘텀(20일) vs forward(10일) 스피어만 IC → n~240시점.
  forward 10일 겹침 → Newey-West HAC(lag=10) t-stat으로 보정(과장 제거).
이게 t<2가 '표본부족'인지 '진짜 신호없음'인지 가른다. in/out 분리.
★2025.6~2026.5. look-ahead 0. sector_fire_map 21섹터.
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


def nw_tstat(x, lag):
    """Newey-West HAC t-stat (평균=0 검정). statsmodels 없으면 수동."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1 - l / (lag + 1)
        cov = (e[l:] @ e[:-l]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    return mu, (mu / se if se > 0 else np.nan)


def main() -> int:
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    d = yaml.safe_load(open(PROJECT_ROOT / "config" / "sector_fire_map.yaml", encoding="utf-8"))
    sect = {n: [c for c in extract_codes(v) if c.isdigit() and len(c) == 6] for n, v in d["sectors"].items()}
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

    mom = Cm / Cm.shift(LOOKBACK) - 1
    fwd = Cm.shift(-FWD) / Cm - 1
    rows = []
    for i in range(LOOKBACK, len(cal) - FWD):  # 매 거래일 (겹침)
        dt = cal[i]
        sm_, sf_ = {}, {}
        for s, codes in sect.items():
            mv = mom.loc[dt][codes].dropna(); fv = fwd.loc[dt][codes].dropna()
            if len(mv) >= 2 and len(fv) >= 2:
                sm_[s] = mv.mean(); sf_[s] = fv.mean()
        common = set(sm_) & set(sf_)
        if len(common) < 5:
            continue
        a = pd.Series({s: sm_[s] for s in common}); b = pd.Series({s: sf_[s] for s in common})
        rows.append((dt, a.corr(b, method="spearman")))
    s = pd.DataFrame(rows, columns=["date", "ic"]).set_index("date")["ic"]

    print(f"섹터 로테이션 t 확정 — 일별 IC {len(s)}시점 + Newey-West(lag{FWD})\n")
    print(f'{"구간":<20}{"평균IC":>9}{"IC>0":>7}{"NW t":>8}{"판정":>10}')
    for lbl, lo, hi in [("in-sample(25.6~11)", "2025-06-01", "2025-11-30"),
                        ("out-of-sample(25.12~)", "2025-12-01", "2026-05-29"),
                        ("전체(25.6~26.5)", "2025-06-01", "2026-05-29")]:
        sub = s[(s.index >= lo) & (s.index <= hi)]
        mu, t = nw_tstat(sub.values, FWD)
        verdict = "✅유의(t>2)" if (t and t > 2) else ("≈경계" if (t and t > 1.5) else "신호약함")
        print(f'{lbl:<20}{mu*100:>+8.2f}%{int((sub>0).mean()*100):>6}%{t:>+8.2f}{verdict:>12}')
    print(f"\n★ 전체+out NW t>2 = 표본부족이 t<2 원인이었고 섹터로테이션은 진짜(유의).")
    print("  NW t<2 유지 = 겹침 보정해도 통계적 미달 = 신호가 약한 것(표본 탓 아님).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
