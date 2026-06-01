"""★선검증 — 봇 모멘텀 top10이 "대장(반도체)"을 담았나 (사장님 "삼전닉스 사면 되나").

봇 모멘텀 집중(+452%)이 삼전닉스(+722%)를 자동 포착했는지 확인:
  매 월리밸 top10 모멘텀 선정 종목 → sector_fire_map(16섹터) 매핑 →
  반도체군 비중 + 삼전/하이닉스 포함율.
높으면 = 봇이 대장 자동포착(분산이라 덜 벌었을뿐) / 낮으면 = 헛다리.
★2025.6~2026.5. look-ahead 0.
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


def build_code2sector():
    d = yaml.safe_load(open(PROJECT_ROOT / "config" / "sector_fire_map.yaml", encoding="utf-8"))
    c2s = {}
    sectors = d.get("sectors", d)
    for sec, v in sectors.items():
        codes = []
        if isinstance(v, dict):
            codes = v.get("tickers", []) or []
        elif isinstance(v, list):
            codes = [x.get("ticker") if isinstance(x, dict) else x for x in v]
        for c in codes:
            c2s[str(c)] = sec
    return c2s, list(sectors.keys())


def main() -> int:
    c2s, sec_names = build_code2sector()
    semi_sectors = [s for s in sec_names if ("반도체" in s or "AI" in s or s in ("액침냉각",))]
    print(f"반도체군 섹터로 분류: {semi_sectors}\n")

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
        if "trading_value" not in df.columns or len(df) < 120:
            continue
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avg[Path(f).stem] = sub["trading_value"].mean(); store[Path(f).stem] = df["close"]
    uni = [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:150]]
    C = pd.DataFrame({c: store[c] for c in uni}).reindex(cal).ffill()
    mom = C / C.shift(20) - 1
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    days = [d for d in cal if S <= d <= E]
    reb = days[::20]

    name_map = {"005930": "삼성전자", "000660": "SK하이닉스", "042700": "한미반도체",
                "403870": "HPSP", "095340": "ISC"}
    semi_ratios = []; snx_incl = []
    print(f'{"리밸일":<12}{"반도체비중":>9}{"삼전닉스":>8}  상위10 종목(섹터)')
    for d in reb:
        m = mom.loc[d].dropna()
        tgt = list(m.nlargest(10).index)
        secs = [c2s.get(t, "기타") for t in tgt]
        nsemi = sum(1 for s in secs if s in semi_sectors)
        semi_ratios.append(nsemi / len(tgt))
        snx = ("005930" in tgt) + ("000660" in tgt)
        snx_incl.append(snx)
        tags = []
        for t in tgt[:6]:
            nm = name_map.get(t, t)
            sc = c2s.get(t, "기타")
            mark = "★" if t in ("005930", "000660") else ""
            tags.append(f"{mark}{nm}")
        print(f'{str(d)[:10]:<12}{nsemi*10:>8.0f}%{("삼"+str(snx)) if snx else "없음":>8}  {", ".join(tags)}')

    print(f"\n=== 종합 ===")
    print(f"봇 top10 평균 반도체군 비중: {np.mean(semi_ratios)*100:.0f}%")
    print(f"삼전·하이닉스 포함 리밸 비율: {sum(1 for x in snx_incl if x>0)}/{len(snx_incl)} "
          f"({sum(1 for x in snx_incl if x>0)/len(snx_incl)*100:.0f}%), 평균 {np.mean(snx_incl):.1f}/2종목")
    print("★ 반도체비중·삼전닉스 포함율 높음 → 봇이 대장 자동포착(분산이라 +452%<+722%). 낮음 → 헛다리.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
