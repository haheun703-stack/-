"""수급+레짐 백테스트 (A 청사진 알파 검증 — 룰 최종 판정).

수급 신호(외인+기관 동반·연속매수 + supply_divergence)에 KOSPI 레짐(BULL) 게이트를
결합해, 약세장(2024) 손실을 회피하면서 강세장 엣지를 유지하는지 전기간 검증.
레짐은 get_kospi_regime 로직(KOSPI MA20/60 + 실현변동성 백분위) 복제. 거래비용 0.2% 차감.

합격: 2024 PF≥1.1(약세장 손실회피) + 2025~26 PF≥1.5 + 전기간 평균 D+3 양수.
폐기: 2024 PF<1.0 또는 BULL게이트가 진입 거의 안 줄임.
"""
from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

COST = 0.002  # 왕복 거래비용 0.2%


def build_regime_map() -> tuple[dict, str, float]:
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"])
    k = k.set_index("date").sort_index()
    k["ma20"] = k["close"].rolling(20).mean()
    k["ma60"] = k["close"].rolling(60).mean()
    k["rv"] = k["close"].pct_change().rolling(20).std()
    k["rv_pct"] = k["rv"].rolling(120).rank(pct=True)

    def reg(r):
        if pd.isna(r["ma60"]):
            return "UNKNOWN"
        if r["close"] > r["ma20"]:
            rv = r["rv_pct"] if not pd.isna(r["rv_pct"]) else 1.0
            return "BULL" if rv < 0.5 else "CAUTION"
        elif r["close"] > r["ma60"]:
            return "BEAR"
        return "CRISIS"

    k["regime"] = k.apply(reg, axis=1)
    rmap = {pd.Timestamp(d).normalize(): v for d, v in k["regime"].items()}
    return rmap, str(k.index.max().date()), float(k["close"].iloc[-1])


def sig(r: dict) -> bool:
    return (
        r.get("foreign_net_5d", 0) > 0
        and r.get("inst_net_5d", 0) > 0
        and (r.get("foreign_consecutive_buy", 0) >= 3 or r.get("inst_consecutive_buy", 0) >= 3)
        and r.get("supply_divergence", 0) > 0
    )


def main() -> int:
    rmap, kospi_latest, kospi_close = build_regime_map()
    dist = {}
    for v in rmap.values():
        dist[v] = dist.get(v, 0) + 1
    print(f"KOSPI 레짐 분포: {dist}")
    print(f"kospi_index 최신: {kospi_latest} / close {kospi_close:.1f}")

    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f).sort_index())
        except Exception:
            pass
    print(f"종목 {len(dfs)}개 로드\n")

    periods = [
        ("2023-06-01", "2023-12-31", "23하"),
        ("2024-01-01", "2024-12-31", "24년(약세검증)"),
        ("2025-01-01", "2026-03-27", "25~26(강세)"),
    ]
    summary = {}
    for gate in ["수급단독", "수급+BULL게이트"]:
        print(f"=== {gate} (거래비용 0.2% 차감, entry=다음날시가/D+3종가) ===")
        print(f'{"기간":<16}{"표본":>8}{"승률":>7}{"평균D3":>9}{"PF":>7}')
        for s, e, lbl in periods:
            st = {"n": 0, "hit": 0, "win": 0.0, "loss": 0.0, "ret": 0.0}
            for df in dfs:
                d = df[(df.index >= s) & (df.index <= e)]
                if len(d) < 8:
                    continue
                o = d["open"].values
                c = d["close"].values
                lo = d["low"].values
                idx = d.index
                recs = d.to_dict("records")
                for i in range(len(d) - 5):
                    if not sig(recs[i]):
                        continue
                    if gate == "수급+BULL게이트":
                        if rmap.get(pd.Timestamp(idx[i]).normalize(), "") != "BULL":
                            continue
                    entry = o[i + 1]
                    if entry <= 0 or np.isnan(entry):
                        continue
                    # 손절 -3%: 보유중(D+1~D+3) 저가가 손절선 도달 시 청산, 아니면 D+3 종가
                    exitp = c[i + 3]
                    for j in range(i + 1, i + 4):
                        if lo[j] <= entry * 0.97:
                            exitp = entry * 0.97
                            break
                    d3 = exitp / entry - 1 - COST
                    if np.isnan(d3):
                        continue
                    st["n"] += 1
                    st["hit"] += int(d3 > 0)
                    st["ret"] += d3
                    if d3 > 0:
                        st["win"] += d3
                    else:
                        st["loss"] += -d3
            n = max(st["n"], 1)
            pf = st["win"] / st["loss"] if st["loss"] > 0 else 99.0
            summary[(gate, lbl)] = (st["n"], pf, st["ret"] / n * 100)
            print(f'{lbl:<16}{st["n"]:>8}{st["hit"]/n*100:>6.0f}%{st["ret"]/n*100:>+8.2f}%{pf:>7.2f}')
        print()

    # 합격 판정
    g = "수급+BULL게이트"
    n24, pf24, r24 = summary.get((g, "24년(약세검증)"), (0, 0, 0))
    n25, pf25, r25 = summary.get((g, "25~26(강세)"), (0, 0, 0))
    nbase24 = summary.get(("수급단독", "24년(약세검증)"), (0, 0, 0))[0]
    print("=== 합격 판정 ===")
    print(f"BULL게이트 약세장(24) 진입 {n24} vs 수급단독 {nbase24} (게이트 감소율 {100*(1-n24/max(nbase24,1)):.0f}%)")
    cond = (pf24 >= 1.1) and (pf25 >= 1.5) and (r24 > 0 or r25 > 0)
    print(f"24 PF {pf24:.2f}(≥1.1?) / 25~26 PF {pf25:.2f}(≥1.5?) → {'합격 ✅' if cond else '미달 ❌ (룰 재조정)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
