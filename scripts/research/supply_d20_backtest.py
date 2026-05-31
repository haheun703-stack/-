"""수급 D+20 스윙 백테스트 재현 + 강건성 검증 (코덱스 2단계, 5/31).

목적: scan_supply_swing.py 주석의 '확정 룰 PF 1.68'은 git에 백테스트 코드가
없었음(재현 불가). 동일 룰을 D+20 보유로 재현하고, 코덱스 2단계 요구
(거래대금 하한 / 슬리피지 / 구간별 OOS / 실효 종목수)를 함께 측정한다.

룰(scan_supply_swing.passes 동일):
  foreign_net_5d>0 AND inst_net_5d>0
  AND (foreign_consecutive_buy>=3 OR inst_consecutive_buy>=3)
  AND supply_divergence>0
진입: 신호 다음날 시가. 청산: D+20 종가(손절 시나리오는 보유중 저가 손절).

★ 생존자편향: data/processed = '현재 생존 종목'만(상폐 미포함) → PF 상방편향 가능.
  절대 PF가 아닌 '시나리오 간 상대 변화'로 해석할 것.
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

HOLD = 20


def sig(r: dict) -> bool:
    return (
        r.get("foreign_net_5d", 0) > 0
        and r.get("inst_net_5d", 0) > 0
        and (r.get("foreign_consecutive_buy", 0) >= 3 or r.get("inst_consecutive_buy", 0) >= 3)
        and r.get("supply_divergence", 0) > 0
    )


def run(dfs, s, e, cost, min_value, stop=None):
    st = {"n": 0, "hit": 0, "win": 0.0, "loss": 0.0, "ret": 0.0, "codes": set()}
    for code, df in dfs:
        d = df[(df.index >= s) & (df.index <= e)]
        if len(d) < HOLD + 2:
            continue
        o = d["open"].values
        c = d["close"].values
        lo = d["low"].values
        tv = d["trading_value"].values if "trading_value" in d.columns else np.full(len(d), 1e12)
        recs = d.to_dict("records")
        for i in range(len(d) - HOLD - 1):
            if not sig(recs[i]):
                continue
            if tv[i] < min_value:
                continue
            entry = o[i + 1]
            if entry <= 0 or np.isnan(entry):
                continue
            exitp = c[i + HOLD]
            if stop:
                for j in range(i + 1, i + 1 + HOLD):
                    if lo[j] <= entry * (1 - stop):
                        exitp = entry * (1 - stop)
                        break
            ret = exitp / entry - 1 - cost
            if np.isnan(ret):
                continue
            st["n"] += 1
            st["hit"] += int(ret > 0)
            st["ret"] += ret
            st["codes"].add(code)
            if ret > 0:
                st["win"] += ret
            else:
                st["loss"] += -ret
    n = max(st["n"], 1)
    pf = st["win"] / st["loss"] if st["loss"] > 0 else 99.0
    return st["n"], len(st["codes"]), st["hit"] / n * 100, st["ret"] / n * 100, pf


def main() -> int:
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    dfs = []
    for f in files:
        try:
            code = Path(f).stem
            dfs.append((code, pd.read_parquet(f).sort_index()))
        except Exception:
            pass
    print(f"종목 {len(dfs)}개 로드 (D+{HOLD} 보유, entry=다음날시가)\n")

    periods = [
        ("2023-06-01", "2023-12-31", "23하"),
        ("2024-01-01", "2024-12-31", "24약세"),
        ("2025-01-01", "2026-05-29", "25~26강세"),
        ("2023-06-01", "2026-05-29", "전체3년"),
    ]
    scenarios = [
        ("기본재현 (비용0.2% / 손절X / 거래대금0)", 0.002, 0, None),
        ("+ 거래대금 ≥10억",                      0.002, 1e9, None),
        ("+ 슬리피지(비용0.5%) + 거래대금≥10억",    0.005, 1e9, None),
        ("+ 손절-8% + 슬리피지 + 거래대금≥10억",     0.005, 1e9, 0.08),
    ]
    for name, cost, mv, stop in scenarios:
        print(f"=== {name} ===")
        print(f'{"기간":<12}{"표본":>7}{"종목":>6}{"승률":>7}{"평균D20":>10}{"PF":>7}')
        for s, e, lbl in periods:
            n, nc, hit, avg, pf = run(dfs, s, e, cost, mv, stop)
            print(f'{lbl:<12}{n:>7}{nc:>6}{hit:>6.0f}%{avg:>+9.2f}%{pf:>7.2f}')
        print()

    print("★ 생존자편향: data/processed=현재 생존종목만(상폐 미포함) → PF 상방편향 가능.")
    print("  절대 PF가 아닌 시나리오 간 상대 변화로 해석.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
