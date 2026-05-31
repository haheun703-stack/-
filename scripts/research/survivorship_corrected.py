"""생존편향 보정 백테스트 — 마지막 관문 (사장님 5/31).

최적 전략(supply_div + D+40 + 손절-8%)을 '생존만' vs '생존+상폐 190종목'으로 비교.
상폐 종목은 마지막 거래일(상폐일) 경과 시 강제 청산(손실 반영) → 생존편향 제거.
초과수익(vs KOSPI)이 보정 후에도 유지되면 진짜 알파, 무너지면 편향이었음.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import scripts.research.full_strategy_grid as fsg


def run(include_delisted, label):
    kospi = fsg.load_kospi()
    cal, cal_pos, overheat, sig_by, C, O, H, L = fsg.build(kospi, include_delisted=include_delisted)
    kclose = {d: float(v) for d, v in kospi["close"].items()}
    sig = sig_by["supply_div"]
    # 종목별 마지막 거래일(상폐일 포함) → 상폐 강제청산용
    last_valid = {}
    for code in O.columns:
        v = O[code].notna()
        if v.any():
            last_valid[code] = cal_pos[O.index[v][-1]]
    n_del = len(list((PROJECT_ROOT / "data" / "delisted").glob("*.parquet"))) if include_delisted else 0
    periods = [
        ("23하", "2023-06-01", "2023-12-31"),
        ("2024약세", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026극강세", "2026-01-01", "2026-05-29"),
        ("전체", "2023-06-01", "2026-05-29"),
    ]
    fsg.SLOTS = 10
    print(f"=== {label} (종목 {len(C.columns)}개{', 상폐 '+str(n_del) if include_delisted else ''}) ===")
    print(f'{"구간":<10}{"전략":>9}{"KOSPI":>9}{"초과":>9}{"승률":>6}{"MaxDD":>7}{"진입":>6}')
    res = {}
    for lbl, s, e in periods:
        sub = [d for d in cal if pd.Timestamp(s) <= d <= pd.Timestamp(e)]
        if len(sub) < 42:
            continue
        kr = (kclose[sub[-1]] / kclose[sub[0]] - 1) * 100
        lv = last_valid if include_delisted else None
        r = fsg.simulate(sub, overheat, sig, C, O, H, L, 40, 0.08, 0.0, 0.0, last_valid=lv)
        res[lbl] = r["ret"] * 100 - kr
        print(f'{lbl:<10}{r["ret"]*100:>+8.1f}%{kr:>+8.1f}%{(r["ret"]*100-kr):>+8.1f}%p'
              f'{r["winrate"]:>5.0f}%{r["mdd"]*100:>6.1f}%{r["entries"]:>6}')
    print()
    return res


def main() -> int:
    print("전략: supply_div + D+40 + 손절-8% / 슬롯10 / 생존편향 보정 비교\n")
    base = run(False, "생존만 (편향 有)")
    corr = run(True, "생존+상폐 (편향 보정)")
    print("=== 보정 영향 (구간별 초과수익 %p 변화) ===")
    for k in base:
        if k in corr:
            print(f'  {k:<10} {base[k]:>+7.1f}%p → {corr[k]:>+7.1f}%p  (Δ{corr[k]-base[k]:>+6.1f}%p)')
    print("\n★ 판정: 보정 후에도 2023~25 초과 +유지 = 진짜 알파. 초과 소멸/음수 = 생존편향이었음.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
