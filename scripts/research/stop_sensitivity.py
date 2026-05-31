"""손절 임계 민감도 — 과최적화 최종 반증 (사장님 5/31).

supply_div + D+40 최고가 '손절-8%'였는데 트레일-7%는 재앙 = 민감도 극심.
손절 -5/8/10/12/15/20%가 모두 KOSPI 초과면 강건, 8%만 좋으면 과최적화.
구간: 전체 + 2024약세 + 2025(robustness 핵심 구간). 슬롯 10 고정.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import scripts.research.full_strategy_grid as fsg


def main() -> int:
    kospi = fsg.load_kospi()
    cal, cal_pos, overheat, sig_by, C, O, H, L = fsg.build(kospi)
    kclose = {d: float(v) for d, v in kospi["close"].items()}
    sig = sig_by["supply_div"]
    fsg.SLOTS = 10

    periods = [
        ("전체", "2023-06-01", "2026-05-29"),
        ("2024약세", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
    ]
    print("손절 임계 민감도 (supply_div + D+40 + 슬롯10, vs KOSPI 구간 초과%p):\n")
    print(f'{"손절":>6}{"전체초과":>12}{"2024약세초과":>14}{"2025초과":>12}')
    for stop in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        out = []
        for lbl, s, e in periods:
            sub = [d for d in cal if pd.Timestamp(s) <= d <= pd.Timestamp(e)]
            kr = (kclose[sub[-1]] / kclose[sub[0]] - 1) * 100
            r = fsg.simulate(sub, overheat, sig, C, O, H, L, 40, stop, 0.0, 0.0)
            out.append(r["ret"] * 100 - kr)
        print(f'{stop*100:>5.0f}%{out[0]:>+11.1f}%p{out[1]:>+13.1f}%p{out[2]:>+11.1f}%p')
    print("\n★ 모든 손절값이 전구간 KOSPI 초과(+) 일관 = 강건(손절8은 우연 아님).")
    print("  특정 손절값만 +, 나머지 - = 과최적화. 생존편향 미보정 가설 유지.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
