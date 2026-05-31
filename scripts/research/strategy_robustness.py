"""최적 전략 적대적 강건성 검증 (사장님 5/31 — 미루지 않고 즉시).

full_strategy_grid 최고 조합(supply_div + D+40 + 손절-8%만)이 진짜인지 공격:
  ① 연도별 워크포워드 (매 구간 KOSPI 이기나 — 한 구간 운 아닌지)
  ② 슬롯 안정성 (슬롯 5/10/20에서 일관되나 — 집중운 아닌지)
손절8 최고 vs 트레일7 재앙 = 민감도 극심 → 과최적화 의심을 정량 반증/입증.
★ 생존편향 미보정(상방편향). 매 구간·슬롯 KOSPI 초과 + 안정성으로만 판정.
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

    periods = [
        ("23하", "2023-06-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-05-29"),
        ("전체", "2023-06-01", "2026-05-29"),
    ]
    print("전략: supply_div + D+40 + 손절-8%만 / 연도 × 슬롯 워크포워드 (vs KOSPI 구간)\n")
    for slots in [5, 10, 20]:
        fsg.SLOTS = slots
        print(f"=== 슬롯 {slots} ===")
        print(f'{"구간":<8}{"전략":>9}{"KOSPI":>9}{"초과":>9}{"승률":>6}{"MaxDD":>7}{"평균보유":>8}{"진입":>6}')
        for lbl, s, e in periods:
            sub = [d for d in cal if pd.Timestamp(s) <= d <= pd.Timestamp(e)]
            if len(sub) < 42:
                continue
            kr = (kclose[sub[-1]] / kclose[sub[0]] - 1) * 100
            r = fsg.simulate(sub, overheat, sig, C, O, H, L, 40, 0.08, 0.0, 0.0)
            print(f'{lbl:<8}{r["ret"]*100:>+8.1f}%{kr:>+8.1f}%{(r["ret"]*100-kr):>+8.1f}%p'
                  f'{r["winrate"]:>5.0f}%{r["mdd"]*100:>6.1f}%{r["avg_hold"]:>7.0f}일{r["entries"]:>6}')
        print()
    print("★ 판정: 매 구간(특히 2024 약세·2026 극강세) + 매 슬롯에서 KOSPI 초과 일관 = 진짜 강건.")
    print("  한 구간/슬롯만 좋고 나머지 음수 = 집중운/과최적화. 생존편향 미보정 가설 유지.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
