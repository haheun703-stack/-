"""Phase 8: ETF 정밀 시그널 (거래대금 + streak + 카테고리)

Phase 6 발견:
- dual_buy 적중률 65.4% (전체)
- theme 카테고리 88.9% (강력)
- 표본 작아서 신뢰도 보강 필요

Phase 8 정밀 분석:
1. 외인+기관 순매수 합계 절대값 (≈ 자금 유입 강도)
2. foreign_streak ≥ 3 AND institution_streak ≥ 3 (연속 매수)
3. 카테고리별 (theme/direction/sector/group/global/bond_commodity)

조합 후보:
- Tier 1: 가장 엄격 (dual_buy + 큰 자금 + 연속 매수 + theme 카테고리)
- Tier 2: 중간 (dual_buy + 큰 자금)
- Tier 3: 기본 (dual_buy만)

출력:
- data/backtest/phase8_etf_precise.parquet
- data/backtest/phase8_report.md
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUT_DIR = PROJECT_ROOT / "data" / "backtest"


def measure(sub, label):
    m = {"label": label, "n": len(sub)}
    for n in [1, 3, 5]:
        v = sub[f"ret_d{n}"].dropna()
        if len(v) >= 3:
            m[f"d{n}_avg"] = v.mean()
            m[f"d{n}_hit"] = (v > 0).mean() * 100
            m[f"d{n}_n"] = len(v)
    return m


def main():
    print("=" * 60)
    print("Phase 8: ETF 정밀 시그널 (거래대금 + streak + 카테고리)")
    print("=" * 60)

    df = pd.read_parquet(OUT_DIR / "phase6_etf_signals.parquet")
    print(f"[load] {len(df)}행")

    # 5일 누적 외+기 (이미 phase6에 fgn_5d, inst_5d 있음)
    # 자금 유입 강도 = abs(fgn_5d) + abs(inst_5d) (외인+기관 활동 총량)
    df["flow_intensity"] = df["fgn_5d"].abs() + df["inst_5d"].abs()
    df["combined_flow"] = df["fgn_5d"] + df["inst_5d"]  # 순매수 합계

    print(f"\n[flow_intensity 분위] 25%={df['flow_intensity'].quantile(0.25):.0f}, "
          f"50%={df['flow_intensity'].quantile(0.50):.0f}, "
          f"75%={df['flow_intensity'].quantile(0.75):.0f}")
    print(f"[foreign_streak] max={df['foreign_streak'].max()}, "
          f"50%={df['foreign_streak'].quantile(0.50):.0f}")
    print(f"[institution_streak] max={df['institution_streak'].max()}, "
          f"50%={df['institution_streak'].quantile(0.50):.0f}")

    flow_q75 = df["flow_intensity"].quantile(0.75)
    print(f"\n[Q75 임계값] flow_intensity >= {flow_q75:.0f}억")

    # 시그널 정의
    df["dual_buy"] = (df["signal"] == "dual_buy").astype(int)
    df["big_flow"] = (df["flow_intensity"] >= flow_q75).astype(int)
    df["long_streak"] = ((df["foreign_streak"] >= 3) & (df["institution_streak"] >= 3)).astype(int)
    df["theme_cat"] = (df["category"] == "theme").astype(int)
    df["leverage_cat"] = df["sector"].fillna("").str.contains("레버리지").astype(int)

    # Tier 분류
    df["tier"] = "none"
    df.loc[df["dual_buy"] == 1, "tier"] = "T3_dual_buy"
    df.loc[(df["dual_buy"] == 1) & (df["big_flow"] == 1), "tier"] = "T2_big_flow"
    df.loc[(df["dual_buy"] == 1) & (df["big_flow"] == 1) & (df["theme_cat"] == 1), "tier"] = "T1_theme"
    df.loc[(df["dual_buy"] == 1) & (df["long_streak"] == 1), "tier"] = "T1_streak"

    df.to_parquet(OUT_DIR / "phase8_etf_precise.parquet", index=False)

    # 측정
    results = []
    for tier_label in ["T3_dual_buy", "T2_big_flow", "T1_theme", "T1_streak"]:
        sub = df[df["tier"].str.startswith(tier_label.split("_")[0])]  # T1 == T1_*
        # 위 방식은 T1_theme이 T1로 잡혀서 모든 T1 합쳐버림. 다시
    results = []
    results.append(measure(df[df["dual_buy"] == 1], "T3_dual_buy (전체 dual_buy)"))
    results.append(measure(df[(df["dual_buy"] == 1) & (df["big_flow"] == 1)], "T2_big_flow (dual_buy + 자금 Q75+)"))
    results.append(measure(df[(df["dual_buy"] == 1) & (df["theme_cat"] == 1)], "T1_theme (dual_buy + theme)"))
    results.append(measure(df[(df["dual_buy"] == 1) & (df["long_streak"] == 1)], "T1_streak (dual_buy + 연속 매수 3일+)"))
    results.append(measure(df[(df["dual_buy"] == 1) & (df["big_flow"] == 1) & (df["theme_cat"] == 1)], "T0_strongest (dual_buy + 자금 + theme)"))
    results.append(measure(df[(df["dual_buy"] == 1) & (df["big_flow"] == 1) & (df["long_streak"] == 1)], "T0_strongest2 (dual_buy + 자금 + 연속)"))

    print()
    print(f"{'Tier':<55} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10} {'D+3 hit':<10}")
    print("-" * 100)
    for r in results:
        if r.get("d1_avg") is None:
            print(f"{r['label']:<55} {r['n']:<8} (표본 부족)")
            continue
        d3a = f"{r.get('d3_avg', 0):+.2f}%" if "d3_avg" in r else "-"
        d3h = f"{r.get('d3_hit', 0):.1f}%" if "d3_hit" in r else "-"
        print(
            f"{r['label']:<55} {r['n']:<8} "
            f"{r['d1_avg']:+.2f}%   {r['d1_hit']:.1f}%      "
            f"{d3a}      {d3h}"
        )

    # 카테고리별 dual_buy 상세
    print()
    print("=== 카테고리별 dual_buy (상세) ===")
    print(f"{'카테고리':<20} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10}")
    print("-" * 60)
    for cat in df["category"].dropna().unique():
        sub = df[(df["category"] == cat) & (df["dual_buy"] == 1)]
        m = measure(sub, cat)
        if m.get("d1_avg") is None:
            print(f"{cat:<20} {m['n']:<8} (표본 부족)")
            continue
        d3a = f"{m.get('d3_avg', 0):+.2f}%" if "d3_avg" in m else "-"
        print(f"{cat:<20} {m['n']:<8} {m['d1_avg']:+.2f}%   {m['d1_hit']:.1f}%      {d3a}")

    # 보고서
    out = OUT_DIR / "phase8_report.md"
    lines = [
        "# Phase 8: ETF 정밀 시그널 (거래대금 + streak + 카테고리)",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**입력**: {len(df)}건",
        f"**flow_intensity Q75 임계값**: {flow_q75:.0f}억",
        "",
        "## Tier 별 시그널 결과",
        "",
        "| Tier | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 |",
        "|------|--------|---------|---------|---------|---------|",
    ]
    for r in results:
        if r.get("d1_avg") is None:
            lines.append(f"| {r['label']} | {r['n']} | - | - | - | - |")
            continue
        d3a = f"{r.get('d3_avg', 0):+.2f}%" if "d3_avg" in r else "-"
        d3h = f"{r.get('d3_hit', 0):.1f}%" if "d3_hit" in r else "-"
        lines.append(
            f"| {r['label']} | {r['n']} | "
            f"{r['d1_avg']:+.2f}% | {r['d1_hit']:.1f}% | {d3a} | {d3h} |"
        )

    lines += [
        "",
        "## 카테고리별 dual_buy",
        "",
        "| 카테고리 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 |",
        "|---------|--------|---------|---------|---------|",
    ]
    for cat in sorted(df["category"].dropna().unique()):
        sub = df[(df["category"] == cat) & (df["dual_buy"] == 1)]
        m = measure(sub, cat)
        if m.get("d1_avg") is None:
            continue
        d3a = f"{m.get('d3_avg', 0):+.2f}%" if "d3_avg" in m else "-"
        lines.append(f"| {cat} | {m['n']} | {m['d1_avg']:+.2f}% | {m['d1_hit']:.1f}% | {d3a} |")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")
    print("\n[OK] Phase 8 완료")


if __name__ == "__main__":
    main()
