"""Phase 4: Q4 (기관 매수 강도 상위 25%) + 시총/거래대금 필터

Phase 3 발견:
- 기관 매수 강도 Q4: D+1 +0.82%, 적중률 47.4%, D+3 적중률 50.0%
- 단조 증가 패턴 명확 (Q1 → Q4)

새 필터:
1. inst_5d > Q4_threshold (예: 50억 추정)
2. 거래대금 100억원 이상 (대형주/ETF만)
3. 외인 음수 또는 미미 (fgn_5d <= +적은수)

목표: 적중률 50%+ + D+3 평균 +1.0%+

출력:
- data/backtest/phase4_filtered_signals.parquet
- data/backtest/phase4_report.md
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUT_DIR = PROJECT_ROOT / "data" / "backtest"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"


def load_avg_turnover(ticker: str, days: int = 5) -> float:
    """ticker의 최근 N일 평균 거래대금 (Close × Volume, 억원)"""
    matches = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not matches:
        return 0.0
    try:
        df = pd.read_csv(matches[0], encoding="utf-8-sig")
        df = df.tail(days)
        # 거래대금(억원) = Close × Volume / 1e8
        turnover_eok = (df["Close"] * df["Volume"]).mean() / 1e8
        return turnover_eok
    except Exception:
        return 0.0


def main():
    print("=" * 60)
    print("Phase 4: Q4 + 거래대금 필터 결합")
    print("=" * 60)

    df = pd.read_parquet(OUT_DIR / "phase3_new_signals.parquet")
    print(f"[load] phase3_new_signals {len(df)}행")

    # 1. Q4 임계값 측정 (inst_5d > 0인 종목의 75 percentile)
    inst_pos = df[df["inst_5d"] > 0]
    q4_threshold = inst_pos["inst_5d"].quantile(0.75)
    print(f"\n[Q4 임계값] inst_5d 상위 25%: {q4_threshold:.1f}억")
    print(f"  Q3 50%: {inst_pos['inst_5d'].quantile(0.50):.1f}억")
    print(f"  Q1 25%: {inst_pos['inst_5d'].quantile(0.25):.1f}억")

    # 2. 거래대금 계산 (캐싱)
    print("\n[turnover] 종목별 5일 평균 거래대금 계산...")
    unique_tickers = df["ticker"].astype(str).str.zfill(6).unique()
    turnover_map = {}
    for t in unique_tickers:
        turnover_map[t] = load_avg_turnover(t)
    print(f"  {len(turnover_map)}종목 거래대금 캐싱 완료")

    df["ticker_str"] = df["ticker"].astype(str).str.zfill(6)
    df["avg_turnover_eok"] = df["ticker_str"].map(turnover_map)
    print(f"  거래대금 분포: 중위 {df['avg_turnover_eok'].median():.1f}억, 75% {df['avg_turnover_eok'].quantile(0.75):.1f}억")

    # 3. 필터링
    # F1: Q4 (inst_5d >= q4_threshold)
    # F2: 거래대금 100억 이상 (대형주/ETF 유동성 기준)
    # F3: 외인 음수 또는 0 (기관 단독 매수에 가까운 조건)
    f1 = df[df["inst_5d"] >= q4_threshold]
    f1_f2 = f1[f1["avg_turnover_eok"] >= 100]
    f1_f2_f3 = f1_f2[f1_f2["fgn_5d"] <= 0]

    print(f"\n[필터 단계별 표본]")
    print(f"  전체: {len(df)}")
    print(f"  F1 (Q4 inst_5d>={q4_threshold:.0f}억): {len(f1)}")
    print(f"  F1+F2 (거래대금>=100억): {len(f1_f2)}")
    print(f"  F1+F2+F3 (외인 음수): {len(f1_f2_f3)}")

    # 4. 각 단계별 성과
    def measure(sub, label):
        print(f"\n--- {label} ({len(sub)}행) ---")
        for n in [1, 3, 5]:
            col = f"ret_d{n}"
            v = sub[col].dropna()
            if len(v) < 3:
                print(f"  D+{n}: 표본 부족")
                continue
            print(f"  D+{n}: 평균 {v.mean():+.2f}%, 적중률 {(v>0).mean()*100:.1f}%, n={len(v)}")
        return {
            "label": label,
            "n": len(sub),
            "d1_avg": sub["ret_d1"].dropna().mean() if "ret_d1" in sub else None,
            "d1_hit": (sub["ret_d1"].dropna() > 0).mean() * 100 if "ret_d1" in sub else None,
            "d3_avg": sub["ret_d3"].dropna().mean() if "ret_d3" in sub else None,
            "d3_hit": (sub["ret_d3"].dropna() > 0).mean() * 100 if "ret_d3" in sub else None,
        }

    results = []
    results.append(measure(df[df["buy_grade"] == "BUY"], "대조군 (기존 BUY)"))
    results.append(measure(df[df["new_simple_buy"] == 1], "Phase 3 new_simple"))
    results.append(measure(f1, "Phase 4 F1 (Q4)"))
    results.append(measure(f1_f2, "Phase 4 F1+F2 (Q4 + 유동성)"))
    results.append(measure(f1_f2_f3, "Phase 4 F1+F2+F3 (Q4 + 유동성 + 외인음수)"))

    # 5. 보고서
    out = OUT_DIR / "phase4_report.md"
    lines = [
        "# Phase 4: Q4 + 거래대금 필터 결합",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Q4 임계값**: inst_5d >= **{q4_threshold:.1f}억**",
        f"**거래대금 필터**: 5일 평균 100억원 이상",
        "",
        "## 결과 비교표",
        "",
        "| 단계 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 |",
        "|------|--------|---------|---------|---------|---------|",
    ]
    for r in results:
        if r["d1_avg"] is None or np.isnan(r["d1_avg"]):
            lines.append(f"| {r['label']} | {r['n']} | - | - | - | - |")
        else:
            d3a = f"{r['d3_avg']:+.2f}%" if r['d3_avg'] is not None and not np.isnan(r['d3_avg']) else "-"
            d3h = f"{r['d3_hit']:.1f}%" if r['d3_hit'] is not None and not np.isnan(r['d3_hit']) else "-"
            lines.append(
                f"| {r['label']} | {r['n']} | {r['d1_avg']:+.2f}% | {r['d1_hit']:.1f}% | {d3a} | {d3h} |"
            )

    lines += [
        "",
        "## 통과 종목 샘플 (F1+F2+F3, 최강 시그널)",
        "",
        "| 일자 | 종목 | 섹터 | inst_5d (억) | fgn_5d (억) | D+1 | D+3 | 거래대금 |",
        "|------|-----|------|-----------|----------|-----|-----|--------|",
    ]
    sample = f1_f2_f3.sort_values("date", ascending=False).head(20)
    for _, r in sample.iterrows():
        d1 = f"{r['ret_d1']:+.2f}%" if not pd.isna(r['ret_d1']) else "-"
        d3 = f"{r['ret_d3']:+.2f}%" if not pd.isna(r['ret_d3']) else "-"
        lines.append(
            f"| {r['date']} | {r['name']}({r['ticker_str']}) | {r['sector']} | "
            f"{r['inst_5d']:.0f} | {r['fgn_5d']:.0f} | {d1} | {d3} | {r['avg_turnover_eok']:.0f}억 |"
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    f1_f2_f3.to_parquet(OUT_DIR / "phase4_filtered_signals.parquet", index=False)
    print(f"\n[report] {out}")
    print(f"[save] phase4_filtered_signals.parquet ({len(f1_f2_f3)}행)")
    print("\n[OK] Phase 4 완료")


if __name__ == "__main__":
    main()
