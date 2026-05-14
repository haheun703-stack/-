"""Phase 7: ETF dual_buy 카테고리 × 종목 stage3 결합 시그널

가설:
- ETF에 외+기 둘 다 매수 (dual_buy) → 그 ETF 카테고리/섹터 강세 신호
- 같은 섹터/카테고리 종목 중 stage3 (금투+연기금+기타+기관+외인) = 추가 강세
- 결합 시 적중률 70%+ 가능성

데이터:
- Phase 5: phase5_3stage.parquet (종목 단위 stage 분류)
- Phase 6: phase6_etf_signals.parquet (ETF 단위 시그널)

분석:
1. ETF dual_buy 날짜+sector 추출
2. 같은 날짜+sector 종목 중 stage3 확인
3. 결합 시그널 D+N 수익률
4. 비교: 단독 stage3 vs ETF+stage3 결합

출력:
- data/backtest/phase7_combined.parquet
- data/backtest/phase7_report.md
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUT_DIR = PROJECT_ROOT / "data" / "backtest"


def main():
    print("=" * 60)
    print("Phase 7: ETF + 종목 결합 시그널")
    print("=" * 60)

    stocks = pd.read_parquet(OUT_DIR / "phase5_3stage.parquet")
    etfs = pd.read_parquet(OUT_DIR / "phase6_etf_signals.parquet")
    print(f"[load] 종목 {len(stocks)}행, ETF {len(etfs)}행")

    # ETF dual_buy 날짜+sector
    etf_buy = etfs[etfs["signal"] == "dual_buy"][["date", "sector", "category", "ticker", "name"]].copy()
    print(f"\n[ETF dual_buy] {len(etf_buy)}건")
    print(f"  카테고리 분포: {etf_buy['category'].value_counts().to_dict()}")
    print(f"  sector 분포 (상위 10): {etf_buy['sector'].value_counts().head(10).to_dict()}")

    # 종목 sector 분포
    print(f"\n[종목 sector 분포 (상위)] {stocks['sector'].value_counts().head(10).to_dict()}")

    # 두 sector 컬럼 교집합
    etf_sectors = set(etf_buy["sector"].dropna().unique())
    stock_sectors = set(stocks["sector"].dropna().unique())
    common = etf_sectors & stock_sectors
    print(f"\n[교집합 sector] {len(common)}개: {sorted(common)[:10]}")

    # ETF dual_buy 날짜+sector → 그 sector 종목 추출
    # 종목 측에서 매칭되는 행 찾기
    print(f"\n[결합] ETF dual_buy 날짜+sector × 종목 stage_class")

    # 1. 모든 종목 행에 ETF_dual_buy 여부 추가
    etf_lookup = set(zip(etf_buy["date"], etf_buy["sector"]))
    stocks["has_etf_dual_buy"] = stocks.apply(
        lambda r: (r["date"], r["sector"]) in etf_lookup, axis=1
    )

    # 2. 그룹별 성과
    def measure(sub, label):
        m = {"label": label, "n": len(sub)}
        for n in [1, 3, 5]:
            v = sub[f"ret_d{n}"].dropna()
            if len(v) >= 3:
                m[f"d{n}_avg"] = v.mean()
                m[f"d{n}_hit"] = (v > 0).mean() * 100
        return m

    print()
    print(f"{'그룹':<55} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10}")
    print("-" * 100)

    results = []
    # 단독 종목 stage3 (대조군)
    sub = stocks[stocks["signal_class"] == "stage3_full"]
    results.append(measure(sub, "단독 stage3 (대조군)"))
    # 단독 ETF dual_buy 동일 sector에서의 일반 종목 (etf 신호 있지만 stage 없음)
    sub = stocks[(stocks["has_etf_dual_buy"]) & (stocks["signal_class"] == "none")]
    results.append(measure(sub, "ETF dual_buy + 종목 시그널 없음"))
    # 결합: stage3 + ETF dual_buy
    sub = stocks[(stocks["has_etf_dual_buy"]) & (stocks["signal_class"] == "stage3_full")]
    results.append(measure(sub, "stage3 + ETF dual_buy ⭐"))
    # 결합: stage2 + ETF dual_buy
    sub = stocks[(stocks["has_etf_dual_buy"]) & (stocks["signal_class"] == "stage2_only")]
    results.append(measure(sub, "stage2 + ETF dual_buy"))
    # 결합: stage1 + ETF dual_buy
    sub = stocks[(stocks["has_etf_dual_buy"]) & (stocks["signal_class"] == "stage1_only")]
    results.append(measure(sub, "stage1 + ETF dual_buy"))

    for r in results:
        if r.get("d1_avg") is None:
            print(f"{r['label']:<55} {r['n']:<8} (표본 부족)")
            continue
        print(
            f"{r['label']:<55} {r['n']:<8} "
            f"{r['d1_avg']:+.2f}%   {r['d1_hit']:.1f}%      "
            f"{r.get('d3_avg', 0):+.2f}%"
        )

    stocks.to_parquet(OUT_DIR / "phase7_combined.parquet", index=False)

    # 보고서
    out = OUT_DIR / "phase7_report.md"
    lines = [
        "# Phase 7: ETF dual_buy × 종목 stage 결합 시그널",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**입력**: 종목 {len(stocks)}행 (Phase 5), ETF {len(etfs)}행 (Phase 6)",
        f"**ETF dual_buy**: {len(etf_buy)}건 (sector 교집합 {len(common)}개)",
        "",
        "## 결과",
        "",
        "| 그룹 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 |",
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
        "## 결합 시그널 통과 종목 (stage3 + ETF dual_buy)",
        "",
        "| 일자 | 종목 | 섹터 | 금투5d | 연5d | 기타5d | 기관5d | 외인5d | D+1 | D+3 |",
        "|------|-----|------|------|------|-------|-------|-------|-----|-----|",
    ]
    sample = stocks[(stocks["has_etf_dual_buy"]) & (stocks["signal_class"] == "stage3_full")].sort_values("date", ascending=False)
    for _, r in sample.iterrows():
        d1 = f"{r['ret_d1']:+.2f}%" if not pd.isna(r.get('ret_d1')) else "-"
        d3 = f"{r['ret_d3']:+.2f}%" if not pd.isna(r.get('ret_d3')) else "-"
        ticker_s = str(r['ticker']).zfill(6)
        lines.append(
            f"| {r['date']} | {r['name']}({ticker_s}) | {r.get('sector', '?')} | "
            f"{r.get('fin_inv_5d', 0):+.0f} | {r.get('pension_5d_db', 0):+.0f} | {r.get('corp_5d', 0):+.0f} | "
            f"{r.get('inst_5d_db', 0):+.0f} | {r.get('fgn_5d_db', 0):+.0f} | "
            f"{d1} | {d3} |"
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")
    print(f"\n[OK] Phase 7 완료")


if __name__ == "__main__":
    main()
