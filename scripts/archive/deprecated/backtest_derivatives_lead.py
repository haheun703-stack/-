"""파생 시그널 → 현물 선행성 검증 백테스트

과거 60일 ETF 데이터에서 매일 파생 시그널을 계산하고,
다음날 KOSPI200 수익률과의 상관관계를 분석한다.

검증 항목:
  1. 파생 composite score ↔ 다음날 KOSPI200 수익률 (Pearson/Spearman)
  2. P/C 프록시 ↔ 다음날 수익률 (역상관 기대)
  3. 레버리지 순유입 ↔ 다음날 수익률 (양상관 기대)
  4. 등급별 다음날 평균 수익률
  5. 방향 적중률

사용: python -u -X utf8 scripts/backtest_derivatives_lead.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# derivatives_collector의 함수 재활용
from scripts.derivatives_collector import (
    ETF_SYMBOLS,
    KOSPI200_SYMBOL,
    _compute_composite_score,
    _compute_futures_basis,
    _compute_leverage_flow,
    _compute_put_call_proxy,
)

import yfinance as yf

OUTPUT_PATH = PROJECT_ROOT / "data" / "derivatives" / "derivatives_lead_backtest.json"


def _fetch_historical(period: str = "6mo") -> dict[str, pd.DataFrame]:
    """6개월 ETF + KOSPI200 데이터 수집."""
    results = {}
    all_symbols = {**ETF_SYMBOLS, "KOSPI200_IDX": KOSPI200_SYMBOL}

    for name, sym in all_symbols.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period=period)
            if not h.empty:
                results[name] = h
        except Exception:
            pass

    return results


def _slice_data(data: dict, end_idx: int, window: int = 30) -> dict:
    """특정 시점까지의 데이터 슬라이스."""
    sliced = {}
    for name, df in data.items():
        start = max(0, end_idx - window)
        sliced[name] = df.iloc[start:end_idx + 1].copy()
    return sliced


def run_backtest():
    """파생 선행성 백테스트."""
    print("=== 파생 시그널 선행성 검증 백테스트 ===\n")

    # 데이터 수집
    print("ETF 데이터 수집 중 (6개월)...")
    data = _fetch_historical("6mo")
    print(f"수집 완료: {list(data.keys())}\n")

    kospi = data.get("KOSPI200_IDX")
    if kospi is None or len(kospi) < 30:
        print("ERROR: KOSPI200 데이터 부족")
        return

    # 공통 날짜 기준으로 정렬
    spot = data.get("KODEX200")
    if spot is None:
        print("ERROR: KODEX200 데이터 없음")
        return

    common_dates = kospi.index.intersection(spot.index)
    if len(common_dates) < 30:
        print("ERROR: 공통 날짜 부족")
        return

    # 매일 시뮬레이션
    records = []
    start_day = 25  # 20일 롤링 통계 + 여유

    for i in range(start_day, len(common_dates) - 1):
        date = common_dates[i]
        next_date = common_dates[i + 1]

        # 해당 날짜까지의 데이터로 파생 시그널 계산
        sliced = _slice_data(data, i, window=30)

        basis = _compute_futures_basis(sliced)
        pc = _compute_put_call_proxy(sliced)
        flow = _compute_leverage_flow(sliced)
        composite = _compute_composite_score(basis, pc, flow)

        # 다음날 KOSPI200 수익률
        kospi_today = kospi.loc[date, "Close"]
        kospi_next = kospi.loc[next_date, "Close"]
        next_ret = (kospi_next / kospi_today - 1)

        records.append({
            "date": date,
            "composite_score": composite["score"],
            "composite_grade": composite["grade"],
            "pc_ratio": pc.get("pc_ratio", 0.5),
            "basis_5d": basis.get("basis_5d_avg", 0),
            "net_flow_5d": flow.get("net_flow_5d_억", 0),
            "kospi_next_ret": next_ret,
        })

    rdf = pd.DataFrame(records).set_index("date")
    print(f"분석 기간: {rdf.index[0].date()} ~ {rdf.index[-1].date()} ({len(rdf)}일)\n")

    # ── 1. 상관관계 분석 ──
    print("=" * 60)
    print("1. 상관관계 (파생 시그널 vs 다음날 KOSPI200 수익률)")
    print("=" * 60)

    metrics = [
        ("composite_score", "종합 스코어"),
        ("pc_ratio", "P/C 프록시 (역상관 기대)"),
        ("basis_5d", "베이시스 5D"),
        ("net_flow_5d", "순유입 5D"),
    ]

    corr_results = {}
    for col, label in metrics:
        valid = rdf[[col, "kospi_next_ret"]].dropna()
        if len(valid) < 10:
            continue

        p_corr, p_pval = pearsonr(valid[col], valid["kospi_next_ret"])
        s_corr, s_pval = spearmanr(valid[col], valid["kospi_next_ret"])

        sig_p = "***" if p_pval < 0.01 else "**" if p_pval < 0.05 else "*" if p_pval < 0.1 else ""
        sig_s = "***" if s_pval < 0.01 else "**" if s_pval < 0.05 else "*" if s_pval < 0.1 else ""

        print(f"  {label:<25} Pearson: {p_corr:+.4f}{sig_p:>4}  Spearman: {s_corr:+.4f}{sig_s:>4}")

        corr_results[col] = {
            "pearson": round(p_corr, 4),
            "pearson_pval": round(p_pval, 4),
            "spearman": round(s_corr, 4),
            "spearman_pval": round(s_pval, 4),
        }

    # ── 2. 등급별 다음날 평균 수익률 ──
    print(f"\n{'=' * 60}")
    print("2. 등급별 다음날 평균 KOSPI200 수익률")
    print("=" * 60)

    grade_order = ["STRONG_BULL", "MILD_BULL", "NEUTRAL", "MILD_BEAR", "STRONG_BEAR"]
    grade_results = {}

    for grade in grade_order:
        mask = rdf["composite_grade"] == grade
        count = mask.sum()
        if count == 0:
            continue

        avg_ret = rdf.loc[mask, "kospi_next_ret"].mean() * 100
        win_rate = (rdf.loc[mask, "kospi_next_ret"] > 0).mean() * 100

        print(f"  {grade:<15} 발생: {count:>3}일  평균수익: {avg_ret:>+6.2f}%  승률: {win_rate:>5.1f}%")

        grade_results[grade] = {
            "count": int(count),
            "avg_return_pct": round(avg_ret, 2),
            "win_rate": round(win_rate, 1),
        }

    # ── 3. 방향 적중률 ──
    print(f"\n{'=' * 60}")
    print("3. 방향 적중률")
    print("=" * 60)

    bull_mask = rdf["composite_grade"].isin(["STRONG_BULL", "MILD_BULL"])
    bear_mask = rdf["composite_grade"].isin(["STRONG_BEAR", "MILD_BEAR"])

    bull_correct = (rdf.loc[bull_mask, "kospi_next_ret"] > 0).sum() if bull_mask.any() else 0
    bull_total = bull_mask.sum()
    bear_correct = (rdf.loc[bear_mask, "kospi_next_ret"] < 0).sum() if bear_mask.any() else 0
    bear_total = bear_mask.sum()

    total_correct = bull_correct + bear_correct
    total_signal = bull_total + bear_total

    if total_signal > 0:
        overall_acc = total_correct / total_signal * 100
    else:
        overall_acc = 0

    print(f"  BULL 방향: {bull_correct}/{bull_total} ({bull_correct/bull_total*100:.1f}%)" if bull_total > 0 else "  BULL 방향: 없음")
    print(f"  BEAR 방향: {bear_correct}/{bear_total} ({bear_correct/bear_total*100:.1f}%)" if bear_total > 0 else "  BEAR 방향: 없음")
    print(f"  전체 방향: {total_correct}/{total_signal} ({overall_acc:.1f}%)" if total_signal > 0 else "  전체 방향: 시그널 없음")

    # ── 4. P/C 비율 극단치 반전 검증 ──
    print(f"\n{'=' * 60}")
    print("4. P/C 극단치 반전 분석")
    print("=" * 60)

    pc_high = rdf["pc_ratio"] > 0.50  # 극도 약세 → 반등 기대
    pc_low = rdf["pc_ratio"] < 0.25   # 극도 강세 → 조정 기대

    if pc_high.any():
        high_ret = rdf.loc[pc_high, "kospi_next_ret"].mean() * 100
        high_win = (rdf.loc[pc_high, "kospi_next_ret"] > 0).mean() * 100
        print(f"  P/C > 0.50 (극약세): {pc_high.sum()}일, 다음날 평균 {high_ret:+.2f}%, 반등률 {high_win:.0f}%")
    else:
        print("  P/C > 0.50: 발생 없음")

    if pc_low.any():
        low_ret = rdf.loc[pc_low, "kospi_next_ret"].mean() * 100
        low_win = (rdf.loc[pc_low, "kospi_next_ret"] < 0).mean() * 100
        print(f"  P/C < 0.25 (극강세): {pc_low.sum()}일, 다음날 평균 {low_ret:+.2f}%, 조정률 {low_win:.0f}%")
    else:
        print("  P/C < 0.25: 발생 없음")

    # ── 결과 저장 ──
    result = {
        "period": f"{rdf.index[0].date()} ~ {rdf.index[-1].date()}",
        "total_days": len(rdf),
        "correlations": corr_results,
        "grade_performance": grade_results,
        "direction_accuracy": {
            "bull": {"correct": int(bull_correct), "total": int(bull_total)},
            "bear": {"correct": int(bear_correct), "total": int(bear_total)},
            "overall_pct": round(overall_acc, 1),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_backtest()
