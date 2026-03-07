"""JARVIS 합산 포트폴리오 백테스트 — v10.3 개별주 60% + ETF 3축 40%.

두 전략을 독립 실행한 뒤 일별 수익률을 60:40 가중 합산하여
통합 포트폴리오의 성과를 검증한다.

사용법:
  python -u -X utf8 scripts/backtest_combined_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── 1. v10.3 개별주 백테스트 (C_new 모드) ───

def run_v10_backtest() -> pd.DataFrame:
    """v10.3 C_new 모드 백테스트 실행 → 일별 equity DataFrame 반환."""
    from scripts.backtest_v2 import (
        load_parquets, load_name_map, load_kospi_index,
        run_backtest as run_v10, DayResult,
    )

    data_dict = load_parquets()
    name_map = load_name_map()
    kospi_df = load_kospi_index()
    print(f"  v10.3: {len(data_dict)}종목 로드")

    trades, daily_results = run_v10(data_dict, name_map, mode="C_new", kospi_df=kospi_df)

    # DayResult → DataFrame
    records = [{"date": d.date, "equity": d.equity, "cash": d.cash,
                "positions": d.positions} for d in daily_results]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 매매 통계
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    total_ret = (df["equity"].iloc[-1] / 100_000_000 - 1) * 100

    eq = df["equity"].values
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak * 100).min()

    daily_ret = df["equity"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    return df, {
        "total_return": round(total_ret, 2),
        "mdd": round(mdd, 2),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
    }


# ─── 2. ETF 3축 백테스트 ───

def run_etf_backtest(start: str, end: str) -> pd.DataFrame:
    """ETF 3축 로테이션 백테스트 실행 → 일별 equity DataFrame 반환."""
    from scripts.backtest_etf_rotation import run_backtest as run_etf

    results = run_etf(start_date=start, end_date=end)
    df = results["daily_values"].copy()

    return df, {
        "total_return": results["total_return_pct"],
        "mdd": results["mdd_pct"],
        "pf": results["profit_factor"],
        "sharpe": results["sharpe_ratio"],
        "trades": results["total_trades"],
        "win_rate": results["win_rate_pct"],
    }


# ─── 3. KOSPI 벤치마크 ───

def load_kospi_benchmark(start: str, end: str) -> pd.Series:
    """KOSPI 일별 종가 → 수익률 시리즈."""
    kospi_path = PROJECT_ROOT / "data" / "kospi_index.csv"
    df = pd.read_csv(kospi_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask, "close"]


# ─── 4. 합산 포트폴리오 구성 ───

def build_combined_portfolio(
    v10_equity: pd.Series,
    etf_equity: pd.Series,
    w_v10: float = 0.6,
    w_etf: float = 0.4,
) -> pd.DataFrame:
    """두 전략의 일별 수익률을 가중 합산하여 통합 포트폴리오 구성.

    Args:
        v10_equity: v10.3 일별 자산가치
        etf_equity: ETF 3축 일별 자산가치
        w_v10: v10.3 비중 (기본 60%)
        w_etf: ETF 3축 비중 (기본 40%)
    """
    # 일별 수익률
    v10_ret = v10_equity.pct_change().fillna(0)
    etf_ret = etf_equity.pct_change().fillna(0)

    # 공통 거래일만
    common_dates = v10_ret.index.intersection(etf_ret.index)
    v10_ret = v10_ret.loc[common_dates]
    etf_ret = etf_ret.loc[common_dates]

    # 가중 합산 수익률
    combined_ret = v10_ret * w_v10 + etf_ret * w_etf

    # 초기 자본 1억 기준 자산 곡선
    initial = 100_000_000
    combined_equity = initial * (1 + combined_ret).cumprod()

    df = pd.DataFrame({
        "combined_equity": combined_equity,
        "combined_return": combined_ret,
        "v10_return": v10_ret,
        "etf_return": etf_ret,
        "v10_equity_norm": initial * w_v10 * (1 + v10_ret).cumprod(),
        "etf_equity_norm": initial * w_etf * (1 + etf_ret).cumprod(),
    })

    return df


def calc_metrics(equity: pd.Series, label: str = "") -> dict:
    """자산 곡선에서 성과 지표 계산."""
    initial = equity.iloc[0]
    final = equity.iloc[-1]
    total_ret = (final / initial - 1) * 100

    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    mdd = dd.min()

    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # 연환산
    days = len(equity)
    annual = (1 + total_ret / 100) ** (252 / max(days, 1)) - 1

    # Calmar Ratio (연환산 수익률 / |MDD|)
    calmar = (annual * 100) / abs(mdd) if mdd != 0 else 0

    # Sortino Ratio
    neg_ret = daily_ret[daily_ret < 0]
    downside_std = neg_ret.std() if len(neg_ret) > 0 else daily_ret.std()
    sortino = (daily_ret.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    return {
        "label": label,
        "total_return": round(total_ret, 2),
        "annual_return": round(annual * 100, 2),
        "mdd": round(mdd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "daily_vol": round(daily_ret.std() * np.sqrt(252) * 100, 2),
    }


def drawdown_comparison(
    combined: pd.Series,
    v10: pd.Series,
    etf: pd.Series,
    kospi: pd.Series,
):
    """드로다운 비교 (Peak-to-Trough)."""
    def max_dd(s):
        peak = s.cummax()
        return ((s - peak) / peak * 100).min()

    print(f"\n  {'전략':>20} | {'MDD':>8} | {'방어 등급'}")
    print(f"  {'-'*20}+{'-'*10}+{'-'*12}")

    items = [
        ("KOSPI B&H", kospi),
        ("v10.3 개별주", v10),
        ("ETF 3축", etf),
        ("합산 60:40", combined),
    ]

    mdds = {}
    for name, series in items:
        mdd = max_dd(series)
        mdds[name] = mdd

    kospi_mdd = mdds["KOSPI B&H"]
    for name, mdd in mdds.items():
        if name == "KOSPI B&H":
            grade = "기준"
        elif mdd > kospi_mdd * 0.5:
            grade = "★★★ 최우수"
        elif mdd > kospi_mdd * 0.7:
            grade = "★★ 우수"
        elif mdd > kospi_mdd:
            grade = "★ 양호"
        else:
            grade = "- 미달"
        print(f"  {name:>20} | {mdd:>+7.2f}% | {grade}")


def correlation_analysis(df: pd.DataFrame):
    """v10.3 vs ETF 상관관계 분석."""
    corr = df["v10_return"].corr(df["etf_return"])

    # 하락일 상관 (둘 다 마이너스인 날)
    down_both = df[(df["v10_return"] < 0) & (df["etf_return"] < 0)]
    up_v10_down_etf = df[(df["v10_return"] > 0) & (df["etf_return"] < 0)]
    down_v10_up_etf = df[(df["v10_return"] < 0) & (df["etf_return"] > 0)]

    total_days = len(df)
    print(f"\n  ■ 상관관계 분석:")
    print(f"    일별 수익률 상관계수: {corr:.3f}")
    print(f"    동시 하락일: {len(down_both)}일 ({len(down_both)/total_days*100:.1f}%)")
    print(f"    v10↑ ETF↓: {len(up_v10_down_etf)}일 (분산 효과)")
    print(f"    v10↓ ETF↑: {len(down_v10_up_etf)}일 (분산 효과)")
    print(f"    분산 효과 일수: {len(up_v10_down_etf) + len(down_v10_up_etf)}일 "
          f"({(len(up_v10_down_etf) + len(down_v10_up_etf))/total_days*100:.1f}%)")

    # 상관계수 해석
    if corr < 0.3:
        interp = "낮은 상관 → 분산 효과 극대화"
    elif corr < 0.6:
        interp = "중간 상관 → 적절한 분산 효과"
    elif corr < 0.8:
        interp = "높은 상관 → 제한적 분산 효과"
    else:
        interp = "매우 높은 상관 → 분산 효과 미미"
    print(f"    해석: {interp}")


def monthly_comparison(
    combined_eq: pd.Series,
    v10_eq: pd.Series,
    etf_eq: pd.Series,
    kospi_close: pd.Series,
):
    """월별 수익률 비교."""
    def monthly_ret(s):
        m = s.resample("ME").last()
        return m.pct_change().fillna((m.iloc[0] / s.iloc[0]) - 1) * 100

    c_m = monthly_ret(combined_eq)
    v_m = monthly_ret(v10_eq)
    e_m = monthly_ret(etf_eq)
    k_m = monthly_ret(kospi_close)

    print(f"\n  {'월':>8} | {'KOSPI':>8} | {'v10.3':>8} | {'ETF3축':>8} | {'합산60:40':>10} | {'vs KOSPI':>9}")
    print(f"  {'-'*8}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*12}+{'-'*11}")

    common_months = c_m.index.intersection(k_m.index)
    beat_count = 0
    total_months = 0

    for month in common_months:
        k = k_m.get(month, 0)
        v = v_m.get(month, 0)
        e = e_m.get(month, 0)
        c = c_m.get(month, 0)
        vs = c - k
        beat = "✓" if vs > 0 else "✗"
        if vs > 0:
            beat_count += 1
        total_months += 1

        print(f"  {month.strftime('%Y-%m'):>8} | {k:>+7.1f}% | {v:>+7.1f}% | {e:>+7.1f}% | "
              f"{c:>+9.1f}% | {vs:>+8.1f}%p {beat}")

    if total_months > 0:
        print(f"\n  KOSPI 대비 초과수익 월: {beat_count}/{total_months}개월 "
              f"({beat_count/total_months*100:.0f}%)")


def scenario_test(v10_ret: pd.Series, etf_ret: pd.Series):
    """다양한 비중 시나리오 테스트."""
    scenarios = [
        (1.0, 0.0, "100:0 (v10.3 단독)"),
        (0.8, 0.2, "80:20"),
        (0.7, 0.3, "70:30"),
        (0.6, 0.4, "60:40 (기본)"),
        (0.5, 0.5, "50:50"),
        (0.4, 0.6, "40:60"),
        (0.0, 1.0, "0:100 (ETF 단독)"),
    ]

    initial = 100_000_000
    print(f"\n  {'비중':>20} | {'수익률':>8} | {'MDD':>8} | {'Sharpe':>7} | {'Calmar':>7} | {'연변동성':>8}")
    print(f"  {'-'*20}+{'-'*10}+{'-'*10}+{'-'*9}+{'-'*9}+{'-'*10}")

    best_calmar = -999
    best_label = ""

    for w_v, w_e, label in scenarios:
        combined = v10_ret * w_v + etf_ret * w_e
        eq = initial * (1 + combined).cumprod()
        m = calc_metrics(eq, label)

        if m["calmar"] > best_calmar:
            best_calmar = m["calmar"]
            best_label = label

        marker = " ★" if label == "60:40 (기본)" else ""
        print(f"  {label:>20} | {m['total_return']:>+7.1f}% | {m['mdd']:>+7.2f}% | "
              f"{m['sharpe']:>6.2f} | {m['calmar']:>6.2f} | {m['daily_vol']:>7.1f}%{marker}")

    print(f"\n  최적 Calmar Ratio: {best_label} ({best_calmar:.2f})")


def main():
    print("=" * 70)
    print("  JARVIS 합산 포트폴리오 백테스트")
    print("  v10.3 개별주 60% + ETF 3축 40%")
    print("=" * 70)

    # ── 기간 설정 ──
    # v10.3: 2025-03-01 ~ 2026-02-13
    # ETF:   2025-06-01 ~ 2026-02-27
    # 공통:  2025-06-01 ~ 2026-02-13
    OVERLAP_START = "2025-06-01"
    OVERLAP_END = "2026-02-13"
    print(f"\n  공통 백테스트 기간: {OVERLAP_START} ~ {OVERLAP_END}")

    # ── 1. v10.3 개별주 백테스트 ──
    print(f"\n{'─'*70}")
    print(f"  [1/4] v10.3 개별주 백테스트 (C_new 모드)")
    print(f"{'─'*70}")
    v10_df, v10_stats = run_v10_backtest()
    # 공통 기간 필터
    v10_df = v10_df[(v10_df.index >= OVERLAP_START) & (v10_df.index <= OVERLAP_END)]
    v10_equity = v10_df["equity"]
    print(f"\n  v10.3 기간 내 거래일: {len(v10_equity)}일")

    # ── 2. ETF 3축 백테스트 ──
    print(f"\n{'─'*70}")
    print(f"  [2/4] ETF 3축 로테이션 백테스트")
    print(f"{'─'*70}")
    etf_df, etf_stats = run_etf_backtest(OVERLAP_START, OVERLAP_END)
    etf_equity = etf_df["total_value"]
    # 공통 기간 필터
    etf_equity = etf_equity[(etf_equity.index >= OVERLAP_START) & (etf_equity.index <= OVERLAP_END)]
    print(f"\n  ETF 3축 기간 내 거래일: {len(etf_equity)}일")

    # ── 3. 합산 포트폴리오 구성 ──
    print(f"\n{'─'*70}")
    print(f"  [3/4] 합산 포트폴리오 구성 (60:40)")
    print(f"{'─'*70}")
    combined_df = build_combined_portfolio(v10_equity, etf_equity)
    print(f"  공통 거래일: {len(combined_df)}일")

    # ── 4. KOSPI 벤치마크 ──
    kospi_close = load_kospi_benchmark(OVERLAP_START, OVERLAP_END)
    # 정규화 (초기값 = 1억)
    kospi_norm = kospi_close / kospi_close.iloc[0] * 100_000_000

    # ── 성과 지표 계산 ──
    print(f"\n{'='*70}")
    print(f"  성과 지표 비교")
    print(f"{'='*70}")

    metrics = {
        "KOSPI B&H": calc_metrics(kospi_norm, "KOSPI B&H"),
        "v10.3 개별주": calc_metrics(v10_equity, "v10.3 개별주"),
        "ETF 3축": calc_metrics(etf_equity, "ETF 3축"),
        "합산 60:40": calc_metrics(combined_df["combined_equity"], "합산 60:40"),
    }

    print(f"\n  {'전략':>15} | {'수익률':>8} | {'연환산':>8} | {'MDD':>8} | {'Sharpe':>7} | "
          f"{'Sortino':>8} | {'Calmar':>7} | {'변동성':>7}")
    print(f"  {'-'*15}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*9}+{'-'*10}+{'-'*9}+{'-'*9}")

    for name, m in metrics.items():
        print(f"  {name:>15} | {m['total_return']:>+7.1f}% | {m['annual_return']:>+7.1f}% | "
              f"{m['mdd']:>+7.2f}% | {m['sharpe']:>6.2f} | {m['sortino']:>7.2f} | "
              f"{m['calmar']:>6.2f} | {m['daily_vol']:>6.1f}%")

    # ── 합산 포트폴리오 핵심 분석 ──
    print(f"\n{'='*70}")
    print(f"  합산 포트폴리오 핵심 분석")
    print(f"{'='*70}")

    cm = metrics["합산 60:40"]
    km = metrics["KOSPI B&H"]
    vm = metrics["v10.3 개별주"]
    em = metrics["ETF 3축"]

    # 초과수익 (알파)
    alpha = cm["total_return"] - km["total_return"]
    print(f"\n  ■ KOSPI 대비 초과수익 (알파): {alpha:+.2f}%p")

    # MDD 개선
    mdd_improve = cm["mdd"] - km["mdd"]
    print(f"  ■ KOSPI 대비 MDD 개선: {mdd_improve:+.2f}%p")

    # 분산 효과: 합산 MDD가 개별 MDD보다 좋은가?
    best_individual_mdd = max(vm["mdd"], em["mdd"])  # 음수이므로 max가 좋은 것
    diversification_mdd = cm["mdd"] - best_individual_mdd
    print(f"  ■ 분산 효과 (MDD): {diversification_mdd:+.2f}%p "
          f"({'개선' if diversification_mdd > 0 else '악화'} vs 최선 개별전략)")

    # 리스크 조정 수익률
    print(f"  ■ Sharpe 비교: KOSPI {km['sharpe']:.2f} → 합산 {cm['sharpe']:.2f} "
          f"({cm['sharpe']/km['sharpe']:.1f}x)" if km['sharpe'] > 0 else "")

    # ── 상관관계 분석 ──
    print(f"\n{'='*70}")
    print(f"  v10.3 vs ETF 3축 상관관계")
    print(f"{'='*70}")
    correlation_analysis(combined_df)

    # ── 드로다운 비교 ──
    print(f"\n{'='*70}")
    print(f"  드로다운 비교")
    print(f"{'='*70}")
    drawdown_comparison(
        combined_df["combined_equity"],
        v10_equity,
        etf_equity,
        kospi_norm,
    )

    # ── 월별 수익률 비교 ──
    print(f"\n{'='*70}")
    print(f"  월별 수익률 비교")
    print(f"{'='*70}")
    monthly_comparison(
        combined_df["combined_equity"],
        v10_equity,
        etf_equity,
        kospi_close,
    )

    # ── 비중 시나리오 테스트 ──
    print(f"\n{'='*70}")
    print(f"  비중 시나리오 테스트 (v10.3:ETF)")
    print(f"{'='*70}")
    common_dates = combined_df.index
    v10_ret = v10_equity.pct_change().fillna(0).loc[common_dates]
    etf_ret = etf_equity.pct_change().fillna(0).loc[common_dates]
    scenario_test(v10_ret, etf_ret)

    # ── 최종 결론 ──
    print(f"\n{'='*70}")
    print(f"  최종 결론")
    print(f"{'='*70}")

    tests = []

    # T1: 합산 MDD < KOSPI MDD
    t1 = cm["mdd"] > km["mdd"]
    tests.append(("합산 MDD < KOSPI MDD", t1,
                   f"합산 {cm['mdd']:.2f}% vs KOSPI {km['mdd']:.2f}%"))

    # T2: 합산 Sharpe > KOSPI Sharpe
    t2 = cm["sharpe"] > km["sharpe"]
    tests.append(("합산 Sharpe > KOSPI Sharpe", t2,
                   f"합산 {cm['sharpe']:.2f} vs KOSPI {km['sharpe']:.2f}"))

    # T3: 합산 Calmar > 2.0
    t3 = cm["calmar"] > 2.0
    tests.append(("합산 Calmar > 2.0", t3, f"합산 {cm['calmar']:.2f}"))

    # T4: 분산 효과 존재 (합산 MDD > 최악 개별 MDD)
    worst_mdd = min(vm["mdd"], em["mdd"])
    t4 = cm["mdd"] > worst_mdd
    tests.append(("분산 효과 존재", t4,
                   f"합산 {cm['mdd']:.2f}% > 최악 {worst_mdd:.2f}%"))

    # T5: 합산 수익률 > 0
    t5 = cm["total_return"] > 0
    tests.append(("합산 수익률 > 0%", t5, f"{cm['total_return']:+.2f}%"))

    passed = sum(1 for _, t, _ in tests if t)
    for name, result, detail in tests:
        print(f"  {'✓' if result else '✗'} {name}: {detail}")

    print(f"\n  종합: {passed}/{len(tests)} PASS")

    if passed >= 4:
        print(f"\n  ✓ JARVIS 합산 포트폴리오 — 실전 투입 적합")
        print(f"    공격수(v10.3) + 수비수(ETF 3축) 시너지 입증")
    elif passed >= 3:
        print(f"\n  △ 부분 검증 — 추가 관찰 필요")
    else:
        print(f"\n  ✗ 검증 미달 — 비중 조정 또는 전략 개선 필요")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
