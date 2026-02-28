"""KOSPI 조정 구간 드로다운 방어력 심층 분석.

Peak-to-trough 기반으로 KOSPI 하락 구간을 정확히 식별하고,
해당 구간에서 ETF 3축 전략이 KOSPI 대비 얼마나 방어했는지 검증.

사용법:
  python -u -X utf8 scripts/analyze_drawdown_defense.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_etf_rotation import (
    load_all_etf_data, load_kospi, run_backtest,
)


def find_drawdown_periods(prices: pd.Series, min_dd_pct: float = -3.0) -> list[dict]:
    """Peak-to-trough 드로다운 구간 식별.

    연속적인 하락 구간을 찾아 peak→trough→recovery 정보를 반환.
    min_dd_pct: 최소 드로다운 % (예: -3.0 = 3% 이상 하락만)
    """
    peak = prices.expanding().max()
    dd = (prices - peak) / peak * 100  # 드로다운 %

    periods = []
    in_dd = False
    peak_date = None
    peak_val = None
    trough_date = None
    trough_val = None
    trough_dd = 0

    for date in prices.index:
        current_dd = dd.loc[date]

        if current_dd < min_dd_pct:
            if not in_dd:
                # 드로다운 시작 — peak 날짜 찾기
                in_dd = True
                # peak는 이 시점의 cummax 위치
                peak_val = peak.loc[date]
                # peak 날짜: 해당 peak 값과 같은 마지막 날짜
                peak_candidates = prices[prices == peak_val]
                peak_date = peak_candidates.index[-1]
                trough_date = date
                trough_val = prices.loc[date]
                trough_dd = current_dd

            # 더 깊은 저점 갱신
            if current_dd < trough_dd:
                trough_date = date
                trough_val = prices.loc[date]
                trough_dd = current_dd
        else:
            if in_dd:
                # 드로다운 종료 (회복)
                recovery_date = date
                duration_days = (trough_date - peak_date).days

                periods.append({
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": recovery_date,
                    "peak_value": peak_val,
                    "trough_value": trough_val,
                    "max_drawdown_pct": round(trough_dd, 2),
                    "duration_days": duration_days,
                    "duration_trading": len(prices[peak_date:trough_date]),
                })
                in_dd = False
                trough_dd = 0

    # 마지막 구간이 아직 회복 안 된 경우
    if in_dd:
        periods.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": None,
            "peak_value": peak_val,
            "trough_value": trough_val,
            "max_drawdown_pct": round(trough_dd, 2),
            "duration_days": (trough_date - peak_date).days,
            "duration_trading": len(prices[peak_date:trough_date]),
        })

    return periods


def analyze_defense(
    kospi_close: pd.Series,
    strategy_values: pd.Series,
    dd_periods: list[dict],
) -> list[dict]:
    """각 드로다운 구간에서 전략의 방어력 계산."""
    results = []

    for p in dd_periods:
        peak_d = p["peak_date"]
        trough_d = p["trough_date"]

        # KOSPI peak→trough 수익률
        k_peak = kospi_close.loc[peak_d] if peak_d in kospi_close.index else None
        k_trough = kospi_close.loc[trough_d] if trough_d in kospi_close.index else None

        if k_peak is None or k_trough is None:
            continue

        k_ret = (k_trough / k_peak - 1) * 100

        # 전략 같은 구간 수익률
        # 가장 가까운 거래일 찾기
        s_dates = strategy_values.index
        s_peak_d = s_dates[s_dates <= peak_d][-1] if any(s_dates <= peak_d) else None
        s_trough_d = s_dates[s_dates <= trough_d][-1] if any(s_dates <= trough_d) else None

        if s_peak_d is None or s_trough_d is None:
            continue

        s_peak_v = strategy_values.loc[s_peak_d]
        s_trough_v = strategy_values.loc[s_trough_d]
        s_ret = (s_trough_v / s_peak_v - 1) * 100

        # 구간 내 전략 최대 낙폭
        s_slice = strategy_values[s_peak_d:s_trough_d]
        if len(s_slice) > 1:
            s_peak_in = s_slice.cummax()
            s_dd_in = ((s_slice - s_peak_in) / s_peak_in * 100).min()
        else:
            s_dd_in = 0

        # 방어 효과 = 전략 수익률 - KOSPI 수익률 (양수면 방어 성공)
        defense = s_ret - k_ret

        # 방어율 = 전략이 KOSPI 하락의 몇 %를 차단했는가
        if k_ret < 0:
            defense_rate = (1 - s_ret / k_ret) * 100 if k_ret != 0 else 0
        else:
            defense_rate = 0

        results.append({
            **p,
            "kospi_return_pct": round(k_ret, 2),
            "strategy_return_pct": round(s_ret, 2),
            "strategy_max_dd_pct": round(s_dd_in, 2),
            "defense_pct": round(defense, 2),
            "defense_rate": round(defense_rate, 1),
        })

    return results


def calc_downside_beta(
    kospi_returns: pd.Series,
    strategy_returns: pd.Series,
) -> float:
    """하방 베타: KOSPI가 하락한 날만 대상으로 한 베타.

    1.0 미만이면 KOSPI 하락 시 전략이 덜 빠짐 → 방어 성공.
    """
    aligned = pd.DataFrame({
        "kospi": kospi_returns,
        "strategy": strategy_returns,
    }).dropna()

    down_days = aligned[aligned["kospi"] < 0]
    if len(down_days) < 10:
        return float("nan")

    cov = np.cov(down_days["kospi"], down_days["strategy"])
    var_k = cov[0, 0]
    if var_k == 0:
        return float("nan")

    return cov[0, 1] / var_k


def calc_weekly_drawdowns(
    kospi_close: pd.Series,
    strategy_values: pd.Series,
) -> pd.DataFrame:
    """주간 단위 수익률 비교 → 하락 주간에서의 방어력."""
    # 주간 수익률
    k_weekly = kospi_close.resample("W-FRI").last().pct_change() * 100
    s_weekly = strategy_values.resample("W-FRI").last().pct_change() * 100

    merged = pd.DataFrame({
        "kospi_weekly": k_weekly,
        "strategy_weekly": s_weekly,
    }).dropna()

    # KOSPI 하락 주간만
    down_weeks = merged[merged["kospi_weekly"] < 0].copy()
    down_weeks["defense"] = down_weeks["strategy_weekly"] - down_weeks["kospi_weekly"]
    down_weeks["defended"] = down_weeks["strategy_weekly"] > down_weeks["kospi_weekly"]

    return down_weeks


def main():
    print("=" * 70)
    print("  KOSPI 조정 구간 드로다운 방어력 심층 분석")
    print("  ETF 3축 vs KOSPI — Peak-to-Trough 기반")
    print("=" * 70)

    # ── 1. 백테스트 실행 (최적 파라미터) ──
    print("\n[1/5] 백테스트 실행 (격주 로테이션 + 손절 -7%)...")
    results = run_backtest(start_date="2025-06-01", end_date="2026-02-27")
    strategy_df = results["daily_values"]
    strategy_values = strategy_df["total_value"]

    # ── 2. KOSPI 데이터 준비 ──
    print("\n[2/5] KOSPI 데이터 준비...")
    kospi = load_kospi()
    kospi_close = kospi["close"]

    # 백테스트 기간 필터
    start = pd.Timestamp("2025-06-01")
    end = pd.Timestamp("2026-02-27")
    kospi_period = kospi_close[(kospi_close.index >= start) & (kospi_close.index <= end)]
    print(f"  KOSPI 데이터: {kospi_period.index[0].strftime('%Y-%m-%d')} ~ "
          f"{kospi_period.index[-1].strftime('%Y-%m-%d')} ({len(kospi_period)}일)")
    print(f"  KOSPI 시작: {kospi_period.iloc[0]:,.0f} → 종료: {kospi_period.iloc[-1]:,.0f} "
          f"({(kospi_period.iloc[-1]/kospi_period.iloc[0]-1)*100:+.1f}%)")

    # ── 3. Peak-to-Trough 드로다운 구간 식별 ──
    print("\n[3/5] KOSPI 드로다운 구간 식별 (Peak-to-Trough)...")

    # -2% 이상 하락 구간 모두 찾기 (세밀한 분석)
    dd_periods = find_drawdown_periods(kospi_period, min_dd_pct=-2.0)
    print(f"  발견된 드로다운 구간: {len(dd_periods)}개 (>= -2%)")

    if not dd_periods:
        print("  ⚠️ 드로다운 구간이 없습니다. 임계값을 -1%로 낮춥니다...")
        dd_periods = find_drawdown_periods(kospi_period, min_dd_pct=-1.0)
        print(f"  발견된 드로다운 구간: {len(dd_periods)}개 (>= -1%)")

    # ── 4. 방어력 분석 ──
    print("\n[4/5] 구간별 방어력 분석...")
    defense_results = analyze_defense(kospi_close, strategy_values, dd_periods)

    if defense_results:
        print(f"\n{'='*80}")
        print(f"  KOSPI 조정 구간별 ETF 3축 방어력")
        print(f"{'='*80}")
        print(f"  {'구간':>20} | {'기간':>5} | {'KOSPI':>8} | {'전략':>8} | {'방어':>8} | {'방어율':>6}")
        print(f"  {'-'*20}+{'-'*7}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*8}")

        total_k = 0
        total_s = 0
        defended_count = 0

        for r in defense_results:
            period_str = f"{r['peak_date'].strftime('%m/%d')}~{r['trough_date'].strftime('%m/%d')}"
            dur = r["duration_trading"]
            k_ret = r["kospi_return_pct"]
            s_ret = r["strategy_return_pct"]
            defense = r["defense_pct"]
            d_rate = r["defense_rate"]

            total_k += k_ret
            total_s += s_ret
            if defense > 0:
                defended_count += 1

            marker = "✓" if defense > 0 else "✗"
            print(f"  {period_str:>20} | {dur:>3}일 | {k_ret:>+7.2f}% | {s_ret:>+7.2f}% | "
                  f"{defense:>+7.2f}%p| {d_rate:>5.1f}% {marker}")

        print(f"  {'-'*20}+{'-'*7}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*8}")
        total_defense = total_s - total_k
        print(f"  {'합계':>20} |       | {total_k:>+7.2f}% | {total_s:>+7.2f}% | "
              f"{total_defense:>+7.2f}%p|")
        print(f"\n  방어 성공: {defended_count}/{len(defense_results)}건 "
              f"({defended_count/len(defense_results)*100:.0f}%)")

        # 주요 조정 구간 (Top 5 깊은 하락)
        major = sorted(defense_results, key=lambda x: x["kospi_return_pct"])[:5]
        print(f"\n{'='*80}")
        print(f"  주요 조정 구간 상세 (KOSPI 하락폭 기준 Top {len(major)})")
        print(f"{'='*80}")
        for r in major:
            print(f"\n  ■ {r['peak_date'].strftime('%Y-%m-%d')} ~ {r['trough_date'].strftime('%Y-%m-%d')} "
                  f"({r['duration_trading']}거래일, {r['duration_days']}캘린더일)")
            print(f"    KOSPI:  {r['peak_value']:,.0f} → {r['trough_value']:,.0f} ({r['kospi_return_pct']:+.2f}%)")
            print(f"    전략:   {r['strategy_return_pct']:+.2f}% (구간내 최대낙폭: {r['strategy_max_dd_pct']:.2f}%)")
            print(f"    방어:   {r['defense_pct']:+.2f}%p (방어율: {r['defense_rate']:.1f}%)")
            if r['defense_pct'] > 0:
                print(f"    결과:   ✓ KOSPI보다 {abs(r['defense_pct']):.2f}%p 덜 하락")
            else:
                print(f"    결과:   ✗ KOSPI보다 {abs(r['defense_pct']):.2f}%p 더 하락")
    else:
        print("  ⚠️ 방어력 분석할 수 있는 구간이 없습니다.")

    # ── 5. 보조 지표 ──
    print(f"\n{'='*80}")
    print(f"  보조 분석 지표")
    print(f"{'='*80}")

    # 5-1. 하방 베타
    kospi_daily_ret = kospi_period.pct_change().dropna()
    strat_daily_ret = strategy_values.pct_change().dropna()
    down_beta = calc_downside_beta(kospi_daily_ret, strat_daily_ret)
    print(f"\n  ■ 하방 베타 (Downside Beta): {down_beta:.3f}")
    if down_beta < 1.0:
        print(f"    → KOSPI 하락 시 전략은 {down_beta:.0%} 수준으로만 하락 (방어 성공)")
    else:
        print(f"    → KOSPI 하락 시 전략이 더 크게 하락 (방어 실패)")

    # 5-2. 주간 하락 방어
    down_weeks = calc_weekly_drawdowns(kospi_period, strategy_values)
    if len(down_weeks) > 0:
        defense_weeks = down_weeks["defended"].sum()
        total_weeks = len(down_weeks)
        avg_k_down = down_weeks["kospi_weekly"].mean()
        avg_s_down = down_weeks["strategy_weekly"].mean()

        print(f"\n  ■ 주간 하락 방어 분석:")
        print(f"    KOSPI 하락 주간: {total_weeks}주")
        print(f"    전략이 덜 빠진 주간: {defense_weeks}주 ({defense_weeks/total_weeks*100:.0f}%)")
        print(f"    KOSPI 평균 주간 하락: {avg_k_down:+.2f}%")
        print(f"    전략 평균 주간 하락:  {avg_s_down:+.2f}%")
        print(f"    평균 방어 효과:       {avg_s_down - avg_k_down:+.2f}%p")

        # 하락 주간 중 가장 큰 5개
        worst_weeks = down_weeks.nsmallest(5, "kospi_weekly")
        print(f"\n    KOSPI 최악 5주간:")
        print(f"    {'주간종료':>12} | {'KOSPI':>8} | {'전략':>8} | {'방어':>8}")
        print(f"    {'-'*12}+{'-'*10}+{'-'*10}+{'-'*10}")
        for date, row in worst_weeks.iterrows():
            defense_str = f"{row['defense']:+.2f}%p"
            marker = "✓" if row["defended"] else "✗"
            print(f"    {date.strftime('%Y-%m-%d'):>12} | {row['kospi_weekly']:>+7.2f}% | "
                  f"{row['strategy_weekly']:>+7.2f}% | {defense_str:>8} {marker}")

    # 5-3. 현금비율 분석 (레짐별)
    print(f"\n  ■ 평균 현금 비율 (방어의 핵심 메커니즘):")
    cash_ratios = strategy_df["cash"] / strategy_df["total_value"] * 100
    print(f"    전체 평균: {cash_ratios.mean():.1f}%")
    print(f"    최대:      {cash_ratios.max():.1f}%")
    print(f"    최소:      {cash_ratios.min():.1f}%")

    # 5-4. 상승/하락 일수 비교
    aligned = pd.DataFrame({
        "kospi": kospi_daily_ret,
        "strategy": strat_daily_ret,
    }).dropna()

    k_up = (aligned["kospi"] > 0).sum()
    k_down = (aligned["kospi"] < 0).sum()
    # KOSPI 하락일에 전략도 하락한 비율
    both_down = ((aligned["kospi"] < 0) & (aligned["strategy"] < 0)).sum()
    # KOSPI 하락일에 전략은 상승한 비율
    strat_up_when_k_down = ((aligned["kospi"] < 0) & (aligned["strategy"] > 0)).sum()

    print(f"\n  ■ 일별 상관 분석:")
    print(f"    KOSPI 상승일: {k_up}일, 하락일: {k_down}일")
    print(f"    KOSPI 하락일 중 전략도 하락: {both_down}일 ({both_down/k_down*100:.0f}%)")
    print(f"    KOSPI 하락일 중 전략은 상승: {strat_up_when_k_down}일 ({strat_up_when_k_down/k_down*100:.0f}%)")

    # 일별 상관계수
    corr = aligned["kospi"].corr(aligned["strategy"])
    print(f"    일별 상관계수: {corr:.3f}")

    # ── 최종 결론 ──
    print(f"\n{'='*80}")
    print(f"  최종 결론: 약세장 수비수 가설 검증")
    print(f"{'='*80}")

    passed = 0
    total_tests = 4

    # 테스트 1: 하방 베타 < 1.0
    t1 = not np.isnan(down_beta) and down_beta < 1.0
    passed += t1
    print(f"\n  1. 하방 베타 < 1.0:  {'✓ PASS' if t1 else '✗ FAIL'} (β={down_beta:.3f})")

    # 테스트 2: 주요 조정 구간에서 50% 이상 방어 성공
    if defense_results:
        dr = defended_count / len(defense_results)
        t2 = dr >= 0.5
        passed += t2
        print(f"  2. 조정구간 방어 ≥50%: {'✓ PASS' if t2 else '✗ FAIL'} ({defended_count}/{len(defense_results)} = {dr*100:.0f}%)")
    else:
        t2 = False
        print(f"  2. 조정구간 방어 ≥50%: N/A (구간 없음)")

    # 테스트 3: 주간 하락 방어율 > 50%
    if len(down_weeks) > 0:
        wr = defense_weeks / total_weeks
        t3 = wr > 0.5
        passed += t3
        print(f"  3. 주간 방어율 > 50%:  {'✓ PASS' if t3 else '✗ FAIL'} ({defense_weeks}/{total_weeks} = {wr*100:.0f}%)")
    else:
        t3 = False
        print(f"  3. 주간 방어율 > 50%:  N/A (데이터 없음)")

    # 테스트 4: 전략 MDD < KOSPI MDD
    kospi_peak = kospi_period.cummax()
    kospi_mdd = ((kospi_period - kospi_peak) / kospi_peak * 100).min()
    strat_mdd = results["mdd_pct"]
    t4 = strat_mdd > kospi_mdd  # MDD는 음수이므로 -6.5 > -10 이면 전략이 방어
    passed += t4
    print(f"  4. 전략 MDD > KOSPI MDD: {'✓ PASS' if t4 else '✗ FAIL'} "
          f"(전략 {strat_mdd:.2f}% vs KOSPI {kospi_mdd:.2f}%)")

    print(f"\n  종합: {passed}/{total_tests} 통과")
    if passed >= 3:
        print(f"  ✓ 약세장 수비수 가설 입증: ETF 3축은 KOSPI 하락 구간에서 유의미한 방어력 보유")
    elif passed >= 2:
        print(f"  △ 부분적 입증: 일부 지표에서 방어력 확인되나 완전한 입증은 아님")
    else:
        print(f"  ✗ 가설 미입증: 추가 데이터 또는 전략 개선 필요")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
