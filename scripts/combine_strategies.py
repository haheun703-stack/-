"""
v10.1 + 그룹순환매 결합 시뮬레이션

두 전략의 일별 수익률 상관관계 분석 + 60/40 포트폴리오 시뮬레이션.
상관 < 0.3이면 MDD가 개별 전략보다 유의미하게 낮아짐.

배분: v10.1 60% (6,000만) + 그룹순환매 40% (4,000만)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPTS_DIR = Path(__file__).parent


def import_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_v101():
    """v10.1 C_new 모드 실행 → daily equity Series 반환"""
    bt = import_module("backtest_v2", SCRIPTS_DIR / "backtest_v2.py")

    data_dict = bt.load_parquets()
    name_map = bt.load_name_map()
    kospi_df = bt.load_kospi_index()

    trades, daily_results = bt.run_backtest(data_dict, name_map, "C_new", kospi_df=kospi_df)

    dates = [d.date for d in daily_results]
    equities = [d.equity for d in daily_results]

    return pd.Series(equities, index=dates, name="v101_equity"), trades


def run_group_rotation():
    """그룹순환매 D 모드 실행 → daily equity Series 반환"""
    bt = import_module("backtest_group_rotation", SCRIPTS_DIR / "backtest_group_rotation.py")

    groups, etfs, ewy, stocks = bt.load_all_data()
    stock_indicators = bt.prepare_indicators(groups, etfs, ewy, stocks)

    trades, daily_results, regime_log = bt.run_backtest(
        "D", groups, etfs, ewy, stock_indicators
    )

    dates = [d["date"] for d in daily_results]
    equities = [d["equity"] for d in daily_results]

    return pd.Series(equities, index=dates, name="group_equity"), trades


def main():
    print("=" * 65)
    print("  v10.1 + 그룹순환매 결합 시뮬레이션")
    print("=" * 65)

    # 1. 각 전략 실행
    print("\n[1] v10.1 C_new 모드 실행...")
    v101_eq, v101_trades = run_v101()
    print(f"  → {len(v101_eq)}일, 최종 {v101_eq.iloc[-1]/1e4:,.0f}만원")

    print("\n[2] 그룹순환매 D 모드 실행...")
    grp_eq, grp_trades = run_group_rotation()
    print(f"  → {len(grp_eq)}일, 최종 {grp_eq.iloc[-1]/1e4:,.0f}만원")

    # 2. 공통 날짜 정렬
    common_dates = v101_eq.index.intersection(grp_eq.index)
    v101 = v101_eq.loc[common_dates]
    grp = grp_eq.loc[common_dates]
    print(f"\n  공통 거래일: {len(common_dates)}일")

    # 3. 일별 수익률 계산
    v101_ret = v101.pct_change().dropna()
    grp_ret = grp.pct_change().dropna()

    common_ret_dates = v101_ret.index.intersection(grp_ret.index)
    v101_ret = v101_ret.loc[common_ret_dates]
    grp_ret = grp_ret.loc[common_ret_dates]

    # 4. 상관관계 분석
    print(f"\n{'=' * 65}")
    print("  [분석 1] 일별 수익률 상관관계")
    print(f"{'=' * 65}")

    corr = v101_ret.corr(grp_ret)
    print(f"\n  전체 상관: {corr:.4f}")

    # Rolling 30일 상관
    df_combined = pd.DataFrame({"v101": v101_ret, "group": grp_ret})
    roll_corr = df_combined["v101"].rolling(30).corr(df_combined["group"]).dropna()
    if len(roll_corr) > 0:
        print(f"  30일 롤링 상관: 평균 {roll_corr.mean():.4f}, "
              f"범위 [{roll_corr.min():.4f}, {roll_corr.max():.4f}]")

    # 상관 판정
    if abs(corr) < 0.2:
        corr_verdict = "매우 낮음 — 분산 효과 극대화 가능"
    elif abs(corr) < 0.3:
        corr_verdict = "낮음 — 분산 효과 유의미"
    elif abs(corr) < 0.5:
        corr_verdict = "중간 — 약간의 분산 효과"
    else:
        corr_verdict = "높음 — 분산 효과 제한적"
    print(f"  판정: {corr_verdict}")

    # 동시 하락/상승 비율
    both_down = ((v101_ret < 0) & (grp_ret < 0)).sum()
    both_up = ((v101_ret > 0) & (grp_ret > 0)).sum()
    total = len(v101_ret)
    print(f"\n  동시 상승: {both_up}일 ({both_up/total*100:.0f}%)")
    print(f"  동시 하락: {both_down}일 ({both_down/total*100:.0f}%)")
    print(f"  역방향:   {total - both_up - both_down}일 ({(total-both_up-both_down)/total*100:.0f}%)")

    # 5. 결합 포트폴리오 시뮬레이션
    print(f"\n{'=' * 65}")
    print("  [분석 2] 결합 포트폴리오 시뮬레이션")
    print(f"{'=' * 65}")

    V101_CAPITAL = 60_000_000   # 6,000만원 (60%)
    GRP_CAPITAL = 40_000_000    # 4,000만원 (40%)
    TOTAL_CAPITAL = V101_CAPITAL + GRP_CAPITAL

    # 각 전략의 자본 비례 equity
    v101_initial = v101.iloc[0]
    grp_initial = grp.iloc[0]

    v101_scaled = v101 / v101_initial * V101_CAPITAL
    grp_scaled = grp / grp_initial * GRP_CAPITAL
    combined_eq = v101_scaled + grp_scaled

    # 개별 전략 통계
    strategies = [
        ("v10.1 단독 (1억)", v101, 100_000_000),
        ("그룹순환매 단독 (4천만)", grp, 40_000_000),
        ("결합 60/40 (1억)", combined_eq, TOTAL_CAPITAL),
    ]

    results = []
    for label, eq, init_cap in strategies:
        final = eq.iloc[-1]
        total_ret = (final / init_cap - 1) * 100

        # MDD
        eq_arr = eq.values
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        mdd = np.min(dd)

        # 일별 수익률 통계
        daily_ret = eq.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(240) if daily_ret.std() > 0 else 0
        sortino_den = daily_ret[daily_ret < 0].std()
        sortino = daily_ret.mean() / sortino_den * np.sqrt(240) if sortino_den > 0 else 0

        # 최대 연속 하락일
        is_down = daily_ret < 0
        max_consecutive_down = 0
        current_streak = 0
        for d in is_down:
            if d:
                current_streak += 1
                max_consecutive_down = max(max_consecutive_down, current_streak)
            else:
                current_streak = 0

        results.append({
            "label": label,
            "init_cap": init_cap,
            "final": final,
            "total_ret": total_ret,
            "mdd": mdd,
            "sharpe": sharpe,
            "sortino": sortino,
            "daily_vol": daily_ret.std() * 100,
            "max_consec_down": max_consecutive_down,
        })

    # 비교 테이블
    print(f"\n  {'전략':<28} {'수익률':>8} {'MDD':>7} {'Sharpe':>8} {'Sortino':>8} {'일변동':>7}")
    print(f"  {'-' * 68}")
    for r in results:
        print(f"  {r['label']:<28} {r['total_ret']:>+7.1f}% {r['mdd']:>6.1f}% "
              f"{r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['daily_vol']:>6.2f}%")

    # 결합 효과 분석
    r_v101 = results[0]
    r_grp = results[1]
    r_combo = results[2]

    print(f"\n  결합 효과:")
    # 가중 평균 MDD (독립 가정 시 이론값)
    weighted_ret = r_v101["total_ret"] * 0.6 + r_grp["total_ret"] * 0.4
    print(f"  가중평균 수익률: {weighted_ret:+.1f}% (실제: {r_combo['total_ret']:+.1f}%)")

    # MDD 개선
    weighted_mdd = r_v101["mdd"] * 0.6 + r_grp["mdd"] * 0.4
    print(f"  가중평균 MDD: {weighted_mdd:.1f}% (실제: {r_combo['mdd']:.1f}%)")
    mdd_improvement = r_combo["mdd"] - weighted_mdd
    print(f"  MDD 분산효과: {mdd_improvement:+.1f}%p ({'개선' if mdd_improvement > 0 else '악화'})")

    # Sharpe 개선
    print(f"  Sharpe: v10.1({r_v101['sharpe']:.2f}) + 순환매({r_grp['sharpe']:.2f}) → 결합({r_combo['sharpe']:.2f})")

    # 6. 월별 비교
    print(f"\n{'=' * 65}")
    print("  [분석 3] 월별 수익률 비교")
    print(f"{'=' * 65}")

    monthly_v101 = v101_scaled.resample("ME").last().pct_change().dropna() * 100
    monthly_grp = grp_scaled.resample("ME").last().pct_change().dropna() * 100
    monthly_combo = combined_eq.resample("ME").last().pct_change().dropna() * 100

    print(f"\n  {'월':>8} {'v10.1':>8} {'순환매':>8} {'결합':>8} {'분산효과':>10}")
    print(f"  {'-' * 45}")

    common_months = monthly_combo.index
    for m in common_months:
        v = monthly_v101.get(m, 0)
        g = monthly_grp.get(m, 0)
        c = monthly_combo.get(m, 0)
        w = v * 0.6 + g * 0.4  # 가중평균 (이론)
        diff = c - w
        print(f"  {m.strftime('%Y-%m'):>8} {v:>+7.1f}% {g:>+7.1f}% {c:>+7.1f}% {diff:>+9.1f}%p")

    # 동시 손실 월
    both_loss = sum(1 for m in common_months
                    if monthly_v101.get(m, 0) < 0 and monthly_grp.get(m, 0) < 0)
    print(f"\n  동시 손실 월: {both_loss}/{len(common_months)}개월")

    # 7. 최종 판정
    print(f"\n{'=' * 65}")
    print("  최종 판정")
    print(f"{'=' * 65}")

    checks = [
        ("상관 < 0.3 (분산 유효)", abs(corr) < 0.3),
        ("결합 MDD > v10.1 MDD", r_combo["mdd"] > r_v101["mdd"]),
        ("결합 Sharpe > v10.1 Sharpe", r_combo["sharpe"] > r_v101["sharpe"]),
        ("결합 수익률 > 15%", r_combo["total_ret"] > 15),
    ]

    for name, passed in checks:
        mark = "O" if passed else "X"
        print(f"  [{mark}] {name}")

    all_pass = all(p for _, p in checks)
    if all_pass:
        print(f"\n  >>> 60/40 결합 전략 채택! <<<")
    elif sum(p for _, p in checks) >= 3:
        print(f"\n  >>> 대부분 통과 — 비율 조정 후 채택 가능 <<<")
    else:
        print(f"\n  >>> 결합 효과 부족 — 개별 운용 권장 <<<")

    # 추천 비율
    print(f"\n  참고: 다른 비율 시뮬레이션이 필요하면 요청하세요")
    print(f"  (50/50, 70/30, 40/60 등)")


if __name__ == "__main__":
    main()
