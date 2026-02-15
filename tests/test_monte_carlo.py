"""
v4.6 Monte Carlo + 표본수 경고 + 통계적 신뢰도 테스트
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.quant_metrics import (
    assess_reliability,
    calc_full_metrics,
)
from src.walk_forward import MonteCarloSimulator


def make_trades_df(n_trades: int, win_rate: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """테스트용 거래 데이터 생성."""
    rng = np.random.RandomState(seed)
    pnl_list = []
    for _ in range(n_trades):
        if rng.random() < win_rate:
            pnl_list.append(rng.uniform(50000, 300000))
        else:
            pnl_list.append(rng.uniform(-200000, -30000))
    pnl = np.array(pnl_list)
    entry = 50000
    pnl_pct = pnl / (entry * 100) * 100  # rough %

    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n_trades)],
        "entry_date": pd.date_range("2020-01-01", periods=n_trades, freq="B"),
        "exit_date": pd.date_range("2020-01-05", periods=n_trades, freq="B"),
        "entry_price": entry,
        "exit_price": entry + pnl / 100,
        "shares": 100,
        "pnl": pnl.astype(int),
        "pnl_pct": np.round(pnl_pct, 2),
        "hold_days": rng.randint(1, 15, n_trades),
        "exit_reason": "test",
        "grade": rng.choice(["A", "B", "C"], n_trades),
        "bes_score": rng.uniform(0.5, 1.0, n_trades),
        "commission": 1000,
        "trigger_type": rng.choice(["impulse", "confirm"], n_trades),
    })


def make_equity_df(n_days: int = 500, initial: float = 50_000_000) -> pd.DataFrame:
    """테스트용 에쿼티 데이터 생성."""
    rng = np.random.RandomState(42)
    returns = rng.normal(0.0003, 0.01, n_days)
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    dates = pd.date_range("2020-01-01", periods=n_days + 1, freq="B")
    return pd.DataFrame({
        "date": [str(d.date()) for d in dates],
        "portfolio_value": [round(e) for e in equity],
        "cash": [round(e * 0.3) for e in equity],
        "n_positions": 2,
    })


# ═════════════════════════════════════════════════
# 테스트 1: 통계적 신뢰도 등급 (표본수 기반)
# ═════════════════════════════════════════════════

def test_reliability_grade_f():
    """30건 미만 → F등급."""
    r = assess_reliability(n_trades=15)
    assert r["grade"] == "F", f"F 기대, 실제: {r['grade']}"
    assert "심각 부족" in r["message"]
    print("  [PASS] test_reliability_grade_f")


def test_reliability_grade_c():
    """30~49건 → C등급."""
    r = assess_reliability(n_trades=35)
    assert r["grade"] == "C", f"C 기대, 실제: {r['grade']}"
    print("  [PASS] test_reliability_grade_c")


def test_reliability_grade_b():
    """50~99건 → B등급."""
    r = assess_reliability(n_trades=75)
    assert r["grade"] == "B", f"B 기대, 실제: {r['grade']}"
    print("  [PASS] test_reliability_grade_b")


def test_reliability_grade_a():
    """100건+ → A등급."""
    r = assess_reliability(n_trades=150)
    assert r["grade"] == "A", f"A 기대, 실제: {r['grade']}"
    print("  [PASS] test_reliability_grade_a")


def test_reliability_with_bootstrap():
    """A등급 + Bootstrap 미유의 → B등급 강등."""
    fake_bs = {
        "status": "completed",
        "shuffle_sharpe": {"significant": False},
        "shuffle_total_return": {"significant": False},
    }
    r = assess_reliability(n_trades=150, bootstrap_results=fake_bs)
    assert r["grade"] == "B", f"B 기대 (BS 미유의), 실제: {r['grade']}"
    assert "Bootstrap 미유의" in r["message"]
    print("  [PASS] test_reliability_with_bootstrap")


# ═════════════════════════════════════════════════
# 테스트 2: calc_full_metrics에 reliability 포함
# ═════════════════════════════════════════════════

def test_metrics_includes_reliability():
    """calc_full_metrics 결과에 statistical_reliability 포함."""
    trades = make_trades_df(60)
    equity = make_equity_df()
    metrics = calc_full_metrics(trades, equity)
    assert "statistical_reliability" in metrics
    assert metrics["statistical_reliability"]["grade"] == "B"  # 60건
    print("  [PASS] test_metrics_includes_reliability")


# ═════════════════════════════════════════════════
# 테스트 3: Monte Carlo 시뮬레이션
# ═════════════════════════════════════════════════

def test_monte_carlo_basic():
    """MC 시뮬레이션 기본 동작."""
    trades = make_trades_df(100, win_rate=0.55)
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    results = mc.run(trades, initial_capital=50_000_000)

    assert results["status"] == "completed"
    r = results["result"]
    assert r.n_simulations == 500
    assert r.n_trades == 100
    assert r.final_equity_mean > 0
    assert r.mdd_mean_pct < 0  # MDD는 항상 음수
    assert 0 <= r.ruin_probability <= 100
    assert 0 <= r.positive_pct <= 100
    assert r.return_p5_pct <= r.return_mean_pct <= r.return_p95_pct
    print("  [PASS] test_monte_carlo_basic")


def test_monte_carlo_winning_strategy():
    """승률 높은 전략: 파산 확률 낮음."""
    trades = make_trades_df(200, win_rate=0.65, seed=99)
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    results = mc.run(trades, initial_capital=50_000_000)
    r = results["result"]
    assert r.positive_pct > 50, f"양수 비율 > 50% 기대, 실제: {r.positive_pct}"
    print("  [PASS] test_monte_carlo_winning_strategy")


def test_monte_carlo_empty():
    """빈 거래 → no_trades."""
    trades = pd.DataFrame()
    mc = MonteCarloSimulator()
    results = mc.run(trades)
    assert results["status"] == "no_trades"
    print("  [PASS] test_monte_carlo_empty")


def test_monte_carlo_equity_curve():
    """에쿼티 커브 정확성 검증."""
    mc = MonteCarloSimulator()
    pnls = np.array([100, -50, 200, -30, 80])
    equity = mc._build_equity_curve(pnls, 1000)
    assert equity[0] == 1000
    assert equity[1] == 1100
    assert equity[2] == 1050
    assert equity[3] == 1250
    assert equity[4] == 1220
    assert equity[5] == 1300
    print("  [PASS] test_monte_carlo_equity_curve")


def test_monte_carlo_mdd():
    """MDD 계산 정확성."""
    mc = MonteCarloSimulator()
    # 1000 → 1200 → 900 → 1100: MDD = (900-1200)/1200 = -25%
    equity = np.array([1000, 1200, 900, 1100])
    mdd = mc._calc_mdd_pct(equity)
    assert abs(mdd - (-25.0)) < 0.1, f"MDD -25% 기대, 실제: {mdd}"
    print("  [PASS] test_monte_carlo_mdd")


# ═════════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("v4.6 Monte Carlo + Reliability Test Suite")
    print("=" * 50)

    tests = [
        test_reliability_grade_f,
        test_reliability_grade_c,
        test_reliability_grade_b,
        test_reliability_grade_a,
        test_reliability_with_bootstrap,
        test_metrics_includes_reliability,
        test_monte_carlo_basic,
        test_monte_carlo_winning_strategy,
        test_monte_carlo_empty,
        test_monte_carlo_equity_curve,
        test_monte_carlo_mdd,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"결과: {passed}/{passed + failed} PASS")
    if failed == 0:
        print("ALL TESTS PASSED")
    print("=" * 50)
