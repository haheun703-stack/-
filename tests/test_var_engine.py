"""risk/var_engine FHS VaR/ES 단위 테스트 (RISK_ENGINE Phase 2a).

불변식:
  1. 정규 수익률 → VaR95 ≈ 1.645·σ (FHS가 표준 분위수 근사).
  2. 표본 < 60 / 데이터 없음 → fail-closed(ok=False, var=1.0) → 게이트가 막는다.
  3. 단조성: var99 ≥ var95, es95 ≥ var95, stress_var95 ≥ var95.
  4. 고변동 종목 VaR > 저변동 종목 VaR / 비중 선형 스케일.
  5. 저상관 다종목 → 분산효과(포트 VaR < 단일 100% VaR).
  6. 이상치 1개가 VaR를 폭발시키지 않는다(z 클립).
  7. 비중 0 종목 제외 / 공통 날짜 정렬.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.var_engine import _MIN_OBS, _ewma_vol, compute_portfolio_var


def _series(vals, start="2023-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(vals), freq="B")
    return pd.Series(list(vals), index=idx)


def _normal_returns(n, sigma, seed) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(rng.normal(0.0, sigma, n))


def test_normal_var95_near_theoretical():
    res = compute_portfolio_var({"A": _normal_returns(400, 0.02, 42)}, {"A": 1.0})
    assert res.ok
    assert 0.024 <= res.var95 <= 0.045, res.var95  # 이론 1.645·0.02 ≈ 0.0329


def test_insufficient_obs_fail_closed():
    res = compute_portfolio_var({"A": _normal_returns(40, 0.02, 1)}, {"A": 1.0})
    assert not res.ok and res.var95 == 1.0 and "insufficient" in res.reason


def test_no_data_fail_closed():
    res = compute_portfolio_var({}, {"A": 1.0})
    assert not res.ok and res.var95 == 1.0 and res.reason == "no_returns_data"


def test_monotonicity():
    res = compute_portfolio_var({"A": _normal_returns(300, 0.02, 7)}, {"A": 1.0})
    assert res.var99 >= res.var95
    assert res.es95 >= res.var95
    assert res.stress_var95 >= res.var95


def test_stress_roughly_1_5x():
    res = compute_portfolio_var({"A": _normal_returns(500, 0.015, 11)}, {"A": 1.0})
    assert 1.3 <= res.stress_var95 / res.var95 <= 1.7  # 변동성 1.5배 → VaR ~1.5배


def test_higher_vol_higher_var():
    lo = compute_portfolio_var({"A": _normal_returns(300, 0.01, 3)}, {"A": 1.0})
    hi = compute_portfolio_var({"A": _normal_returns(300, 0.03, 3)}, {"A": 1.0})
    assert hi.var95 > lo.var95 * 1.5


def test_weight_scales_var():
    r = _normal_returns(300, 0.02, 5)
    full = compute_portfolio_var({"A": r}, {"A": 1.0})
    half = compute_portfolio_var({"A": r}, {"A": 0.5})
    assert abs(half.var95 - full.var95 * 0.5) < full.var95 * 0.05  # 선형


def test_diversification_effect():
    a = _normal_returns(500, 0.02, 100)
    b = _normal_returns(500, 0.02, 200)  # 독립
    port = compute_portfolio_var({"A": a, "B": b}, {"A": 0.5, "B": 0.5})
    single = compute_portfolio_var({"A": a}, {"A": 1.0})
    assert port.var95 < single.var95  # 독립 분산 → 포트 σ ≈ 0.707·단일


def test_zero_weight_excluded():
    r = _normal_returns(300, 0.02, 9)
    res = compute_portfolio_var(
        {"A": r, "B": _normal_returns(300, 0.9, 9)}, {"A": 1.0, "B": 0.0}
    )
    only_a = compute_portfolio_var({"A": r}, {"A": 1.0})
    assert abs(res.var95 - only_a.var95) < 1e-9


def test_outlier_clipped():
    vals = list(np.random.default_rng(0).normal(0, 0.02, 300))
    vals[150] = -5.0  # -500% 글리치 1개(손실 쪽)
    res = compute_portfolio_var({"A": _series(vals)}, {"A": 1.0})
    assert res.ok and res.var95 < 0.5  # 클립 없으면 폭발, 클립으로 합리적 범위


def test_ewma_vol_positive_and_responsive():
    rng = np.random.default_rng(1)
    r = np.concatenate([rng.normal(0, 0.01, 100), rng.normal(0, 0.05, 100)])
    sigma = _ewma_vol(r, 0.94)
    assert (sigma > 0).all()
    assert sigma[-1] > sigma[100]  # 고변동 구간 진입 후 σ 상승


def test_common_dates_alignment():
    a = _series(np.random.default_rng(2).normal(0, 0.02, 200), start="2023-01-01")
    b = _series(np.random.default_rng(3).normal(0, 0.02, 200), start="2023-03-01")
    res = compute_portfolio_var({"A": a, "B": b}, {"A": 0.5, "B": 0.5})
    assert res.ok and res.n_obs < 200  # 교집합이라 200 미만


def test_min_obs_boundary():
    # 정확히 _MIN_OBS면 통과(>= 경계)
    res = compute_portfolio_var({"A": _normal_returns(_MIN_OBS, 0.02, 13)}, {"A": 1.0})
    assert res.ok and res.n_obs == _MIN_OBS
