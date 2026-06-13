"""risk/component_var Component VaR 엔진 단위 — RISK_ENGINE Phase 4 (§3.3).

불변식:
  1. Euler 합산성: 기여율 합 = 1, component_var 합 = var95 (공분산 분해 정확성).
  2. 동질(동일 변동성·무상관·동일 비중) → 각 기여율 ≈ 1/k.
  3. 고변동 종목은 비중보다 큰 기여(비중≠리스크 기여를 드러냄 = §3.3 목적).
  4. 표본 부족/데이터 없음 → ok=False fail-closed(보수적 var95=1.0).
  5. 단일 포지션 → 기여율 ≈ 1.0.
  6. 헤지(음의 상관) 종목은 기여를 낮춘다(음수 기여 가능).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.component_var import compute_component_var


def _returns(sigma, n=400, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, sigma, n), index=idx)


def test_contributions_sum_to_one_and_var():
    # Euler 정확성: 기여율 합 = 1, component_var 합 = var95
    rets = {"A": _returns(0.01, seed=1), "B": _returns(0.02, seed=2), "C": _returns(0.015, seed=3)}
    weights = {"A": 0.1, "B": 0.1, "C": 0.1}
    cv = compute_component_var(rets, weights)
    assert cv.ok
    assert cv.n_positions == 3
    assert abs(sum(cv.contributions.values()) - 1.0) < 1e-9
    assert abs(sum(cv.component_var.values()) - cv.var95) < 1e-9


def test_homogeneous_assets_equal_contribution():
    # 동일 변동성·무상관·동일 비중 → 각 기여율 ≈ 1/3 (독립 시드로 상관 거의 0)
    rets = {t: _returns(0.02, seed=s) for t, s in (("A", 10), ("B", 20), ("C", 30))}
    weights = {"A": 0.1, "B": 0.1, "C": 0.1}
    cv = compute_component_var(rets, weights)
    for t in ("A", "B", "C"):
        assert abs(cv.contributions[t] - 1 / 3) < 0.10  # 표본 잡음 허용


def test_high_vol_contributes_more_than_weight_share():
    # 비중은 같지만 변동성 10배 종목이 리스크 기여를 압도 (§3.3: 비중≠리스크 기여)
    rets = {"LOW": _returns(0.005, seed=1), "HIGH": _returns(0.05, seed=2)}
    weights = {"LOW": 0.1, "HIGH": 0.1}
    cv = compute_component_var(rets, weights)
    assert cv.contributions["HIGH"] > 0.8  # 비중 50%인데 기여 80%+
    assert cv.contributions["HIGH"] > cv.contributions["LOW"]


def test_insufficient_data_fail_closed():
    # 표본 40 < 60 → ok=False, 보수적 var95=1.0, 빈 기여 dict
    cv = compute_component_var({"A": _returns(0.02, n=40)}, {"A": 0.1})
    assert cv.ok is False
    assert cv.var95 == 1.0
    assert cv.contributions == {}
    assert "insufficient_obs" in cv.reason


def test_no_data_fail_closed():
    cv = compute_component_var({}, {"A": 0.1})
    assert cv.ok is False
    assert cv.reason == "no_returns_data"


def test_single_position_contributes_all():
    # 단일 포지션 → 그 종목이 리스크의 100%
    cv = compute_component_var({"A": _returns(0.02)}, {"A": 0.1})
    assert cv.ok
    assert cv.n_positions == 1
    assert abs(cv.contributions["A"] - 1.0) < 1e-9


def test_contribution_helper_absent_ticker():
    cv = compute_component_var({"A": _returns(0.02)}, {"A": 0.1})
    assert cv.contribution("A") == cv.contributions["A"]
    assert cv.contribution("ZZZ") == 0.0  # 부재 종목 → 0


def test_hedge_lowers_contribution():
    # 강한 음의 공분산(B = -0.25·A + 잡음): B의 음의 공분산이 자기분산을 압도 → B 기여율 음수.
    #   포트 분산은 양수로 유지(부분 헤지) → 분해 안정. Σ=1 보존.
    base = _returns(0.04, seed=7)
    noise = _returns(0.005, seed=8)
    rets = {"A": base, "B": -0.25 * base + noise}
    cv = compute_component_var(rets, {"A": 0.1, "B": 0.1})
    assert cv.ok
    assert abs(sum(cv.contributions.values()) - 1.0) < 1e-9
    assert cv.contributions["B"] < 0.0  # 헤지 종목은 음의 리스크 기여
    assert cv.contributions["A"] > 1.0  # 합=1이므로 다른 쪽이 1 초과


def test_perfect_hedge_degenerate_fail_closed():
    # 완전 헤지(B=-A) → 포트 분산 ≈ 0(부동소수 잡음) → degenerate fail-closed(기여율 발산 차단)
    base = _returns(0.02, seed=7)
    cv = compute_component_var({"A": base, "B": -base}, {"A": 0.1, "B": 0.1})
    assert cv.ok is False
    assert cv.reason == "degenerate_zero_variance"
    assert cv.contributions == {}


def test_weights_zero_excluded():
    # 비중 0 종목은 패널에서 제외 (compute_portfolio_var와 동일 선별)
    rets = {"A": _returns(0.02, seed=1), "B": _returns(0.02, seed=2)}
    cv = compute_component_var(rets, {"A": 0.1, "B": 0.0})
    assert cv.n_positions == 1
    assert "B" not in cv.contributions
