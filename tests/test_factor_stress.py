"""risk/factor_stress — factor_exposure × stress_test 오케스트레이터(§3.1 + §4.1).

불변식:
  1. run_factor_stress_test가 베타를 자동 주입 → S1/S2/S5가 (수동 주입 없이) 평가된다.
  2. 역사 H1~H5는 factor_returns가 그 일자를 포함할 때만 평가(짧으면 graceful 생략).
  3. H 충격 = 팩터 근사(Σ 베타 × 잔차화된 그날 팩터). 실제 종목 수익률이 있으면 그것 우선.
  4. ★잔차화 정합: semi 잔차 베타에는 잔차화된 semi 충격이 곱해진다(raw -15%가 아님).
  5. 빈 holdings / market 팩터 없음 → S3·S4만 평가(기존 stress_test 동작 보존).
  6. 결과는 (exposure, stress) 쌍.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.factor_exposure import build_factor_panel, compute_factor_exposure
from risk.factor_stress import (
    FactorStressResult,
    factor_historical_shocks,
    run_factor_stress_test,
)

_ASOF = pd.Timestamp("2026-06-12")


def _mk(idx, seed):
    rng = np.random.default_rng(seed)
    return {
        "market": pd.Series(rng.normal(0, 0.012, len(idx)), index=idx),
        "smallcap": pd.Series(rng.normal(0, 0.014, len(idx)), index=idx),
        "fx": pd.Series(rng.normal(0, 0.005, len(idx)), index=idx),
        "semi": pd.Series(rng.normal(0, 0.018, len(idx)), index=idx),
        "rate": pd.Series(rng.normal(0, 0.003, len(idx)), index=idx),
    }


def _factor_inputs(n=250, seed=7):
    """H 일자를 포함하지 않는 일반 팩터 시계열(end=2026 → 2008~2024 H 미포함)."""
    return _mk(pd.bdate_range(end=_ASOF, periods=n), seed)


def _factor_inputs_ending(day, n=300, seed=7, shock=None):
    """day(영업일)를 마지막 봉으로 갖는 시계열 — 그날 팩터 충격을 직접 제어."""
    idx = pd.bdate_range(end=pd.Timestamp(day), periods=n)
    f = _mk(idx, seed)
    if shock:
        for k, v in shock.items():
            f[k].iloc[-1] = v
    return f


def _by_id(report):
    return {r.scenario_id: r for r in report.results}


# ── 1. 베타 자동 주입(가상 시나리오) ──────────────────────────────────────────
def test_auto_injects_betas_for_s2():
    f = _factor_inputs(seed=44)
    y = 0.5 * f["market"] + 1.0 * f["semi"]  # 반도체 강노출
    res = run_factor_stress_test({"S": 0.5}, {"S": y}, f)
    s2 = _by_id(res.stress)["S2"]
    assert s2.evaluable                       # 수동 주입 없이 자동 평가
    assert s2.portfolio_pnl < 0               # 반도체 +베타 × -15% = 손실
    assert "S" in {sb.ticker for sb in res.exposure.stock_betas}


def test_auto_s5_combined_evaluated():
    f = _factor_inputs(seed=8)
    y = 1.0 * f["fx"] + 1.0 * f["semi"]       # fx·semi 둘 다 노출 → S1·S2 → S5
    res = run_factor_stress_test({"X": 1.0}, {"X": y}, f)
    assert _by_id(res.stress)["S5"].evaluable


# ── 2. 역사 재생 H — 팩터 근사 ────────────────────────────────────────────────
def test_historical_factor_approximation():
    f = _factor_inputs_ending("2020-03-19", shock={"market": -0.10})  # H3 그날 시장 -10%
    y = 1.0 * f["market"]                                              # 시장 베타 1 종목
    res = run_factor_stress_test({"M": 1.0}, {"M": y}, f)
    h3 = _by_id(res.stress)["H3"]
    assert h3.evaluable
    assert h3.portfolio_pnl < 0
    assert abs(h3.portfolio_pnl - (-0.10)) < 0.03                      # ≈ 베타1 × -10%


def test_historical_shocks_direct_helper():
    f = _factor_inputs_ending("2020-03-19", shock={"market": -0.10})
    rep = compute_factor_exposure({"M": 1.0}, {"M": 1.0 * f["market"]}, f)
    hist = factor_historical_shocks(rep, f)
    assert "H3" in hist and "M" in hist["H3"]
    assert hist["H3"]["M"] < 0


def test_historical_skipped_when_date_absent():
    f = _factor_inputs()                                              # 2026말 끝 → H 일자 없음
    res = run_factor_stress_test({"M": 1.0}, {"M": 1.0 * f["market"]}, f)
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert not _by_id(res.stress)[h].evaluable                   # graceful 생략


# ── 3. 실측 우선 + 잔차화 정합 ────────────────────────────────────────────────
def test_actual_return_preferred_over_approximation():
    day = "2020-03-19"
    f = _factor_inputs_ending(day, shock={"market": -0.10})
    actual = (1.0 * f["market"]).copy()
    actual.loc[pd.Timestamp(day)] = -0.30                            # 실제 그날 종목 -30%(근사 -10%와 다르게)
    res = run_factor_stress_test({"M": 1.0}, {"M": actual}, f)
    h3 = _by_id(res.stress)["H3"]
    assert h3.evaluable
    assert abs(h3.portfolio_pnl - (-0.30)) < 1e-9                    # 팩터근사 아닌 실측 우선


def test_residualized_shock_matches_beta_units():
    """semi 잔차 베타 × 잔차화된 semi 충격 = H 손실. raw -15%를 썼다면 불일치할 값."""
    day = "2024-08-05"  # H5
    f = _factor_inputs_ending(day, seed=13, shock={"market": -0.06, "semi": -0.15})
    panel = build_factor_panel(f)
    # 순수 semi 잔차 베타 ≈ 1.0, 시장 노출 0 종목.
    y = pd.Series(
        1.0 * panel["semi"].to_numpy() + np.random.default_rng(1).normal(0, 0.0005, len(panel)),
        index=panel.index,
    )
    res = run_factor_stress_test({"Q": 1.0}, {"Q": y}, f)
    h5 = _by_id(res.stress)["H5"]
    assert h5.evaluable
    semi_resid_shock = float(panel.loc[pd.Timestamp(day)]["semi"])   # 잔차화된 그날 semi(≠ raw -0.15)
    assert abs(h5.portfolio_pnl - semi_resid_shock) < 0.02           # 잔차 단위로 정합
    assert abs(semi_resid_shock - (-0.15)) > 1e-3                    # raw와 실제로 다름을 입증


# ── 4. graceful (빈 포트 / market 없음) ───────────────────────────────────────
def test_empty_holdings_only_s3_s4():
    res = run_factor_stress_test({}, {}, _factor_inputs())
    ev = {r.scenario_id for r in res.stress.results if r.evaluable}
    assert ev == {"S3", "S4"}
    assert res.exposure.stock_betas == ()


def test_no_market_factor_graceful():
    f = _factor_inputs()
    dummy = f["fx"]
    del f["market"]                                                  # market 없음 → 패널 None
    res = run_factor_stress_test({"A": 0.5}, {"A": dummy}, f)
    assert res.exposure.stock_betas == ()
    ev = {r.scenario_id for r in res.stress.results if r.evaluable}
    assert ev == {"S3", "S4"}                                        # 베타 미주입 → S3 베타1 폴백


# ── 5. 결과 형태 ──────────────────────────────────────────────────────────────
def test_result_is_exposure_stress_pair():
    f = _factor_inputs(seed=5)
    res = run_factor_stress_test({"A": 0.4}, {"A": 1.2 * f["market"]}, f)
    assert isinstance(res, FactorStressResult)
    assert abs(res.exposure.covered_weight - 0.4) < 1e-9
    assert len(res.stress.results) == 10
    assert "KOSPI" in res.exposure.interpretation
