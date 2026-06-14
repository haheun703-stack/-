"""vol_targeting (§4.3 변동성 타겟팅) 검증 — RISK_ENGINE Phase 3.

불변식:
  1. 고변동성(연>15%) → scale<1.0(축소); 저변동성(<15%) → scale=1.0(레버리지 확대 금지, min cap).
  2. scale = min(1.0, target/realized) 공식 일치.
  3. None/표본<20 → 미평가, scale=1.0(중립). 상수(변동성0) → 평가, scale=1.0.
  4. scale은 어떤 입력에도 ≤ 1.0.
"""
import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG
from risk.vol_targeting import vol_target_scale


def _series(arr):
    idx = pd.bdate_range(end="2026-06-12", periods=len(arr))
    return pd.Series(arr, index=idx)


def _normal_returns(sigma, n=120, seed=0):
    return _series(np.random.default_rng(seed).normal(0.0, sigma, n))


def test_high_vol_scales_down():
    st = vol_target_scale(_normal_returns(0.03))  # 연 ~47% >> 15%
    assert st.evaluable
    assert st.scale < 0.5
    assert st.realized_vol_annual > 0.30


def test_low_vol_capped_at_one():
    st = vol_target_scale(_normal_returns(0.004))  # 연 ~6% < 15%
    assert st.evaluable
    assert st.scale == 1.0  # min cap — 레버리지 확대 안 함


def test_scale_formula_matches():
    st = vol_target_scale(_normal_returns(0.02, seed=3))
    expected = min(1.0, RISK_CONFIG.target_vol_annual / st.realized_vol_annual)
    assert abs(st.scale - expected) < 1e-12


def test_none_neutral():
    st = vol_target_scale(None)
    assert not st.evaluable
    assert st.scale == 1.0
    assert st.realized_vol_annual is None


def test_insufficient_obs_neutral():
    st = vol_target_scale(_normal_returns(0.03, n=10))  # 10 < 20
    assert not st.evaluable
    assert st.scale == 1.0


def test_constant_returns_neutral_scale():
    st = vol_target_scale(_series([0.001] * 60))  # 변동성 0
    assert st.evaluable
    assert st.realized_vol_annual == 0.0
    assert st.scale == 1.0


def test_scale_never_exceeds_one():
    for sigma in (0.001, 0.005, 0.01, 0.02, 0.05):
        st = vol_target_scale(_normal_returns(sigma, seed=7))
        assert st.scale <= 1.0
