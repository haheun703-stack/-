"""crowding (§4.4 동질화 모니터) 검증 — RISK_ENGINE Phase 4.

불변식:
  1. C1 고상관 보유 → 경고; 저상관 → 아님; 1종목/공통표본<60 → 미평가.
  2. C2 VKOSPI 20 초과 AND 5일 30%↑ '동시'만 경고; 한쪽만 → 아님; 미주입/표본<6 → 미평가.
  3. C3 외인선물 2년 분위수 상/하위 5% → 경고; 중앙 → 아님; 표본<252 → 미평가.
  4. 평가 가능 경고 ≥2 → gross_exposure_mult 0.80(-20%p); 미만 → 1.0; 3개여도 -20%p 고정.
  5. evaluable=False는 warning_count에서 제외(데이터 없음 ≠ 위험, 과경고 차단 = freeze 일관).
"""
import numpy as np
import pandas as pd

from risk.crowding import crowding_state


def _series(arr):
    idx = pd.bdate_range(end="2026-06-12", periods=len(arr))
    return pd.Series(arr, index=idx)


def _correlated_returns(n=80, k=3, idio=0.002, seed=0):
    """k개 종목, 공유 shock 지배(idio 작음) → ρ≈0.99 고상관."""
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 0.02, n)
    return {
        f"T{i}": _series(shock + np.random.default_rng(100 + i).normal(0.0, idio, n))
        for i in range(k)
    }


def _independent_returns(n=80, k=3):
    return {
        f"T{i}": _series(np.random.default_rng(200 + i).normal(0.0, 0.02, n))
        for i in range(k)
    }


def _ff_series(current, n=300, seed=3):
    """외인선물 순포지션 — 균등분포 배경[-100,100] + 마지막에 current 값."""
    rng = np.random.default_rng(seed)
    body = rng.uniform(-100.0, 100.0, n - 1)
    return _series(np.append(body, current))


# 5일전 18 → 현재 25: +38.9%(>30%) & 25>20 → C2 발동용 시계열
_VK_SURGE = [18.0, 19.0, 20.0, 22.0, 23.0, 25.0]


# ── C1: 평균 쌍상관 ──────────────────────────────────────────────────────────
def test_c1_high_corr_triggers():
    st = crowding_state(holding_returns=_correlated_returns())
    assert st.c1_evaluable
    assert st.c1_value > 0.70
    assert st.c1_triggered


def test_c1_low_corr_not_triggered():
    st = crowding_state(holding_returns=_independent_returns())
    assert st.c1_evaluable
    assert st.c1_value < 0.70
    assert not st.c1_triggered


def test_c1_single_holding_not_evaluable():
    st = crowding_state(holding_returns={"T0": _series(np.random.default_rng(1).normal(0, 0.02, 80))})
    assert not st.c1_evaluable
    assert st.c1_value is None
    assert not st.c1_triggered


def test_c1_insufficient_obs_not_evaluable():
    st = crowding_state(holding_returns=_correlated_returns(n=40))  # 공통 40 < 60
    assert not st.c1_evaluable
    assert st.c1_value is None


# ── C2: VKOSPI 레짐 전환 ─────────────────────────────────────────────────────
def test_c2_surge_and_level_triggers():
    st = crowding_state(vkospi_series=_series(_VK_SURGE))
    assert st.c2_evaluable
    assert st.c2_value > 0.30
    assert st.c2_triggered


def test_c2_level_without_surge_not_triggered():
    # 현재 25>20 이지만 5일전 24 → +4.2%(<30%)
    st = crowding_state(vkospi_series=_series([24.0, 24.0, 24.0, 24.5, 24.8, 25.0]))
    assert st.c2_evaluable
    assert not st.c2_triggered


def test_c2_surge_without_level_not_triggered():
    # 5일전 10 → 현재 14(+40% 급등)이지만 14<20
    st = crowding_state(vkospi_series=_series([10.0, 11.0, 12.0, 13.0, 13.5, 14.0]))
    assert st.c2_evaluable
    assert st.c2_value > 0.30
    assert not st.c2_triggered


def test_c2_exact_surge_boundary_triggers():
    # 5일전 20 → 현재 26 = +30% 정확히(>=) & 26>20
    st = crowding_state(vkospi_series=_series([20.0, 21.0, 22.0, 23.0, 24.0, 26.0]))
    assert abs(st.c2_value - 0.30) < 1e-9
    assert st.c2_triggered


def test_c2_none_not_evaluable():
    st = crowding_state(vkospi_series=None)
    assert not st.c2_evaluable
    assert not st.c2_triggered
    assert st.c2_value is None


def test_c2_insufficient_not_evaluable():
    st = crowding_state(vkospi_series=_series([20.0, 21.0, 22.0]))  # len 3 < 6
    assert not st.c2_evaluable


# ── C3: 외인선물 쏠림 ────────────────────────────────────────────────────────
def test_c3_extreme_high_triggers():
    st = crowding_state(foreign_futures_series=_ff_series(current=200.0))  # 배경 최대 초과
    assert st.c3_evaluable
    assert st.c3_value >= 0.95
    assert st.c3_triggered


def test_c3_extreme_low_triggers():
    st = crowding_state(foreign_futures_series=_ff_series(current=-200.0))
    assert st.c3_evaluable
    assert st.c3_value <= 0.05
    assert st.c3_triggered


def test_c3_central_not_triggered():
    st = crowding_state(foreign_futures_series=_ff_series(current=0.0))
    assert st.c3_evaluable
    assert 0.05 < st.c3_value < 0.95
    assert not st.c3_triggered


def test_c3_insufficient_not_evaluable():
    st = crowding_state(foreign_futures_series=_ff_series(current=0.0, n=100))  # 100 < 252
    assert not st.c3_evaluable
    assert st.c3_value is None


def test_c3_constant_series_not_evaluable():
    # 상수 시계열: 분위수 무의미인데 (s<=current).mean()=1.0 → 거짓 경고 방지(미평가 처리)
    st = crowding_state(foreign_futures_series=_series([50.0] * 300))
    assert not st.c3_evaluable
    assert st.c3_value is None
    assert not st.c3_triggered


# ── 집계: 노출 조정 ──────────────────────────────────────────────────────────
def test_two_warnings_haircut_20():
    st = crowding_state(holding_returns=_correlated_returns(), vkospi_series=_series(_VK_SURGE))
    assert st.warning_count == 2
    assert st.exposure_haircut == 0.20
    assert st.gross_exposure_mult == 0.80


def test_one_warning_no_haircut():
    st = crowding_state(holding_returns=_correlated_returns())  # C1만
    assert st.warning_count == 1
    assert st.exposure_haircut == 0.0
    assert st.gross_exposure_mult == 1.0


def test_three_warnings_haircut_fixed_at_20():
    st = crowding_state(
        holding_returns=_correlated_returns(),
        vkospi_series=_series(_VK_SURGE),
        foreign_futures_series=_ff_series(current=200.0),
    )
    assert st.warning_count == 3
    assert st.gross_exposure_mult == 0.80  # 경고 2개 이상 → -20%p 고정(누적 아님)


def test_no_data_full_exposure():
    st = crowding_state()  # 전부 None
    assert not (st.c1_evaluable or st.c2_evaluable or st.c3_evaluable)
    assert st.warning_count == 0
    assert st.gross_exposure_mult == 1.0


def test_evaluable_warning_with_unevaluable_no_haircut():
    # C1 경고(평가) + C2/C3 미주입(미평가) → 경고 1개 → haircut 없음(미평가는 안 셈)
    st = crowding_state(holding_returns=_correlated_returns())
    assert st.c1_triggered and not st.c2_evaluable and not st.c3_evaluable
    assert st.warning_count == 1
    assert st.gross_exposure_mult == 1.0
