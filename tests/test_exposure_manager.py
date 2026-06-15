"""risk/exposure_manager — L3 노출 관리 계층 계수 합성(§4.2·§4.3·§4.4).

불변식:
  1. 세 계수 곱셈; 컴포넌트 None(미주입) → 그 축 1.0(과조절 방지).
  2. 전부 미주입 → 최종 1.0(현행 노출 불변 = freeze 안전).
  3. 각 ≤ 1.0 → 곱도 ≤ 1.0(축소 전용, 레버리지 확대 금지).
  4. ★사다리는 gross_exposure(총노출)를 쓴다 — new_size_mult(G8 게이트 영역)가 아님.
  5. 사다리 step3(DD -10% 초과) → 0.0 + to_kill_switch.
  6. base 주입 → target_exposure_krw = base × 최종 계수.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.crowding import CrowdingState
from risk.drawdown_ladder import LadderState
from risk.exposure_manager import (
    combine_exposure_multipliers,
    compute_exposure_plan,
)
from risk.vol_targeting import VolTargetState

_ASOF = pd.Timestamp("2026-06-12")


def _ladder(gross, *, kill=False, step=1):
    return LadderState(step=step, gross_exposure=gross, new_entry_allowed=not kill,
                       new_size_mult=0.5, to_kill_switch=kill, dd=-0.05)


def _vol(scale):
    return VolTargetState(evaluable=True, realized_vol_annual=0.2,
                          target_vol_annual=0.15, scale=scale)


def _crowd(mult):
    return CrowdingState(
        c1_triggered=mult < 1.0, c1_evaluable=True, c1_value=0.8,
        c2_triggered=mult < 1.0, c2_evaluable=True, c2_value=0.4,
        c3_triggered=False, c3_evaluable=False, c3_value=None,
        warning_count=0 if mult >= 1.0 else 2,
        exposure_haircut=round(1.0 - mult, 4), gross_exposure_mult=mult,
    )


# ── 1. combine 단위(None 중립 + 곱셈) ─────────────────────────────────────────
def test_all_none_neutral():
    assert combine_exposure_multipliers(None, None, None) == (1.0, 1.0, 1.0, 1.0)


def test_product_of_three():
    lm, vs, cm, final = combine_exposure_multipliers(_ladder(0.7), _vol(0.8), _crowd(0.8))
    assert (lm, vs, cm) == (0.7, 0.8, 0.8)
    assert abs(final - 0.7 * 0.8 * 0.8) < 1e-12


def test_each_axis_independent():
    assert combine_exposure_multipliers(_ladder(0.4), None, None)[3] == 0.4
    assert combine_exposure_multipliers(None, _vol(0.5), None)[3] == 0.5
    assert combine_exposure_multipliers(None, None, _crowd(0.8))[3] == 0.8


def test_product_never_exceeds_one():
    assert combine_exposure_multipliers(_ladder(1.0), _vol(1.0), _crowd(1.0))[3] == 1.0


# ── 2. compute_exposure_plan 통합 ─────────────────────────────────────────────
def test_plan_all_absent_is_neutral():
    plan = compute_exposure_plan()
    assert plan.gross_exposure_mult == 1.0
    assert plan.target_exposure_krw is None
    assert not plan.to_kill_switch
    assert "조절 없음" in plan.interpretation


def test_plan_ladder_from_dd():
    plan = compute_exposure_plan(current_dd=-0.05)        # -4~-7% → step1 → 0.7
    assert plan.ladder_mult == 0.7
    assert abs(plan.gross_exposure_mult - 0.7) < 1e-12


def test_ladder_uses_gross_not_new_size():
    """★사다리 step1: gross_exposure 0.7(노출관리) vs new_size_mult 0.5(G8). 0.7을 써야 한다."""
    plan = compute_exposure_plan(current_dd=-0.05)
    assert plan.ladder_mult == 0.7                        # NOT 0.5(new_size_mult = 게이트 G8 영역)


def test_plan_kill_switch_step3():
    plan = compute_exposure_plan(current_dd=-0.11)        # -10% 초과 → 0.0 + kill
    assert plan.gross_exposure_mult == 0.0
    assert plan.to_kill_switch
    assert "킬스위치" in plan.interpretation


def test_plan_base_scaled():
    plan = compute_exposure_plan(base_exposure_krw=10_000_000, current_dd=-0.05)
    assert abs(plan.target_exposure_krw - 7_000_000) < 1e-6


def test_plan_high_vol_reduces():
    idx = pd.bdate_range(end=_ASOF, periods=60)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.05, 60), index=idx)  # 일 5% = 초고변동
    plan = compute_exposure_plan(portfolio_returns=r)
    assert plan.vol_scale < 1.0
    assert plan.gross_exposure_mult < 1.0


def test_plan_crowding_two_warnings():
    idx = pd.bdate_range(end=_ASOF, periods=70)
    base = np.random.default_rng(2).normal(0, 0.02, 70)
    a = pd.Series(base, index=idx)
    b = pd.Series(base + np.random.default_rng(3).normal(0, 0.0005, 70), index=idx)  # 고상관(C1)
    vk = pd.Series([15.0] * 64 + [16, 17, 18, 19, 20, 22], index=idx)  # C2: past16→cur22(+37%), >20
    plan = compute_exposure_plan(holding_returns={"A": a, "B": b}, vkospi_series=vk)
    assert plan.crowding_mult == 0.80                    # 경고 2개 → -20%p
    assert abs(plan.gross_exposure_mult - 0.80) < 1e-12


def test_plan_combined_product():
    """사다리 + 고변동성 + 크라우딩 동시 → 세 계수 곱 = 최종(분해 일관)."""
    idx = pd.bdate_range(end=_ASOF, periods=70)
    base = np.random.default_rng(4).normal(0, 0.04, 70)   # 고변동
    a = pd.Series(base, index=idx)
    b = pd.Series(base + np.random.default_rng(5).normal(0, 0.0005, 70), index=idx)
    vk = pd.Series([15.0] * 64 + [16, 17, 18, 19, 20, 22], index=idx)
    plan = compute_exposure_plan(
        current_dd=-0.05, portfolio_returns=a,
        holding_returns={"A": a, "B": b}, vkospi_series=vk,
    )
    assert plan.ladder_mult == 0.7
    assert plan.crowding_mult == 0.80
    assert plan.vol_scale < 1.0
    expected = plan.ladder_mult * plan.vol_scale * plan.crowding_mult
    assert abs(plan.gross_exposure_mult - expected) < 1e-12
    assert plan.gross_exposure_mult <= 1.0
