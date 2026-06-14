"""risk/factor_exposure — EWMA 팩터 회귀(§3.1).

엔진 불변식:
  1. 합성 종목 r = 1.5·market + 0.8·semi + 잡음 → market≈1.5, semi≈0.8 회복.
  2. smallcap·semi는 시장 잔차화 → market과 무상관 합성 시 베타 분리.
  3. 표본 < factor_min_obs / market 팩터 없음 → graceful(None/빈 리포트).
  4. EWMA vs 252 단순 market 베타 괴리 ≥50% → unstable=True.
  5. 포트 노출 = Σ weight×beta, covered_weight로 커버리지 노출.
  6. betas_for("fx"/"semi"/"market") → stress_test 주입 → S1/S2/S3 실평가.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG
from risk.factor_exposure import (
    FACTORS,
    build_factor_panel,
    compute_factor_exposure,
    estimate_stock_betas,
)
from risk.stress_test import run_stress_test

_ASOF = date(2026, 6, 12)


def _series(arr) -> pd.Series:
    idx = pd.bdate_range(end=pd.Timestamp(_ASOF), periods=len(arr))
    return pd.Series(np.asarray(arr, dtype=float), index=idx)


def _factor_panel_inputs(n=250, seed=7):
    """공통 인덱스의 독립 팩터 수익률 5종."""
    rng = np.random.default_rng(seed)
    return {
        "market": _series(rng.normal(0, 0.012, n)),
        "smallcap": _series(rng.normal(0, 0.014, n)),
        "fx": _series(rng.normal(0, 0.005, n)),
        "semi": _series(rng.normal(0, 0.018, n)),
        "rate": _series(rng.normal(0, 0.003, n)),
    }


# ── 1. 베타 회복 ──────────────────────────────────────────────────────────────
def test_recovers_known_betas():
    f = _factor_panel_inputs()
    panel = build_factor_panel(f)
    rng = np.random.default_rng(99)
    # 잔차화 전 원 팩터로 합성하되, market과 semi는 독립 생성이라 잔차≈원본.
    y = 1.5 * f["market"] + 0.8 * f["semi"] + _series(rng.normal(0, 0.001, len(panel)))
    sb = estimate_stock_betas("X", y, panel)
    assert sb is not None
    assert abs(sb.betas["market"] - 1.5) < 0.1
    assert abs(sb.betas["semi"] - 0.8) < 0.1
    assert abs(sb.betas["fx"]) < 0.1       # 노출 없는 팩터 ≈ 0
    assert abs(sb.betas["rate"]) < 0.1


def test_pure_market_bet():
    f = _factor_panel_inputs(seed=3)
    panel = build_factor_panel(f)
    y = 1.0 * f["market"]
    sb = estimate_stock_betas("M", y, panel)
    assert abs(sb.betas["market"] - 1.0) < 0.05
    assert abs(sb.betas["semi"]) < 0.05


# ── 2. 잔차화 ─────────────────────────────────────────────────────────────────
def test_residualized_factors_orthogonal_to_market():
    f = _factor_panel_inputs(seed=11)
    panel = build_factor_panel(f)
    # 잔차화된 smallcap·semi는 market과 거의 무상관이어야 한다.
    corr_semi = np.corrcoef(panel["semi"], panel["market"])[0, 1]
    corr_small = np.corrcoef(panel["smallcap"], panel["market"])[0, 1]
    assert abs(corr_semi) < 1e-6
    assert abs(corr_small) < 1e-6


def test_panel_none_without_market():
    f = _factor_panel_inputs()
    del f["market"]
    assert build_factor_panel(f) is None


def test_panel_missing_factor_filled_zero():
    f = _factor_panel_inputs()
    del f["rate"]  # 누락 팩터
    panel = build_factor_panel(f)
    assert panel is not None
    assert "rate" in panel.columns
    assert float(np.sum(np.abs(panel["rate"]))) == 0.0


# ── 3. graceful ───────────────────────────────────────────────────────────────
def test_short_history_returns_none():
    f = _factor_panel_inputs(n=40)  # < factor_min_obs(60)
    panel = build_factor_panel(f)
    y = f["market"] * 1.0
    assert estimate_stock_betas("S", y, panel) is None


def test_empty_holdings_empty_report():
    f = _factor_panel_inputs()
    rep = compute_factor_exposure({}, {}, f)
    assert rep.stock_betas == ()
    assert rep.covered_weight == 0.0
    assert "불가" in rep.interpretation


def test_missing_returns_skipped():
    f = _factor_panel_inputs()
    panel_y = 1.2 * f["market"]
    rep = compute_factor_exposure(
        {"A": 0.1, "B": 0.1},
        {"A": panel_y},          # B는 수익률 없음 → 생략
        f,
    )
    tickers = {sb.ticker for sb in rep.stock_betas}
    assert tickers == {"A"}
    assert abs(rep.covered_weight - 0.1) < 1e-9


# ── 4. 불안정 플래그 ──────────────────────────────────────────────────────────
def test_unstable_when_recent_beta_diverges():
    f = _factor_panel_inputs(n=252, seed=21)
    panel = build_factor_panel(f)
    m = f["market"].to_numpy()
    # 과거 절반 market 베타 0.2, 최근 절반 2.0 → EWMA(최근가중) vs 252단순 괴리 큼.
    half = len(m) // 2
    coef = np.concatenate([np.full(half, 0.2), np.full(len(m) - half, 2.0)])
    y = _series(coef * m + np.random.default_rng(5).normal(0, 0.0005, len(m)))
    sb = estimate_stock_betas("U", y, panel)
    assert sb is not None
    assert sb.unstable is True


def test_stable_when_constant_beta():
    f = _factor_panel_inputs(n=252, seed=22)
    panel = build_factor_panel(f)
    y = 1.3 * f["market"] + _series(np.random.default_rng(6).normal(0, 0.0005, len(panel)))
    sb = estimate_stock_betas("St", y, panel)
    assert sb.unstable is False


# ── 5. 포트 집계 ──────────────────────────────────────────────────────────────
def test_portfolio_aggregation():
    f = _factor_panel_inputs(seed=33)
    a = 1.0 * f["market"]
    b = 2.0 * f["market"]
    rep = compute_factor_exposure(
        {"A": 0.3, "B": 0.2},
        {"A": a, "B": b},
        f,
    )
    # 0.3·1.0 + 0.2·2.0 = 0.7
    assert abs(rep.portfolio["market"] - 0.7) < 0.05
    assert abs(rep.covered_weight - 0.5) < 1e-9
    assert "KOSPI" in rep.interpretation


# ── 6. stress_test 통합 ───────────────────────────────────────────────────────
def test_betas_feed_stress_test():
    f = _factor_panel_inputs(seed=44)
    # semi에 강하게 노출된 종목 → S2(반도체 -15%)에서 큰 손실이 평가돼야.
    y = 0.5 * f["market"] + 1.0 * f["semi"]
    holdings = {"S": 0.5}
    rep = compute_factor_exposure(holdings, {"S": y}, f)
    semi_betas = rep.betas_for("semi")
    market_betas = rep.betas_for("market")
    assert "S" in semi_betas

    # factor_exposure 주입 전: S1/S2 미평가(베타 미주입).
    before = run_stress_test(holdings)
    s2_before = next(r for r in before.results if r.scenario_id == "S2")
    assert s2_before.evaluable is False

    # 주입 후: S2 평가 + 손실(음수).
    after = run_stress_test(
        holdings,
        fx_betas=rep.betas_for("fx"),
        semi_betas=semi_betas,
        market_betas=market_betas,
    )
    s2_after = next(r for r in after.results if r.scenario_id == "S2")
    assert s2_after.evaluable is True
    assert s2_after.portfolio_pnl < 0  # 반도체 양의 베타 × -15% = 손실


def test_betas_for_unknown_factor_empty():
    f = _factor_panel_inputs()
    y = 1.0 * f["market"]
    rep = compute_factor_exposure({"A": 0.1}, {"A": y}, f)
    assert rep.betas_for("nonexistent") == {}
