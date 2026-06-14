"""stress_test (§4.1 시나리오 라이브러리) 검증 — RISK_ENGINE Phase 3.

불변식:
  1. S3(시장갭)·S4(하한가)는 비중만으로 항상 평가; S3은 시장 베타 1 폴백.
  2. S1/S2/S5·H1~H5는 베타/충격 주입 시 평가, 미주입 → 미평가(과소평가 방지).
  3. _portfolio_impact: 보유 종목 충격/베타 일부 누락 → 미평가(None).
  4. worst = 평가 가능 시나리오 중 최소 pnl.
  5. 빈 포트 → 평가 가능 시나리오 손실 0.
"""
from risk.stress_test import run_stress_test


def _by_id(report):
    return {r.scenario_id: r for r in report.results}


# ── S3 시장 갭 ───────────────────────────────────────────────────────────────
def test_s3_market_beta_one_fallback():
    rep = run_stress_test({"A": 0.6, "B": 0.4})  # gross 1.0, 베타 미주입
    s3 = _by_id(rep)["S3"]
    assert s3.evaluable
    assert abs(s3.portfolio_pnl - (-0.08)) < 1e-12  # -8% × gross 1.0
    assert "베타 1" in s3.note


def test_s3_scales_with_gross():
    rep = run_stress_test({"A": 0.3, "B": 0.2})  # gross 0.5
    assert abs(_by_id(rep)["S3"].portfolio_pnl - (-0.04)) < 1e-12


def test_s3_with_market_betas():
    rep = run_stress_test({"A": 0.5, "B": 0.5}, market_betas={"A": 1.2, "B": 0.8})
    # (0.5×1.2 + 0.5×0.8) × -0.08 = 1.0 × -0.08 = -0.08
    assert abs(_by_id(rep)["S3"].portfolio_pnl - (-0.08)) < 1e-12


def test_s3_partial_market_betas_not_evaluable():
    rep = run_stress_test({"A": 0.5, "B": 0.5}, market_betas={"A": 1.0})  # B 누락
    assert not _by_id(rep)["S3"].evaluable


# ── S4 하한가 ────────────────────────────────────────────────────────────────
def test_s4_max_weight_limit_down():
    rep = run_stress_test({"A": 0.5, "B": 0.3, "C": 0.2})
    s4 = _by_id(rep)["S4"]
    assert s4.evaluable
    assert abs(s4.portfolio_pnl - (0.5 * -0.30)) < 1e-12  # -0.15
    assert "A" in s4.note


def test_s4_empty_zero():
    assert _by_id(run_stress_test({}))["S4"].portfolio_pnl == 0.0


# ── S1/S2 팩터 ───────────────────────────────────────────────────────────────
def test_s1_fx_betas():
    rep = run_stress_test({"A": 0.5, "B": 0.5}, fx_betas={"A": 0.3, "B": -0.1})
    # (0.5×0.3 + 0.5×-0.1) × 0.07 = 0.1 × 0.07 = 0.007
    assert abs(_by_id(rep)["S1"].portfolio_pnl - 0.007) < 1e-12


def test_s1_not_evaluable_without_betas():
    s1 = _by_id(run_stress_test({"A": 1.0}))["S1"]
    assert not s1.evaluable
    assert s1.portfolio_pnl is None


def test_s2_semi_betas():
    rep = run_stress_test({"A": 1.0}, semi_betas={"A": 1.0})
    assert abs(_by_id(rep)["S2"].portfolio_pnl - (-0.15)) < 1e-12


def test_s2_not_evaluable_without_betas():
    assert not _by_id(run_stress_test({"A": 1.0}))["S2"].evaluable


# ── S5 복합 ──────────────────────────────────────────────────────────────────
def test_s5_combines_s1_s2():
    rep = run_stress_test({"A": 1.0}, fx_betas={"A": 1.0}, semi_betas={"A": 1.0})
    # S1 = 0.07, S2 = -0.15, S5 = -0.08
    assert abs(_by_id(rep)["S5"].portfolio_pnl - (-0.08)) < 1e-12


def test_s5_not_evaluable_if_one_missing():
    rep = run_stress_test({"A": 1.0}, fx_betas={"A": 1.0})  # semi 없음
    assert not _by_id(rep)["S5"].evaluable


# ── H 역사 재생 ──────────────────────────────────────────────────────────────
def test_historical_with_shocks():
    rep = run_stress_test({"A": 0.5, "B": 0.5},
                          historical_shocks={"H3": {"A": -0.10, "B": -0.20}})
    h3 = _by_id(rep)["H3"]
    assert h3.evaluable
    assert abs(h3.portfolio_pnl - (-0.15)) < 1e-12  # 0.5×-0.1 + 0.5×-0.2


def test_historical_not_evaluable_without_shocks():
    rep = run_stress_test({"A": 1.0})
    assert all(not _by_id(rep)[h].evaluable for h in ("H1", "H2", "H3", "H4", "H5"))


def test_historical_partial_shocks_not_evaluable():
    rep = run_stress_test({"A": 0.5, "B": 0.5}, historical_shocks={"H1": {"A": -0.1}})  # B 누락
    assert not _by_id(rep)["H1"].evaluable


# ── worst + 기본 상태 ────────────────────────────────────────────────────────
def test_worst_is_minimum_pnl():
    rep = run_stress_test({"A": 0.5, "B": 0.5},
                          historical_shocks={"H1": {"A": -0.30, "B": -0.30}})
    # H1 -0.30이 S3(-0.08)·S4(0.5×-0.3=-0.15)보다 나쁨
    assert rep.worst.scenario_id == "H1"
    assert abs(rep.worst.portfolio_pnl - (-0.30)) < 1e-12


def test_default_only_s3_s4_evaluable():
    # production 기본(베타·역사 미주입): S3(폴백)·S4만 평가, 나머지 미평가
    rep = run_stress_test({"A": 0.6, "B": 0.4})
    ev = {r.scenario_id for r in rep.results if r.evaluable}
    assert ev == {"S3", "S4"}
    # S3 -0.08, S4 0.6×-0.30 = -0.18 → worst S4
    assert rep.worst.scenario_id == "S4"
    assert abs(rep.worst.portfolio_pnl - (-0.18)) < 1e-12


def test_report_has_ten_scenarios():
    rep = run_stress_test({"A": 1.0})
    assert len(rep.results) == 10
    assert {r.scenario_id for r in rep.results} == {
        "H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "S4", "S5"
    }
