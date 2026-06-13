"""pre_trade_gate G7 (Component VaR) 통합 — RISK_ENGINE Phase 4 (§3.5).

returns_by_ticker 주입 시 G7(신규 종목 Component VaR 기여 ≤ component_var_limit)이 RESIZE 루프에
통합되는지 검증. returns=None이면 not_active(Phase 1a 호환).

불변식:
  1. 분산된 포트(신규 기여 낮음) → G7 pass.
  2. 신규 고변동 종목이 리스크 지배(>25% 기여) → RESIZE/REJECT + G7 사유 감사.
  3. 단일 포지션(신규만) → G7 pass(single_position_no_basis, 부트스트랩 과차단 방지).
  4. returns 미주입 → G7 not_active.
  5. VaR 데이터 부족 → G7 fail-closed(G1과 동시).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG
from risk.pre_trade_gate import GateRequest, Holding, evaluate_pre_trade


def _returns(sigma, n=400, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, sigma, n), index=idx)


def _req(size, equity=100_000_000, ticker="NEW", sector="NEW_SEC") -> GateRequest:
    # adv20 매우 큼 → G6 통과시켜 G7만 검증
    return GateRequest(ticker=ticker, sector=sector, proposed_size_krw=size,
                       equity_krw=equity, adv20_krw=1e15)


def _diversified_holdings():
    # 5개 저변동 보유, 서로 다른 섹터(G4 회피) — 신규와 함께 분산 기반(n_positions≥2) 형성
    return [Holding(ticker=f"H{i}", value_krw=10_000_000, sector=f"S{i}") for i in range(5)]


def _low_vol_returns():
    return {f"H{i}": _returns(0.005, seed=100 + i) for i in range(5)}


def test_g7_diversified_small_contribution_passes():
    # 신규가 보유들과 동질(저변동)·소비중 → 기여 ≤ 25% → PASS
    rets = _low_vol_returns()
    rets["NEW"] = _returns(0.005, seed=1)
    res = evaluate_pre_trade(_req(5_000_000), _diversified_holdings(), returns_by_ticker=rets)
    assert res.checks["G7"]["status"] == "pass", (res.verdict, res.checks.get("G7"))
    assert res.checks["G7"]["n_positions"] == 6
    assert res.checks["G7"]["contribution"] <= RISK_CONFIG.component_var_limit + 1e-9


def test_g7_dominant_new_position_resizes():
    # 신규 고변동(10배)이 리스크 지배 → 기여 >> 25% → RESIZE(축소) 후 한도 내 수렴
    rets = _low_vol_returns()
    rets["NEW"] = _returns(0.05, seed=1)  # 10배 변동성
    res = evaluate_pre_trade(_req(10_000_000), _diversified_holdings(), returns_by_ticker=rets)
    assert res.verdict in ("RESIZE", "REJECT"), (res.verdict, res.checks.get("G7"))
    # G7이 축소를 유발했음이 감사 이력에 남아야 (G1도 같이 잡힐 수 있음)
    assert any(v.get("gate") == "G7" for v in res.violations), res.violations
    if res.verdict == "RESIZE":
        assert res.final_size_krw < 10_000_000
        assert res.checks["G7"]["status"] == "pass"  # 축소 후 한도 내 수렴
        assert res.checks["G7"]["contribution"] <= RISK_CONFIG.component_var_limit + 0.02


def test_g7_single_position_no_basis_passes():
    # 보유 없음 + 신규만 → 분산 대상 없음 → G7 pass(부트스트랩 과차단 방지)
    res = evaluate_pre_trade(_req(5_000_000), [], returns_by_ticker={"NEW": _returns(0.05)})
    assert res.checks["G7"]["status"] == "pass"
    assert res.checks["G7"]["reason"] == "single_position_no_basis"
    assert res.checks["G7"]["n_positions"] == 1


def test_g7_not_active_without_returns():
    # returns 미주입 → G7 not_active (Phase 1a 호환)
    res = evaluate_pre_trade(_req(5_000_000), _diversified_holdings())
    assert "not_active" in res.checks["G7"]["status"]


def test_g7_insufficient_data_fail_closed():
    # 신규 표본 40 < 60 → 게이트가 returns에서 못 거름(직접 주입) → G1/G7 동시 fail-closed REJECT
    rets = {"NEW": _returns(0.02, n=40)}
    res = evaluate_pre_trade(_req(1_000_000), [], returns_by_ticker=rets)
    assert res.verdict == "REJECT"
    assert res.checks["G7"]["data_ok"] is False
    assert res.checks["G7"]["status"] == "violation"


def test_g7_audit_records_contribution():
    # 감사: 활성 G7은 contribution/limit/n_positions를 checks에 남긴다(사후 추적)
    rets = _low_vol_returns()
    rets["NEW"] = _returns(0.005, seed=1)
    res = evaluate_pre_trade(_req(3_000_000), _diversified_holdings(), returns_by_ticker=rets)
    g7 = res.checks["G7"]
    assert "contribution" in g7 and "limit" in g7 and "n_positions" in g7
    assert g7["limit"] == RISK_CONFIG.component_var_limit
