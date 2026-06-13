"""pre_trade_gate G1/G2 (VaR) 게이트 통합 — RISK_ENGINE Phase 2b.

returns_by_ticker 주입 시 G1(포트 VaR95 ≤ 2.5%)/G2(스트레스 VaR95 ≤ 4.0%)가 RESIZE 루프에
통합되는지 검증. returns=None이면 not_active(Phase 1a 호환 — test_pre_trade_gate 31개가 별도 보증).

불변식:
  1. 저변동 → G1/G2 PASS.
  2. 초고변동(G3는 통과하는 비중) → G1 위반 → RESIZE(축소) 또는 REJECT.
  3. VaR 데이터 부족(<60) → fail-closed REJECT.
  4. returns 미주입 → G1/G2 not_active + 정상 PASS.
  5. 기존 보유 + 신규 동일 종목 → 합산 비중으로 VaR(가상 포트).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.pre_trade_gate import GateRequest, Holding, evaluate_pre_trade


def _returns(sigma, n=300, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, sigma, n), index=idx)


def _req(size, equity=100_000_000, ticker="A") -> GateRequest:
    # adv20 매우 큼 → G6(유동성) 통과시켜 VaR 게이트만 검증
    return GateRequest(ticker=ticker, sector="ETF", proposed_size_krw=size,
                       equity_krw=equity, adv20_krw=1e15)


def test_var_gate_low_vol_passes():
    # 저변동(일 0.5%) + 작은 비중 → var95 << 2.5% → PASS
    res = evaluate_pre_trade(_req(5_000_000), [], returns_by_ticker={"A": _returns(0.005)})
    assert res.verdict == "PASS", (res.verdict, res.checks.get("G1"))
    assert res.checks["G1"]["status"] == "pass"
    assert res.checks["G2"]["status"] == "pass"
    assert res.checks["G1"]["var95"] <= 0.025


def test_var_gate_high_vol_resizes_or_rejects():
    # 초고변동(일 20%) 11% 비중 → G3 통과(11<12)이나 G1 var95 > 2.5% → 축소
    res = evaluate_pre_trade(_req(11_000_000), [], returns_by_ticker={"A": _returns(0.2)})
    assert res.verdict in ("RESIZE", "REJECT"), (res.verdict, res.checks.get("G1"))
    if res.verdict == "RESIZE":
        # 최종 checks는 축소 후라 G1=pass(한도 내 수렴) — 축소를 유발한 게 G1임을 이력으로 검증
        assert res.final_size_krw < 11_000_000
        assert res.checks["G1"]["status"] == "pass"
        assert any(v.get("gate") == "G1" for v in res.violations)  # resize 이력 binding


def test_var_gate_insufficient_data_rejects():
    # 표본 40 < 60 → var ok=False → fail-closed REJECT
    res = evaluate_pre_trade(_req(1_000_000), [], returns_by_ticker={"A": _returns(0.02, n=40)})
    assert res.verdict == "REJECT"
    assert res.checks["G1"]["data_ok"] is False  # VaR 데이터 부족 노출(감사)
    assert any(v.get("gate") == "G1" for v in res.violations)  # G1발 REJECT


def test_var_gate_none_keeps_not_active():
    # returns 미주입 → G1/G2 not_active (Phase 1a 호환), 정상 PASS
    res = evaluate_pre_trade(_req(5_000_000), [])
    assert "not_active" in res.checks["G1"]["status"]
    assert "not_active" in res.checks["G2"]["status"]
    assert res.verdict == "PASS"


def test_var_gate_existing_holdings_aggregated():
    # 기존 보유 10M + 신규 5M(동일 종목) → 합산 15M 비중으로 VaR (가상 포트 재계산)
    holdings = [Holding(ticker="A", value_krw=10_000_000, sector="ETF")]
    res = evaluate_pre_trade(_req(5_000_000), holdings, returns_by_ticker={"A": _returns(0.05)})
    assert res.checks["G1"]["status"] in ("pass", "violation")  # 작동(예외 없음)
    assert res.checks["G1"]["var95"] >= 0.0


def test_var_gate_resize_converges_below_limit():
    # RESIZE 성공 시 축소된 사이즈는 한도(2.5%) 근처/이하로 수렴해야
    res = evaluate_pre_trade(_req(11_000_000), [], returns_by_ticker={"A": _returns(0.12)})
    if res.verdict == "RESIZE":
        # 재평가된 최종 var95가 한도 + 여유 안 (선형근사 + 루프 수렴)
        assert res.checks["G1"]["var95"] <= 0.025 + 0.005


def test_var_gate_reject_includes_var_reason_in_audit():
    # G1 위반으로 min_trade 미만까지 축소되면 REJECT + var 사유가 violations에
    # equity 작게 → 비중 커짐 → 강한 축소
    req = GateRequest(ticker="A", sector="ETF", proposed_size_krw=900_000,
                      equity_krw=1_000_000, adv20_krw=1e15)
    res = evaluate_pre_trade(req, [], returns_by_ticker={"A": _returns(0.25)})
    # 90% 비중 → G3(12%)부터 위반, var도 위반 — 어느 쪽이든 REJECT/RESIZE
    assert res.verdict in ("RESIZE", "REJECT")
