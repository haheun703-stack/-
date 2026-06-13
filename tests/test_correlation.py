"""risk/correlation 이중 상관(ρ_normal/ρ_stress) + G5 실활성화 — RISK_ENGINE Phase 4 (§3.4).

엔진 불변식:
  1. 완전 양의 상관 → ρ_normal≈1, ρ_stress≈1.
  2. 무상관 → ρ_normal≈0, ρ_stress≈0.5(1 방향 슈링크).
  3. stress_shrink 공식: 0→0.5, 0.6→0.8, 1→1, 음의 상관도 1 방향으로 당김.
  4. 공통 표본<60 / 신규 없음 / 상수(분산0) → 생략(corr_with_new=None 처리).
G5 통합:
  5. 고상관 보유 + 신규 합산 비중 > 12% → G5 REJECT(클러스터=유효 단일 포지션).
  6. returns 미가용 → corr None → G5 비활성(군집 없음, Phase 1a 호환).
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from risk.config import KST
from risk.correlation import stress_correlation_with, stress_shrink
from src.use_cases.gate_wiring import build_gate_result

_KEY = "corr-test-key"
_ASOF = date(2026, 6, 10)  # 수요일(거래일)


def _series(arr) -> pd.Series:
    idx = pd.bdate_range(end=pd.Timestamp(_ASOF), periods=len(arr))
    return pd.Series(arr, index=idx)


def _rets(sigma=0.02, n=200, seed=1):
    rng = np.random.default_rng(seed)
    return _series(rng.normal(0.0, sigma, n))


# ── 엔진: stress_shrink ──────────────────────────────────────────────────────
def test_stress_shrink_formula():
    assert stress_shrink(0.0) == 0.5
    assert abs(stress_shrink(0.6) - 0.8) < 1e-12
    assert stress_shrink(1.0) == 1.0
    assert stress_shrink(-1.0) == 0.0  # -1 → 0 (1 방향으로 당김, 헤지 과신 차단)


# ── 엔진: stress_correlation_with ────────────────────────────────────────────
def test_perfectly_correlated_high_stress():
    base = _rets(seed=5)
    rets = {"NEW": base, "H1": base * 1.0}  # 동일 = 완전 상관
    out = stress_correlation_with("NEW", ["H1"], rets)
    assert out["H1"] > 0.99  # ρ_stress ≈ 1


def test_uncorrelated_shrinks_to_half():
    rets = {"NEW": _rets(seed=1), "H1": _rets(seed=2)}  # 독립
    out = stress_correlation_with("NEW", ["H1"], rets)
    assert abs(out["H1"] - 0.5) < 0.15  # ρ_normal≈0 → ρ_stress≈0.5(표본 잡음)


def test_negative_corr_pulled_toward_one():
    base = _rets(seed=7)
    rets = {"NEW": base, "H1": -base}  # 완전 음의 상관
    out = stress_correlation_with("NEW", ["H1"], rets)
    assert abs(out["H1"] - 0.0) < 1e-6  # ρ_normal=-1 → ρ_stress=0 (헤지 과신 방어)


def test_insufficient_common_obs_omitted():
    rets = {"NEW": _rets(n=40), "H1": _rets(n=40, seed=2)}  # 40 < 60
    out = stress_correlation_with("NEW", ["H1"], rets)
    assert "H1" not in out


def test_new_ticker_missing_returns_empty():
    out = stress_correlation_with("NEW", ["H1"], {"H1": _rets()})
    assert out == {}


def test_constant_series_omitted():
    rets = {"NEW": _rets(seed=1), "H1": _series(np.full(200, 0.0))}  # 분산 0
    out = stress_correlation_with("NEW", ["H1"], rets)
    assert "H1" not in out


# ── G5 통합: build_gate_result로 실활성화 ────────────────────────────────────
class _FakeBalance:
    def __init__(self, payload):
        self._p = payload

    def fetch_balance(self):
        return self._p


def _correlated_ohlcv(seed_shared=0, n=130, idio=0.003, volume=80000):
    """고상관 NEW/H1 OHLCV 생성 — 공유 shock + 작은 개별 잡음(ρ 높음). 130행(>60), 기준일 최신."""
    rng = np.random.default_rng(seed_shared)
    shock = rng.normal(0.0, 0.02, n)
    idx = pd.bdate_range(end=pd.Timestamp(_ASOF), periods=n)

    def _df(extra_seed):
        r = shock + np.random.default_rng(extra_seed).normal(0.0, idio, n)
        close = 10000.0 * np.cumprod(1.0 + r)
        return pd.DataFrame({"close": close, "volume": [volume] * n}, index=idx)

    return {"NEW": _df(101), "H1": _df(202)}


def test_g5_high_corr_cluster_rejects():
    # 보유 H1 9% + 신규 8%(고상관) → 클러스터 합 17% > 12% → G5 REJECT
    dfs = _correlated_ohlcv()
    balance = _FakeBalance({
        "ok": True, "available_cash": 91_000_000,
        "holdings": [{"ticker": "H1", "eval_amount": 9_000_000}],
    })
    r = build_gate_result(
        "NEW", 8_000_000, balance_port=balance,
        ohlcv_loader=lambda t: dfs.get(t),
        sector_resolver=lambda t: {"NEW": "B", "H1": "A"}.get(t),  # 다른 섹터 → G4 회피
        hmac_key=_KEY, as_of_date=_ASOF,
        now_kst=datetime(2026, 6, 10, 10, 0, tzinfo=KST),
    )
    assert r.verdict == "REJECT", (r.verdict, r.checks.get("G5"))
    assert r.checks["G5"]["status"] == "violation"
    assert "H1" in r.checks["G5"]["cluster_tickers"]  # 고상관 → 클러스터 편입
    assert any(v.get("gate") == "G5" for v in r.violations)


def test_g5_inactive_without_returns():
    # 짧은 이력(상수 30행) → returns 미가용 → corr None → G5 군집 없음(unknown_corr_count로 노출)
    short_df = pd.DataFrame(
        {"close": [10000] * 30, "volume": [80000] * 30},
        index=pd.bdate_range(end=pd.Timestamp(_ASOF), periods=30),
    )
    balance = _FakeBalance({
        "ok": True, "available_cash": 91_000_000,
        "holdings": [{"ticker": "H1", "eval_amount": 9_000_000}],
    })
    r = build_gate_result(
        "NEW", 5_000_000, balance_port=balance,
        ohlcv_loader=lambda t: short_df,
        sector_resolver=lambda t: {"NEW": "B", "H1": "A"}.get(t),
        hmac_key=_KEY, as_of_date=_ASOF,
        now_kst=datetime(2026, 6, 10, 10, 0, tzinfo=KST),
    )
    # corr 미계산 → 클러스터 비어있음, unknown_corr_count로 투명 노출
    assert r.checks["G5"]["cluster_tickers"] == []
    assert r.checks["G5"]["unknown_corr_count"] == 1
