"""risk/var_backtest 검증 — RISK_ENGINE Phase 2d.

불변식:
  1. 정규 iid 수익률 → VaR95 초과율 ≈ 5%(목표밴드 3~8%), Kupiec 통과.
  2. Kupiec LR: 정확히 5%면 ~0, 크게 벗어나면 > 3.84(기각).
  3. 표본 ≤ window → 평가 0(insufficient).
  4. 변동성 클러스터에서도 FHS(EWMA 필터)라 초과율 폭주 안 함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.var_backtest import _kupiec_lr, backtest_var


def _series(vals, start="2020-01-01") -> pd.Series:
    return pd.Series(list(vals), index=pd.date_range(start, periods=len(vals), freq="B"))


def test_backtest_normal_in_band():
    rng = np.random.default_rng(42)
    r = _series(rng.normal(0.0, 0.02, 1500))
    res = backtest_var(r, window=120)
    assert res.n_eval > 1000
    assert 0.02 <= res.exceed_rate <= 0.09  # 정규 iid → ~5% 근방
    assert not res.kupiec_reject            # 모델 정상


def test_kupiec_lr_near_zero_at_expected():
    assert _kupiec_lr(1000, 50, 0.05) < 0.5  # 정확히 5% → LR ≈ 0


def test_kupiec_lr_rejects_far_rate():
    assert _kupiec_lr(1000, 150, 0.05) > 3.84  # 15%(과소추정) → 기각
    assert _kupiec_lr(1000, 5, 0.05) > 3.84    # 0.5%(과대추정) → 기각


def test_kupiec_lr_edge_zero_exceed():
    assert _kupiec_lr(500, 0, 0.05) == 0.0  # 초과 0 → 검정 불능(보수 0)


def test_backtest_insufficient_returns():
    r = _series(np.random.default_rng(1).normal(0, 0.02, 100))
    res = backtest_var(r, window=120)  # window >= len
    assert res.n_eval == 0
    assert "insufficient" in res.reason


def test_backtest_high_vol_still_calibrated():
    # 변동성 클러스터 — FHS(EWMA 시변 변동성)라 단순 히스토리컬보다 초과율 안정
    rng = np.random.default_rng(7)
    r = _series(np.concatenate([
        rng.normal(0, 0.01, 700),
        rng.normal(0, 0.04, 700),
        rng.normal(0, 0.01, 700),
    ]))
    res = backtest_var(r, window=120)
    assert res.exceed_rate < 0.12  # 과도 초과 없음(FHS 추종)
