"""risk/correlation.py — 이중 상관행렬 (평시 ρ_normal + 스트레스 ρ_stress). RISK_ENGINE Phase 4.

스펙: docs/01-plan/RISK_ENGINE_SPEC_v2.md §3.4.

핵심 원칙: **상관관계는 위기 때 1로 수렴한다.** 평시 상관 하나로 분산을 평가하면 안 된다.
  ρ_normal  = 최근 lookback일 EWMA 가중 상관 (최근 관측에 가중 — 국면 변화 반영)
  ρ_stress  = ρ_normal + 0.5 × (1 - ρ_normal)        # 1 방향으로 슈링크(위기 동조화 가정)
  → 게이트(G5 상관 클러스터)는 **둘 중 나쁜 값(=ρ_stress, 더 큼)** 기준으로 군집을 판정한다.
    위기에 함께 무너질 종목을 평시 상관이 낮다고 따로 세면 분산 착시 → ρ_stress로 보수화.

용도: G5(ρ_stress ≥ corr_cluster_threshold 보유 = 신규와 '유효 단일 포지션' 합산 → G3 한도).
  Phase 4 이전에는 gate_wiring이 corr_with_new=None만 넘겨 G5가 production에서 비활성이었고
  (unknown_corr_count로 투명 노출), 이 모듈이 실상관을 주입해 G5를 실활성화한다.

★순수 계산만 — 실주문 경로 접촉 0, 파일 write 0. risk/config 외 프로젝트 모듈 미import(격리).
  상관 계산 불가(공통 표본 부족)는 None으로 흘려보낸다 — 호출부(gate_wiring)가 그 종목을
  corr_with_new=None으로 두면 G5가 군집에서 제외하고 unknown_corr_count로 카운트한다(기존 계약).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG, RiskConfig

# 상관 추정 최소 공통 표본. VaR(_MIN_OBS=60)과 동일 보수 기준 — 짧은 공통구간 상관은 신뢰 불가.
_MIN_PAIR_OBS = 60


def _ewma_weights(n: int, lam: float) -> np.ndarray:
    """최근 관측에 큰 가중을 주는 정규화 EWMA 가중치(합=1). w_t ∝ λ^(거리)."""
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def _ewma_corr(x: np.ndarray, y: np.ndarray, lam: float) -> float | None:
    """EWMA 가중 피어슨 상관 ρ_normal. 분산 0(상수)면 None(상관 정의 불가)."""
    n = len(x)
    w = _ewma_weights(n, lam)
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    dx = x - mx
    dy = y - my
    cov = float(np.sum(w * dx * dy))
    vx = float(np.sum(w * dx * dx))
    vy = float(np.sum(w * dy * dy))
    if vx <= 0.0 or vy <= 0.0:
        return None
    rho = cov / np.sqrt(vx * vy)
    return float(np.clip(rho, -1.0, 1.0))


def stress_shrink(rho_normal: float) -> float:
    """ρ_stress = ρ_normal + 0.5(1-ρ_normal). 1 방향 슈링크(위기 동조화). [-1,1] 클립.

    ρ=0 → 0.5, ρ=0.6 → 0.8, ρ=1 → 1. 음의 상관도 1 방향으로 당겨 헤지 과신을 차단한다.
    """
    rho_stress = rho_normal + 0.5 * (1.0 - rho_normal)
    return float(np.clip(rho_stress, -1.0, 1.0))


def stress_correlation_with(
    new_ticker: str,
    other_tickers: list[str],
    returns_by_ticker: dict[str, pd.Series],
    *,
    cfg: RiskConfig = RISK_CONFIG,
    lookback: int = 252,
) -> dict[str, float]:
    """신규 종목 vs 각 보유 종목의 **스트레스 상관 ρ_stress** 맵.

    절차: 공통 날짜 inner join → 최근 lookback일 → EWMA ρ_normal → stress_shrink.
    공통 표본 < _MIN_PAIR_OBS / 신규 수익률 없음 / 분산 0 → 해당 종목은 결과에서 **생략**
    (호출부가 corr_with_new=None으로 두어 G5 군집에서 제외 + unknown_corr_count 카운트).

    Args:
        new_ticker: 신규 주문 종목.
        other_tickers: 기존 보유 종목들.
        returns_by_ticker: {ticker: pd.Series(일별 수익률)} — gate_wiring이 VaR용으로 이미 로드.

    Returns:
        {other_ticker: ρ_stress}. 계산 가능한 종목만 포함.
    """
    out: dict[str, float] = {}
    new_r = returns_by_ticker.get(new_ticker)
    if new_r is None or len(new_r) == 0:
        return out
    for t in other_tickers:
        if not t or t == new_ticker:
            continue
        other_r = returns_by_ticker.get(t)
        if other_r is None or len(other_r) == 0:
            continue
        df = pd.DataFrame({"a": new_r, "b": other_r}).dropna()
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        if len(df) < _MIN_PAIR_OBS:
            continue
        rho_n = _ewma_corr(df["a"].to_numpy(dtype=float), df["b"].to_numpy(dtype=float),
                           cfg.ewma_lambda)
        if rho_n is None:
            continue
        out[t] = stress_shrink(rho_n)
    return out
