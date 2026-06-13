"""risk/var_engine.py — L2 VaR/ES 엔진 (Filtered Historical Simulation). RISK_ENGINE Phase 2.

스펙: docs/01-plan/RISK_ENGINE_SPEC_v2.md §3.2.

★순수 히스토리컬 VaR 금지(최근 저변동 구간이면 위험을 체계적으로 과소평가) → FHS 절차:
  ① 종목별 EWMA 변동성 σ_t (λ=0.94 RiskMetrics 표준)
  ② 과거 수익률 표준화:        z_t = r_t / σ_t
  ③ 현재 변동성으로 리스케일:  r̃_t = z_t × σ_now
  ④ 비중 벡터 w를 곱해 가상 포트폴리오 손익 분포 생성 (lookback 최대 750일)
  ⑤ 분위수 추출: VaR95, VaR99, ES95(= 하위 5% 평균)
스트레스 VaR: σ_now × 1.5 (변동성 1.5배). historical 동시성이 종목 간 상관을 *내재*한다
  (같은 날짜의 실제 동반 수익률을 가중합 → 위기 구간의 상관 급등이 분포에 자동 반영).
  ※ ρ_stress 명시 슈링크(스펙 §3.4)와 Component VaR(§3.3)는 상관행렬 배선 후 Phase 4.

설계 철학(§0, 협상 불가): VaR는 예측이 아니라 '평소 분포에서의 하한 추정'이며 최악이 아니다
  (최악은 스트레스 테스트 + 킬스위치 담당). 모르면 보수적으로 크게 — 표본 부족/이상 입력은
  fail-closed(ok=False + 보수적 VaR=1.0)로 흘려보내 게이트가 REJECT/RESIZE 방향이 되게 한다.

★이 모듈은 순수 계산만 한다 — 실주문 경로 접촉 0, 파일 write 0. 게이트(pre_trade_gate)가
  주입받은 수익률로 호출한다. risk/config 외 프로젝트 모듈 미import(격리).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG, RiskConfig

# FHS 최소 표본(공통 구간). 이 미만이면 VaR를 신뢰할 수 없어 fail-closed.
# 스펙 권장 lookback은 750일이나, 실데이터(상장 이력·공통 구간)가 그보다 짧을 수 있어
# 하한만 강제하고 가용 최대를 쓴다. 60 = sizing.gap_min_samples와 동일 보수 기준.
_MIN_OBS = 60
_Z_CLIP = 10.0  # 표준화 잔차 이상치 클립(데이터 글리치 방어 — 분할/배당 미조정 점프 등)


@dataclass(frozen=True)
class VarResult:
    """VaR 산출 결과. var/es/stress는 모두 *양수 손실률*(예: 0.021 = -2.1% 손실).

    ok=False면 표본 부족/입력 이상 → var=1.0(=100% 손실 가정, fail-closed). 게이트는 이 값을
    한도와 비교하므로 자동으로 위반(REJECT/RESIZE) 처리된다.
    """

    ok: bool
    var95: float
    var99: float
    es95: float
    stress_var95: float
    n_obs: int
    reason: str = ""


def _ewma_vol(r: np.ndarray, lam: float) -> np.ndarray:
    """EWMA 변동성 시계열 σ_t (RiskMetrics: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}).

    σ_t는 t-1 시점까지의 정보로 만든 t 시점 *사전(ex-ante)* 변동성. 첫 σ²는 앞 구간 표본분산 시드.
    σ_now = σ[-1] (최신 변동성) — FHS 리스케일에 사용한다.
    """
    n = len(r)
    var = np.empty(n)
    seed = float(np.var(r[: min(n, 30)]))
    var[0] = seed if seed > 0 else 1e-8
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return np.sqrt(np.maximum(var, 1e-12))


def compute_portfolio_var(
    returns_by_ticker: dict[str, pd.Series],
    weights_by_ticker: dict[str, float],
    *,
    cfg: RiskConfig = RISK_CONFIG,
    lookback: int = 750,
    stress_vol_mult: float = 1.5,
) -> VarResult:
    """가상 포트폴리오(보유 + 신규)의 FHS VaR95/VaR99/ES95 + 스트레스 VaR95.

    Args:
        returns_by_ticker: {ticker: pd.Series(일별 단순수익률, DatetimeIndex)}.
        weights_by_ticker: {ticker: 포트폴리오 비중 = value_i / equity}. 합 ≤ 1(현금 제외 가능).
            신규 종목 비중도 포함해 호출(가상 포트폴리오). 비중 0/부재 종목은 제외.
        cfg: ewma_lambda 등. lookback: 사용할 최대 일수(가용 데이터가 적으면 그만큼).
        stress_vol_mult: 스트레스 변동성 배수(스펙 1.5).

    Returns:
        VarResult. 표본 < _MIN_OBS / 데이터 없음 / 비중 합 0 → ok=False + 보수적 VaR(1.0).
    """
    # 비중 유효(>0) + 수익률 존재 종목만
    tickers = [
        t for t, w in weights_by_ticker.items()
        if abs(float(w)) > 0 and returns_by_ticker.get(t) is not None and len(returns_by_ticker[t]) > 0
    ]
    if not tickers:
        return VarResult(False, 1.0, 1.0, 1.0, 1.0, 0, "no_returns_data")

    # 공통 날짜 정렬 (inner join → dropna): 같은 날짜의 동반 수익률이라야 상관이 내재된다.
    df = pd.DataFrame({t: returns_by_ticker[t] for t in tickers}).dropna()
    if df.empty:
        return VarResult(False, 1.0, 1.0, 1.0, 1.0, 0, "no_common_dates")
    if len(df) > lookback:
        df = df.iloc[-lookback:]
    n = len(df)
    if n < _MIN_OBS:
        return VarResult(False, 1.0, 1.0, 1.0, 1.0, n, f"insufficient_obs:{n}<{_MIN_OBS}")

    w = np.array([float(weights_by_ticker[t]) for t in tickers], dtype=float)

    # FHS 종목별 필터링: 표준화 잔차 z → 현재 변동성 리스케일(평시 σ_now, 스트레스 σ_now×mult)
    filtered = np.empty((n, len(tickers)))
    stressed = np.empty((n, len(tickers)))
    for j, t in enumerate(tickers):
        r = df[t].to_numpy(dtype=float)
        sigma = _ewma_vol(r, cfg.ewma_lambda)
        sigma_now = float(sigma[-1])
        z = np.clip(r / np.where(sigma > 0, sigma, 1e-12), -_Z_CLIP, _Z_CLIP)
        filtered[:, j] = z * sigma_now
        stressed[:, j] = z * sigma_now * stress_vol_mult

    # 포트폴리오 손익 분포(각 날짜 가중합)
    port = filtered @ w
    port_stress = stressed @ w

    def _var(dist: np.ndarray, conf: float) -> float:
        # 손실 = -수익. VaR(conf%) = 손익 분포의 (100-conf) 분위수를 손실(양수)로.
        return float(-np.percentile(dist, 100.0 - conf))

    var95 = max(_var(port, 95.0), 0.0)
    var99 = max(_var(port, 99.0), var95)
    cutoff = np.percentile(port, 5.0)
    tail = port[port <= cutoff]
    es95 = float(-tail.mean()) if tail.size else var95
    es95 = max(es95, var95)
    # 스트레스(변동성↑)는 평시보다 작을 수 없다 — 단조성 강제(수치 안정).
    stress_var95 = max(_var(port_stress, 95.0), var95)

    return VarResult(True, var95, var99, es95, stress_var95, n, "ok")
