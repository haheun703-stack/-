"""risk/component_var.py — L2 Component VaR (포지션별 VaR 기여 분해). RISK_ENGINE Phase 4.

스펙: docs/01-plan/RISK_ENGINE_SPEC_v2.md §3.3.

목적: 총 VaR을 포지션별 기여도로 분해한다 — "종목 A: 비중 8% / VaR 기여 31%"처럼
  비중과 리스크 기여의 불일치를 드러낸다. 게이트 G7(신규 포지션 기여 ≤ component_var_limit)과
  일일 리포트(1위 기여 40% 초과 경고)에 쓴다.

★분해 방식 = Euler(한계기여) 공분산 분해를 **FHS 필터링 수익률** 위에서 계산:
    기여율_i = w_i · Cov(r̃_i, r̃_p) / Var(r̃_p)         (Σ_i 기여율_i = 1, 정확)
    Component VaR_i = 기여율_i × VaR95
  r̃ = var_engine._build_fhs_panel의 필터링 수익률(현재 변동성 리스케일·공통 날짜로 상관 내재),
  r̃_p = 가중합 포트폴리오 분포. 모분산(ddof=0)으로 계산해 Σ w_i·Cov(r̃_i,r̃_p)=Var(r̃_p)가
  성립 → 기여율 합이 정확히 1.

★왜 '꼬리 조건부 기댓값'(VaR 분위수 한 점의 조건부 평균)이 아니라 공분산 분해인가:
  실데이터 공통구간이 60~수백일이면 95% 꼬리 표본이 3~수십 개라 분위수 한 점 조건부 추정은
  분산이 너무 크다(노이즈). 공분산 분해는 전체 표본을 써 안정적이며 Euler 합산성을 정확히 만족한다.
  타원분포 가정이 들어가나, 게이트가 보는 것은 '기여 *비율*'(상대 집중도)이라 robust하다.

★이 모듈은 순수 계산만 한다 — 실주문 경로 접촉 0, 파일 write 0. var_engine.FhsPanel을 공유해
  VaR와 정합(로직 1곳=버그 1곳). risk/config·var_engine 외 프로젝트 모듈 미import(격리).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from risk.config import RISK_CONFIG, RiskConfig
from risk.var_engine import _build_fhs_panel


@dataclass(frozen=True)
class ComponentVarResult:
    """Component VaR 분해 결과.

    contributions: {ticker: 기여율}(Σ=1, 음수 가능=헤지). component_var: {ticker: 기여율×VaR95}.
    ok=False면 표본 부족/데이터 없음/분산 0(분해 불가) → 빈 dict + 보수적 var95=1.0(fail-closed).
    호출부(게이트)는 ok=False를 차단 방향으로 흘려보낸다.
    """

    ok: bool
    contributions: dict[str, float] = field(default_factory=dict)
    component_var: dict[str, float] = field(default_factory=dict)
    var95: float = 1.0
    n_positions: int = 0
    n_obs: int = 0
    reason: str = ""

    def contribution(self, ticker: str) -> float:
        """종목 기여율(부재=0.0). 게이트 G7이 신규 종목 기여를 한도와 비교할 때 사용."""
        return float(self.contributions.get(ticker, 0.0))


def compute_component_var(
    returns_by_ticker: dict[str, "np.ndarray"],
    weights_by_ticker: dict[str, float],
    *,
    cfg: RiskConfig = RISK_CONFIG,
    lookback: int = 750,
) -> ComponentVarResult:
    """가상 포트폴리오(보유 + 신규)의 포지션별 VaR 기여를 Euler 공분산 분해로 산출.

    Args:
        returns_by_ticker: {ticker: pd.Series(일별 단순수익률)} — compute_portfolio_var와 동일 입력.
        weights_by_ticker: {ticker: 비중 = value_i/equity}. 신규 포함(가상 포트폴리오).
        cfg/lookback: var_engine과 공유 파라미터.

    Returns:
        ComponentVarResult. 표본<_MIN_OBS·데이터 없음·포트분산 0 → ok=False + 빈 dict(fail-closed).
    """
    panel = _build_fhs_panel(returns_by_ticker, weights_by_ticker, cfg, lookback, 1.5)
    if not panel.ok:
        return ComponentVarResult(False, {}, {}, 1.0, 0, panel.n_obs, panel.reason)

    port = panel.port
    var_p = float(np.var(port))  # 모분산(ddof=0) — Σ w_i·Cov(r_i,port)=Var(port) 정합용
    var95 = max(float(-np.percentile(port, 5.0)), 0.0)
    # ★near-zero 분산 가드: 완전/근사 헤지(B≈-A)면 port 분산이 부동소수 잡음 수준(예: 1e-19)으로
    #   떨어져 1/var_p가 폭발(기여율 발산)한다. 절대 0뿐 아니라 '그로스 분산 대비 무의미하게 작은'
    #   상쇄도 degenerate로 본다. gross=Σ w_i²·Var(r̃_i)(대각 분산, 상쇄 전 규모 기준).
    gross = float(np.sum((panel.w ** 2) * np.var(panel.filtered, axis=0)))
    if var_p <= 1e-12 * max(gross, 1e-300) or var95 <= 0.0:
        # 포트 분산 0/근사0(상수·완전 헤지) → 한계기여 정의 불가 → fail-closed(분해 신뢰 불가)
        return ComponentVarResult(False, {}, {}, 1.0, len(panel.tickers), panel.n_obs,
                                  "degenerate_zero_variance")

    # 기여율_i = w_i · Cov(r̃_i, r̃_p) / Var(r̃_p). 모공분산(bias=True=ddof0)으로 Σ=1 정확.
    contributions: dict[str, float] = {}
    component_var: dict[str, float] = {}
    for j, t in enumerate(panel.tickers):
        cov_i = float(np.cov(panel.filtered[:, j], port, bias=True)[0, 1])
        frac = float(panel.w[j]) * cov_i / var_p
        contributions[t] = frac
        component_var[t] = frac * var95

    return ComponentVarResult(
        True, contributions, component_var, var95, len(panel.tickers), panel.n_obs, "ok"
    )
