"""risk/vol_targeting.py — 포트폴리오 변동성 타겟팅. RISK_ENGINE Phase 3 / 스펙 §4.3.

변동성이 치솟으면 노출을 자동으로 줄인다 — 알라딘식 "폭풍 전 돛 줄이기"의 가장 단순한 구현.

  target_vol   = 연 target_vol_annual(15%)
  realized_vol = 포트폴리오 일별 수익률의 EWMA 변동성(halflife 20일) × √252
  scale        = min(1.0, target_vol / realized_vol)   # ≤1.0
  총노출       = 기본노출 × scale × 사다리 계수(§4.2) × 크라우딩 계수(§4.4)

★scale ≤ 1.0이 핵심: 변동성이 높으면 축소하되, 낮아도 1.0을 넘기지 않는다(레버리지 확대 금지 —
  §0 "생존이 목표"). 노출을 늘리는 방향으로는 절대 작동하지 않는다.

★게이트(G1~G8)가 아니라 L3 노출 조절(crowding·drawdown_ladder와 동형). 순수 계산 —
  실주문 경로 접촉 0, 파일 write 0, risk/config 외 프로젝트 모듈 미import(격리).
  포트폴리오 수익률 시계열은 호출처가 제공(drawdown_ladder가 DD를 받듯). 표본 부족/변동성 0 →
  scale=1.0(중립) — 데이터가 없으면 노출을 건드리지 않는다(과조절 방지, crowding과 동일 철학).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG, RiskConfig

_HALFLIFE = 20        # §4.3 "halflife 20일"
_MIN_OBS = 20         # EWMA 변동성 신뢰 최소 표본(미만이면 중립 scale=1.0)
_TRADING_DAYS = 252   # 연율화


@dataclass(frozen=True)
class VolTargetState:
    """변동성 타겟팅 판정.

    evaluable=False는 데이터 미주입/표본 부족 — scale은 1.0(중립). '변동성 낮음'(scale 1.0,
    evaluable=True)과 구분된다.
    """

    evaluable: bool
    realized_vol_annual: float | None  # 연율 실현변동성(평가 가능 시)
    target_vol_annual: float
    scale: float                       # min(1.0, target/realized); 미평가/변동성0 → 1.0


def vol_target_scale(
    portfolio_returns: pd.Series | None,
    *,
    cfg: RiskConfig = RISK_CONFIG,
) -> VolTargetState:
    """포트폴리오 일별 수익률로 변동성 타겟 노출 계수를 산출(§4.3).

    Args:
        portfolio_returns: 포트폴리오 일별 수익률 시계열(최신이 마지막). 미주입/표본<20 → scale 1.0.
        cfg: target_vol_annual(0.15) 단일 출처.

    Returns:
        VolTargetState. scale은 항상 ≤ 1.0(레버리지 확대 금지).
    """
    target = cfg.target_vol_annual
    if portfolio_returns is None:
        return VolTargetState(False, None, target, 1.0)
    r = portfolio_returns.dropna()
    if len(r) < _MIN_OBS:
        return VolTargetState(False, None, target, 1.0)
    daily_vol = float(r.ewm(halflife=_HALFLIFE).std().iloc[-1])
    if not np.isfinite(daily_vol) or daily_vol <= 0.0:
        # 변동성 0(상수)/비유한 → 타겟 이하로 간주, 노출 그대로(scale 1.0)
        return VolTargetState(True, 0.0, target, 1.0)
    realized = daily_vol * np.sqrt(_TRADING_DAYS)
    scale = min(1.0, target / realized)
    return VolTargetState(True, realized, target, float(scale))
