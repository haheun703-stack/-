"""risk/drawdown_ladder.py — 드로다운 디레버리징 사다리 (게이트 G8). RISK_ENGINE Phase 3 / 스펙 §4.2.

계좌 고점 대비 누적 드로다운(DD)으로 총노출(gross exposure)을 *기계적으로* 조절한다.
모델이 아니라 하드 룰 — 손실이 깊어질수록 베팅을 줄여 계좌 생존을 강제한다(§0 철학).

  DD  0 ~ -4%   : 노출 100%, 정상 운영
  DD -4 ~ -7%   : 노출 70%,  신규 진입 사이즈 50% 축소
  DD -7 ~ -10%  : 노출 40%,  신규 진입 금지(기존 포지션 관리만)
  DD -10% 초과  : L4 킬스위치 이관

★히스테리시스(필수): 단계 복귀(완화)는 그 단계 진입 경계보다 ladder_hysteresis(1.5%p) 회복해야
  발동한다(예: -4% 단계 진입 후 정상 복귀는 DD가 -2.5%까지 회복돼야). 경계에서 매일 노출이
  출렁이는 것을 막는다. 복귀는 한 번에 한 단계씩(점진), 악화는 즉시 반영(보수).

순수 계산. DD 자체(계좌 고점 추적)는 호출처가 제공. 게이트 G8 연결은 별도(pre_trade_gate).
config의 ladder_steps/ladder_hysteresis(Phase 1a 선등록)를 단일 출처로 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk.config import RISK_CONFIG, RiskConfig


@dataclass(frozen=True)
class LadderState:
    """드로다운 사다리 판정. step 0=정상 / 1=-4~-7 / 2=-7~-10 / 3=킬스위치 이관."""

    step: int
    gross_exposure: float    # 총노출 계수 (1.0 / 0.7 / 0.4 / 0.0)
    new_entry_allowed: bool  # 신규 진입 허용 여부
    new_size_mult: float     # 신규 진입 사이즈 배수 (1.0 / 0.5 / 0.0)
    to_kill_switch: bool     # L4 킬스위치 이관(step 3)
    dd: float                # 평가에 쓴 DD(≤0)


# step → (노출, 신규허용, 신규사이즈배수, 킬스위치)
_STEP_POLICY = {
    0: (1.0, True, 1.0, False),
    1: (0.7, True, 0.5, False),
    2: (0.4, False, 0.0, False),
    3: (0.0, False, 0.0, True),
}


def ladder_state(
    current_dd: float,
    prev_step: int = 0,
    cfg: RiskConfig = RISK_CONFIG,
) -> LadderState:
    """현재 DD와 직전 단계로 사다리 상태를 판정(히스테리시스 적용).

    Args:
        current_dd: 계좌 고점 대비 DD(음수, 예 -0.05 = -5%). 양수/0은 0으로 클램프.
        prev_step: 직전 호출의 step(복귀 히스테리시스용). 기본 0(정상).
        cfg: ladder_steps((-0.04,1.0),(-0.07,0.7),(-0.10,0.4)) + ladder_hysteresis(0.015).
    """
    dd = min(float(current_dd), 0.0)
    boundaries = [s[0] for s in cfg.ladder_steps]   # [-0.04, -0.07, -0.10]
    hyst = cfg.ladder_hysteresis

    # 1) 악화 기준 raw step — dd가 각 경계 이하이면 단계 상승
    raw = 0
    for i, b in enumerate(boundaries):
        if dd <= b:
            raw = i + 1

    # 2) 히스테리시스: 복귀(prev_step > raw)는 한 단계씩 + 경계 +hyst 회복 확인
    step = raw
    if prev_step > raw:
        # prev_step → prev_step-1 복귀 임계 = prev_step 진입 경계 + hyst
        recover_threshold = boundaries[prev_step - 1] + hyst
        if dd > recover_threshold:
            step = prev_step - 1     # 한 단계만 완화
        else:
            step = prev_step         # 임계 미달 → 단계 유지
        step = max(step, raw)        # 악화(raw)보다 아래로는 안 내려간다(즉시 반영)

    exposure, entry_ok, size_mult, kill = _STEP_POLICY[step]
    return LadderState(
        step=step,
        gross_exposure=exposure,
        new_entry_allowed=entry_ok,
        new_size_mult=size_mult,
        to_kill_switch=kill,
        dd=dd,
    )
