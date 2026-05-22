"""피보나치 단계별 매수 로직 — 역매수 전략 핵심 (5/22 퐝가님 인사이트).

배경:
  개미털기 추정 종목 (shake_out_detector 확정 후) → 피보나치 단계별 분할 매수.
  하락폭이 깊을수록 더 많이 매수 (황금비 0.618 영역에서 최대).
  진짜 문제 시 -20% 손절 (총 자본 손실 한도 X)

피보나치 단계 (진입가 대비):
  1차 (신규):     0% — 초기 매수 (자비스 entry_score 통과 시)
  관망:        -5~0% — 정상 조정
  2차 매수:   -5~-10% — 0.382~0.5 영역 (30% 자본 추가)
  ★ 3차 매수: -10~-15% — 0.618 황금비 (40% 자본, 최대)
  4차 매수:  -15~-18% — 0.786 (30% 자본, 마지막)
  손절:      -20% 이하 — 진짜 문제 확정, 전량 매도

총 자본 배분 (종목당):
  1차 30% (초기) + 2차 30% + 3차 40% (황금비 최대) + 4차 (선택) = 100%
  단순화 (워밍업): 1차 50% + 2차 30% + 3차 20%

워밍업 단계 (5/26~):
  종목당 max 1주 (워밍업), 자본 분할 X
  → fibonacci_buyer는 추매 신호만 발동 (실제 추매는 5/30+ 결단 후)
  → 일단 로직 구현 + 신호 발동만, 매수 실행 X (워밍업 안전)

사용:
  from src.use_cases.fibonacci_buyer import calculate_fib_buy_step
  result = calculate_fib_buy_step(entry_price, current_price, already_bought_steps)
  if result["action"] == "BUY_STEP_2":
      # 2차 매수 신호 발동
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 피보나치 단계 임계 (진입가 대비 하락률 %)
STEP_2_RANGE = (-10.0, -5.0)    # 2차 매수: -10~-5% (0.382~0.5)
STEP_3_RANGE = (-15.0, -10.0)   # 3차 매수: -15~-10% (0.618 황금비) ★ 최대
STEP_4_RANGE = (-18.0, -15.0)   # 4차 매수: -18~-15% (0.786)
STOP_LOSS_THRESHOLD = -20.0      # 손절 임계

# 자본 배분 (워밍업 = 단계당 1주, 5/30+ 정식 = 실제 분할)
CAPITAL_ALLOCATION = {
    1: 0.30,    # 1차 신규: 30%
    2: 0.30,    # 2차 매수: 30%
    3: 0.40,    # 3차 매수: 40% (황금비 최대) ★
    4: 0.00,    # 4차 매수: 0% (워밍업은 3차까지만, 정식 가동 후 30% 활성화)
}

# 신호 종류
ACTIONS = {
    "HOLD": "보유 유지 (정상 영역)",
    "WATCH": "관망 (정상 조정 -5~0%)",
    "BUY_STEP_2": "2차 매수 시그널 (-10~-5%)",
    "BUY_STEP_3": "★3차 매수 시그널 (-15~-10%, 황금비)",
    "BUY_STEP_4": "4차 매수 시그널 (-18~-15%, 마지막)",
    "STOP_LOSS": "손절 (-20% 이하)",
    "REBUY_BLOCKED": "이미 해당 단계 매수 완료",
    "INVALID": "진입가 부재 또는 무효",
}


def calculate_fib_buy_step(
    entry_price: float,
    current_price: float,
    initial_capital: float = 1_000_000,
    already_bought_steps: list[int] | None = None,
    warmup_mode: bool = True,
) -> dict[str, Any]:
    """피보나치 단계별 매수 결정.

    Args:
        entry_price: 1차 진입가 (보유 시점 가격)
        current_price: 현재가
        initial_capital: 종목당 초기 배분 자본 (기본 100만, 워밍업은 1주 한도)
        already_bought_steps: 이미 매수한 단계 [1, 2, ...] (1차는 항상 포함)
        warmup_mode: 워밍업 모드 (True = 신호만, 실제 매수 X)

    Returns:
        {
            "action": str,                     # ACTIONS 키
            "step": int,                        # 다음 매수 단계 (0 = 매수 안 함)
            "decline_pct": float,                # 진입가 대비 하락률
            "next_buy_price": float,             # 다음 매수 권장가 (현재가)
            "next_capital_pct": float,           # 다음 단계 자본 비율
            "next_capital_won": int,             # 다음 단계 자본 금액
            "next_qty_estimate": int,            # 추정 수량 (자본/현재가, 워밍업 1주 한도)
            "stop_loss_price": float,            # 손절가 (-20%)
            "reason": str,
        }
    """
    if entry_price <= 0 or current_price <= 0:
        return {
            "action": "INVALID",
            "step": 0,
            "decline_pct": 0.0,
            "next_buy_price": 0.0,
            "next_capital_pct": 0.0,
            "next_capital_won": 0,
            "next_qty_estimate": 0,
            "stop_loss_price": 0.0,
            "reason": ACTIONS["INVALID"],
        }

    if already_bought_steps is None:
        already_bought_steps = [1]  # 1차는 항상 매수됨 (보유 종목)

    decline_pct = (current_price - entry_price) / entry_price * 100
    stop_loss_price = entry_price * (1 + STOP_LOSS_THRESHOLD / 100)

    # 손절 임계
    if decline_pct <= STOP_LOSS_THRESHOLD:
        return {
            "action": "STOP_LOSS",
            "step": 0,
            "decline_pct": round(decline_pct, 2),
            "next_buy_price": 0.0,
            "next_capital_pct": 0.0,
            "next_capital_won": 0,
            "next_qty_estimate": 0,
            "stop_loss_price": round(stop_loss_price, 2),
            "reason": f"손절 ({decline_pct:.2f}% ≤ {STOP_LOSS_THRESHOLD}%) — 진짜 문제 확정",
        }

    # 단계 판정 (가장 깊은 매수 가능 단계 찾기)
    step = 0
    action = "HOLD"

    if STEP_4_RANGE[0] <= decline_pct < STEP_4_RANGE[1]:
        step = 4
        action = "BUY_STEP_4"
    elif STEP_3_RANGE[0] <= decline_pct < STEP_3_RANGE[1]:
        step = 3
        action = "BUY_STEP_3"
    elif STEP_2_RANGE[0] <= decline_pct < STEP_2_RANGE[1]:
        step = 2
        action = "BUY_STEP_2"
    elif decline_pct >= -5:
        action = "WATCH" if decline_pct < 0 else "HOLD"
    else:
        action = "HOLD"

    # 이미 해당 단계 매수했는지 확인 (중복 방지)
    if step > 0 and step in already_bought_steps:
        return {
            "action": "REBUY_BLOCKED",
            "step": step,
            "decline_pct": round(decline_pct, 2),
            "next_buy_price": 0.0,
            "next_capital_pct": 0.0,
            "next_capital_won": 0,
            "next_qty_estimate": 0,
            "stop_loss_price": round(stop_loss_price, 2),
            "reason": f"{step}차 매수 이미 완료 (단계: {already_bought_steps})",
        }

    # 자본 + 수량 계산
    next_capital_pct = CAPITAL_ALLOCATION.get(step, 0.0)
    next_capital_won = int(initial_capital * next_capital_pct)
    next_qty_estimate = int(next_capital_won / current_price) if current_price > 0 else 0

    # 워밍업 모드: 1주 한도
    if warmup_mode and next_qty_estimate > 0:
        next_qty_estimate = 1

    if step > 0:
        reason = (
            f"{ACTIONS[action]} | 하락 {decline_pct:.2f}% | "
            f"진입가 {entry_price:,.0f}원 → 현재 {current_price:,.0f}원 | "
            f"수량 {next_qty_estimate}주 (자본 {next_capital_pct*100:.0f}%)"
        )
    else:
        reason = f"{ACTIONS[action]} | 하락 {decline_pct:.2f}%"

    return {
        "action": action,
        "step": step,
        "decline_pct": round(decline_pct, 2),
        "next_buy_price": round(current_price, 2),
        "next_capital_pct": next_capital_pct,
        "next_capital_won": next_capital_won,
        "next_qty_estimate": next_qty_estimate,
        "stop_loss_price": round(stop_loss_price, 2),
        "reason": reason,
    }


def get_fib_levels(entry_price: float) -> dict[str, float]:
    """피보나치 임계가 계산 (참고용 — 차트 표기 등).

    Returns:
        {
            "entry":      entry_price (0% 기준),
            "fib_236":    -2.36% (관망 영역),
            "fib_382":    -3.82% (2차 매수 시작 근처),
            "fib_5":      -5.0%  (2차 진입),
            "fib_618":    -6.18% (실제 황금비 영역),
            "step_2_min": -5.0%,
            "step_2_max": -10.0%,
            "step_3_min": -10.0%,  ★ 3차 매수 시작 (실용 황금비)
            "step_3_max": -15.0%,
            "step_4_min": -15.0%,
            "step_4_max": -18.0%,
            "stop_loss":  -20.0%,
        }
    """
    return {
        "entry": entry_price,
        "fib_236": entry_price * 0.9764,
        "fib_382": entry_price * 0.9618,
        "fib_5": entry_price * 0.95,
        "fib_618": entry_price * 0.9382,
        "step_2_min": entry_price * 0.95,
        "step_2_max": entry_price * 0.90,
        "step_3_min": entry_price * 0.90,
        "step_3_max": entry_price * 0.85,
        "step_4_min": entry_price * 0.85,
        "step_4_max": entry_price * 0.82,
        "stop_loss": entry_price * (1 + STOP_LOSS_THRESHOLD / 100),
    }
