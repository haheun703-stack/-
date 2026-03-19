"""LENS 1: GAME BOARD — 공격/방어 모드 판정 (STEP 7-2)

brain_decision.json의 effective_regime + confidence를 읽어
4단계 전투 모드를 결정한다.

AGGRESSIVE: BULL + confidence > 0.7
BALANCED:   BULL + confidence ≤ 0.7
DEFENSIVE:  CAUTION / BEAR
RETREAT:    CRISIS
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 레짐 → 모드 매핑 (confidence 무관한 기본값)
_REGIME_BASE_MODE = {
    "BULL": "BALANCED",
    "CAUTION": "DEFENSIVE",
    "PRE_BEAR": "DEFENSIVE",
    "BEAR": "DEFENSIVE",
    "CRISIS": "RETREAT",
}


def compute(brain: dict) -> dict:
    """GAME BOARD 렌즈 계산.

    Args:
        brain: brain_decision.json 로드 결과

    Returns:
        {"mode": str, "reason": str}
    """
    regime = brain.get("effective_regime", "CAUTION").upper()
    confidence = brain.get("confidence", 0.5)

    # 레짐-신뢰도 조합 판정
    if regime == "CRISIS":
        mode = "RETREAT"
        reason = f"CRISIS 레짐 — 전면 방어"
    elif regime == "BULL" and confidence > 0.7:
        mode = "AGGRESSIVE"
        reason = f"BULL + 고신뢰({confidence:.0%}) — 공격 모드"
    elif regime == "BULL":
        mode = "BALANCED"
        reason = f"BULL + 중신뢰({confidence:.0%}) — 균형 모드"
    elif regime in ("BEAR", "PRE_BEAR"):
        mode = "DEFENSIVE"
        reason = f"{regime} 레짐 — 방어 모드"
    else:
        mode = _REGIME_BASE_MODE.get(regime, "DEFENSIVE")
        reason = f"{regime} 레짐 + 신뢰도 {confidence:.0%}"

    return {"mode": mode, "reason": reason}
