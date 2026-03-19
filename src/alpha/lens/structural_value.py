"""LENS 3: STRUCTURAL VALUE — 밸류트랩 필터 (STEP 7-4)

V_Score(밸류에이션)와 Q_Score(퀄리티)를 교차 분석하여
밸류트랩 종목을 식별하고 레짐별 최소 퀄리티 기준을 설정한다.

밸류트랩: V_Score 상위 30% AND Q_Score 하위 30%
구조적 가치: V_Score 상위 30% AND Q_Score 상위 30%
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 레짐별 최소 퀄리티 스코어
_REGIME_MIN_QUALITY = {
    "BULL": 0.3,
    "CAUTION": 0.4,
    "PRE_BEAR": 0.4,
    "BEAR": 0.5,
    "CRISIS": 0.6,
}

# 레짐별 밸류에이션 모드
_REGIME_VALUATION_MODE = {
    "BULL": "NORMAL",      # V만 충족하면 OK
    "CAUTION": "NORMAL",
    "PRE_BEAR": "STRICT",  # V+Q 동시 충족
    "BEAR": "STRICT",
    "CRISIS": "STRICT",
}


def compute(regime: str, lens_cfg: dict) -> dict:
    """STRUCTURAL VALUE 렌즈 계산.

    Args:
        regime: effective_regime (BULL/CAUTION/BEAR 등)
        lens_cfg: settings.yaml의 alpha_v2.lens 설정

    Returns:
        {
            "min_quality_score": float,
            "valuation_mode": str,  # STRICT / NORMAL
            "trap_filter": bool
        }
    """
    sv_cfg = lens_cfg.get("structural_value", {})
    regime_upper = regime.upper()

    # 설정 오버라이드 or 기본값
    min_q = sv_cfg.get("min_quality", {}).get(
        regime_upper.lower(),
        _REGIME_MIN_QUALITY.get(regime_upper, 0.4),
    )

    val_mode = _REGIME_VALUATION_MODE.get(regime_upper, "NORMAL")
    trap_filter = sv_cfg.get("trap_filter_enabled", True)

    return {
        "min_quality_score": min_q,
        "valuation_mode": val_mode,
        "trap_filter": trap_filter,
    }
