"""LENS 4: ASYMMETRY — 레짐별 R:R 동적 조정 (STEP 7-5)

레짐에 따라 최소 보상/위험 비율과 ATR 배수를 동적으로 설정한다.

BULL:    min_rr=1.2, target=3.0×ATR, stop=2.0×ATR
CAUTION: min_rr=1.5, target=3.0×ATR, stop=2.0×ATR
BEAR:    min_rr=2.0, target=4.0×ATR, stop=2.0×ATR
CRISIS:  min_rr=3.0, target=5.0×ATR, stop=1.5×ATR
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGIME_ASYMMETRY = {
    "BULL":     {"min_rr": 1.2, "target_atr_mult": 3.0, "stop_atr_mult": 2.0},
    "CAUTION":  {"min_rr": 1.5, "target_atr_mult": 3.0, "stop_atr_mult": 2.0},
    "PRE_BEAR": {"min_rr": 1.5, "target_atr_mult": 3.5, "stop_atr_mult": 2.0},
    "BEAR":     {"min_rr": 2.0, "target_atr_mult": 4.0, "stop_atr_mult": 2.0},
    "CRISIS":   {"min_rr": 3.0, "target_atr_mult": 5.0, "stop_atr_mult": 1.5},
}


def compute(regime: str, lens_cfg: dict) -> dict:
    """ASYMMETRY 렌즈 계산.

    Args:
        regime: effective_regime
        lens_cfg: settings.yaml의 alpha_v2.lens 설정

    Returns:
        {"min_rr_ratio": float, "target_atr_mult": float, "stop_atr_mult": float}
    """
    asym_cfg = lens_cfg.get("asymmetry", {})
    regime_upper = regime.upper()

    defaults = _REGIME_ASYMMETRY.get(regime_upper, _REGIME_ASYMMETRY["CAUTION"])

    # settings.yaml 오버라이드 지원
    regime_overrides = asym_cfg.get(regime_upper.lower(), {})

    return {
        "min_rr_ratio": regime_overrides.get("min_rr", defaults["min_rr"]),
        "target_atr_mult": regime_overrides.get("target_atr_mult", defaults["target_atr_mult"]),
        "stop_atr_mult": regime_overrides.get("stop_atr_mult", defaults["stop_atr_mult"]),
    }
