"""LENS 3: STRUCTURAL VALUE — 밸류트랩 필터 (STEP 7-4)

V_Score(밸류에이션)와 Q_Score(퀄리티)를 교차 분석하여
밸류트랩 종목을 식별하고 레짐별 최소 퀄리티 기준을 설정한다.

밸류트랩: V_Score 상위 30% AND Q_Score 하위 30%
구조적 가치: V_Score 상위 30% AND Q_Score 상위 30%

공매도 연동 (STEP 10+):
  - 공매도 잔고율 상위 종목 → trap_score 증가 (밸류트랩 의심 강화)
  - 공매도 극단 + Q_Score 높은 → 숏커버 반등 후보 태깅
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SHORT_SIGNAL_PATH = Path("data") / "short_selling" / "daily_short.json"

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


def _load_short_data() -> dict:
    """공매도 시그널 로드."""
    try:
        with open(_SHORT_SIGNAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def compute(regime: str, lens_cfg: dict) -> dict:
    """STRUCTURAL VALUE 렌즈 계산.

    Args:
        regime: effective_regime (BULL/CAUTION/BEAR 등)
        lens_cfg: settings.yaml의 alpha_v2.lens 설정

    Returns:
        {
            "min_quality_score": float,
            "valuation_mode": str,  # STRICT / NORMAL
            "trap_filter": bool,
            "short_selling": {
                "available": bool,
                "surge_tickers": list,      # SH1 급증 종목 (밸류트랩 의심)
                "cover_tickers": list,       # SH2 숏커버 후보 (반등 기대)
                "extreme_tickers": list,     # SH3 극단 잔고
                "market_pressure": str,      # "HIGH" / "NORMAL" / "LOW"
            }
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

    # 공매도 데이터 연동
    short_data = _load_short_data()
    short_ctx = _compute_short_context(short_data)

    # 공매도 잔고 급증 시 밸류트랩 필터 강화
    if short_ctx["available"] and short_ctx["market_pressure"] == "HIGH":
        if val_mode == "NORMAL":
            val_mode = "STRICT"
            logger.info("LENS-3: 시장 공매도 압력 HIGH → STRICT 모드 전환")

    return {
        "min_quality_score": min_q,
        "valuation_mode": val_mode,
        "trap_filter": trap_filter,
        "short_selling": short_ctx,
    }


def _compute_short_context(short_data: dict) -> dict:
    """공매도 시그널에서 LENS 3 맥락 추출."""
    if not short_data or short_data.get("short_banned"):
        return {
            "available": False,
            "surge_tickers": [],
            "cover_tickers": [],
            "extreme_tickers": [],
            "market_pressure": "NORMAL",
        }

    surge_top = short_data.get("surge_top", [])
    cover_top = short_data.get("cover_top", [])
    extreme_top = short_data.get("extreme_top", [])
    market = short_data.get("market_signal", {})

    # 시장 압력 판정
    surge_ratio = market.get("surge_ratio_pct", 0)
    if surge_ratio >= 15:
        pressure = "HIGH"
    elif surge_ratio >= 8:
        pressure = "MODERATE"
    else:
        pressure = "NORMAL"

    return {
        "available": True,
        "surge_tickers": [s["ticker"] for s in surge_top[:10]],
        "cover_tickers": [s["ticker"] for s in cover_top[:10]],
        "extreme_tickers": [s["ticker"] for s in extreme_top[:10]],
        "market_pressure": pressure,
    }
