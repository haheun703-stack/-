"""KOSPI 매크로 가드 — 약세장 자동 차단 (5/22 백테스트 결과 반영).

배경 (5/22 picks_history 382건 백테스트):
  C2 필터 시장 환경별 D+1 결과:
    STRONG_BULL (≥+0.5%): D+1 +22.46%, 승률 89.1%  ★ 적극 진입
    NEUTRAL (-0.5~+0.5%): D+1 +26.97%, 승률 90.0%  ★ 적극 진입
    CAUTION (-2~-0.5%):   D+1 +4.49%,  승률 69.2%  🟡 신중 진입
    BEARISH (-2%↓):       D+1 -7.35%,  승률 0.0%   ❌ 진입 차단 필수

가드 임계 (KOSPI 전일 일봉 등락률):
  ≥ -0.5%: NEUTRAL/STRONG_BULL → 진입 허용 (적극)
  -0.5 ~ -1.5%: CAUTION → 진입 허용 (신중)
  ≤ -1.5%: BEARISH 추정 → 매수 차단 ← .env AUTO_TRADING_KOSPI_BEARISH_THRESHOLD

추가:
  KOSPI MA5 위치도 보조 (MA5 -3% 이하 = 추세 약세)

사용:
  from src.use_cases.market_regime_guard import check_market_regime_guard
  result = check_market_regime_guard()
  if not result["passed"]:
      # BEARISH 진입 차단
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KOSPI_INDEX_PATH = PROJECT_ROOT / "data" / "kospi_index.csv"

# 임계 (.env 동적, 기본값 5/22 백테스트 권장)
BEARISH_THRESHOLD = float(os.getenv("AUTO_TRADING_KOSPI_BEARISH_THRESHOLD", "-1.5"))  # %
CAUTION_THRESHOLD = float(os.getenv("AUTO_TRADING_KOSPI_CAUTION_THRESHOLD", "-0.5"))  # %
MA5_DROP_THRESHOLD = float(os.getenv("AUTO_TRADING_KOSPI_MA5_DROP", "-3.0"))           # KOSPI vs MA5


def _load_kospi_history(n_days: int = 10) -> list[dict[str, Any]]:
    """KOSPI 일봉 최근 N일 로드."""
    if not KOSPI_INDEX_PATH.exists():
        logger.warning("kospi_index.csv 없음: %s", KOSPI_INDEX_PATH)
        return []
    try:
        with open(KOSPI_INDEX_PATH, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return rows[-n_days:] if len(rows) > n_days else rows
    except Exception as e:
        logger.warning("kospi_index.csv 로드 실패: %s", e)
        return []


def get_kospi_regime() -> dict[str, Any]:
    """현재 KOSPI 매크로 환경 판정.

    Returns:
        {
            "regime": "STRONG_BULL" | "NEUTRAL" | "CAUTION" | "BEARISH" | "UNKNOWN",
            "kospi_chg_pct": float,        # 전일 일봉 등락률
            "kospi_close": float,           # 전일 종가
            "kospi_ma5": float,             # 5일 이동평균
            "vs_ma5_pct": float,            # 전일 종가 vs MA5
            "reason": str,
        }
    """
    history = _load_kospi_history(n_days=10)
    if len(history) < 2:
        return {
            "regime": "UNKNOWN",
            "kospi_chg_pct": 0.0,
            "kospi_close": 0.0,
            "kospi_ma5": 0.0,
            "vs_ma5_pct": 0.0,
            "reason": "KOSPI 데이터 부족",
        }

    # 전일 vs 전전일 등락률
    try:
        prev = float(history[-1]["close"])
        prev_prev = float(history[-2]["close"])
        chg_pct = (prev - prev_prev) / prev_prev * 100
    except (KeyError, ValueError, ZeroDivisionError) as e:
        logger.warning("KOSPI 등락률 계산 실패: %s", e)
        return {
            "regime": "UNKNOWN",
            "kospi_chg_pct": 0.0,
            "kospi_close": 0.0,
            "kospi_ma5": 0.0,
            "vs_ma5_pct": 0.0,
            "reason": "KOSPI 계산 실패",
        }

    # MA5 계산 (최근 5일 종가 평균)
    try:
        last_5 = [float(r["close"]) for r in history[-5:]]
        ma5 = sum(last_5) / len(last_5) if last_5 else prev
        vs_ma5 = (prev - ma5) / ma5 * 100 if ma5 > 0 else 0
    except Exception:
        ma5 = prev
        vs_ma5 = 0

    # Regime 판정
    if chg_pct <= BEARISH_THRESHOLD:
        regime = "BEARISH"
        reason = f"KOSPI {chg_pct:+.2f}% ≤ {BEARISH_THRESHOLD}% — 약세장 진입 차단"
    elif chg_pct <= CAUTION_THRESHOLD:
        regime = "CAUTION"
        reason = f"KOSPI {chg_pct:+.2f}% (CAUTION 영역)"
    elif chg_pct >= 0.5:
        regime = "STRONG_BULL"
        reason = f"KOSPI {chg_pct:+.2f}% ≥ +0.5% (강세장)"
    else:
        regime = "NEUTRAL"
        reason = f"KOSPI {chg_pct:+.2f}% (NEUTRAL)"

    # MA5 보조 가드 — MA5 -3% 이하면 BEARISH 추정 강화
    if vs_ma5 <= MA5_DROP_THRESHOLD and regime not in ("BEARISH",):
        original = regime
        regime = "CAUTION" if regime != "CAUTION" else "BEARISH"
        reason = (
            f"{reason}; MA5 대비 {vs_ma5:+.2f}% ≤ {MA5_DROP_THRESHOLD}% "
            f"({original}→{regime} 강등)"
        )

    return {
        "regime": regime,
        "kospi_chg_pct": round(chg_pct, 2),
        "kospi_close": round(prev, 2),
        "kospi_ma5": round(ma5, 2),
        "vs_ma5_pct": round(vs_ma5, 2),
        "reason": reason,
    }


def check_market_regime_guard() -> dict[str, Any]:
    """매수 진입 가드 — BEARISH 시 차단.

    Returns:
        {
            "passed": bool,                  # True = 진입 허용
            "regime": str,
            "kospi_chg_pct": float,
            "block_reason": str | None,
        }
    """
    info = get_kospi_regime()
    regime = info["regime"]

    if regime == "BEARISH":
        return {
            "passed": False,
            "regime": regime,
            "kospi_chg_pct": info["kospi_chg_pct"],
            "kospi_close": info["kospi_close"],
            "block_reason": info["reason"],
            "info": info,
        }
    elif regime == "UNKNOWN":
        # 데이터 부재 시 안전 차단 (5/22 사고 교훈)
        return {
            "passed": False,
            "regime": regime,
            "kospi_chg_pct": 0.0,
            "kospi_close": 0.0,
            "block_reason": "KOSPI 데이터 없음 — 안전 차단",
            "info": info,
        }

    return {
        "passed": True,
        "regime": regime,
        "kospi_chg_pct": info["kospi_chg_pct"],
        "kospi_close": info["kospi_close"],
        "block_reason": None,
        "info": info,
    }
