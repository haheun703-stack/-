"""ETF 자동 진입 advisory (2026-05-18 신규)

배경: 5/18 사장님 "돈 될만한거 뭐든지" + paper_portfolio.json 검증
- 자비스 ETF 트랙 (KODEX 레버리지) 4/16~5/13 27일 +66.32% 단일 거래
- 일반 포지션 7주 +8.98% MDD -3.4% vs ETF 트랙 +59.8% = ETF 7배 효율

regime 기반 자동 ETF 추천:
  STRONG_BULL → KODEX 레버리지 (122630) — 강세장 추격
  MILD_BULL   → KODEX 200 (069500) — 안정 추종
  NEUTRAL     → 관망 (보유 유지)
  CAUTION     → 현금 비중 ↑
  BEAR        → KODEX 인버스 (114800) — 약세 베팅
  CRISIS      → KODEX 인버스2X (252670) — 강한 약세 베팅

KODEX 화이트리스트 (6개, 5/16 등록):
- 069500 KODEX 200
- 122630 KODEX 레버리지
- 114800 KODEX 인버스
- 252670 KODEX 200선물인버스2X
- 233740 KODEX 코스닥150 레버리지
- 251340 KODEX 코스닥150선물인버스
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ETFRecommendation:
    """ETF 추천 결과."""

    action: str  # 'BUY_LEVERAGE' | 'BUY_LONG' | 'HOLD' | 'BUY_INVERSE' | 'BUY_INVERSE_2X' | 'CASH_UP'
    ticker: Optional[str]
    name: Optional[str]
    grade: str  # 'STRONG' | 'MEDIUM' | 'WATCH' | 'AVOID'
    reasoning: str
    suggested_size_won: int  # 권장 매수 사이즈 (1주 기준)


REGIME_TO_ETF = {
    "STRONG_BULL": {
        "action": "BUY_LEVERAGE",
        "ticker": "122630",
        "name": "KODEX 레버리지",
        "grade": "STRONG",
        "size": 100_000,  # 5/20 출격 안전선 (1주 10만)
        "reasoning": "강세장 진입 — KODEX 레버리지 진입 권장 (+66% 검증 패턴)",
    },
    "MILD_BULL": {
        "action": "BUY_LONG",
        "ticker": "069500",
        "name": "KODEX 200",
        "grade": "MEDIUM",
        "size": 100_000,
        "reasoning": "약한 강세 — KODEX 200 안정 추종 권장",
    },
    "NEUTRAL": {
        "action": "HOLD",
        "ticker": None,
        "name": None,
        "grade": "WATCH",
        "size": 0,
        "reasoning": "보합 — ETF 진입 보류, 보유 유지",
    },
    "CAUTION": {
        "action": "CASH_UP",
        "ticker": None,
        "name": None,
        "grade": "WATCH",
        "size": 0,
        "reasoning": "주의 — ETF 진입 자제, 현금 비중 ↑",
    },
    "CAUTION_TO_NEUTRAL": {
        "action": "HOLD",
        "ticker": None,
        "name": None,
        "grade": "WATCH",
        "size": 0,
        "reasoning": "전환 중 — 한 단계 더 확인",
    },
    "BEAR": {
        "action": "BUY_INVERSE",
        "ticker": "114800",
        "name": "KODEX 인버스",
        "grade": "MEDIUM",
        "size": 100_000,
        "reasoning": "약세장 — KODEX 인버스 단기 헤지 권장",
    },
    "CRISIS": {
        "action": "BUY_INVERSE_2X",
        "ticker": "252670",
        "name": "KODEX 200선물인버스2X",
        "grade": "STRONG",
        "size": 100_000,
        "reasoning": "위기 — KODEX 인버스2X 강한 헤지 권장 (단, 곱버스 손실 위험)",
    },
}


def suggest_etf_position(
    market_regime: str,
    inverse_strength: float = 100.0,
    market_strength: float = 100.0,
) -> ETFRecommendation:
    """regime 기반 ETF 추천 + 강도 보정.

    Args:
        market_regime: snapshot_session.determine_regime() 결과
        inverse_strength: KODEX 200선물인버스2X 평균 강도 (intraday_minute DB)
        market_strength: 시장 전체 평균 강도

    Returns:
        ETFRecommendation
    """
    config = REGIME_TO_ETF.get(market_regime, REGIME_TO_ETF["NEUTRAL"])

    reasoning = config["reasoning"]
    grade = config["grade"]

    # 보정: 인버스 강도 매우 강함 (≥150) + 시장 약함 (<90) → grade 격상
    if inverse_strength >= 150 and market_strength < 90:
        if config["action"] in ("BUY_INVERSE", "BUY_INVERSE_2X"):
            grade = "STRONG"
            reasoning += f" | 인버스 강도 {inverse_strength:.0f} 폭발 + 시장 강도 {market_strength:.0f} 약화 = 격상"

    # 보정: 인버스 강도 약함 (≤100) + 시장 강함 (≥100) → 강세 ETF 격상
    elif inverse_strength <= 100 and market_strength >= 100:
        if config["action"] in ("BUY_LEVERAGE", "BUY_LONG"):
            grade = "STRONG"
            reasoning += f" | 인버스 강도 {inverse_strength:.0f} 약화 + 시장 강도 {market_strength:.0f} 회복 = 격상"

    # 보정: 보수적 grade 다운 (NEUTRAL 강도 → MEDIUM이 STRONG이면 다운)
    elif market_regime == "NEUTRAL" and abs(market_strength - 100) < 5:
        # 진짜 보합이면 WATCH 유지
        pass

    return ETFRecommendation(
        action=config["action"],
        ticker=config["ticker"],
        name=config["name"],
        grade=grade,
        reasoning=reasoning,
        suggested_size_won=config["size"],
    )


def format_etf_for_advisory(rec: ETFRecommendation) -> dict:
    """advisory reasoning(JSONB)에 포함할 dict 변환."""
    return {
        "etf_action": rec.action,
        "etf_ticker": rec.ticker,
        "etf_name": rec.name,
        "etf_grade": rec.grade,
        "etf_reasoning": rec.reasoning,
        "etf_size_won": rec.suggested_size_won,
    }


def format_etf_for_telegram(rec: ETFRecommendation) -> str:
    """텔레그램 메시지용 1줄."""
    if rec.action in ("HOLD", "CASH_UP"):
        return f"📦 ETF: {rec.action} — {rec.reasoning}"
    grade_emoji = {"STRONG": "🟢", "MEDIUM": "🟡", "WATCH": "🟠"}.get(rec.grade, "⚪")
    return f"📦 ETF {grade_emoji} [{rec.grade}] {rec.name}({rec.ticker}) — {rec.reasoning}"
