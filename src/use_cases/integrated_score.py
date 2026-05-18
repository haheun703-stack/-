"""사장님 EYE 통합 점수 함수 (2026-05-18 신규)

배경: 5/18 모든 부품 EYE 통합 (사장님 13:00 통찰 "맵핑 + 연결")
- 자비스가 가진 모든 신호를 한 점수로 통합 → STRONG/MEDIUM/WATCH/SKIP 자동 분류

점수 구성 (0~100):
  ① EYE 필터 통과 (-30 가산 if SKIP)  : 사전 위험 회피
  ② 가격 시그널 (시초가 대비, 0~25)    : 강세 종목 우선
  ③ 프로그램 매수 (0~15)                : 프로그램 + 가격 일치
  ④ 거래량 (0~15)                       : 활발한 거래
  ⑤ advisory regime (0~15)              : 시장 매크로 보정
  ⑥ EYE-07 알림 횟수 (0~10)             : 자비스 자체 EYE (5/19 통합 예정)
  ⑦ VWAP 과열 (0~10)                    : 강세 추격 가능 (5/19 통합 예정)

등급 (5/18 잠정):
  STRONG  (80~100): 1주/10만원 매수 가능 (5/20 안전선)
  MEDIUM  (60~79):  관찰 + 추가 시그널 대기
  WATCH   (40~59):  관찰만
  SKIP    (0~39):   진입 금지

Note: 5/19 새벽 정밀화 예정 (EYE-07 알림 횟수 + VWAP 과열 통합)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class IntegratedScore:
    """종합 점수 결과."""

    ticker: str
    name: str
    score: float  # 0~100
    grade: str  # STRONG / MEDIUM / WATCH / SKIP
    components: dict = field(default_factory=dict)  # 점수 구성 상세
    reasoning: list[str] = field(default_factory=list)


def grade_from_score(score: float) -> str:
    if score >= 80:
        return "STRONG"
    if score >= 60:
        return "MEDIUM"
    if score >= 40:
        return "WATCH"
    return "SKIP"


def calculate_integrated_score(
    broker,
    ticker: str,
    name: str = "",
    eye_filter_result: dict | None = None,
    market_regime: str = "NEUTRAL",
) -> IntegratedScore:
    """종합 점수 계산.

    Args:
        broker: KIS mojito broker
        ticker: 6자리 종목코드
        name: 종목명
        eye_filter_result: src.use_cases.eye_filters.evaluate_filters() 결과
        market_regime: snapshot_session.py의 advisory regime (NEUTRAL/MILD_BULL/STRONG_BULL/CAUTION/BEAR)
    """
    components = {}
    reasoning = []
    score = 50.0  # 기본 시작점 (중립)

    # ① EYE 필터 (사전 위험 회피)
    if eye_filter_result:
        if eye_filter_result.get("should_skip"):
            score -= 30
            components["eye_filter"] = -30
            reasoning.append(f"EYE SKIP: {', '.join(eye_filter_result.get('skip_reasons', []))}")
        else:
            score += 10  # EYE 통과 가산
            components["eye_filter"] = +10
            reasoning.append("EYE 필터 통과")

    # KIS API fetch_price
    try:
        resp = broker.fetch_price(ticker)
        out = resp.get("output", {}) if resp else {}
    except Exception as e:
        logger.warning("fetch_price 실패 %s: %s", ticker, e)
        return IntegratedScore(
            ticker=ticker, name=name, score=0, grade="SKIP",
            components={"error": "fetch_failed"},
            reasoning=[f"KIS 호출 실패: {e}"]
        )

    cur = int(out.get("stck_prpr", 0) or 0)
    opn = int(out.get("stck_oprc", 0) or 0)
    pgtr = int(out.get("pgtr_ntby_qty", 0) or 0)
    vol_rate = float(out.get("prdy_vrss_vol_rate", 0) or 0)
    intra_chg = ((cur - opn) / opn * 100) if opn > 0 else 0

    # ② 가격 시그널 (시초가 대비 0~25)
    if intra_chg >= 7:
        score += 25
        components["price"] = +25
        reasoning.append(f"강세 {intra_chg:+.2f}% (+25)")
    elif intra_chg >= 4:
        score += 20
        components["price"] = +20
        reasoning.append(f"상승 {intra_chg:+.2f}% (+20)")
    elif intra_chg >= 2:
        score += 12
        components["price"] = +12
        reasoning.append(f"양봉 {intra_chg:+.2f}% (+12)")
    elif intra_chg >= 0:
        score += 5
        components["price"] = +5
        reasoning.append(f"보합 {intra_chg:+.2f}% (+5)")
    elif intra_chg >= -2:
        score -= 5
        components["price"] = -5
        reasoning.append(f"약세 {intra_chg:+.2f}% (-5)")
    else:
        score -= 15
        components["price"] = -15
        reasoning.append(f"하락 {intra_chg:+.2f}% (-15)")

    # ③ 프로그램 매수 (0~15)
    if pgtr >= 100_000:
        score += 15
        components["program"] = +15
        reasoning.append(f"프로그램 대량 매수 {pgtr:+,} (+15)")
    elif pgtr >= 30_000:
        score += 10
        components["program"] = +10
        reasoning.append(f"프로그램 매수 {pgtr:+,} (+10)")
    elif pgtr >= 0:
        score += 3
        components["program"] = +3
        reasoning.append(f"프로그램 중립 {pgtr:+,} (+3)")
    elif pgtr >= -100_000:
        score -= 3
        components["program"] = -3
        reasoning.append(f"프로그램 매도 {pgtr:+,} (-3)")
    else:
        # 대량 매도지만 가격 강세면 외인/기관 매수 압도 = 감점 적게
        score -= 5 if intra_chg > 0 else 10
        components["program"] = -5 if intra_chg > 0 else -10
        reasoning.append(f"프로그램 대량매도 {pgtr:+,} (가격 {intra_chg:+.2f}%)")

    # ④ 거래량 (0~15)
    if vol_rate >= 100:
        score += 15
        components["volume"] = +15
        reasoning.append(f"거래량 폭발 {vol_rate:.0f}% (+15)")
    elif vol_rate >= 60:
        score += 10
        components["volume"] = +10
        reasoning.append(f"거래량 활발 {vol_rate:.0f}% (+10)")
    elif vol_rate >= 30:
        score += 3
        components["volume"] = +3
        reasoning.append(f"거래량 보통 {vol_rate:.0f}% (+3)")
    else:
        score -= 5
        components["volume"] = -5
        reasoning.append(f"거래량 부진 {vol_rate:.0f}% (-5)")

    # ⑤ advisory regime (0~15)
    regime_score = {
        "STRONG_BULL": 15, "MILD_BULL": 10, "NEUTRAL": 0,
        "CAUTION": -8, "BEAR": -15, "CRISIS": -20,
    }.get(market_regime, 0)
    score += regime_score
    components["regime"] = regime_score
    reasoning.append(f"시장 {market_regime} ({regime_score:+d})")

    # 점수 범위 0~100 클램프
    score = max(0, min(100, score))
    grade = grade_from_score(score)

    return IntegratedScore(
        ticker=ticker, name=name, score=round(score, 1), grade=grade,
        components=components, reasoning=reasoning,
    )


def evaluate_picks_integrated(
    broker,
    picks: list[tuple[str, str]],
    eye_filters_results: dict[str, dict] | None = None,
    market_regime: str = "NEUTRAL",
) -> list[IntegratedScore]:
    """여러 종목 일괄 종합 점수 계산."""
    eye_filters_results = eye_filters_results or {}
    results = []
    for tk, nm in picks:
        eye_res = eye_filters_results.get(tk)
        sc = calculate_integrated_score(broker, tk, nm, eye_res, market_regime)
        results.append(sc)
    return sorted(results, key=lambda x: x.score, reverse=True)
