"""진입 점수 시스템 — Level 1 필수 + Level 2 가중 점수.

배경 (5/22 백테스트 + 퐝가님 지식):
  단순 임계 (3/3 모두 통과)는 정밀도 부족 → 시그널 강도별 가중 점수로 정밀 진입.
  VWAP 상/중/하, 호가 강도, 체결강도 90+ 추세 모두 반영한 통합 시스템.

Level 1 — 필수 조건 (모두 통과 필요, AND 게이트):
  L1-1. VWAP × 0.995 이상 (하향이탈 차단)
  L1-2. 5분봉 양봉 비율 ≥ 50% (직전 30분 기준)
  L1-3. 마지막 5분봉 양봉
  L1-4. 체결강도 ≥ 70 (매도 우세 차단)

Level 2 — 가중 점수 (총 -1 ~ +13점):
  VWAP 위치 (0~+3)     | 양봉비율 (0~+2)
  거래량 서지 (0~+2)    | 호가 강도 (0~+2)
  체결강도+추세 (-1~+3) | ← 5/22 퐝가님 90+ 인사이트

진입 결정 (3단계):
  ≥ 10점: ★적극 매수 (한도 100%)
  7~9점:  신중 매수 (차트영웅 D+1 양봉 확인 후)
  < 7점:  진입 차단

사용:
  from src.use_cases.entry_score import calculate_entry_score
  result = calculate_entry_score(broker, "067310")
  if result["passed_active"]:
      매수()
"""

from __future__ import annotations

import logging
from typing import Any

from src.use_cases.entry_gates import (
    check_intraday_gate,
    check_orderbook_gate,
    check_volume_power_gate,
)
from src.use_cases.volume_power_tracker import (
    calculate_vp_score,
    record_vp,
)

logger = logging.getLogger(__name__)

# Level 1 필수 임계 (절대 통과 조건)
REQUIRED_VWAP_FLOOR = 0.995      # 현재가 ≥ VWAP × 0.995 (하향 -0.5% 허용)
REQUIRED_BULL_RATIO = 0.5         # 5분봉 양봉 비율 ≥ 50%
REQUIRED_VP_MIN = 70.0            # 체결강도 ≥ 70 (매도 우세 차단)

# Level 2 진입 결정 임계
SCORE_ACTIVE_MIN = 10             # 적극 매수 (≥10점)
SCORE_ADJUSTABLE_MIN = 7          # 신중 매수 (7~9점)


def _score_vwap(current: float, vwap: float) -> tuple[int, str]:
    """VWAP 위치 점수 (0~+3)."""
    if vwap <= 0 or current <= 0:
        return 0, "VWAP 부재"
    ratio = current / vwap
    pct = (ratio - 1) * 100
    if ratio >= 1.015:
        return 3, f"VWAP상 ({pct:+.2f}%)"
    elif ratio >= 1.005:
        return 2, f"VWAP중상 ({pct:+.2f}%)"
    elif ratio >= 0.995:
        return 1, f"VWAP중 ({pct:+.2f}%)"
    else:
        return 0, f"VWAP하 ({pct:+.2f}%)"


def _score_bull_ratio(bull_ratio: float) -> tuple[int, str]:
    """양봉 비율 점수 (0~+2)."""
    pct = bull_ratio * 100
    if bull_ratio >= 0.7:
        return 2, f"양봉강 ({pct:.0f}%)"
    elif bull_ratio >= 0.5:
        return 1, f"양봉중 ({pct:.0f}%)"
    else:
        return 0, f"양봉약 ({pct:.0f}%)"


def _score_volume(vol_surge: bool) -> tuple[int, str]:
    """거래량 서지 점수 (0~+2).

    현재 entry_gates는 bool만 반환 (vol_surge ≥ 1.5x).
    추후 entry_gates 수정 후 1.5x / 2.0x 구분 가능.
    """
    if vol_surge:
        return 2, "거래량 서지 (≥1.5x)"
    return 0, "거래량 평이"


def _score_orderbook(bid_ask_ratio: float, depth_ratio: float) -> tuple[int, str]:
    """호가 강도 점수 (0~+2)."""
    if bid_ask_ratio >= 2.0:
        return 2, f"호가강 (매수1/매도1 {bid_ask_ratio:.2f}x)"
    elif bid_ask_ratio >= 1.0:
        return 1, f"호가중 (매수1/매도1 {bid_ask_ratio:.2f}x)"
    else:
        return 0, f"호가약 (매수1/매도1 {bid_ask_ratio:.2f}x)"


def calculate_entry_score(
    broker,
    ticker: str,
    current_price: int | None = None,
) -> dict[str, Any]:
    """진입 점수 계산 (Level 1 필수 + Level 2 가중).

    Args:
        broker: KIS broker (mojito 또는 CachedBroker)
        ticker: 종목 코드
        current_price: 평가 시점 현재가 (None이면 broker.fetch_price 호출)

    Returns:
        {
            "passed_required": bool,    # Level 1 모든 필수 통과
            "passed_active": bool,        # 총점 ≥ 10 (적극 매수)
            "passed_adjustable": bool,    # 총점 ≥ 7 (신중 매수)
            "score": int,                  # Level 2 총점 (-1 ~ +13)
            "breakdown": dict,             # 시그널별 {score, reason}
            "blocks": list[str],           # Level 1 실패 사유
            "reasoning": str,              # 종합 요약
            "vp": float,                   # 체결강도 현재 측정값
            "vwap_last": float,
            "current_price": int,
        }
    """
    blocks: list[str] = []

    # ── A. 5분봉 + VWAP + 거래량 게이트 ──
    intraday = check_intraday_gate(broker, ticker, current_price=current_price)
    bull_ratio = float(intraday.get("bull_ratio", 0) or 0)
    last_bull = bool(intraday.get("last_bull", False))
    vol_surge = bool(intraday.get("vol_surge", False))
    vwap_last = float(intraday.get("vwap_last", 0) or 0)
    current = int(intraday.get("current_price", current_price or 0) or 0)

    # ── B. 호가 게이트 ──
    orderbook = check_orderbook_gate(broker, ticker)
    bid_ask_ratio = float(orderbook.get("bid_ask_ratio", 0) or 0)
    depth_ratio = float(orderbook.get("depth_ratio", 0) or 0)

    # ── C. 체결강도 (disabled 모드로 호출 → 측정값만 추출) ──
    vp_result = check_volume_power_gate(broker, ticker, threshold=0)
    vp = float(vp_result.get("volume_power", 0) or 0)

    # 체결강도 시계열 저장 (다음 cron 비교용, 추세 판정)
    if vp > 0:
        record_vp(ticker, vp)
    vp_score_info = calculate_vp_score(ticker, vp)

    # ── Level 1 필수 조건 검증 ──
    # L1-1: VWAP 하향이탈 차단
    if vwap_last > 0 and current > 0:
        vwap_ratio = current / vwap_last
        if vwap_ratio < REQUIRED_VWAP_FLOOR:
            blocks.append(f"VWAP하향이탈 ({(vwap_ratio-1)*100:+.2f}% < -0.5%)")

    # L1-2: 양봉 비율 50%+
    if bull_ratio < REQUIRED_BULL_RATIO:
        blocks.append(f"양봉비율 {bull_ratio*100:.0f}% < 50%")

    # L1-3: 마지막 5분봉 양봉
    if not last_bull:
        blocks.append("마지막5분봉음봉")

    # L1-4: 체결강도 70+ (매도 우세 차단)
    if vp < REQUIRED_VP_MIN:
        blocks.append(f"체결강도 {vp:.0f} < 70")

    passed_required = len(blocks) == 0

    # ── Level 2 가중 점수 ──
    s_vwap, r_vwap = _score_vwap(current, vwap_last)
    s_bull, r_bull = _score_bull_ratio(bull_ratio)
    s_vol, r_vol = _score_volume(vol_surge)
    s_ob, r_ob = _score_orderbook(bid_ask_ratio, depth_ratio)
    s_vp = vp_score_info["score"]
    r_vp = vp_score_info["reason"]

    breakdown = {
        "vwap":      {"score": s_vwap, "reason": r_vwap},
        "bull":      {"score": s_bull, "reason": r_bull},
        "volume":    {"score": s_vol,  "reason": r_vol},
        "orderbook": {"score": s_ob,   "reason": r_ob},
        "vp":        {"score": s_vp,   "reason": r_vp},
    }

    total_score = s_vwap + s_bull + s_vol + s_ob + s_vp

    passed_active = passed_required and total_score >= SCORE_ACTIVE_MIN
    passed_adjustable = passed_required and total_score >= SCORE_ADJUSTABLE_MIN

    if not passed_required:
        reasoning = f"진입차단: {'; '.join(blocks)}"
    elif passed_active:
        reasoning = f"★적극매수 (총점 {total_score} ≥ 10)"
    elif passed_adjustable:
        reasoning = f"신중매수 (총점 {total_score}, 7~9, D+1확인후)"
    else:
        reasoning = f"진입차단 (총점 {total_score} < 7)"

    return {
        "passed_required": passed_required,
        "passed_active": passed_active,
        "passed_adjustable": passed_adjustable,
        "score": total_score,
        "breakdown": breakdown,
        "blocks": blocks,
        "reasoning": reasoning,
        "vp": vp,
        "vp_trend": vp_score_info["trend"],
        "vp_breakout_90": vp_score_info["breakout_90"],
        "vp_breakout_100": vp_score_info["breakout_100"],
        "vwap_last": vwap_last,
        "current_price": current,
    }


def format_entry_score_summary(result: dict[str, Any]) -> str:
    """진입 점수 결과를 사람이 읽기 좋게 포맷 (로그용)."""
    bd = result.get("breakdown", {})
    lines = [
        f"  진입점수: {result['score']:+d}점 | {result['reasoning']}",
    ]
    if bd:
        parts = []
        for k in ["vwap", "bull", "volume", "orderbook", "vp"]:
            v = bd.get(k, {})
            parts.append(f"{k}={v.get('score',0):+d}({v.get('reason','')})")
        lines.append(f"  세부: {' / '.join(parts)}")
    if result.get("vp_breakout_90"):
        lines.append(f"  ★ 90 돌파 시그널 (체결강도 {result['vp']:.0f}, 추세 {result['vp_trend']})")
    return "\n".join(lines)
