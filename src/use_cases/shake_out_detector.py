"""개미털기 vs 진짜 문제 판별 — 역매수 전략 핵심 (5/22 퐝가님 인사이트).

배경:
  기가막힌 조합 (C2) 통과 종목이 D+1~D+3 마이너스 진입 시:
  - 개미털기 (역매수 OK): 외인/기관 매수 유지 + 호가 균형 + 거래량 평이 + 적정 하락
  - 진짜 문제 (회피): 대량 매도 + 호가 매도 우세 + 거래량 폭증 + 큰 폭 하락

5/22 백테스트 (382건) 마이너스 종목 105건 중:
  대부분 5/15~5/22 약세 전환 시기 = 시장 영향 (개미털기 + 동시 약세 혼재)
  → 매크로 가드 통과 + 개별 종목 판별 필수

5종 시그널 (각 -2 ~ +2 점, 총 -10 ~ +10):
  1. 외인 5일 누적 매수
  2. 기관 5일 누적 매수
  3. 호가 매수1/매도1 비율
  4. 거래량 vol_ratio (vs 평균)
  5. 체결강도 + 추세

판별:
  +5 이상: 명백한 개미털기 (역매수 진행)
  0 ~ +4:   불확실 (보수적 관망)
  -5 이하: 진짜 문제 (회피 + 손절 검토)

사용:
  from src.use_cases.shake_out_detector import detect_shake_out
  result = detect_shake_out(broker, ticker, current_price, entry_price, pick_record)
  if result["is_shakeout"]:
      # 역매수 시그널 발동
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 시그널 임계 (백테스트 검증 후 조정 가능)
FOREIGN_NET_5D_OK = 0           # > 0 = 외인 매수 유지
INST_NET_5D_OK = 0               # > 0 = 기관 매수 유지
BID_ASK_OK = 0.8                  # ≥ 0.8 = 호가 균형
BID_ASK_BAD = 0.5                 # < 0.5 = 매도 우세
VOL_RATIO_NORMAL = 2.0           # < 2.0 = 거래량 평이
VOL_RATIO_PANIC = 3.0             # ≥ 3.0 = 거래량 폭증 (패닉 매도)
VP_OK = 70.0                      # ≥ 70 = 수급 유지
VP_BAD = 50.0                     # < 50 = 대량 매도

# 판별 임계
SHAKEOUT_SCORE_MIN = 5            # +5 이상 = 명백한 개미털기
PROBLEM_SCORE_MAX = -5            # -5 이하 = 진짜 문제

# 하락폭 임계
DECLINE_HEALTHY_RANGE = (-10.0, -5.0)    # 적정 (개미털기 가능)
DECLINE_PANIC_THRESHOLD = -15.0           # 큰 폭 (진짜 문제 추정)


def _score_foreign_inst(foreign_5d: float, inst_5d: float) -> tuple[int, str]:
    """외인 + 기관 5일 누적 점수 (-4 ~ +4)."""
    score = 0
    reason_parts = []
    if foreign_5d > 0:
        score += 2
        reason_parts.append(f"외인5d +{foreign_5d/1e8:.1f}억(+2)")
    else:
        score -= 2
        reason_parts.append(f"외인5d {foreign_5d/1e8:.1f}억(-2)")
    if inst_5d > 0:
        score += 2
        reason_parts.append(f"기관5d +{inst_5d/1e8:.1f}억(+2)")
    else:
        score -= 2
        reason_parts.append(f"기관5d {inst_5d/1e8:.1f}억(-2)")
    return score, " / ".join(reason_parts)


def _score_orderbook(bid_ask_ratio: float) -> tuple[int, str]:
    """호가 매수1/매도1 점수 (-2 ~ +2)."""
    if bid_ask_ratio >= 1.5:
        return 2, f"호가매수우세 ({bid_ask_ratio:.2f}x +2)"
    elif bid_ask_ratio >= BID_ASK_OK:
        return 1, f"호가균형 ({bid_ask_ratio:.2f}x +1)"
    elif bid_ask_ratio >= BID_ASK_BAD:
        return 0, f"호가중립 ({bid_ask_ratio:.2f}x 0)"
    else:
        return -2, f"호가매도우세 ({bid_ask_ratio:.2f}x -2)"


def _score_volume(vol_ratio: float) -> tuple[int, str]:
    """거래량 비율 점수 (-2 ~ +2)."""
    if vol_ratio < 1.0:
        return 2, f"거래량 감소 ({vol_ratio:.2f}x +2 — 개미털기 신호)"
    elif vol_ratio < VOL_RATIO_NORMAL:
        return 1, f"거래량 평이 ({vol_ratio:.2f}x +1)"
    elif vol_ratio < VOL_RATIO_PANIC:
        return 0, f"거래량 증가 ({vol_ratio:.2f}x 0)"
    else:
        return -2, f"거래량 폭증 ({vol_ratio:.2f}x -2 — 패닉 매도 의심)"


def _score_volume_power(vp: float) -> tuple[int, str]:
    """체결강도 점수 (-2 ~ +2)."""
    if vp >= 100:
        return 2, f"체결강도 강세 ({vp:.0f} +2)"
    elif vp >= VP_OK:
        return 1, f"체결강도 유지 ({vp:.0f} +1)"
    elif vp >= VP_BAD:
        return 0, f"체결강도 중립 ({vp:.0f} 0)"
    else:
        return -2, f"체결강도 약세 ({vp:.0f} -2 — 대량 매도)"


def detect_shake_out(
    foreign_5d: float,
    inst_5d: float,
    current_price: float,
    entry_price: float,
    bid_ask_ratio: Optional[float] = None,
    vol_ratio: Optional[float] = None,
    volume_power: Optional[float] = None,
) -> dict[str, Any]:
    """개미털기 vs 진짜 문제 판별.

    Args:
        foreign_5d: 외인 5일 누적 매수 (원, picks_history.foreign_5d)
        inst_5d: 기관 5일 누적 매수 (원, picks_history.inst_5d)
        current_price: 현재가
        entry_price: 진입가 (보유 종목) 또는 픽업 시점 close
        bid_ask_ratio: 호가 매수1/매도1 (None = 데이터 부재)
        vol_ratio: 거래량 비율 vs 평균 (None = 데이터 부재)
        volume_power: 체결강도 (None = 데이터 부재)

    Returns:
        {
            "is_shakeout": bool,            # +5 이상 = 명백한 개미털기
            "is_problem": bool,              # -5 이하 = 진짜 문제
            "score": int,                     # -10 ~ +10
            "decline_pct": float,             # 진입가 대비 하락률
            "signals": dict,                  # 각 시그널 {score, reason}
            "n_data_available": int,          # 시그널 데이터 수
            "reason": str,                    # 종합 사유
            "recommendation": str,             # "REBUY" | "WATCH" | "CUT"
        }
    """
    signals = {}
    total_score = 0
    n_available = 0

    # 1. 외인 + 기관 (picks_history 데이터, 거의 항상 있음)
    fi_score, fi_reason = _score_foreign_inst(foreign_5d or 0, inst_5d or 0)
    signals["foreign_inst"] = {"score": fi_score, "reason": fi_reason}
    total_score += fi_score
    n_available += 2

    # 2. 호가 (실시간 KIS API 필요, 없으면 skip)
    if bid_ask_ratio is not None and bid_ask_ratio > 0:
        ob_score, ob_reason = _score_orderbook(bid_ask_ratio)
        signals["orderbook"] = {"score": ob_score, "reason": ob_reason}
        total_score += ob_score
        n_available += 1
    else:
        signals["orderbook"] = {"score": 0, "reason": "호가 데이터 부재 (skip)"}

    # 3. 거래량 (jgis OHLCV 또는 분봉)
    if vol_ratio is not None and vol_ratio > 0:
        v_score, v_reason = _score_volume(vol_ratio)
        signals["volume"] = {"score": v_score, "reason": v_reason}
        total_score += v_score
        n_available += 1
    else:
        signals["volume"] = {"score": 0, "reason": "거래량 데이터 부재 (skip)"}

    # 4. 체결강도 (volume_power_tracker)
    if volume_power is not None and volume_power > 0:
        vp_score, vp_reason = _score_volume_power(volume_power)
        signals["vp"] = {"score": vp_score, "reason": vp_reason}
        total_score += vp_score
        n_available += 1
    else:
        signals["vp"] = {"score": 0, "reason": "체결강도 부재 (skip)"}

    # 하락률
    if entry_price > 0:
        decline_pct = (current_price - entry_price) / entry_price * 100
    else:
        decline_pct = 0

    # 큰 폭 하락 추가 페널티 (진짜 문제 의심)
    if decline_pct <= DECLINE_PANIC_THRESHOLD:
        total_score -= 2
        signals["panic_decline"] = {
            "score": -2,
            "reason": f"하락폭 {decline_pct:.2f}% ≤ -15% (큰 폭 하락 -2)",
        }

    # 판별
    is_shakeout = total_score >= SHAKEOUT_SCORE_MIN
    is_problem = total_score <= PROBLEM_SCORE_MAX

    # 추천
    if is_shakeout:
        recommendation = "REBUY"
        reason = f"★개미털기 추정 (점수 {total_score:+d} ≥ +5) — 역매수 진행"
    elif is_problem:
        recommendation = "CUT"
        reason = f"진짜 문제 추정 (점수 {total_score:+d} ≤ -5) — 손절 검토"
    else:
        recommendation = "WATCH"
        reason = f"불확실 (점수 {total_score:+d}) — 추가 관망"

    return {
        "is_shakeout": is_shakeout,
        "is_problem": is_problem,
        "score": total_score,
        "decline_pct": round(decline_pct, 2),
        "signals": signals,
        "n_data_available": n_available,
        "reason": reason,
        "recommendation": recommendation,
    }
