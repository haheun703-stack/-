"""체결강도 시계열 추적 — 90 돌파 + 추세 감지.

배경 (5/22 퐝가님 인사이트):
  "체결강도 90 이상부터 수급이 들어오면서 오른다"

기존 entry_gates.py의 C 게이트는 단일 임계 150만 사용 → 항상 늦은 진입.
이 모듈은 자비스 cron (매 5분) 측정값을 JSON 시계열로 저장하고,
직전 cron 대비 추세 판정 + 90 돌파 시점 감지로 1~2 cron 빠른 조기 진입 시그널 제공.

흐름:
  자비스 cron (매 5분 14:00~14:55):
    1. entry_gates._fetch_volume_power(ticker) → 체결강도 측정
    2. volume_power_tracker.record_vp(ticker, vp) → 시계열 저장
    3. volume_power_tracker.calculate_vp_score(ticker, vp) → 추세 + 점수 산출
    4. entry_score 통합 점수에 합산

저장: data/volume_power_history.json (종목당 최근 12건 = 1시간 윈도우)
TTL: 24시간 자동 만료
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_PATH = PROJECT_ROOT / "data" / "volume_power_history.json"

# 추세 임계 (5/22 퐝가님 지식 반영)
TREND_BREAKOUT_THRESHOLD = 90    # 수급 진입 시작 (조기 시그널)
TREND_STABLE_THRESHOLD = 100     # 매수 안정
TREND_STRONG_THRESHOLD = 150     # 강한 매수세 (단타봇 동급)

# 추세 판정 (delta 기준)
TREND_RISING_DELTA = 2.0    # 직전 대비 +2pt 이상 = 상승
TREND_FALLING_DELTA = -2.0  # 직전 대비 -2pt 이하 = 하락

# 시계열 관리
MAX_POINTS_PER_TICKER = 12   # 종목당 최대 12건 (5분 × 12 = 1시간)
HISTORY_TTL_HOURS = 24       # 24시간 후 만료


def _load_history() -> dict:
    """JSON 파일에서 체결강도 시계열 로드 + 만료 정리."""
    if not HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        cutoff = datetime.now() - timedelta(hours=HISTORY_TTL_HOURS)
        cleaned = {}
        for ticker, points in data.items():
            kept = []
            for p in points:
                try:
                    t = datetime.fromisoformat(p["time"])
                    if t > cutoff:
                        kept.append(p)
                except (ValueError, KeyError):
                    continue
            if kept:
                cleaned[ticker] = kept
        return cleaned
    except Exception as e:
        logger.warning("volume_power_history 로드 실패: %s", e)
        return {}


def _save_history(data: dict) -> None:
    """JSON 파일에 시계열 저장."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        HISTORY_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("volume_power_history 저장 실패: %s", e)


def record_vp(ticker: str, vp: float) -> None:
    """체결강도 측정값을 시계열에 추가.

    자비스 cron 매 5분 호출 시 직후 저장.
    종목당 최근 12건만 유지 (1시간 윈도우).
    """
    if vp is None or vp < 0:
        return
    history = _load_history()
    if ticker not in history:
        history[ticker] = []
    history[ticker].append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "vp": round(float(vp), 2),
    })
    history[ticker] = history[ticker][-MAX_POINTS_PER_TICKER:]
    _save_history(history)


def analyze_trend(ticker: str, current_vp: float) -> dict:
    """체결강도 추세 분석 (직전 cron 대비).

    Returns:
        {
            "current": float,                    # 현재 측정값
            "previous": float | None,            # 직전 cron 측정값
            "trend": "rising"|"falling"|"flat"|"first",
            "delta": float,                       # current - previous
            "breakout_90": bool,                  # 90 신규 돌파 (직전 < 90 ≤ 현재)
            "breakout_100": bool,                 # 100 신규 돌파
            "n_history": int,                     # 시계열 데이터 수
        }
    """
    history = _load_history()
    points = history.get(ticker, [])

    result = {
        "current": current_vp,
        "previous": None,
        "trend": "first",
        "delta": 0.0,
        "breakout_90": False,
        "breakout_100": False,
        "n_history": len(points),
    }

    if not points:
        return result

    prev_vp = float(points[-1].get("vp", 0))
    result["previous"] = prev_vp
    result["delta"] = round(current_vp - prev_vp, 2)

    if result["delta"] >= TREND_RISING_DELTA:
        result["trend"] = "rising"
    elif result["delta"] <= TREND_FALLING_DELTA:
        result["trend"] = "falling"
    else:
        result["trend"] = "flat"

    # 돌파 판정 (직전이 임계 미만 + 현재가 임계 이상)
    if prev_vp < TREND_BREAKOUT_THRESHOLD <= current_vp:
        result["breakout_90"] = True
    if prev_vp < TREND_STABLE_THRESHOLD <= current_vp:
        result["breakout_100"] = True

    return result


def calculate_vp_score(ticker: str, current_vp: float) -> dict:
    """체결강도 점수 + 추세 반영 (5/22 퐝가님 지식).

    등급:
        ≥ 150              : +3 (강한 매수세)
        100~150 + rising   : +3 (안정 + 상승 확인)
        100~150 + flat     : +2 (안정)
        100~150 + falling  : +1 (안정 그러나 약세 전환)
        90~100 + rising    : +3 ★ 조기 진입 (퐝가님 인사이트)
        90~100 + flat      : +2 (수급 진입 시작 정체)
        90~100 + falling   : +1 (수급 진입 후 후퇴)
        70~90 + delta≥+20  : +2 (수급 급반등)
        70~90 + rising     : +1
        70~90 + flat/falling: 0 (수급 부재)
        < 70               : -1 (매도 우세 페널티)

    Returns:
        {
            "score": int,                   # -1 ~ +3
            "vp": float,
            "trend": str,
            "previous_vp": float | None,
            "breakout_90": bool,
            "breakout_100": bool,
            "reason": str,                   # 사람이 읽을 수 있는 사유
        }
    """
    trend_info = analyze_trend(ticker, current_vp)
    t = trend_info["trend"]
    delta = trend_info["delta"]

    score = 0
    reason = ""

    if current_vp >= TREND_STRONG_THRESHOLD:
        score = 3
        reason = f"강세 ({current_vp:.0f} ≥ 150)"
    elif current_vp >= TREND_STABLE_THRESHOLD:
        if t == "rising":
            score = 3
            reason = f"안정+상승추세 ({current_vp:.0f}, 직전 {trend_info['previous']:.0f}, +{delta:.0f})"
        elif t == "flat":
            score = 2
            reason = f"안정 ({current_vp:.0f}, 정체)"
        elif t == "falling":
            score = 1
            reason = f"안정→약세 ({current_vp:.0f}, {delta:+.0f})"
        else:
            score = 2
            reason = f"안정 ({current_vp:.0f}, 첫측정)"
    elif current_vp >= TREND_BREAKOUT_THRESHOLD:
        if t == "rising":
            score = 3
            reason = f"★조기진입 ({current_vp:.0f}, 90+ 돌파 추세 {delta:+.0f})"
        elif t == "flat":
            score = 2
            reason = f"수급진입시작 ({current_vp:.0f}, 정체)"
        elif t == "falling":
            score = 1
            reason = f"수급진입후후퇴 ({current_vp:.0f}, {delta:+.0f})"
        else:
            score = 2
            reason = f"수급진입 ({current_vp:.0f}, 첫측정)"
    elif current_vp >= 70:
        if delta >= 20:
            score = 2
            reason = f"수급급반등 ({current_vp:.0f}, 직전 {trend_info['previous']:.0f}→ {delta:+.0f})"
        elif t == "rising":
            score = 1
            reason = f"수급회복 ({current_vp:.0f}, {delta:+.0f})"
        else:
            score = 0
            reason = f"수급부재 ({current_vp:.0f})"
    else:
        score = -1
        reason = f"매도우세 ({current_vp:.0f} < 70) — 진입 페널티"

    return {
        "score": score,
        "vp": round(current_vp, 2),
        "trend": t,
        "previous_vp": trend_info["previous"],
        "delta": delta,
        "breakout_90": trend_info["breakout_90"],
        "breakout_100": trend_info["breakout_100"],
        "n_history": trend_info["n_history"],
        "reason": reason,
    }
