"""
Step 3: ICT 필터 — 스윙봇 진입 시 프리미엄 레벨 + OR/IR bias 참조

핵심 원칙:
  - ICT 필터는 "차단"하지 않는다. 신뢰도 조절만 한다.
  - 차단은 BRAIN 캡의 역할 (CRISIS → 신규진입 금지).
  - daily_bias가 unknown이면 개입하지 않는다 (graceful degradation).
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.ict.premium_levels import load_premium_levels
from src.ict.opening_range import load_or_ir

logger = logging.getLogger(__name__)


def ict_check(symbol: str, signal_type: str, date_str: str | None = None) -> dict:
    """ICT 기반 진입 품질 판단.

    Args:
        symbol: 종목코드 (예: "005930")
        signal_type: "buy" 또는 "sell"
        date_str: 날짜 (YYYY-MM-DD). None이면 오늘.

    Returns:
        {
            "pass": True,  # ICT는 항상 pass (차단 안 함)
            "confidence_adjust": float,  # -0.2 ~ +0.1
            "reason": str,
            "premium_levels": dict | None,
            "daily_bias": str,
            "suggested_target": int | None,
            "suggested_stop": int | None,
            "tags": list[str],
        }
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    result = {
        "pass": True,
        "confidence_adjust": 0.0,
        "reason": "ICT 데이터 없음, 기존 로직 유지",
        "premium_levels": None,
        "daily_bias": "unknown",
        "suggested_target": None,
        "suggested_stop": None,
        "tags": [],
    }

    # 1. 프리미엄 레벨 로드
    levels = load_premium_levels(date_str, symbol)

    # 2. OR/IR 바이어스 로드
    or_ir = load_or_ir(date_str, symbol)
    daily_bias = or_ir.get("daily_bias", "unknown") if or_ir else "unknown"

    result["premium_levels"] = levels
    result["daily_bias"] = daily_bias

    if not levels and not or_ir:
        return result

    # ── 매수 시그널 판단 ──
    if signal_type == "buy":
        adjustments = []

        # 3. bias vs 매수 방향 일치 여부
        if daily_bias == "bearish":
            adjustments.append((-0.15, "당일 bias bearish (OR저 이탈)"))
            result["tags"].append("BIAS_AGAINST")
        elif daily_bias == "bullish":
            adjustments.append((+0.05, "당일 bias bullish 일치"))
            result["tags"].append("BIAS_ALIGNED")

        # 4. 저항 근접 매수 → 주의
        if levels:
            res = levels.get("nearest_resistance")
            if res and abs(res.get("distance_pct", 99)) < 1.0:
                level_name = _level_display(res["level"])
                adjustments.append((
                    -0.10,
                    f"저항 근접: {level_name} {res['price']:,} ({res['distance_pct']:+.1f}%)",
                ))
                result["tags"].append("NEAR_RESISTANCE")

            # 5. 지지 근접 매수 → 좋은 위치
            sup = levels.get("nearest_support")
            if sup and abs(sup.get("distance_pct", 99)) < 1.5:
                level_name = _level_display(sup["level"])
                adjustments.append((
                    +0.05,
                    f"지지 근접: {level_name} {sup['price']:,} ({sup['distance_pct']:+.1f}%)",
                ))
                result["tags"].append("NEAR_SUPPORT")

        # 6. OR 폭이 극도로 좁음 → 축적 가능성
        if or_ir:
            or_vs_atr = or_ir.get("or_vs_atr")
            if or_vs_atr is not None and or_vs_atr < 0.3:
                adjustments.append((+0.05, f"OR 극도 좁음 (ATR 대비 {or_vs_atr:.2f}) → 축적 가능"))
                result["tags"].append("NARROW_OR")

        # 합산
        if adjustments:
            total_adj = sum(a[0] for a in adjustments)
            # 클램핑: -0.2 ~ +0.1
            total_adj = max(-0.20, min(+0.10, total_adj))
            result["confidence_adjust"] = round(total_adj, 2)
            result["reason"] = " / ".join(a[1] for a in adjustments)
        else:
            result["reason"] = "ICT 특이사항 없음"

        # 7. 타겟/손절 제안
        if levels:
            res = levels.get("nearest_resistance")
            if res:
                result["suggested_target"] = res["price"]

            sup = levels.get("nearest_support")
            if sup:
                result["suggested_stop"] = sup["price"]

    # ── 매도 시그널 → ICT 개입 안 함 ──
    elif signal_type == "sell":
        result["reason"] = "매도 시그널 — ICT 개입 없음"

    return result


def _level_display(level_key: str) -> str:
    """레벨 키 → 한글"""
    mapping = {
        "prev_day_high": "전일고",
        "prev_day_low": "전일저",
        "prev_week_high": "주간고",
        "prev_week_low": "주간저",
        "prev_month_high": "월간고",
        "prev_month_low": "월간저",
    }
    return mapping.get(level_key, level_key)
