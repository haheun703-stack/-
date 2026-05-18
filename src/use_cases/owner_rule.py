"""사장님 이중 안전망 룰 — 5/20 출격일 적용 (2026-05-18 신규)

사장님 룰 (5/18 결정):
  ① 절대 손절: 진입가 -3% 도달 → 즉시 청산
  ② 트레일링 스톱: 고가 대비 -3% 도달 → 청산 (이익 보존)
  사장님 한 마디: "막 올랐다가 추세가 빠지면 -3% 손절해도 된다"

기존 paper_trading_unified와 다른 점:
- 기존 STOP_LOSS_PCT = -7% (-7%) → 사장님 룰 -3% (타이트)
- 기존 TRAILING_ACTIVATE_PCT = +10% (T1 익절 후 활성화) → 사장님 룰 +3% (빠른 활성화)
- 기존 TRAILING_STOP_PCT = -3% (peak 대비) → 사장님 룰 -3% (동일)

5/20 출격일에만 적용 (환경변수 OWNER_RULE_MODE=true).
이후 paper_portfolio.json에서 실전 검증 후 점진적 확대.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 사장님 룰 임계값 (5/18 결정)
OWNER_STOP_LOSS_PCT = -0.03       # 진입가 -3% 절대 손절
OWNER_TRAILING_ACTIVATE_PCT = 0.03  # 진입가 +3% 도달 시 트레일링 활성화
OWNER_TRAILING_STOP_PCT = -0.03    # peak 대비 -3% 청산
OWNER_FORCE_CLOSE_TIME = "15:20"   # 15:20 강제 청산 (NXT 안전마진)

# 사장님 룰 ④ 수급 지속 이월 (5/18 추가)
OWNER_HOLD_OVERNIGHT_MIN_GAIN_PCT = 0.03   # 종가 +3% 이상 양봉
OWNER_HOLD_OVERNIGHT_MIN_SUPPLY_EOK = 1.0  # 외인+기관+연기금 매수 +1억 이상
OWNER_MAX_HOLDING_DAYS = 5                  # 최대 보유 5일 (자비스 기존 패턴)


@dataclass
class OwnerRuleVerdict:
    """사장님 룰 평가 결과."""

    action: str  # 'HOLD' | 'SELL_STOP_LOSS' | 'SELL_TRAILING' | 'SELL_FORCE_CLOSE'
    reason: str
    entry_price: int
    current_price: int
    peak_price: int
    pnl_pct: float           # 진입가 대비 %
    peak_drop_pct: float     # peak 대비 %
    trailing_active: bool


def evaluate_owner_rule(
    entry_price: int,
    current_price: int,
    peak_price: int,
    trailing_active: bool = False,
    current_time: str = "",
) -> OwnerRuleVerdict:
    """사장님 이중 안전망 룰 평가.

    Args:
        entry_price: 진입가 (정수)
        current_price: 현재가 (정수)
        peak_price: 진입 이후 최고가 (정수)
        trailing_active: 트레일링 활성화 여부 (이전 상태)
        current_time: 현재 시각 "HH:MM" (15:20 강제 청산 체크용)

    Returns:
        OwnerRuleVerdict
    """
    if entry_price <= 0:
        return OwnerRuleVerdict(
            action="HOLD",
            reason="진입가 0 — 룰 평가 불가",
            entry_price=entry_price,
            current_price=current_price,
            peak_price=peak_price,
            pnl_pct=0,
            peak_drop_pct=0,
            trailing_active=trailing_active,
        )

    # 거래정지 / 응답 누락 가드 (5/18 E2 보강) — 0원에 매도 시도 방지
    if current_price <= 0:
        return OwnerRuleVerdict(
            action="HOLD",
            reason="현재가 0 (거래정지/응답 누락) — 매도 보류, 다음 cron 재시도",
            entry_price=entry_price,
            current_price=current_price,
            peak_price=peak_price,
            pnl_pct=0,
            peak_drop_pct=0,
            trailing_active=trailing_active,
        )

    # 진입가 대비 PnL
    pnl_pct = (current_price - entry_price) / entry_price

    # peak 갱신
    new_peak = max(peak_price, current_price)
    peak_drop_pct = (current_price - new_peak) / new_peak if new_peak > 0 else 0

    # 트레일링 활성화 여부 갱신
    if pnl_pct >= OWNER_TRAILING_ACTIVATE_PCT:
        trailing_active = True

    # 룰 ① 절대 손절 (-3%)
    if pnl_pct <= OWNER_STOP_LOSS_PCT:
        return OwnerRuleVerdict(
            action="SELL_STOP_LOSS",
            reason=f"진입가 -3% 절대 손절 (현재 {pnl_pct*100:+.2f}%)",
            entry_price=entry_price,
            current_price=current_price,
            peak_price=new_peak,
            pnl_pct=pnl_pct * 100,
            peak_drop_pct=peak_drop_pct * 100,
            trailing_active=trailing_active,
        )

    # 룰 ② 트레일링 스톱 (peak -3%, 활성화 상태에서)
    if trailing_active and peak_drop_pct <= OWNER_TRAILING_STOP_PCT:
        return OwnerRuleVerdict(
            action="SELL_TRAILING",
            reason=(
                f"트레일링 청산 (peak {new_peak:,} → 현재 {current_price:,}, "
                f"하락 {peak_drop_pct*100:+.2f}%, 이익 보존 {pnl_pct*100:+.2f}%)"
            ),
            entry_price=entry_price,
            current_price=current_price,
            peak_price=new_peak,
            pnl_pct=pnl_pct * 100,
            peak_drop_pct=peak_drop_pct * 100,
            trailing_active=trailing_active,
        )

    # 룰 ③ 15:20 강제 청산 (단, 룰 ④ 이월 조건 미달 시만)
    # 룰 ④는 호출자가 hold_overnight_eligible 인자로 결정 (수급 데이터 필요)
    if current_time and current_time >= OWNER_FORCE_CLOSE_TIME:
        return OwnerRuleVerdict(
            action="SELL_FORCE_CLOSE",
            reason=f"15:20 강제 청산 (NXT 안전마진, 현재 {pnl_pct*100:+.2f}%)",
            entry_price=entry_price,
            current_price=current_price,
            peak_price=new_peak,
            pnl_pct=pnl_pct * 100,
            peak_drop_pct=peak_drop_pct * 100,
            trailing_active=trailing_active,
        )

    # 보유 유지
    return OwnerRuleVerdict(
        action="HOLD",
        reason=(
            f"보유 유지 (PnL {pnl_pct*100:+.2f}%, peak {new_peak:,}, "
            f"trailing {'ON' if trailing_active else 'OFF'})"
        ),
        entry_price=entry_price,
        current_price=current_price,
        peak_price=new_peak,
        pnl_pct=pnl_pct * 100,
        peak_drop_pct=peak_drop_pct * 100,
        trailing_active=trailing_active,
    )


def evaluate_hold_overnight(
    entry_price: int,
    current_price: int,
    peak_price: int,
    trailing_active: bool,
    days_held: int,
    foreign_net_eok: float = 0.0,
    inst_net_eok: float = 0.0,
    pension_net_eok: float = 0.0,
    eye_filter_passed: bool = True,
) -> tuple[bool, dict]:
    """사장님 룰 ④ — 15:20 시점 익일 이월 가능 여부 평가.

    4 조건 ALL 통과 시 익일 이월:
    ① 종가 +3% 이상 양봉 (모멘텀)
    ② 외인+기관+연기금 매수 누적 +1억 이상 (수급 양호)
    ③ EYE 필터 5종 PASS (위험 신호 없음)
    ④ 트레일링 미발동 (peak 대비 -3% 이내)
    + ⑤ 최대 보유 5일 미만

    Returns:
        (이월 가능 여부, 상세 dict)
    """
    pnl_pct = (current_price - entry_price) / entry_price * 100
    new_peak = max(peak_price, current_price)
    peak_drop_pct = (current_price - new_peak) / new_peak * 100 if new_peak > 0 else 0
    supply_total_eok = foreign_net_eok + inst_net_eok + pension_net_eok

    checks = {
        "gain_3pct_plus": pnl_pct >= OWNER_HOLD_OVERNIGHT_MIN_GAIN_PCT * 100,
        "supply_positive": supply_total_eok >= OWNER_HOLD_OVERNIGHT_MIN_SUPPLY_EOK,
        "eye_passed": eye_filter_passed,
        "trailing_safe": peak_drop_pct > OWNER_TRAILING_STOP_PCT * 100,
        "holding_days_ok": days_held < OWNER_MAX_HOLDING_DAYS,
    }

    can_hold = all(checks.values())

    details = {
        "checks": checks,
        "pnl_pct": round(pnl_pct, 2),
        "peak_drop_pct": round(peak_drop_pct, 2),
        "supply_total_eok": round(supply_total_eok, 1),
        "days_held": days_held,
        "verdict": "HOLD_OVERNIGHT" if can_hold else "SELL_FORCE_CLOSE",
        "reason": (
            f"수급 지속 이월 (PnL {pnl_pct:+.2f}% / 수급 +{supply_total_eok:.1f}억)"
            if can_hold
            else "이월 조건 미달 (4 조건 중 미통과)"
        ),
    }
    return can_hold, details


def format_telegram(v: OwnerRuleVerdict, ticker: str, name: str = "") -> str:
    """텔레그램 알림 포맷."""
    if v.action == "HOLD":
        return ""  # HOLD는 알림 안 보냄

    emoji = {
        "SELL_STOP_LOSS": "🔴",
        "SELL_TRAILING": "🟡",
        "SELL_FORCE_CLOSE": "⏰",
    }.get(v.action, "⚪")

    return (
        f"{emoji} [사장님 룰] {name}({ticker}) {v.action}\n"
        f"  진입 {v.entry_price:,} → 현재 {v.current_price:,} ({v.pnl_pct:+.2f}%)\n"
        f"  peak {v.peak_price:,} 대비 {v.peak_drop_pct:+.2f}%\n"
        f"  사유: {v.reason}"
    )
