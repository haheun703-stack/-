"""
매수/매도 실행 규칙
========================
VWAP/전일고가/15분고가 레벨 기반 매수·매도 판정.

대장주: A형(재돌파) / B형(눌림)
2차 연동주: 대장주 강세 유지 시만 진입
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSignal:
    """실행 신호."""
    ticker: str
    name: str
    role: str                  # "kr_leader" or "kr_secondary"
    action: str                # BUY_A / BUY_B / STANDBY / WATCH_BREAKOUT / BLOCKED
    buy_price: float = 0       # 매수 기준가
    stop_price: float = 0      # 손절가
    tp1_price: float = 0       # 1차 익절가
    tp2_price: float = 0       # 2차 익절가
    reason: str = ""
    conditions: dict = field(default_factory=dict)


def generate_kr_leader_signals(
    sector_config: dict,
    phase: int,
    common_rules: dict,
) -> list[ExecutionSignal]:
    """한국 대장주 실행 신호 생성.

    Phase 3+ 에서만 실행 신호 생성.
    Phase 2: WATCH_BREAKOUT (돌파 대기)
    Phase 0~1: STANDBY
    """
    signals = []
    kr_leaders = sector_config.get("kr_leaders", [])
    tp_range = sector_config.get("take_profit_leader", [4, 6])

    for leader in kr_leaders:
        ticker = leader["ticker"]
        name = leader["name"]

        if phase >= 3:
            # Phase 3+: 매수 준비 완료
            sig = ExecutionSignal(
                ticker=ticker,
                name=name,
                role="kr_leader",
                action="BUY_A",  # A형 재돌파
                reason="전일 고가 재돌파 시 매수",
                conditions={
                    "buy_type": "prev_high_breakout",
                    "buy_multiplier": 1.003,  # 전일고가 × 1.003
                    "stop_method": common_rules.get("stop_method", "vwap_or_pullback"),
                    "take_profit_1_pct": tp_range[0],
                    "take_profit_2_pct": tp_range[1] if len(tp_range) > 1 else tp_range[0] + 2,
                    "gap_up_ban_pct": common_rules.get("gap_up_chase_ban_pct", 7.0),
                    "first_5min_ban": common_rules.get("first_5min_ban", True),
                },
            )
        elif phase == 2:
            sig = ExecutionSignal(
                ticker=ticker,
                name=name,
                role="kr_leader",
                action="WATCH_BREAKOUT",
                reason="US 확인 완료, KR 돌파 대기",
            )
        else:
            sig = ExecutionSignal(
                ticker=ticker,
                name=name,
                role="kr_leader",
                action="STANDBY",
                reason="조건 미충족",
            )
        signals.append(sig)

    return signals


def generate_kr_secondary_signals(
    sector_config: dict,
    phase: int,
    kr_leaders_active: bool,
    common_rules: dict,
) -> list[ExecutionSignal]:
    """한국 2차 연동주 실행 신호 생성.

    대장주가 활성 상태(Phase 3+)일 때만 진입 가능.
    """
    signals = []
    kr_secondaries = sector_config.get("kr_secondaries", [])
    tp_range = sector_config.get("take_profit_secondary", [5, 8])

    for sec in kr_secondaries:
        ticker = sec["ticker"]
        name = sec["name"]

        if phase >= 3 and kr_leaders_active:
            # 특수 진입 규칙 체크
            entry_rule = sector_config.get("entry_rule", "")

            if entry_rule == "2day_above_prev_high":
                sig = ExecutionSignal(
                    ticker=ticker,
                    name=name,
                    role="kr_secondary",
                    action="BUY_A",
                    reason="전일고가 위 2일 연속 안착 후 진입",
                    conditions={
                        "buy_type": "2day_above_prev_high",
                        "take_profit_1_pct": tp_range[0],
                        "take_profit_2_pct": tp_range[1] if len(tp_range) > 1 else tp_range[0] + 3,
                    },
                )
            elif entry_rule == "weekly_pullback_swing":
                sig = ExecutionSignal(
                    ticker=ticker,
                    name=name,
                    role="kr_secondary",
                    action="BUY_B",
                    reason="주봉 눌림 스윙 (전일저점 미이탈 + 전일고가 돌파)",
                    conditions={
                        "buy_type": "weekly_pullback_swing",
                        "take_profit_1_pct": tp_range[0],
                        "take_profit_2_pct": tp_range[1] if len(tp_range) > 1 else tp_range[0] + 3,
                    },
                )
            else:
                sig = ExecutionSignal(
                    ticker=ticker,
                    name=name,
                    role="kr_secondary",
                    action="BUY_A",
                    reason="대장주 강세 확인 → 전일 고가 돌파 시 후행 진입",
                    conditions={
                        "buy_type": "prev_high_breakout",
                        "stop_method": "breakout_candle_low",
                        "take_profit_1_pct": tp_range[0],
                        "take_profit_2_pct": tp_range[1] if len(tp_range) > 1 else tp_range[0] + 3,
                        "leader_must_be_strong": True,
                    },
                )
        elif phase >= 2:
            sig = ExecutionSignal(
                ticker=ticker,
                name=name,
                role="kr_secondary",
                action="STANDBY",
                reason="대장주 미확인 → 후행 대기",
            )
        else:
            sig = ExecutionSignal(
                ticker=ticker,
                name=name,
                role="kr_secondary",
                action="STANDBY",
                reason="조건 미충족",
            )
        signals.append(sig)

    return signals


def format_execution_summary(signals: list[ExecutionSignal]) -> dict:
    """실행 신호를 요약 딕셔너리로 변환."""
    result = {}
    for sig in signals:
        result[sig.name] = {
            "ticker": sig.ticker,
            "role": sig.role,
            "action": sig.action,
            "reason": sig.reason,
        }
        if sig.conditions:
            result[sig.name]["conditions"] = sig.conditions
    return result
