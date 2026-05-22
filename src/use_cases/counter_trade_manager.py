"""역매수 포지션 관리 — 개미털기 판별 + 피보나치 단계별 추매 통합 (5/22 퐝가님 인사이트).

배경:
  보유 종목 D+1~D+N 마이너스 진입 시:
  1. shake_out_detector → 개미털기 vs 진짜 문제 판별
  2. fibonacci_buyer → 매수 단계 결정 (2/3/4차)
  3. 개미털기 + 매수 단계 매칭 → 추매 시그널
  4. 진짜 문제 → 손절 시그널

워밍업 모드 (5/26~):
  - 신호 발동 + 텔레그램 알림만
  - 실제 추매 매수 X (사용자 검증 후 5/30+ 활성화)
  - 종목당 1주 한도 유지

정식 모드 (5/30+):
  - 추매 + 손절 자동 실행
  - 자본 분할: 1차 30% + 2차 30% + 3차 40%

owner_rule_monitor와 관계:
  - owner_rule: -3% 손절 / Trailing -3% (단순)
  - counter_trade: 개미털기 판별 + 피보나치 추매 (정밀)
  - 워밍업 단계: counter_trade 신호만, owner_rule 그대로 가동
  - 정식 단계: counter_trade 추매 활성화, owner_rule 임계 완화 결단 필요

데이터 저장:
  data/counter_trade_state.json — 종목별 매수 단계 + 누적 자본

사용:
  from src.use_cases.counter_trade_manager import evaluate_position
  signal = evaluate_position(broker, ticker, entry_price, current_price, pick_record)
  if signal["recommend"] == "REBUY":
      텔레그램 알림 + (정식 모드 시) 추매 주문
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.use_cases.shake_out_detector import detect_shake_out
from src.use_cases.fibonacci_buyer import calculate_fib_buy_step, get_fib_levels

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_PATH = PROJECT_ROOT / "data" / "counter_trade_state.json"

# 워밍업 모드 — 신호만 발동 (실제 매수 X)
WARMUP_MODE = os.getenv("COUNTER_TRADE_WARMUP", "1") == "1"

# 추천 종류
RECOMMENDATIONS = {
    "REBUY": "역매수 진행 (개미털기 확정 + 피보나치 단계 도달)",
    "WATCH": "관망 (개미털기 가능성 있으나 매수 단계 미도달)",
    "HOLD": "보유 유지 (정상 영역)",
    "CUT_LOSS": "손절 시그널 (진짜 문제 추정)",
    "STOP_LOSS": "강제 손절 (-20% 이하)",
    "BLOCKED": "추매 차단 (이미 해당 단계 매수 완료)",
}


def _load_state() -> dict:
    """역매수 상태 로드 (종목별 매수 단계)."""
    if not STATE_PATH.exists():
        return {"positions": {}, "updated_at": None}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("counter_trade_state.json 로드 실패: %s", e)
        return {"positions": {}, "updated_at": None}


def _save_state(state: dict) -> None:
    """역매수 상태 저장."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        STATE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("counter_trade_state.json 저장 실패: %s", e)


def get_position_steps(ticker: str) -> list[int]:
    """종목별 이미 매수한 피보나치 단계 리스트."""
    state = _load_state()
    pos = state.get("positions", {}).get(ticker, {})
    return pos.get("steps", [1])  # 1차는 항상 (보유 종목)


def record_position_step(
    ticker: str,
    step: int,
    buy_price: float,
    qty: int,
) -> None:
    """피보나치 단계 매수 기록."""
    state = _load_state()
    positions = state.setdefault("positions", {})
    pos = positions.setdefault(ticker, {"steps": [1], "buys": []})

    if step not in pos["steps"]:
        pos["steps"].append(step)
        pos["steps"].sort()

    pos["buys"].append({
        "step": step,
        "buy_price": round(buy_price, 2),
        "qty": qty,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    _save_state(state)
    logger.info(
        "[counter_trade] %s 단계 %d 매수 기록 (price=%s, qty=%d)",
        ticker, step, buy_price, qty,
    )


def evaluate_position(
    ticker: str,
    name: str,
    entry_price: float,
    current_price: float,
    foreign_5d: float = 0,
    inst_5d: float = 0,
    bid_ask_ratio: Optional[float] = None,
    vol_ratio: Optional[float] = None,
    volume_power: Optional[float] = None,
    initial_capital: float = 1_000_000,
) -> dict[str, Any]:
    """보유 종목 평가 — 개미털기 + 피보나치 통합 판단.

    Returns:
        {
            "ticker": str,
            "name": str,
            "recommend": "REBUY" | "WATCH" | "HOLD" | "CUT_LOSS" | "STOP_LOSS" | "BLOCKED",
            "step": int,                    # 추매 단계 (0 = 없음)
            "decline_pct": float,
            "shake_out": dict,              # detect_shake_out 결과
            "fib": dict,                     # calculate_fib_buy_step 결과
            "already_bought_steps": list,
            "warmup_mode": bool,
            "action_text": str,              # 사용자/텔레그램 표시용
            "reason": str,
        }
    """
    already_bought_steps = get_position_steps(ticker)

    # ── 1. 개미털기 판별 ──
    shake = detect_shake_out(
        foreign_5d=foreign_5d,
        inst_5d=inst_5d,
        current_price=current_price,
        entry_price=entry_price,
        bid_ask_ratio=bid_ask_ratio,
        vol_ratio=vol_ratio,
        volume_power=volume_power,
    )

    # ── 2. 피보나치 단계 판정 ──
    fib = calculate_fib_buy_step(
        entry_price=entry_price,
        current_price=current_price,
        initial_capital=initial_capital,
        already_bought_steps=already_bought_steps,
        warmup_mode=WARMUP_MODE,
    )

    decline_pct = fib["decline_pct"]
    fib_action = fib["action"]
    fib_step = fib["step"]

    # ── 3. 통합 판단 ──
    # 손절 (피보나치 -20% 또는 진짜 문제)
    if fib_action == "STOP_LOSS":
        recommend = "STOP_LOSS"
        reason = f"피보나치 손절 ({decline_pct:.2f}% ≤ -20%): {fib['reason']}"
    elif shake["is_problem"]:
        recommend = "CUT_LOSS"
        reason = f"진짜 문제 시그널: {shake['reason']}"
    # 추매 (개미털기 확정 + 매수 단계 도달)
    elif shake["is_shakeout"] and fib_action.startswith("BUY_STEP"):
        recommend = "REBUY"
        reason = (
            f"★역매수 시그널: 개미털기 (점수 {shake['score']:+d}) + "
            f"{fib_action} ({decline_pct:.2f}%)"
        )
    # 추매 차단 (이미 매수)
    elif fib_action == "REBUY_BLOCKED":
        recommend = "BLOCKED"
        reason = fib["reason"]
    # 관망 (불확실)
    elif fib_action.startswith("BUY_STEP") or fib_action == "WATCH":
        recommend = "WATCH"
        reason = (
            f"관망: 매수 단계 도달했으나 개미털기 미확정 "
            f"(shake 점수 {shake['score']:+d}, n_data={shake['n_data_available']})"
        )
    else:
        recommend = "HOLD"
        reason = f"정상 영역 ({decline_pct:+.2f}%)"

    # ── 4. 액션 텍스트 (텔레그램용) ──
    action_text = _format_action(
        ticker, name, recommend, fib, shake, current_price, entry_price,
    )

    return {
        "ticker": ticker,
        "name": name,
        "recommend": recommend,
        "step": fib_step if recommend == "REBUY" else 0,
        "decline_pct": decline_pct,
        "shake_out": shake,
        "fib": fib,
        "already_bought_steps": already_bought_steps,
        "warmup_mode": WARMUP_MODE,
        "action_text": action_text,
        "reason": reason,
    }


def _format_action(
    ticker: str,
    name: str,
    recommend: str,
    fib: dict,
    shake: dict,
    current_price: float,
    entry_price: float,
) -> str:
    """텔레그램 알림용 액션 텍스트 포맷."""
    icon = {
        "REBUY": "🔵 역매수",
        "WATCH": "👀 관망",
        "HOLD": "➖ 보유",
        "CUT_LOSS": "🟡 손절검토",
        "STOP_LOSS": "🔴 손절",
        "BLOCKED": "🔒 중복차단",
    }.get(recommend, "?")

    lines = [
        f"{icon} {name}({ticker})",
        f"  진입 {entry_price:,.0f}원 → 현재 {current_price:,.0f}원 ({fib['decline_pct']:+.2f}%)",
    ]

    if recommend == "REBUY":
        lines.append(f"  추매 단계 {fib['step']}: {fib['next_qty_estimate']}주 (자본 {fib['next_capital_pct']*100:.0f}%)")
        lines.append(f"  개미털기 점수 {shake['score']:+d} (외인/기관 매수 유지)")
    elif recommend in ("CUT_LOSS", "STOP_LOSS"):
        lines.append(f"  진짜 문제 점수 {shake['score']:+d}")
        if recommend == "STOP_LOSS":
            lines.append(f"  강제 손절 ({fib['decline_pct']:.2f}% ≤ -20%)")
    elif recommend == "WATCH":
        lines.append(f"  shake 점수 {shake['score']:+d} (확실하지 않음)")

    return "\n".join(lines)


def evaluate_all_positions(
    broker,
    positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """모든 보유 종목 일괄 평가.

    Args:
        broker: KIS broker
        positions: [{"ticker", "name", "entry_price", "current_price",
                     "foreign_5d", "inst_5d", ...}, ...]

    Returns:
        각 종목별 evaluate_position 결과 리스트
    """
    results = []
    for pos in positions:
        try:
            r = evaluate_position(
                ticker=pos["ticker"],
                name=pos.get("name", pos["ticker"]),
                entry_price=float(pos.get("entry_price", 0)),
                current_price=float(pos.get("current_price", 0)),
                foreign_5d=float(pos.get("foreign_5d", 0)),
                inst_5d=float(pos.get("inst_5d", 0)),
                bid_ask_ratio=pos.get("bid_ask_ratio"),
                vol_ratio=pos.get("vol_ratio"),
                volume_power=pos.get("volume_power"),
                initial_capital=float(pos.get("initial_capital", 1_000_000)),
            )
            results.append(r)
        except Exception as e:
            logger.warning("evaluate_position 실패 %s: %s", pos.get("ticker"), e)
    return results
