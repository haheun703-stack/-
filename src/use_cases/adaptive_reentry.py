"""적응형 포지션 매매법 MVP-4 — 자동 재진입 (받침 + STEP 5 + 자비스 9 안전선).

배경 (퐝가님 5/23 흐름):
  6단계 흐름의 [6단계 재진입]:
    "받침 형성 확인 후 다시 사고 → 끝까지 오르고 -3%면 팔고 (MVP-1으로 회귀)"

  단순 받침 ≠ 자동 매수. 3중 검증 ALL 통과해야 진짜 재진입.

  3중 검증 게이트:
    1. ★ MVP-3 받침 시그널 trigger=True
    2. ★ STEP 5 ★★★ 이상 (soubujang_pool 통과 종목)
    3. ★ 자비스 9 안전선 통과 (KILL_SWITCH / 매크로 가드 / 시간대 등)

  ALL → 자동 매수 (1주차는 알림만, AUTO_REENTRY=0)
  → 매수 성공 → 다음 MVP-1 사이클 회귀 (6단계 완전 자동화)

MVP-4 기능:
  1. evaluate_reentry: 종목 단일 평가 (3중 검증)
  2. execute_auto_reentry: 자동 매수 실행 (AUTO_REENTRY=1)
  3. scan_pool_for_reentry: 후보 풀 일괄 평가
  4. format_reentry_for_telegram

매수 사이즈 결정:
  - 재진입은 분할매수 큐의 L1 단위(가용 30% × cap)에 맞춤
  - 이미 큐에 등록된 종목 = 큐 매수로 위임 (MVP-2가 처리, 중복 방지)
  - 신규 종목만 MVP-4 단독 매수

5/17 자기반성 #1 적용: import + 함수 호출 + 통합 흐름 검증.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

from src.use_cases.gate_wiring import gate_check
from src.utils.trade_runtime_safety import assert_runtime_orders_allowed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


# === 임계 (.env 동적, 1주차는 보수적) ===
AUTO_REENTRY = os.getenv("ADAPTIVE_AUTO_REENTRY", "0") == "1"             # 1주차 알림만
REENTRY_MAX_AMOUNT = int(os.getenv("ADAPTIVE_REENTRY_MAX_AMOUNT", "1000000"))  # 1단위 100만
REENTRY_MAX_QTY = int(os.getenv("ADAPTIVE_REENTRY_MAX_QTY", "0"))         # 1주차 1주 cap (0=무제한)
STEP5_MIN_STARS = int(os.getenv("ADAPTIVE_STEP5_MIN_STARS", "3"))         # ★★★ 이상
# 3종목 한도 (1주차 안전). MVP-4 매수 실행 직전 재검사용. 미설정 시 기본 3.
MAX_POSITIONS = int(os.getenv("ADAPTIVE_MAX_POSITIONS", "3"))


@dataclass
class ReentryDecision:
    """재진입 평가."""

    ticker: str
    name: str = ""
    # 게이트별 결과
    support_pass: bool = False
    support_reasons: list[str] = field(default_factory=list)
    step5_pass: bool = False
    step5_stars: int = 0
    step5_upside: float = 0.0
    jarvis_pass: bool = False
    jarvis_failed_checks: list[str] = field(default_factory=list)
    # 종합
    trigger: bool = False
    already_in_queue: bool = False           # MVP-2 큐에 있으면 위임
    auto_reentry_eligible: bool = False
    # 매수 정보
    target_price: int = 0
    target_qty: int = 0
    target_amount: int = 0
    # 실행 결과
    order_id: Optional[str] = None
    error: Optional[str] = None


def _is_kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


def _is_in_buy_queue(ticker: str) -> bool:
    """MVP-2 큐에 종목이 활성 상태로 있는지."""
    try:
        from src.use_cases.adaptive_buy_queue import (
            get_queue_status,
            STATUS_PENDING,
            STATUS_TRIGGERED,
        )

        entry = get_queue_status(ticker)
        if not entry:
            return False
        for stage in entry.get("stages", []):
            if stage.get("status") in (STATUS_PENDING, STATUS_TRIGGERED):
                return True
        return False
    except Exception:
        return False


def _count_active_positions(exclude_ticker: Optional[str] = None) -> int:
    """현재 활성 포지션 수 (MVP-2 큐 + MVP-4 신규 매수 직전 검사용).

    카운트 소스 선택: **MVP-2 큐 PENDING/TRIGGERED/FILLED 종목 수**.
    이유:
      - MVP-2 큐가 6단계 흐름의 단일 진실 원천 (MVP-1 매도 후 자동 등록 + 분할매수 진행)
      - 실제 잔고 조회는 broker 의존성 + 추가 API 호출 → 매 cron마다 비용 ↑
      - PENDING도 카운트해야 "큐 등록 완료 + 아직 도달 안 한" 종목까지 한도 보호
      - MVP-4 재진입은 별도 ticker에 한해서만 발동 (큐에 이미 있으면 _is_in_buy_queue로 위임 처리)
      - 잔고 조회는 추후 보강 가능 — 1주차는 큐 기반으로 보수적 운영

    Args:
        exclude_ticker: 카운트 제외 종목 (재진입 평가 대상 본인 — 큐에 있어도 신규 매수 슬롯으로 간주 X)

    Returns:
        활성 포지션 수
    """
    try:
        from src.use_cases.adaptive_buy_queue import (
            load_queues,
            STATUS_PENDING,
            STATUS_TRIGGERED,
            STATUS_FILLED,
            STATUS_QUICK_ARMED,
        )

        active_statuses = (STATUS_PENDING, STATUS_TRIGGERED, STATUS_FILLED, STATUS_QUICK_ARMED)
        queues = load_queues()
        count = 0
        for ticker, entry in queues.items():
            if exclude_ticker and ticker == exclude_ticker:
                continue
            for stage in entry.get("stages", []):
                if stage.get("status") in active_statuses:
                    count += 1
                    break
        return count
    except Exception as e:
        # 카운트 실패 시 안전하게 한도값 반환 → 매수 차단
        logger.warning("활성 포지션 카운트 실패: %s — 한도 도달로 간주", e)
        return MAX_POSITIONS


def evaluate_reentry(
    broker,
    ticker: str,
    name: str = "",
    step5_pool: Optional[dict] = None,
    jarvis_safety_check: Optional[Callable[[str], dict]] = None,
) -> ReentryDecision:
    """3중 검증 게이트 평가.

    Args:
        broker: KIS broker
        ticker: 종목 코드
        name: 종목명 (알림용)
        step5_pool: STEP 5 후보 풀 dict {ticker: {stars, upside, ...}}
        jarvis_safety_check: 자비스 9 안전선 함수 (ticker → {"pass": bool, "failed": [...]}).
                              None이면 KILL_SWITCH만 검사.

    Returns:
        ReentryDecision
    """
    dec = ReentryDecision(ticker=ticker, name=name)

    # KILL_SWITCH 최우선
    if _is_kill_switch_active():
        dec.jarvis_failed_checks.append("KILL_SWITCH 발동")
        return dec

    # === 게이트 0: 중복 진입 방지 (MVP-2 큐 활성 여부) ===
    if _is_in_buy_queue(ticker):
        dec.already_in_queue = True
        # 큐 위임 — 재진입은 큐가 도달 시 자동 처리
        return dec

    # === 게이트 1: MVP-3 받침 패턴 ===
    try:
        from src.use_cases.support_pattern_detector import detect_support_pattern

        support_sig = detect_support_pattern(broker, ticker)
        dec.support_pass = bool(support_sig.trigger)
        dec.support_reasons = (
            support_sig.reasons_pass if dec.support_pass else support_sig.reasons_fail
        )
    except Exception as e:
        dec.support_pass = False
        dec.support_reasons = [f"받침 평가 오류: {e}"]

    # === 게이트 2: STEP 5 ★★★ 이상 ===
    if step5_pool and ticker in step5_pool:
        info = step5_pool[ticker]
        stars = int(info.get("stars", 0))
        upside = float(info.get("upside", 0.0))
        dec.step5_stars = stars
        dec.step5_upside = upside
        dec.step5_pass = (stars >= STEP5_MIN_STARS)
    else:
        # STEP 5 풀에 없으면 보수적으로 차단
        dec.step5_pass = False

    # === 게이트 3: 자비스 9 안전선 ===
    if jarvis_safety_check is not None:
        try:
            jcheck = jarvis_safety_check(ticker)
            dec.jarvis_pass = bool(jcheck.get("pass", False))
            dec.jarvis_failed_checks = list(jcheck.get("failed", []))
        except Exception as e:
            dec.jarvis_pass = False
            dec.jarvis_failed_checks = [f"자비스 체크 오류: {e}"]
    else:
        # 자비스 함수 없으면 KILL_SWITCH만 통과한 것으로 간주
        dec.jarvis_pass = True

    # === 종합: 3 게이트 ALL 통과 ===
    dec.trigger = dec.support_pass and dec.step5_pass and dec.jarvis_pass

    if dec.trigger and AUTO_REENTRY:
        dec.auto_reentry_eligible = True

    # 매수 금액 산정 (현재가 기준)
    if dec.trigger:
        try:
            price_res = broker.fetch_price(ticker)
            current_price = int(
                str(price_res.get("output", {}).get("stck_prpr", 0)).replace(",", "") or 0
            )
            if current_price > 0:
                dec.target_price = current_price
                dec.target_amount = REENTRY_MAX_AMOUNT
                qty_calc = REENTRY_MAX_AMOUNT // current_price
                # 1주차 안전: REENTRY_MAX_QTY > 0이면 수량 cap
                if REENTRY_MAX_QTY > 0:
                    qty_calc = min(qty_calc, REENTRY_MAX_QTY)
                dec.target_qty = qty_calc
        except Exception as e:
            dec.error = f"현재가 fetch 오류: {e}"

    return dec


def execute_auto_reentry(
    broker, decision: ReentryDecision,
    *, mode: str | None = None, executor_bot: str | None = None,
) -> dict:
    """자동 재진입 — 5/28 코덱스 결정문: mode/executor_bot 명시 강제 가능.

    backward compat: 둘 다 None → 기존 _guard 9중만.
    명시 시 → L10 (order_intents_gate) 강제.
    """
    """자동 재진입 매수 (AUTO_REENTRY=1일 때만).

    Returns:
        {"success": bool, "order_id": str, "price": int, "qty": int, "error": str}
    """
    if decision.already_in_queue:
        return {"success": False, "error": "이미 MVP-2 큐 활성 — 큐 위임"}

    if not decision.trigger:
        failed = []
        if not decision.support_pass:
            failed.append("받침")
        if not decision.step5_pass:
            failed.append("STEP5")
        if not decision.jarvis_pass:
            failed.append("자비스")
        return {"success": False, "error": f"게이트 미통과: {','.join(failed)}"}

    if not AUTO_REENTRY:
        return {"success": False, "error": "ADAPTIVE_AUTO_REENTRY=0 — 알림만"}

    if decision.target_qty <= 0 or decision.target_price <= 0:
        return {
            "success": False,
            "error": f"target qty/price 부적합 ({decision.target_qty}/{decision.target_price})",
        }

    # ★ P0-3 (5/24): 매수 트리거 직전 한도 재검사.
    # 큐 등록 시점과 매수 시점 사이에 다른 종목이 추가 매수되어
    # 한도 초과될 위험을 차단. 본인(decision.ticker)은 제외하고 카운트.
    active_count = _count_active_positions(exclude_ticker=decision.ticker)
    if active_count >= MAX_POSITIONS:
        msg = (
            f"{MAX_POSITIONS}종목 한도 도달 (현재 {active_count}) — "
            f"매수 SKIP [{decision.ticker}]"
        )
        logger.warning("[MVP-4] %s", msg)
        decision.error = msg
        return {"success": False, "error": msg, "max_positions_blocked": True}

    try:
        # 시장가 매수 (받침 확인된 종목 → 즉시 진입)
        if hasattr(broker, "buy_market"):
            adapter_kwargs = {}
            if mode is not None or executor_bot is not None:
                adapter_kwargs = {"mode": mode, "executor_bot": executor_bot}
            # ★RISK_ENGINE C-ii-b: 게이트 통행증(시장가라 사이징 단가=target_price). REAL만 차단.
            proceed, risk_gate, qty = gate_check(
                broker, decision.ticker, decision.target_price, decision.target_qty,
            )
            if not proceed:
                return {"success": False, "error": "risk_gate_reject"}
            order = broker.buy_market(decision.ticker, qty, gate_result=risk_gate, **adapter_kwargs)
            order_id = getattr(order, "order_id", "") or ""
        else:
            raise RuntimeError(
                "[P0-D] raw mojito broker 호출 차단 — KisOrderAdapter 인스턴스 필수. "
                "호출자가 KisOrderAdapter를 broker 인자로 전달해야 함."
            )

        decision.order_id = order_id
        return {
            "success": True,
            "order_id": order_id,
            "price": decision.target_price,
            "qty": qty,
        }
    except Exception as e:
        logger.error("auto reentry %s 실패: %s", decision.ticker, e)
        decision.error = str(e)
        return {"success": False, "error": str(e)}


def scan_pool_for_reentry(
    broker,
    candidates: list[tuple[str, str]],
    step5_pool: Optional[dict] = None,
    jarvis_safety_check: Optional[Callable[[str], dict]] = None,
) -> list[ReentryDecision]:
    """후보 풀 일괄 재진입 평가.

    Args:
        candidates: [(ticker, name), ...]
    """
    decisions: list[ReentryDecision] = []
    for ticker, name in candidates:
        dec = evaluate_reentry(broker, ticker, name, step5_pool, jarvis_safety_check)
        decisions.append(dec)
    # trigger 우선
    decisions.sort(key=lambda d: (not d.trigger, d.ticker))
    return decisions


def format_reentry_for_telegram(decision: ReentryDecision) -> str:
    """텔레그램 알림용 포맷."""
    name = decision.name or decision.ticker

    if decision.already_in_queue:
        return f"⏭️ 재진입 평가 스킵 [{name}] — MVP-2 큐에 이미 있음 (위임)"

    head = f"🔁 적응형 재진입 평가 [{name}]"

    gates = [
        f"  받침: {'✓' if decision.support_pass else '✗'}",
        f"  STEP5: {'✓' if decision.step5_pass else '✗'} "
        f"({'★' * decision.step5_stars}, 업사이드 {decision.step5_upside:.2f}x)",
        f"  자비스: {'✓' if decision.jarvis_pass else '✗'} "
        f"({len(decision.jarvis_failed_checks)}건 실패)",
    ]

    if decision.trigger:
        body = [head] + gates
        body.append(
            f"  ➜ 매수 {decision.target_qty}주 @ {decision.target_price:,} "
            f"({decision.target_amount:,}원)"
        )
        if decision.order_id:
            body.append(f"  ✅ 주문 ID: {decision.order_id}")
        elif not decision.auto_reentry_eligible:
            body.append("  💤 ADAPTIVE_AUTO_REENTRY=0 — 알림만")
        return "\n".join(body)
    else:
        failed = []
        if not decision.support_pass:
            failed.append("받침")
        if not decision.step5_pass:
            failed.append("STEP5")
        if not decision.jarvis_pass:
            failed.append("자비스")
        return f"{head} 미트리거 — 실패: {','.join(failed)}"
