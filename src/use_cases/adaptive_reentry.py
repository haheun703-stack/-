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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


# === 임계 (.env 동적, 1주차는 보수적) ===
AUTO_REENTRY = os.getenv("ADAPTIVE_AUTO_REENTRY", "0") == "1"             # 1주차 알림만
REENTRY_MAX_AMOUNT = int(os.getenv("ADAPTIVE_REENTRY_MAX_AMOUNT", "1000000"))  # 1단위 100만
STEP5_MIN_STARS = int(os.getenv("ADAPTIVE_STEP5_MIN_STARS", "3"))         # ★★★ 이상


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
                dec.target_qty = REENTRY_MAX_AMOUNT // current_price
        except Exception as e:
            dec.error = f"현재가 fetch 오류: {e}"

    return dec


def execute_auto_reentry(broker, decision: ReentryDecision) -> dict:
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

    try:
        # 시장가 매수 (받침 확인된 종목 → 즉시 진입)
        if hasattr(broker, "buy_market"):
            order = broker.buy_market(decision.ticker, decision.target_qty)
            order_id = getattr(order, "order_id", "") or ""
        else:
            # mojito2 broker fallback
            res = broker.create_market_buy_order(decision.ticker, decision.target_qty)
            order_id = res.get("output", {}).get("ODNO", "") if res else ""

        decision.order_id = order_id
        return {
            "success": True,
            "order_id": order_id,
            "price": decision.target_price,
            "qty": decision.target_qty,
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
