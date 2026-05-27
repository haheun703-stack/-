"""적응형 포지션 매매법 MVP-2.5 v2 — Trailing Quick Profit (5/24 보강).

배경 (퐝가님 5/24 두 번째 지적):
  "원익이 그날에 15%까지 올라가도 우린 7%만 먹고 나온다는 거야?"

  v1 (+7% 고정 익절)의 한계:
    매수가 +7% 도달 → 즉시 매도 → 그 후 +15%까지 가도 못 먹음

  v2 Trailing 보강:
    1. 매수가 +7% 도달 → "QUICK_ARMED" 상태 (안 팔음, 추적 시작)
    2. 현재가가 trailing_peak를 갱신할 때마다 자동 갱신
    3. 현재가 ≤ trailing_peak × (1 - TRAILING_DROP_PCT/100) 도달 시 매도
    4. → 천장 다 먹고 진짜 꺾일 때만 매도

  예시:
    L1 22,500 매수 → 24,075 (+7%) 도달 → ARMED ★ 안 팔음
    25,000 → trailing_peak 갱신
    25,800 → trailing_peak 갱신 (현재 고점)
    25,284 (고점 -2% 꺾임) → 매도! +12.4% 확정

  → 단순 +7% 대비 +5.4%p 추가 수익 (15% 가는 케이스에서)
  → 진짜 빠르게 떨어지면 +7% 부근에서 매도 (안전망 그대로)

룰 요약:
  매수 → +7%까지 대기 → +7% 도달 = ARMED (trailing 시작)
  ARMED 상태:
    현재가 > trailing_peak → trailing_peak 갱신 (계속 추적)
    현재가 ≤ trailing_peak × 0.98 (-2%) → 매도 ★

핵심: "올라가는 동안 절대 안 팜, 꺾일 때만 팜" — MVP-1과 동일 철학.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from src.utils.trade_runtime_safety import assert_runtime_orders_allowed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


# === 임계 (.env 동적) ===
QUICK_PROFIT_ENABLED = os.getenv("ADAPTIVE_QUICK_PROFIT_ENABLED", "1") == "1"
QUICK_PROFIT_PCT = float(os.getenv("ADAPTIVE_QUICK_PROFIT_PCT", "7"))      # +7% trailing 진입
QUICK_PROFIT_RATIO = float(os.getenv("ADAPTIVE_QUICK_PROFIT_RATIO", "1.0"))  # 1.0=전량
# v2 신규: trailing 매도 임계
TRAILING_DROP_PCT = float(os.getenv("ADAPTIVE_TRAILING_DROP_PCT", "2"))    # 고점 -2% 꺾임


def _is_kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


def _fetch_current_price(broker, ticker: str) -> int:
    try:
        res = broker.fetch_price(ticker)
        output = res.get("output", {}) if res else {}
        return int(str(output.get("stck_prpr", 0)).replace(",", "") or 0)
    except Exception as e:
        logger.warning("price fetch %s 실패: %s", ticker, e)
        return 0


def execute_trailing_sell(broker, ticker: str, stage: dict, sell_price: int) -> dict:
    """Trailing 꺾임 시 시장가 매도 (즉시 체결 보장).

    Returns:
        {"success": bool, "order_id": str, "price": int, "qty": int, "error": str}
    """
    if not QUICK_PROFIT_ENABLED:
        return {"success": False, "error": "ADAPTIVE_QUICK_PROFIT_ENABLED=0"}

    actual_qty = int(stage.get("actual_qty", 0))
    if actual_qty <= 0:
        return {"success": False, "error": f"actual_qty 부적합 ({actual_qty})"}

    sell_qty = max(1, int(actual_qty * QUICK_PROFIT_RATIO))

    try:
        # 5/26 지정가 매도 (시장가 슬리피지 회피, ADAPTIVE_SELL_USE_LIMIT=1 기본).
        # trailing은 꺾이는 순간이지만 -0.3% 미세 슬리피지 지정가가 시장가보다 유리.
        # 시장가 fallback: 환경변수 ADAPTIVE_SELL_USE_LIMIT=0 또는 sell_limit 미지원.
        use_limit = os.getenv("ADAPTIVE_SELL_USE_LIMIT", "1") == "1"
        sell_slippage_pct = float(os.getenv("ADAPTIVE_SELL_LIMIT_SLIPPAGE_PCT", "0.3"))

        if use_limit and hasattr(broker, "sell_limit") and sell_price > 0:
            # 지정가 = 현재가 - slippage% (체결 우선)
            limit_price = int(sell_price * (1 - sell_slippage_pct / 100))
            order = broker.sell_limit(ticker, limit_price, sell_qty)
            order_id = getattr(order, "order_id", "") or ""
            logger.info(
                "[trailing] 지정가 매도 %s %d주 @ %d (현재 %d, slippage -%.1f%%)",
                ticker, sell_qty, limit_price, sell_price, sell_slippage_pct,
            )
        elif hasattr(broker, "sell_market"):
            order = broker.sell_market(ticker, sell_qty)
            order_id = getattr(order, "order_id", "") or ""
            limit_price = sell_price  # 시장가
        else:
            assert_runtime_orders_allowed()
            res = broker.create_market_sell_order(ticker, sell_qty)
            order_id = res.get("output", {}).get("ODNO", "") if res else ""
            limit_price = sell_price

        return {
            "success": True,
            "order_id": order_id,
            "price": limit_price,
            "qty": sell_qty,
        }
    except Exception as e:
        logger.error("trailing sell %s L%s 실패: %s", ticker, stage.get("level"), e)
        return {"success": False, "error": str(e)}


def check_quick_profit_triggers(broker) -> list[dict]:
    """모든 FILLED/QUICK_ARMED stage 순회 + Trailing Quick Profit 평가.

    v2 흐름:
      FILLED → 현재가 ≥ +7% target → QUICK_ARMED (trailing 시작)
      QUICK_ARMED:
        현재가 > trailing_peak → peak 갱신
        현재가 ≤ peak × 0.98 → 매도 (QUICK_SOLD)
    """
    triggers: list[dict] = []

    if not QUICK_PROFIT_ENABLED:
        logger.info("ADAPTIVE_QUICK_PROFIT_ENABLED=0 — MVP-2.5 비활성")
        return triggers

    # P0-4 (5/25): KILL_SWITCH는 매수만 차단, Trailing Quick Profit 매도는 계속 진행
    # 빠른 익절은 손실 방지 메커니즘이므로 KILL_SWITCH 무관 (꺾일 때 매도 보장)
    if _is_kill_switch_active():
        logger.warning(
            "KILL_SWITCH 활성 — Trailing Quick Profit 매도는 계속 진행 (P0-4 보장)"
        )
        # return 없음 — 매도 평가 계속

    from src.use_cases.adaptive_buy_queue import (
        _load_queues_raw,
        _save_queues_raw,
        STATUS_FILLED,
        STATUS_QUICK_ARMED,
        STATUS_QUICK_SOLD,
    )

    raw = _load_queues_raw()
    queues = raw.get("queues", {})
    modified = False
    now_iso = datetime.now().isoformat(timespec="seconds")

    for ticker, entry in queues.items():
        current_price = _fetch_current_price(broker, ticker)
        if current_price <= 0:
            continue

        for stage in entry.get("stages", []):
            status = stage.get("status")

            # === 케이스 1: FILLED → +7% 도달 시 ARMED 전환 ===
            if status == STATUS_FILLED:
                target = int(stage.get("quick_profit_target", 0))
                if target <= 0:
                    continue

                if current_price >= target:
                    # ARMED 전환 (안 팔음, trailing 시작)
                    stage["status"] = STATUS_QUICK_ARMED
                    stage["trailing_peak"] = current_price
                    stage["trailing_armed_at"] = now_iso
                    stage["trailing_peak_updated_at"] = now_iso
                    modified = True

                    actual_buy = int(stage.get("actual_price", 0))
                    pct = (current_price / actual_buy - 1) * 100 if actual_buy else 0
                    triggers.append({
                        "ticker": ticker,
                        "name": entry.get("name", ""),
                        "level": stage.get("level"),
                        "event": "ARMED",
                        "actual_buy_price": actual_buy,
                        "current_price": current_price,
                        "profit_pct_so_far": round(pct, 2),
                        "trailing_peak": current_price,
                    })

            # === 케이스 2: QUICK_ARMED → trailing 업데이트 또는 매도 ===
            elif status == STATUS_QUICK_ARMED:
                trailing_peak = int(stage.get("trailing_peak", 0))

                # 고점 갱신
                if current_price > trailing_peak:
                    stage["trailing_peak"] = current_price
                    stage["trailing_peak_updated_at"] = now_iso
                    modified = True
                    # 알림 발송 X (너무 시끄러움) — 매도 시에만 알림

                else:
                    # 꺾임 체크: 현재가 ≤ trailing_peak × (1 - DROP/100)
                    sell_threshold = trailing_peak * (1 - TRAILING_DROP_PCT / 100)

                    if current_price <= sell_threshold:
                        # 매도 실행
                        sell_result = execute_trailing_sell(
                            broker, ticker, stage, current_price
                        )

                        if sell_result["success"]:
                            actual_buy = int(stage.get("actual_price", 0))
                            sold = sell_result.get("price", current_price)
                            profit_pct = (
                                (sold / actual_buy - 1) * 100 if actual_buy else 0
                            )
                            peak_pct = (
                                (trailing_peak / actual_buy - 1) * 100 if actual_buy else 0
                            )

                            stage["status"] = STATUS_QUICK_SOLD
                            stage["quick_profit_order_id"] = sell_result.get("order_id", "")
                            stage["quick_profit_sold_at"] = now_iso
                            stage["quick_profit_sold_price"] = sold
                            modified = True

                            triggers.append({
                                "ticker": ticker,
                                "name": entry.get("name", ""),
                                "level": stage.get("level"),
                                "event": "SOLD",
                                "actual_buy_price": actual_buy,
                                "trailing_peak": trailing_peak,
                                "peak_pct": round(peak_pct, 2),
                                "sold_price": sold,
                                "profit_pct": round(profit_pct, 2),
                                "qty": sell_result.get("qty", 0),
                                "order_id": sell_result.get("order_id", ""),
                                "current_price": current_price,
                            })

    if modified:
        _save_queues_raw(raw)

    return triggers


def format_quick_profit_for_telegram(trigger: dict) -> str:
    """텔레그램 알림 — ARMED / SOLD 이벤트 구분."""
    name = trigger.get("name") or trigger.get("ticker", "")
    level = trigger.get("level", "?")
    event = trigger.get("event", "UNKNOWN")

    if event == "ARMED":
        buy = int(trigger.get("actual_buy_price", 0))
        cur = int(trigger.get("current_price", 0))
        pct = trigger.get("profit_pct_so_far", 0.0)
        return (
            f"🎯 빠른 익절 L{level} 추적 시작 [{name}]\n"
            f"  매수: {buy:,} → 현재: {cur:,} ({pct:+.2f}%)\n"
            f"  ▶ Trailing 모드 진입 — 천장 다 먹고 -{TRAILING_DROP_PCT:.0f}% 꺾일 때 매도"
        )

    elif event == "SOLD":
        buy = int(trigger.get("actual_buy_price", 0))
        peak = int(trigger.get("trailing_peak", 0))
        sold = int(trigger.get("sold_price", 0))
        profit_pct = trigger.get("profit_pct", 0.0)
        peak_pct = trigger.get("peak_pct", 0.0)
        qty = int(trigger.get("qty", 0))
        profit_amount = (sold - buy) * qty
        gap = peak_pct - profit_pct

        return (
            f"💰 빠른 익절 L{level} 체결 [{name}]\n"
            f"  매수: {buy:,} → 고점: {peak:,} ({peak_pct:+.2f}%) → 매도: {sold:,} ({profit_pct:+.2f}%)\n"
            f"  Trailing 꺾임 -{gap:.2f}%p, 수량 {qty}주, 수익 {profit_amount:,}원\n"
            f"  주문 ID: {trigger.get('order_id', '')}"
        )

    return f"❓ 빠른 익절 L{level} 알 수 없는 이벤트 [{name}] {event}"


def reset_quick_sold_for_reentry(ticker: str) -> bool:
    """QUICK_SOLD stage → PENDING 복원 (재진입용).

    v2: trailing 필드도 함께 초기화.
    """
    from src.use_cases.adaptive_buy_queue import (
        _load_queues_raw,
        _save_queues_raw,
        STATUS_PENDING,
        STATUS_QUICK_SOLD,
    )

    raw = _load_queues_raw()
    queues = raw.get("queues", {})
    if ticker not in queues:
        return False

    modified = False
    for stage in queues[ticker].get("stages", []):
        if stage.get("status") == STATUS_QUICK_SOLD:
            stage["status"] = STATUS_PENDING
            stage["actual_price"] = 0
            stage["actual_qty"] = 0
            stage["quick_profit_target"] = 0
            stage["order_id"] = None
            stage["quick_profit_order_id"] = None
            stage["quick_profit_sold_at"] = None
            stage["quick_profit_sold_price"] = 0
            # v2 trailing 필드 초기화
            stage["trailing_peak"] = 0
            stage["trailing_armed_at"] = None
            stage["trailing_peak_updated_at"] = None
            modified = True

    if modified:
        _save_queues_raw(raw)

    return modified


# v1 호환성 (단계별 +7% 직접 매도 — 보존)
def execute_quick_sell(broker, ticker: str, stage: dict) -> dict:
    """v1 직접 매도 (legacy 호환용)."""
    target_price = int(stage.get("quick_profit_target", 0))
    if target_price <= 0:
        return {"success": False, "error": f"quick_profit_target 부적합 ({target_price})"}
    return execute_trailing_sell(broker, ticker, stage, target_price)
