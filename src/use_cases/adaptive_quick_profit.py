"""적응형 포지션 매매법 MVP-2.5 — 단계별 독립 빠른 익절 (+7%).

배경 (퐝가님 5/24 결단):
  "1개 종목 로테이션 3일이야. 적응형은 너무 오래 들고 있는 거 아닌가?"
  "단기로 벌어서 중장기에 시드를 계속 늘려가는 거지... 복리 타입으로 제격이다"

  순수 적응형(MVP-1 천장 -5% 매도)만으로는 평균 1~2개월 보유 → 회전 느림.
  보강: L1/L2/L3 각 단계 매수 후 +7% 도달 시 즉시 익절 (단계별 독립).

  → 평균 회전 3~10일/단계, 사이클 전체 1~2주에 완성
  → 단타(차트영웅)와 중기(적응형) 중간 = "준-중기" 매매법

MVP-2.5 작동 흐름:
  L1 22,500 매수 (체결) → quick_profit_target = 24,075 자동 계산 (×1.07)
                       ↓
  매 30분 cron → 현재가 ≥ 24,075 ?
                       ↓ YES
  지정가 매도 24,075 × qty=1 → status QUICK_SOLD
                       ↓
  +7% 수익 확정, 다음 사이클 (L1 재진입 또는 신규 종목)

MVP-1 천장 매도와의 관계:
  - 단계별 독립: L1 익절 후에도 L2/L3 보유 중이면 MVP-1 천장 매도 그대로 작동
  - 모든 stage가 QUICK_SOLD 되면 종목 완전 청산 (큐 정리)
  - 부분 익절 vs 천장 회복 매도 = 둘 다 가능 (단계별로)

5/17 자기반성 #1 적용: import + 함수 호출 + 상태 머신 흐름 검증.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


# === 임계 (.env 동적) ===
QUICK_PROFIT_ENABLED = os.getenv("ADAPTIVE_QUICK_PROFIT_ENABLED", "1") == "1"
QUICK_PROFIT_PCT = float(os.getenv("ADAPTIVE_QUICK_PROFIT_PCT", "7"))      # +7% 익절
QUICK_PROFIT_RATIO = float(os.getenv("ADAPTIVE_QUICK_PROFIT_RATIO", "1.0"))  # 1.0=전량


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


def execute_quick_sell(broker, ticker: str, stage: dict) -> dict:
    """단계별 빠른 익절 매도 실행 (지정가).

    Returns:
        {"success": bool, "order_id": str, "price": int, "qty": int, "error": str}
    """
    if not QUICK_PROFIT_ENABLED:
        return {"success": False, "error": "ADAPTIVE_QUICK_PROFIT_ENABLED=0"}

    actual_qty = int(stage.get("actual_qty", 0))
    if actual_qty <= 0:
        return {"success": False, "error": f"actual_qty 부적합 ({actual_qty})"}

    target_price = int(stage.get("quick_profit_target", 0))
    if target_price <= 0:
        return {"success": False, "error": f"quick_profit_target 부적합 ({target_price})"}

    sell_qty = max(1, int(actual_qty * QUICK_PROFIT_RATIO))

    try:
        # 지정가 매도 (target_price) — 즉시 익절 확정
        if hasattr(broker, "sell_limit"):
            order = broker.sell_limit(ticker, target_price, sell_qty)
            order_id = getattr(order, "order_id", "") or ""
        else:
            # mojito2 fallback
            res = broker.create_limit_sell_order(ticker, target_price, sell_qty)
            order_id = res.get("output", {}).get("ODNO", "") if res else ""

        return {
            "success": True,
            "order_id": order_id,
            "price": target_price,
            "qty": sell_qty,
        }
    except Exception as e:
        logger.error("quick sell %s L%s 실패: %s", ticker, stage.get("level"), e)
        return {"success": False, "error": str(e)}


def check_quick_profit_triggers(broker) -> list[dict]:
    """모든 FILLED stage 순회 + 빠른 익절 트리거 평가.

    매 30분 cron 호출 (MVP-2와 동일 주기).
    """
    triggers: list[dict] = []

    if not QUICK_PROFIT_ENABLED:
        logger.info("ADAPTIVE_QUICK_PROFIT_ENABLED=0 — MVP-2.5 비활성")
        return triggers

    if _is_kill_switch_active():
        logger.info("KILL_SWITCH 발동 — 빠른 익절 정지")
        return triggers

    # MVP-2 큐 로드
    from src.use_cases.adaptive_buy_queue import (
        _load_queues_raw,
        _save_queues_raw,
        STATUS_FILLED,
        STATUS_QUICK_SOLD,
    )

    raw = _load_queues_raw()
    queues = raw.get("queues", {})
    modified = False

    for ticker, entry in queues.items():
        current_price = _fetch_current_price(broker, ticker)
        if current_price <= 0:
            continue

        for stage in entry.get("stages", []):
            if stage.get("status") != STATUS_FILLED:
                continue

            target = int(stage.get("quick_profit_target", 0))
            if target <= 0:
                continue

            # 빠른 익절 트리거: 현재가 ≥ 매수가 × 1.07
            if current_price >= target:
                sell_result = execute_quick_sell(broker, ticker, stage)

                if sell_result["success"]:
                    actual_buy = int(stage.get("actual_price", 0))
                    profit_pct = (
                        (target / actual_buy - 1) * 100 if actual_buy > 0 else 0
                    )
                    stage["status"] = STATUS_QUICK_SOLD
                    stage["quick_profit_order_id"] = sell_result.get("order_id", "")
                    stage["quick_profit_sold_at"] = datetime.now().isoformat(
                        timespec="seconds"
                    )
                    stage["quick_profit_sold_price"] = sell_result.get("price", target)
                    modified = True

                    triggers.append({
                        "ticker": ticker,
                        "name": entry.get("name", ""),
                        "level": stage.get("level"),
                        "actual_buy_price": actual_buy,
                        "sold_price": sell_result.get("price", target),
                        "profit_pct": round(profit_pct, 2),
                        "qty": sell_result.get("qty", 0),
                        "order_id": sell_result.get("order_id", ""),
                        "current_price": current_price,
                    })

    if modified:
        _save_queues_raw(raw)

    return triggers


def format_quick_profit_for_telegram(trigger: dict) -> str:
    """텔레그램 알림용 포맷."""
    name = trigger.get("name") or trigger.get("ticker", "")
    level = trigger.get("level", "?")
    buy = int(trigger.get("actual_buy_price", 0))
    sold = int(trigger.get("sold_price", 0))
    profit = trigger.get("profit_pct", 0.0)
    qty = int(trigger.get("qty", 0))
    profit_amount = (sold - buy) * qty

    return (
        f"💰 빠른 익절 L{level} 체결 [{name}]\n"
        f"  매수: {buy:,} → 매도: {sold:,} ({profit:+.2f}%)\n"
        f"  수량: {qty}주, 수익: {profit_amount:,}원\n"
        f"  주문 ID: {trigger.get('order_id', '')}"
    )


def reset_quick_sold_for_reentry(ticker: str) -> bool:
    """QUICK_SOLD stage를 PENDING으로 재설정 — 동일 종목 사이클 재시작용.

    사용 예: L1 익절 후 가격 다시 떨어지면 L1 재매수 가능하도록.
    1주차는 보수적으로 사용 X (수동 호출만).
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
            modified = True

    if modified:
        _save_queues_raw(raw)

    return modified
