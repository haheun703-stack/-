"""적응형 매매법 MVP-2.6 — 자동 손절 -5% (5/25 백테스트 + 퐝가님 지시).

배경:
  5/25 백테스트 (3개월, 64종목, 130 매수):
    R0 손절X: MDD -20.84%, 누적 +851.9% (강세장 편향)
    R1 -3%:   MDD -3%, 누적 +221.5% (너무 빈번)
    R2 -5%:   MDD -5%, 누적 +338.5%, PF 2.21 ★ 채택
    R3 -7%:   MDD -7%, 누적 +700.4%, PF 4.75

  퐝가님 5/25 22:00 지시: -5% 자동 손절 (시드 보전 우선)
  메모리 [[feedback-backtest-first]]: 사용자 지정값 즉시 적용 + 사후 검증

룰:
  보유 종목 FILLED stage 매수가 대비 -5% 도달 시 즉시 시장가 매도.

Quick Profit 충돌 방지:
  FILLED 상태에서만 손절 평가 (QUICK_ARMED는 trailing이 더 우월)
  → 매수 → +7% 도달 전까지는 -5% 손절 활성
  → +7% 도달 후 ARMED 진입 → 손절 비활성 (Quick Profit이 익절 보호)

P0-4 보장:
  KILL_SWITCH 활성 시에도 손절 매도 계속 (매수만 차단)
  손실 차단이 손절의 목적

손실 한도 시뮬:
  종목당 -5% × 1주 평균 26만원 ≈ 13,000원
  3종목 동시 손절 시 최대 39,000원 (시드 25,050,012원의 0.16%)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from src.utils.trade_runtime_safety import assert_runtime_orders_allowed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"

# === 임계 (.env 동적) ===
STOP_LOSS_ENABLED = os.getenv("ADAPTIVE_STOP_LOSS_ENABLED", "1") == "1"
STOP_LOSS_PCT = float(os.getenv("ADAPTIVE_STOP_LOSS_PCT", "5"))  # 5% 손절

# 상태 상수 (adaptive_buy_queue.py와 동일)
STATUS_FILLED = "FILLED"
STATUS_STOP_LOSS_SOLD = "STOP_LOSS_SOLD"


def _is_kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


def _fetch_current_price(broker, ticker: str) -> int:
    """현재가 fetch (실패 시 0)."""
    try:
        if hasattr(broker, "fetch_price"):
            r = broker.fetch_price(ticker)
            return int(r.get("output", {}).get("stck_prpr", 0))
        return 0
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger.warning("price fetch %s 실패: %s", ticker, e)
        return 0


def execute_stop_loss_sell(
    broker, ticker: str, stage: dict,
    *, mode: str | None = None, executor_bot: str | None = None,
) -> dict:
    """손절 매도 — 5/28 코덱스 결정문: mode/executor_bot 명시 강제 가능.

    backward compat: 둘 다 None → 기존 _guard 9중만.
    명시 시 → L10 (order_intents_gate) 강제.
    """
    """손절 시장가 매도 실행.

    Returns:
        {"success": bool, "order_id": str, "qty": int, "price": int,
         "loss_pct": float, "error": str}
    """
    qty = int(stage.get("actual_qty", 0))
    if qty <= 0:
        return {"success": False, "error": f"actual_qty={qty}"}

    actual_buy = int(stage.get("actual_price", 0))
    current = _fetch_current_price(broker, ticker)

    try:
        if hasattr(broker, "sell_market"):
            adapter_kwargs = {}
            if mode is not None or executor_bot is not None:
                adapter_kwargs = {"mode": mode, "executor_bot": executor_bot}
            order = broker.sell_market(ticker, qty, **adapter_kwargs)
            order_id = getattr(order, "order_id", "") or ""
        else:
            raise RuntimeError(
                "[P0-D] raw mojito broker 호출 차단 — KisOrderAdapter 인스턴스 필수. "
                "호출자가 KisOrderAdapter를 broker 인자로 전달해야 함."
            )

        loss_pct = (current / actual_buy - 1) * 100 if actual_buy > 0 else 0

        return {
            "success": True,
            "order_id": order_id,
            "qty": qty,
            "price": current,
            "loss_pct": loss_pct,
        }
    except Exception as e:
        logger.error("stop loss sell %s L%s 실패: %s", ticker, stage.get("level"), e)
        return {"success": False, "error": str(e)}


def check_stop_loss_triggers(
    broker, *, mode: str | None = None, executor_bot: str | None = None,
) -> list[dict]:
    """모든 FILLED stage 순회 + -5% 손절 평가.

    매 30분 cron에서 호출. P0-1 락 + P0-4 매도 계속 보장.
    5/28 코덱스 검수 후속: mode/executor_bot 명시 시 execute_stop_loss_sell에 전달.

    Returns:
        손절 매도 발생한 stage 리스트
    """
    triggers: list[dict] = []

    if not STOP_LOSS_ENABLED:
        logger.info("ADAPTIVE_STOP_LOSS_ENABLED=0 — MVP-2.6 비활성")
        return triggers

    # P0-4: KILL_SWITCH는 매수만 차단, 손절 매도는 계속 진행
    if _is_kill_switch_active():
        logger.warning(
            "KILL_SWITCH 활성 — Stop Loss 매도는 계속 진행 (P0-4 보장)"
        )
        # return 없음 — 매도 평가 계속

    # P0-1: 락 + atomic write
    from src.use_cases.adaptive_buy_queue import (
        _locked_read_modify_write,
        QUEUE_PATH,
    )

    def _modify(raw: dict[str, Any]) -> Any:
        queues = raw.setdefault("queues", {})
        modified = False
        now_iso = datetime.now().isoformat(timespec="seconds")

        for ticker, entry in queues.items():
            current_price = _fetch_current_price(broker, ticker)
            if current_price <= 0:
                continue

            for stage in entry.get("stages", []):
                # FILLED 상태만 평가 (QUICK_ARMED는 trailing이 보호)
                if stage.get("status") != STATUS_FILLED:
                    continue

                actual_buy = int(stage.get("actual_price", 0))
                if actual_buy <= 0:
                    continue

                # 손절 임계 도달 평가 (백테스트 R2: -5%)
                stop_threshold = actual_buy * (1 - STOP_LOSS_PCT / 100)
                if current_price <= stop_threshold:
                    sell_result = execute_stop_loss_sell(
                        broker, ticker, stage,
                        mode=mode, executor_bot=executor_bot,
                    )
                    if sell_result["success"]:
                        stage["status"] = STATUS_STOP_LOSS_SOLD
                        stage["stop_loss_sold_at"] = now_iso
                        stage["stop_loss_sold_price"] = sell_result["price"]
                        stage["stop_loss_pct"] = sell_result["loss_pct"]
                        stage["stop_loss_order_id"] = sell_result["order_id"]
                        modified = True

                        triggers.append({
                            "ticker": ticker,
                            "name": entry.get("name", ""),
                            "level": stage.get("level"),
                            "status": STATUS_STOP_LOSS_SOLD,
                            "actual_buy": actual_buy,
                            "current_price": current_price,
                            "loss_pct": sell_result["loss_pct"],
                            "qty": stage.get("actual_qty", 0),
                            "order_id": sell_result["order_id"],
                        })

                        logger.warning(
                            "🛑 손절 매도: %s L%s — 매수가 %d → 현재가 %d (%.2f%%)",
                            ticker, stage.get("level"), actual_buy, current_price,
                            sell_result["loss_pct"]
                        )
                    else:
                        logger.error(
                            "손절 매도 실패 %s L%s: %s",
                            ticker, stage.get("level"), sell_result.get("error", "")
                        )

        # 락 우회 방지 마커: 변경 없을 때만 _skip_save
        if not modified:
            raw["_skip_save"] = True
        return raw

    _locked_read_modify_write(QUEUE_PATH, _modify)
    return triggers


def format_stop_loss_for_telegram(trigger: dict) -> str:
    """텔레그램 알림 포맷."""
    return (
        f"🛑 손절 매도 [{trigger.get('name') or trigger['ticker']}] L{trigger.get('level')}\n"
        f"  매수가: {trigger.get('actual_buy', 0):,}원\n"
        f"  매도가: {trigger.get('current_price', 0):,}원\n"
        f"  손실률: {trigger.get('loss_pct', 0):+.2f}%\n"
        f"  수량: {trigger.get('qty', 0)}주\n"
        f"  주문번호: {trigger.get('order_id', '')}\n"
        f"  → MVP-2.6 자동 손절 -5% (5/25 백테스트 R2 채택)"
    )
