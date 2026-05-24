"""적응형 포지션 매매법 MVP-2 — 분할매수 큐 (-10% / -20% / -30%).

배경 (퐝가님 5/23 흐름 + 5/24 일요일 작업):
  6단계 흐름 중 [4단계 분할매수 대기]:
    "내리면 사고, 끝까지 오르고 -3%면 팔고,
     조정의 시작점에서 다시 사고 — 단, 한 번에 사지 말고 3 단계로 나눠서."

  MVP-1 매도 후 천장가 + 가용 현금 알면 자동 큐 등록.
  종목별 3 단계 지정가:
    - L1: 천장 -10%  (가용 30%) — 1차 진입
    - L2: 천장 -20%  (가용 30%) — 평균 단가 낮춤
    - L3: 천장 -30%  (가용 40%) — 바닥 매수 (가장 큰 비중)

  평단가 효과:
    0.30 × 0.90 + 0.30 × 0.80 + 0.40 × 0.70
    = 0.27 + 0.24 + 0.28 = 0.79
    → 평단가 = 천장 × 79% (= 천장 대비 -21% 효과)

  매 30 분 cron으로 단계별 도달 여부 확인 + (옵션) 자동 매수.

MVP-2 기능:
  1. register_buy_queue: 매도 후 천장가 + 가용 현금 → 3 단계 큐 등록
  2. check_and_trigger_queues: cron에서 호출, 도달 단계 자동 매수 / 알림
  3. data/adaptive_buy_queue.json 영속 저장 (상태 추적)
  4. KILL_SWITCH 발동 시 정지
  5. 3 종목 한도 (ADAPTIVE_MAX_POSITIONS)
  6. 1주차 안전: ADAPTIVE_AUTO_BUY=0 → 알림만, 자동매수 X

상태 머신:
  PENDING ──도달──> TRIGGERED ──자동매수──> FILLED
                            └──알림만──> NOTIFIED
                                          └──다음 cron 재시도

사용:
  from src.use_cases.adaptive_buy_queue import register_buy_queue, check_and_trigger_queues
  register_buy_queue(ticker="240810", peak_price=25000, available_cash=3_000_000, name="원익IPS")
  results = check_and_trigger_queues(broker)  # cron
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QUEUE_PATH = PROJECT_ROOT / "data" / "adaptive_buy_queue.json"
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


# === 임계 (.env 동적, 1주차는 보수적) ===
def _parse_int_list(env_val: str, default: list[int]) -> list[int]:
    try:
        return [int(x.strip()) for x in env_val.split(",") if x.strip()]
    except (ValueError, AttributeError):
        return default


SPLIT_LEVELS = _parse_int_list(os.getenv("ADAPTIVE_SPLIT_LEVELS", "10,20,30"), [10, 20, 30])
SPLIT_RATIOS = _parse_int_list(os.getenv("ADAPTIVE_SPLIT_RATIOS", "30,30,40"), [30, 30, 40])
SPLIT_MAX_AMOUNT = int(os.getenv("ADAPTIVE_SPLIT_MAX_AMOUNT", "1000000"))     # 1단위 100만
SPLIT_MAX_QTY = int(os.getenv("ADAPTIVE_SPLIT_MAX_QTY", "0"))                  # 1주차 1주 cap (0=무제한)
MAX_POSITIONS = int(os.getenv("ADAPTIVE_MAX_POSITIONS", "3"))                  # 3종목 한도
AUTO_BUY = os.getenv("ADAPTIVE_AUTO_BUY", "0") == "1"                          # 1주차 알림만
QUEUE_EXPIRY_DAYS = int(os.getenv("ADAPTIVE_QUEUE_EXPIRY_DAYS", "60"))


# === 상태 상수 ===
STATUS_PENDING = "PENDING"      # 가격 미도달
STATUS_TRIGGERED = "TRIGGERED"  # 가격 도달 + 알림 (AUTO_BUY=0)
STATUS_FILLED = "FILLED"        # 자동매수 성공
STATUS_EXPIRED = "EXPIRED"      # 만료
STATUS_FAILED = "FAILED"        # 매수 실패


@dataclass
class QueueStage:
    """큐 단계."""

    level: int                  # 1, 2, 3
    target_pct: float           # 0.90, 0.80, 0.70 (천장 대비)
    target_price: int           # 지정가
    alloc_ratio: float          # 0.30, 0.30, 0.40
    alloc_amount: int           # 배정 금액 (KRW)
    qty: int                    # 매수 수량
    status: str = STATUS_PENDING
    triggered_at: Optional[str] = None
    order_id: Optional[str] = None
    actual_price: int = 0       # 실제 체결가
    actual_qty: int = 0
    error: Optional[str] = None


def _is_kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


def _load_queues_raw() -> dict[str, Any]:
    if not QUEUE_PATH.exists():
        return {"queues": {}}
    try:
        with QUEUE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("queue load 실패: %s — 빈 큐로 시작", e)
        return {"queues": {}}


def _save_queues_raw(data: dict[str, Any]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_queues() -> dict[str, dict]:
    """저장된 큐 전체 dict 반환 ({ticker: queue_entry})."""
    return _load_queues_raw().get("queues", {})


def _count_active_positions(queues: dict[str, dict]) -> int:
    """활성 큐 수 (PENDING/TRIGGERED 단계 1개 이상 보유 종목)."""
    count = 0
    for entry in queues.values():
        for stage in entry.get("stages", []):
            if stage.get("status") in (STATUS_PENDING, STATUS_TRIGGERED):
                count += 1
                break
    return count


def _build_stages(peak_price: int, available_cash: int) -> list[QueueStage]:
    """3단계 큐 빌드 (LEVELS/RATIOS .env)."""
    stages: list[QueueStage] = []
    if len(SPLIT_LEVELS) != len(SPLIT_RATIOS):
        logger.warning("LEVELS/RATIOS 길이 불일치 — 기본값으로 fallback")
        levels, ratios = [10, 20, 30], [30, 30, 40]
    else:
        levels, ratios = SPLIT_LEVELS, SPLIT_RATIOS

    for i, (pct_drop, alloc_pct) in enumerate(zip(levels, ratios), start=1):
        target_pct = 1.0 - pct_drop / 100.0          # 0.90, 0.80, 0.70
        target_price = int(peak_price * target_pct)
        alloc_amount = min(int(available_cash * alloc_pct / 100), SPLIT_MAX_AMOUNT)
        qty = alloc_amount // max(target_price, 1)
        # 1주차 안전: SPLIT_MAX_QTY > 0이면 수량 cap (차트영웅 max-qty 1 동일 안전망)
        if SPLIT_MAX_QTY > 0:
            qty = min(qty, SPLIT_MAX_QTY)
        stages.append(QueueStage(
            level=i,
            target_pct=target_pct,
            target_price=target_price,
            alloc_ratio=alloc_pct / 100.0,
            alloc_amount=alloc_amount,
            qty=qty,
        ))
    return stages


def register_buy_queue(
    ticker: str,
    peak_price: int,
    available_cash: int,
    name: str = "",
) -> dict:
    """매도 후 천장가 + 가용 현금 → 3단계 큐 자동 등록.

    Args:
        ticker: 종목 코드
        peak_price: 천장가 (MVP-1 detect_peak_signal 결과)
        available_cash: 가용 현금 (KRW)
        name: 종목명 (알림용)

    Returns:
        {"success": bool, "ticker": str, "stages": [...], "error": str}
    """
    if _is_kill_switch_active():
        return {"success": False, "error": "KILL_SWITCH 발동 중 — 등록 정지"}

    if peak_price <= 0:
        return {"success": False, "error": f"peak_price 부적합: {peak_price}"}

    if available_cash < 100_000:
        return {"success": False, "error": f"가용 현금 부족: {available_cash:,}"}

    raw = _load_queues_raw()
    queues = raw.get("queues", {})

    # 동일 종목 기존 큐 있으면 덮어씀 (천장 갱신 케이스)
    is_update = ticker in queues
    if not is_update and _count_active_positions(queues) >= MAX_POSITIONS:
        return {
            "success": False,
            "error": f"3종목 한도 도달 ({MAX_POSITIONS}) — 기존 종목 청산 후 재등록",
        }

    stages = _build_stages(peak_price, available_cash)

    queues[ticker] = {
        "ticker": ticker,
        "name": name,
        "peak_price": int(peak_price),
        "available_cash": int(available_cash),
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "stages": [stage.__dict__ for stage in stages],
    }
    raw["queues"] = queues
    _save_queues_raw(raw)

    return {
        "success": True,
        "ticker": ticker,
        "name": name,
        "is_update": is_update,
        "peak_price": peak_price,
        "available_cash": available_cash,
        "stages": [stage.__dict__ for stage in stages],
    }


def _fetch_current_price(broker, ticker: str) -> int:
    """현재가 fetch (adaptive_position_manager와 동일 패턴)."""
    try:
        res = broker.fetch_price(ticker)
        output = res.get("output", {}) if res else {}
        return int(str(output.get("stck_prpr", 0)).replace(",", "") or 0)
    except Exception as e:
        logger.warning("price fetch %s 실패: %s", ticker, e)
        return 0


def _is_expired(registered_at: str) -> bool:
    """등록 후 N일 경과 여부."""
    try:
        reg = datetime.fromisoformat(registered_at)
        return (datetime.now() - reg).days > QUEUE_EXPIRY_DAYS
    except (ValueError, TypeError):
        return False


def execute_auto_buy(broker, ticker: str, stage_dict: dict) -> dict:
    """단계별 자동 매수 (ADAPTIVE_AUTO_BUY=1일 때만).

    Returns:
        {"success": bool, "order_id": str, "price": int, "qty": int, "error": str}
    """
    if not AUTO_BUY:
        return {"success": False, "error": "ADAPTIVE_AUTO_BUY=0 — 알림만"}

    target_price = int(stage_dict.get("target_price", 0))
    qty = int(stage_dict.get("qty", 0))
    if target_price <= 0 or qty <= 0:
        return {"success": False, "error": f"target_price/qty 부적합 ({target_price}/{qty})"}

    try:
        # 지정가 매수 (target_price)
        order = broker.buy_limit(ticker, target_price, qty)
        order_id = getattr(order, "order_id", "") or ""
        return {
            "success": True,
            "order_id": order_id,
            "price": target_price,
            "qty": qty,
        }
    except Exception as e:
        logger.error("auto buy %s L%s 실패: %s", ticker, stage_dict.get("level"), e)
        return {"success": False, "error": str(e)}


def check_and_trigger_queues(broker) -> list[dict]:
    """모든 활성 큐 순회 + 가격 도달 단계 트리거.

    매 30 분 cron에서 호출.

    Returns:
        [{ticker, name, level, status, target_price, current_price, ...}, ...]
        trigger 발생한 단계만 반환.
    """
    triggers: list[dict] = []

    if _is_kill_switch_active():
        logger.info("KILL_SWITCH 발동 — 큐 트리거 정지")
        return triggers

    raw = _load_queues_raw()
    queues = raw.get("queues", {})
    modified = False

    for ticker, entry in list(queues.items()):
        # 만료 체크
        if _is_expired(entry.get("registered_at", "")):
            for stage in entry.get("stages", []):
                if stage.get("status") in (STATUS_PENDING, STATUS_TRIGGERED):
                    stage["status"] = STATUS_EXPIRED
                    modified = True
            triggers.append({
                "ticker": ticker,
                "name": entry.get("name", ""),
                "event": "EXPIRED",
                "registered_at": entry.get("registered_at", ""),
            })
            continue

        current_price = _fetch_current_price(broker, ticker)
        if current_price <= 0:
            continue

        for stage in entry.get("stages", []):
            if stage.get("status") != STATUS_PENDING:
                continue

            target_price = int(stage.get("target_price", 0))
            if target_price <= 0:
                continue

            # 도달: 현재가 ≤ 지정가 (떨어져서 지정가 도달)
            if current_price <= target_price:
                stage["triggered_at"] = datetime.now().isoformat(timespec="seconds")

                # 자동 매수 시도
                buy_result = execute_auto_buy(broker, ticker, stage)
                if buy_result["success"]:
                    stage["status"] = STATUS_FILLED
                    stage["order_id"] = buy_result.get("order_id", "")
                    stage["actual_price"] = buy_result.get("price", target_price)
                    stage["actual_qty"] = buy_result.get("qty", stage.get("qty", 0))
                elif buy_result.get("error", "").startswith("ADAPTIVE_AUTO_BUY=0"):
                    # 알림만 모드 — TRIGGERED로 표시 후 추후 수동/자동
                    stage["status"] = STATUS_TRIGGERED
                else:
                    stage["status"] = STATUS_FAILED
                    stage["error"] = buy_result.get("error", "")

                modified = True
                triggers.append({
                    "ticker": ticker,
                    "name": entry.get("name", ""),
                    "level": stage.get("level"),
                    "status": stage.get("status"),
                    "target_price": target_price,
                    "current_price": current_price,
                    "peak_price": entry.get("peak_price", 0),
                    "qty": stage.get("qty", 0),
                    "alloc_amount": stage.get("alloc_amount", 0),
                    "order_id": stage.get("order_id"),
                    "error": stage.get("error"),
                })

    if modified:
        _save_queues_raw(raw)

    return triggers


def get_queue_status(ticker: str) -> Optional[dict]:
    """특정 종목 큐 상태."""
    queues = load_queues()
    return queues.get(ticker)


def format_trigger_for_telegram(trigger: dict) -> str:
    """텔레그램 알림용 포맷."""
    name = trigger.get("name") or trigger.get("ticker", "")

    if trigger.get("event") == "EXPIRED":
        return (
            f"⏰ 분할매수 큐 만료 [{name}]\n"
            f"  등록일: {trigger.get('registered_at', '')[:10]}\n"
            f"  {QUEUE_EXPIRY_DAYS}일 경과 — 큐 자동 정리"
        )

    status = trigger.get("status", "")
    level = trigger.get("level", "?")
    target = int(trigger.get("target_price", 0))
    current = int(trigger.get("current_price", 0))
    peak = int(trigger.get("peak_price", 0))
    qty = int(trigger.get("qty", 0))
    alloc = int(trigger.get("alloc_amount", 0))

    pct_from_peak = ((current / peak) - 1) * 100 if peak > 0 else 0.0

    if status == STATUS_FILLED:
        head = f"✅ 분할매수 L{level} 체결 [{name}]"
        action_line = f"  주문ID: {trigger.get('order_id', '')}"
    elif status == STATUS_TRIGGERED:
        head = f"🔔 분할매수 L{level} 가격 도달 [{name}] (알림만)"
        action_line = f"  ADAPTIVE_AUTO_BUY=0 — 수동 매수 또는 활성화 검토"
    elif status == STATUS_FAILED:
        head = f"⚠️ 분할매수 L{level} 실패 [{name}]"
        action_line = f"  오류: {trigger.get('error', '')[:60]}"
    else:
        head = f"❓ 분할매수 L{level} 상태={status} [{name}]"
        action_line = ""

    lines = [
        head,
        f"  천장: {peak:,} → 현재: {current:,} ({pct_from_peak:+.2f}%)",
        f"  지정가: {target:,} (L{level} = 천장 -{int((1 - target/peak)*100) if peak else 0}%)",
        f"  배정: {alloc:,}원 / {qty}주",
    ]
    if action_line:
        lines.append(action_line)
    return "\n".join(lines)


def clear_queue(ticker: str) -> bool:
    """특정 종목 큐 삭제 (수동 청산용)."""
    raw = _load_queues_raw()
    queues = raw.get("queues", {})
    if ticker in queues:
        del queues[ticker]
        raw["queues"] = queues
        _save_queues_raw(raw)
        return True
    return False
