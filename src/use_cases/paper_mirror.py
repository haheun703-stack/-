"""Paper Mirror Use Case — 5/20 자비스 자율 매매 병행 시뮬 (§13-2-1 명세, 2026-05-19)

배경: 사장님 5/18 결단 옵션 B — 실주문과 같은 시그널로 paper 시뮬 동시 가동
- 관찰자 역할 (실주문 결정에 영향 0)
- 슬리피지·수수료·거래세 가정 누적 → 5/21+ 가정 실측 보정

흐름:
  1. auto_buy_executor가 BUY 결정 시 paper_record_entry() 호출
  2. paper_order_adapter.buy_limit으로 가상 체결
  3. data/paper_mirror/{date}_positions.json 갱신
  4. owner_rule_monitor가 paper_evaluate_positions()로 시뮬 청산

모드 스위치: PAPER_MIRROR_MODE=true (.env)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.adapters.paper_order_adapter import PaperOrderAdapter
from src.entities.trading_models import OrderStatus

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_POSITIONS_DIR = PROJECT_ROOT / "data" / "paper_mirror"
PAPER_LOG_DIR = PROJECT_ROOT / "logs"


def is_paper_mirror_enabled() -> bool:
    """PAPER_MIRROR_MODE=true 시만 가동."""
    return os.environ.get("PAPER_MIRROR_MODE", "false").lower() == "true"


def _positions_path(today: str) -> Path:
    """data/paper_mirror/{YYYY-MM-DD}_positions.json"""
    PAPER_POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_POSITIONS_DIR / f"{today}_positions.json"


def load_paper_positions(today: str) -> dict:
    """오늘 paper 포지션 로드."""
    path = _positions_path(today)
    if not path.exists():
        return {"positions": {}, "updated_at": None, "trades": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("paper_mirror positions 로드 실패: %s — 빈 dict", e)
        return {"positions": {}, "updated_at": None, "trades": []}


def save_paper_positions(today: str, state: dict) -> None:
    """저장."""
    state["updated_at"] = datetime.now().isoformat()
    _positions_path(today).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def paper_record_entry(ticker: str, name: str, price: int, score: float, today: str) -> dict | None:
    """auto_buy_executor의 실주문 BUY 결정 직후 호출.

    같은 시그널·같은 후보·같은 타이밍으로 paper 매수 시뮬.
    실주문 결정에 영향 0 — 예외 발생해도 호출 측에서 무시 가능.

    Returns:
        시뮬 결과 dict (filled_price, fee, total_cost) 또는 None (모드 OFF)
    """
    if not is_paper_mirror_enabled():
        logger.debug("PAPER_MIRROR_MODE=false — 시뮬 스킵")
        return None

    state = load_paper_positions(today)
    positions = state.get("positions", {})

    # 일일 1건 한도 (live와 동일)
    if positions:
        logger.info("[PAPER] 이미 진입 1건 보유 — 추가 진입 스킵")
        return None

    adapter = PaperOrderAdapter()
    # C2-paper_mirror fix (5/28 코덱스 검수): mode/executor_bot 명시 — L10 가드 통과 필수
    try:
        order = adapter.buy_limit(
            ticker, price, 1, orderbook_available=False,
            mode="paper", executor_bot="quant",
        )
    except Exception as e:
        # NoIntentError / IntentSignatureError / IntentExpiredError / ValueError 모두 명시 로깅
        logger.error("[PAPER 매수 시뮬 차단] %s(%s) @ %d: %s — %s",
                     name, ticker, price, type(e).__name__, e)
        return None

    if order.status != OrderStatus.FILLED:
        logger.warning("[PAPER] 매수 시뮬 실패 — status=%s", order.status)
        return None

    # positions 갱신 (live owner_rule_positions와 같은 스키마)
    positions[ticker] = {
        "entry_price": int(order.filled_price),
        "entry_date": today,
        "name": name,
        "qty": 1,
        "peak_price": int(order.filled_price),
        "trailing_active": False,
        "order_id": order.order_id,
        "integrated_score": score,
        "live_intended_price": price,  # 실주문 의도 가격 (슬리피지 비교용)
        "slippage_abs": int(order.filled_price) - price,
        "fee_assumed": int(order.message.split("fee=")[1].split()[0]) if "fee=" in order.message else 0,
        "created_at": datetime.now().isoformat(),
        "mode": "paper_mirror",
    }
    state["positions"] = positions

    # 거래 히스토리 추가
    trades = state.get("trades", [])
    trades.append({
        "ticker": ticker,
        "name": name,
        "side": "BUY",
        "live_price": price,
        "paper_filled": int(order.filled_price),
        "slippage_abs": int(order.filled_price) - price,
        "slippage_pct": round((int(order.filled_price) - price) / price * 100, 3),
        "score": score,
        "ts": datetime.now().isoformat(),
    })
    state["trades"] = trades

    save_paper_positions(today, state)

    logger.info(
        "[PAPER ✅] 매수 시뮬 %s(%s) @ %d원 (실주문 의도 %d, 슬리피지 +%d원 %.3f%%)",
        name, ticker, int(order.filled_price), price,
        int(order.filled_price) - price,
        (int(order.filled_price) - price) / price * 100,
    )
    return {
        "filled_price": int(order.filled_price),
        "slippage_abs": int(order.filled_price) - price,
        "order_id": order.order_id,
        "message": order.message,
    }


def paper_record_exit(ticker: str, current_price: int, today: str, reason: str = "SELL_FORCE_CLOSE", market: str = "KOSPI") -> dict | None:
    """owner_rule_monitor의 SELL 결정 시 호출 — paper 청산 시뮬.

    Args:
        ticker: 종목 코드
        current_price: 청산 시점 KIS 현재가
        today: YYYY-MM-DD
        reason: SELL_STOP_LOSS / SELL_TRAILING / SELL_FORCE_CLOSE
        market: KOSPI / KOSDAQ (거래세 계산용)
    """
    if not is_paper_mirror_enabled():
        return None

    state = load_paper_positions(today)
    positions = state.get("positions", {})
    pos = positions.get(ticker)

    if not pos:
        logger.debug("[PAPER] %s 포지션 없음 — 청산 스킵", ticker)
        return None

    adapter = PaperOrderAdapter()
    # C2-paper_mirror fix (5/28): mode/executor_bot 명시 + 예외 명시 로깅 (silent X)
    try:
        order = adapter.sell_limit(
            ticker, current_price, pos["qty"],
            orderbook_available=False, market=market,
            mode="paper", executor_bot="quant",
        )
    except Exception as e:
        logger.error("[PAPER 청산 시뮬 차단] %s @ %d: %s — %s",
                     ticker, current_price, type(e).__name__, e)
        return None

    # 손익 계산
    entry = pos["entry_price"]
    filled = int(order.filled_price)
    pnl_abs = (filled - entry) * pos["qty"]
    pnl_pct = round((filled - entry) / entry * 100, 3)

    # 거래 히스토리 추가
    trades = state.get("trades", [])
    trades.append({
        "ticker": ticker,
        "name": pos.get("name", ticker),
        "side": "SELL",
        "live_price": current_price,
        "paper_filled": filled,
        "entry_price": entry,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "ts": datetime.now().isoformat(),
    })
    state["trades"] = trades

    # 포지션 제거
    positions.pop(ticker, None)
    state["positions"] = positions

    save_paper_positions(today, state)

    logger.info(
        "[PAPER ✅] 매도 시뮬 %s(%s) @ %d원 (진입 %d, %s%+.3f%%, 사유 %s)",
        pos.get("name", ticker), ticker, filled, entry,
        "+" if pnl_pct >= 0 else "", pnl_pct, reason,
    )
    return {
        "filled_price": filled,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "reason": reason,
    }
