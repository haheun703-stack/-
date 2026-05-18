"""사장님 룰 자동 모니터 — 5/20 출격일 매 5분 자동 청산 평가 (4번 작업, 2026-05-18)

배경: 사장님 자동매매 ON 결단 + 사장님 룰 (-3%/-3%)
- 5/20 자비스가 자동 매수 → 매 5분 모니터링 → 청산 조건 충족 시 자동 매도

흐름:
  1. KIS fetch_balance → 보유 종목 추출
  2. 각 종목에 owner_rule.evaluate_owner_rule() 적용
  3. action ∈ {SELL_STOP_LOSS, SELL_TRAILING, SELL_FORCE_CLOSE} → 자동 매도
  4. 즉시 텔레그램 카톡 사장님 알림

가동: 5/20 09:00~15:30 매 5분 (cron 등록)
환경변수: AUTO_TRADE_5_20=true 시만 작동
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

POSITIONS_STATE_PATH = PROJECT_ROOT / "data" / "owner_rule_positions.json"


def load_positions_state() -> dict:
    """진입가/peak_price 추적 상태 로드."""
    if not POSITIONS_STATE_PATH.exists():
        return {"positions": {}, "updated_at": None}
    try:
        return json.loads(POSITIONS_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"positions": {}, "updated_at": None}


def save_positions_state(state: dict) -> None:
    POSITIONS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def fetch_current_balance(broker) -> list[dict]:
    """KIS fetch_balance → 보유 종목 리스트."""
    try:
        # mojito broker.fetch_balance() 사용
        resp = broker.fetch_balance()
        out = resp.get("output1", [])
        holdings = []
        for r in out:
            qty = int(r.get("hldg_qty", 0) or 0)
            if qty <= 0:
                continue
            holdings.append({
                "ticker": r.get("pdno"),
                "name": r.get("prdt_name"),
                "qty": qty,
                "avg_price": int(float(r.get("pchs_avg_pric", 0) or 0)),
                "current_price": int(float(r.get("prpr", 0) or 0)),
            })
        return holdings
    except Exception as e:
        logger.error("fetch_balance 실패: %s", e)
        return []


def execute_sell(broker, ticker: str, qty: int, current_price: int) -> tuple[bool, str]:
    """KIS 시장가 매도 주문."""
    try:
        resp = broker.create_market_sell_order(symbol=ticker, quantity=qty)
        if resp.get("rt_cd") == "0":
            return True, "OK"
        return False, str(resp.get("msg1", "unknown"))
    except Exception as e:
        return False, str(e)


def send_telegram(msg: str) -> None:
    try:
        from src.telegram_sender import send_message
        send_message(msg)
    except Exception as e:
        logger.warning("텔레그램 발송 실패: %s", e)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 5/20 출격일 환경변수 가드
    if os.environ.get("AUTO_TRADE_5_20", "false").lower() != "true":
        logger.info("AUTO_TRADE_5_20 != true — 스킵")
        return 0

    # KILL_SWITCH 존재 시 즉시 종료
    if (PROJECT_ROOT / "data" / "KILL_SWITCH").exists():
        logger.info("KILL_SWITCH 존재 — 스킵")
        return 0

    from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
    from src.use_cases.owner_rule import evaluate_owner_rule

    adp = KisStockDataAdapter()
    broker = adp.broker

    holdings = fetch_current_balance(broker)
    if not holdings:
        logger.info("보유 종목 0건 — 스킵")
        return 0

    state = load_positions_state()
    positions = state.get("positions", {})

    now_hhmm = datetime.now().strftime("%H:%M")

    for h in holdings:
        tk = h["ticker"]
        pos = positions.get(tk, {})

        entry_price = pos.get("entry_price") or h["avg_price"]
        peak_price = max(pos.get("peak_price", 0), h["current_price"], entry_price)
        trailing_active = pos.get("trailing_active", False)

        # 사장님 룰 평가
        verdict = evaluate_owner_rule(
            entry_price=entry_price,
            current_price=h["current_price"],
            peak_price=peak_price,
            trailing_active=trailing_active,
            current_time=now_hhmm,
        )

        # 상태 갱신
        positions[tk] = {
            "entry_price": entry_price,
            "peak_price": verdict.peak_price,
            "trailing_active": verdict.trailing_active,
            "last_check_at": datetime.now().isoformat(),
            "name": h["name"],
            "qty": h["qty"],
        }

        if verdict.action != "HOLD":
            # 자동 매도
            logger.info("청산 결정 %s: %s", tk, verdict.reason)
            ok, msg = execute_sell(broker, tk, h["qty"], h["current_price"])
            emoji = "✅" if ok else "❌"

            tg_msg = (
                f"{emoji} [사장님 룰 자동 청산] {h['name']}({tk})\n"
                f"  {verdict.action}\n"
                f"  진입 {entry_price:,} → 현재 {h['current_price']:,} ({verdict.pnl_pct:+.2f}%)\n"
                f"  peak {verdict.peak_price:,} 대비 {verdict.peak_drop_pct:+.2f}%\n"
                f"  사유: {verdict.reason}\n"
                f"  주문: {'성공' if ok else '실패 — ' + msg}"
            )
            send_telegram(tg_msg)
            print(tg_msg)

            # 매도 성공 시 상태에서 제거
            if ok:
                positions.pop(tk, None)
        else:
            logger.debug("HOLD %s: %s", tk, verdict.reason)

    state["positions"] = positions
    state["updated_at"] = datetime.now().isoformat()
    save_positions_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
