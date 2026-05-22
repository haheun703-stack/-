"""역매수 신호 모니터 — 보유 종목 매 30분 평가 (워밍업 모드).

배경 (5/22 퐝가님 인사이트):
  기가막힌 조합 통과 종목이 D+1 마이너스 진입 시:
  - 개미털기 (외인+기관 매수 유지) → 역매수 신호 발동
  - 진짜 문제 (대량 매도) → 손절 시그널

워밍업 모드 (5/26~):
  - 신호만 텔레그램 알림 (실제 추매 매수 X)
  - 사용자 검증 후 5/30+ 정식 활성화 결단
  - .env COUNTER_TRADE_WARMUP=1 (default)

cron 등록 (VPS):
  */30 9-15 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 scripts/counter_trade_monitor.py
"""

from __future__ import annotations

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


def send_telegram(msg: str) -> None:
    try:
        from src.telegram_sender import send_message
        send_message(msg)
    except Exception as e:
        logger.warning("텔레그램 발송 실패: %s", e)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # KILL_SWITCH 가드 (5/22 안전 우선)
    if (PROJECT_ROOT / "data" / "KILL_SWITCH").exists():
        logger.info("KILL_SWITCH 존재 — 스킵")
        return 0

    from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
    from src.use_cases.counter_trade_manager import evaluate_position, fetch_supply_5d_from_jgis

    adp = KisStockDataAdapter()
    broker = adp.broker

    # 보유 종목 fetch
    try:
        resp = broker.fetch_balance()
        out = resp.get("output1", [])
        holdings = []
        for r in out:
            qty = int(r.get("hldg_qty", 0) or 0)
            if qty <= 0: continue
            holdings.append({
                "ticker": r.get("pdno"),
                "name": r.get("prdt_name", ""),
                "qty": qty,
                "avg_price": float(r.get("pchs_avg_pric", 0) or 0),
                "current_price": float(r.get("prpr", 0) or 0),
            })
    except Exception as e:
        logger.error("fetch_balance 실패: %s", e)
        return 1

    if not holdings:
        logger.info("보유 종목 0건 — 평가 종료")
        return 0

    # 각 보유 종목 evaluate
    rebuy_signals = []
    cut_signals = []
    for h in holdings:
        try:
            result = evaluate_position(
                ticker=h["ticker"],
                name=h["name"],
                entry_price=h["avg_price"],
                current_price=h["current_price"],
            )
            if result["recommend"] == "REBUY":
                rebuy_signals.append(result)
            elif result["recommend"] in ("CUT_LOSS", "STOP_LOSS"):
                cut_signals.append(result)
            logger.info(
                "[counter_trade] %s %s — %s (%+.2f%%, score %d)",
                h["ticker"], h["name"], result["recommend"],
                result["decline_pct"], result["shake_out"]["score"],
            )
        except Exception as e:
            logger.warning("evaluate_position 실패 %s: %s", h["ticker"], e)

    # 텔레그램 알림 (워밍업 모드 — 신호만, 실제 매수 X)
    if rebuy_signals or cut_signals:
        warmup = os.environ.get("COUNTER_TRADE_WARMUP", "1") == "1"
        mode = "🟡 워밍업 (신호만)" if warmup else "🔴 정식 (자동 실행)"
        msg_lines = [
            f"📊 <b>역매수 모니터 [{datetime.now().strftime('%H:%M')}] — {mode}</b>",
            "",
        ]
        if rebuy_signals:
            msg_lines.append(f"<b>🔵 역매수 시그널 ({len(rebuy_signals)}건)</b>:")
            for r in rebuy_signals:
                msg_lines.append(r["action_text"])
                msg_lines.append("")
        if cut_signals:
            msg_lines.append(f"<b>🔴 손절 검토 ({len(cut_signals)}건)</b>:")
            for r in cut_signals:
                msg_lines.append(r["action_text"])
                msg_lines.append("")
        send_telegram("\n".join(msg_lines))

    logger.info(
        "counter_trade 모니터 완료: 보유 %d건, REBUY %d, CUT %d",
        len(holdings), len(rebuy_signals), len(cut_signals),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
