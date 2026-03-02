"""장중 AI 매도 모니터 — sell_brain + 자동 매도 실행

BAT-E 이후 09:30~14:50까지 주기적으로 실행:
  - 10:00, 12:00, 14:00: sell_brain 장중 체크
  - 14:30: preclose 체크
  - SELL_NOW → 즉시 시장가 매도
  - PARTIAL_SELL → 50% 시장가 매도
  - HOLD/WATCH → 유지

안전장치:
  - data/KILL_SWITCH 존재 시 중단
  - AI 장애 시 전종목 HOLD (매도 안 함)
  - 텔레그램 알림 (매도 실행 / 판단 변경)

사용법:
  python scripts/sell_monitor.py              # 풀 세션 (09:30~14:50)
  python scripts/sell_monitor.py --once       # 1회 체크 후 종료
  python scripts/sell_monitor.py --dry-run    # 매도 없이 판단만
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sell_monitor")

ROOT = Path(__file__).resolve().parent.parent
KILL_SWITCH = ROOT / "data" / "KILL_SWITCH"

# 체크 스케줄 (장중)
CHECK_TIMES = ["10:00", "12:00", "14:00"]
PRECLOSE_TIME = "14:30"


def load_json(name: str, default=None):
    p = ROOT / "data" / name
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default or {}


def save_json(name: str, data: dict):
    p = ROOT / "data" / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_holdings() -> list[dict]:
    """KIS API에서 현재 보유 종목 조회 → sell_brain 입력 형식으로 변환"""
    import mojito

    broker = mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY"),
        api_secret=os.getenv("KIS_APP_SECRET"),
        acc_no=os.getenv("KIS_ACC_NO"),
        mock=os.getenv("MODEL") != "REAL",
    )

    resp = broker.fetch_balance()
    holdings = resp.get("output1", [])

    positions = []
    for h in holdings:
        qty = int(h.get("hldg_qty", 0))
        if qty == 0:
            continue

        ticker = h.get("pdno", "")
        name = h.get("prdt_name", "?")
        avg_price = float(h.get("pchs_avg_pric", 0))
        curr_price = int(h.get("prpr", 0))
        pnl_pct = float(h.get("evlu_pfls_rt", 0))

        positions.append({
            "ticker": ticker,
            "name": name,
            "entry_price": int(avg_price),
            "current_price": curr_price,
            "quantity": qty,
            "pnl_pct": pnl_pct,
            "hold_days": "?",
            "grade": "수동매수",
            "trigger_type": "manual",
            "stop_loss": int(avg_price * 0.92),  # 기본 -8%
        })

    return positions, broker


def execute_sell(broker, ticker: str, name: str, qty: int, sell_type: str, dry_run: bool) -> dict:
    """시장가 매도 실행"""
    if sell_type == "PARTIAL_SELL":
        qty = max(1, qty // 2)  # 50% 매도

    if dry_run:
        logger.info("[DRY-RUN] 매도 스킵: %s %d주", name, qty)
        return {"status": "dry_run", "ticker": ticker, "qty": qty}

    try:
        resp = broker.create_market_sell_order(ticker, qty)
        rt_cd = resp.get("rt_cd", "?")
        msg = resp.get("msg1", "")
        odno = resp.get("output", {}).get("ODNO", "")

        if rt_cd == "0":
            logger.info("[매도 성공] %s %d주, 주문번호=%s", name, qty, odno)
            return {"status": "ok", "ticker": ticker, "qty": qty, "order_no": odno}
        else:
            logger.error("[매도 실패] %s: %s", name, msg)
            return {"status": "failed", "ticker": ticker, "msg": msg}
    except Exception as e:
        logger.error("[매도 에러] %s: %s", name, e)
        return {"status": "error", "ticker": ticker, "msg": str(e)}


def send_telegram(text: str):
    """텔레그램 알림"""
    try:
        from src.telegram_sender import send_message
        send_message(text)
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


async def run_check(positions: list[dict], broker, check_type: str, dry_run: bool) -> dict:
    """1회 매도 체크 + 실행"""
    from src.agents.sell_brain import SellBrainAgent

    strategic = load_json("ai_strategic_analysis.json")
    sector_focus = load_json("ai_sector_focus.json")

    agent = SellBrainAgent()

    if check_type == "preclose":
        overnight = load_json("overnight_signal.json")
        result = await agent.preclose_check(positions, strategic, overnight)
    else:
        result = await agent.check_sell(positions, strategic, sector_focus)

    # 결과 저장
    save_json("ai_sell_cache.json", result)

    # 매도 실행
    sells_executed = []
    for p in result.get("positions", []):
        action = p.get("action", "HOLD")
        ticker = p.get("ticker", "")
        name = p.get("name", "")
        reasoning = p.get("reasoning", "")

        if action in ("SELL_NOW", "PARTIAL_SELL"):
            # 해당 종목의 수량 찾기
            pos = next((x for x in positions if x["ticker"] == ticker), None)
            if pos is None:
                continue

            qty = pos["quantity"]
            pnl = pos.get("pnl_pct", 0)

            logger.info("[매도 결정] %s → %s (사유: %s)", name, action, reasoning[:50])

            sell_result = execute_sell(broker, ticker, name, qty, action, dry_run)
            sells_executed.append({
                "ticker": ticker,
                "name": name,
                "action": action,
                "qty": qty if action == "SELL_NOW" else max(1, qty // 2),
                "pnl_pct": pnl,
                "reasoning": reasoning,
                "result": sell_result,
            })

    # 텔레그램 알림
    if sells_executed:
        lines = [f"🔴 AI 매도 실행 ({check_type})"]
        for s in sells_executed:
            mode = "DRY" if dry_run else "LIVE"
            status = s["result"].get("status", "?")
            lines.append(f"  {s['name']}: {s['action']} {s['qty']}주 [{status}]")
            lines.append(f"    손익 {s['pnl_pct']:+.1f}% | {s['reasoning'][:40]}")
        send_telegram("\n".join(lines))
    else:
        # 매도 없으면 요약만
        hold_count = sum(1 for p in result.get("positions", []) if p.get("action") == "HOLD")
        watch_count = sum(1 for p in result.get("positions", []) if p.get("action") == "WATCH")
        if check_type == "preclose":
            send_telegram(f"🟡 프리클로즈 체크: HOLD {hold_count} / WATCH {watch_count}")

    return {
        "check_type": check_type,
        "total": len(positions),
        "sells": len(sells_executed),
        "details": sells_executed,
    }


def main():
    parser = argparse.ArgumentParser(description="장중 AI 매도 모니터")
    parser.add_argument("--once", action="store_true", help="1회 체크 후 종료")
    parser.add_argument("--dry-run", action="store_true", help="매도 없이 판단만")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("  Sell Monitor 시작 (%s)", "DRY-RUN" if args.dry_run else "LIVE")
    logger.info("=" * 50)

    # KILL_SWITCH 체크
    if KILL_SWITCH.exists():
        logger.warning("KILL_SWITCH 존재 → 중단")
        return

    # 보유 종목 조회
    positions, broker = fetch_holdings()
    if not positions:
        logger.info("보유 종목 없음 → 종료")
        return

    logger.info("보유 종목: %d개", len(positions))
    for p in positions:
        logger.info("  %s(%s): %+.1f%%", p["name"], p["ticker"], p["pnl_pct"])

    if args.once:
        # 1회 체크
        now = datetime.now().strftime("%H:%M")
        check_type = "preclose" if now >= "14:20" else "intraday"
        result = asyncio.run(run_check(positions, broker, check_type, args.dry_run))
        logger.info("결과: %d종목 중 %d매도", result["total"], result["sells"])
        return

    # 풀 세션: 스케줄대로 체크
    checked = set()
    logger.info("풀 세션 모드: %s + %s 대기 중...", CHECK_TIMES, PRECLOSE_TIME)

    while True:
        if KILL_SWITCH.exists():
            logger.warning("KILL_SWITCH 감지 → 중단")
            send_telegram("🛑 Sell Monitor: KILL_SWITCH로 중단됨")
            break

        now = datetime.now()
        now_str = now.strftime("%H:%M")

        # 14:50 이후 종료
        if now_str >= "14:50":
            logger.info("14:50 → 세션 종료")
            break

        # 체크 시간 확인
        for ct in CHECK_TIMES:
            if ct not in checked and now_str >= ct:
                logger.info("=== %s 장중 체크 ===", ct)
                # 최신 잔고 다시 조회
                positions, broker = fetch_holdings()
                if positions:
                    result = asyncio.run(run_check(positions, broker, "intraday", args.dry_run))
                    logger.info("결과: %d종목 중 %d매도", result["total"], result["sells"])
                checked.add(ct)

        # 프리클로즈
        if PRECLOSE_TIME not in checked and now_str >= PRECLOSE_TIME:
            logger.info("=== %s 프리클로즈 체크 ===", PRECLOSE_TIME)
            positions, broker = fetch_holdings()
            if positions:
                result = asyncio.run(run_check(positions, broker, "preclose", args.dry_run))
                logger.info("프리클로즈: %d종목 중 %d매도", result["total"], result["sells"])
            checked.add(PRECLOSE_TIME)

        # 30초 대기
        time.sleep(30)


if __name__ == "__main__":
    main()
