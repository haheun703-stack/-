"""
일일 매매일지 자동 업데이트

장마감 후 실행: KIS 잔고 스냅샷 → 전일 대비 변동 감지 → 매매 기록 저장.

저장 구조:
  data/trade_journal/
    snapshots/YYYY-MM-DD.json   — 일일 잔고 스냅샷
    trades.json                 — 전체 매매 이력 (누적)
    monthly/YYYY-MM.json        — 월별 요약

Usage:
    python scripts/update_trade_journal.py
    python scripts/update_trade_journal.py --date 2026-02-21
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

JOURNAL_DIR = PROJECT_ROOT / "data" / "trade_journal"
SNAPSHOT_DIR = JOURNAL_DIR / "snapshots"
TRADES_FILE = JOURNAL_DIR / "trades.json"
MONTHLY_DIR = JOURNAL_DIR / "monthly"


def fetch_kis_balance() -> dict:
    """KIS API에서 현재 잔고 조회."""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    kis = KisOrderAdapter()
    return kis.fetch_balance()


def save_snapshot(today: str, balance: dict) -> Path:
    """일일 잔고 스냅샷 저장."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "date": today,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "holdings": balance.get("holdings", []),
        "total_eval": balance.get("total_eval", 0),
        "total_pnl": balance.get("total_pnl", 0),
        "available_cash": balance.get("available_cash", 0),
    }

    path = SNAPSHOT_DIR / f"{today}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[스냅샷] 저장: %s (%d종목)", path.name, len(snapshot["holdings"]))
    return path


def load_snapshot(date_str: str) -> dict | None:
    """특정 날짜 스냅샷 로드."""
    path = SNAPSHOT_DIR / f"{date_str}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_previous_snapshot(today: str) -> dict | None:
    """오늘 이전 가장 최근 스냅샷 찾기."""
    if not SNAPSHOT_DIR.exists():
        return None

    files = sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True)
    for f in files:
        if f.stem < today:
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def detect_trades(prev: dict | None, curr: dict) -> list[dict]:
    """전일 vs 오늘 스냅샷 비교하여 매매 감지."""
    trades = []
    today = curr["date"]
    now = datetime.now().strftime("%H:%M:%S")

    prev_holdings = {}
    if prev:
        for h in prev.get("holdings", []):
            prev_holdings[h["ticker"]] = h

    curr_holdings = {}
    for h in curr.get("holdings", []):
        curr_holdings[h["ticker"]] = h

    # 신규 매수: 오늘 있지만 어제 없음
    for ticker, h in curr_holdings.items():
        if ticker not in prev_holdings:
            trades.append({
                "date": today,
                "time": now,
                "type": "BUY",
                "ticker": h["ticker"],
                "name": h["name"],
                "quantity": h["quantity"],
                "price": round(h["avg_price"]),
                "amount": round(h["avg_price"] * h["quantity"]),
                "note": "신규 매수 감지",
            })
        else:
            # 추가 매수: 수량 증가
            prev_qty = prev_holdings[ticker]["quantity"]
            if h["quantity"] > prev_qty:
                added = h["quantity"] - prev_qty
                trades.append({
                    "date": today,
                    "time": now,
                    "type": "BUY",
                    "ticker": h["ticker"],
                    "name": h["name"],
                    "quantity": added,
                    "price": round(h["avg_price"]),
                    "amount": round(h["avg_price"] * added),
                    "note": f"추가 매수 ({prev_qty} -> {h['quantity']}주)",
                })
            # 부분 매도: 수량 감소
            elif h["quantity"] < prev_qty:
                sold = prev_qty - h["quantity"]
                sell_price = h["current_price"]
                buy_price = round(prev_holdings[ticker]["avg_price"])
                pnl = round((sell_price - buy_price) * sold)
                pnl_pct = round((sell_price / buy_price - 1) * 100, 2) if buy_price > 0 else 0
                trades.append({
                    "date": today,
                    "time": now,
                    "type": "SELL",
                    "ticker": h["ticker"],
                    "name": h["name"],
                    "quantity": sold,
                    "price": sell_price,
                    "amount": sell_price * sold,
                    "buy_price": buy_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "note": f"부분 매도 ({prev_qty} -> {h['quantity']}주)",
                })

    # 전량 매도: 어제 있지만 오늘 없음
    for ticker, h in prev_holdings.items():
        if ticker not in curr_holdings:
            # 매도 가격은 알 수 없으므로 전일 현재가 사용
            sell_price = h.get("current_price", 0)
            buy_price = round(h.get("avg_price", 0))
            pnl = round((sell_price - buy_price) * h["quantity"]) if sell_price > 0 else 0
            pnl_pct = round((sell_price / buy_price - 1) * 100, 2) if buy_price > 0 else 0
            trades.append({
                "date": today,
                "time": now,
                "type": "SELL",
                "ticker": h["ticker"],
                "name": h["name"],
                "quantity": h["quantity"],
                "price": sell_price,
                "amount": sell_price * h["quantity"],
                "buy_price": buy_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "note": "전량 매도 감지",
            })

    return trades


def load_all_trades() -> list[dict]:
    """전체 매매 이력 로드."""
    if TRADES_FILE.exists():
        return json.loads(TRADES_FILE.read_text(encoding="utf-8"))
    return []


def save_trades(trades: list[dict]) -> None:
    """전체 매매 이력 저장."""
    TRADES_FILE.write_text(
        json.dumps(trades, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_monthly_summary(today: str, all_trades: list[dict]) -> None:
    """월별 요약 업데이트."""
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    month = today[:7]  # YYYY-MM

    month_trades = [t for t in all_trades if t["date"].startswith(month)]
    buys = [t for t in month_trades if t["type"] == "BUY"]
    sells = [t for t in month_trades if t["type"] == "SELL"]

    total_pnl = sum(t.get("pnl", 0) for t in sells)
    win_trades = [t for t in sells if t.get("pnl", 0) > 0]
    lose_trades = [t for t in sells if t.get("pnl", 0) < 0]

    summary = {
        "month": month,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "total_pnl": total_pnl,
        "win_count": len(win_trades),
        "lose_count": len(lose_trades),
        "win_rate": round(len(win_trades) / len(sells) * 100, 1) if sells else 0,
        "avg_pnl_pct": round(
            sum(t.get("pnl_pct", 0) for t in sells) / len(sells), 2
        ) if sells else 0,
        "best_trade": max(sells, key=lambda t: t.get("pnl", 0)) if sells else None,
        "worst_trade": min(sells, key=lambda t: t.get("pnl", 0)) if sells else None,
        "trades": month_trades,
    }

    path = MONTHLY_DIR / f"{month}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[월별] %s 저장: 매수 %d건 매도 %d건 손익 %+d원", month, len(buys), len(sells), total_pnl)


def print_journal(today: str, snapshot: dict, new_trades: list[dict], all_trades: list[dict]):
    """터미널 일지 출력."""
    print(f"\n{'='*50}")
    print(f"  일일 매매일지 — {today}")
    print(f"{'='*50}")

    # 포트폴리오 현황
    holdings = snapshot.get("holdings", [])
    total_eval = snapshot.get("total_eval", 0)
    total_pnl = snapshot.get("total_pnl", 0)
    cash = snapshot.get("available_cash", 0)

    print(f"\n[포트폴리오]")
    print(f"  총평가: {total_eval:>15,}원")
    print(f"  손  익: {total_pnl:>+15,}원")
    print(f"  예수금: {cash:>15,}원")
    print(f"  종  목: {len(holdings)}개")

    if holdings:
        print(f"\n  {'종목명':<12} {'수량':>6} {'평단가':>10} {'현재가':>10} {'수익률':>8} {'손익':>12}")
        print(f"  {'-'*62}")
        for h in holdings:
            print(
                f"  {h['name']:<12} {h['quantity']:>6}주 "
                f"{round(h['avg_price']):>10,} {h['current_price']:>10,} "
                f"{h['pnl_pct']:>+7.1f}% {h['pnl_amount']:>+12,}"
            )

    # 오늘 매매
    if new_trades:
        print(f"\n[오늘 매매] {len(new_trades)}건")
        for t in new_trades:
            emoji = "BUY " if t["type"] == "BUY" else "SELL"
            pnl_str = f" | 손익 {t.get('pnl', 0):+,}원 ({t.get('pnl_pct', 0):+.1f}%)" if t["type"] == "SELL" else ""
            print(f"  [{emoji}] {t['name']} {t['quantity']}주 @ {t['price']:,}원{pnl_str}")
    else:
        print(f"\n[오늘 매매] 없음")

    # 월간 통계
    month = today[:7]
    month_sells = [t for t in all_trades if t["date"].startswith(month) and t["type"] == "SELL"]
    if month_sells:
        month_pnl = sum(t.get("pnl", 0) for t in month_sells)
        wins = sum(1 for t in month_sells if t.get("pnl", 0) > 0)
        print(f"\n[{month} 월간] 매도 {len(month_sells)}건 | 승률 {wins}/{len(month_sells)} | 손익 {month_pnl:+,}원")

    print(f"\n{'='*50}\n")


def add_manual_trade(
    trade_type: str,
    name: str,
    ticker: str,
    quantity: int,
    price: int,
    trade_date: str,
    trade_time: str = "09:00:00",
    buy_price: int = 0,
    note: str = "",
) -> None:
    """수동 매매 기록 추가."""
    trade = {
        "date": trade_date,
        "time": trade_time,
        "type": trade_type.upper(),
        "ticker": ticker,
        "name": name,
        "quantity": quantity,
        "price": price,
        "amount": price * quantity,
        "note": note or f"수동 등록",
    }

    if trade_type.upper() == "SELL" and buy_price > 0:
        pnl = round((price - buy_price) * quantity)
        pnl_pct = round((price / buy_price - 1) * 100, 2) if buy_price > 0 else 0
        trade["buy_price"] = buy_price
        trade["pnl"] = pnl
        trade["pnl_pct"] = pnl_pct

    all_trades = load_all_trades()
    all_trades.append(trade)
    all_trades.sort(key=lambda t: (t["date"], t["time"]))
    save_trades(all_trades)

    # 월별 요약 갱신
    update_monthly_summary(trade_date, all_trades)

    pnl_str = ""
    if trade_type.upper() == "SELL" and buy_price > 0:
        pnl_str = f" | 손익 {trade['pnl']:+,}원 ({trade['pnl_pct']:+.1f}%)"
    logger.info(
        "[수동등록] %s %s %s %d주 @ %d원%s",
        trade_date, trade_type, name, quantity, price, pnl_str,
    )
    print(f"  등록 완료: [{trade_type}] {name} {quantity}주 @ {price:,}원{pnl_str}")


def cmd_add(args):
    """수동 매매 등록 서브커맨드."""
    add_manual_trade(
        trade_type=args.type,
        name=args.name,
        ticker=args.ticker,
        quantity=args.qty,
        price=args.price,
        trade_date=args.date,
        trade_time=args.time or "09:00:00",
        buy_price=args.buy_price or 0,
        note=args.note or "",
    )


def cmd_update(args):
    """일일 업데이트 서브커맨드."""
    today = args.date or date.today().isoformat()
    logger.info("[매매일지] %s 업데이트 시작", today)

    # 1. KIS 잔고 조회
    if not os.getenv("KIS_APP_KEY"):
        logger.error("KIS API 키 미설정")
        return

    balance = fetch_kis_balance()
    if not balance.get("holdings") and balance.get("total_eval", 0) == 0:
        logger.warning("잔고 조회 결과 없음")

    # 2. 스냅샷 저장
    snapshot = {
        "date": today,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "holdings": balance.get("holdings", []),
        "total_eval": balance.get("total_eval", 0),
        "total_pnl": balance.get("total_pnl", 0),
        "available_cash": balance.get("available_cash", 0),
    }
    save_snapshot(today, balance)

    # 3. 전일 스냅샷과 비교 → 매매 감지
    prev = find_previous_snapshot(today)
    new_trades = detect_trades(prev, snapshot)

    # 4. 매매 이력 업데이트
    all_trades = load_all_trades()
    # 오늘 날짜 기존 자동감지 기록 제거 (중복 방지, 수동 등록은 유지)
    all_trades = [t for t in all_trades if not (t["date"] == today and "수동" not in t.get("note", ""))]
    all_trades.extend(new_trades)
    all_trades.sort(key=lambda t: (t["date"], t["time"]))
    save_trades(all_trades)

    if new_trades:
        logger.info("[매매] %d건 감지: %s", len(new_trades),
                     ", ".join(f"{t['type']} {t['name']}" for t in new_trades))
    else:
        logger.info("[매매] 오늘 변동 없음")

    # 5. 월별 요약
    update_monthly_summary(today, all_trades)

    # 6. 터미널 출력
    print_journal(today, snapshot, new_trades, all_trades)

    logger.info("[매매일지] 완료")


def cmd_list(args):
    """매매 이력 조회 서브커맨드."""
    all_trades = load_all_trades()
    if args.month:
        all_trades = [t for t in all_trades if t["date"].startswith(args.month)]

    if not all_trades:
        print("매매 기록 없음")
        return

    print(f"\n{'='*70}")
    print(f"  매매 이력 ({len(all_trades)}건)")
    print(f"{'='*70}")
    for t in all_trades:
        pnl_str = ""
        if t["type"] == "SELL" and "pnl" in t:
            pnl_str = f" | 손익 {t['pnl']:+,}원 ({t.get('pnl_pct', 0):+.1f}%)"
        print(f"  {t['date']} {t['time']} [{t['type']:4s}] {t['name']:<12s} {t['quantity']:>5d}주 @ {t['price']:>10,}원{pnl_str}")
    print()


def main():
    parser = argparse.ArgumentParser(description="일일 매매일지 관리")
    sub = parser.add_subparsers(dest="command")

    # update (기본: 인자 없이 실행)
    p_update = sub.add_parser("update", help="일일 업데이트 (KIS 잔고 스냅샷)")
    p_update.add_argument("--date", default=None, help="날짜 (YYYY-MM-DD)")

    # add: 수동 매매 등록
    p_add = sub.add_parser("add", help="수동 매매 등록")
    p_add.add_argument("type", choices=["BUY", "SELL", "buy", "sell"], help="매수/매도")
    p_add.add_argument("name", help="종목명")
    p_add.add_argument("ticker", help="종목코드 (6자리)")
    p_add.add_argument("qty", type=int, help="수량")
    p_add.add_argument("price", type=int, help="체결가")
    p_add.add_argument("--date", required=True, help="체결일 (YYYY-MM-DD)")
    p_add.add_argument("--time", default="09:00:00", help="체결시간 (HH:MM:SS)")
    p_add.add_argument("--buy-price", type=int, default=0, help="매수단가 (매도 시)")
    p_add.add_argument("--note", default="", help="비고")

    # list: 이력 조회
    p_list = sub.add_parser("list", help="매매 이력 조회")
    p_list.add_argument("--month", default=None, help="월 필터 (YYYY-MM)")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "update":
        cmd_update(args)
    else:
        # 인자 없이 실행 시 기본 update
        args.date = None
        cmd_update(args)


if __name__ == "__main__":
    main()
