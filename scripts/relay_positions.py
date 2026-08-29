"""릴레이 포지션 관리 — 진입/청산/조회/이력.

relay_stock_picker → relay_sizer → [진입] → relay_positions
relay_exit → [청산] → relay_positions

저장 파일:
  data/sector_rotation/relay_positions.json   — 현재 보유 포지션
  data/sector_rotation/relay_history.json     — 청산 이력 (누적)

사용법:
  python scripts/relay_positions.py list                      # 보유 포지션 조회
  python scripts/relay_positions.py add 088350 한화생명 생명보험 4850 600 --fired 증권 --win-rate 70.6 --lag 1
  python scripts/relay_positions.py check                     # 청산 조건 일괄 체크
  python scripts/relay_positions.py close 088350 5100         # 수동 청산
  python scripts/relay_positions.py history                   # 청산 이력
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"
POSITIONS_FILE = DATA_DIR / "relay_positions.json"
HISTORY_FILE = DATA_DIR / "relay_history.json"

from relay_exit import check_exit_conditions, get_profit_target, get_sector_today_return


# ─────────────────────────────────────────────
# 포지션 파일 I/O
# ─────────────────────────────────────────────

def load_positions() -> list[dict]:
    """현재 보유 포지션 로드."""
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("positions", [])


def save_positions(positions: list[dict]):
    """보유 포지션 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "count": len(positions),
        "positions": positions,
    }
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_history() -> list[dict]:
    """청산 이력 로드."""
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", [])


def save_history(history: list[dict]):
    """청산 이력 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 통계 계산
    pnls = [h["pnl_pct"] for h in history if "pnl_pct" in h]
    stats = {}
    if pnls:
        wins = [p for p in pnls if p > 0]
        stats = {
            "total_trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1),
            "avg_pnl": round(float(np.mean(pnls)), 2),
            "best": round(max(pnls), 2),
            "worst": round(min(pnls), 2),
        }
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "count": len(history),
        "stats": stats,
        "history": history,
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 포지션 조작
# ─────────────────────────────────────────────

def add_position(
    ticker: str,
    name: str,
    sector: str,
    entry_price: float,
    quantity: int,
    fired_sector: str = "",
    win_rate: float = 0,
    best_lag: int = 1,
    confidence: str = "HIGH",
    weight_pct: float = 0,
) -> dict:
    """포지션 추가 (매수 진입)."""
    positions = load_positions()

    # 이미 보유 중인지 확인
    for p in positions:
        if p["ticker"] == ticker:
            print(f"  이미 보유 중: {name}({ticker}) — 추가 매수 불가")
            return p

    target_pct = get_profit_target(win_rate)

    new_pos = {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "entry_price": entry_price,
        "quantity": quantity,
        "investment": int(entry_price * quantity),
        "fired_sector": fired_sector,
        "win_rate": win_rate,
        "best_lag": best_lag,
        "confidence": confidence,
        "target_pct": target_pct,
        "timeout_days": best_lag + 1,
        "weight_pct": weight_pct,
        "trading_days_held": 0,
    }

    positions.append(new_pos)
    save_positions(positions)
    print(f"  진입: {name}({ticker}) {entry_price:,}원 x {quantity}주 "
          f"= {new_pos['investment']:,}원")
    print(f"  패턴: {fired_sector}→{sector} 래그{best_lag}일 승률{win_rate:.0f}%")
    print(f"  목표: +{target_pct}%, 타임아웃: {new_pos['timeout_days']}일")
    return new_pos


def close_position(
    ticker: str,
    exit_price: float,
    reason: str = "수동 청산",
    action: str = "MANUAL",
) -> dict | None:
    """포지션 청산."""
    positions = load_positions()
    history = load_history()

    target = None
    remaining = []
    for p in positions:
        if p["ticker"] == ticker:
            target = p
        else:
            remaining.append(p)

    if not target:
        print(f"  {ticker}: 보유 포지션 없음")
        return None

    pnl_pct = (exit_price - target["entry_price"]) / target["entry_price"] * 100
    pnl_amount = int((exit_price - target["entry_price"]) * target["quantity"])

    record = {
        **target,
        "exit_date": datetime.now().strftime("%Y-%m-%d"),
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_amount": pnl_amount,
        "exit_reason": reason,
        "exit_action": action,
    }

    history.append(record)
    save_positions(remaining)
    save_history(history)

    win_str = "수익" if pnl_pct > 0 else "손실"
    print(f"  청산: {target['name']}({ticker}) {exit_price:,}원")
    print(f"  {win_str}: {pnl_pct:+.1f}% ({pnl_amount:+,}원)")
    print(f"  사유: {reason}")
    return record


# ─────────────────────────────────────────────
# 일괄 청산 체크
# ─────────────────────────────────────────────

def get_current_price(ticker: str) -> float:
    """종목 최신 종가 조회."""
    matches = list(DAILY_DIR.glob(f"*_{ticker}.csv"))
    if not matches:
        return 0
    try:
        df = pd.read_csv(matches[0], usecols=["Date", "Close"])
        df = df.dropna().sort_values("Date")
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0


def check_all_positions() -> list[dict]:
    """모든 보유 포지션 청산 조건 일괄 체크."""
    positions = load_positions()
    if not positions:
        print("  보유 포지션 없음")
        return []

    results = []
    updated_positions = []

    for pos in positions:
        current_price = get_current_price(pos["ticker"])
        if current_price <= 0:
            updated_positions.append(pos)
            continue

        # 보유일 +1
        pos["trading_days_held"] = pos.get("trading_days_held", 0) + 1

        # 후행 섹터 현재 등락률
        follow_ret, follow_breadth = get_sector_today_return(pos["sector"])

        # 청산 조건 체크
        exit_result = check_exit_conditions(
            entry_price=pos["entry_price"],
            current_price=current_price,
            entry_date=pos["entry_date"],
            current_date=datetime.now().strftime("%Y-%m-%d"),
            win_rate=pos["win_rate"],
            best_lag=pos["best_lag"],
            follow_sector_return=follow_ret,
            follow_sector_breadth=follow_breadth,
            trading_days_held=pos["trading_days_held"],
        )

        result = {
            "ticker": pos["ticker"],
            "name": pos["name"],
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "pnl_pct": exit_result["pnl_pct"],
            "days_held": pos["trading_days_held"],
            "exit": exit_result["exit"],
            "action": exit_result["action"],
            "reason": exit_result["reason"],
        }
        results.append(result)

        if exit_result["exit"]:
            # 자동 청산
            close_position(
                pos["ticker"], current_price,
                reason=exit_result["reason"],
                action=exit_result["action"],
            )
        else:
            updated_positions.append(pos)

    # 보유일 업데이트 저장
    save_positions(updated_positions)
    return results


# ─────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────

def print_positions():
    """보유 포지션 출력."""
    positions = load_positions()

    print(f"\n{'=' * 65}")
    print(f"  릴레이 보유 포지션 — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'=' * 65}")

    if not positions:
        print("  보유 포지션 없음")
        print(f"{'=' * 65}\n")
        return

    total_invest = 0
    total_pnl = 0

    print(f"  {'종목':<10} {'코드':>8} {'진입가':>8} {'현재가':>8} "
          f"{'수익률':>7} {'보유일':>4} {'패턴'}")
    print(f"  {'─' * 60}")

    for pos in positions:
        current = get_current_price(pos["ticker"])
        pnl_pct = (current - pos["entry_price"]) / pos["entry_price"] * 100 if current > 0 else 0
        pnl_amount = int((current - pos["entry_price"]) * pos["quantity"]) if current > 0 else 0
        total_invest += pos["investment"]
        total_pnl += pnl_amount

        print(f"  {pos['name']:<10} ({pos['ticker']}) "
              f"{pos['entry_price']:>7,} {current:>7,} "
              f"{pnl_pct:>+6.1f}% "
              f"{pos.get('trading_days_held', 0):>3}일 "
              f"{pos['fired_sector']}→{pos['sector']}")

    if total_invest > 0:
        print(f"\n  투입: {total_invest:,}원 | 평가손익: {total_pnl:+,}원 "
              f"({total_pnl/total_invest*100:+.1f}%)")
    print(f"{'=' * 65}\n")


def print_history():
    """청산 이력 출력."""
    history = load_history()

    print(f"\n{'=' * 65}")
    print(f"  릴레이 청산 이력")
    print(f"{'=' * 65}")

    if not history:
        print("  청산 이력 없음")
        print(f"{'=' * 65}\n")
        return

    print(f"  {'종목':<10} {'진입일':>10} {'청산일':>10} {'수익률':>7} {'사유'}")
    print(f"  {'─' * 60}")

    for h in history[-20:]:  # 최근 20건
        icon = "+" if h["pnl_pct"] > 0 else ""
        print(f"  {h['name']:<10} {h['entry_date']:>10} {h['exit_date']:>10} "
              f"{icon}{h['pnl_pct']:.1f}%  {h['exit_action']}")

    # 통계
    pnls = [h["pnl_pct"] for h in history]
    wins = [p for p in pnls if p > 0]
    if pnls:
        print(f"\n  총 {len(pnls)}건 | 승률 {len(wins)/len(pnls)*100:.0f}% | "
              f"평균 {np.mean(pnls):+.1f}% | "
              f"최고 {max(pnls):+.1f}% | 최저 {min(pnls):+.1f}%")
    print(f"{'=' * 65}\n")


def print_check_results(results: list[dict]):
    """일괄 체크 결과 출력."""
    print(f"\n{'=' * 65}")
    print(f"  릴레이 청산 체크 — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'=' * 65}")

    for r in results:
        icon = "EXIT" if r["exit"] else "HOLD"
        print(f"  [{icon}] {r['name']}({r['ticker']}) "
              f"{r['pnl_pct']:+.1f}% {r['days_held']}일 — {r['reason']}")

    exits = [r for r in results if r["exit"]]
    holds = [r for r in results if not r["exit"]]
    print(f"\n  청산: {len(exits)}건, 보유유지: {len(holds)}건")
    print(f"{'=' * 65}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="릴레이 포지션 관리")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="보유 포지션 조회")

    # add
    add_p = sub.add_parser("add", help="포지션 추가")
    add_p.add_argument("ticker", help="종목 코드")
    add_p.add_argument("name", help="종목명")
    add_p.add_argument("sector", help="후행 섹터")
    add_p.add_argument("price", type=float, help="진입가")
    add_p.add_argument("qty", type=int, help="수량")
    add_p.add_argument("--fired", default="", help="발화 섹터")
    add_p.add_argument("--win-rate", type=float, default=0, help="승률")
    add_p.add_argument("--lag", type=int, default=1, help="래그")
    add_p.add_argument("--confidence", default="HIGH", help="신뢰도")

    # check
    sub.add_parser("check", help="청산 조건 일괄 체크")

    # close
    close_p = sub.add_parser("close", help="수동 청산")
    close_p.add_argument("ticker", help="종목 코드")
    close_p.add_argument("price", type=float, help="청산가")

    # history
    sub.add_parser("history", help="청산 이력")

    args = parser.parse_args()

    if args.command == "list":
        print_positions()
    elif args.command == "add":
        add_position(
            args.ticker, args.name, args.sector,
            args.price, args.qty,
            fired_sector=args.fired,
            win_rate=args.win_rate,
            best_lag=args.lag,
            confidence=args.confidence,
        )
    elif args.command == "check":
        results = check_all_positions()
        if results:
            print_check_results(results)
    elif args.command == "close":
        close_position(args.ticker, args.price)
    elif args.command == "history":
        print_history()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
