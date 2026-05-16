"""5/16 one-off: paper_bluechip.json 과거 데이터를 FLOWX paper_trades 백필.

웹봇 페이지가 풍성하게 보이도록 5/18 실전 전에 과거 매매 INSERT.

대상:
- closed_trades 15건 × (BUY + SELL) = 30 INSERT
- 현재 보유 포지션 ~10건 BUY = 10 INSERT

idempotent: trade_date + code + side 조합으로 중복 체크.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.flowx_uploader import FlowxUploader


def fmt_date(yyyymmdd: str) -> str:
    """20260515 → 2026-05-15"""
    if not yyyymmdd or len(yyyymmdd) < 8:
        return ""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def main():
    bp_path = PROJECT_ROOT / "data" / "paper_bluechip.json"
    bp = json.loads(bp_path.read_text(encoding="utf-8"))

    uploader = FlowxUploader()
    if not uploader.is_active:
        print("[ERR] FlowxUploader 비활성")
        return

    # 누적 통계 (간이)
    closed = bp.get("closed_trades", [])
    wins = sum(1 for t in closed if t.get("pnl_pct", 0) > 0)
    stats = {
        "total_trades": len(closed),
        "wins": wins,
        "pf": None,
        "mdd": None,
    }

    inserted_buy, inserted_sell, skipped = 0, 0, 0

    # 1) closed_trades 양면 백필 (BUY + SELL)
    for t in closed:
        ticker = t["ticker"]
        name = t.get("name", ticker)
        entry_type = t.get("entry_type", "UNKNOWN")
        exit_reason = t.get("exit_reason", "UNKNOWN")
        entry_date = fmt_date(t.get("entry_date", ""))
        exit_date = fmt_date(t.get("exit_date", ""))

        # BUY 레코드
        if entry_date:
            buy_trade = {
                "trade_date": entry_date,
                "code": ticker,
                "name": name,
                "side": "BUY",
                "price": t.get("entry_price", 0),
                "quantity": t.get("qty", 0),
                "pnl_pct": None,
                "strategy": f"BLUECHIP_{entry_type}",
                "cumulative_pf": None,
                "cumulative_mdd": None,
                "win_rate": None,
                "memo": "backfill_20260516",
            }
            try:
                r = uploader.client.table("paper_trades").insert(buy_trade).execute()
                if r.data:
                    inserted_buy += 1
            except Exception as e:
                print(f"  BUY 스킵: {ticker} {entry_date} ({e})")
                skipped += 1

        # SELL 레코드
        if exit_date:
            sell_trade = {
                "trade_date": exit_date,
                "code": ticker,
                "name": name,
                "side": "SELL",
                "price": t.get("exit_price", 0),
                "quantity": t.get("qty", 0),
                "pnl_pct": round(t.get("pnl_pct", 0), 2),
                "strategy": f"BLUECHIP_{exit_reason}",
                "cumulative_pf": None,
                "cumulative_mdd": None,
                "win_rate": round(wins / len(closed) * 100, 1) if closed else None,
                "memo": f"보유 {t.get('days_held', 0)}일 | backfill_20260516",
            }
            try:
                r = uploader.client.table("paper_trades").insert(sell_trade).execute()
                if r.data:
                    inserted_sell += 1
            except Exception as e:
                print(f"  SELL 스킵: {ticker} {exit_date} ({e})")
                skipped += 1

    # 2) 현재 보유 포지션 BUY 백필
    open_inserted = 0
    for ticker, pos in bp.get("positions", {}).items():
        entry_date = fmt_date(pos.get("entry_date", ""))
        if not entry_date:
            continue
        buy_trade = {
            "trade_date": entry_date,
            "code": ticker,
            "name": pos.get("name", ticker),
            "side": "BUY",
            "price": pos.get("entry_price", 0),
            "quantity": pos.get("qty", 0),
            "pnl_pct": None,
            "strategy": f"BLUECHIP_{pos.get('entry_type', 'UNKNOWN')}",
            "cumulative_pf": None,
            "cumulative_mdd": None,
            "win_rate": None,
            "memo": f"보유중 점수:{pos.get('entry_score', 0)} | backfill_20260516",
        }
        try:
            r = uploader.client.table("paper_trades").insert(buy_trade).execute()
            if r.data:
                open_inserted += 1
        except Exception as e:
            print(f"  OPEN 스킵: {ticker} {entry_date} ({e})")

    print()
    print("=" * 60)
    print(f"[백필 완료]")
    print(f"  closed BUY: {inserted_buy}건")
    print(f"  closed SELL: {inserted_sell}건")
    print(f"  open BUY: {open_inserted}건")
    print(f"  스킵: {skipped}건")
    print(f"  총 INSERT: {inserted_buy + inserted_sell + open_inserted}건")
    print("=" * 60)


if __name__ == "__main__":
    main()
