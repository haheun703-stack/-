"""FLOWX STEP 3 — 단타봇 일괄 청산.

장마감 직전(15:20) DAYTRADING OPEN 시그널을 전부 종료 처리.
단타는 당일 청산이 원칙 — 익일 보유 없음.

종료 로직:
  1. Supabase에서 DAYTRADING + OPEN 시그널 조회
  2. parquet 최신 종가로 return_pct 계산
  3. return_pct > 0 → CLOSED (close_reason='EOD_PROFIT')
     return_pct <= 0 → STOPPED (close_reason='EOD_LOSS')
  4. 성적표 집계 (DAYTRADING 전용)

실행 시점: 매일 15:20 KST

Usage:
    python scripts/daily_close_daytrading.py
    python scripts/daily_close_daytrading.py --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PERIODS = {"30D": 30, "60D": 60, "90D": 90, "ALL": 3650}


def _get_latest_close(ticker: str) -> int:
    """parquet에서 최신 종가."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq, columns=["close"])
            if len(df) > 0:
                return int(df.iloc[-1]["close"])
        except Exception:
            pass
    return 0


def close_all_daytrading(dry_run: bool = False) -> dict:
    """DAYTRADING OPEN 시그널 전부 종료.

    Returns:
        {"total": N, "closed": N, "stopped": N}
    """
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()

    if not uploader.is_active:
        print("  [WARN] Supabase 미연결")
        return {"total": 0, "closed": 0, "stopped": 0}

    open_signals = uploader.fetch_open_signals(bot_type="DAYTRADING")
    print(f"  DAYTRADING OPEN: {len(open_signals)}건")

    if not open_signals:
        return {"total": 0, "closed": 0, "stopped": 0}

    today_str = datetime.now().strftime("%Y-%m-%d")
    closed = 0
    stopped = 0

    for sig in open_signals:
        signal_id = sig.get("id", "")
        ticker = sig.get("ticker", "")
        name = sig.get("ticker_name", "?")
        entry_price = sig.get("entry_price", 0)

        current_price = _get_latest_close(ticker) or entry_price
        if entry_price <= 0:
            continue

        return_pct = round((current_price - entry_price) / entry_price * 100, 2)
        prev_max = float(sig.get("max_return_pct", 0) or 0)
        max_return_pct = round(max(prev_max, return_pct), 2)

        if return_pct > 0:
            status = "CLOSED"
            reason = "EOD_PROFIT"
            closed += 1
            icon = "✅"
        else:
            status = "STOPPED"
            reason = "EOD_LOSS"
            stopped += 1
            icon = "🛑"

        print(f"    {icon} {name}({ticker}): {return_pct:+.2f}% → {status}")

        if not dry_run:
            uploader.close_signal(signal_id, {
                "status": status,
                "close_date": today_str,
                "close_reason": reason,
                "current_price": current_price,
                "return_pct": return_pct,
                "max_return_pct": max_return_pct,
            })

    result = {"total": len(open_signals), "closed": closed, "stopped": stopped}
    print(f"\n  결과: CLOSED {closed} / STOPPED {stopped}")
    return result


def aggregate_daytrading_scoreboard(dry_run: bool = False) -> list[dict]:
    """DAYTRADING 성적표 집계."""
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()

    if not uploader.is_active:
        return []

    today = datetime.now().date()
    rows = []

    for period_name, days in PERIODS.items():
        from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        signals = uploader.fetch_signals_by_period(
            bot_type="DAYTRADING",
            status_list=["CLOSED", "STOPPED"],
            from_date=from_date,
        )

        returns = [float(s.get("return_pct", 0) or 0) for s in signals]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        total = len(returns)

        best = max(signals, key=lambda s: float(s.get("return_pct", 0) or 0), default=None)
        worst = min(signals, key=lambda s: float(s.get("return_pct", 0) or 0), default=None)

        def _summary(s):
            if not s:
                return None
            return {
                "ticker": s.get("ticker", ""),
                "ticker_name": s.get("ticker_name", ""),
                "return_pct": float(s.get("return_pct", 0) or 0),
                "signal_date": s.get("signal_date", ""),
            }

        row = {
            "bot_type": "DAYTRADING",
            "period": period_name,
            "total_signals": total,
            "win_count": len(wins),
            "lose_count": len(losses),
            "win_rate": round(len(wins) / total * 100, 2) if total > 0 else 0,
            "avg_return_pct": round(sum(returns) / total, 2) if total > 0 else 0,
            "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_lose_pct": round(sum(losses) / len(losses), 2) if losses else 0,
            "best_signal": json.loads(json.dumps(_summary(best))) if best else None,
            "worst_signal": json.loads(json.dumps(_summary(worst))) if worst else None,
            "calculated_at": datetime.now().isoformat(),
        }
        rows.append(row)
        print(f"  DAYTRADING/{period_name}: {total}건, 승률 {row['win_rate']}%")

    if not dry_run and rows:
        ok = uploader.upsert_scoreboard(rows)
        print(f"  Supabase: {'OK' if ok else 'FAIL'}")

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOWX 단타 일괄 청산")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*50}")
    print(f"  FLOWX 단타 일괄 청산 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # 1. 전부 청산
    print("\n[1/2] 단타 시그널 일괄 청산...")
    close_all_daytrading(dry_run=args.dry_run)

    # 2. 성적표 집계
    print("\n[2/2] 단타 성적표 집계...")
    aggregate_daytrading_scoreboard(dry_run=args.dry_run)

    print(f"\n{'='*50}")
    print(f"  완료 | {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
