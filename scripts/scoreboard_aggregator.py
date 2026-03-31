"""FLOWX STEP 2 — 성적표 집계기.

CLOSED/STOPPED 시그널을 기간별로 집계하여 scoreboard 테이블에 UPSERT.

집계 기간: 30D, 60D, 90D, ALL
집계 항목: total_signals, win_count, lose_count, win_rate, avg_return_pct,
           avg_win_pct, avg_lose_pct, best_signal, worst_signal

실행 시점: 매일 16:30 (signal_closer 이후)

Usage:
    python scripts/scoreboard_aggregator.py
    python scripts/scoreboard_aggregator.py --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# 집계 기간 정의 (일수)
PERIODS = {
    "30D": 30,
    "60D": 60,
    "90D": 90,
    "ALL": 3650,  # ~10년
}


def _calc_stats(signals: list[dict]) -> dict:
    """시그널 리스트 → 성적표 통계 계산."""
    if not signals:
        return {
            "total_signals": 0,
            "win_count": 0,
            "lose_count": 0,
            "win_rate": 0,
            "avg_return_pct": 0,
            "avg_win_pct": 0,
            "avg_lose_pct": 0,
            "best_signal": None,
            "worst_signal": None,
        }

    returns = []
    wins = []
    losses = []
    best = None
    worst = None

    for sig in signals:
        ret = float(sig.get("return_pct", 0) or 0)
        returns.append(ret)

        if ret > 0:
            wins.append(ret)
        else:
            losses.append(ret)

        if best is None or ret > float(best.get("return_pct", 0) or 0):
            best = sig
        if worst is None or ret < float(worst.get("return_pct", 0) or 0):
            worst = sig

    total = len(returns)
    win_count = len(wins)
    lose_count = len(losses)

    # best/worst 시그널 요약 (전체 데이터가 아닌 요약만 저장)
    def _signal_summary(s: dict | None) -> dict | None:
        if not s:
            return None
        return {
            "ticker": s.get("ticker", ""),
            "ticker_name": s.get("ticker_name", ""),
            "return_pct": float(s.get("return_pct", 0) or 0),
            "signal_date": s.get("signal_date", ""),
            "close_date": s.get("close_date", ""),
            "close_reason": s.get("close_reason", ""),
        }

    return {
        "total_signals": total,
        "win_count": win_count,
        "lose_count": lose_count,
        "win_rate": round(win_count / total * 100, 2) if total > 0 else 0,
        "avg_return_pct": round(sum(returns) / total, 2) if total > 0 else 0,
        "avg_win_pct": round(sum(wins) / win_count, 2) if win_count > 0 else 0,
        "avg_lose_pct": round(sum(losses) / lose_count, 2) if lose_count > 0 else 0,
        "best_signal": _signal_summary(best),
        "worst_signal": _signal_summary(worst),
    }


def aggregate_scoreboard(dry_run: bool = False) -> list[dict]:
    """기간별 성적표 집계 → Supabase UPSERT.

    Returns:
        집계된 scoreboard 행 리스트
    """
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()

    if not uploader.is_active:
        print("  [WARN] Supabase 미연결")
        return []

    today = datetime.now().date()
    rows = []

    for bot_type in ["QUANT", "ALL"]:
        for period_name, days in PERIODS.items():
            from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")

            signals = uploader.fetch_signals_by_period(
                bot_type=bot_type,
                status_list=["CLOSED", "STOPPED"],
                from_date=from_date,
            )

            stats = _calc_stats(signals)

            row = {
                "bot_type": bot_type,
                "period": period_name,
                **stats,
                "calculated_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            # jsonb 필드는 json 직렬화
            if row.get("best_signal"):
                row["best_signal"] = json.loads(json.dumps(row["best_signal"]))
            if row.get("worst_signal"):
                row["worst_signal"] = json.loads(json.dumps(row["worst_signal"]))

            rows.append(row)

            print(f"  {bot_type}/{period_name}: {stats['total_signals']}건, "
                  f"승률 {stats['win_rate']}%, 평균 {stats['avg_return_pct']:+.2f}%")

    if not dry_run and rows:
        ok = uploader.upsert_scoreboard(rows)
        print(f"\n  Supabase 업로드: {'OK' if ok else 'FAIL'} ({len(rows)}건)")

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOWX 성적표 집계")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*40}")
    print(f"  FLOWX 성적표 집계 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}\n")

    aggregate_scoreboard(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
