"""FLOWX STEP 2 — 시그널 성과 추적기.

장마감 후(16:10) OPEN 시그널의 현재가/수익률/최대수익률을 업데이트.

로직:
  1. Supabase에서 OPEN 시그널 조회
  2. 각 종목 최신 종가 (parquet)
  3. return_pct = (current - entry) / entry * 100
  4. max_return_pct = max(기존 max, 오늘 return_pct)
  5. Supabase UPDATE

Usage:
    python scripts/performance_tracker.py
    python scripts/performance_tracker.py --dry-run
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _get_latest_close(ticker: str) -> int:
    """parquet에서 최신 종가 조회."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq, columns=["close"])
            if len(df) > 0:
                return int(df.iloc[-1]["close"])
        except Exception:
            pass
    return 0


def _get_sar_trend(ticker: str) -> int:
    """parquet에서 최신 SAR 트렌드 조회 (1=상승, -1=하락)."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq, columns=["sar_trend"])
            if len(df) > 0 and "sar_trend" in df.columns:
                return int(df.iloc[-1]["sar_trend"])
        except Exception:
            pass
    return 0


def update_performance(dry_run: bool = False) -> dict:
    """OPEN 시그널 성과 업데이트.

    Returns:
        {"total": N, "updated": N, "skipped": N}
    """
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()

    if not uploader.is_active:
        print("  [WARN] Supabase 미연결")
        return {"total": 0, "updated": 0, "skipped": 0}

    # OPEN 시그널 조회
    open_signals = uploader.fetch_open_signals(bot_type="QUANT")
    print(f"  OPEN 시그널: {len(open_signals)}건")

    if not open_signals:
        return {"total": 0, "updated": 0, "skipped": 0}

    updated = 0
    skipped = 0

    for sig in open_signals:
        ticker = sig.get("ticker", "")
        signal_id = sig.get("id", "")
        entry_price = sig.get("entry_price", 0)

        if not ticker or entry_price <= 0:
            skipped += 1
            continue

        current_price = _get_latest_close(ticker)
        if current_price <= 0:
            print(f"    {sig.get('ticker_name', '?')}({ticker}): 종가 없음 — 스킵")
            skipped += 1
            continue

        return_pct = round((current_price - entry_price) / entry_price * 100, 2)
        prev_max = float(sig.get("max_return_pct", 0) or 0)
        max_return_pct = round(max(prev_max, return_pct), 2)

        print(f"    {sig.get('ticker_name', '?')}({ticker}): "
              f"{entry_price:,} → {current_price:,} ({return_pct:+.2f}%, max {max_return_pct:+.2f}%)")

        if dry_run:
            updated += 1
            continue

        ok = uploader.update_signal_performance(signal_id, {
            "current_price": current_price,
            "return_pct": return_pct,
            "max_return_pct": max_return_pct,
        })
        if ok:
            updated += 1
        else:
            skipped += 1

    result = {"total": len(open_signals), "updated": updated, "skipped": skipped}
    print(f"\n  성과 업데이트: {updated}/{len(open_signals)}건")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOWX 시그널 성과 추적")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*40}")
    print(f"  FLOWX 성과 추적 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}\n")

    update_performance(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
