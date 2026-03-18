"""FLOWX STEP 2 — 시그널 종료 판정기.

OPEN 시그널을 체크하여 종료 조건 충족 시 CLOSED/STOPPED 처리.

종료 조건:
  1. TARGET: return_pct >= target 도달
  2. STOP_LOSS: return_pct <= stop 도달
  3. SAR_REVERSAL: SAR 트렌드 하락 전환
  4. TIMEOUT: 보유일 초과 (기본 10거래일)

실행 시점: 매일 16:15 (performance_tracker 이후)

Usage:
    python scripts/signal_closer.py
    python scripts/signal_closer.py --dry-run
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 기본 종료 기준
MAX_HOLDING_DAYS = 10  # 최대 보유 거래일


def _get_sar_trend(ticker: str) -> int:
    """parquet에서 최신 SAR 트렌드 (1=상승, -1=하락)."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq, columns=["sar_trend"])
            if len(df) > 0 and "sar_trend" in df.columns:
                return int(df.iloc[-1]["sar_trend"])
        except Exception:
            pass
    return 0


def _calc_holding_days(signal_date_str: str) -> int:
    """시그널 날짜부터 오늘까지 거래일 (주말 제외 근사)."""
    try:
        sig_date = datetime.strptime(signal_date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        delta = (today - sig_date).days
        # 거래일 근사: weekday 제외
        weeks = delta // 7
        remainder = delta % 7
        return weeks * 5 + min(remainder, 5)
    except Exception:
        return 0


def close_signals(dry_run: bool = False) -> dict:
    """OPEN 시그널 종료 판정.

    Returns:
        {"total": N, "closed": N, "stopped": N, "open": N}
    """
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()

    if not uploader.is_active:
        print("  [WARN] Supabase 미연결")
        return {"total": 0, "closed": 0, "stopped": 0, "open": 0}

    open_signals = uploader.fetch_open_signals(bot_type="QUANT")
    print(f"  OPEN 시그널: {len(open_signals)}건")

    if not open_signals:
        return {"total": 0, "closed": 0, "stopped": 0, "open": 0}

    today_str = datetime.now().strftime("%Y-%m-%d")
    closed_count = 0
    stopped_count = 0
    still_open = 0

    for sig in open_signals:
        signal_id = sig.get("id", "")
        ticker = sig.get("ticker", "")
        name = sig.get("ticker_name", "?")
        entry_price = sig.get("entry_price", 0)
        target_price = sig.get("target_price", 0)
        stop_price = sig.get("stop_price", 0)
        current_price = sig.get("current_price", 0)
        return_pct = float(sig.get("return_pct", 0) or 0)
        signal_date = sig.get("signal_date", "")

        if entry_price <= 0 or current_price <= 0:
            still_open += 1
            continue

        close_data = None

        # 1. 목표 도달
        if target_price > 0 and current_price >= target_price:
            close_data = {
                "status": "CLOSED",
                "close_date": today_str,
                "close_reason": "TARGET",
                "return_pct": return_pct,
            }
            print(f"    ✅ {name}({ticker}): TARGET 도달 ({return_pct:+.2f}%)")
            closed_count += 1

        # 2. 손절
        elif stop_price > 0 and current_price <= stop_price:
            close_data = {
                "status": "STOPPED",
                "close_date": today_str,
                "close_reason": "STOP_LOSS",
                "return_pct": return_pct,
            }
            print(f"    🛑 {name}({ticker}): STOP_LOSS ({return_pct:+.2f}%)")
            stopped_count += 1

        # 3. SAR 반전
        elif _get_sar_trend(ticker) == -1:
            close_data = {
                "status": "STOPPED",
                "close_date": today_str,
                "close_reason": "SAR_REVERSAL",
                "return_pct": return_pct,
            }
            print(f"    📉 {name}({ticker}): SAR 반전 ({return_pct:+.2f}%)")
            stopped_count += 1

        # 4. 시간 초과
        elif _calc_holding_days(signal_date) >= MAX_HOLDING_DAYS:
            status = "CLOSED" if return_pct >= 0 else "STOPPED"
            close_data = {
                "status": status,
                "close_date": today_str,
                "close_reason": "TIMEOUT",
                "return_pct": return_pct,
            }
            print(f"    ⏰ {name}({ticker}): TIMEOUT {MAX_HOLDING_DAYS}일 ({return_pct:+.2f}%)")
            if status == "CLOSED":
                closed_count += 1
            else:
                stopped_count += 1

        else:
            still_open += 1
            continue

        if close_data and not dry_run:
            uploader.close_signal(signal_id, close_data)

    result = {
        "total": len(open_signals),
        "closed": closed_count,
        "stopped": stopped_count,
        "open": still_open,
    }

    print(f"\n  결과: CLOSED {closed_count} / STOPPED {stopped_count} / OPEN {still_open}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOWX 시그널 종료 판정")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*40}")
    print(f"  FLOWX 시그널 종료 판정 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}\n")

    close_signals(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
