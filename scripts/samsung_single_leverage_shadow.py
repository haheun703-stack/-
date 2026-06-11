#!/usr/bin/env python3
"""Generate Samsung single-stock leverage shadow ledger/report.

Shadow-only safety rules:
- No real order.
- No external order adapter import.
- No scheduler/systemd modification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.etf.samsung_single_leverage_shadow import (  # noqa: E402
    DEFAULT_LEVERAGE_TICKER,
    SEMI_LEVERAGE_TICKER,
    UNDERLYING_TICKER,
    build_common_period_comparison,
    build_samsung_single_leverage_shadow_ledger,
    latest_provisional_warning,
    load_daily_ohlcv,
    save_samsung_single_leverage_outputs,
)
from src.etf.c60_shadow import build_c60_report, build_c60_shadow_ledger  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SAMSUNG-LEV] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("samsung_single_leverage_shadow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Samsung single-stock leverage shadow")
    parser.add_argument("--underlying", default=UNDERLYING_TICKER)
    parser.add_argument("--leverage-ticker", default=DEFAULT_LEVERAGE_TICKER)
    parser.add_argument("--days", type=int, default=1260)
    parser.add_argument("--print-report", action="store_true")
    parser.add_argument("--write", action="store_true",
                        help="ledger/report 기록(기본 dry). 장마감 15:30 KST 후에만.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.underlying != UNDERLYING_TICKER:
        logger.warning("Samsung shadow primary is designed for %s; got %s", UNDERLYING_TICKER, args.underlying)

    underlying = load_daily_ohlcv(args.underlying, days=args.days, prefer_remote=True)
    leverage = load_daily_ohlcv(args.leverage_ticker, days=args.days, prefer_remote=True)
    if leverage.empty:
        logger.warning("Leverage ticker %s data unavailable; synthetic 2x close will be used", args.leverage_ticker)
    semi_leverage = load_daily_ohlcv(SEMI_LEVERAGE_TICKER, days=args.days, prefer_remote=True)

    rows = build_samsung_single_leverage_shadow_ledger(
        underlying,
        leverage_prices=leverage if not leverage.empty else None,
        leverage_ticker=args.leverage_ticker,
    )
    if not rows:
        logger.error("Samsung single leverage shadow failed: insufficient OHLCV data")
        return 1

    c60_488080_reference = {}
    common_period_comparison = {}
    if not semi_leverage.empty:
        semi_rows = build_c60_shadow_ledger(semi_leverage, ticker=SEMI_LEVERAGE_TICKER)
        if semi_rows:
            semi_report = build_c60_report(semi_rows)
            common_period_comparison = build_common_period_comparison(rows, semi_rows)
            c60_488080_reference = {
                "ticker": SEMI_LEVERAGE_TICKER,
                "period_start": semi_rows[0].date,
                "period_end": semi_rows[-1].date,
                "ledger_rows": len(semi_rows),
                "latest_date": semi_report.get("latest_date"),
                "latest_signal": semi_report.get("latest_signal"),
                "latest_c60_position_state": semi_report.get("latest_c60_position_state"),
                "c60_final_return": semi_report.get("c60_final_return"),
                "buyhold_final_return": semi_report.get("buyhold_final_return"),
                "c60_mdd": semi_report.get("c60_mdd"),
                "buyhold_mdd": semi_report.get("buyhold_mdd"),
                "days_in_cash": semi_report.get("days_in_cash"),
                "whipsaw_count": semi_report.get("whipsaw_count"),
            }
    else:
        logger.warning("488080 reference data unavailable; Samsung-only report will be saved")

    ledger_path, report_path, report = save_samsung_single_leverage_outputs(
        rows,
        c60_488080_reference=c60_488080_reference,
        common_period_comparison=common_period_comparison,
        write=args.write,
    )
    if args.write:
        logger.info("Ledger saved: %s (%d rows)", ledger_path, len(rows))
        logger.info("Report saved: %s", report_path)
    else:
        logger.info("dry-run (기본) — 파일 미기록. 기록하려면 --write (장마감 후)")
    warn = latest_provisional_warning(rows)
    if warn:
        logger.warning("%s", warn)
    logger.info("Safety proof: real orders 0 / HOLD maintained / scheduler untouched")

    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
