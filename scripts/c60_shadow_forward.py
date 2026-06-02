#!/usr/bin/env python3
"""Generate 488080 C60 shadow forward ledger/report.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etf.c60_shadow import (  # noqa: E402
    DEFAULT_TICKER,
    build_c60_shadow_ledger,
    load_daily_close,
    save_shadow_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="488080 C60 shadow forward ledger")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--days", type=int, default=260)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ticker != DEFAULT_TICKER:
        logger.warning("C60 shadow primary is designed for %s; got %s", DEFAULT_TICKER, args.ticker)

    df = load_daily_close(args.ticker, days=args.days)
    rows = build_c60_shadow_ledger(df, ticker=args.ticker)
    if not rows:
        logger.error("C60 shadow ledger 생성 실패: 데이터 부족 또는 OHLCV 조회 실패")
        return 1

    ledger_path, report_path, report = save_shadow_outputs(rows)
    logger.info("C60 shadow ledger 저장: %s (%d행)", ledger_path, len(rows))
    logger.info("C60 shadow report 저장: %s", report_path)
    logger.info("안전 증빙: 실주문 0건/HOLD 유지")

    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
