#!/usr/bin/env python3
"""Generate accelerated historical replay for 488080 C60 shadow.

This is analytics only:
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
    SHADOW_DIR,
    build_accelerated_c60_validation,
    build_c60_shadow_ledger,
    load_daily_close,
)

DEFAULT_JSON_PATH = SHADOW_DIR / f"{DEFAULT_TICKER}_c60_accelerated_replay.json"
DEFAULT_MD_PATH = SHADOW_DIR / f"{DEFAULT_TICKER}_c60_accelerated_replay.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_windows(raw: str) -> list[int]:
    windows = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 1:
            raise ValueError("window must be greater than 1")
        windows.append(value)
    if not windows:
        raise ValueError("at least one window is required")
    return windows


def pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def render_markdown(report: dict) -> str:
    lines = [
        "# 488080 C60 Accelerated Shadow Replay",
        "",
        "## Safety",
        "",
        "- Real orders: 0",
        "- Live trading state: HOLD",
        "- Scheduler/systemd changes: 0",
        "- Purpose: analytics-only historical rolling replay",
        "",
        "## Base",
        "",
        f"- Ledger: {report.get('ledger_start')} ~ {report.get('ledger_end')} ({report.get('ledger_rows')} rows)",
        f"- Latest signal: {report.get('latest_signal')}",
        f"- Latest state: {report.get('latest_c60_position_state')}",
        "",
        "## Rolling Window Summary",
        "",
        "| Window | Segments | C60 MDD Better | C60 Return Better | Avg Return Delta | Avg MDD Edge | Latest C60 | Latest Buyhold | Latest MDD Edge |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key in sorted(report.get("windows", {}), key=lambda x: int(x)):
        row = report["windows"][key]
        latest = row["latest_segment"]
        lines.append(
            "| {window} | {segments} | {mdd_better} | {return_better} | {avg_return_delta} | {avg_mdd_edge} | {latest_c60} | {latest_buyhold} | {latest_mdd_edge} |".format(
                window=row["window_days"],
                segments=row["segment_count"],
                mdd_better=pct(row["c60_mdd_better_rate"]),
                return_better=pct(row["c60_return_better_rate"]),
                avg_return_delta=pct(row["avg_return_delta"]),
                avg_mdd_edge=pct(row["avg_mdd_edge"]),
                latest_c60=pct(latest["c60_return"]),
                latest_buyhold=pct(latest["buyhold_return"]),
                latest_mdd_edge=pct(latest["mdd_edge"]),
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Positive Avg MDD Edge means C60 reduced drawdown versus buyhold.",
        "- Negative Avg Return Delta means C60 paid insurance cost versus buyhold.",
        "- Latest segment is not a live trading instruction. It only shortens the first evidence loop.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="488080 C60 accelerated shadow replay")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--days", type=int, default=760)
    parser.add_argument("--windows", default="20,40,60,120")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--md-path", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ticker != DEFAULT_TICKER:
        logger.warning("C60 shadow primary is designed for %s; got %s", DEFAULT_TICKER, args.ticker)

    windows = parse_windows(args.windows)
    df = load_daily_close(args.ticker, days=args.days)
    rows = build_c60_shadow_ledger(df, ticker=args.ticker)
    if not rows:
        logger.error("C60 accelerated replay failed: insufficient OHLCV data")
        return 1

    report = build_accelerated_c60_validation(rows, windows=windows)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md_path.write_text(render_markdown(report), encoding="utf-8")

    logger.info("C60 accelerated replay JSON saved: %s", args.json_path)
    logger.info("C60 accelerated replay markdown saved: %s", args.md_path)
    logger.info("Safety proof: real orders 0 / HOLD maintained / scheduler untouched")

    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
