"""Run the quant bot as a 3 trading-day paper pilot.

Window: 2026-05-27(Wed) through 2026-05-29(Fri), KST.

This wrapper never creates a real-order adapter. It sets QUANT_3DAY_PILOT=1,
forces PaperOrderAdapter via run_cycle(is_paper=True), and writes a compact
JSON summary under results/quant_3day_pilot/.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PILOT_START = dt.date(2026, 5, 27)
PILOT_END = dt.date(2026, 5, 29)
RESULT_DIR = PROJECT_ROOT / "results" / "quant_3day_pilot"
RUNS_CSV = RESULT_DIR / "quant_3day_pilot_runs.csv"
LATEST_MD = RESULT_DIR / "quant_3day_pilot_latest.md"
ORDER_LOG = RESULT_DIR / "paper_orders.jsonl"


def is_pilot_day(today: dt.date) -> bool:
    return PILOT_START <= today <= PILOT_END and today.weekday() < 5


def write_summary(summary: dict) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULT_DIR / f"quant_3day_pilot_{stamp}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summarize_candidate_pool() -> dict:
    from scripts.run_adaptive_cycle import load_latest_soubujang_pool, passed_candidates

    pool = load_latest_soubujang_pool()
    passed = passed_candidates(pool)
    latest_files = sorted((PROJECT_ROOT / "data" / "soubujang_pool").glob("*.json"), reverse=True)
    latest_path = latest_files[0] if latest_files else None

    fail_reasons: Counter[str] = Counter()
    sectors: Counter[str] = Counter()
    passed_rows = []

    for ticker, info in pool.items():
        sectors[str(info.get("sector_key", "unknown"))] += 1
        reasons = info.get("filter_reasons", []) or []
        if not info.get("passed"):
            for reason in reasons:
                text = str(reason)
                if "OK" not in text:
                    fail_reasons[text] += 1
        else:
            step5 = info.get("step5", {}) or {}
            passed_rows.append({
                "ticker": ticker,
                "name": info.get("name", ""),
                "sector_key": info.get("sector_key", ""),
                "current_price": (info.get("kis", {}) or {}).get("current_price", 0),
                "upside_ratio": step5.get("upside_ratio"),
                "annual_return": step5.get("annual_return"),
                "filter_reasons": reasons,
            })

    passed_rows.sort(
        key=lambda x: float(x["upside_ratio"] or 0),
        reverse=True,
    )

    mtime = None
    age_hours = None
    if latest_path and latest_path.exists():
        mtime = dt.datetime.fromtimestamp(latest_path.stat().st_mtime)
        age_hours = round((dt.datetime.now() - mtime).total_seconds() / 3600, 1)

    return {
        "pool_file": str(latest_path) if latest_path else "",
        "pool_file_mtime": mtime.isoformat(timespec="seconds") if mtime else "",
        "pool_file_age_hours": age_hours,
        "total_candidates": len(pool),
        "passed_candidates": len(passed),
        "failed_candidates": max(0, len(pool) - len(passed)),
        "sector_counts": dict(sectors.most_common()),
        "failed_reason_top": [
            {"reason": reason, "count": count}
            for reason, count in fail_reasons.most_common(10)
        ],
        "passed_top": passed_rows[:10],
    }


def summarize_triggers(summary: dict) -> dict:
    keys = ["mvp1", "mvp2", "mvp2_5", "mvp2_6", "mvp2_7", "mvp2_8", "mvp3", "mvp4", "mvp5", "mvp6", "mvp7"]
    rows = []
    total_triggers = 0
    total_errors = 0
    for key in keys:
        item = summary.get(key, {}) or {}
        triggers = int(item.get("triggers", 0) or 0)
        errors = item.get("errors", []) or []
        total_triggers += triggers
        total_errors += len(errors)
        rows.append({
            "module": key,
            "executed": bool(item.get("executed", False)),
            "triggers": triggers,
            "errors": errors,
        })
    return {
        "total_triggers": total_triggers,
        "total_errors": total_errors,
        "modules": rows,
    }


def read_order_rows(run_id: str) -> list[dict]:
    if not ORDER_LOG.exists():
        return []
    rows = []
    for line in ORDER_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("run_id") == run_id:
            rows.append(row)
    return rows


def build_no_trade_reasons(summary: dict, candidate_report: dict, trigger_report: dict) -> list[str]:
    if trigger_report["total_triggers"] > 0:
        return []

    reasons = ["all adaptive-cycle trigger counts were 0"]
    if candidate_report.get("passed_candidates", 0) > 0:
        reasons.append(
            f"{candidate_report['passed_candidates']} candidates passed the pool, but no intraday/entry trigger fired"
        )
    else:
        reasons.append("candidate pool had no passed candidates")

    age = candidate_report.get("pool_file_age_hours")
    if isinstance(age, (int, float)) and age > 24:
        reasons.append(f"candidate pool is stale ({age}h old); refresh Step 5 pool before next run")

    if candidate_report.get("failed_reason_top"):
        top = candidate_report["failed_reason_top"][0]
        reasons.append(f"top pool rejection: {top['reason']} ({top['count']} names)")

    return reasons


def append_run_csv(summary: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pilot = summary.get("pilot", {})
    candidates = summary.get("candidate_report", {})
    triggers = summary.get("trigger_report", {})
    row = {
        "run_id": pilot.get("run_id", ""),
        "started_at": summary.get("started_at", ""),
        "ended_at": summary.get("ended_at", ""),
        "mode": summary.get("mode", ""),
        "total_candidates": candidates.get("total_candidates", 0),
        "passed_candidates": candidates.get("passed_candidates", 0),
        "total_triggers": triggers.get("total_triggers", 0),
        "total_errors": triggers.get("total_errors", 0),
        "paper_orders": len(summary.get("paper_orders", [])),
        "no_trade_reasons": " | ".join(summary.get("no_trade_reasons", [])),
    }
    exists = RUNS_CSV.exists()
    with RUNS_CSV.open("a", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_latest_markdown(summary: dict) -> None:
    candidates = summary.get("candidate_report", {})
    triggers = summary.get("trigger_report", {})
    orders = summary.get("paper_orders", [])
    lines = [
        "# Quant 3-Day Pilot",
        "",
        f"- Window: {PILOT_START} ~ {PILOT_END}",
        f"- Run: {summary.get('started_at', '')} -> {summary.get('ended_at', '')}",
        f"- Mode: {summary.get('mode', '')}",
        f"- Real orders allowed: {summary.get('pilot', {}).get('real_orders_allowed', False)}",
        "",
        "## Candidate Pool",
        "",
        f"- Total candidates: {candidates.get('total_candidates', 0)}",
        f"- Passed candidates: {candidates.get('passed_candidates', 0)}",
        f"- Pool file age: {candidates.get('pool_file_age_hours', '')}h",
        "",
        "## Trigger Summary",
        "",
        "| Module | Executed | Triggers | Errors |",
        "|---|---:|---:|---:|",
    ]
    for row in triggers.get("modules", []):
        lines.append(
            f"| {row['module']} | {row['executed']} | {row['triggers']} | {len(row['errors'])} |"
        )

    lines += ["", "## No-Trade Reasons", ""]
    for reason in summary.get("no_trade_reasons", []):
        lines.append(f"- {reason}")
    if not summary.get("no_trade_reasons"):
        lines.append("- Trades/orders were generated in this run.")

    lines += ["", "## Passed Candidates Top", "", "| Ticker | Name | Sector | Upside | Annual Return |", "|---|---|---|---:|---:|"]
    for row in candidates.get("passed_top", [])[:10]:
        lines.append(
            f"| {row.get('ticker', '')} | {row.get('name', '')} | {row.get('sector_key', '')} | "
            f"{row.get('upside_ratio', '')} | {row.get('annual_return', '')} |"
        )

    lines += ["", "## Paper Orders", ""]
    if orders:
        lines += ["| Time | Side | Ticker | Qty | Filled | Cash Effect |", "|---|---|---|---:|---:|---:|"]
        for row in orders:
            lines.append(
                f"| {row.get('timestamp', '')} | {row.get('side', '')} | {row.get('ticker', '')} | "
                f"{row.get('filled_quantity', 0)} | {row.get('filled_price', 0)} | {row.get('cash_effect', 0)} |"
            )
    else:
        lines.append("- No paper orders in this run.")

    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    today = dt.date.today()
    now = dt.datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")

    os.environ["QUANT_3DAY_PILOT"] = "1"
    os.environ["QUANT_3DAY_PILOT_RUN_ID"] = run_id

    base = {
        "pilot": {
            "name": "quant_3day_wed_thu_fri",
            "run_id": run_id,
            "start": PILOT_START.isoformat(),
            "end": PILOT_END.isoformat(),
            "today": today.isoformat(),
            "forced_mode": "PAPER",
            "real_orders_allowed": False,
        },
        "started_at": now.isoformat(timespec="seconds"),
    }

    if not is_pilot_day(today):
        summary = {
            **base,
            "skipped": True,
            "skip_reason": "outside 2026-05-27~2026-05-29 pilot window",
        }
        path = write_summary(summary)
        print(f"3-day pilot skipped: {summary['skip_reason']}")
        print(f"summary: {path}")
        return 0

    from scripts.run_adaptive_cycle import run_cycle

    summary = run_cycle(is_paper=True, skip=set(), dry_run=False)
    summary.update(base)
    summary["ended_at"] = dt.datetime.now().isoformat(timespec="seconds")
    summary["candidate_report"] = summarize_candidate_pool()
    summary["trigger_report"] = summarize_triggers(summary)
    summary["paper_orders"] = read_order_rows(run_id)
    summary["no_trade_reasons"] = build_no_trade_reasons(
        summary,
        summary["candidate_report"],
        summary["trigger_report"],
    )
    path = write_summary(summary)
    append_run_csv(summary)
    write_latest_markdown(summary)

    print("3-day quant pilot complete")
    print("mode: PAPER")
    print(f"window: {PILOT_START} ~ {PILOT_END}")
    print(f"candidates: {summary['candidate_report']['total_candidates']} total / {summary['candidate_report']['passed_candidates']} passed")
    print(f"triggers: {summary['trigger_report']['total_triggers']}")
    print(f"paper orders: {len(summary['paper_orders'])}")
    print(f"summary: {path}")
    print(f"latest report: {LATEST_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
