"""Mirror scheduled KR paper portfolios into the FLOWX append-only ledger."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.forward_paper_ledger import ForwardPaperLedger  # noqa: E402
from src.use_cases.forward_paper_portfolio_mapper import map_legacy_portfolio  # noqa: E402
from src.use_cases.forward_paper_market_mapper import (  # noqa: E402
    map_index_benchmarks,
    map_wave_book,
)


PROFILES = {
    "main_a": ("data/paper_portfolio.json", "paper_main_a"),
    "main_b": ("data/paper_portfolio_b.json", "paper_main_b"),
    "value_fib": ("data/paper_portfolio_vf.json", "paper_value_fib"),
    "holdnav": ("data/paper_portfolio_holdnav.json", "paper_holding_nav"),
}
MARKET_PROFILES = {
    "index_bh": "data/paper_portfolio_indexbh.json",
    "wave_kr": "data/paper_portfolio_wave_kr.json",
    "wave_us": "data/paper_portfolio_wave_us.json",
}
ALL_PROFILES = {**PROFILES, **MARKET_PROFILES}
DEFAULT_LEDGER = ROOT / "data" / "forward_paper" / "events.jsonl"


def _git_revision() -> str:
    """Read the deployed repository revision without spawning a subprocess."""
    git_dir = ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            return (git_dir / ref).read_text(encoding="ascii").strip()[:12]
        return head[:12]
    except OSError:
        return "unknown-revision"


def mirror_profile(profile: str, ledger: ForwardPaperLedger, strategy_version: str) -> dict:
    if profile in PROFILES:
        rel_path, portfolio_id = PROFILES[profile]
    else:
        rel_path, portfolio_id = MARKET_PROFILES[profile], None
    source_path = ROOT / rel_path
    if not source_path.exists():
        return {"profile": profile, "status": "SOURCE_MISSING", "appended": 0}
    snapshot = json.loads(source_path.read_text(encoding="utf-8"))
    if portfolio_id:
        events = map_legacy_portfolio(
            snapshot, portfolio_id=portfolio_id, strategy_version=strategy_version,
            source_name=source_path.name,
        )
    elif profile == "index_bh":
        events = map_index_benchmarks(
            snapshot, strategy_version=strategy_version, source_name=source_path.name,
        )
    else:
        events = map_wave_book(
            snapshot, market=profile.removeprefix("wave_").upper(),
            strategy_version=strategy_version, source_name=source_path.name,
        )
    existing_ids = {row["payload"]["event_id"] for row in ledger.load(verify=True)}
    appended = 0
    for event in events:
        if event.event_id in existing_ids:
            continue
        ledger.append(event)
        existing_ids.add(event.event_id)
        appended += 1
    return {"profile": profile, "status": "OK", "events": len(events), "appended": appended}


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror KR paper portfolios to forward ledger")
    parser.add_argument("--profile", choices=["all", *ALL_PROFILES], default="all")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--strategy-version",
        default=os.getenv("FLOWX_STRATEGY_VERSION") or _git_revision(),
    )
    args = parser.parse_args()

    ledger = ForwardPaperLedger(args.ledger)
    selected = list(ALL_PROFILES) if args.profile == "all" else [args.profile]
    results = [mirror_profile(profile, ledger, args.strategy_version) for profile in selected]
    ledger.verify()
    print(json.dumps({"ledger": str(args.ledger), "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
