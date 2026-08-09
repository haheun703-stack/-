"""Export the verified forward paper ledger to the FLOWX web contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.forward_paper_ledger import ForwardPaperLedger  # noqa: E402
from src.entities.forward_paper_event import ForwardPaperEvent  # noqa: E402
from src.use_cases.forward_paper_scoreboard import (  # noqa: E402
    build_strategy_validation_batch,
)

DEFAULT_LEDGER = ROOT / "data" / "forward_paper" / "events.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "flowx_public"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def export_batch(ledger_path: Path, output_dir: Path, as_of: datetime) -> dict:
    source = ledger_path.resolve()
    output_dir = output_dir.resolve()
    dated = output_dir / f"strategy_validation_{as_of.date().isoformat()}.json"
    latest = output_dir / "strategy_validation_latest.json"
    if source in {dated.resolve(), latest.resolve()}:
        raise ValueError("scoreboard output must not overwrite the source ledger")

    rows = ForwardPaperLedger(source).load(verify=True)
    events = [ForwardPaperEvent.from_dict(row["payload"]) for row in rows]
    batch = build_strategy_validation_batch(events, as_of=as_of)
    _atomic_json(dated, batch)
    _atomic_json(latest, batch)
    return {"batch": batch, "paths": [str(dated), str(latest)]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FLOWX paper strategy validation batch")
    parser.add_argument("--as-of", required=True, help="timezone-aware ISO-8601 cutoff")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    result = export_batch(args.ledger, args.output_dir, as_of)
    print(
        json.dumps(
            {
                "paper_only": True,
                "results": len(result["batch"]["results"]),
                "run_id": result["batch"]["run_id"],
                "paths": result["paths"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
