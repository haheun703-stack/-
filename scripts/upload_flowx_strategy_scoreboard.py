"""Upload the paper-only strategy scoreboard batch to the FLOWX API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_BATCH = ROOT / "data" / "flowx_public" / "strategy_validation_latest.json"
DEFAULT_ENDPOINT = "https://flowx.kr/api/strategy-scoreboard"
MAX_BODY_BYTES = 256 * 1024


def upload_batch(
    batch_path: Path,
    *,
    token: str,
    endpoint: str = DEFAULT_ENDPOINT,
    post: Callable[..., object] = requests.post,
) -> dict:
    if not token.strip():
        raise ValueError("FLOWX_SCOREBOARD_TOKEN is required")
    if not endpoint.startswith("https://"):
        raise ValueError("FLOWX scoreboard endpoint must use HTTPS")

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if batch.get("schema_version") != "1.0" or batch.get("producer") != "quant-bot":
        raise ValueError("unexpected FLOWX strategy scoreboard batch identity")
    results = batch.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("FLOWX strategy scoreboard batch has no results")

    body = json.dumps(batch, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("FLOWX strategy scoreboard batch exceeds 256KiB")

    response = post(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    accepted = payload.get("data", {}).get("accepted")
    run_id = payload.get("data", {}).get("runId")
    if accepted != len(results) or run_id != batch.get("run_id"):
        raise RuntimeError("FLOWX scoreboard receipt does not match the submitted batch")
    return {"accepted": accepted, "run_id": run_id, "producer": "quant-bot"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload FLOWX paper strategy scoreboard batch")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    receipt = upload_batch(
        args.batch,
        token=os.getenv("FLOWX_SCOREBOARD_TOKEN", ""),
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
