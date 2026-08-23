from __future__ import annotations

import json

import pytest

from scripts.upload_flowx_strategy_scoreboard import MAX_BODY_BYTES, upload_batch


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def _batch(tmp_path, *, results: list[dict] | None = None):
    payload = {
        "schema_version": "1.0",
        "run_id": "quant-20260823T100000Z",
        "producer": "quant-bot",
        "generated_at": "2026-08-23T10:00:00Z",
        "results": results or [{"strategy_id": "paper-kr", "market": "KR"}],
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_upload_uses_bearer_without_logging_or_mutating_batch(tmp_path) -> None:
    path, batch = _batch(tmp_path)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(
            {"data": {"accepted": 1, "runId": batch["run_id"], "producer": "quant-bot"}}
        )

    receipt = upload_batch(path, token="test-token", post=post)

    assert receipt == {"accepted": 1, "run_id": batch["run_id"], "producer": "quant-bot"}
    assert calls[0][0].startswith("https://")
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-token"
    assert calls[0][1]["timeout"] == 30
    assert json.loads(calls[0][1]["data"]) == batch


def test_upload_fails_closed_for_missing_token_and_bad_identity(tmp_path) -> None:
    path, _ = _batch(tmp_path)
    with pytest.raises(ValueError, match="TOKEN"):
        upload_batch(path, token="")

    path.write_text(json.dumps({"schema_version": "1.0", "producer": "other", "results": [{}]}))
    with pytest.raises(ValueError, match="identity"):
        upload_batch(path, token="token")


def test_upload_rejects_oversize_and_mismatched_receipt(tmp_path) -> None:
    path, batch = _batch(tmp_path, results=[{"strategy_id": "x", "note": "a" * MAX_BODY_BYTES}])
    with pytest.raises(ValueError, match="256KiB"):
        upload_batch(path, token="token")

    path, batch = _batch(tmp_path)

    def post(_url, **_kwargs):
        return _Response({"data": {"accepted": 0, "runId": batch["run_id"]}})

    with pytest.raises(RuntimeError, match="receipt"):
        upload_batch(path, token="token", post=post)
