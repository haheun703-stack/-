"""Append-only, hash-chained storage for FLOWX forward paper events."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.entities.forward_paper_event import ForwardPaperEvent


class ForwardPaperLedgerError(RuntimeError):
    pass


class DuplicatePaperEventError(ForwardPaperLedgerError):
    pass


class PaperLedgerIntegrityError(ForwardPaperLedgerError):
    pass


def _canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(payload: dict, previous_hash: str) -> str:
    material = f"{previous_hash}\n{_canonical_json(payload)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@contextmanager
def _directory_lock(lock_path: Path, timeout_sec: float = 5.0) -> Iterator[None]:
    """Portable process lock using atomic directory creation."""
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            lock_path.mkdir(parents=False)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"forward paper ledger lock timeout: {lock_path}")
            time.sleep(0.02)
    try:
        yield
    finally:
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


class ForwardPaperLedger:
    GENESIS_HASH = "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")

    def append(self, event: ForwardPaperEvent) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _directory_lock(self.lock_path):
            records = self.load(verify=True)
            if any(row["payload"]["event_id"] == event.event_id for row in records):
                raise DuplicatePaperEventError(f"duplicate event_id: {event.event_id}")

            previous_hash = records[-1]["record_hash"] if records else self.GENESIS_HASH
            payload = event.to_dict()
            envelope = {
                "sequence": len(records) + 1,
                "previous_hash": previous_hash,
                "payload": payload,
                "record_hash": _record_hash(payload, previous_hash),
            }
            line = _canonical_json(envelope) + "\n"
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return envelope

    def load(self, verify: bool = True) -> list[dict]:
        if not self.path.exists():
            return []
        records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PaperLedgerIntegrityError(f"invalid JSON at line {line_no}") from exc
                records.append(row)
        if verify:
            self.verify(records)
        return records

    def verify(self, records: list[dict] | None = None) -> None:
        rows = self.load(verify=False) if records is None else records
        previous_hash = self.GENESIS_HASH
        event_ids: set[str] = set()
        for expected_sequence, row in enumerate(rows, start=1):
            if row.get("sequence") != expected_sequence:
                raise PaperLedgerIntegrityError(
                    f"sequence mismatch: expected {expected_sequence}, got {row.get('sequence')}"
                )
            if row.get("previous_hash") != previous_hash:
                raise PaperLedgerIntegrityError(f"previous_hash mismatch at sequence {expected_sequence}")
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise PaperLedgerIntegrityError(f"payload missing at sequence {expected_sequence}")
            event_id = payload.get("event_id")
            if event_id in event_ids:
                raise PaperLedgerIntegrityError(f"duplicate event_id in ledger: {event_id}")
            event_ids.add(event_id)
            expected_hash = _record_hash(payload, previous_hash)
            if row.get("record_hash") != expected_hash:
                raise PaperLedgerIntegrityError(f"record_hash mismatch at sequence {expected_sequence}")
            ForwardPaperEvent.from_dict(payload)
            previous_hash = expected_hash
