from __future__ import annotations

import json

import pytest

from src.adapters.forward_paper_ledger import (
    DuplicatePaperEventError,
    ForwardPaperLedger,
    PaperLedgerIntegrityError,
)
from src.entities.forward_paper_event import (
    ForwardPaperEvent,
    PaperEventType,
    PaperMarket,
)


def _event(event_id: str = "evt-1", **overrides) -> ForwardPaperEvent:
    values = {
        "event_id": event_id,
        "trade_id": "trade-1",
        "event_type": PaperEventType.SIGNAL_FROZEN,
        "producer_bot": "quant",
        "strategy_id": "morning_plan_07",
        "strategy_version": "420b1d2",
        "market": PaperMarket.KR,
        "currency": "KRW",
        "ticker": "005930",
        "event_at": "2026-08-08T08:20:00+09:00",
        "data_asof": "2026-08-07T15:30:00+09:00",
        "signal_at": "2026-08-08T08:20:00+09:00",
        "eligible_at": "2026-08-10T09:00:00+09:00",
        "reference_price": 70000,
        "real_order": False,
    }
    values.update(overrides)
    return ForwardPaperEvent(**values)


def test_append_roundtrip_and_hash_chain(tmp_path) -> None:
    ledger = ForwardPaperLedger(tmp_path / "forward.jsonl")
    first = ledger.append(_event())
    second = ledger.append(
        _event(
            "evt-2",
            event_type=PaperEventType.FILL,
            status="PARTIAL",
            filled_quantity=0.5,
            fill_price=70100,
            fill_ratio=0.5,
            fee=10,
            tax=0,
            slippage_cost=50,
            spread_cost=20,
        )
    )

    rows = ledger.load()
    assert len(rows) == 2
    assert second["previous_hash"] == first["record_hash"]
    assert rows[1]["payload"]["fill_ratio"] == 0.5


def test_duplicate_event_id_is_rejected(tmp_path) -> None:
    ledger = ForwardPaperLedger(tmp_path / "forward.jsonl")
    ledger.append(_event())
    with pytest.raises(DuplicatePaperEventError, match="duplicate event_id"):
        ledger.append(_event())


def test_tampered_history_is_detected(tmp_path) -> None:
    path = tmp_path / "forward.jsonl"
    ledger = ForwardPaperLedger(path)
    ledger.append(_event())
    row = json.loads(path.read_text(encoding="utf-8"))
    row["payload"]["reference_price"] = 1
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(PaperLedgerIntegrityError, match="record_hash mismatch"):
        ledger.load()


def test_real_order_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="real_order=true"):
        _event(real_order=True)


def test_us_event_requires_usd_and_supports_benchmark() -> None:
    event = _event(
        market=PaperMarket.US,
        currency="USD",
        ticker="NVDA",
        benchmark_id="SPY",
        benchmark_return=0.01,
        exposure=0.25,
    )
    assert event.to_dict()["market"] == "US"
    assert event.benchmark_id == "SPY"

    with pytest.raises(ValueError, match="US paper events must use USD"):
        _event(market=PaperMarket.US, currency="KRW", ticker="NVDA")
