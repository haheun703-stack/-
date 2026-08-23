from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.export_forward_paper_scoreboard import export_batch
from src.adapters.forward_paper_ledger import ForwardPaperLedger
from src.entities.forward_paper_event import ForwardPaperEvent, PaperEventType, PaperMarket
from src.use_cases.forward_paper_scoreboard import build_strategy_validation_batch

AS_OF = datetime.fromisoformat("2026-08-09T18:00:00+09:00")


def _event(event_id: str, event_type: PaperEventType, **overrides) -> ForwardPaperEvent:
    values = {
        "event_id": event_id,
        "trade_id": "trade-1",
        "event_type": event_type,
        "producer_bot": "quant",
        "strategy_id": "paper_wave_kr",
        "strategy_version": "abc123",
        "market": PaperMarket.KR,
        "currency": "KRW",
        "ticker": "069500",
        "event_at": "2026-08-08T15:30:00+09:00",
        "data_asof": "2026-08-08T15:30:00+09:00",
        "filled_quantity": 10,
        "metadata": {"cost_complete": False},
    }
    values.update(overrides)
    return ForwardPaperEvent(**values)


def test_reference_batch_computes_return_mdd_but_does_not_certify_costs() -> None:
    events = [
        _event(
            "fill",
            PaperEventType.FILL,
            fee=10,
            metadata={"normalized_equity": True, "cost_complete": False},
        ),
        _event(
            "mark-1",
            PaperEventType.MARK,
            trade_id="portfolio-1",
            ticker="__PORTFOLIO__",
            event_at="2026-08-06T15:30:00+09:00",
            metadata={"equity": 1000, "cost_complete": False},
        ),
        _event(
            "mark-2",
            PaperEventType.MARK,
            trade_id="portfolio-2",
            ticker="__PORTFOLIO__",
            event_at="2026-08-07T15:30:00+09:00",
            metadata={"equity": 800, "cost_complete": False},
        ),
        _event(
            "mark-3",
            PaperEventType.MARK,
            trade_id="portfolio-3",
            ticker="__PORTFOLIO__",
            metadata={"equity": 1100, "cost_complete": False},
        ),
        _event("exit", PaperEventType.EXIT),
    ]

    batch = build_strategy_validation_batch(events, as_of=AS_OF)
    result = batch["results"][0]
    assert result["strategy_return_pct"] == 10.0
    assert result["mdd_pct"] == -20.0
    assert result["trade_count"] == 1
    assert result["cost_complete"] is False
    assert result["same_exposure_benchmark"] is False
    assert result["benchmark_return_pct"] is None


def test_future_event_is_excluded_and_cross_market_trade_is_rejected() -> None:
    future = _event("future", PaperEventType.MARK, event_at="2026-08-10T15:30:00+09:00")
    current = _event("fill", PaperEventType.FILL)
    batch = build_strategy_validation_batch([current, future], as_of=AS_OF)
    assert "events=1" in batch["results"][0]["methodology_note"]

    us = _event("us-exit", PaperEventType.EXIT, market=PaperMarket.US, currency="USD", ticker="SPY")
    with pytest.raises(ValueError, match="crosses strategy or market"):
        build_strategy_validation_batch([current, us], as_of=AS_OF)


def test_partial_exits_are_allowed_but_over_exit_fails_closed() -> None:
    batch = build_strategy_validation_batch(
        [
            _event("fill", PaperEventType.FILL, filled_quantity=10),
            _event("exit-1", PaperEventType.EXIT, filled_quantity=4),
            _event("exit-2", PaperEventType.EXIT, filled_quantity=6),
        ],
        as_of=AS_OF,
    )
    assert batch["results"][0]["trade_count"] == 1

    with pytest.raises(ValueError, match="exceeds FILL"):
        build_strategy_validation_batch(
            [
                _event("fill", PaperEventType.FILL, filled_quantity=10),
                _event("exit-1", PaperEventType.EXIT, filled_quantity=7),
                _event("exit-2", PaperEventType.EXIT, filled_quantity=4),
            ],
            as_of=AS_OF,
        )


def test_export_verifies_ledger_and_writes_dated_and_latest(tmp_path) -> None:
    ledger_path = tmp_path / "events.jsonl"
    ledger = ForwardPaperLedger(ledger_path)
    ledger.append(_event("fill", PaperEventType.FILL))
    result = export_batch(ledger_path, tmp_path / "public", AS_OF)

    assert len(result["paths"]) == 2
    latest = json.loads((tmp_path / "public" / "strategy_validation_latest.json").read_text(encoding="utf-8"))
    assert latest["producer"] == "quant-bot"
    assert latest["results"][0]["cost_complete"] is False


def test_export_refuses_to_overwrite_source_ledger(tmp_path) -> None:
    source = tmp_path / "strategy_validation_latest.json"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite"):
        export_batch(source, tmp_path, AS_OF)


def test_cron_exports_scoreboard_immediately_after_paper_ledger_mirror() -> None:
    cron = (Path(__file__).parents[1] / "scripts" / "cron" / "run_bat.sh").read_text(
        encoding="utf-8"
    )
    mirror = "run_py scripts/mirror_forward_paper_ledger.py --profile all"
    export = (
        'run_py scripts/export_forward_paper_scoreboard.py '
        '--as-of "$(date --iso-8601=seconds)"'
    )

    assert cron.count(mirror) == 1
    assert cron.count(export) == 1
    assert cron.index(mirror) < cron.index(export)
    assert "--live" not in export
    assert "--real" not in export
