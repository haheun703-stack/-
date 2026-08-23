"""Build a FLOWX strategy-validation batch from verified paper events."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from src.entities.forward_paper_event import ForwardPaperEvent, PaperEventType


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"paper event timestamp must be timezone-aware: {value}")
    return parsed.astimezone(UTC)


def _equity(event: ForwardPaperEvent) -> float | None:
    if event.event_type != PaperEventType.MARK:
        return None
    raw = event.metadata.get("equity", event.metadata.get("normalized_equity"))
    if raw is None or isinstance(raw, bool):
        return None
    value = float(raw)
    return value if isfinite(value) and value > 0 else None


def _return_and_mdd(events: list[ForwardPaperEvent]) -> tuple[float | None, float | None, int]:
    marks = sorted(
        ((event.event_at, value) for event in events if (value := _equity(event)) is not None),
        key=lambda item: item[0],
    )
    if len(marks) < 2:
        return None, None, len(marks)

    first = marks[0][1]
    strategy_return = (marks[-1][1] / first - 1.0) * 100.0
    peak = first
    mdd = 0.0
    for _, equity in marks:
        peak = max(peak, equity)
        mdd = min(mdd, (equity / peak - 1.0) * 100.0)
    return round(strategy_return, 6), round(mdd, 6), len(marks)


def _cost_evidence(events: list[ForwardPaperEvent]) -> tuple[bool, dict[str, bool | None]]:
    transactions = [event for event in events if event.event_type in {PaperEventType.FILL, PaperEventType.EXIT}]
    commission = bool(transactions) and all(
        event.metadata.get("commission_complete") is True or event.metadata.get("fee_complete") is True
        for event in transactions
    )
    tax = bool(transactions) and all(event.metadata.get("tax_complete") is True for event in transactions)
    slippage = bool(transactions) and all(
        event.metadata.get("slippage_complete") is True and event.metadata.get("spread_complete") is True
        for event in transactions
    )
    declared_complete = bool(transactions) and all(
        event.metadata.get("cost_complete") is True for event in transactions
    )
    complete = declared_complete and commission and tax and slippage
    return complete, {
        "commission": commission,
        "tax": tax,
        "slippage": slippage,
        "borrow": None,
        "borrow_required": False,
    }


def build_strategy_validation_batch(events: Iterable[ForwardPaperEvent], *, as_of: datetime) -> dict:
    """Aggregate immutable paper events without claiming unproven net performance."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = as_of.astimezone(UTC)

    groups: dict[tuple[str, str, str, str], list[ForwardPaperEvent]] = defaultdict(list)
    trade_identity: dict[str, tuple[str, str, str, str, str]] = {}
    trade_quantities: dict[str, dict[str, object]] = {}

    for event in events:
        if event.real_order:
            raise ValueError("strategy validation rejects real_order=true")
        if _instant(event.event_at) > cutoff:
            continue
        identity = (
            event.producer_bot,
            event.strategy_id,
            event.market.value,
            event.currency,
            event.ticker,
        )
        previous = trade_identity.setdefault(event.trade_id, identity)
        if previous != identity:
            raise ValueError(f"trade_id crosses strategy or market identity: {event.trade_id}")
        group_key = identity[:4]
        if event.event_type in {PaperEventType.FILL, PaperEventType.EXIT}:
            quantity = float(event.filled_quantity or 0.0)
            if quantity <= 0:
                raise ValueError(f"transaction event has no filled quantity: {event.event_id}")
            totals = trade_quantities.setdefault(event.trade_id, {"fill": 0.0, "exit": 0.0, "group": group_key})
            bucket = "fill" if event.event_type == PaperEventType.FILL else "exit"
            totals[bucket] = float(totals[bucket]) + quantity
        groups[group_key].append(event)

    closed_trades: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for trade_id, totals in trade_quantities.items():
        filled = float(totals["fill"])
        exited = float(totals["exit"])
        tolerance = max(1e-9, filled * 1e-9)
        if filled <= 0 and exited > 0:
            raise ValueError(f"EXIT has no matching FILL: {trade_id}")
        if exited > filled + tolerance:
            raise ValueError(f"EXIT quantity exceeds FILL quantity: {trade_id}")
        if filled > 0 and abs(exited - filled) <= tolerance:
            closed_trades[totals["group"]].add(trade_id)

    if not groups:
        raise ValueError("no paper events at or before as_of")
    if len(groups) > 100:
        raise ValueError("strategy validation batch exceeds 100 results")

    results = []
    for key in sorted(groups):
        producer, strategy_id, market, _currency = key
        strategy_events = groups[key]
        ordered_events = sorted(strategy_events, key=lambda event: _instant(event.event_at))
        strategy_version = ordered_events[-1].strategy_version
        version_count = len({event.strategy_version for event in strategy_events})
        strategy_return, mdd, mark_count = _return_and_mdd(strategy_events)
        cost_complete, costs = _cost_evidence(strategy_events)
        closed_trade_count = len(closed_trades[key])
        event_dates = sorted(_instant(event.event_at).date().isoformat() for event in strategy_events)
        data_as_of = max(_instant(event.data_asof) for event in strategy_events)
        period_start, period_end = event_dates[0], event_dates[-1]

        results.append(
            {
                "strategy_id": strategy_id,
                "market": market,
                "strategy_name": strategy_id,
                "strategy_version": strategy_version,
                "strategy_return_pct": strategy_return,
                "benchmark_return_pct": None,
                "excess_return_pct": None,
                "benchmark_label": None,
                "same_exposure_benchmark": False,
                "mdd_pct": mdd,
                "trade_count": closed_trade_count,
                "period_start": period_start,
                "period_end": period_end,
                "period_label": f"{period_start}~{period_end}",
                "cost_complete": cost_complete,
                "costs": costs,
                "data_as_of": data_as_of.isoformat().replace("+00:00", "Z"),
                "source_label": "quant forward paper ledger (hash-chain verified)",
                "methodology_note": (
                    f"paper-only; events={len(strategy_events)}; closed={closed_trade_count}; "
                    f"equity_marks={mark_count}; versions={version_count}; "
                    f"latest_version={strategy_version}; unmatched-exposure benchmark not claimed"
                ),
            }
        )

    generated_at = cutoff.isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.0",
        "run_id": f"quant-{cutoff.strftime('%Y%m%dT%H%M%SZ')}",
        "producer": "quant-bot",
        "generated_at": generated_at,
        "results": results,
    }
