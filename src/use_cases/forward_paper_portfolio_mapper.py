"""Map legacy paper portfolio snapshots to immutable forward paper events."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.entities.forward_paper_event import ForwardPaperEvent, PaperEventType, PaperMarket


KST = ZoneInfo("Asia/Seoul")


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _timestamp(value: object, fallback_date: str) -> str:
    text = str(value or "").strip()
    candidates = (text, fallback_date)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=KST)
        return parsed.isoformat(timespec="seconds")
    raise ValueError(f"invalid paper event timestamp: value={value!r}, fallback={fallback_date!r}")


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _trade_identity(portfolio_id: str, row: dict[str, Any]) -> tuple[str, str, str, float, float]:
    ticker = str(row.get("ticker") or row.get("code") or "").strip()
    entry_date = str(row.get("orig_entry_date") or row.get("entry_date") or "").strip()
    entry_price = _float(row.get("avg_price") or row.get("entry_price")) or 0.0
    quantity = _float(row.get("qty") or row.get("quantity")) or 0.0
    if not ticker or not entry_date or entry_price <= 0 or quantity <= 0:
        raise ValueError(f"invalid legacy paper trade identity: {row!r}")
    trade_id = _stable_id("trade", portfolio_id, ticker, entry_date, entry_price)
    return trade_id, ticker, entry_date, entry_price, quantity


def _strategy_id(portfolio_id: str, row: dict[str, Any]) -> str:
    source = str(row.get("strategy") or "mixed").strip().lower()
    return f"{portfolio_id}:{source}"


def _fill_event(
    row: dict[str, Any], *, portfolio_id: str, strategy_version: str, source_name: str,
    reconstructed_quantity: float | None = None,
) -> ForwardPaperEvent:
    trade_id, ticker, entry_date, entry_price, quantity = _trade_identity(portfolio_id, row)
    if reconstructed_quantity is not None:
        quantity = reconstructed_quantity
    strategy_id = _strategy_id(portfolio_id, row)
    return ForwardPaperEvent(
        event_id=_stable_id("fill", trade_id),
        trade_id=trade_id,
        event_type=PaperEventType.FILL,
        producer_bot="quant",
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        market=PaperMarket.KR,
        currency="KRW",
        ticker=ticker,
        event_at=_timestamp(entry_date, entry_date),
        data_asof=_timestamp(entry_date, entry_date),
        side="BUY",
        status="FILLED",
        reference_price=entry_price,
        requested_quantity=quantity,
        requested_notional=round(entry_price * quantity, 4),
        filled_quantity=quantity,
        fill_price=entry_price,
        fill_ratio=1.0,
        source_record_id=f"{source_name}:position:{trade_id}",
        reason=str(row.get("reason") or "") or None,
        metadata={
            "portfolio_id": portfolio_id,
            "legacy_import": True,
            "cost_complete": False,
            "cost_note": "legacy entry does not expose fee, spread, and slippage separately",
            "quantity_reconstructed_from_snapshot": reconstructed_quantity is not None,
            "grade": row.get("grade"),
            "name": row.get("name"),
        },
    )


def _exit_event(
    row: dict[str, Any], *, portfolio_id: str, strategy_version: str, source_name: str
) -> ForwardPaperEvent:
    trade_id, ticker, entry_date, entry_price, quantity = _trade_identity(portfolio_id, row)
    exit_date = str(row.get("exit_date") or "").strip()
    exit_price = _float(row.get("exit_price"))
    if not exit_date or exit_price is None or exit_price <= 0:
        raise ValueError(f"invalid legacy paper exit: {row!r}")
    reason = str(row.get("exit_reason") or row.get("reason") or "UNKNOWN")
    source_pnl_pct = _float(row.get("pnl_pct"))
    event_id = _stable_id(
        "exit", trade_id, exit_date, exit_price, quantity, reason, source_pnl_pct
    )
    return ForwardPaperEvent(
        event_id=event_id,
        trade_id=trade_id,
        event_type=PaperEventType.EXIT,
        producer_bot="quant",
        strategy_id=_strategy_id(portfolio_id, row),
        strategy_version=strategy_version,
        market=PaperMarket.KR,
        currency="KRW",
        ticker=ticker,
        event_at=_timestamp(exit_date, exit_date),
        data_asof=_timestamp(exit_date, exit_date),
        side="SELL",
        status="FILLED",
        reference_price=exit_price,
        requested_quantity=quantity,
        requested_notional=round(exit_price * quantity, 4),
        filled_quantity=quantity,
        fill_price=exit_price,
        fill_ratio=1.0,
        net_pnl=None,
        net_return=None,
        source_record_id=f"{source_name}:closed:{event_id}",
        reason=reason,
        metadata={
            "portfolio_id": portfolio_id,
            "legacy_import": True,
            "cost_complete": False,
            "cost_note": "legacy pnl_pct is retained as source value, not certified net return",
            "source_pnl_pct": source_pnl_pct,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "days_held": row.get("days_held"),
            "partial": reason == "TAKE_PROFIT_T1",
            "name": row.get("name"),
        },
    )


def _mark_event(
    row: dict[str, Any], *, portfolio_id: str, strategy_version: str, source_name: str
) -> ForwardPaperEvent:
    mark_date = str(row.get("date") or "").strip()
    equity = _float(row.get("equity"))
    if not mark_date or equity is None or equity < 0:
        raise ValueError(f"invalid legacy paper equity mark: {row!r}")
    trade_id = _stable_id("portfolio", portfolio_id, mark_date)
    return ForwardPaperEvent(
        event_id=_stable_id("mark", portfolio_id, mark_date, equity),
        trade_id=trade_id,
        event_type=PaperEventType.MARK,
        producer_bot="quant",
        strategy_id=portfolio_id,
        strategy_version=strategy_version,
        market=PaperMarket.KR,
        currency="KRW",
        ticker="__PORTFOLIO__",
        event_at=_timestamp(mark_date, mark_date),
        data_asof=_timestamp(mark_date, mark_date),
        status="VALUED",
        source_record_id=f"{source_name}:equity:{mark_date}",
        metadata={
            "portfolio_id": portfolio_id,
            "equity": equity,
            "capital": _float(row.get("capital")),
            "positions": row.get("positions"),
            "stock_ratio": _float(row.get("stock_ratio")),
            "cost_complete": False,
        },
    )


def map_legacy_portfolio(
    snapshot: dict[str, Any], *, portfolio_id: str, strategy_version: str, source_name: str
) -> list[ForwardPaperEvent]:
    """Return deterministic events without writing files or changing the source snapshot."""
    if not isinstance(snapshot, dict):
        raise ValueError("paper portfolio snapshot must be a dict")
    events: dict[str, ForwardPaperEvent] = {}

    positions = snapshot.get("positions") or {}
    position_rows = [dict(row) for row in (positions.values() if isinstance(positions, dict) else positions)]
    closed_rows = [dict(row) for row in (snapshot.get("closed_trades") or [])]

    # Partial exits leave only the remaining quantity in positions. Reconstruct the
    # original paper fill from remaining + all closed tranches sharing one entry key.
    fill_groups: dict[str, dict[str, Any]] = {}
    for row in [*position_rows, *closed_rows]:
        trade_id, _, _, _, quantity = _trade_identity(portfolio_id, row)
        group = fill_groups.setdefault(trade_id, {"row": row, "quantity": 0.0})
        group["quantity"] += quantity
    for group in fill_groups.values():
        event = _fill_event(
            group["row"], portfolio_id=portfolio_id,
            strategy_version=strategy_version, source_name=source_name,
            reconstructed_quantity=group["quantity"],
        )
        events[event.event_id] = event

    for row in closed_rows:
        exit_event = _exit_event(
            row, portfolio_id=portfolio_id,
            strategy_version=strategy_version, source_name=source_name,
        )
        events[exit_event.event_id] = exit_event

    for raw in snapshot.get("daily_equity") or []:
        mark = _mark_event(
            dict(raw), portfolio_id=portfolio_id,
            strategy_version=strategy_version, source_name=source_name,
        )
        events[mark.event_id] = mark

    order = {PaperEventType.FILL: 0, PaperEventType.EXIT: 1, PaperEventType.MARK: 2}
    return sorted(events.values(), key=lambda event: (event.event_at, order[event.event_type], event.event_id))
