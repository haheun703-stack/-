"""Map index benchmarks and KR/US wave paper books to forward events."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.entities.forward_paper_event import ForwardPaperEvent, PaperEventType, PaperMarket


FEE_RATE = 0.0005
TIMEZONES = {PaperMarket.KR: ZoneInfo("Asia/Seoul"), PaperMarket.US: ZoneInfo("America/New_York")}
WAVE_TICKERS = {
    "KODEX레버리지": "KODEX_LEV",
    "KODEX200": "KODEX200",
    "SSO(2x)": "SSO",
    "SPY": "SPY",
}


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _timestamp(value: object, market: PaperMarket) -> str:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TIMEZONES[market])
    return parsed.isoformat(timespec="seconds")


def _market(value: object) -> PaperMarket:
    return PaperMarket(str(value).upper())


def _currency(market: PaperMarket) -> str:
    return "KRW" if market == PaperMarket.KR else "USD"


def map_wave_book(
    snapshot: dict[str, Any], *, market: str, strategy_version: str, source_name: str
) -> list[ForwardPaperEvent]:
    """Convert a wave-rider paper book without promoting fee-only PnL to certified net PnL."""
    paper_market = _market(market)
    currency = _currency(paper_market)
    strategy_id = f"paper_wave_{paper_market.value.lower()}"
    events: dict[str, ForwardPaperEvent] = {}
    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)

    for ordinal, raw in enumerate(snapshot.get("trades") or []):
        row = dict(raw)
        side = str(row.get("side") or "").upper()
        name = str(row.get("name") or "").strip()
        ticker = WAVE_TICKERS.get(name, name)
        date = str(row.get("date") or "").strip()
        price = float(row.get("price") or 0)
        quantity = float(row.get("qty") or 0)
        if side not in {"BUY", "SELL"} or not ticker or not date or price <= 0 or quantity <= 0:
            raise ValueError(f"invalid wave trade: {row!r}")

        if side == "BUY":
            trade_id = _stable_id("trade", strategy_id, ticker, date, price, ordinal)
            open_lots[ticker].append({"trade_id": trade_id, "price": price, "quantity": quantity})
            event = ForwardPaperEvent(
                event_id=_stable_id("fill", trade_id), trade_id=trade_id,
                event_type=PaperEventType.FILL, producer_bot="quant",
                strategy_id=strategy_id, strategy_version=strategy_version,
                market=paper_market, currency=currency, ticker=ticker,
                event_at=_timestamp(date, paper_market), data_asof=_timestamp(row.get("price_date") or date, paper_market),
                side="BUY", status="FILLED", reference_price=price,
                requested_quantity=quantity, requested_notional=round(price * quantity, 8),
                filled_quantity=quantity, fill_price=price, fill_ratio=1.0,
                fee=round(price * quantity * FEE_RATE, 8),
                source_record_id=f"{source_name}:trade:{ordinal}",
                metadata={"name": name, "cost_complete": False, "fee_complete": True,
                          "cost_note": "paper model includes fee but not spread or slippage"},
            )
        else:
            if not open_lots[ticker]:
                raise ValueError(f"wave SELL has no matching BUY: {row!r}")
            lot = open_lots[ticker].popleft()
            if quantity != lot["quantity"]:
                raise ValueError(f"wave SELL quantity does not match BUY: {row!r}")
            declared_fee_return = (
                price * (1 - FEE_RATE) - lot["price"] * (1 + FEE_RATE)
            ) / (lot["price"] * (1 + FEE_RATE))
            event = ForwardPaperEvent(
                event_id=_stable_id("exit", lot["trade_id"], date, price, ordinal),
                trade_id=lot["trade_id"], event_type=PaperEventType.EXIT,
                producer_bot="quant", strategy_id=strategy_id, strategy_version=strategy_version,
                market=paper_market, currency=currency, ticker=ticker,
                event_at=_timestamp(date, paper_market), data_asof=_timestamp(date, paper_market),
                side="SELL", status="FILLED", reference_price=price,
                requested_quantity=quantity, requested_notional=round(price * quantity, 8),
                filled_quantity=quantity, fill_price=price, fill_ratio=1.0,
                fee=round(price * quantity * FEE_RATE, 8), net_pnl=None, net_return=None,
                reason=str(row.get("reason") or "") or None,
                source_record_id=f"{source_name}:trade:{ordinal}",
                metadata={"name": name, "entry_price": lot["price"],
                          "source_pnl_pct": row.get("pnl_pct"),
                          "declared_fee_net_return": declared_fee_return,
                          "cost_complete": False, "fee_complete": True,
                          "cost_note": "spread and slippage are absent; declared fee return is not certified net return"},
            )
        events[event.event_id] = event

    for row in snapshot.get("daily_equity") or []:
        date = str(row.get("date") or "").strip()
        equity = float(row.get("equity") or 0)
        if not date or equity < 0:
            raise ValueError(f"invalid wave equity mark: {row!r}")
        event = ForwardPaperEvent(
            event_id=_stable_id("mark", strategy_id, date, equity),
            trade_id=_stable_id("portfolio", strategy_id, date), event_type=PaperEventType.MARK,
            producer_bot="quant", strategy_id=strategy_id, strategy_version=strategy_version,
            market=paper_market, currency=currency, ticker="__PORTFOLIO__",
            event_at=_timestamp(date, paper_market), data_asof=_timestamp(row.get("asof") or date, paper_market),
            status="VALUED", benchmark_id="KODEX200" if paper_market == PaperMarket.KR else "SPY",
            exposure=0.0 if str(row.get("holding") or "").upper() in {"", "CASH", "현금"} else 1.0,
            regime=str(row.get("regime") or "") or None,
            source_record_id=f"{source_name}:equity:{date}",
            metadata={"equity": equity, "holding": row.get("holding"), "cost_complete": False},
        )
        events[event.event_id] = event

    order = {PaperEventType.FILL: 0, PaperEventType.EXIT: 1, PaperEventType.MARK: 2}
    return sorted(events.values(), key=lambda event: (event.event_at, order[event.event_type], event.event_id))


def map_index_benchmarks(
    snapshot: dict[str, Any], *, strategy_version: str, source_name: str
) -> list[ForwardPaperEvent]:
    """Convert normalized buy-and-hold series for KR and US benchmarks."""
    initial_capital = float(snapshot.get("initial_capital") or 0)
    if initial_capital <= 0:
        raise ValueError("index benchmark initial_capital must be positive")
    events: dict[str, ForwardPaperEvent] = {}
    for key, raw in (snapshot.get("benchmarks") or {}).items():
        row = dict(raw)
        market = _market(row.get("market"))
        currency = _currency(market)
        ticker = str(row.get("symbol") or key)
        entry_date = str(row.get("entry_date") or "")
        entry_price = float(row.get("entry_px") or 0)
        if not entry_date or entry_price <= 0:
            raise ValueError(f"invalid index benchmark: {key}")
        trade_id = _stable_id("benchmark", key, entry_date, entry_price)
        fill = ForwardPaperEvent(
            event_id=_stable_id("fill", trade_id), trade_id=trade_id,
            event_type=PaperEventType.FILL, producer_bot="quant", strategy_id=f"index_buyhold:{key}",
            strategy_version=strategy_version, market=market, currency=currency, ticker=ticker,
            event_at=_timestamp(entry_date, market), data_asof=_timestamp(entry_date, market),
            side="BUY", status="FILLED", reference_price=entry_price,
            requested_quantity=initial_capital / entry_price, requested_notional=initial_capital,
            filled_quantity=initial_capital / entry_price, fill_price=entry_price, fill_ratio=1.0,
            benchmark_id=str(key), exposure=1.0, source_record_id=f"{source_name}:{key}:entry",
            metadata={"normalized_equity": True, "cost_complete": False,
                      "cost_note": "buy-and-hold benchmark excludes execution costs",
                      "name": row.get("name"), "group": row.get("group"), "mult": row.get("mult")},
        )
        events[fill.event_id] = fill
        for daily in row.get("daily") or []:
            date = str(daily.get("date") or "")
            equity = float(daily.get("equity") or 0)
            benchmark_return = float(daily.get("return_pct") or 0) / 100.0
            mark = ForwardPaperEvent(
                event_id=_stable_id("mark", trade_id, date, equity), trade_id=trade_id,
                event_type=PaperEventType.MARK, producer_bot="quant", strategy_id=f"index_buyhold:{key}",
                strategy_version=strategy_version, market=market, currency=currency, ticker=ticker,
                event_at=_timestamp(date, market), data_asof=_timestamp(date, market), status="VALUED",
                reference_price=float(daily.get("px") or 0), benchmark_id=str(key),
                benchmark_return=benchmark_return, exposure=1.0,
                source_record_id=f"{source_name}:{key}:daily:{date}",
                metadata={"normalized_equity": equity, "cost_complete": False},
            )
            events[mark.event_id] = mark
    order = {PaperEventType.FILL: 0, PaperEventType.MARK: 1}
    return sorted(events.values(), key=lambda event: (event.event_at, order[event.event_type], event.event_id))
