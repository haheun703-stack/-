"""FLOWX forward paper event entities.

Pure domain objects shared by KR/US paper producers. They contain no file,
broker, scheduler, or network behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class PaperEventType(str, Enum):
    SIGNAL_FROZEN = "SIGNAL_FROZEN"
    ORDER_INTENT = "ORDER_INTENT"
    FILL = "FILL"
    REJECT = "REJECT"
    MARK = "MARK"
    EXIT = "EXIT"


class PaperMarket(str, Enum):
    KR = "KR"
    US = "US"


@dataclass(frozen=True)
class ForwardPaperEvent:
    event_id: str
    trade_id: str
    event_type: PaperEventType
    producer_bot: str
    strategy_id: str
    strategy_version: str
    market: PaperMarket
    currency: str
    ticker: str
    event_at: str
    data_asof: str
    side: str | None = None
    status: str | None = None
    signal_at: str | None = None
    eligible_at: str | None = None
    reference_price: float | None = None
    order_type: str | None = None
    requested_quantity: float | None = None
    requested_notional: float | None = None
    filled_quantity: float | None = None
    fill_price: float | None = None
    fill_ratio: float | None = None
    fee: float = 0.0
    tax: float = 0.0
    slippage_cost: float = 0.0
    spread_cost: float = 0.0
    net_pnl: float | None = None
    net_return: float | None = None
    benchmark_id: str | None = None
    benchmark_return: float | None = None
    exposure: float | None = None
    regime: str | None = None
    reason: str | None = None
    source_record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    real_order: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        required = {
            "event_id": self.event_id,
            "trade_id": self.trade_id,
            "producer_bot": self.producer_bot,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "currency": self.currency,
            "ticker": self.ticker,
            "event_at": self.event_at,
            "data_asof": self.data_asof,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"forward paper event required fields missing: {', '.join(missing)}")
        if self.real_order:
            raise ValueError("forward paper ledger rejects real_order=true")
        if self.market == PaperMarket.KR and self.currency != "KRW":
            raise ValueError("KR paper events must use KRW")
        if self.market == PaperMarket.US and self.currency != "USD":
            raise ValueError("US paper events must use USD")

        non_negative = {
            "reference_price": self.reference_price,
            "requested_quantity": self.requested_quantity,
            "requested_notional": self.requested_notional,
            "filled_quantity": self.filled_quantity,
            "fill_price": self.fill_price,
            "fee": self.fee,
            "tax": self.tax,
            "slippage_cost": self.slippage_cost,
            "spread_cost": self.spread_cost,
        }
        for name, value in non_negative.items():
            if value is not None and (not isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.fill_ratio is not None and not 0.0 <= self.fill_ratio <= 1.0:
            raise ValueError("fill_ratio must be between 0 and 1")
        if self.exposure is not None and not 0.0 <= self.exposure <= 1.0:
            raise ValueError("exposure must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        payload["market"] = self.market.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForwardPaperEvent":
        values = dict(payload)
        values["event_type"] = PaperEventType(values["event_type"])
        values["market"] = PaperMarket(values["market"])
        return cls(**values)
