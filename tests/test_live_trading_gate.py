"""RISK_ENGINE C-ii part2 — live_trading 매수 경로 게이트 배선 통합.

검증: _execute_single_buy가 build_gate_result를 거쳐 gate_result를 주문 포트에 전달하고,
게이트 REJECT 시 주문이 나가지 않음.
"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from risk.config import KST
from src.entities.trading_models import Order, OrderSide, OrderStatus, OrderType
from src.use_cases.live_trading import LiveTradingEngine

_KEY = "live-gate-test-key"
_ASOF = date(2026, 6, 10)


def _fresh_ohlcv(close=10000, volume=5000, rows=30):
    idx = pd.bdate_range(end=pd.Timestamp(_ASOF), periods=rows)
    return pd.DataFrame({"close": [close] * rows, "volume": [volume] * rows}, index=idx)


def _make_engine(order_port, balance_payload, ohlcv_df):
    tracker = MagicMock()
    tracker.positions = []
    guard = MagicMock()
    balance_port = MagicMock(fetch_balance=lambda: balance_payload)
    balance_port._is_mock = False  # REAL 강제 경로 모사(게이트 enforce=True)
    engine = LiveTradingEngine(
        order_port=order_port,
        balance_port=balance_port,
        price_port=MagicMock(),
        tracker=tracker,
        guard=guard,
        config={
            "live_trading": {"order": {"default_type": "market", "max_retry": 1}},
            "backtest": {
                "max_risk_pct": 0.005, "max_single_position_pct": 0.12,
                "max_portfolio_risk_pct": 0.02, "trailing_stop_atr_mult": 2.0,
            },
        },
        mode="live", executor_bot="quant",
        gate_ohlcv_loader=(lambda t: ohlcv_df),
        gate_sector_resolver=(lambda t: "반도체"),
    )
    engine.sizer.calculate = MagicMock(return_value={"shares": 10, "investment": 100_000})
    engine._wait_for_fill = MagicMock(side_effect=lambda o, **k: _filled(o))
    engine._send_buy_alert = MagicMock()
    return engine


def _filled(order):
    order.status = OrderStatus.FILLED
    order.filled_quantity = order.quantity
    order.filled_price = 10000
    return order


def _ok_balance(cash=10_000_000):
    return {"ok": True, "available_cash": cash, "holdings": [], "total_eval": 0}


def _pending_order(ticker="005930", qty=10):
    return Order(ticker=ticker, side=OrderSide.BUY, order_type=OrderType.MARKET,
                 price=0, quantity=qty, status=OrderStatus.PENDING)


@pytest.fixture(autouse=True)
def _hmac_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", _KEY)
    # 게이트 audit 로그를 tmp로 격리
    monkeypatch.setattr("src.use_cases.gate_wiring.DEFAULT_GATE_LOG_DIR", tmp_path / "gatelog")


# ── 1. 정상: gate_result(PASS 토큰)가 주문 포트로 전달 ───────────────────────
def test_buy_passes_gate_result_to_order_port():
    order_port = MagicMock()
    order_port.buy_market.return_value = _pending_order()
    engine = _make_engine(order_port, _ok_balance(), _fresh_ohlcv())

    signal = {"ticker": "005930", "entry_price": 10000, "grade": "A", "atr_value": 200}
    result = engine._execute_single_buy(signal, available_cash=5_000_000)

    assert result["success"] is True
    assert order_port.buy_market.called
    kwargs = order_port.buy_market.call_args.kwargs
    gate = kwargs.get("gate_result")
    assert gate is not None, "gate_result가 주문 포트로 전달돼야 함"
    assert gate.verdict in ("PASS", "RESIZE")
    assert gate.token and gate.signed
    assert gate.ticker == "005930"


# ── 2. 게이트 REJECT(잔고 조회 실패) → 주문 안 나감 ──────────────────────────
def test_gate_reject_blocks_order():
    order_port = MagicMock()
    engine = _make_engine(order_port, {"ok": False, "available_cash": 0, "holdings": []}, _fresh_ohlcv())

    signal = {"ticker": "005930", "entry_price": 10000, "grade": "A", "atr_value": 200}
    result = engine._execute_single_buy(signal, available_cash=5_000_000)

    assert result["success"] is False
    assert "게이트 거부" in result["reason"]
    assert not order_port.buy_market.called  # 주문 자체가 나가지 않음


# ── 3. adv stale → REJECT → 주문 안 나감 ─────────────────────────────────────
def test_gate_reject_on_stale_data_blocks_order():
    order_port = MagicMock()
    stale = _fresh_ohlcv()
    stale.index = pd.bdate_range(end=pd.Timestamp(date(2026, 5, 20)), periods=len(stale))  # 오래된 봉
    engine = _make_engine(order_port, _ok_balance(), stale)

    signal = {"ticker": "005930", "entry_price": 10000, "grade": "A", "atr_value": 200}
    result = engine._execute_single_buy(signal, available_cash=5_000_000)

    assert result["success"] is False and not order_port.buy_market.called
