"""split_order 평균가 — M-2 결함수정 가드 (5/31).

체결가 미상(PENDING 0원) 분할이 avg_price를 0쪽으로 왜곡하지 않는지 검증.
수량/성공 집계(total_filled)는 기존대로 유지.
"""
from src.entities.trading_models import Order, OrderSide, OrderStatus, OrderType
from src.split_order import SplitOrderExecutor


class _SeqAdapter:
    """호출 순서대로 미리 준비한 Order를 반환."""
    def __init__(self, orders):
        self._orders = list(orders)
        self._i = 0

    def _next(self):
        o = self._orders[self._i]
        self._i += 1
        return o

    def buy_market(self, ticker, qty, **kw):
        return self._next()

    def sell_market(self, ticker, qty, **kw):
        return self._next()

    def fetch_current_price(self, ticker):
        return {"current_price": 1000}


def _mk(status, fq, fp):
    return Order(
        order_id="x", ticker="005930", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=1000, quantity=50,
        filled_quantity=fq, filled_price=fp, status=status,
    )


def test_avg_price_excludes_unpriced_pending():
    """split1 체결 50@1000 + split2 PENDING(0원) → avg=1000 (500 왜곡 아님)."""
    adapter = _SeqAdapter([
        _mk(OrderStatus.FILLED, 50, 1000),
        _mk(OrderStatus.PENDING, 0, 0),
    ])
    ex = SplitOrderExecutor(adapter)
    res = ex._execute_splits("005930", 100, 2, 0, side="buy")
    assert res.filled_quantity == 100          # 수량 집계는 그대로(시장가 체결 가정)
    assert res.success is True
    assert abs(res.avg_price - 1000.0) < 1e-6, f"avg_price 왜곡됨: {res.avg_price}"


def test_avg_price_normal_two_fills():
    """둘 다 체결: 50@1000 + 50@1200 → avg=1100 (회귀)."""
    adapter = _SeqAdapter([
        _mk(OrderStatus.FILLED, 50, 1000),
        _mk(OrderStatus.FILLED, 50, 1200),
    ])
    ex = SplitOrderExecutor(adapter)
    res = ex._execute_splits("005930", 100, 2, 0, side="buy")
    assert res.filled_quantity == 100
    assert abs(res.avg_price - 1100.0) < 1e-6, f"avg_price: {res.avg_price}"
