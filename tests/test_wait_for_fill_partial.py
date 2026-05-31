"""_wait_for_fill 부분체결/취소 회귀 테스트 — C-1/C-3 결함수정 가드 (5/31).

C-1: 부분체결 후 타임아웃 시 체결분을 FILLED로 보존 (유령 포지션 방지).
C-3: cancel() 실패해도 체결분 등록 + CANCELLED 단정 안 함.
회귀: 전량체결/완전미체결 경로는 기존대로 유지.
"""
from src.entities.trading_models import Order, OrderSide, OrderStatus, OrderType
from src.use_cases.live_trading import LiveTradingEngine


def _order(qty: int = 100) -> Order:
    return Order(
        order_id="X1", ticker="005930", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=1000, quantity=qty,
    )


class _Shim:
    """_wait_for_fill가 참조하는 속성만 가진 최소 self."""
    def __init__(self, port, timeout: int = 1):
        self.order_port = port
        self.cancel_after_sec = timeout


class _PartialPort:
    """60/100 부분체결 상태를 계속 반환."""
    def __init__(self, cancel_return: bool = True):
        self.cancel_called = False
        self._cr = cancel_return

    def get_order_status(self, order_id):
        return Order(order_id=order_id, status=OrderStatus.PARTIAL,
                     filled_quantity=60, filled_price=1000)

    def cancel(self, order):
        self.cancel_called = True
        return self._cr


def test_partial_fill_survives_timeout():
    """C-1: 60/100 부분체결 + 타임아웃 → FILLED 60주 (취소로 버리지 않음)."""
    port = _PartialPort(cancel_return=True)
    res = LiveTradingEngine._wait_for_fill(_Shim(port), _order(), timeout=1)
    assert res.status == OrderStatus.FILLED
    assert res.filled_quantity == 60
    assert port.cancel_called  # 잔여분은 취소


def test_partial_fill_survives_even_if_cancel_fails():
    """C-3: 취소 실패해도 체결분 보존 (CANCELLED 단정 금지)."""
    res = LiveTradingEngine._wait_for_fill(
        _Shim(_PartialPort(cancel_return=False)), _order(), timeout=1,
    )
    assert res.status == OrderStatus.FILLED
    assert res.filled_quantity == 60


def test_full_fill_unchanged():
    """회귀: 전량체결은 그대로 FILLED 100주."""
    class _FullPort(_PartialPort):
        def get_order_status(self, order_id):
            return Order(order_id=order_id, status=OrderStatus.FILLED,
                         filled_quantity=100, filled_price=1000)
    res = LiveTradingEngine._wait_for_fill(_Shim(_FullPort()), _order(), timeout=1)
    assert res.status == OrderStatus.FILLED
    assert res.filled_quantity == 100


def test_no_fill_cancelled():
    """회귀: 완전 미체결 → CANCELLED (없는 포지션 안 만듦)."""
    class _NoFillPort(_PartialPort):
        def get_order_status(self, order_id):
            return Order(order_id=order_id, status=OrderStatus.PENDING,
                         filled_quantity=0, filled_price=0)
    res = LiveTradingEngine._wait_for_fill(_Shim(_NoFillPort()), _order(), timeout=1)
    assert res.status == OrderStatus.CANCELLED
