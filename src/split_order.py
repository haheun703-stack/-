"""
분할 주문 실행기 — 호가 충격 방지.

영상 인사이트: "5천만 원 시가에 한 번에 던져서 시세조작 주의 떴다"
→ 대량 주문을 2~3분할로 나눠서 시간 간격을 두고 실행.

사용법:
    from src.split_order import SplitOrderExecutor

    executor = SplitOrderExecutor(order_adapter)
    results = executor.buy_split("005930", quantity=100, splits=3, interval_sec=30)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.entities.trading_models import Order, OrderStatus

logger = logging.getLogger(__name__)

# 분할 매수 기준
SPLIT_THRESHOLD_AMOUNT = 10_000_000  # 1,000만원 이상이면 분할
DEFAULT_SPLITS = 3                   # 기본 3분할
DEFAULT_INTERVAL_SEC = 30            # 분할 간격 30초
MAX_SPLITS = 5                       # 최대 5분할


@dataclass
class SplitResult:
    """분할 주문 결과."""
    ticker: str
    total_quantity: int
    filled_quantity: int
    avg_price: float
    orders: list[Order]
    split_count: int
    success: bool

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0.0


class SplitOrderExecutor:
    """분할 주문 실행기.

    KisOrderAdapter를 래핑하여 대량 주문을 분할 실행.
    소량 주문은 그대로 1회 실행.
    """

    def __init__(self, order_adapter, config: dict | None = None):
        self.adapter = order_adapter
        cfg = config or {}
        self.split_threshold = cfg.get("split_threshold_amount", SPLIT_THRESHOLD_AMOUNT)
        self.default_splits = cfg.get("default_splits", DEFAULT_SPLITS)
        self.interval_sec = cfg.get("split_interval_sec", DEFAULT_INTERVAL_SEC)

    def buy_split(
        self,
        ticker: str,
        quantity: int,
        price: float = 0,
        splits: int = 0,
        interval_sec: int = 0,
    ) -> SplitResult:
        """분할 시장가 매수.

        Args:
            ticker: 종목코드
            quantity: 총 매수 수량
            price: 예상 단가 (분할 판단용, 0이면 현재가 조회)
            splits: 분할 횟수 (0이면 자동 판단)
            interval_sec: 분할 간격 초 (0이면 기본값)
        """
        if not interval_sec:
            interval_sec = self.interval_sec

        # 예상 주문 금액으로 분할 필요 여부 판단
        if price <= 0:
            try:
                data = self.adapter.fetch_current_price(ticker)
                price = data.get("current_price", 0)
            except Exception:
                price = 0

        estimated_amount = price * quantity if price > 0 else 0

        # 분할 횟수 결정
        if splits <= 0:
            splits = self._calc_splits(estimated_amount)

        if splits <= 1:
            # 분할 불필요 → 단일 주문
            order = self.adapter.buy_market(ticker, quantity)
            return SplitResult(
                ticker=ticker,
                total_quantity=quantity,
                filled_quantity=order.filled_quantity if order.status == OrderStatus.FILLED else 0,
                avg_price=order.filled_price,
                orders=[order],
                split_count=1,
                success=order.status in (OrderStatus.FILLED, OrderStatus.PENDING),
            )

        # 분할 실행
        return self._execute_splits(
            ticker, quantity, splits, interval_sec, side="buy"
        )

    def sell_split(
        self,
        ticker: str,
        quantity: int,
        price: float = 0,
        splits: int = 0,
        interval_sec: int = 0,
    ) -> SplitResult:
        """분할 시장가 매도."""
        if not interval_sec:
            interval_sec = self.interval_sec

        if price <= 0:
            try:
                data = self.adapter.fetch_current_price(ticker)
                price = data.get("current_price", 0)
            except Exception:
                price = 0

        estimated_amount = price * quantity if price > 0 else 0

        if splits <= 0:
            splits = self._calc_splits(estimated_amount)

        if splits <= 1:
            order = self.adapter.sell_market(ticker, quantity)
            return SplitResult(
                ticker=ticker,
                total_quantity=quantity,
                filled_quantity=order.filled_quantity if order.status == OrderStatus.FILLED else 0,
                avg_price=order.filled_price,
                orders=[order],
                split_count=1,
                success=order.status in (OrderStatus.FILLED, OrderStatus.PENDING),
            )

        return self._execute_splits(
            ticker, quantity, splits, interval_sec, side="sell"
        )

    def _calc_splits(self, estimated_amount: float) -> int:
        """주문 금액 기반 분할 횟수 자동 결정.

        < 1,000만: 1회 (분할 불필요)
        1,000만~2,000만: 2분할
        2,000만 이상: 3분할
        """
        if estimated_amount < self.split_threshold:
            return 1
        elif estimated_amount < self.split_threshold * 2:
            return 2
        else:
            return min(self.default_splits, MAX_SPLITS)

    def _execute_splits(
        self,
        ticker: str,
        total_qty: int,
        splits: int,
        interval_sec: int,
        side: str,
    ) -> SplitResult:
        """실제 분할 주문 실행."""
        # 수량 분배 (균등 분할, 나머지는 마지막에)
        base_qty = total_qty // splits
        remainder = total_qty % splits
        quantities = [base_qty] * splits
        quantities[-1] += remainder  # 나머지를 마지막 주문에

        # 0수량 제거
        quantities = [q for q in quantities if q > 0]
        actual_splits = len(quantities)

        logger.info(
            "[분할주문] %s %s: %d주를 %d분할 (%s), 간격 %d초",
            side.upper(), ticker, total_qty, actual_splits,
            "+".join(str(q) for q in quantities), interval_sec,
        )

        orders = []
        total_filled = 0
        total_cost = 0.0

        for i, qty in enumerate(quantities):
            if i > 0:
                logger.info("[분할주문] %d/%d 대기 (%d초)...", i + 1, actual_splits, interval_sec)
                time.sleep(interval_sec)

            logger.info("[분할주문] %d/%d 실행: %d주", i + 1, actual_splits, qty)

            if side == "buy":
                order = self.adapter.buy_market(ticker, qty)
            else:
                order = self.adapter.sell_market(ticker, qty)

            orders.append(order)

            if order.status in (OrderStatus.FILLED, OrderStatus.PENDING):
                filled = order.filled_quantity if order.filled_quantity > 0 else qty
                fill_price = order.filled_price if order.filled_price > 0 else 0
                total_filled += filled
                total_cost += filled * fill_price
            else:
                logger.warning(
                    "[분할주문] %d/%d 실패: %s", i + 1, actual_splits, order.message
                )
                # 실패 시 남은 분할 중단
                break

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        success = total_filled > 0

        logger.info(
            "[분할주문] 완료: %s %s %d/%d주 체결 (평균가 %.0f원)",
            side.upper(), ticker, total_filled, total_qty, avg_price,
        )

        return SplitResult(
            ticker=ticker,
            total_quantity=total_qty,
            filled_quantity=total_filled,
            avg_price=avg_price,
            orders=orders,
            split_count=actual_splits,
            success=success,
        )
