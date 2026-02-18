"""텔레그램 대화형 상태 관리 (FSM).

매수/매도/분석처럼 파라미터가 필요한 명령의 순차 입력을 관리.
Single-user 전제 (TELEGRAM_CHAT_ID 하나).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class ConvState(Enum):
    IDLE = "idle"
    # 매수
    BUY_WAIT_STOCK = "buy_wait_stock"
    BUY_WAIT_QTY = "buy_wait_qty"
    # 매도
    SELL_WAIT_STOCK = "sell_wait_stock"
    SELL_WAIT_QTY = "sell_wait_qty"
    # 분석
    ANALYZE_WAIT_STOCK = "analyze_wait_stock"


@dataclass
class ConversationContext:
    state: ConvState = ConvState.IDLE
    action: str = ""
    ticker: str = ""
    stock_name: str = ""
    quantity: int = 0
    price: int = 0
    extra: dict = field(default_factory=dict)


class ConversationManager:
    """사용자 대화 상태 관리."""

    TIMEOUT = 300  # 5분

    def __init__(self):
        self._ctx = ConversationContext()
        self._last_activity = time.time()

    @property
    def state(self) -> ConvState:
        return self._ctx.state

    @property
    def context(self) -> ConversationContext:
        return self._ctx

    def is_idle(self) -> bool:
        return self._ctx.state == ConvState.IDLE

    def reset(self) -> None:
        self._ctx = ConversationContext()
        self._last_activity = time.time()

    def set_state(self, state: ConvState, **kwargs) -> None:
        self._ctx.state = state
        for k, v in kwargs.items():
            if hasattr(self._ctx, k):
                setattr(self._ctx, k, v)
        self._last_activity = time.time()

    def check_timeout(self) -> bool:
        """타임아웃 발생 시 True 반환 + 자동 reset."""
        if not self.is_idle() and (time.time() - self._last_activity) > self.TIMEOUT:
            self.reset()
            return True
        return False
