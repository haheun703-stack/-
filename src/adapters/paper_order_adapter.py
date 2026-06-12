"""Paper Order Adapter — 실주문 미러 시뮬레이터 (§13-2-1 명세, 2026-05-19)

배경: 5/18 사장님 결단 옵션 B — 5/20 자비스 자율 매매 첫 가동 시 paper mirror 병행
- 실주문 (auto_buy_executor) 결정에 영향 0 (관찰자 역할)
- 같은 시그널·같은 후보·같은 타이밍으로 시뮬 진입
- 슬리피지·수수료·거래세 가정 누적 → 5/21+ 슬리피지 가정 실측 보정

§13-2-1 (2) 체결 시뮬레이션 가정:
  슬리피지 (매수): 현재가 +0.05% (호가창 미수신 fallback +0.10%)
  슬리피지 (매도): 현재가 -0.05% (호가창 미수신 fallback -0.10%)
  수수료: 매수·매도 각 0.015%
  거래세: 매도 시 0.18% (코스피) / 0.20% (코스닥) — v1은 코스피 디폴트 (보수)
  호가 단위: kis_order_adapter._adjust_to_tick 재사용
  체결 수량: 100% 즉시 (10만원 사이즈 한정 가정)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.adapters.kis_order_adapter import KisOrderAdapter
from src.entities.trading_models import Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)

# §13-2-1 (8) 환경변수 (5/20 가동 디폴트)
DEFAULT_SLIPPAGE_BASE_PCT = 0.05
DEFAULT_SLIPPAGE_FALLBACK_PCT = 0.10
FEE_PCT = 0.015  # 매수·매도 각 0.015% (KIS 키움 동일)
TAX_PCT_KOSPI = 0.18  # 매도 시 (2026 현행)
TAX_PCT_KOSDAQ = 0.20


class PaperOrderAdapter:
    """KisOrderAdapter 인터페이스 미러 — 실주문 X, 시뮬만.

    인터페이스 동일:
      - buy_limit(ticker, price, qty) -> Order
      - sell_limit(ticker, price, qty) -> Order
      - buy_market(ticker, qty) -> Order
      - sell_market(ticker, qty) -> Order

    차이:
      - 실제 KIS API 호출 없음 (mojito broker 미사용)
      - filled_price = 가정 슬리피지 적용 결과
      - order_id = "PAPER_{YYYYMMDDHHMMSS}_{ticker}"
      - status = 즉시 FILLED (PENDING 단계 생략)
    """

    def __init__(self):
        self.slippage_base = float(os.getenv("PAPER_MIRROR_SLIPPAGE_BASE_PCT", DEFAULT_SLIPPAGE_BASE_PCT))
        self.slippage_fallback = float(os.getenv("PAPER_MIRROR_SLIPPAGE_FALLBACK_PCT", DEFAULT_SLIPPAGE_FALLBACK_PCT))
        self._market_data = None
        logger.info(
            "PaperOrderAdapter 초기화 — slippage_base=%.2f%%, fallback=%.2f%%",
            self.slippage_base, self.slippage_fallback,
        )

    def _get_market_data(self):
        """Lazy read-only KIS adapter for price/OHLCV data in paper mode."""
        if self._market_data is None:
            self._market_data = KisOrderAdapter()
        return self._market_data

    # ──────────────────────────────────────────
    # 슬리피지·수수료·거래세 계산
    # ──────────────────────────────────────────
    @staticmethod
    def _get_tick_size(price: int) -> int:
        """호가 단위 테이블 (2023+, kis_order_adapter._adjust_to_tick과 동일)."""
        if price < 1000: return 1
        if price < 5000: return 5
        if price < 20000: return 10
        if price < 50000: return 50
        if price < 200000: return 100
        if price < 500000: return 500
        return 1000

    @staticmethod
    def _round_to_tick(price: int, ceil: bool = False) -> int:
        """호가 단위 보정 — 매수는 ceil, 매도는 floor (슬리피지 보존)."""
        tick = PaperOrderAdapter._get_tick_size(price)
        if ceil:
            return ((price + tick - 1) // tick) * tick
        return (price // tick) * tick

    def _apply_slippage(self, price: int, side: OrderSide, orderbook_available: bool = False) -> int:
        """슬리피지 적용 후 체결가.

        매수는 호가 단위 올림 (불리), 매도는 내림 (불리) — 슬리피지 보존.
        """
        slip_pct = self.slippage_base if orderbook_available else self.slippage_fallback
        if side == OrderSide.BUY:
            filled = price * (1 + slip_pct / 100.0)
            return self._round_to_tick(int(filled), ceil=True)
        else:
            filled = price * (1 - slip_pct / 100.0)
            return self._round_to_tick(int(filled), ceil=False)

    @staticmethod
    def _calc_fee(filled_price: int, qty: int) -> int:
        """수수료 (원 단위 반올림)."""
        return round(filled_price * qty * FEE_PCT / 100.0)

    @staticmethod
    def _calc_tax(filled_price: int, qty: int, market: str = "KOSPI") -> int:
        """거래세 (매도 시만). v1은 코스피 디폴트.

        Args:
            market: "KOSPI" 또는 "KOSDAQ" (5/21+ 종목코드 → 시장 매핑 후 정확화)
        """
        tax_pct = TAX_PCT_KOSDAQ if market == "KOSDAQ" else TAX_PCT_KOSPI
        return round(filled_price * qty * tax_pct / 100.0)

    @staticmethod
    def _make_order_id(ticker: str) -> str:
        return f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S')}_{ticker}"

    def _record_pilot_order(self, order: Order, *, fee: int = 0, tax: int = 0,
                            cash_effect: int = 0) -> None:
        """Append paper fills to the 3-day pilot order journal."""
        if os.getenv("QUANT_3DAY_PILOT", "0") != "1":
            return
        try:
            root = Path(__file__).resolve().parents[2]
            out = root / "results" / "quant_3day_pilot" / "paper_orders.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "run_id": os.getenv("QUANT_3DAY_PILOT_RUN_ID", ""),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "order_id": order.order_id,
                "ticker": order.ticker,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "price": order.price,
                "quantity": order.quantity,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "filled_price": order.filled_price,
                "fee": fee,
                "tax": tax,
                "cash_effect": cash_effect,
                "message": order.message,
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[PAPER] pilot order journal write failed: %s", e)

    # ──────────────────────────────────────────
    # 주문 인터페이스 (KisOrderAdapter 미러)
    # ──────────────────────────────────────────
    def buy_limit(
        self, ticker: str, price: int, quantity: int,
        orderbook_available: bool = False,
        *, mode: str | None = None, executor_bot: str | None = None,
        gate_result: object | None = None,
    ) -> Order:
        """지정가 매수 시뮬.

        Trading Factory v1 (코덱스 4차 5/28): mode + executor_bot 명시 시 order_intents_gate 강제.
        P0-2: PaperOrderAdapter는 mode='paper'만 허용 (mode='live' 즉시 차단).
        RISK_ENGINE Phase 1b: gate_result는 인터페이스 정합성 위해 수용하나 페이퍼는 실탄이
            아니므로 강제하지 않는다(실전 강제는 KisOrderAdapter 전담). 무시.
        """
        _ = gate_result  # 페이퍼는 통행증 강제 안 함(인터페이스 호환만)
        # 10번째 가드 — order_intents_gate (mode/executor_bot 명시 시 강제)
        if mode is not None or executor_bot is not None:
            # P0-2 (코덱스 4차 응답): PaperOrderAdapter는 mode='live' 거부
            # paper 시뮬 어댑터 → live 권한 없음
            if mode != "paper":
                raise ValueError(
                    f"[PAPER GUARD] PaperOrderAdapter는 mode='paper'만 허용 (received='{mode}'). "
                    "live 매매는 KisOrderAdapter 사용. "
                    "mode/executor_bot 인자는 둘 다 명시 또는 둘 다 생략 (backward compat)."
                )
            if executor_bot is None:
                raise ValueError(
                    "[PAPER GUARD] PaperOrderAdapter.buy_limit: mode 명시 시 executor_bot도 명시 필수."
                )
            from src.use_cases.order_intents_gate import assert_order_intent_exists
            assert_order_intent_exists(
                ticker=ticker, side="BUY", mode=mode, executor_bot=executor_bot,
            )
        adjusted = KisOrderAdapter._adjust_to_tick(price)
        filled_price = self._apply_slippage(adjusted, OrderSide.BUY, orderbook_available)
        fee = self._calc_fee(filled_price, quantity)
        total_cost = filled_price * quantity + fee
        logger.info(
            "[PAPER 매수] %s %d주 지정가 %d → 체결 %d (슬리피지 %s, 수수료 %d원, 총 %d원)",
            ticker, quantity, adjusted, filled_price,
            f"{self.slippage_base if orderbook_available else self.slippage_fallback}%",
            fee, total_cost,
        )
        order = Order(
            order_id=self._make_order_id(ticker),
            ticker=ticker,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=adjusted,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=float(filled_price),
            message=f"PAPER_FILLED fee={fee} total={total_cost}",
        )
        self._record_pilot_order(order, fee=fee, cash_effect=-total_cost)
        return order

    def sell_limit(
        self, ticker: str, price: int, quantity: int,
        orderbook_available: bool = False, market: str = "KOSPI",
        *, mode: str | None = None, executor_bot: str | None = None,
    ) -> Order:
        """지정가 매도 시뮬.

        Trading Factory v1 (코덱스 4차 5/28): mode + executor_bot 명시 시 order_intents_gate 강제.
        P0-2: PaperOrderAdapter는 mode='paper'만 허용 (mode='live' 즉시 차단).
        """
        # 10번째 가드 — order_intents_gate
        if mode is not None or executor_bot is not None:
            # P0-2 (코덱스 4차 응답): PaperOrderAdapter는 mode='live' 거부
            if mode != "paper":
                raise ValueError(
                    f"[PAPER GUARD] PaperOrderAdapter는 mode='paper'만 허용 (received='{mode}'). "
                    "live 매매는 KisOrderAdapter 사용."
                )
            if executor_bot is None:
                raise ValueError(
                    "[PAPER GUARD] PaperOrderAdapter.sell_limit: mode 명시 시 executor_bot도 명시 필수."
                )
            from src.use_cases.order_intents_gate import assert_order_intent_exists
            assert_order_intent_exists(
                ticker=ticker, side="SELL", mode=mode, executor_bot=executor_bot,
            )
        adjusted = KisOrderAdapter._adjust_to_tick(price)
        filled_price = self._apply_slippage(adjusted, OrderSide.SELL, orderbook_available)
        fee = self._calc_fee(filled_price, quantity)
        tax = self._calc_tax(filled_price, quantity, market)
        net_proceeds = filled_price * quantity - fee - tax
        logger.info(
            "[PAPER 매도] %s %d주 지정가 %d → 체결 %d (슬리피지 %s, 수수료 %d원, 거래세 %d원, 순수령 %d원)",
            ticker, quantity, adjusted, filled_price,
            f"{self.slippage_base if orderbook_available else self.slippage_fallback}%",
            fee, tax, net_proceeds,
        )
        order = Order(
            order_id=self._make_order_id(ticker),
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=adjusted,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=float(filled_price),
            message=f"PAPER_FILLED fee={fee} tax={tax} net={net_proceeds}",
        )
        self._record_pilot_order(order, fee=fee, tax=tax, cash_effect=net_proceeds)
        return order

    def buy_market(
        self, ticker: str, quantity: int, current_price: int = 0,
        orderbook_available: bool = False,
        *, mode: str | None = None, executor_bot: str | None = None,
        gate_result: object | None = None,
    ) -> Order:
        """시장가 매수 시뮬.

        ★ C1 fix (5/26): current_price 옵셔널.
        Trading Factory v1: mode + executor_bot 명시 시 order_intents_gate 강제.
        RISK_ENGINE Phase 1b: gate_result 수용하나 페이퍼는 강제 안 함(KisOrderAdapter 전담).
        """
        if current_price <= 0:
            try:
                res = self.fetch_price(ticker)
                current_price = int(res.get("output", {}).get("stck_prpr", 0))
            except Exception:
                current_price = 1
        return self.buy_limit(
            ticker, current_price, quantity, orderbook_available,
            mode=mode, executor_bot=executor_bot, gate_result=gate_result,
        )

    def sell_market(
        self, ticker: str, quantity: int, current_price: int = 0,
        orderbook_available: bool = False, market: str = "KOSPI",
        *, mode: str | None = None, executor_bot: str | None = None,
    ) -> Order:
        """시장가 매도 시뮬.

        ★ C1 fix (5/26): current_price 옵셔널.
        Trading Factory v1: mode + executor_bot 명시 시 order_intents_gate 강제.
        """
        if current_price <= 0:
            try:
                res = self.fetch_price(ticker)
                current_price = int(res.get("output", {}).get("stck_prpr", 0))
            except Exception:
                current_price = 1
        return self.sell_limit(
            ticker, current_price, quantity, orderbook_available, market,
            mode=mode, executor_bot=executor_bot,
        )

    def fetch_price(self, ticker: str) -> dict:
        """Read-only market price passthrough for paper-mode gates."""
        return self._get_market_data().fetch_price(ticker)

    def fetch_current_price(self, ticker: str) -> dict:
        """Processed current-price passthrough for paper-mode consumers."""
        return self._get_market_data().fetch_current_price(ticker)

    def fetch_ohlcv(self, ticker: str, timeframe: str = "D",
                    start_day: str = "", end_day: str = "", adj_price: bool = True):
        """Read-only OHLCV passthrough for paper-mode entry/exit gates."""
        return self._get_market_data().fetch_ohlcv(
            ticker,
            timeframe=timeframe,
            start_day=start_day,
            end_day=end_day,
            adj_price=adj_price,
        )

    def fetch_today_1m_ohlcv(self, ticker: str):
        """Read-only intraday OHLCV passthrough when a gate requests it."""
        return self._get_market_data().fetch_today_1m_ohlcv(ticker)

    def fetch_balance(self) -> dict:
        """Paper account snapshot detached from the real KIS account."""
        return {"holdings": [], "total_eval": 0, "total_pnl": 0, "available_cash": 0, "ok": True}

    def fetch_holdings(self) -> list[dict]:
        return self.fetch_balance()["holdings"]

    def get_available_cash(self) -> float:
        return float(self.fetch_balance()["available_cash"])
