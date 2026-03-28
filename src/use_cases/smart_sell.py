"""Smart Sell Executor — X1~X5 유형별 매도 전략 (EX-4)

X1 하드스톱: 즉시 시장가 (슬리피지 감수)
X2 수급이탈: 지정가 5분 → 시장가
X3 트레일링: 지정가 2분 → 시장가
X4 시간제한: VWAP 기준 15분 대기 → 현재가 정정 → 5분 → 시장가
X5 목표달성: 지정가 5분 → 시장가 (프리미엄 최대)

toggle: execution_alpha.smart_sell.enabled: false → 기존 매도 로직 폴백
"""

from __future__ import annotations

import logging
import time

from src.entities.trading_models import Order, OrderSide, OrderType, OrderStatus, tick_round

logger = logging.getLogger(__name__)


class SmartSellExecutor:
    """X1~X5 유형별 스마트 매도."""

    def __init__(self, settings: dict, order_adapter=None, intraday_adapter=None):
        ea_cfg = settings.get("execution_alpha", {})
        self.sell_cfg = ea_cfg.get("smart_sell", {})
        self.enabled = self.sell_cfg.get("enabled", False)
        self.order = order_adapter
        self.intraday = intraday_adapter

    def execute(
        self,
        ticker: str,
        qty: int,
        exit_rule: str,
        current_price: int,
        dry_run: bool = True,
    ) -> dict:
        """유형별 매도 전략 실행.

        Returns:
            {"method": str, "price": int, "qty": int, "filled": bool, "detail": str}
        """
        rule = exit_rule.upper() if exit_rule else ""

        if rule == "X1":
            return self._market_immediate(ticker, qty, current_price, dry_run)
        elif rule == "X5":
            premium = self.sell_cfg.get("x5_premium_pct", 0.3)
            timeout = self.sell_cfg.get("x5_timeout_min", 5)
            return self._limit_then_market(
                ticker, qty, current_price, premium, timeout, dry_run
            )
        elif rule == "X3":
            premium = self.sell_cfg.get("x3_premium_pct", 0.2)
            timeout = self.sell_cfg.get("x3_timeout_min", 2)
            return self._limit_then_market(
                ticker, qty, current_price, premium, timeout, dry_run
            )
        elif rule == "X4":
            return self._patient_sell(ticker, qty, current_price, dry_run)
        elif rule == "X2":
            premium = self.sell_cfg.get("x2_premium_pct", 0.1)
            timeout = self.sell_cfg.get("x2_timeout_min", 5)
            return self._limit_then_market(
                ticker, qty, current_price, premium, timeout, dry_run
            )
        else:
            logger.warning("SmartSell: 알 수 없는 exit_rule=%s → 시장가 폴백", rule)
            return self._market_immediate(ticker, qty, current_price, dry_run)

    # ── 매도 전략 구현 ──────────────────────────────

    def _market_immediate(
        self, ticker: str, qty: int, price: int, dry_run: bool
    ) -> dict:
        """X1: 즉시 시장가 (슬리피지 감수)."""
        logger.info(
            "SmartSell X1: %s %d주 시장가 매도 (dry_run=%s)", ticker, qty, dry_run
        )
        if dry_run or not self.order:
            return {
                "method": "market_immediate",
                "price": price,
                "qty": qty,
                "filled": True,
                "detail": "X1 하드스톱 → 즉시 시장가",
            }

        result = self.order.sell_market(ticker, qty)
        filled = result.status.value != "failed"
        return {
            "method": "market_immediate",
            "price": int(result.filled_price or price),
            "qty": qty,
            "filled": filled,
            "detail": f"X1 시장가 → {'체결' if filled else '실패'}",
            "order_id": result.order_id,
        }

    def _limit_then_market(
        self,
        ticker: str,
        qty: int,
        price: int,
        premium_pct: float,
        timeout_min: int,
        dry_run: bool,
    ) -> dict:
        """지정가 접수 → timeout 후 미체결 시 시장가 전환."""
        limit_price = tick_round(
            int(price * (1 + premium_pct / 100)), price
        )
        logger.info(
            "SmartSell: %s %d주 지정가 %d (+%.1f%%, %d분 대기, dry_run=%s)",
            ticker, qty, limit_price, premium_pct, timeout_min, dry_run,
        )

        if dry_run or not self.order:
            return {
                "method": "limit_then_market",
                "price": limit_price,
                "qty": qty,
                "filled": True,
                "detail": f"지정가 {limit_price:,}원 → {timeout_min}분 → 시장가",
            }

        # 1단계: 지정가 접수
        order = self.order.sell_limit(ticker, limit_price, qty)
        if order.status.value == "failed":
            # 지정가 실패 → 즉시 시장가
            logger.warning("SmartSell: %s 지정가 실패 → 시장가 전환", ticker)
            return self._market_immediate(ticker, qty, price, dry_run)

        # 2단계: timeout 동안 30초 간격 체결 확인
        check_interval = 30
        max_checks = (timeout_min * 60) // check_interval
        for i in range(max_checks):
            time.sleep(check_interval)
            status = self.order.get_order_status(order.order_id)
            if status.status.value == "filled":
                filled_price = int(status.filled_price or limit_price)
                logger.info(
                    "SmartSell: %s 지정가 체결 %d원 (%d/%d초)",
                    ticker, filled_price, (i + 1) * check_interval, timeout_min * 60,
                )
                return {
                    "method": "limit_filled",
                    "price": filled_price,
                    "qty": qty,
                    "filled": True,
                    "detail": f"지정가 {filled_price:,}원 체결",
                    "order_id": order.order_id,
                }
            # 부분 체결 시 누적 체결 수량 추적
            if status.status.value == "partial" and status.filled_quantity > 0:
                cumulative_filled = status.filled_quantity

        # 3단계: 미체결 → 취소 → 시장가 (잔여 수량만)
        cumulative_filled = getattr(status, "filled_quantity", 0) or 0
        remaining = qty - cumulative_filled
        if remaining <= 0:
            logger.info("SmartSell: %s 대기 중 전량 체결됨 (%d주)", ticker, qty)
            return {
                "method": "limit_filled_during_wait",
                "price": limit_price,
                "qty": qty,
                "filled": True,
                "detail": f"대기 중 전량 체결 {qty}주",
                "order_id": order.order_id,
            }
        logger.info("SmartSell: %s %d분 미체결 → 취소 + 시장가 (잔여 %d주)", ticker, timeout_min, remaining)
        cancelled = self.order.cancel(order)
        if not cancelled:
            logger.warning("SmartSell: %s 취소 실패 → 시장가 전환 중단", ticker)
            return {
                "method": "limit_cancel_failed",
                "price": limit_price,
                "qty": remaining,
                "filled": False,
                "detail": "취소 실패 → 시장가 전환 중단 (이중 매도 방지)",
                "order_id": order.order_id,
            }
        time.sleep(1)
        return self._market_immediate(ticker, remaining, price, dry_run=False)

    def _patient_sell(
        self, ticker: str, qty: int, price: int, dry_run: bool
    ) -> dict:
        """X4: VWAP 이상 지정가 → 15분 → 현재가 정정 → 5분 → 시장가."""
        timeout = self.sell_cfg.get("x4_timeout_min", 15)
        logger.info(
            "SmartSell X4: %s %d주 VWAP 기준 대기 %d분 (dry_run=%s)",
            ticker, qty, timeout, dry_run,
        )

        if dry_run or not self.order:
            return {
                "method": "patient_sell",
                "price": price,
                "qty": qty,
                "filled": True,
                "detail": f"X4 시간제한 → VWAP 기준 {timeout}분 대기",
            }

        # VWAP 조회
        vwap_price = price
        if self.intraday:
            try:
                candles = self.intraday.fetch_full_day_1m_candles(ticker)
                if candles:
                    cum_tp_vol = 0.0
                    cum_vol = 0
                    for c in candles:
                        h, l, cl, v = c.get("high", 0), c.get("low", 0), c.get("close", 0), c.get("volume", 0)
                        if h > 0 and l > 0 and cl > 0 and v > 0:
                            cum_tp_vol += (h + l + cl) / 3.0 * v
                            cum_vol += v
                    if cum_vol > 0:
                        vwap_price = tick_round(int(cum_tp_vol / cum_vol), price)
            except Exception as e:
                logger.warning("SmartSell X4: VWAP 조회 실패 → 현재가 사용: %s", e)

        # VWAP 이상 지정가
        limit_price = max(price, vwap_price)
        order = self.order.sell_limit(ticker, limit_price, qty)
        if order.status.value == "failed":
            return self._market_immediate(ticker, qty, price, dry_run=False)

        # Phase 1: timeout분 대기
        for i in range(timeout * 2):  # 30초 간격
            time.sleep(30)
            status = self.order.get_order_status(order.order_id)
            if status.status.value == "filled":
                return {
                    "method": "patient_filled",
                    "price": int(status.filled_price or limit_price),
                    "qty": qty,
                    "filled": True,
                    "detail": f"X4 VWAP 지정가 체결 ({(i + 1) * 30}초)",
                    "order_id": order.order_id,
                }

        # Phase 2: 현재가로 정정 → 5분 대기
        logger.info("SmartSell X4: %s %d분 미체결 → 현재가 정정 + 5분", ticker, timeout)
        try:
            resp = self.order.fetch_current_price(ticker)
            new_price = int(resp.get("current_price", price)) if isinstance(resp, dict) else int(resp or price)
        except Exception:
            new_price = price
        self.order.modify(order, new_price, qty)

        for i in range(10):  # 5분 = 30초 × 10
            time.sleep(30)
            status = self.order.get_order_status(order.order_id)
            if status.status.value == "filled":
                return {
                    "method": "patient_modified_filled",
                    "price": int(status.filled_price or new_price),
                    "qty": qty,
                    "filled": True,
                    "detail": f"X4 정정 후 체결 ({new_price:,}원)",
                    "order_id": order.order_id,
                }

        # Phase 3: 시장가
        logger.info("SmartSell X4: %s 최종 시장가 전환", ticker)
        cancelled = self.order.cancel(order)
        if not cancelled:
            logger.warning("SmartSell X4: %s 취소 실패 → 시장가 전환 중단", ticker)
            return {
                "method": "patient_cancel_failed",
                "price": limit_price,
                "qty": qty,
                "filled": False,
                "detail": "X4 취소 실패 → 이중 매도 방지",
                "order_id": order.order_id,
            }
        time.sleep(1)
        return self._market_immediate(ticker, qty, price, dry_run=False)
