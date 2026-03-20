"""Smart Sell Executor — X1~X5 유형별 매도 전략 (EX-4)

X1 하드스톱: 즉시 시장가 (슬리피지 감수)
X2 수급이탈: 지정가 5분 → 시장가
X3 트레일링: 지정가 2분 → 시장가
X4 시간제한: VWAP 기준 15분 대기 → 정정 → 시장가
X5 목표달성: 지정가 5분 → 시장가 (프리미엄 최대)

toggle: execution_alpha.smart_sell.enabled: false → 기존 매도 로직 폴백
"""

from __future__ import annotations

import logging
import time

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

        Args:
            ticker: 종목코드
            qty: 매도 수량
            exit_rule: "X1" ~ "X5"
            current_price: 현재가
            dry_run: True면 로그만

        Returns:
            {"method": str, "price": int, "qty": int, "filled": bool, "detail": str}
        """
        # STEP 6에서 구현
        rule = exit_rule.upper()

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
        # STEP 6에서 구현
        logger.info(
            "SmartSell X1: %s %d주 시장가 매도 (dry_run=%s)", ticker, qty, dry_run
        )
        return {
            "method": "market_immediate",
            "price": price,
            "qty": qty,
            "filled": dry_run,  # dry_run=True면 시뮬레이션 체결
            "detail": "X1 하드스톱 → 즉시 시장가",
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
        # STEP 6에서 구현
        limit_price = int(price * (1 + premium_pct / 100))
        logger.info(
            "SmartSell: %s %d주 지정가 %d (프리미엄 +%.1f%%, %d분 대기, dry_run=%s)",
            ticker, qty, limit_price, premium_pct, timeout_min, dry_run,
        )
        return {
            "method": "limit_then_market",
            "price": limit_price,
            "qty": qty,
            "filled": dry_run,
            "detail": f"지정가 {limit_price:,}원 → {timeout_min}분 → 시장가",
        }

    def _patient_sell(
        self, ticker: str, qty: int, price: int, dry_run: bool
    ) -> dict:
        """X4: VWAP 이상 지정가 → 15분 → 현재가 정정 → 5분 → 시장가."""
        # STEP 6에서 구현
        timeout = self.sell_cfg.get("x4_timeout_min", 15)
        logger.info(
            "SmartSell X4: %s %d주 VWAP 기준 대기 %d분 (dry_run=%s)",
            ticker, qty, timeout, dry_run,
        )
        return {
            "method": "patient_sell",
            "price": price,
            "qty": qty,
            "filled": dry_run,
            "detail": f"X4 시간제한 → VWAP 기준 {timeout}분 대기",
        }
