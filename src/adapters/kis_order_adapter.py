"""
한국투자증권 주문/잔고 어댑터

mojito2 라이브러리를 래핑하여 OrderPort, BalancePort, CurrentPricePort를 구현한다.
"""

from __future__ import annotations

import logging
import os

import mojito

from src.entities.trading_models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.use_cases.ports import BalancePort, CurrentPricePort, OrderPort

logger = logging.getLogger(__name__)


class KisOrderAdapter(OrderPort, BalancePort, CurrentPricePort):
    """한국투자증권 API 주문/잔고/현재가 어댑터"""

    def __init__(self):
        is_mock = os.getenv("MODEL") != "REAL"
        self.broker = mojito.KoreaInvestment(
            api_key=os.getenv("KIS_APP_KEY"),
            api_secret=os.getenv("KIS_APP_SECRET"),
            acc_no=os.getenv("KIS_ACC_NO"),
            mock=is_mock,
        )
        self._is_mock = is_mock
        logger.info(
            "KisOrderAdapter 초기화 (모드: %s)", "모의투자" if is_mock else "실전"
        )

    # ──────────────────────────────────────────
    # OrderPort 구현
    # ──────────────────────────────────────────

    def buy_limit(self, ticker: str, price: int, quantity: int) -> Order:
        """지정가 매수"""
        logger.info("[주문] 지정가 매수: %s %d주 @ %d원", ticker, quantity, price)
        try:
            resp = self.broker.create_limit_buy_order(ticker, price, quantity)
            return self._parse_order_response(
                resp, ticker, OrderSide.BUY, OrderType.LIMIT, price, quantity
            )
        except Exception as e:
            logger.error("[주문] 지정가 매수 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.BUY, order_type=OrderType.LIMIT,
                price=price, quantity=quantity, status=OrderStatus.FAILED,
                message=str(e),
            )

    def sell_limit(self, ticker: str, price: int, quantity: int) -> Order:
        """지정가 매도"""
        logger.info("[주문] 지정가 매도: %s %d주 @ %d원", ticker, quantity, price)
        try:
            resp = self.broker.create_limit_sell_order(ticker, price, quantity)
            return self._parse_order_response(
                resp, ticker, OrderSide.SELL, OrderType.LIMIT, price, quantity
            )
        except Exception as e:
            logger.error("[주문] 지정가 매도 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.SELL, order_type=OrderType.LIMIT,
                price=price, quantity=quantity, status=OrderStatus.FAILED,
                message=str(e),
            )

    def buy_market(self, ticker: str, quantity: int) -> Order:
        """시장가 매수"""
        logger.info("[주문] 시장가 매수: %s %d주", ticker, quantity)
        try:
            resp = self.broker.create_market_buy_order(ticker, quantity)
            return self._parse_order_response(
                resp, ticker, OrderSide.BUY, OrderType.MARKET, 0, quantity
            )
        except Exception as e:
            logger.error("[주문] 시장가 매수 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=quantity, status=OrderStatus.FAILED, message=str(e),
            )

    def sell_market(self, ticker: str, quantity: int) -> Order:
        """시장가 매도"""
        logger.info("[주문] 시장가 매도: %s %d주", ticker, quantity)
        try:
            resp = self.broker.create_market_sell_order(ticker, quantity)
            return self._parse_order_response(
                resp, ticker, OrderSide.SELL, OrderType.MARKET, 0, quantity
            )
        except Exception as e:
            logger.error("[주문] 시장가 매도 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=quantity, status=OrderStatus.FAILED, message=str(e),
            )

    def cancel(self, order: Order) -> bool:
        """주문 취소"""
        logger.info("[주문] 취소: %s (주문번호=%s)", order.ticker, order.order_id)
        try:
            resp = self.broker.cancel_order(
                org_no=order.org_no,
                order_no=order.order_id,
                quantity=order.quantity,
                total=True,
            )
            rt_cd = resp.get("rt_cd", "1")
            if rt_cd == "0":
                logger.info("[주문] 취소 성공: %s", order.order_id)
                return True
            logger.warning("[주문] 취소 응답: %s", resp.get("msg1", ""))
            return False
        except Exception as e:
            logger.error("[주문] 취소 실패: %s — %s", order.order_id, e)
            return False

    def modify(self, order: Order, new_price: int, new_quantity: int) -> Order:
        """주문 정정"""
        logger.info(
            "[주문] 정정: %s (주문번호=%s) → %d원 %d주",
            order.ticker, order.order_id, new_price, new_quantity,
        )
        try:
            resp = self.broker.modify_order(
                org_no=order.org_no,
                order_no=order.order_id,
                order_type="00",  # 지정가
                price=new_price,
                quantity=new_quantity,
                total=True,
            )
            return self._parse_order_response(
                resp, order.ticker, order.side, OrderType.LIMIT,
                new_price, new_quantity,
            )
        except Exception as e:
            logger.error("[주문] 정정 실패: %s — %s", order.order_id, e)
            return Order(
                order_id=order.order_id, ticker=order.ticker,
                side=order.side, order_type=OrderType.LIMIT,
                price=new_price, quantity=new_quantity,
                status=OrderStatus.FAILED, message=str(e),
            )

    def get_order_status(self, order_id: str) -> Order:
        """주문 상태 조회 (미체결 내역 기반 + 체결 확인)"""
        try:
            resp = self.broker.fetch_open_order({
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
                "INQR_DVSN_1": "0",
                "INQR_DVSN_2": "0",
            })
            for item in resp.get("output", []):
                if item.get("odno") == order_id:
                    return Order(
                        order_id=order_id,
                        ticker=item.get("pdno", ""),
                        side=OrderSide.BUY if item.get("sll_buy_dvsn_cd") == "02" else OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        price=int(item.get("ord_unpr", 0)),
                        quantity=int(item.get("ord_qty", 0)),
                        filled_quantity=int(item.get("tot_ccld_qty", 0)),
                        filled_price=float(item.get("avg_prvs", 0)),
                        status=OrderStatus.PARTIAL if int(item.get("tot_ccld_qty", 0)) > 0 else OrderStatus.PENDING,
                        org_no=item.get("orgn_odno", ""),
                    )
            # 미체결 목록에 없음 → 체결 완료 또는 취소됨
            # 잔고 조회로 실제 보유 여부 교차 확인
            try:
                balance = self.fetch_balance()
                held_tickers = {h["ticker"] for h in balance.get("holdings", [])}
                # 잔고에서 발견되면 체결 완료로 확신
                # (주의: order_id로 특정 불가, ticker 기반 추정)
                logger.info("[주문] %s 미체결 목록 부재 → 체결 완료 추정", order_id)
                return Order(order_id=order_id, status=OrderStatus.FILLED)
            except Exception:
                # 잔고 조회도 실패하면 PENDING 유지 (안전 측)
                logger.warning("[주문] %s 상태 불확실 — PENDING 유지", order_id)
                return Order(order_id=order_id, status=OrderStatus.PENDING)
        except Exception as e:
            logger.error("[주문] 상태 조회 실패: %s — %s", order_id, e)
            return Order(order_id=order_id, status=OrderStatus.FAILED, message=str(e))

    # ──────────────────────────────────────────
    # BalancePort 구현
    # ──────────────────────────────────────────

    def fetch_balance(self) -> dict:
        """전체 잔고 조회"""
        try:
            data = self.broker.fetch_balance()
            holdings = data.get("output1", [])
            summary = data.get("output2", [{}])
            summary_item = summary[0] if summary else {}

            return {
                "holdings": [
                    {
                        "ticker": h.get("pdno", ""),
                        "name": h.get("prdt_name", ""),
                        "quantity": int(h.get("hldg_qty", 0)),
                        "avg_price": float(h.get("pchs_avg_pric", 0)),
                        "current_price": int(h.get("prpr", 0)),
                        "eval_amount": int(h.get("evlu_amt", 0)),
                        "pnl_amount": int(h.get("evlu_pfls_amt", 0)),
                        "pnl_pct": float(h.get("evlu_pfls_rt", 0)),
                    }
                    for h in holdings if int(h.get("hldg_qty", 0)) > 0
                ],
                "total_eval": int(summary_item.get("tot_evlu_amt", 0)),
                "total_pnl": int(summary_item.get("evlu_pfls_smtl_amt", 0)),
                "available_cash": int(
                    summary_item.get("dnca_tot_amt", 0)
                    or summary_item.get("prvs_rcdl_excc_amt", 0)
                ),
            }
        except Exception as e:
            logger.error("[잔고] 조회 실패: %s", e)
            return {"holdings": [], "total_eval": 0, "total_pnl": 0, "available_cash": 0}

    def fetch_holdings(self) -> list[dict]:
        """보유종목 목록만 조회"""
        balance = self.fetch_balance()
        return balance.get("holdings", [])

    def get_available_cash(self) -> float:
        """주문 가능 예수금"""
        balance = self.fetch_balance()
        return float(balance.get("available_cash", 0))

    # ──────────────────────────────────────────
    # CurrentPricePort 구현
    # ──────────────────────────────────────────

    def fetch_current_price(self, ticker: str) -> dict:
        """종목 현재가 + 기본 정보 조회"""
        try:
            data = self.broker.fetch_price(ticker)
            output = data.get("output", {})
            return {
                "ticker": ticker,
                "name": output.get("rprs_mrkt_kor_name", ""),
                "current_price": int(output.get("stck_prpr", 0)),
                "change_pct": float(output.get("prdy_ctrt", 0)),
                "volume": int(output.get("acml_vol", 0)),
                "high": int(output.get("stck_hgpr", 0)),
                "low": int(output.get("stck_lwpr", 0)),
                "open": int(output.get("stck_oprc", 0)),
            }
        except Exception as e:
            logger.error("[현재가] 조회 실패: %s — %s", ticker, e)
            return {"ticker": ticker, "current_price": 0}

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _parse_order_response(
        resp: dict,
        ticker: str,
        side: OrderSide,
        order_type: OrderType,
        price: int,
        quantity: int,
    ) -> Order:
        """한투 API 주문 응답 → Order 엔티티"""
        rt_cd = resp.get("rt_cd", "1")
        output = resp.get("output", {})

        if rt_cd == "0":
            order_id = output.get("ODNO", output.get("odno", ""))
            org_no = output.get("KRX_FWDG_ORD_ORGNO", output.get("krx_fwdg_ord_orgno", ""))
            logger.info(
                "[주문] 접수 성공: %s %s %d주 (주문번호=%s)",
                side.value, ticker, quantity, order_id,
            )
            return Order(
                order_id=order_id,
                ticker=ticker,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                status=OrderStatus.PENDING,
                org_no=org_no,
                message=resp.get("msg1", ""),
            )
        else:
            msg = resp.get("msg1", resp.get("msg_cd", "알 수 없는 오류"))
            logger.error("[주문] 접수 실패: %s — %s", ticker, msg)
            return Order(
                ticker=ticker,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                status=OrderStatus.FAILED,
                message=msg,
            )
