"""
한국투자증권 주문/잔고 어댑터

mojito2 라이브러리를 래핑하여 OrderPort, BalancePort, CurrentPricePort를 구현한다.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time as dtime

import mojito

from src.entities.trading_models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.trading_calendar import is_kr_trading_day
from src.use_cases.ports import BalancePort, CurrentPricePort, OrderPort
from src.utils.auto_trading_volume import check_daily_limits, record_buy

logger = logging.getLogger(__name__)


class KisOrderAdapter(OrderPort, BalancePort, CurrentPricePort):
    """한국투자증권 API 주문/잔고/현재가 어댑터

    P0 가드레일 (5/14 추가, Phase 8 백테스트 기반):
    - AUTO_TRADING_ENABLED=true 필수 (기본 false)
    - AUTO_TRADING_MAX_QTY 종목당 최대 수량 (기본 1)
    - AUTO_TRADING_WHITELIST 매매 화이트리스트 (theme ETF 20개)
    - 거래시간 09:00~15:30 외 차단
    - 휴장일 차단
    """

    # theme ETF 20개 (Phase 8 백테스트: dual_buy 적중률 88.9%, D+3 +1.52%)
    THEME_ETF_WHITELIST = {
        "487240", "487230",  # KODEX AI전력
        "395160",            # KODEX AI반도체TOP2플러스
        "466920",            # SOL 조선TOP3플러스
        "367760", "367770",  # RISE 네트워크인프라/수소경제
        "228810", "228800",  # TIGER 미디어컨텐츠/여행레저
        "401170", "401470",  # RISE/KODEX 메타버스
        "337160",            # KODEX 200ESG
        "464310", "394660", "394670",  # TIGER 글로벌AI&로보틱스/자율주행/리튬
        "411420",            # KODEX 미국나스닥AI테크액티브
        "489030", "210780", "211560", "211900", "237370",  # 배당 ETF 5개
    }

    # 인버스/레버리지 ETF (5/16 추가, 약세장/강세장 방향성 베팅 도구)
    # 5/12~15 실측: KODEX 200선물인버스2X 4일 +7.3% (외인 5일 -25조 매도 패턴)
    # 위험: 일일 변동성 高, 음의 복리, 단기용 (1~5일 보유 권장)
    INVERSE_LEVERAGE_WHITELIST = {
        # 인버스 (약세장 베팅)
        "114800",  # KODEX 인버스 (1배)
        "252670",  # KODEX 200선물인버스2X (2배 레버리지) ⭐
        "251340",  # KODEX 코스닥150선물인버스
        # 레버리지 (강세장 베팅)
        "122630",  # KODEX 레버리지 (2배)
        "233160",  # TIGER 코스닥150 레버리지 (2배)
        "243880",  # TIGER 200IT레버리지 (2배)
    }

    # 자동매매 통합 화이트리스트 (theme + 인버스/레버리지)
    AUTO_TRADING_WHITELIST = THEME_ETF_WHITELIST | INVERSE_LEVERAGE_WHITELIST

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
            "KisOrderAdapter 초기화 (모드: %s, 가드레일: %s)",
            "모의투자" if is_mock else "실전",
            "ON" if os.getenv("AUTO_TRADING_ENABLED", "0") == "1" else "OFF (자동매매 비활성)",
        )

    # ──────────────────────────────────────────
    # P0 가드레일 (모든 주문 진입 시 호출)
    # ──────────────────────────────────────────
    def _guard(self, ticker: str, quantity: int, price: int | None = None, side: str = "BUY"):
        """주문 차단 가드레일. 실패 시 PermissionError/ValueError/RuntimeError 발생.

        Args:
            ticker: 종목 코드
            quantity: 수량
            price: 주문 단가 (지정가만, 시장가는 None). 현재가 ±X% 검증에 사용.
            side: BUY / SELL

        Checks (9개):
            1. AUTO_TRADING_ENABLED=1 활성화
            2. AUTO_TRADING_MAX_QTY 수량 한도
            3. AUTO_TRADING_WHITELIST_ONLY 화이트리스트
            4. 거래시간 09:00~15:30
            5. 거래일 (주말 + 공휴일, is_kr_trading_day)
            6. 일일 매수 금액/횟수 한도 (BUY만)
            7. 현재가 ±X% 범위 (지정가, price 전달 시)
            8. 통과 로그
        """
        # 1. 자동매매 활성화 체크
        if os.getenv("AUTO_TRADING_ENABLED", "0") != "1":
            raise PermissionError(
                "[GUARD] AUTO_TRADING_ENABLED=1 필수 (기본 비활성, 백테스트 60%+ 검증 후 활성)"
            )
        # 2. 종목당 최대 수량
        max_qty = int(os.getenv("AUTO_TRADING_MAX_QTY", "1"))
        if quantity > max_qty:
            raise ValueError(f"[GUARD] 수량 한도 초과: {quantity} > {max_qty}")
        # 3. 화이트리스트 (선택적, AUTO_TRADING_WHITELIST_ONLY=1일 때만)
        if os.getenv("AUTO_TRADING_WHITELIST_ONLY", "0") == "1":
            wl_env = os.getenv("AUTO_TRADING_WHITELIST", "")
            wl = set(wl_env.split(",")) if wl_env else self.AUTO_TRADING_WHITELIST
            if ticker not in wl:
                raise PermissionError(
                    f"[GUARD] 화이트리스트 외 종목: {ticker} (허용: {len(wl)}개)"
                )
        # 4. 거래시간 (09:00~15:30 KST)
        now = datetime.now().time()
        if not (dtime(9, 0) <= now <= dtime(15, 30)):
            raise RuntimeError(f"[GUARD] 거래시간 외: {now}")
        # 5. 거래일 (주말 + KRX 공휴일 통합)
        if not is_kr_trading_day():
            raise RuntimeError(f"[GUARD] KRX 휴장일: {date.today()}")
        # 6. 일일 매수 금액/횟수 한도 (BUY만 적용 — SELL은 한도 무관)
        if side == "BUY":
            max_amount = int(os.getenv("AUTO_TRADING_MAX_AMOUNT", "300000"))
            max_trades = int(os.getenv("AUTO_TRADING_MAX_TRADES_PER_DAY", "5"))
            # 지정가는 price, 시장가는 현재가 추정 (없으면 0으로 사전 검증만)
            est_price = price if price is not None else self._estimate_price(ticker)
            est_amount = quantity * est_price
            ok, reason = check_daily_limits(est_amount, max_amount, max_trades)
            if not ok:
                raise ValueError(f"[GUARD] {reason}")
        # 7. 현재가 ±X% 범위 검증 (지정가만)
        if price is not None:
            range_pct = float(os.getenv("AUTO_TRADING_PRICE_RANGE_PCT", "5"))
            current = self._estimate_price(ticker)
            if current > 0:
                diff_pct = abs(price - current) / current * 100
                if diff_pct > range_pct:
                    raise ValueError(
                        f"[GUARD] 지정가 현재가 ±{range_pct}% 초과: "
                        f"지정 {price:,} vs 현재 {current:,} (편차 {diff_pct:.1f}%)"
                    )
        # 8. 통과 로그
        logger.warning(
            "[GUARD PASS] %s %s %d주 @ %s (max_qty=%d, mode=%s)",
            side, ticker, quantity,
            f"{price:,}" if price else "시장가",
            max_qty, "MOCK" if self._is_mock else "REAL"
        )

    def _estimate_price(self, ticker: str) -> int:
        """현재가 조회 (오류 시 0). 가드레일 검증 전용 (실제 주문에는 영향 없음)."""
        try:
            return int(self.fetch_current_price(ticker).get("current_price", 0))
        except Exception:
            return 0

    def _send_telegram_alert(self, action: str, ticker: str, quantity: int,
                              price: int, order_type: str = "지정가") -> None:
        """매수/매도 시 텔레그램 알림. TELEGRAM_ALERT=0이면 스킵."""
        if os.getenv("AUTO_TRADING_TELEGRAM_ALERT", "1") != "1":
            return
        try:
            from src.telegram_sender import send_message
            from src.utils.auto_trading_volume import get_today_volume
            volume = get_today_volume()
            max_amount = int(os.getenv("AUTO_TRADING_MAX_AMOUNT", "300000"))
            max_trades = int(os.getenv("AUTO_TRADING_MAX_TRADES_PER_DAY", "5"))
            amount = quantity * price if price else 0
            msg = (
                f"[자동매매] {action}\n"
                f"종목: {ticker}\n"
                f"수량: {quantity}주\n"
                f"가격: {price:,}원 ({order_type})\n"
                f"금액: {amount:,}원\n"
                f"일일 누적: {volume['total_amount']:,}원 / {max_amount:,}원 "
                f"({volume['total_amount']/max_amount*100:.1f}%)\n"
                f"일일 횟수: {volume['total_trades']}회 / {max_trades}회\n"
                f"시각: {datetime.now().strftime('%H:%M:%S')}"
            )
            send_message(msg)
        except Exception as e:
            logger.warning("[자동매매 알림] 텔레그램 발송 실패: %s", e)

    # ──────────────────────────────────────────
    # OrderPort 구현
    # ──────────────────────────────────────────

    def buy_limit(self, ticker: str, price: int, quantity: int) -> Order:
        """지정가 매수"""
        self._guard(ticker, quantity, price=price, side="BUY")
        logger.info("[주문] 지정가 매수: %s %d주 @ %d원", ticker, quantity, price)
        try:
            resp = self.broker.create_limit_buy_order(ticker, price, quantity)
            order = self._parse_order_response(
                resp, ticker, OrderSide.BUY, OrderType.LIMIT, price, quantity
            )
            if order.status == OrderStatus.PENDING:
                record_buy(ticker, quantity, price)
                self._send_telegram_alert("매수 접수", ticker, quantity, price, "지정가")
            return order
        except Exception as e:
            logger.error("[주문] 지정가 매수 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.BUY, order_type=OrderType.LIMIT,
                price=price, quantity=quantity, status=OrderStatus.FAILED,
                message=str(e),
            )

    def sell_limit(self, ticker: str, price: int, quantity: int) -> Order:
        """지정가 매도"""
        self._guard(ticker, quantity, price=price, side="SELL")
        logger.info("[주문] 지정가 매도: %s %d주 @ %d원", ticker, quantity, price)
        try:
            resp = self.broker.create_limit_sell_order(ticker, price, quantity)
            order = self._parse_order_response(
                resp, ticker, OrderSide.SELL, OrderType.LIMIT, price, quantity
            )
            if order.status == OrderStatus.PENDING:
                self._send_telegram_alert("매도 접수", ticker, quantity, price, "지정가")
            return order
        except Exception as e:
            logger.error("[주문] 지정가 매도 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.SELL, order_type=OrderType.LIMIT,
                price=price, quantity=quantity, status=OrderStatus.FAILED,
                message=str(e),
            )

    def buy_market(self, ticker: str, quantity: int) -> Order:
        """시장가 매수"""
        self._guard(ticker, quantity, price=None, side="BUY")
        logger.info("[주문] 시장가 매수: %s %d주", ticker, quantity)
        try:
            resp = self.broker.create_market_buy_order(ticker, quantity)
            order = self._parse_order_response(
                resp, ticker, OrderSide.BUY, OrderType.MARKET, 0, quantity
            )
            if order.status == OrderStatus.PENDING:
                est_price = self._estimate_price(ticker)
                record_buy(ticker, quantity, est_price)
                self._send_telegram_alert("매수 접수", ticker, quantity, est_price, "시장가")
            return order
        except Exception as e:
            logger.error("[주문] 시장가 매수 실패: %s — %s", ticker, e)
            return Order(
                ticker=ticker, side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=quantity, status=OrderStatus.FAILED, message=str(e),
            )

    def sell_market(self, ticker: str, quantity: int) -> Order:
        """시장가 매도"""
        self._guard(ticker, quantity, price=None, side="SELL")
        logger.info("[주문] 시장가 매도: %s %d주", ticker, quantity)
        try:
            resp = self.broker.create_market_sell_order(ticker, quantity)
            order = self._parse_order_response(
                resp, ticker, OrderSide.SELL, OrderType.MARKET, 0, quantity
            )
            if order.status == OrderStatus.PENDING:
                est_price = self._estimate_price(ticker)
                self._send_telegram_alert("매도 접수", ticker, quantity, est_price, "시장가")
            return order
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
            # 미체결 목록에 없음 → 체결 완료 또는 취소/거부됨
            # 당일 체결 내역 조회로 교차 확인
            try:
                filled_order = self._check_filled_today(order_id)
                if filled_order:
                    logger.info("[주문] %s 당일 체결 확인됨", order_id)
                    return filled_order
            except Exception as e:
                logger.warning("[주문] %s 체결 확인 조회 실패: %s", order_id, e)

            # 체결 내역에도 없음 → 취소/거부로 간주 (안전 측)
            logger.warning("[주문] %s 미체결+체결 모두 부재 → CANCELLED 추정 (취소/거부)", order_id)
            return Order(order_id=order_id, status=OrderStatus.CANCELLED)
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

            cash = int(
                summary_item.get("dnca_tot_amt", 0)
                or summary_item.get("prvs_rcdl_excc_amt", 0)
                or summary_item.get("nass_amt", 0)
            )
            if cash <= 0 and summary_item:
                logger.warning(
                    "[잔고] available_cash=0 — 필드 확인 필요: keys=%s",
                    list(summary_item.keys())[:10],
                )
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
                "available_cash": cash,
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

    def _check_filled_today(self, order_id: str) -> Order | None:
        """당일 체결 내역에서 order_id 확인. 체결되었으면 Order, 아니면 None."""
        try:
            from datetime import date
            today = date.today().strftime("%Y%m%d")
            resp = self.broker.fetch_today_execution({
                "INQR_STRT_DT": today,
                "INQR_END_DT": today,
                "SLL_BUY_DVSN_CD": "00",  # 전체
                "INQR_DVSN": "00",
                "PDNO": "",
                "CCLD_DVSN": "01",  # 체결분만
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "INQR_DVSN_3": "00",
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            })
            for item in resp.get("output1", []):
                if item.get("odno") == order_id:
                    return Order(
                        order_id=order_id,
                        ticker=item.get("pdno", ""),
                        side=OrderSide.BUY if item.get("sll_buy_dvsn_cd") == "02" else OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        price=int(item.get("ord_unpr", 0)),
                        quantity=int(item.get("ord_qty", 0)),
                        filled_quantity=int(item.get("tot_ccld_qty", 0)),
                        filled_price=float(item.get("avg_prvs", 0) or item.get("ccld_pric", 0)),
                        status=OrderStatus.FILLED,
                    )
        except AttributeError:
            # mojito에 fetch_today_execution이 없으면 잔고 기반 fallback
            logger.debug("[주문] fetch_today_execution 미지원 → 잔고 기반 확인")
        except Exception as e:
            logger.warning("[주문] 당일 체결 조회 실패: %s", e)
        return None

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
