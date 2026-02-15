"""pykrx 기반 수급 데이터 수집 어댑터 — Phase 1

수집 데이터:
  Layer 1: 공매도 잔고/거래비중/대차잔고
  Layer 4: 프로그램매매 동향 (차익/비차익)
  Layer 5: 투자자별 매매동향 (외국인/기관/연기금/개인)

의존성: pip install pykrx
갱신 주기: 매일 장마감 후 1회 (daily_scheduler Phase 8 이후)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from src.entities.supply_demand_models import (
    InvestorFlowData,
    ProgramTradingData,
    ShortSellingData,
)

logger = logging.getLogger(__name__)


class PykrxSupplyAdapter:
    """pykrx 라이브러리를 사용한 수급 데이터 수집"""

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self._pykrx = None

    def _ensure_pykrx(self):
        """pykrx 지연 임포트 (설치 안 된 환경 대응)"""
        if self._pykrx is None:
            try:
                from pykrx import stock
                self._pykrx = stock
            except ImportError:
                logger.warning("pykrx 미설치: pip install pykrx")
                raise ImportError("pykrx 라이브러리가 필요합니다: pip install pykrx")
        return self._pykrx

    def _date_range(self, days: int | None = None) -> tuple[str, str]:
        """조회 기간 (YYYYMMDD 형식)"""
        end = datetime.today()
        start = end - timedelta(days=days or self.lookback_days)
        return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    # ─────────────────────────────────────────
    # Layer 1: 공매도
    # ─────────────────────────────────────────
    def fetch_short_selling(self, ticker: str) -> ShortSellingData:
        """공매도 데이터 수집 (잔고 + 거래비중 + 대차)"""
        stock = self._ensure_pykrx()
        start, end = self._date_range()
        today = datetime.today().strftime("%Y%m%d")

        result = ShortSellingData(ticker=ticker, date=today)

        try:
            # 공매도 거래 현황 (일별)
            short_df = stock.get_shorting_status_by_date(start, end, ticker)
            if short_df is not None and not short_df.empty:
                latest = short_df.iloc[-1]
                result.short_volume = int(latest.get("공매도거래량", 0))
                result.total_volume = int(latest.get("총거래량", 0))
                result.short_ratio = float(latest.get("비중", 0))

                # 40일 평균 대비 스파이크
                if "비중" in short_df.columns and len(short_df) >= 40:
                    avg_40 = short_df["비중"].tail(40).mean()
                    result.avg_short_ratio_40d = float(avg_40)
                    result.short_spike_ratio = (
                        result.short_ratio / avg_40 if avg_40 > 0 else 1.0
                    )
        except Exception as e:
            logger.warning(f"[{ticker}] 공매도 거래현황 조회 실패: {e}")

        try:
            # 공매도 잔고
            balance_df = stock.get_shorting_balance_by_ticker(start, end, ticker)
            if balance_df is not None and not balance_df.empty:
                latest = balance_df.iloc[-1]
                result.short_balance = int(latest.get("공매도잔고", 0))
                result.short_balance_ratio = float(latest.get("비중", 0))

                # 대차잔고 5일 변화율
                if "대차잔고" in balance_df.columns and len(balance_df) >= 6:
                    result.lending_balance = int(balance_df["대차잔고"].iloc[-1])
                    lb_5d_ago = balance_df["대차잔고"].iloc[-6]
                    if lb_5d_ago > 0:
                        result.lending_change_5d = (
                            (result.lending_balance - lb_5d_ago) / lb_5d_ago * 100
                        )
        except Exception as e:
            logger.warning(f"[{ticker}] 공매도 잔고 조회 실패: {e}")

        return result

    # ─────────────────────────────────────────
    # Layer 4: 프로그램매매
    # ─────────────────────────────────────────
    def fetch_program_trading(self, date: str | None = None) -> ProgramTradingData:
        """프로그램매매 동향 (차익/비차익)"""
        stock = self._ensure_pykrx()
        target = date or datetime.today().strftime("%Y%m%d")

        result = ProgramTradingData(date=target)

        try:
            # pykrx 프로그램매매 데이터
            # get_market_trading_value_by_date로 프로그램 순매수 추출
            prog_df = stock.get_market_net_purchases_of_equities_by_ticker(
                target, target, "KOSPI"
            )
            if prog_df is not None and not prog_df.empty:
                # 시장 전체 프로그램매매 합계
                if "기관합계" in prog_df.columns:
                    result.non_arbitrage_buy = int(
                        prog_df["기관합계"].clip(lower=0).sum()
                    )
                    result.non_arbitrage_sell = int(
                        (-prog_df["기관합계"].clip(upper=0)).sum()
                    )
        except Exception as e:
            logger.warning(f"프로그램매매 동향 조회 실패: {e}")

        return result

    # ─────────────────────────────────────────
    # Layer 5: 기관/외인 수급
    # ─────────────────────────────────────────
    def fetch_investor_flow(self, ticker: str) -> InvestorFlowData:
        """투자자별 매매동향 (외국인/기관/연기금/개인)"""
        stock = self._ensure_pykrx()
        start, end = self._date_range(days=30)
        today = datetime.today().strftime("%Y%m%d")

        result = InvestorFlowData(ticker=ticker, date=today)

        try:
            inv_df = stock.get_market_trading_value_by_date(start, end, ticker)
            if inv_df is None or inv_df.empty:
                return result

            latest = inv_df.iloc[-1]

            # 투자자별 순매수
            result.foreign_net = int(latest.get("외국인합계", 0))
            result.institution_net = int(latest.get("기관합계", 0))
            result.individual_net = int(latest.get("개인", 0))

            # 연기금 (있는 경우)
            if "연기금등" in inv_df.columns:
                result.pension_net = int(latest.get("연기금등", 0))

            # 외국인 연속 순매수 일수
            if "외국인합계" in inv_df.columns:
                foreign_series = inv_df["외국인합계"]
                consecutive = 0
                for val in reversed(foreign_series.values):
                    if val > 0:
                        consecutive += 1
                    else:
                        break
                result.foreign_consecutive_days = consecutive

            # 기관 20일 누적 순매수
            if "기관합계" in inv_df.columns and len(inv_df) >= 20:
                result.institution_cumulative_20d = int(
                    inv_df["기관합계"].tail(20).sum()
                )

        except Exception as e:
            logger.warning(f"[{ticker}] 투자자별 매매동향 조회 실패: {e}")

        return result

    # ─────────────────────────────────────────
    # 일괄 수집
    # ─────────────────────────────────────────
    def collect_all(self, tickers: list[str]) -> dict:
        """전 종목 수급 데이터 일괄 수집

        Returns:
            {ticker: {"short": ShortSellingData, "flow": InvestorFlowData}}
        """
        results = {}
        for ticker in tickers:
            logger.info(f"[수급 수집] {ticker}")
            try:
                short = self.fetch_short_selling(ticker)
                flow = self.fetch_investor_flow(ticker)
                results[ticker] = {"short": short, "flow": flow}
            except ImportError:
                logger.error("pykrx 미설치, 수급 수집 중단")
                break
            except Exception as e:
                logger.warning(f"[{ticker}] 수급 수집 실패: {e}")
                results[ticker] = {"short": None, "flow": None}

        return results
