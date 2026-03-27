"""pykrx 기반 수급 데이터 수집 어댑터 — Phase 1

수집 데이터:
  Layer 1: 공매도 잔고/거래비중/대차잔고
  Layer 4: 프로그램매매 동향 (차익/비차익)
  Layer 5: 투자자별 매매동향 (외국인/기관/연기금/개인)

의존성: pip install pykrx
갱신 주기: 매일 장마감 후 1회 (daily_scheduler Phase 8 이후)
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.entities.supply_demand_models import (
    InstitutionalDetailData,
    InvestorFlowData,
    ProgramTradingData,
    ShortSellingData,
)

# PyKRX 컬럼명 → 모델 필드 매핑
_INST_COL_MAP = {
    "금융투자": "securities_net",
    "보험": "insurance_net",
    "투신": "asset_mgmt_net",
    "사모": "private_equity_net",
    "은행": "bank_net",
    "기타금융": "other_financial_net",
    "연기금등": "pension_net",
}

# KIS API fallback (pykrx 수급 장애 대비)
try:
    from src.adapters.kis_investor_adapter import fetch_investor_by_ticker as kis_fetch_investor
    KIS_INVESTOR_AVAILABLE = True
except ImportError:
    KIS_INVESTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class PykrxSupplyAdapter:
    """pykrx 라이브러리를 사용한 수급 데이터 수집"""

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self._pykrx = None
        self._blacklist_path = Path("data/supply_blacklist.json")
        self._max_fails = 3
        self._blacklist_reset_days = 30

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
        """투자자별 매매동향 (외국인/기관/연기금/개인)

        pykrx 우선 시도, 실패 시 KIS API fallback.
        """
        start, end = self._date_range(days=30)
        today = datetime.today().strftime("%Y%m%d")

        result = InvestorFlowData(ticker=ticker, date=today)
        inv_df = None

        # 1) pykrx 시도
        try:
            stock = self._ensure_pykrx()
            inv_df = stock.get_market_trading_value_by_date(start, end, ticker)
            if inv_df is None or inv_df.empty:
                inv_df = None
        except Exception as e:
            logger.debug(f"[{ticker}] pykrx 수급 실패: {e}")
            inv_df = None

        # 2) KIS API fallback
        if inv_df is None and KIS_INVESTOR_AVAILABLE:
            try:
                inv_df = kis_fetch_investor(ticker)
                if inv_df is not None and not inv_df.empty:
                    logger.debug(f"[{ticker}] KIS fallback 수급 조회 성공")
                else:
                    inv_df = None
            except Exception as e:
                logger.debug(f"[{ticker}] KIS fallback 실패: {e}")
                inv_df = None

        if inv_df is None or inv_df.empty:
            return result

        try:
            latest = inv_df.iloc[-1]

            # 투자자별 순매수
            result.foreign_net = int(latest.get("외국인합계", 0))
            result.institution_net = int(latest.get("기관합계", 0))
            result.individual_net = int(latest.get("개인", 0))

            # 연기금 (pykrx에서만 제공)
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
            logger.warning(f"[{ticker}] 투자자별 매매동향 파싱 실패: {e}")

        return result

    # ─────────────────────────────────────────
    # Layer 5-B: 기관 유형별 세분화 (TIER2)
    # ─────────────────────────────────────────
    def fetch_institutional_detail(self, ticker: str) -> InstitutionalDetailData:
        """기관 유형별(금투/보험/투신/사모/은행/기타금융/연기금) 세분화 수급.

        PyKRX get_market_trading_value_by_date()는 기본적으로
        금융투자, 보험, 투신, 사모, 은행, 기타금융, 연기금등 컬럼을 반환한다.
        """
        start, end = self._date_range(days=30)
        today = datetime.today().strftime("%Y%m%d")

        result = InstitutionalDetailData(ticker=ticker, date=today)

        try:
            stock = self._ensure_pykrx()
            inv_df = stock.get_market_trading_value_by_date(start, end, ticker)
            if inv_df is None or inv_df.empty:
                return result
        except Exception as e:
            logger.debug(f"[{ticker}] 기관 세분화 수급 실패: {e}")
            return result

        try:
            latest = inv_df.iloc[-1]

            # 당일 기관 유형별 순매수
            for col, attr in _INST_COL_MAP.items():
                if col in inv_df.columns:
                    setattr(result, attr, int(latest.get(col, 0)))

            # 연기금 연속 순매수 일수
            if "연기금등" in inv_df.columns:
                pension_series = inv_df["연기금등"]
                consecutive = 0
                for val in reversed(pension_series.values):
                    if val > 0:
                        consecutive += 1
                    else:
                        break
                result.pension_consecutive_days = consecutive

                # 연기금 5/20일 누적
                if len(inv_df) >= 5:
                    result.pension_cumulative_5d = int(pension_series.tail(5).sum())
                if len(inv_df) >= 20:
                    result.pension_cumulative_20d = int(pension_series.tail(20).sum())

            # 보험 5일 누적
            if "보험" in inv_df.columns and len(inv_df) >= 5:
                result.insurance_cumulative_5d = int(
                    inv_df["보험"].tail(5).sum()
                )

            # 투신 5일 누적
            if "투신" in inv_df.columns and len(inv_df) >= 5:
                result.asset_mgmt_cumulative_5d = int(
                    inv_df["투신"].tail(5).sum()
                )

            # 스마트머니 비율: (연기금+보험) / |기관합계|
            if "기관합계" in inv_df.columns:
                inst_total = abs(int(latest.get("기관합계", 0)))
                if inst_total > 0:
                    smart = abs(result.pension_net) + abs(result.insurance_net)
                    result.smart_money_ratio = round(min(smart / inst_total, 1.0), 3)

        except Exception as e:
            logger.warning(f"[{ticker}] 기관 세분화 파싱 실패: {e}")

        return result

    # ─────────────────────────────────────────
    # 블랙리스트 관리
    # ─────────────────────────────────────────
    def _load_blacklist(self) -> dict:
        """반복 실패 종목 블랙리스트 로드"""
        if self._blacklist_path.exists():
            try:
                with open(self._blacklist_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"tickers": {}, "updated_at": ""}

    def _save_blacklist(self, bl: dict):
        """블랙리스트 저장"""
        self._blacklist_path.parent.mkdir(parents=True, exist_ok=True)
        bl["updated_at"] = datetime.today().strftime("%Y-%m-%d %H:%M")
        with open(self._blacklist_path, "w", encoding="utf-8") as f:
            json.dump(bl, f, ensure_ascii=False, indent=2)

    def _is_blacklisted(self, ticker: str, bl: dict) -> bool:
        """블랙리스트 여부 (30일 경과 시 자동 해제)"""
        info = bl.get("tickers", {}).get(ticker)
        if not info or info.get("fail_count", 0) < self._max_fails:
            return False
        last_fail = info.get("last_fail", "")
        if last_fail:
            try:
                days = (datetime.today() - datetime.strptime(last_fail, "%Y-%m-%d")).days
                if days >= self._blacklist_reset_days:
                    return False
            except ValueError:
                pass
        return True

    def _record_failure(self, ticker: str, bl: dict, reason: str):
        """실패 기록 (연속 실패 카운트 증가)"""
        tickers = bl.setdefault("tickers", {})
        info = tickers.setdefault(ticker, {"fail_count": 0})
        info["fail_count"] = info.get("fail_count", 0) + 1
        info["last_fail"] = datetime.today().strftime("%Y-%m-%d")
        info["reason"] = reason[:100]

    def _record_success(self, ticker: str, bl: dict):
        """성공 시 블랙리스트에서 제거"""
        if ticker in bl.get("tickers", {}):
            del bl["tickers"][ticker]

    # ─────────────────────────────────────────
    # 일괄 수집 (병렬 + 블랙리스트)
    # ─────────────────────────────────────────
    def _collect_single(self, ticker: str) -> tuple:
        """단일 종목 수급 수집 (스레드용)"""
        try:
            short = self.fetch_short_selling(ticker)
            flow = self.fetch_investor_flow(ticker)

            # 빈 결과 감지: fetch 내부에서 에러가 삼켜져 default 값만 남은 경우
            # (Expecting value 등 pykrx JSON 파싱 실패)
            # ※ short.total_volume은 공매도 금지 기간에 정상 0이므로 제외
            #    수급(flow) 3종이 모두 0 = 데이터 자체가 없는 종목
            is_empty = (
                flow.foreign_net == 0
                and flow.institution_net == 0
                and flow.individual_net == 0
            )
            err = "empty_response" if is_empty else None
            return (ticker, {"short": short, "flow": flow}, err)
        except ImportError:
            raise
        except Exception as e:
            return (ticker, {"short": None, "flow": None}, str(e))

    def collect_all(self, tickers: list[str], max_workers: int = 5) -> dict:
        """전 종목 수급 데이터 병렬 수집 (블랙리스트 적용)

        Returns:
            {ticker: {"short": ShortSellingData, "flow": InvestorFlowData}}
        """
        # pykrx 사전 확인
        try:
            self._ensure_pykrx()
        except ImportError:
            logger.error("pykrx 미설치, 수급 수집 중단")
            return {}

        bl = self._load_blacklist()

        # 블랙리스트 필터링
        active = []
        skipped_count = 0
        for t in tickers:
            if self._is_blacklisted(t, bl):
                skipped_count += 1
            else:
                active.append(t)
        if skipped_count:
            logger.info(f"[블랙리스트] {skipped_count}종목 스킵 (연속 {self._max_fails}회+ 실패)")

        total = len(active)
        logger.info(f"[수급 수집] {total}종목 병렬 수집 시작 (workers={max_workers})")
        t0 = time.time()

        results = {}
        err_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._collect_single, t): t for t in active}
            for i, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    t, data, err = future.result()
                    if err:
                        err_count += 1
                        self._record_failure(t, bl, err)
                    else:
                        self._record_success(t, bl)
                    results[t] = data  # 빈 결과도 보존 (downstream 기본값 처리)
                except Exception as e:
                    err_count += 1
                    self._record_failure(ticker, bl, str(e))
                    results[ticker] = {"short": None, "flow": None}

                if i % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"  수급 진행: {i}/{total} ({elapsed:.0f}초) | 오류: {err_count}"
                    )

        elapsed = time.time() - t0
        logger.info(
            f"[수급 수집] 완료: {total}종목, {elapsed:.0f}초 ({elapsed/60:.1f}분) | "
            f"오류: {err_count} | 블랙리스트: {len(bl.get('tickers', {}))}종목"
        )

        self._save_blacklist(bl)
        return results
