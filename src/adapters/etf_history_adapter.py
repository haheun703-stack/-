"""ETF History Adapter — ETF 월봉/주봉 OHLCV 수집 + parquet 캐시

Structure Score(S1 연간돌파, S2 주봉 StochRSI) 계산에 사용.
데이터 소스: yfinance (한국 ETF: {code}.KS 형식)
Fallback: FinanceDataReader
캐시: data/etf_history/{ticker}_{monthly|weekly}.parquet
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/etf_history")
MONTHLY_MAX_AGE_DAYS = 7
WEEKLY_MAX_AGE_DAYS = 1


class EtfHistoryAdapter:
    """한국 ETF/종목 월봉·주봉 수집 + parquet 캐시"""

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._yf = None
        self._fdr = None

    # ── lazy import ──────────────────────────

    def _ensure_yf(self):
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.warning("yfinance 미설치: pip install yfinance")
        return self._yf

    def _ensure_fdr(self):
        if self._fdr is None:
            try:
                import FinanceDataReader as fdr
                self._fdr = fdr
            except ImportError:
                logger.warning("FinanceDataReader 미설치")
        return self._fdr

    # ── 캐시 ─────────────────────────────────

    @staticmethod
    def _load_cache(path: Path, max_age_days: int) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > timedelta(days=max_age_days):
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            return df
        except Exception as e:
            logger.warning("캐시 로드 실패 %s: %s", path.name, e)
            return None

    @staticmethod
    def _save_cache(df: pd.DataFrame, path: Path):
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning("캐시 저장 실패 %s: %s", path.name, e)

    # ── 월봉 수집 ────────────────────────────

    def fetch_monthly(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """ETF/종목 월봉 OHLCV 수집 (캐시 우선).

        Returns: DataFrame[Date index, Open, High, Low, Close, Volume]
        """
        cache_path = CACHE_DIR / f"{ticker}_monthly.parquet"
        cached = self._load_cache(cache_path, MONTHLY_MAX_AGE_DAYS)
        if cached is not None:
            logger.debug("월봉 캐시 히트: %s (%d행)", ticker, len(cached))
            return cached

        end = datetime.now()
        start = end - timedelta(days=years * 365 + 30)

        df = self._fetch_daily_raw(ticker, start, end)
        if df.empty:
            return pd.DataFrame()

        # 일봉 → 월봉 리샘플
        monthly = df.resample("ME").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])

        if not monthly.empty:
            self._save_cache(monthly, cache_path)
            logger.info("월봉 수집 완료: %s (%d개월)", ticker, len(monthly))

        return monthly

    # ── 주봉 수집 ────────────────────────────

    def fetch_weekly(self, ticker: str, weeks: int = 52) -> pd.DataFrame:
        """ETF/종목 주봉 OHLCV 수집 (캐시 우선).

        Returns: DataFrame[Date index, Open, High, Low, Close, Volume]
        """
        cache_path = CACHE_DIR / f"{ticker}_weekly.parquet"
        cached = self._load_cache(cache_path, WEEKLY_MAX_AGE_DAYS)
        if cached is not None:
            logger.debug("주봉 캐시 히트: %s (%d행)", ticker, len(cached))
            return cached

        end = datetime.now()
        start = end - timedelta(days=weeks * 7 + 30)

        df = self._fetch_daily_raw(ticker, start, end)
        if df.empty:
            return pd.DataFrame()

        # 일봉 → 주봉 리샘플
        weekly = df.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])

        if not weekly.empty:
            self._save_cache(weekly, cache_path)
            logger.info("주봉 수집 완료: %s (%d주)", ticker, len(weekly))

        return weekly

    # ── 연봉 리샘플 ──────────────────────────

    def get_yearly_from_monthly(self, monthly: pd.DataFrame) -> pd.DataFrame:
        """월봉 → 연봉 리샘플. S1 연간돌파 계산용."""
        if monthly.empty:
            return pd.DataFrame()
        yearly = monthly.resample("YE").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])
        return yearly

    # ── 내부: 일봉 데이터 수집 ────────────────

    def _fetch_daily_raw(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        """yfinance → FDR fallback으로 일봉 수집."""
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        # 1차: yfinance
        df = self._fetch_yfinance(ticker, start_str, end_str)
        if df is not None and not df.empty:
            return df

        # 2차: FinanceDataReader
        df = self._fetch_fdr(ticker, start_str, end_str)
        if df is not None and not df.empty:
            return df

        logger.warning("데이터 수집 실패: %s (yfinance + FDR 모두 실패)", ticker)
        return pd.DataFrame()

    def _fetch_yfinance(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        yf = self._ensure_yf()
        if yf is None:
            return None

        # 한국 ETF/종목: 6자리 숫자 → {code}.KS
        yf_ticker = f"{ticker}.KS" if ticker.isdigit() and len(ticker) == 6 else ticker
        try:
            t = yf.Ticker(yf_ticker)
            df = t.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = df.index.tz_localize(None)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                return df
        except Exception as e:
            logger.debug("yfinance 실패 %s: %s", yf_ticker, e)
        return None

    def _fetch_fdr(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        fdr = self._ensure_fdr()
        if fdr is None:
            return None
        # FDR은 ^ prefix 불필요 (예: ^KS11 → KS11)
        fdr_ticker = ticker.lstrip("^") if ticker.startswith("^") else ticker
        try:
            df = fdr.DataReader(fdr_ticker, start, end)
            if df is not None and not df.empty:
                cols = {c: c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns}
                if "Close" in cols:
                    return df[list(cols.values())]
        except Exception as e:
            logger.debug("FDR 실패 %s: %s", ticker, e)
        return None
