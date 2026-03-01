"""
L4 글로벌 매크로 어댑터

수집 데이터:
  - VIX (CBOE 변동성 지수) — FinanceDataReader
  - USD/KRX (달러/원 환율) — FinanceDataReader
  - KOSPI — FinanceDataReader
  - SOXX (반도체 ETF) — yfinance (fallback: FinanceDataReader)

저장: data/macro/global_indices.parquet (단일 파일, 날짜 인덱스)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MACRO_DIR = Path("data/macro")


class MacroAdapter:
    """글로벌 매크로 데이터 수집"""

    def __init__(self):
        MACRO_DIR.mkdir(parents=True, exist_ok=True)
        self._fdr = None
        self._yf = None

    def _ensure_fdr(self):
        if self._fdr is None:
            import FinanceDataReader as fdr
            self._fdr = fdr
        return self._fdr

    def _ensure_yf(self):
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.warning("yfinance 미설치: pip install yfinance")
                return None
        return self._yf

    def fetch_vix(self, start: str, end: str) -> pd.DataFrame:
        """VIX 지수"""
        fdr = self._ensure_fdr()
        try:
            df = fdr.DataReader("VIX", start, end)
            if df is not None and not df.empty:
                return df[["Close"]].rename(columns={"Close": "vix_close"})
        except Exception as e:
            logger.warning(f"VIX 수집 실패: {e}")
        return pd.DataFrame()

    def fetch_usdkrw(self, start: str, end: str) -> pd.DataFrame:
        """달러/원 환율"""
        fdr = self._ensure_fdr()
        try:
            df = fdr.DataReader("USD/KRW", start, end)
            if df is not None and not df.empty:
                return df[["Close"]].rename(columns={"Close": "usdkrw_close"})
        except Exception as e:
            logger.warning(f"USD/KRW 수집 실패: {e}")
        return pd.DataFrame()

    def fetch_kospi(self, start: str, end: str) -> pd.DataFrame:
        """KOSPI 지수"""
        fdr = self._ensure_fdr()
        try:
            df = fdr.DataReader("KS11", start, end)
            if df is not None and not df.empty:
                return df[["Close"]].rename(columns={"Close": "kospi_close"})
        except Exception as e:
            logger.warning(f"KOSPI 수집 실패: {e}")
        return pd.DataFrame()

    def fetch_soxx(self, start: str, end: str) -> pd.DataFrame:
        """SOXX (반도체 ETF) — yfinance 우선, fallback FDR"""
        yf = self._ensure_yf()
        if yf is not None:
            try:
                ticker = yf.Ticker("SOXX")
                df = ticker.history(start=start, end=end)
                if df is not None and not df.empty:
                    df.index = df.index.tz_localize(None)
                    return df[["Close"]].rename(columns={"Close": "soxx_close"})
            except Exception as e:
                logger.warning(f"SOXX(yfinance) 수집 실패: {e}")

        # fallback: FinanceDataReader
        fdr = self._ensure_fdr()
        try:
            df = fdr.DataReader("SOXX", start, end)
            if df is not None and not df.empty:
                return df[["Close"]].rename(columns={"Close": "soxx_close"})
        except Exception as e:
            logger.warning(f"SOXX(FDR) 수집 실패: {e}")
        return pd.DataFrame()

    def fetch_all(self, start: str, end: str) -> pd.DataFrame:
        """4개 글로벌 지수 합산 DataFrame (날짜 인덱스, ffill 처리)"""
        vix = self.fetch_vix(start, end)
        usdkrw = self.fetch_usdkrw(start, end)
        kospi = self.fetch_kospi(start, end)
        soxx = self.fetch_soxx(start, end)

        frames = [f for f in [vix, usdkrw, kospi, soxx] if not f.empty]
        if not frames:
            logger.error("글로벌 매크로 데이터 수집 실패: 모든 소스 없음")
            return pd.DataFrame()

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, how="outer")

        # 영업일 불일치 처리
        merged = merged.ffill()
        merged.index.name = "date"

        return merged

    def save(self, df: pd.DataFrame) -> Path:
        """parquet 저장"""
        path = MACRO_DIR / "global_indices.parquet"
        df.to_parquet(path)
        logger.info(f"매크로 데이터 저장: {path} ({len(df)}일)")
        return path

    def load(self) -> pd.DataFrame:
        """parquet 로드"""
        path = MACRO_DIR / "global_indices.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
