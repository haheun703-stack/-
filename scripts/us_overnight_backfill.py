"""
US Overnight Signal — Step 1: 히스토리 백필

yfinance로 미국 시장 핵심 지표 3년치 다운로드:
  - SPY (S&P 500), QQQ (NASDAQ 100), DIA (Dow 30)
  - XLK (Tech), XLF (Finance), XLE (Energy), XLI (Industrial), XLV (Healthcare)
  - SOXX (반도체), ^VIX (변동성)
  - TLT (20Y Treasury), DXY (달러인덱스 → UUP 대용)

저장: data/us_market/us_daily.parquet (날짜 인덱스, 멀티 티커)

사용법:
    python scripts/us_overnight_backfill.py [--years 3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

US_DIR = Path("data/us_market")
PARQUET_PATH = US_DIR / "us_daily.parquet"

# 핵심 US 티커 → 한국 시장 연관 섹터
TICKERS = {
    # 대형 지수
    "SPY": "S&P500",
    "QQQ": "NASDAQ100",
    "DIA": "DOW30",
    # 섹터 ETF
    "XLK": "Tech",
    "XLF": "Finance",
    "XLE": "Energy",
    "XLI": "Industrial",
    "XLV": "Healthcare",
    "SOXX": "Semiconductor",
    # 변동성 & 채권 & 달러
    "^VIX": "VIX",
    "TLT": "Treasury20Y",
    "UUP": "DollarIndex",
    # 한국 프록시 (미국 상장 Korea ETF)
    "EWY": "KoreaETF",
}

# 한국 섹터 매핑 (US ETF → KR 업종)
US_KR_SECTOR_MAP = {
    "XLK": ["반도체", "IT", "소프트웨어", "전자부품"],
    "SOXX": ["반도체", "전자부품"],
    "XLF": ["은행", "증권", "보험", "금융"],
    "XLE": ["에너지", "정유", "화학"],
    "XLI": ["조선", "기계", "건설", "자동차"],
    "XLV": ["제약", "바이오", "의료기기"],
}


def backfill(years: int = 3) -> pd.DataFrame:
    """yfinance로 US 시장 데이터 백필."""
    import yfinance as yf

    US_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    logger.info(f"US 시장 백필: {start.date()} ~ {end.date()} ({len(TICKERS)} 티커)")

    all_data = {}

    for ticker, label in TICKERS.items():
        try:
            obj = yf.Ticker(ticker)
            df = obj.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is None or df.empty:
                logger.warning(f"  {ticker} ({label}): 데이터 없음")
                continue

            # 날짜 인덱스 정리 (timezone 제거)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

            # 필요 컬럼만
            prefix = ticker.replace("^", "").lower()
            cols = {
                "Close": f"{prefix}_close",
                "Volume": f"{prefix}_volume",
                "High": f"{prefix}_high",
                "Low": f"{prefix}_low",
            }
            # rename 먼저, 그 다음 존재하는 컬럼만 선택
            df = df.rename(columns=cols)
            available = [c for c in cols.values() if c in df.columns]
            df = df[available]

            all_data[ticker] = df
            logger.info(f"  {ticker} ({label}): {len(df)}일")
        except Exception as e:
            logger.warning(f"  {ticker} ({label}): 실패 — {e}")

    if not all_data:
        logger.error("수집된 데이터 없음!")
        return pd.DataFrame()

    # 날짜 기준 병합
    merged = pd.DataFrame()
    for ticker, df in all_data.items():
        if merged.empty:
            merged = df
        else:
            merged = merged.join(df, how="outer")

    # 결측값 전방 채움 (휴일 등)
    merged = merged.ffill()

    # 파생 지표 계산
    merged = _calc_derived(merged)

    # 저장
    merged.to_parquet(PARQUET_PATH)
    logger.info(f"저장: {PARQUET_PATH} ({len(merged)}일 × {len(merged.columns)}컬럼)")

    return merged


def _calc_derived(df: pd.DataFrame) -> pd.DataFrame:
    """파생 지표 (수익률, 이평, 변동성 등)."""
    # 주요 지수 일간 수익률
    for prefix in ["spy", "qqq", "dia", "soxx", "ewy"]:
        col = f"{prefix}_close"
        if col in df.columns:
            df[f"{prefix}_ret_1d"] = df[col].pct_change()
            df[f"{prefix}_ret_5d"] = df[col].pct_change(5)
            df[f"{prefix}_sma_20"] = df[col].rolling(20).mean()
            df[f"{prefix}_above_sma20"] = (df[col] > df[f"{prefix}_sma_20"]).astype(int)

    # VIX 파생
    if "vix_close" in df.columns:
        df["vix_sma_20"] = df["vix_close"].rolling(20).mean()
        df["vix_zscore"] = (
            (df["vix_close"] - df["vix_close"].rolling(60).mean())
            / df["vix_close"].rolling(60).std()
        )
        df["vix_spike"] = (df["vix_close"] > df["vix_sma_20"] * 1.2).astype(int)

    # TLT (채권): 상승 = risk-off
    if "tlt_close" in df.columns:
        df["tlt_ret_1d"] = df["tlt_close"].pct_change()

    # 섹터 상대 강도 (vs SPY)
    if "spy_close" in df.columns:
        spy_ret = df["spy_close"].pct_change(5)
        for prefix in ["xlk", "xlf", "xle", "xli", "xlv", "soxx", "ewy"]:
            col = f"{prefix}_close"
            if col in df.columns:
                sector_ret = df[col].pct_change(5)
                df[f"{prefix}_rel_spy_5d"] = sector_ret - spy_ret

    return df


def main():
    parser = argparse.ArgumentParser(description="US Overnight Signal 히스토리 백필")
    parser.add_argument("--years", type=int, default=3, help="백필 년수 (기본: 3)")
    args = parser.parse_args()

    backfill(args.years)


if __name__ == "__main__":
    main()
