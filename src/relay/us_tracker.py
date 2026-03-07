"""
US 대장주 가격 추적기
========================
yfinance로 US 개별 대장주의 최신 가격/레벨을 수집하고
strength/weakness 판정.

출력: data/relay/us_leaders.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

from src.relay.config import load_relay_config, get_sectors, DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "us_leaders.json"


def fetch_us_leaders(config: dict = None) -> dict:
    """모든 섹터의 US 대장주 데이터 수집 + 레벨 계산.

    Returns:
        {sector_key: {ticker: {close, prev_high, prev_low, sma20,
                               strength_level, weakness_level,
                               is_strong, is_weak, ret_1d, ret_5d, trend}}}
    """
    if config is None:
        config = load_relay_config()
    sectors = get_sectors(config)

    # 모든 US 티커 수집 (중복 제거)
    ticker_to_sectors: dict[str, list[str]] = {}
    ticker_info: dict[str, dict] = {}
    for sec_key, sec in sectors.items():
        for leader in sec.get("us_leaders", []):
            t = leader["ticker"]
            if t not in ticker_to_sectors:
                ticker_to_sectors[t] = []
            ticker_to_sectors[t].append(sec_key)
            ticker_info[t] = leader

    all_tickers = list(ticker_to_sectors.keys())
    if not all_tickers:
        logger.warning("US 대장주 티커 없음")
        return {}

    # yfinance 일괄 다운로드 (최근 30일, SMA20 계산용)
    logger.info("US 대장주 %d개 다운로드: %s", len(all_tickers), all_tickers)
    try:
        raw = yf.download(all_tickers, period="1mo", group_by="ticker", progress=False)
    except Exception as e:
        logger.error("yfinance 다운로드 실패: %s", e)
        return {}

    # 티커별 분석
    ticker_data: dict[str, dict] = {}
    for ticker in all_tickers:
        try:
            if len(all_tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            df = df.dropna(subset=["Close"])
            if len(df) < 5:
                logger.warning("%s: 데이터 부족 (%d일)", ticker, len(df))
                continue

            closes = df["Close"].values
            highs = df["High"].values
            lows = df["Low"].values

            close = float(closes[-1])
            prev_high = float(highs[-2]) if len(highs) >= 2 else close
            prev_low = float(lows[-2]) if len(lows) >= 2 else close
            prev_close = float(closes[-2]) if len(closes) >= 2 else close

            # SMA20
            sma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else close

            # 수익률
            ret_1d = (close / prev_close - 1) * 100 if prev_close > 0 else 0
            ret_5d = (close / float(closes[-6]) - 1) * 100 if len(closes) >= 6 else 0

            # strength / weakness 레벨
            strength_level = round(prev_high * 1.001, 2)   # 전일고가 +0.1%
            weakness_level = round(prev_low, 2)              # 전일 저가

            is_strong = close > strength_level
            is_weak = close < weakness_level

            # 추세 판정
            above_sma20 = close > sma20
            if ret_1d > 1.0:
                trend = "strong_up"
            elif ret_1d > 0:
                trend = "up"
            elif ret_1d > -1.0:
                trend = "flat"
            elif ret_1d > -2.0:
                trend = "down"
            else:
                trend = "strong_down"

            ticker_data[ticker] = {
                "ticker": ticker,
                "name": ticker_info.get(ticker, {}).get("name", ticker),
                "role": ticker_info.get(ticker, {}).get("role", ""),
                "close": round(close, 2),
                "prev_high": round(prev_high, 2),
                "prev_low": round(prev_low, 2),
                "prev_close": round(prev_close, 2),
                "sma20": round(sma20, 2),
                "strength_level": strength_level,
                "weakness_level": weakness_level,
                "is_strong": bool(is_strong),
                "is_weak": bool(is_weak),
                "above_sma20": bool(above_sma20),
                "ret_1d": round(ret_1d, 2),
                "ret_5d": round(ret_5d, 2),
                "trend": trend,
                "last_date": str(df.index[-1].date()),
            }
            logger.info(
                "%s: $%.2f (1d:%+.1f%%) strong=%s weak=%s",
                ticker, close, ret_1d, is_strong, is_weak,
            )
        except Exception as e:
            logger.warning("%s 분석 실패: %s", ticker, e)

    # 섹터별 정리
    result = {}
    for sec_key, sec in sectors.items():
        sec_leaders = {}
        for leader in sec.get("us_leaders", []):
            t = leader["ticker"]
            if t in ticker_data:
                sec_leaders[t] = ticker_data[t]
        result[sec_key] = {
            "name": sec.get("name", sec_key),
            "type": sec.get("type", "persistent"),
            "leaders": sec_leaders,
            "strong_count": sum(1 for v in sec_leaders.values() if v.get("is_strong")),
            "weak_count": sum(1 for v in sec_leaders.values() if v.get("is_weak")),
            "min_strong": sec.get("us_leader_min_strong", 2),
            "all_strong_enough": (
                sum(1 for v in sec_leaders.values() if v.get("is_strong"))
                >= sec.get("us_leader_min_strong", 2)
            ),
        }

    return result


def save_us_leaders(data: dict) -> Path:
    """US 대장주 데이터를 JSON으로 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sectors": data,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("US 대장주 데이터 저장: %s", OUTPUT_PATH)
    return OUTPUT_PATH


def load_us_leaders() -> dict:
    """저장된 US 대장주 데이터 로드."""
    if not OUTPUT_PATH.exists():
        return {}
    try:
        raw = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return raw.get("sectors", {})
    except Exception as e:
        logger.warning("US 대장주 데이터 로드 실패: %s", e)
        return {}


def update_and_save(config: dict = None) -> dict:
    """US 대장주 업데이트 + 저장 (원스텝)."""
    data = fetch_us_leaders(config)
    if data:
        save_us_leaders(data)
    return data
