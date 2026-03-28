"""KIS API 기반 투자자별 매매동향 어댑터

pykrx get_market_trading_value_by_date 대체.
KIS REST API (FHKST01010900)로 외국인/기관/개인 순매수 데이터 수집.

사용처:
  1. extend_parquet_data.py → parquet 외국인합계/기관합계/개인 컬럼
  2. sector_investor_flow.py → 섹터별 스마트머니 흐름
  3. pykrx_supply_adapter.py → 개별종목 수급 분석
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_int(val, default: int = 0) -> int:
    """빈 문자열/None 안전 int 변환."""
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# KIS API 설정
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")

# 토큰 캐시 (모듈 수준 싱글턴)
_token_cache: dict = {"token": "", "expires": 0}


def _issue_token(max_retries: int = 3) -> str:
    """KIS 접근 토큰 발급 (캐시 활용, rate limit 재시도)."""
    now = time.time()
    if _token_cache["token"] and _token_cache["expires"] > now:
        return _token_cache["token"]

    url = f"{KIS_BASE_URL}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
    }

    last_error = None
    for attempt in range(max_retries):
        resp = requests.post(url, json=body, timeout=10)
        data = resp.json()

        token = data.get("access_token", "")
        if token:
            _token_cache["token"] = token
            _token_cache["expires"] = time.time() + 23 * 3600
            logger.info("KIS 토큰 발급 성공")
            return token

        error_code = data.get("error_code", "")
        last_error = data

        # EGW00133: 1분당 1회 제한 → 65초 대기 후 재시도
        if error_code == "EGW00133" and attempt < max_retries - 1:
            wait = 65
            logger.warning("KIS 토큰 rate limit — %d초 대기 후 재시도 (%d/%d)",
                           wait, attempt + 1, max_retries)
            time.sleep(wait)
            continue

        break

    raise RuntimeError(f"KIS 토큰 발급 실패: {last_error}")


def fetch_investor_by_ticker(ticker: str) -> pd.DataFrame:
    """종목별 투자자 매매동향 30일치 조회.

    KIS API: FHKST01010900 (주식현재가 투자자)

    Args:
        ticker: 종목코드 (6자리)

    Returns:
        DataFrame with columns:
            date, close, 외국인합계, 기관합계, 개인,
            foreign_buy_vol, foreign_sell_vol,
            inst_buy_vol, inst_sell_vol
        Index: date (datetime)
    """
    token = _issue_token()

    url = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST01010900",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    data = resp.json()

    if data.get("rt_cd") != "0":
        logger.warning("[%s] KIS 투자자 조회 실패: %s", ticker, data.get("msg1", ""))
        return pd.DataFrame()

    rows = data.get("output", [])
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        date_str = r.get("stck_bsop_date", "")
        if not date_str:
            continue
        records.append({
            "date": pd.Timestamp(datetime.strptime(date_str, "%Y%m%d")),
            "close": _safe_int(r.get("stck_clpr")),
            "외국인합계": _safe_int(r.get("frgn_ntby_tr_pbmn")) * 1_000_000,  # 백만→원
            "기관합계": _safe_int(r.get("orgn_ntby_tr_pbmn")) * 1_000_000,
            "개인": _safe_int(r.get("prsn_ntby_tr_pbmn")) * 1_000_000,
            "foreign_net_qty": _safe_int(r.get("frgn_ntby_qty")),
            "inst_net_qty": _safe_int(r.get("orgn_ntby_qty")),
            "individual_net_qty": _safe_int(r.get("prsn_ntby_qty")),
            "foreign_buy_vol": _safe_int(r.get("frgn_shnu_vol")),
            "foreign_sell_vol": _safe_int(r.get("frgn_seln_vol")),
            "inst_buy_vol": _safe_int(r.get("orgn_shnu_vol")),
            "inst_sell_vol": _safe_int(r.get("orgn_seln_vol")),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.set_index("date").sort_index()
    return df


def fetch_investor_batch(
    tickers: list[str],
    delay: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """여러 종목 투자자 매매동향 배치 조회.

    Args:
        tickers: 종목코드 리스트
        delay: API 호출 간 대기 (초, rate limit 방지)

    Returns:
        {ticker: DataFrame}
    """
    results = {}
    for i, ticker in enumerate(tickers):
        try:
            df = fetch_investor_by_ticker(ticker)
            results[ticker] = df
            if i > 0 and delay > 0:
                time.sleep(delay)
        except Exception as e:
            logger.warning("[%s] 투자자 데이터 실패: %s", ticker, e)
            results[ticker] = pd.DataFrame()

    success = sum(1 for df in results.values() if not df.empty)
    logger.info("KIS 투자자 배치 완료: %d/%d 성공", success, len(tickers))
    return results
