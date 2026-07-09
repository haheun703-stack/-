# -*- coding: utf-8 -*-
"""시세 폴백 어댑터 — KIS 키가 없어도 성과 추적이 돌도록.

1순위: KIS 현재가 (kis_flow_client 쪽에서 시도)
2순위: 네이버 폴링 API
3순위: 네이버 종목 페이지 파싱
"""
from __future__ import annotations

import logging
import re

import requests

log = logging.getLogger("insight_signals.price")

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

_POLLING_URL = "https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
_SISE_URL = "https://finance.naver.com/item/sise.naver?code={code}"
_SISE_RE = re.compile(r'id="_nowVal"[^>]*>([\d,]+)<')


def naver_price(stock_code: str, timeout: int = 10):
    """네이버에서 현재가(또는 종가). 실패 시 None."""
    # 1) 폴링 API
    try:
        r = requests.get(_POLLING_URL.format(code=stock_code), headers=UA, timeout=timeout)
        if r.ok:
            data = r.json()
            datas = data.get("datas") or []
            if datas:
                v = str(datas[0].get("closePrice", "")).replace(",", "")
                if v:
                    return float(v)
    except Exception as e:  # noqa: BLE001
        log.debug("네이버 폴링 실패 [%s]: %s", stock_code, e)

    # 2) 종목 페이지 파싱
    try:
        r = requests.get(_SISE_URL.format(code=stock_code), headers=UA, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "euc-kr"
        m = _SISE_RE.search(r.text)
        if m:
            return float(m.group(1).replace(",", ""))
    except Exception as e:  # noqa: BLE001
        log.warning("네이버 시세 파싱 실패 [%s]: %s", stock_code, e)
    return None


def get_price(stock_code: str, kis_client=None):
    """KIS 우선, 실패하면 네이버 폴백."""
    if kis_client is not None and getattr(kis_client, "available", False):
        p = kis_client.current_price(stock_code)
        if p:
            return p
    return naver_price(stock_code)
