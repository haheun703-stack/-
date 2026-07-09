# -*- coding: utf-8 -*-
"""일봉 히스토리 어댑터 — 백테스트용 과거 종가.

네이버 fchart (키 불필요):
  https://fchart.stock.naver.com/sise.nhn?symbol=005930&timeframe=day&count=300&requestType=0
  응답: XML, <item data="YYYYMMDD|시가|고가|저가|종가|거래량"/>
"""
from __future__ import annotations

import logging
import re

import requests

log = logging.getLogger("insight_signals.history")

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

_FCHART_URL = "https://fchart.stock.naver.com/sise.nhn"
_ITEM_RE = re.compile(r'<item\s+data="([^"]+)"')


def daily_closes(stock_code: str, count: int = 300, timeout: int = 15):
    """과거 일봉 종가 [(YYYYMMDD, close), ...] 오름차순. 실패 시 빈 리스트."""
    try:
        r = requests.get(
            _FCHART_URL,
            params={
                "symbol": stock_code,
                "timeframe": "day",
                "count": count,
                "requestType": "0",
            },
            headers=UA,
            timeout=timeout,
        )
        r.raise_for_status()
        out = []
        for m in _ITEM_RE.finditer(r.text):
            parts = m.group(1).split("|")
            if len(parts) >= 5:
                date, close = parts[0].strip(), parts[4].strip()
                if len(date) == 8 and close:
                    try:
                        out.append((date, float(close)))
                    except ValueError:
                        continue
        out.sort(key=lambda x: x[0])
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("일봉 조회 실패 [%s]: %s", stock_code, e)
        return []
