"""국내 개장일(휴장일) 판정 어댑터 — KIS CTCA0903R.

배경 (7/20, B-13): crontab이 요일 기반(1-5)이라 평일 공휴일에도 전체 배치가 실행됨.
7/17(제헌절) 실측 — 종가 0%·수급 0종목·DART 0건을 "장 운영일 비정상"으로 판정해
**🚨 D등급(13/18) 가짜 경보**가 발송됐다. 휴장일엔 국내시장 데이터가 없는 게 정상이다.

★판정 소스를 KIS 공식 API로 쓰는 이유: 우리 수집 파이프라인과 **독립**이어야 한다.
  kospi_index.csv 같은 자체 산출물로 판정하면 "수집 실패"가 "휴장"으로 오판되어
  진짜 장애를 은폐한다 (경보 시스템의 자기기만).

폴백 원칙: 판정 불가 시 **장날로 간주**(True). 휴장 오판(경보 억제)보다
알 수 없을 때 경보를 내는 쪽이 안전하다.

사용:
    from src.adapters.kis_holiday_adapter import is_trading_day
    if is_trading_day():  # 오늘
        ...
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

CACHE_PATH = PROJECT_ROOT / "data" / "trading_calendar.json"
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"


def _load_cache() -> dict[str, bool]:
    if CACHE_PATH.exists():
        try:
            return json.load(open(CACHE_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cal: dict[str, bool]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1, sort_keys=True)
    tmp.replace(CACHE_PATH)  # 원자 쓰기


def fetch_calendar(base_yyyymmdd: str) -> dict[str, bool]:
    """KIS 국내휴장일조회 → {YYYYMMDD: 개장여부}. 실패 시 빈 dict."""
    from src.adapters.kis_investor_adapter import (
        _issue_token, KIS_APP_KEY, KIS_APP_SECRET,
    )

    try:
        token = _issue_token()
        resp = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/chk-holiday",
            headers={
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {token}",
                "appkey": KIS_APP_KEY,
                "appsecret": KIS_APP_SECRET,
                "tr_id": "CTCA0903R",
                "custtype": "P",
            },
            params={"BASS_DT": base_yyyymmdd, "CTX_AREA_NK": "", "CTX_AREA_FK": ""},
            timeout=15,
        )
        data = resp.json()
        if data.get("rt_cd") != "0":
            logger.warning("[휴장일] KIS 조회 실패: %s", data.get("msg1", ""))
            return {}
        return {
            str(r.get("bass_dt", "")): (r.get("opnd_yn", "") == "Y")
            for r in data.get("output", [])
            if r.get("bass_dt")
        }
    except Exception as e:
        logger.warning("[휴장일] KIS 조회 예외: %s", e)
        return {}


def is_trading_day(target: date | None = None, *, allow_fetch: bool = True) -> bool:
    """국내 증시 개장일이면 True.

    캐시 → (미스 시) KIS 조회 → (실패 시) 주말만 휴장으로 보고 나머지는 True.
    ★판정 불가 시 True인 이유: 휴장 오판으로 경보를 잠재우는 것보다
      장날로 보고 경보를 내는 편이 안전하다.
    """
    d = target or date.today()
    key = d.strftime("%Y%m%d")

    cal = _load_cache()
    if key in cal:
        return cal[key]

    if allow_fetch:
        fetched = fetch_calendar(key)
        if fetched:
            cal.update(fetched)
            _save_cache(cal)
            if key in cal:
                return cal[key]

    if d.weekday() >= 5:  # 토·일
        return False
    logger.warning("[휴장일] %s 판정 불가 — 장날로 간주(안전측)", key)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for s in ("20260716", "20260717", "20260718", "20260720"):
        dt = datetime.strptime(s, "%Y%m%d").date()
        print(f"{s}: {'개장' if is_trading_day(dt) else '휴장'}")
