"""KIS API NX 통합 시세 + WebSocket 실시간 — 단일 파일 키트.

KISBOT(2026-05-13)에서 검증된 NXT 통합 패턴을 퀀트봇에 이식.
- fid_cond_mrkt_div_code='NX' → KRX + NXT 통합 시세 (HTS [0416] 통합모드와 동일)
- 매매동향(외인/기관)은 J/NX 동일. 가격/거래량만 차이 (NX = KRX + NXT 합산)
- 별도 NXT API 신청 불필요. 모든 KIS 사용자 즉시 사용 가능

본 키트는 *신규* 분석/도구 전용. 기존 코드는 변경하지 않음 (백테스트 일관성 보호).

사용 예:
    from src.adapters.kis_nxt_kit import get_nx_price, get_supply_5day, get_session
    p = get_nx_price('005930')  # 삼성전자 NX 통합 현재가
    s = get_supply_5day('005930')  # 5일 외인+기관 양매수 판정
"""

import asyncio
import json
import os
import sys
from datetime import datetime, time as dtime
from pathlib import Path

import requests
# C4 보강: 표준 python-dotenv 사용 (따옴표/이스케이프 자동 처리)
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

_BASE = "https://openapi.koreainvestment.com:9443"
_token_cache: dict = {"token": None, "time": None}
_approval_cache: dict = {"key": None, "time": None}


def get_token() -> str | None:
    """KIS 접근 토큰 (1시간 캐시)."""
    if _token_cache["token"] and (datetime.now() - _token_cache["time"]).total_seconds() < 3600:
        return _token_cache["token"]
    r = requests.post(
        f"{_BASE}/oauth2/tokenP",
        json={
            "grant_type": "client_credentials",
            "appkey": os.getenv("KIS_APP_KEY"),
            "appsecret": os.getenv("KIS_APP_SECRET"),
        },
        timeout=10,
    )
    t = r.json().get("access_token")
    _token_cache.update({"token": t, "time": datetime.now()})
    return t


def _headers(tr_id: str) -> dict:
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {get_token()}",
        "appkey": os.getenv("KIS_APP_KEY"),
        "appsecret": os.getenv("KIS_APP_SECRET"),
        "tr_id": tr_id,
        "custtype": "P",
    }


def get_nx_price(code: str) -> dict | None:
    """NX(KRX+NXT 통합) 현재가."""
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=_headers("FHKST01010100"),
        params={"fid_cond_mrkt_div_code": "NX", "fid_input_iscd": code},
        timeout=5,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    o = r.json().get("output", {})
    # C1 보강 #1: stck_prpr=0 (거래정지/시간외) 시 None 반환 — 가짜 손절 방지
    price = int(o.get("stck_prpr", 0) or 0)
    if price <= 0:
        return None
    return {
        "code": code,
        "price": price,
        "change_pct": float(o.get("prdy_ctrt", 0) or 0),
        "volume": int(o.get("acml_vol", 0) or 0),
        "open": int(o.get("stck_oprc", 0) or 0),
        "high": int(o.get("stck_hgpr", 0) or 0),
        "low": int(o.get("stck_lwpr", 0) or 0),
    }


def get_nx_supply(code: str, days: int = 30) -> list[dict] | None:
    """NX(KRX+NXT 통합) 투자자별 매매. J/NX 동일 데이터(검증완료) but 명시."""
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-investor",
        headers=_headers("FHKST01010900"),
        params={"fid_cond_mrkt_div_code": "NX", "fid_input_iscd": code},
        timeout=10,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    out = []
    for row in r.json().get("output", [])[:days]:
        try:
            out.append({
                "date": row.get("stck_bsop_date"),
                "close": int(row.get("stck_clpr", 0) or 0),
                "foreign_백만원": int(row.get("frgn_ntby_tr_pbmn", 0) or 0),
                "institution_백만원": int(row.get("orgn_ntby_tr_pbmn", 0) or 0),
                "individual_백만원": int(row.get("prsn_ntby_tr_pbmn", 0) or 0),
            })
        except Exception:
            pass
    return out


def get_nx_daily_chart(code: str, fromdate: str, todate: str) -> list[dict] | None:
    """NX 일봉 차트 (KRX+NXT 통합 가격/거래량)."""
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        headers=_headers("FHKST03010100"),
        params={
            "fid_cond_mrkt_div_code": "NX",
            "fid_input_iscd": code,
            "fid_input_date_1": fromdate,
            "fid_input_date_2": todate,
            "fid_period_div_code": "D",
            "fid_org_adj_prc": "0",
        },
        timeout=15,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    return r.json().get("output2", [])


def get_supply_5day(code: str) -> dict | None:
    """5일 외인+기관 매매 요약 + 양매수 판정."""
    d = get_nx_supply(code, days=5)
    if not d:
        return None
    f = sum(x["foreign_백만원"] for x in d)
    i = sum(x["institution_백만원"] for x in d)
    return {
        "code": code,
        "foreign_5d_백만원": f,
        "institution_5d_백만원": i,
        "is_yangmaesoo": f > 0 and i > 0,
        "is_불일치": (f > 0) != (i > 0),
        "daily": d,
    }


# === WebSocket 실시간 ===
def get_approval_key(force_refresh: bool = False) -> str | None:
    """WebSocket 전용 approval_key (24시간 캐시).

    Args:
        force_refresh: True면 캐시 무시하고 새로 발급 (자정 넘긴 stale key 대응).
    """
    if (
        not force_refresh
        and _approval_cache["key"]
        and (datetime.now() - _approval_cache["time"]).total_seconds() < 24 * 3600
    ):
        return _approval_cache["key"]
    r = requests.post(
        f"{_BASE}/oauth2/Approval",
        headers={"content-type": "application/json"},
        json={
            "grant_type": "client_credentials",
            "appkey": os.getenv("KIS_APP_KEY"),
            "secretkey": os.getenv("KIS_APP_SECRET"),
        },
        timeout=10,
    )
    ak = r.json().get("approval_key")
    if ak:
        _approval_cache.update({"key": ak, "time": datetime.now()})
    return ak


def is_yangmaesoo_5d(rows: list[dict]) -> bool:
    """5일 양매수 판정 — 외인+기관 동시 매수 3일+ (단타봇 5/14 검증 패턴).

    Args:
        rows: get_nx_supply() 반환 (최신순). 각 dict는 foreign_백만원/institution_백만원 필드 보유.

    Returns:
        True if 외인+기관 동시 매수 3일 이상 (최근 5일 중)
    """
    if not rows or len(rows) < 5:
        return False
    last5 = rows[:5]
    co_buy_days = sum(
        1 for r in last5
        if r.get("foreign_백만원", 0) > 0 and r.get("institution_백만원", 0) > 0
    )
    return co_buy_days >= 3


async def subscribe_realtime(codes: list[str], on_tick, tr_id: str = "H0STCNT0"):
    """실시간 체결가 구독.

    Args:
        codes: 6자리 종목코드 리스트 (예: ['005930', '000660'])
        on_tick: callback(code, price, volume, time)
        tr_id: H0STCNT0=체결가 / H0STASP0=호가 / H0STMOM0=지수
    """
    import websockets  # 지연 import (WebSocket 안 쓰면 의존 강제 X)

    ak = get_approval_key()
    uri = "ws://ops.koreainvestment.com:21000"
    async with websockets.connect(uri) as ws:
        for code in codes:
            req = {
                "header": {
                    "approval_key": ak,
                    "custtype": "P",
                    "tr_type": "1",
                    "content-type": "utf-8",
                },
                "body": {"input": {"tr_id": tr_id, "tr_key": code}},
            }
            await ws.send(json.dumps(req))
            await asyncio.sleep(0.1)
        while True:
            data = await ws.recv()
            if data[0] in ("0", "1"):
                parts = data.split("|")
                if len(parts) >= 4:
                    f = parts[3].split("^")
                    on_tick(
                        f[0],
                        int(f[2]) if f[2] else 0,
                        int(f[12]) if f[12] else 0,
                        f[1],
                    )


# === 시간대 분기 ===
def get_session() -> str:
    """현재 시간대 → 데이터 소스 추천."""
    n = datetime.now()
    if n.weekday() >= 5:
        return "WEEKEND_GLOBAL"
    t = n.time()
    if dtime(8, 0) <= t < dtime(8, 50):
        return "NXT_PREMARKET"
    if dtime(8, 55) <= t < dtime(15, 30):
        return "KRX_WEBSOCKET"
    if dtime(15, 30) <= t < dtime(20, 0):
        return "NXT_AFTERMARKET"
    if t >= dtime(20, 0) or t < dtime(8, 0):
        return "GLOBAL_ETF"
    return "IDLE"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    test_code = "005930"  # 삼성전자 — NXT 참여 확실
    print("=== KIS NX 통합 시세 검증 ===")
    print(f"현재 세션: {get_session()}")
    p = get_nx_price(test_code)
    print(f"{test_code} 현재가 NX: {p}")
    s = get_supply_5day(test_code)
    if s:
        print(f"{test_code} 5일 외인:  {s['foreign_5d_백만원']:>12,} 백만원")
        print(f"{test_code} 5일 기관:  {s['institution_5d_백만원']:>12,} 백만원")
        print(f"{test_code} 양매수:    {s['is_yangmaesoo']}")
    else:
        print(f"{test_code} 매매동향 응답 없음")
