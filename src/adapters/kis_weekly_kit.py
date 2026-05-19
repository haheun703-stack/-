"""KIS API 주봉 차트 — 종목 + 지수 (차트영웅 매매법 Gate 5 + 매크로 Gate 1).

차트영웅 룰: 주봉 스토캐스틱 K < 30 = 저점 진입 신호.
- 종목 주봉: inquire-daily-itemchartprice (FID_PERIOD_DIV_CODE=W)
- 지수 주봉: inquire-daily-indexchartprice (KOSPI=0001, KOSDAQ=1001)

검증: 2026-05-19, 005930 28주 + KOSPI 29주 정상 수집.

사용 예:
    from src.adapters.kis_weekly_kit import get_stock_weekly, get_index_weekly, compute_weekly_stoch_k
    w = get_stock_weekly('005930', '20251101', '20260519')   # 삼성전자 28주
    k = get_index_weekly('0001', '20251101', '20260519')     # KOSPI 29주
    stoch = compute_weekly_stoch_k(w, period=14)             # 주봉 스토캐스틱 K
"""

import requests

from src.adapters.kis_nxt_kit import _BASE, _headers


def get_stock_weekly(code: str, from_date: str, to_date: str) -> list[dict] | None:
    """종목 주봉 OHLCV.

    Args:
        code: 6자리 종목코드 ('005930')
        from_date: 'YYYYMMDD'
        to_date:   'YYYYMMDD'
    Returns:
        주봉 리스트 (최신 → 과거 순). 각 dict:
            { date, open, high, low, close, volume }
    """
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        headers=_headers("FHKST03010100"),
        params={
            "fid_cond_mrkt_div_code": "J",      # KRX (주봉은 NX보다 J가 안정적)
            "fid_input_iscd": code,
            "fid_input_date_1": from_date,
            "fid_input_date_2": to_date,
            "fid_period_div_code": "W",         # 주봉
            "fid_org_adj_prc": "0",             # 0 = 수정주가, 1 = 원주가
        },
        timeout=15,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    out = []
    for row in r.json().get("output2", []):
        try:
            out.append({
                "date":   row.get("stck_bsop_date"),
                "open":   int(row.get("stck_oprc", 0) or 0),
                "high":   int(row.get("stck_hgpr", 0) or 0),
                "low":    int(row.get("stck_lwpr", 0) or 0),
                "close":  int(row.get("stck_clpr", 0) or 0),
                "volume": int(row.get("acml_vol",  0) or 0),
            })
        except Exception:
            pass
    return out


def get_index_weekly(idx_code: str, from_date: str, to_date: str) -> list[dict] | None:
    """지수 주봉 OHLC (KOSPI='0001', KOSDAQ='1001').

    KOSPI 주봉 스토캐스틱은 차트영웅 매크로 4-시그널 1번 — 시장 저점 잡기 핵심.
    """
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
        headers=_headers("FHKUP03500100"),
        params={
            "fid_cond_mrkt_div_code": "U",      # 업종/지수
            "fid_input_iscd": idx_code,
            "fid_input_date_1": from_date,
            "fid_input_date_2": to_date,
            "fid_period_div_code": "W",
        },
        timeout=15,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    out = []
    for row in r.json().get("output2", []):
        try:
            out.append({
                "date":  row.get("stck_bsop_date"),
                "open":  float(row.get("bstp_nmix_oprc", 0) or 0),
                "high":  float(row.get("bstp_nmix_hgpr", 0) or 0),
                "low":   float(row.get("bstp_nmix_lwpr", 0) or 0),
                "close": float(row.get("bstp_nmix_prpr", 0) or 0),
            })
        except Exception:
            pass
    return out


def compute_weekly_stoch_k(weekly: list[dict], period: int = 14) -> float | None:
    """주봉 스토캐스틱 %K (Fast) — 가장 최근 주차 기준.

    K = (current_close - min_low_N) / (max_high_N - min_low_N) * 100

    차트영웅 룰: K < 30 = 과매도, 매수 신호 가능.

    Args:
        weekly: get_stock_weekly() 또는 get_index_weekly() 출력 (최신→과거)
        period: 룩백 주차 (default 14)
    Returns:
        %K 값 (0~100), 데이터 부족 시 None
    """
    if not weekly or len(weekly) < period:
        return None
    window = weekly[:period]   # 최신 period 주차
    current_close = window[0]["close"]
    max_high = max(r["high"] for r in window)
    min_low  = min(r["low"]  for r in window)
    if max_high == min_low:
        return 50.0
    return round((current_close - min_low) / (max_high - min_low) * 100, 2)


def is_weekly_oversold(weekly: list[dict], period: int = 14, threshold: float = 30.0) -> bool:
    """주봉 과매도 판정 (차트영웅 매매법 Gate 5).

    K < threshold = 매수 진입 가능 (저점 시그널).
    """
    k = compute_weekly_stoch_k(weekly, period)
    return k is not None and k < threshold


def get_stock_daily(code: str, from_date: str, to_date: str) -> list[dict] | None:
    """종목 일봉 OHLCV (Gate 3 MA20 dev 계산용).

    Returns: [{date, open, high, low, close, volume}, ...] (최신→과거)
    """
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        headers=_headers("FHKST03010100"),
        params={
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
            "fid_input_date_1": from_date,
            "fid_input_date_2": to_date,
            "fid_period_div_code": "D",
            "fid_org_adj_prc": "0",
        },
        timeout=15,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    out = []
    for row in r.json().get("output2", []):
        try:
            out.append({
                "date": row.get("stck_bsop_date"),
                "open": int(row.get("stck_oprc", 0) or 0),
                "high": int(row.get("stck_hgpr", 0) or 0),
                "low":  int(row.get("stck_lwpr", 0) or 0),
                "close": int(row.get("stck_clpr", 0) or 0),
                "volume": int(row.get("acml_vol", 0) or 0),
            })
        except Exception:
            pass
    return out


def compute_ma20_dev(code: str) -> float | None:
    """MA20 대비 현재가 편차 % (Gate 3).

    차트영웅 룰 완화 버전: -5 ~ +5% = 저변동 진입 영역.
    """
    import datetime as dt
    end = dt.date.today()
    start = end - dt.timedelta(days=45)   # 영업일 20일 + 여유
    daily = get_stock_daily(code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    if not daily or len(daily) < 20:
        return None
    closes = [d["close"] for d in daily[:20]]   # 최근 20영업일
    ma20 = sum(closes) / 20
    current = daily[0]["close"]
    if ma20 <= 0:
        return None
    return round((current - ma20) / ma20 * 100, 2)


if __name__ == "__main__":
    # 자가 검증
    print("=== KIS 주봉 어댑터 검증 (2026-05-19) ===")
    targets = [
        ("005930", "삼성전자", get_stock_weekly),
        ("000660", "SK하이닉스", get_stock_weekly),
        ("0001",   "KOSPI 지수", get_index_weekly),
        ("1001",   "KOSDAQ 지수", get_index_weekly),
    ]
    for code, name, fn in targets:
        data = fn(code, "20251101", "20260519")
        if not data:
            print(f"[FAIL] {code} {name}")
            continue
        latest = data[0]
        k = compute_weekly_stoch_k(data, 14)
        oversold = is_weekly_oversold(data, 14, 30.0)
        print(f"[OK]  {code:6} {name:12} 주차={len(data):>2} "
              f"최신={latest['date']} 종가={latest['close']:>10.2f} "
              f"%K={k:>6.2f} 과매도={'YES⭐' if oversold else 'NO'}")
