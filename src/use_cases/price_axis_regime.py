"""FLOWX Market OS v1 관측 레이어 — 가격축 레짐 (지시서 1단계).

단일 종목 일봉 OHLCV에서 "가격의 위치"를 라벨로만 뽑는다. 매수/매도 신호가 아니라
관측 라벨이다. C60 단독 hard gate를 바꾸지 않는다 — 전부 shadow / label.

포함:
- price_axis    : 전일/주봉/월봉/반기/연초 시가 대비 현재 위치(ABOVE/BELOW)
- candle_turn   : 음양(바닥전환 후보)/양음(고점경고) 캔들 전환 라벨
- annual_overheat: 1년 수익률 + 과열 등급(OVERHEAT_300/500/1000)
- ipo_reversion : 상장 시초가 대비 낙폭(상장일 메타 주입 시에만, 없으면 빈 결과)

★주문/매도/스케줄러/SAJANG 경로 import·호출 0. 입력은 OHLCV·메타뿐, 출력은 라벨뿐.
라벨은 매수 조건이 아니다(즉시 hard gate 승격 금지). 6/8~6/12 관측 후 성과 비교.

설계: 진행 지시서 "주봉/반기/시가축 관측 레이어" §A·§3·§4·§6.
"""

from __future__ import annotations

import pandas as pd

from src.etf.c60_shadow import normalize_ohlcv
from src.etf.samsung_single_leverage_shadow import load_daily_ohlcv

PRICE_AXIS_VERSION = "price_axis_regime_v1"

# 과열 등급 임계(연간 수익률 %)
OVERHEAT_300 = 300.0
OVERHEAT_500 = 500.0
OVERHEAT_1000 = 1000.0

# near_half_year_high 판정: 반기 고가의 이 비율 이상이면 신고가 근처
NEAR_HIGH_RATIO = 0.97
# new_half_year_high_20d 판정 창(거래일)
NEW_HIGH_WINDOW = 20

# IPO 되돌림 분류 임계(상장 시초가 대비 낙폭 %)
IPO_CORE_DRAWDOWN = -50.0
IPO_WATCH_DRAWDOWN = -30.0
IPO_LOW_QUALITY_TAGS = {"LOW_QUALITY", "MANAGED", "관리종목", "SPAC_RISK"}


def _pct(value: float, base: float) -> float | None:
    if not base:
        return None
    return round((value - base) / base * 100, 2)


def _state(current_close: float, anchor_open: float | None) -> str | None:
    """현재 종가가 기준 시가 위면 ABOVE, 아래면 BELOW."""
    if anchor_open is None or anchor_open <= 0:
        return None
    return "ABOVE" if current_close >= anchor_open else "BELOW"


def _first_open_in_period(df: pd.DataFrame, mask: pd.Series) -> float | None:
    """mask(현재 기간) True인 첫 거래일의 시가."""
    seg = df.loc[mask]
    if seg.empty or "open" not in seg.columns:
        return None
    val = float(seg["open"].iloc[0])
    return val if val > 0 else None


def compute_price_axis(df: pd.DataFrame) -> dict:
    """일봉/주봉/월봉/반기/연초 시가 대비 현재 위치. 순수 함수.

    df: 정규화된 OHLCV(DatetimeIndex 오름차순, open/high/low/close). 마지막 행=현재.
    각 기간 시가 = 그 기간 첫 거래일의 시가(주봉=현재 주의 첫날, 반기=1/7월 첫날 등).
    전일 시가 = 직전 거래일의 시가.
    """
    if df is None or df.empty or "open" not in df.columns:
        return {"data_available": False, "labels": []}

    last_ts = df.index[-1]
    current_close = float(df["close"].iloc[-1])

    # 전일 시가
    daily_open = float(df["open"].iloc[-2]) if len(df) >= 2 else float(df["open"].iloc[-1])

    # 주봉/월봉/연초 시가 — 현재 기간 첫 거래일 시가
    iso = df.index.isocalendar()
    week_mask = (iso["year"].values == last_ts.isocalendar()[0]) & (
        iso["week"].values == last_ts.isocalendar()[1]
    )
    weekly_open = _first_open_in_period(df, pd.Series(week_mask, index=df.index))
    monthly_open = _first_open_in_period(
        df, (df.index.year == last_ts.year) & (df.index.month == last_ts.month)
    )
    year_open = _first_open_in_period(df, df.index.year == last_ts.year)

    # 반기 시가 — 1~6월=1월 첫거래일, 7~12월=7월 첫거래일
    is_first_half = last_ts.month <= 6
    half_mask = (df.index.year == last_ts.year) & (
        (df.index.month <= 6) if is_first_half else (df.index.month >= 7)
    )
    half_year_open = _first_open_in_period(df, half_mask)

    def _axis(anchor: float | None) -> dict:
        return {
            "open": int(anchor) if anchor else None,
            "state": _state(current_close, anchor),
            "distance_pct": _pct(current_close, anchor) if anchor else None,
        }

    daily = _axis(daily_open)
    weekly = _axis(weekly_open)
    monthly = _axis(monthly_open)
    half = _axis(half_year_open)
    year = _axis(year_open)

    labels: list[str] = []
    for name, ax in (
        ("DAILY", daily), ("WEEKLY", weekly), ("MONTHLY", monthly),
        ("HALF_YEAR", half), ("YEAR", year),
    ):
        if ax["state"]:
            labels.append(f"{name}_OPEN_{ax['state']}")

    return {
        "data_available": True,
        "current_close": int(current_close),
        "as_of_date": pd.Timestamp(last_ts).strftime("%Y-%m-%d"),
        "current_half": "H1" if is_first_half else "H2",
        "daily_open": daily["open"],
        "daily_open_state": daily["state"],
        "weekly_open": weekly["open"],
        "weekly_open_state": weekly["state"],
        "weekly_open_broken": weekly["state"] == "BELOW",
        "monthly_open": monthly["open"],
        "monthly_open_state": monthly["state"],
        "monthly_open_broken": monthly["state"] == "BELOW",
        "half_year_open": half["open"],
        "half_year_open_state": half["state"],
        "distance_from_half_year_open_pct": half["distance_pct"],
        "year_open": year["open"],
        "year_open_state": year["state"],
        "labels": labels,
    }


def compute_candle_turn(df: pd.DataFrame) -> dict:
    """음양(전일 음봉→금일 양봉=바닥전환 후보)/양음(전일 양봉→금일 음봉=고점경고). 순수 함수.

    매수/매도 신호가 아니라 관측 라벨이다.
    """
    if df is None or len(df) < 2 or "open" not in df.columns:
        return {"candle_turn_type": "NO_TURN", "label": "NO_TURN"}

    prev_open = float(df["open"].iloc[-2])
    prev_close = float(df["close"].iloc[-2])
    cur_open = float(df["open"].iloc[-1])
    cur_close = float(df["close"].iloc[-1])

    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open
    cur_bullish = cur_close > cur_open
    cur_bearish = cur_close < cur_open

    if prev_bearish and cur_bullish:
        turn_type, label = "EUM_YANG", "EUM_YANG_REVERSAL"
    elif prev_bullish and cur_bearish:
        turn_type, label = "YANG_EUM", "YANG_EUM_WARNING"
    else:
        turn_type, label = "NO_TURN", "NO_TURN"

    return {
        "candle_turn_type": turn_type,
        "label": label,
        "prev_open": int(prev_open),
        "prev_close": int(prev_close),
        "open": int(cur_open),
        "close": int(cur_close),
    }


def compute_annual_overheat(df: pd.DataFrame) -> dict:
    """1년 수익률 + 과열 등급. 순수 함수.

    1년 전 종가 = 마지막 날짜 - 365일에 가장 가까운(이전) 거래일 종가.
    데이터가 1년 미만이면 가장 오래된 종가를 기준으로 하되 부족 플래그를 남긴다.
    """
    if df is None or df.empty:
        return {"return_1y_pct": None, "annual_overheat_warning": False, "overheat_grade": None}

    last_ts = df.index[-1]
    current_close = float(df["close"].iloc[-1])
    one_year_ago = last_ts - pd.Timedelta(days=365)
    prior = df.loc[df.index <= one_year_ago]
    insufficient = prior.empty
    base = float((prior["close"].iloc[-1]) if not prior.empty else df["close"].iloc[0])

    ret = _pct(current_close, base)
    grade = None
    if ret is not None:
        if ret >= OVERHEAT_1000:
            grade = "OVERHEAT_1000"
        elif ret >= OVERHEAT_500:
            grade = "OVERHEAT_500"
        elif ret >= OVERHEAT_300:
            grade = "OVERHEAT_300"

    return {
        "return_1y_pct": ret,
        "annual_overheat_warning": bool(ret is not None and ret >= OVERHEAT_300),
        "overheat_grade": grade,
        "lookback_insufficient": insufficient,
    }


def build_ipo_reversion(
    listing_date: str | None,
    listing_open: float | None,
    current_close: float | None,
    quality_tag: str | None = None,
    liquidity_ok: bool = True,
    managed_risk: bool = False,
) -> dict:
    """상장 시초가 대비 되돌림 라벨. 순수 함수. 상장일 메타 없으면 빈 결과.

    바로 매수 후보가 아니라 SHOW ME "시초가 회복 여력" 표시용 관측 라벨이다.
    """
    if not listing_date or not listing_open or not current_close or listing_open <= 0:
        return {"data_available": False, "ipo_reversion_state": None}

    drawdown = _pct(current_close, listing_open)
    recovery_target = _pct(listing_open, current_close)  # 시초가 회복까지 필요 수익률
    low_quality = bool(
        managed_risk
        or not liquidity_ok
        or (quality_tag in IPO_LOW_QUALITY_TAGS if quality_tag else False)
    )

    if low_quality:
        state = "IPO_REVERSION_AVOID"
    elif drawdown is not None and drawdown <= IPO_CORE_DRAWDOWN:
        state = "IPO_REVERSION_CORE"
    elif drawdown is not None and drawdown <= IPO_WATCH_DRAWDOWN:
        state = "IPO_REVERSION_WATCH"
    else:
        state = "IPO_NOT_REVERSION"

    return {
        "data_available": True,
        "listing_date": listing_date,
        "listing_open": int(listing_open),
        "current_close": int(current_close),
        "drawdown_from_listing_open_pct": drawdown,
        "recovery_target_pct": recovery_target,
        "quality_tag": quality_tag,
        "ipo_reversion_state": state,
    }


def build_price_axis_labels(df: pd.DataFrame, ipo_meta: dict | None = None) -> dict:
    """단일 종목 OHLCV → 가격축/캔들전환/연간과열/IPO 라벨 번들. 순수 함수.

    df는 정규화된 OHLCV. ipo_meta(선택): {listing_date, listing_open, quality_tag,
    liquidity_ok, managed_risk}. 없으면 ipo_reversion은 빈 결과.
    """
    if df is None or df.empty or "close" not in df.columns:
        return {
            "version": PRICE_AXIS_VERSION,
            "data_available": False,
            "price_axis": {"data_available": False, "labels": []},
            "candle_turn": {"candle_turn_type": "NO_TURN", "label": "NO_TURN"},
            "annual_overheat": {"return_1y_pct": None, "annual_overheat_warning": False, "overheat_grade": None},
            "ipo_reversion": {"data_available": False, "ipo_reversion_state": None},
        }

    current_close = float(df["close"].iloc[-1])
    meta = ipo_meta or {}
    ipo = build_ipo_reversion(
        meta.get("listing_date"),
        meta.get("listing_open"),
        current_close,
        quality_tag=meta.get("quality_tag"),
        liquidity_ok=meta.get("liquidity_ok", True),
        managed_risk=meta.get("managed_risk", False),
    )
    return {
        "version": PRICE_AXIS_VERSION,
        "data_available": True,
        "price_axis": compute_price_axis(df),
        "candle_turn": compute_candle_turn(df),
        "annual_overheat": compute_annual_overheat(df),
        "ipo_reversion": ipo,
    }


def load_price_axis_for_ticker(
    ticker: str, days: int = 400, prefer_remote: bool = True, ipo_meta: dict | None = None
) -> dict:
    """편의 로더(유일한 IO 함수): OHLCV 로드 → 라벨 번들. 실패 시 data_available=False."""
    try:
        df = normalize_ohlcv(load_daily_ohlcv(ticker, days=days, prefer_remote=prefer_remote))
    except Exception:
        df = None
    return build_price_axis_labels(df, ipo_meta=ipo_meta)
