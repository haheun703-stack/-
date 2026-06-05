"""PAPER 모의 원장 추적/스키마 보강.

data/paper_ledger.json의 PAPER_OPEN 포지션을 processed parquet 기준으로 추적한다.

보강 필드:
  - candidate: 주봉 게이트, 일봉 눌림/지지, 과열, 손익비, 4주체 수급, 진입/회피 사유
  - entry: T0 종가 진입 + T+1 시가 진입 병행 기록
  - tracking: 일별 close/high/low, MFE/MAE, MA10/MA20, exit_check
  - exit: 지지이탈/MA10/MA20/D+10/손절 발생 시 paper 청산 기록

실주문 0 / KIS 미접촉 / scheduler·SAJANG 무변경.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

LEDGER = PROJECT_ROOT / "data" / "paper_ledger.json"
PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data"
PULLBACK_SCAN = DATA_DIR / "pullback_scan.json"
SCHEMA_VERSION = "quant_paper_ledger_v2"


def _date(value: Any) -> str:
    if value in (None, ""):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 2) -> float:
    return round(_float(value), digits)


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator - 1) * 100, 2)


def load_price_df(ticker: str) -> pd.DataFrame | None:
    """processed/{ticker}.parquet 로드. OHLCV 0 행은 제거한다."""
    f = PROCESSED / f"{ticker}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return None
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    if df.empty:
        return None
    return df


def latest_close(ticker: str) -> tuple[float | None, str | None]:
    df = load_price_df(ticker)
    if df is None or df.empty:
        return None, None
    return float(df["close"].iloc[-1]), _date(df.index[-1])


def _row_on_or_before(df: pd.DataFrame, date_str: str) -> tuple[pd.Timestamp, pd.Series] | None:
    hist = df.loc[df.index <= pd.Timestamp(date_str)]
    if hist.empty:
        return None
    return hist.index[-1], hist.iloc[-1]


def _row_after(df: pd.DataFrame, date_str: str) -> tuple[pd.Timestamp, pd.Series] | None:
    future = df.loc[df.index > pd.Timestamp(date_str)]
    if future.empty:
        return None
    return future.index[0], future.iloc[0]


def _ma(row: pd.Series, hist: pd.DataFrame, col: str, window: int) -> float:
    if col in row and _float(row.get(col)) > 0:
        return _float(row.get(col))
    if len(hist) < window:
        return 0.0
    return _float(hist["close"].tail(window).mean())


def build_weekly_gate(df: pd.DataFrame, asof_date: str) -> dict:
    """주봉 20/60선 기반 LONG_ALLOWED/WATCH/AVOID 관찰 게이트."""
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    if hist.empty:
        return {"data_available": False, "gate": "WATCH", "weekly_LONG_ALLOWED": False}

    weekly = hist["close"].resample("W-FRI").last().dropna()
    if len(weekly) < 20:
        return {
            "data_available": False,
            "gate": "WATCH",
            "weekly_LONG_ALLOWED": False,
            "reason": "weekly_data_insufficient",
            "weeks": int(len(weekly)),
        }

    ma20 = weekly.rolling(20).mean()
    ma60 = weekly.rolling(60).mean()
    close = _float(weekly.iloc[-1])
    ma20_now = _float(ma20.iloc[-1])
    ma60_now = _float(ma60.iloc[-1]) if len(weekly) >= 60 else 0.0
    ma20_prev = _float(ma20.iloc[-4]) if len(weekly) >= 24 else ma20_now
    ma60_prev = _float(ma60.iloc[-4]) if len(weekly) >= 64 else ma60_now
    slope20 = _pct(ma20_now, ma20_prev) if ma20_prev else 0.0
    slope60 = _pct(ma60_now, ma60_prev) if ma60_prev else 0.0
    vs20 = _pct(close, ma20_now) if ma20_now else 0.0
    vs60 = _pct(close, ma60_now) if ma60_now else 0.0

    if close >= ma20_now and slope20 >= -1.0 and (not ma60_now or close >= ma60_now * 0.95 or slope60 >= 0):
        gate = "LONG_ALLOWED"
    elif close >= ma20_now * 0.95 or slope20 >= 0:
        gate = "WATCH"
    else:
        gate = "AVOID"

    return {
        "data_available": True,
        "gate": gate,
        "weekly_LONG_ALLOWED": gate == "LONG_ALLOWED",
        "weekly_close": int(close),
        "weekly_close_vs_ma20_pct": round(vs20, 2),
        "weekly_close_vs_ma60_pct": round(vs60, 2),
        "weekly_ma20": round(ma20_now, 2),
        "weekly_ma60": round(ma60_now, 2) if ma60_now else None,
        "weekly_ma20_slope_pct_4w": round(slope20, 2),
        "weekly_ma60_slope_pct_4w": round(slope60, 2) if ma60_now else None,
    }


def build_daily_setup(df: pd.DataFrame, asof_date: str) -> dict:
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    if hist.empty:
        return {"data_available": False, "daily_pullback_support": False}

    row = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else row
    close = _float(row["close"])
    open_ = _float(row["open"])
    low = _float(row["low"])
    high = _float(row["high"])
    ma5 = _ma(row, hist, "sma_5", 5)
    ma10 = _float(hist["close"].tail(10).mean()) if len(hist) >= 10 else 0.0
    ma20 = _ma(row, hist, "sma_20", 20)
    ma60 = _ma(row, hist, "sma_60", 60)
    rsi = _float(row.get("rsi_14"))
    adx = _float(row.get("adx_14"))
    distance_to_ma20 = _pct(close, ma20) if ma20 else 0.0
    support_test = bool(ma20 and low <= ma20 * 1.03 and close >= ma20 * 0.97)
    bounce_confirmed = bool(support_test and close >= open_ and close >= _float(prev["close"]) * 0.995)
    overheated = bool(distance_to_ma20 >= 12.0 or rsi >= 75.0)

    if bounce_confirmed:
        setup = "BOUNCE_CONFIRMED"
    elif support_test:
        setup = "SUPPORT_TEST"
    elif ma20 and close < ma20 * 1.08:
        setup = "PULLBACK"
    else:
        setup = "NONE"

    return {
        "data_available": True,
        "daily_setup": setup,
        "daily_pullback_support": setup in ("SUPPORT_TEST", "BOUNCE_CONFIRMED"),
        "support_price": int(ma20) if ma20 else int(low),
        "ma5": round(ma5, 2) if ma5 else None,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "adx": round(adx, 2) if adx else None,
        "rsi": round(rsi, 2) if rsi else None,
        "distance_to_ma20_pct": round(distance_to_ma20, 2),
        "overheated": overheated,
        "candle": {
            "open": int(open_),
            "high": int(high),
            "low": int(low),
            "close": int(close),
            "is_bullish": close >= open_,
        },
    }


def build_supply_4_actor(df: pd.DataFrame, asof_date: str, lookback: int = 5) -> dict:
    hist = df.loc[df.index <= pd.Timestamp(asof_date)].tail(lookback)
    cols = {
        "foreign": "외국인합계",
        "institution": "기관합계",
        "individual": "개인",
        "other_corp": "기타법인",
    }
    if hist.empty or not any(col in hist.columns for col in cols.values()):
        return {
            "data_available": False,
            "lookback_days": lookback,
            "score": 0,
            "hard_gate_used": False,
            "reason": "4_actor_flow_missing",
        }

    sums = {key: int(_float(hist[col].sum())) if col in hist.columns else 0 for key, col in cols.items()}
    score = 0
    score += 2 if sums["foreign"] > 0 else -2 if sums["foreign"] < 0 else 0
    score += 2 if sums["institution"] > 0 else -2 if sums["institution"] < 0 else 0
    score += 1 if sums["other_corp"] > 0 else -1 if sums["other_corp"] < 0 else 0
    if sums["individual"] > 0 and (sums["foreign"] < 0 or sums["institution"] < 0):
        score -= 1

    buyers = [k for k, v in sums.items() if v > 0]
    if sums["foreign"] > 0 and sums["institution"] > 0:
        alignment = "FOREIGN_INST"
    elif sums["institution"] > 0:
        alignment = "INST_ONLY"
    elif sums["foreign"] > 0:
        alignment = "FOREIGN_ONLY"
    elif sums["individual"] > 0:
        alignment = "RETAIL_ABSORB"
    else:
        alignment = "NONE"

    return {
        "data_available": True,
        "lookback_days": int(len(hist)),
        "score": int(score),
        "hard_gate_used": False,
        "alignment": alignment,
        "buyers": buyers,
        "net_shares": sums,
    }


def build_risk_reward(trade: dict, df: pd.DataFrame, asof_date: str) -> dict:
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    entry = _float(trade.get("entry_price"))
    stop = _float(trade.get("stop_loss_price"))
    if entry <= 0:
        return {"data_available": False, "rr": 0.0}
    if stop <= 0:
        daily = build_daily_setup(df, asof_date)
        support_stop = _float(daily.get("support_price")) * 0.97 if daily.get("support_price") else 0.0
        stop = support_stop if 0 < support_stop < entry else entry * 0.92
    elif stop >= entry:
        stop = entry * 0.92

    recent_high = _float(hist["high"].tail(60).max()) if not hist.empty else entry
    risk = max(entry - stop, 1.0)
    target = recent_high if recent_high > entry else entry + risk * 2.0
    reward = max(target - entry, 0.0)
    rr = reward / risk if risk > 0 else 0.0
    return {
        "data_available": True,
        "entry_price": int(entry),
        "stop_loss_price": int(stop),
        "target_reference_price": int(target),
        "risk_per_share": int(risk),
        "reward_per_share": int(reward),
        "rr": round(rr, 2),
    }


def build_floor_quality(df: pd.DataFrame, asof_date: str) -> dict:
    """바닥 품질 필터 (floor_quality_score) — 보조 feature, hard gate 미사용.

    공식: 밸류 저점 × 추세 둔화/전환 × 수급 회복 × 하락이유(구조악재 아님).
    ⚠️ "바닥 판정"이 아니라 "바닥일 가능성 조건"을 점수화한 필터다. 확정 표현 금지.
    진입 게이트는 차트 4조건(주봉LA·눌림지지·과열아님·RR)이며, 이 점수는 누적·사후비교용.
    """
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    if hist.empty or len(hist) < 60:
        return {"data_available": False, "floor_quality_score": 0,
                "label": "data_insufficient", "hard_gate_used": False}

    close = _float(hist.iloc[-1]["close"])
    score = 0
    comp: dict = {}

    # 1) 밸류 저점 (fund 데이터 있는 종목만 — 대형~중형 한정)
    value: dict = {"available": False}
    if "fund_PER" in hist.columns and "fund_PBR" in hist.columns:
        per = hist["fund_PER"][(hist["fund_PER"] > 0) & (hist["fund_PER"] < 300)]
        pbr = hist["fund_PBR"][(hist["fund_PBR"] > 0) & (hist["fund_PBR"] < 50)]
        if len(per) >= 250 and len(pbr) >= 250:
            per_pct = float((per < per.iloc[-1]).mean() * 100)
            pbr_pct = float((pbr < pbr.iloc[-1]).mean() * 100)
            state = ("저평가" if (per_pct <= 30 and pbr_pct <= 30)
                     else "중립이하" if (per_pct <= 50 or pbr_pct <= 50) else "고평가")
            value = {"available": True, "per": round(float(per.iloc[-1]), 1),
                     "per_pctile": round(per_pct), "pbr": round(float(pbr.iloc[-1]), 2),
                     "pbr_pctile": round(pbr_pct), "state": state}
            score += 2 if state == "저평가" else 1 if state == "중립이하" else 0
    comp["value"] = value

    # 2) 추세 둔화/전환
    ma60 = _float(hist["close"].tail(60).mean())
    ma120 = _float(hist["close"].tail(120).mean()) if len(hist) >= 120 else 0.0
    if ma120 and close > ma60:
        trend = "전환/상향"; score += 2
    elif ma120 and close < ma60 < ma120:
        trend = "우하향"; score -= 1
    else:
        trend = "횡보"; score += 1
    high252 = _float(hist["close"].tail(252).max())
    drawdown = _pct(close, high252) if high252 else 0.0
    comp["trend"] = {"state": trend, "drawdown_from_52w_high_pct": round(drawdown, 1),
                     "ma60": round(ma60), "ma120": round(ma120) if ma120 else None}

    # 3) 수급 회복 (4주체 재사용)
    supply = build_supply_4_actor(df, asof_date)
    supply_recover = supply.get("score", 0) > 0
    if supply_recover:
        score += 1
    comp["supply_recover"] = {"recovered": supply_recover,
                              "supply_score": supply.get("score", 0),
                              "alignment": supply.get("alignment")}

    # 4) 하락 이유 — 구조악재 근사 (저평가인데 우하향 = 가치함정 의심)
    structural_suspect = (trend == "우하향" and value.get("state") == "저평가")
    if structural_suspect:
        score -= 2
    comp["drop_reason"] = {
        "structural_suspect": structural_suspect,
        "note": "저평가인데 우하향=가치함정 의심" if structural_suspect else "구조악재 징후 약함",
        "_todo": "KOSPI/섹터 대비 상대성과로 시장조정 vs 개별악재 구분(후속)",
    }

    # 라벨 (확정 아님 — 등급만)
    if structural_suspect or trend == "우하향":
        label = "관찰(위험)"
    elif trend == "전환/상향" and (not value["available"] or value.get("state") in ("저평가", "중립이하")):
        label = "진짜바닥후보"
    elif trend == "횡보":
        label = "바닥다지기후보"
    else:
        label = "중립"

    return {
        "data_available": True,
        "floor_quality_score": int(score),
        "label": label,
        "hard_gate_used": False,
        "components": comp,
        "note": "확정 아님 — 바닥 가능성 조건 필터(보조 feature). 진입게이트는 차트 4조건.",
    }


KOSPI_INDEX_CSV = DATA_DIR / "kospi_index.csv"
_MARKET_CACHE: dict = {}


def load_market_index() -> pd.Series:
    """KOSPI 지수 종가 시계열 (캐시). KOSDAQ 지수 미보유 → KOSPI 단일 기준."""
    if "kospi" in _MARKET_CACHE:
        return _MARKET_CACHE["kospi"]
    s = pd.Series(dtype=float)
    if KOSPI_INDEX_CSV.exists():
        try:
            k = pd.read_csv(KOSPI_INDEX_CSV)
            dc = "Date" if "Date" in k.columns else k.columns[0]
            k[dc] = pd.to_datetime(k[dc])
            k = k.sort_values(dc)
            s = pd.Series(pd.to_numeric(k["close"], errors="coerce").values, index=k[dc]).dropna()
        except Exception:  # noqa
            pass
    _MARKET_CACHE["kospi"] = s
    return s


def build_market_context(df: pd.DataFrame, asof_date: str) -> dict:
    """market_beta_20d, relative_return_5d/20d, drop_context (KOSPI 기준 근사).

    4단 결합 중 '시장 대비 버팀' 축. hard gate 미사용, feature-only.
    """
    out = {"data_available": False, "market_beta_20d": None,
           "relative_return_5d": None, "relative_return_20d": None,
           "drop_context": "unknown", "hard_gate_used": False}
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    kospi = load_market_index()
    if hist.empty or kospi.empty or len(hist) < 22:
        return out
    asof = pd.Timestamp(asof_date)
    k = kospi.loc[kospi.index <= asof]
    if len(k) < 22:
        return out

    sc = hist["close"].astype(float)
    s_ret = sc.pct_change()
    k_ret = k.pct_change()
    common = s_ret.index.intersection(k_ret.index)
    if len(common) >= 21:
        c = common[-20:]
        sr = s_ret.loc[c].values
        kr = k_ret.loc[c].values
    else:
        srx = s_ret.dropna().tail(20)
        krx = k_ret.dropna().tail(20)
        n = min(len(srx), len(krx))
        if n < 10:
            return out
        sr = srx.tail(n).values
        kr = krx.tail(n).values

    import numpy as np
    mask = np.isfinite(sr) & np.isfinite(kr)   # nan + inf(가격0 pct_change) 동시 제거
    sr, kr = sr[mask], kr[mask]
    var_m = float(np.var(kr)) if len(kr) >= 10 else 0.0
    beta = float(np.cov(sr, kr)[0, 1] / var_m) if var_m > 0 and len(sr) >= 10 else None

    def _ret(series: pd.Series, n: int):
        ss = series.dropna()
        return float((ss.iloc[-1] / ss.iloc[-1 - n] - 1) * 100) if len(ss) > n else None

    s5, s20 = _ret(sc, 5), _ret(sc, 20)
    m5, m20 = _ret(k, 5), _ret(k, 20)
    rr5 = (s5 - m5) if (s5 is not None and m5 is not None) else None
    rr20 = (s20 - m20) if (s20 is not None and m20 is not None) else None

    # drop_context: 당일(1일) 시장 급락 우선 감지 → 5일 약세 보조
    m1, s1 = _ret(k, 1), _ret(sc, 1)
    rel1 = (s1 - m1) if (s1 is not None and m1 is not None) else None
    ctx = "unknown"
    if m1 is not None and m1 <= -1.5 and rel1 is not None:    # 당일 시장 급락
        if rel1 >= 1.5:
            ctx = "resilient_pullback"   # 시장 폭락에도 버팀
        elif rel1 <= -3.0:
            ctx = "stock_specific_drop"  # 시장보다 훨씬 더 빠짐
        else:
            ctx = "market_selloff"       # 시장만큼 같이 빠짐
    elif m5 is not None and m5 <= -3.0 and rr5 is not None:   # 5일 시장 약세
        if rr5 >= 3.0:
            ctx = "resilient_pullback"
        elif rr5 <= -5.0:
            ctx = "stock_specific_drop"
        else:
            ctx = "market_selloff"
    elif s5 is not None and s5 <= -5.0 and (rr5 is None or rr5 <= -3.0):
        ctx = "stock_specific_drop"      # 시장 멀쩡한데 종목만 폭락
    else:
        ctx = "normal"

    return {"data_available": True,
            "market_beta_20d": round(beta, 2) if beta is not None else None,
            "relative_return_5d": round(rr5, 2) if rr5 is not None else None,
            "relative_return_20d": round(rr20, 2) if rr20 is not None else None,
            "stock_return_5d": round(s5, 2) if s5 is not None else None,
            "market_return_5d": round(m5, 2) if m5 is not None else None,
            "stock_return_1d": round(s1, 2) if s1 is not None else None,
            "market_return_1d": round(m1, 2) if m1 is not None else None,
            "drop_context": ctx,
            "hard_gate_used": False,
            "_note": "KOSDAQ 지수 미보유 → KOSPI 단일 근사. sector_selloff는 후속(섹터지수 필요)."}


def build_supply_confirmation(df: pd.DataFrame, asof_date: str) -> dict:
    """supply_confirmation_score, supply_state — 외국인/기관 3·5일 누적·전환.

    4단 결합 중 '수급 이탈 멈춤' 축. hard gate 미사용, feature-only.
    """
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    cols = {"foreign": "외국인합계", "institution": "기관합계"}
    if hist.empty or not all(c in hist.columns for c in cols.values()):
        return {"data_available": False, "supply_state": "unknown",
                "supply_confirmation_score": 0, "hard_gate_used": False}

    last = hist.iloc[-1]
    f5 = _float(last.get("foreign_net_5d")) if "foreign_net_5d" in hist.columns else _float(hist["외국인합계"].tail(5).sum())
    i5 = _float(last.get("inst_net_5d")) if "inst_net_5d" in hist.columns else _float(hist["기관합계"].tail(5).sum())
    f3 = _float(hist["외국인합계"].tail(3).sum())
    i3 = _float(hist["기관합계"].tail(3).sum())
    fcb = _float(last.get("foreign_consecutive_buy", 0))
    icb = _float(last.get("inst_consecutive_buy", 0))

    if f5 > 0 and i5 > 0:
        state = "dual_buying"
    elif f5 > 0:
        state = "foreign_accumulation"
    elif i5 > 0:
        state = "institution_accumulation"
    elif f5 < 0 and i5 < 0:
        state = "distribution_warning"
    else:
        state = "supply_neutral"

    score = 0
    score += 2 if f5 > 0 else (-2 if f5 < 0 else 0)
    score += 2 if i5 > 0 else (-2 if i5 < 0 else 0)
    score += 1 if (f3 > 0 and f5 > 0) else 0   # 3일도 순매수 = 회복 가속
    score += 1 if (i3 > 0 and i5 > 0) else 0
    score += 1 if fcb >= 2 else 0
    score += 1 if icb >= 2 else 0

    return {"data_available": True, "supply_state": state,
            "supply_confirmation_score": int(score),
            "foreign_net_5d": int(f5), "inst_net_5d": int(i5),
            "foreign_net_3d": int(f3), "inst_net_3d": int(i3),
            "foreign_consecutive_buy": int(fcb), "inst_consecutive_buy": int(icb),
            "hard_gate_used": False}


def build_candidate(trade: dict, df: pd.DataFrame) -> dict:
    asof_date = trade.get("entry_date") or _date(df.index[-1])
    weekly = build_weekly_gate(df, asof_date)
    daily = build_daily_setup(df, asof_date)
    supply = build_supply_4_actor(df, asof_date)
    rr = build_risk_reward(trade, df, asof_date)
    floor = build_floor_quality(df, asof_date)
    market = build_market_context(df, asof_date)
    supply_conf = build_supply_confirmation(df, asof_date)

    blockers: list[str] = []
    if weekly.get("gate") != "LONG_ALLOWED":
        blockers.append(f"weekly_gate={weekly.get('gate')}")
    if not daily.get("daily_pullback_support"):
        blockers.append(f"daily_setup={daily.get('daily_setup')}")
    if daily.get("overheated"):
        blockers.append("overheated")
    if rr.get("rr", 0) < 1.5:
        blockers.append(f"rr<{1.5}")

    decision = "진입" if not blockers else "회피"
    decision_shadow = "ENTER_CANDIDATE" if not blockers else "AVOID"
    reason = "조건 충족: LONG_ALLOWED + 눌림/지지 + 과열아님 + RR>=1.5" if not blockers else "; ".join(blockers)

    return {
        "ticker": trade.get("ticker", ""),
        "name": trade.get("name", ""),
        "date": asof_date,
        "weekly_LONG_ALLOWED": bool(weekly.get("weekly_LONG_ALLOWED", False)),
        "weekly_gate": weekly,
        "daily_pullback_support": bool(daily.get("daily_pullback_support", False)),
        "daily_setup": daily,
        "overheated": bool(daily.get("overheated", False)),
        "risk_reward": rr,
        "supply_4주체점수": supply.get("score", 0),
        "stock_flow_4_actor": supply,
        "decision": decision,
        "decision_shadow": decision_shadow,
        "reason": reason,
        "hard_gate_notes": "수급은 feature/log only. hard gate 미사용.",
        "floor_quality": floor,
        "market_context": market,
        "supply_confirmation": supply_conf,
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _name_map() -> dict[str, str]:
    result: dict[str, str] = {}

    name_path = DATA_DIR / "universe" / "name_map.json"
    data = _read_json(name_path)
    for ticker, name in data.items():
        result[str(ticker).zfill(6)] = str(name)

    sector_path = DATA_DIR / "universe" / "sector_map.csv"
    if sector_path.exists():
        try:
            with open(sector_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    ticker = str(row.get("ticker", "")).zfill(6)
                    name = str(row.get("name", "")).strip()
                    if ticker and name:
                        result.setdefault(ticker, name)
        except Exception:
            pass

    return result


def _source_score(item: dict) -> float:
    for key in ("total_score", "score"):
        if key in item:
            return _round(item.get(key), 2)
    return 0.0


def _source_reasons(item: dict) -> list[str]:
    reasons = item.get("reasons")
    if isinstance(reasons, list):
        return [str(x) for x in reasons[:8]]
    return []


def _manual_candidate_items(tickers: list[str]) -> list[dict]:
    names = _name_map()
    items: list[dict] = []
    for idx, ticker in enumerate(tickers, start=1):
        normalized = str(ticker).strip().zfill(6)
        if not normalized:
            continue
        items.append({
            "ticker": normalized,
            "name": names.get(normalized, normalized),
            "source_rank": idx,
            "source": "manual",
        })
    return items


def _load_pullback_items(limit: int) -> list[dict]:
    data = _read_json(PULLBACK_SCAN)
    rows = data.get("candidates") or data.get("all_uptrend") or []
    items: list[dict] = []
    for idx, row in enumerate(rows[:limit], start=1):
        if not row.get("ticker"):
            continue
        item = dict(row)
        item["source_rank"] = idx
        item["source"] = "pullback_scan"
        items.append(item)
    return items


def _load_tomorrow_pick_items(limit: int) -> list[dict]:
    paths = [
        DATA_DIR / "tomorrow_picks.json",
        DATA_DIR / "tomorrow_picks_flowx.json",
    ]
    data = {}
    for path in paths:
        data = _read_json(path)
        if data:
            break
    rows = data.get("picks") or []
    items: list[dict] = []
    for idx, row in enumerate(rows[:limit], start=1):
        if not row.get("ticker"):
            continue
        item = dict(row)
        item["source_rank"] = idx
        item["source"] = "tomorrow_picks"
        items.append(item)
    return items


def load_candidate_items(source: str, tickers: list[str], limit: int) -> list[dict]:
    if source == "manual":
        return _manual_candidate_items(tickers)
    if source == "pullback":
        return _load_pullback_items(limit)
    if source == "tomorrow_picks":
        return _load_tomorrow_pick_items(limit)
    raise ValueError(f"unknown candidate source: {source}")


def evaluate_candidate_item(item: dict, asof_date: str = "") -> tuple[dict, list[str]]:
    issues: list[str] = []
    ticker = str(item.get("ticker", "")).strip().zfill(6)
    name = str(item.get("name") or ticker)
    df = load_price_df(ticker)
    if df is None:
        return {
            "ticker": ticker,
            "name": name,
            "date": asof_date,
            "decision": "회피",
            "decision_shadow": "AVOID",
            "reason": "price_data_missing",
            "source": item.get("source", ""),
            "source_rank": item.get("source_rank"),
            "real_order": False,
        }, [f"{ticker}: candidate price_data_missing"]

    eval_date = asof_date or _date(df.index[-1])
    row_pair = _row_on_or_before(df, eval_date)
    if row_pair is None:
        return {
            "ticker": ticker,
            "name": name,
            "date": eval_date,
            "decision": "회피",
            "decision_shadow": "AVOID",
            "reason": "asof_price_missing",
            "source": item.get("source", ""),
            "source_rank": item.get("source_rank"),
            "real_order": False,
        }, [f"{ticker}: candidate asof_price_missing"]

    row_date, row = row_pair
    entry_price = _float(item.get("entry_price")) or _float(item.get("close")) or _float(row["close"])
    stop_price = _float(item.get("stop_loss")) or _float(item.get("stop_loss_price"))
    trade = {
        "ticker": ticker,
        "name": name,
        "entry_date": _date(row_date),
        "entry_price": entry_price,
        "stop_loss_price": stop_price,
        "qty": 0,
    }
    candidate = build_candidate(trade, df)
    candidate.update({
        "source": item.get("source", ""),
        "source_rank": item.get("source_rank"),
        "source_grade": item.get("grade", ""),
        "source_score": _source_score(item),
        "source_strategy": item.get("strategy", ""),
        "source_reasons": _source_reasons(item),
        "real_order": False,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })
    return candidate, issues


def build_candidate_log_entry(
    source: str,
    source_label: str,
    source_note: str,
    candidates: list[dict],
    asof_date: str,
) -> dict:
    enter_count = sum(1 for c in candidates if c.get("decision") == "진입")
    avoid_count = sum(1 for c in candidates if c.get("decision") == "회피")
    label = source_label or source
    date_key = (asof_date or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
    return {
        "id": f"CAND-{date_key}-{label}",
        "schema_version": SCHEMA_VERSION,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": asof_date,
        "source": source,
        "source_label": label,
        "source_note": source_note,
        "total": len(candidates),
        "enter_count": enter_count,
        "avoid_count": avoid_count,
        "real_order": False,
        "candidates": candidates,
    }


def upsert_candidate_log(data: dict, entry: dict) -> None:
    logs = data.get("candidate_log")
    if not isinstance(logs, list):
        logs = []
    logs = [log for log in logs if log.get("id") != entry.get("id")]
    logs.append(entry)
    logs.sort(key=lambda x: (x.get("as_of_date", ""), x.get("source_label", "")))
    data["candidate_log"] = logs
    data["_candidate_log_last_at"] = entry.get("evaluated_at")


def validate_candidate_log(data: dict) -> list[str]:
    issues: list[str] = []
    logs = data.get("candidate_log") or []
    for idx, log in enumerate(logs):
        for key in ("as_of_date", "source", "total", "enter_count", "avoid_count", "candidates"):
            if key not in log:
                issues.append(f"candidate_log[{idx}].{key}")
        for c_idx, candidate in enumerate(log.get("candidates", [])):
            for key in ("ticker", "decision", "reason", "weekly_LONG_ALLOWED", "daily_pullback_support", "overheated"):
                if key not in candidate:
                    issues.append(f"candidate_log[{idx}].candidates[{c_idx}].{key}")
    return issues


def print_candidate_log(entry: dict) -> None:
    print(
        f"[CANDIDATE LOG] {entry.get('source_label')} | 기준일 {entry.get('as_of_date')} "
        f"| 진입 {entry.get('enter_count')} / 회피 {entry.get('avoid_count')}"
    )
    for c in entry.get("candidates", []):
        rr = c.get("risk_reward", {}).get("rr", 0)
        weekly = c.get("weekly_gate", {}).get("gate", "?")
        daily = c.get("daily_setup", {}).get("daily_setup", "?")
        supply = c.get("supply_4주체점수", 0)
        print(
            f"  {c.get('source_rank', '-')}. {c.get('name', c.get('ticker'))}({c.get('ticker')}) "
            f"{c.get('decision')} | weekly={weekly} daily={daily} "
            f"RR={rr} 4주체={supply} | {c.get('reason')}"
        )
    print()


def build_entry(trade: dict, df: pd.DataFrame) -> dict:
    entry_date = trade.get("entry_date") or _date(df.index[-1])
    t0 = _row_on_or_before(df, entry_date)
    t1 = _row_after(df, entry_date)
    t0_date, t0_row = t0 if t0 else (None, None)
    t1_date, t1_row = t1 if t1 else (None, None)
    entry_price = _float(trade.get("entry_price")) or (_float(t0_row["close"]) if t0_row is not None else 0.0)
    stop_price = _float(trade.get("stop_loss_price")) or entry_price * 0.92
    qty = int(_float(trade.get("qty")))
    return {
        "variant_recorded": ["T0_CLOSE", "T1_OPEN"],
        "selected_variant": "T0_CLOSE",
        "entry_date": entry_date,
        "t0_date": _date(t0_date) if t0_date is not None else entry_date,
        "t0_close_price": int(_float(t0_row["close"])) if t0_row is not None else int(entry_price),
        "t1_date": _date(t1_date) if t1_date is not None else None,
        "t1_open_price": int(_float(t1_row["open"])) if t1_row is not None else None,
        "actual_entry_price": int(entry_price),
        "qty": qty,
        "capital_won": int(entry_price * qty),
        "stop_loss": {
            "price": int(stop_price),
            "pct": _round(trade.get("stop_loss_pct", -8.0), 2),
        },
    }


def build_tracking_record(trade: dict, df: pd.DataFrame) -> dict:
    entry_date = trade.get("entry_date") or _date(df.index[-1])
    entry_price = _float(trade.get("entry_price"))
    qty = int(_float(trade.get("qty")))
    period = df.loc[df.index >= pd.Timestamp(entry_date)]
    if period.empty:
        period = df.tail(1)
    row = period.iloc[-1]
    hist = df.loc[df.index <= period.index[-1]]
    close = _float(row["close"])
    high = _float(row["high"])
    low = _float(row["low"])
    mfe_price = _float(period["high"].max())
    mae_price = _float(period["low"].min())
    ma10 = _float(hist["close"].tail(10).mean()) if len(hist) >= 10 else 0.0
    ma20 = _ma(row, hist, "sma_20", 20)
    support_price = ma20 if ma20 else _float(period["low"].tail(5).min())
    pnl_pct = _pct(close, entry_price) if entry_price else 0.0
    pnl_won = int((close - entry_price) * qty) if entry_price else 0
    mfe_pct = _pct(mfe_price, entry_price) if entry_price else 0.0
    mae_pct = _pct(mae_price, entry_price) if entry_price else 0.0
    days_held = max(0, len(period) - 1)
    exit_check = check_exit(trade, close, low, ma10, ma20, support_price, days_held)
    return {
        "date": _date(period.index[-1]),
        "close": int(close),
        "high": int(high),
        "low": int(low),
        "qty": qty,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_won": pnl_won,
        "MFE_pct": round(mfe_pct, 2),
        "MAE_pct": round(mae_pct, 2),
        "MFE_price": int(mfe_price),
        "MAE_price": int(mae_price),
        "days_held": days_held,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "support_price": int(support_price) if support_price else None,
        "exit_check": exit_check,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def check_exit(
    trade: dict,
    close: float,
    low: float,
    ma10: float,
    ma20: float,
    support_price: float,
    days_held: int,
) -> dict:
    stop = _float(trade.get("stop_loss_price"))
    if stop and low <= stop:
        return {"should_exit": True, "reason": "STOP_LOSS", "detail": "손절선 터치"}
    if support_price and close < support_price * 0.97:
        return {"should_exit": True, "reason": "SUPPORT_BREAK", "detail": "지지선 3% 이상 이탈"}
    if ma20 and close < ma20:
        return {"should_exit": True, "reason": "MA20_BREAK", "detail": "MA20 이탈"}
    if ma10 and close < ma10:
        return {"should_exit": True, "reason": "MA10_BREAK", "detail": "MA10 이탈"}
    if days_held >= 10:
        return {"should_exit": True, "reason": "D+10_REVIEW", "detail": "D+10 점검 기준"}
    return {"should_exit": False, "reason": "HOLD", "detail": "보유 관찰"}


def apply_exit_if_needed(trade: dict, record: dict) -> None:
    if trade.get("status") != "PAPER_OPEN":
        return
    exit_check = record.get("exit_check", {})
    if not exit_check.get("should_exit"):
        return
    entry_price = _float(trade.get("entry_price"))
    exit_price = _float(record.get("close"))
    qty = int(_float(trade.get("qty")))
    pnl_won = int((exit_price - entry_price) * qty)
    pnl_pct = _pct(exit_price, entry_price) if entry_price else 0.0
    trade["status"] = "PAPER_CLOSED"
    trade["exit"] = {
        "date": record.get("date"),
        "reason": exit_check.get("reason"),
        "detail": exit_check.get("detail"),
        "price": int(exit_price),
        "qty": qty,
        "pnl_won": pnl_won,
        "pnl_pct": round(pnl_pct, 2),
        "real_order": False,
    }


def upsert_tracking(trade: dict, record: dict) -> None:
    tracking = trade.get("tracking")
    if not isinstance(tracking, list):
        tracking = []
    tracking = [r for r in tracking if r.get("date") != record.get("date")]
    tracking.append(record)
    tracking.sort(key=lambda x: x.get("date", ""))
    trade["tracking"] = tracking


def enrich_trade(trade: dict) -> tuple[dict, list[str]]:
    issues: list[str] = []
    ticker = trade.get("ticker", "")
    df = load_price_df(ticker)
    if df is None:
        issues.append(f"{ticker}: processed parquet 없음")
        trade.setdefault("candidate", {"ticker": ticker, "decision": "회피", "reason": "price_data_missing"})
        return trade, issues

    trade["schema_version"] = SCHEMA_VERSION
    trade["candidate"] = build_candidate(trade, df)
    trade["entry"] = build_entry(trade, df)

    if trade.get("status") == "PAPER_OPEN":
        record = build_tracking_record(trade, df)
        upsert_tracking(trade, record)
        apply_exit_if_needed(trade, record)
    elif trade.get("tracking"):
        # 닫힌 거래도 최신 MFE/MAE 필드가 없는 과거 tracking은 그대로 두되 스키마만 표시한다.
        pass

    return trade, issues


def validate_trade(trade: dict) -> list[str]:
    missing: list[str] = []
    candidate = trade.get("candidate") or {}
    entry = trade.get("entry") or {}
    tracking = trade.get("tracking") or []
    for key in ("ticker", "weekly_LONG_ALLOWED", "daily_pullback_support", "overheated", "risk_reward", "decision", "reason"):
        if key not in candidate:
            missing.append(f"candidate.{key}")
    for key in ("t0_close_price", "t1_open_price", "qty", "stop_loss"):
        if key not in entry:
            missing.append(f"entry.{key}")
    if not tracking:
        missing.append("tracking[]")
    else:
        last = tracking[-1]
        for key in ("close", "MFE_pct", "MAE_pct", "exit_check"):
            if key not in last:
                missing.append(f"tracking[-1].{key}")
    if trade.get("status") == "PAPER_CLOSED" and "exit" not in trade:
        missing.append("exit")
    return missing


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {
            "_note": "PAPER 전용 모의 원장. 실주문 0 / KIS 미접촉.",
            "_created": datetime.now().strftime("%Y-%m-%d"),
            "paper_trades": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_trade(trade: dict) -> None:
    name = trade.get("name", trade.get("ticker", ""))
    ticker = trade.get("ticker", "")
    sector = trade.get("sector", "")
    candidate = trade.get("candidate", {})
    entry = trade.get("entry", {})
    tracking = trade.get("tracking", [])
    latest = tracking[-1] if tracking else {}
    status = trade.get("status", "")
    status_icon = "🔴 청산" if status == "PAPER_CLOSED" else "🟢 보유중" if status == "PAPER_OPEN" else "⚪ 기타"

    print(f"{name}({ticker}) [{sector}] | {status_icon}")
    print(
        f"  후보: {candidate.get('decision', '?')} "
        f"| weekly={candidate.get('weekly_gate', {}).get('gate', '?')} "
        f"| daily={candidate.get('daily_setup', {}).get('daily_setup', '?')} "
        f"| RR={candidate.get('risk_reward', {}).get('rr', 0)} "
        f"| 4주체={candidate.get('supply_4주체점수', 0)}"
    )
    print(f"  사유: {candidate.get('reason', '')}")
    print(
        f"  entry: T0 {entry.get('t0_close_price')}원 / "
        f"T+1 open {entry.get('t1_open_price')}원 / 실제 {entry.get('actual_entry_price')}원 x{entry.get('qty')}주"
    )
    if latest:
        print(
            f"  latest {latest.get('date')}: close {latest.get('close'):,}원 "
            f"| PnL {latest.get('pnl_pct'):+.2f}% ({latest.get('pnl_won'):+,}원) "
            f"| MFE {latest.get('MFE_pct'):+.2f}% / MAE {latest.get('MAE_pct'):+.2f}%"
        )
        ex = latest.get("exit_check", {})
        print(f"  exit_check: {ex.get('reason')} — {ex.get('detail')}")
    if trade.get("exit"):
        ex = trade["exit"]
        print(f"  EXIT: {ex.get('date')} {ex.get('reason')} @{ex.get('price'):,}원 PnL {ex.get('pnl_pct'):+.2f}%")
    print()


def run(
    path: Path,
    write: bool,
    check_only: bool,
    candidate_source: str = "",
    candidate_tickers: list[str] | None = None,
    candidate_limit: int = 30,
    candidate_label: str = "",
    candidate_note: str = "",
    candidate_asof: str = "",
) -> int:
    data = load_ledger(path)
    data["_schema_version"] = SCHEMA_VERSION
    data["_last_tracked_at"] = datetime.now().isoformat(timespec="seconds")
    data["_safety"] = {
        "real_order": False,
        "kis_touch": False,
        "scheduler_changed": False,
        "sajang_changed": False,
    }

    print("=== PAPER 모의 원장 추적 / v2 스키마 점검 (실주문 0) ===\n")
    issues: list[str] = []
    trades = data.get("paper_trades", [])
    for idx, trade in enumerate(trades):
        if not check_only:
            enriched, trade_issues = enrich_trade(trade)
            trades[idx] = enriched
            issues.extend(trade_issues)
            trade = enriched
        missing = validate_trade(trade)
        if missing:
            issues.append(f"{trade.get('ticker', '?')}: " + ", ".join(missing))
        print_trade(trade)

    if candidate_source and not check_only:
        candidate_items = load_candidate_items(candidate_source, candidate_tickers or [], candidate_limit)
        candidates: list[dict] = []
        for item in candidate_items:
            candidate, candidate_issues = evaluate_candidate_item(item, candidate_asof)
            candidates.append(candidate)
            issues.extend(candidate_issues)
        log_asof = candidate_asof or (candidates[0].get("date") if candidates else datetime.now().strftime("%Y-%m-%d"))
        entry = build_candidate_log_entry(
            source=candidate_source,
            source_label=candidate_label,
            source_note=candidate_note,
            candidates=candidates,
            asof_date=log_asof,
        )
        upsert_candidate_log(data, entry)
        print_candidate_log(entry)

    issues.extend(validate_candidate_log(data))

    data["paper_trades"] = trades
    if write and not check_only:
        save_ledger(path, data)
        print(f"[WRITE] ledger 저장 완료: {path}")
    else:
        print("[NO-WRITE] ledger 파일은 변경하지 않음")

    if issues:
        print("\n[LEDGER CHECK] FAIL")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("\n[LEDGER CHECK] PASS — candidate/entry/tracking/exit 스키마 충족")
    print("★ 실주문 0 / KIS 미접촉 / AUTO_TRADING_ENABLED=0 불변.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PAPER ledger v2 추적/점검")
    parser.add_argument("--ledger", default=str(LEDGER), help="ledger JSON 경로")
    parser.add_argument("--no-write", action="store_true", help="미리보기만 수행")
    parser.add_argument("--check-only", action="store_true", help="현재 ledger 스키마만 점검")
    parser.add_argument(
        "--candidate-source",
        choices=["manual", "pullback", "tomorrow_picks"],
        default="",
        help="후보 평가 로그 소스",
    )
    parser.add_argument("--candidate-tickers", default="", help="manual 후보 티커 CSV")
    parser.add_argument("--candidate-limit", type=int, default=30, help="파일 기반 후보 평가 상위 N개")
    parser.add_argument("--candidate-label", default="", help="candidate_log id/source_label")
    parser.add_argument("--candidate-note", default="", help="후보 로그 설명")
    parser.add_argument("--candidate-asof", default="", help="평가 기준일 YYYY-MM-DD")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.candidate_tickers.split(",") if t.strip()]
    return run(
        Path(args.ledger),
        write=not args.no_write,
        check_only=args.check_only,
        candidate_source=args.candidate_source,
        candidate_tickers=tickers,
        candidate_limit=args.candidate_limit,
        candidate_label=args.candidate_label,
        candidate_note=args.candidate_note,
        candidate_asof=args.candidate_asof,
    )


if __name__ == "__main__":
    raise SystemExit(main())
