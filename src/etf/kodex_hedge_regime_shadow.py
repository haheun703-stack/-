"""KODEX 레버리지 + 인버스 국면별 헤지 — shadow 관측 모듈 (실매매 0).

발견(적대검증, research 스크립트 2종): **국면이 헤지 룰을 결정한다.**
같은 신호가 강세장엔 최고·약세장엔 최악 →
  - BULL  : volatility_dynamic (변동성 확대 때만 단기 헤지, 레버 수익 살림)
  - BEAR  : fixed_thick_80 (고정 두꺼운 헤지, 데드캣 휩쏘 회피·MDD 방어)
  - UNKNOWN: minimal_or_no_hedge (애매하면 관망/소형, 고정50 같은 중간값 제외)

★shadow 전용: 실매매/주문/브로커/SAJANG/scheduler 무접촉. 추천·실전 신호 아님.
read-only 데이터 조회 + shadow ledger 기록만. 6/10~6/12 매일 관측 → 6/12 실전 판정.

regime classifier가 먼저고, regime 없이는 hedge 전략도 없다(사장님 지시).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER_DIR = PROJECT_ROOT / "data_store" / "kodex_hedge_shadow"
LEDGER_PATH = LEDGER_DIR / "ledger.json"

VERSION = "kodex_hedge_regime_shadow_v1"

TICKER_CORE = "122630"      # KODEX 레버리지(2x)
TICKER_INVERSE = "114800"   # KODEX 인버스(1x)
TICKER_INDEX = "069500"     # KODEX200(1x) — regime 판정 기준

REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_UNKNOWN = "UNKNOWN"

HEDGE_VOL_DYNAMIC = "volatility_dynamic"
HEDGE_FIXED_THICK = "fixed_thick_80"
HEDGE_MINIMAL = "minimal_or_no_hedge"

THICK_RATIO = 0.8  # 약세장 고정 헤지 비중


def _load_close(ticker: str, start: str, end: str) -> pd.Series:
    """pykrx ETF 종가(읽기 전용, KRX 로그 억제)."""
    from pykrx import stock as krx
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = krx.get_market_ohlcv(start, end, ticker)
    finally:
        logging.disable(logging.NOTSET)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} empty")
    df.index = pd.to_datetime(df.index)
    return df["종가"].astype(float).rename(ticker)


def _load_fx() -> pd.Series | None:
    try:
        import FinanceDataReader as fdr
        s = fdr.DataReader("USD/KRW")["Close"]
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return None


def classify_regime(index_close: pd.Series) -> tuple[str, str]:
    """KODEX200 추세로 국면 판정. (regime, reason). ★헤지 신호 아닌 큰 국면 판정."""
    if index_close is None or len(index_close) < 60:
        return REGIME_UNKNOWN, "데이터 부족(<60일)"
    close = float(index_close.iloc[-1])
    ma20 = float(index_close.rolling(20).mean().iloc[-1])
    ma60_series = index_close.rolling(60).mean()
    ma60 = float(ma60_series.iloc[-1])
    ma60_prev = float(ma60_series.iloc[-6]) if len(ma60_series) >= 6 else ma60
    slope = ma60 - ma60_prev
    if close > ma60 and ma20 > ma60 and slope > 0:
        return REGIME_BULL, f"정배열(close>{ma60:.0f}>ma60)+60일선 상승"
    if close < ma60 and ma20 < ma60 and slope < 0:
        return REGIME_BEAR, f"역배열(close<ma60)+60일선 하락"
    return REGIME_UNKNOWN, "혼조(추세 불명)"


def hedge_policy_for(regime: str) -> str:
    return {
        REGIME_BULL: HEDGE_VOL_DYNAMIC,
        REGIME_BEAR: HEDGE_FIXED_THICK,
        REGIME_UNKNOWN: HEDGE_MINIMAL,
    }.get(regime, HEDGE_MINIMAL)


def _signals(lev: pd.Series, fx: pd.Series | None) -> dict[str, bool]:
    """전일 기준 헤지 on/off 신호(look-ahead 회피)."""
    lev_r = lev.pct_change()
    vol20 = lev_r.rolling(20).std()
    vol60 = lev_r.rolling(60).std()
    vol_on = bool(len(vol60.dropna()) and vol20.iloc[-2] > vol60.iloc[-2]) if len(lev) >= 62 else False
    ma60 = lev.rolling(60).mean()
    c60_on = bool(lev.iloc[-2] < ma60.iloc[-2]) if len(lev) >= 61 else False
    fx_on = False
    if fx is not None:
        fxa = fx.reindex(lev.index).ffill()
        hi = fxa.rolling(20).max()
        if len(fxa) >= 21:
            fx_on = bool(fxa.iloc[-2] >= hi.iloc[-2])
    return {"vol_on": vol_on, "c60_on": c60_on, "fx_on": fx_on}


def _portfolios_1d(lev_r: float, inv_r: float, sig: dict[str, bool]) -> dict[str, float]:
    """당일 7종 shadow 포트폴리오 1일 수익률(%)."""
    def p(ratio: float) -> float:
        return round((lev_r + inv_r * ratio) * 100, 4)
    return {
        "leverage_core_only": p(0.0),
        "leverage_plus_fixed_30": p(0.3),
        "leverage_plus_fixed_50": p(0.5),
        "leverage_plus_fixed_80": p(0.8),
        "leverage_plus_vol_dynamic": p(THICK_RATIO if sig["vol_on"] else 0.0),
        "leverage_plus_fx_dynamic": p(THICK_RATIO if sig["fx_on"] else 0.0),
        "leverage_plus_c60_dynamic": p(THICK_RATIO if sig["c60_on"] else 0.0),
    }


def _adaptive_ratio(regime: str, sig: dict[str, bool]) -> float:
    """regime별 권장정책의 당일 적용 헤지 비중."""
    if regime == REGIME_BEAR:
        return THICK_RATIO
    if regime == REGIME_BULL:
        return THICK_RATIO if sig["vol_on"] else 0.0
    return 0.0  # UNKNOWN = minimal(무헤지 관망)


def build_record(as_of: str, lev: pd.Series, inv: pd.Series, idx: pd.Series, fx: pd.Series | None) -> dict[str, Any]:
    """당일 shadow 관측 record(16필드 + 7 portfolio). 순수 계산."""
    lev_r = float(lev.pct_change().iloc[-1])
    inv_r = float(inv.pct_change().iloc[-1])
    idx_r = float(idx.pct_change().iloc[-1])
    regime, reason = classify_regime(idx)
    policy = hedge_policy_for(regime)
    sig = _signals(lev, fx)
    ratio = _adaptive_ratio(regime, sig)
    ports = _portfolios_1d(lev_r, inv_r, sig)
    # 권장(regime-adaptive) 당일 수익
    adaptive_1d = round((lev_r + inv_r * ratio) * 100, 4)
    realized_vol = float(lev.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100)
    hedge_reason = f"{regime}→{policy} (vol_on={sig['vol_on']}, c60_on={sig['c60_on']}, fx_on={sig['fx_on']})"
    return {
        "date": as_of,
        "kodex200_close": round(float(idx.iloc[-1]), 2),
        "leverage_close": round(float(lev.iloc[-1]), 2),
        "inverse_close": round(float(inv.iloc[-1]), 2),
        "kodex200_ret_1d": round(idx_r * 100, 4),
        "leverage_ret_1d": round(lev_r * 100, 4),
        "inverse_ret_1d": round(inv_r * 100, 4),
        "regime": regime,
        "hedge_policy": policy,
        "hedge_ratio": ratio,
        "hedge_reason": hedge_reason,
        "portfolio_ret_1d": adaptive_1d,          # regime-adaptive 당일
        "portfolio_cum_ret": None,                 # _recompute에서 채움
        "mdd": None,                               # _recompute에서 채움
        "realized_vol": round(realized_vol, 2),
        "whipsaw_flag": None,                      # _recompute에서 채움
        "notes": reason,
        "portfolios_ret_1d": ports,
        "signals": sig,
    }


def _recompute(records: list[dict]) -> list[dict]:
    """regime-adaptive 누적수익·MDD·whipsaw 재계산(ledger 전체)."""
    records = sorted(records, key=lambda r: r.get("date", ""))
    cum = 1.0
    peak = 1.0
    ratios: list[float] = []
    for r in records:
        cum *= (1 + (r.get("portfolio_ret_1d") or 0) / 100)
        peak = max(peak, cum)
        r["portfolio_cum_ret"] = round((cum - 1) * 100, 2)
        r["mdd"] = round((cum / peak - 1) * 100, 2)
        ratios.append(1.0 if (r.get("hedge_ratio") or 0) > 0 else 0.0)
        # whipsaw: 최근 5거래일 헤지 on/off 전환 2회 이상
        recent = ratios[-5:]
        flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        r["whipsaw_flag"] = flips >= 2
    return records


def load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8")).get("records", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_ledger(records: list[dict]) -> Path:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "KODEX 레버리지+인버스 국면별 헤지 shadow 관측 (실매매 0)",
        "safety": {"real_order": False, "broker": "None", "sajang_changed": False,
                   "scheduler_changed": False, "is_recommendation": False},
        "records": records,
    }
    LEDGER_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return LEDGER_PATH


def run_snapshot(as_of: str | None = None, prefer_remote: bool = True, write: bool = True) -> dict[str, Any]:
    """오늘(또는 as_of) shadow 관측 record 생성 + ledger append. 실매매 0."""
    end = (as_of or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
    # ma60/vol60 워밍업 위해 충분히(약 150거래일)
    start = (pd.Timestamp(end) - pd.Timedelta(days=260)).strftime("%Y%m%d")
    lev = _load_close(TICKER_CORE, start, end)
    inv = _load_close(TICKER_INVERSE, start, end)
    idx = _load_close(TICKER_INDEX, start, end)
    fx = _load_fx()
    actual_date = str(idx.index[-1].date())
    record = build_record(actual_date, lev, inv, idx, fx)
    if write:
        records = [r for r in load_ledger() if r.get("date") != actual_date]
        records.append(record)
        records = _recompute(records)
        save_ledger(records)
        # 갱신된 누적값을 record에 반영해 반환
        record = next(r for r in records if r["date"] == actual_date)
    return record


if __name__ == "__main__":
    import sys
    rec = run_snapshot(write="--no-write" not in sys.argv)
    print(json.dumps({k: rec[k] for k in (
        "date", "regime", "hedge_policy", "hedge_ratio", "portfolio_ret_1d",
        "portfolio_cum_ret", "mdd", "realized_vol", "whipsaw_flag")}, ensure_ascii=False, indent=2))
