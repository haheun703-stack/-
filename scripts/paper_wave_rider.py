#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""파도타기 페이퍼 (7번째 트랙) — 레짐 방향성 페이퍼 (KR+US, 7/7 퐝가님 비전).

"오르면 레버리지, 내리면 (일단) 현금" — V3b 레짐 판정의 실전(페이퍼) 검증 트랙.
  BULL → 레버리지 2x / CAUTION → 지수 1x / BEAR·CRISIS → 현금
  (하강면 인버스는 phase12 기각 — 지수BH 인버스 3종 관측으로 신호 축적 후 재론)

- 레짐 = V3b(close>MA20 AND 하방변동 252일 백분위<60 → BULL). 근거=brain_bull_relax_report.
- KR: kospi_index.csv 판정 → KODEX레버리지(KODEX_LEV)/KODEX200 체결(벤치마크 CSV 종가)
- US: us_daily.parquet spy_close 판정 → SSO/SPY 체결. ★US는 전일 미장 종가 기준(1일 지연 구조).
- 관측 전용·실주문 0·freeze 무관. 주말 가드·일 1회 idempotent·가격 5일 stale 시 매매 스킵.
- 수수료 0.05%/편도. 출력: data/paper_portfolio_wave_{kr,us}.json

실행(BAT-D, update_benchmarks 이후): python -X utf8 scripts/paper_wave_rider.py
"""
import sys
import os

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

import io
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA = os.path.join(ROOT, "data")
INITIAL = 100_000_000
FEE = 0.0005          # 편도 0.05%
STALE_DAYS = 5

MARKETS = {
    "kr": {
        "label": "한국",
        "assets": {"LEV": "KODEX_LEV", "INDEX": "KODEX200"},   # data/benchmark/{key}.csv
        "names": {"LEV": "KODEX레버리지", "INDEX": "KODEX200", "CASH": "현금"},
    },
    "us": {
        "label": "미국",
        "assets": {"LEV": "SSO", "INDEX": "SPY"},
        "names": {"LEV": "SSO(2x)", "INDEX": "SPY", "CASH": "현금"},
    },
}
POLICY = {"BULL": "LEV", "CAUTION": "INDEX", "BEAR": "CASH", "CRISIS": "CASH"}


def _index_series(market: str) -> pd.Series | None:
    """레짐 판정용 지수 종가 시계열 (시간순)."""
    try:
        if market == "kr":
            k = pd.read_csv(os.path.join(DATA, "kospi_index.csv"))
            k["Date"] = pd.to_datetime(k["Date"])
            return pd.to_numeric(k.sort_values("Date").set_index("Date")["close"],
                                 errors="coerce").dropna()
        df = pd.read_parquet(os.path.join(DATA, "us_market", "us_daily.parquet"))
        if not isinstance(df.index, pd.DatetimeIndex):
            dc = next((c for c in df.columns if "date" in c.lower()), None)
            if dc:
                df = df.assign(_d=pd.to_datetime(df[dc], errors="coerce")).set_index("_d")
        return pd.to_numeric(df.sort_index()["spy_close"], errors="coerce").dropna()
    except Exception as e:
        print(f"[wave:{market}] 지수 로드 실패(graceful): {e}")
        return None


def v3b_regime(close: pd.Series) -> tuple[str, str]:
    """V3b 레짐 (BULL = MA20 위 AND 하방변동 252일 백분위<60). returns (regime, asof)."""
    if close is None or len(close) < 60:
        return "CAUTION", "?"
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    ret = close.pct_change()
    dn = ret.where(ret < 0, 0.0)
    drv = dn.rolling(20).std() * np.sqrt(252) * 100
    drv_pct = (drv.rolling(252, min_periods=60).rank(pct=True) * 100).fillna(50).iloc[-1]
    c = close.iloc[-1]
    if c > ma20:
        reg = "BULL" if drv_pct < 60 else "CAUTION"
    elif c > ma60:
        reg = "BEAR"
    else:
        reg = "CRISIS"
    return reg, str(close.index[-1].date())


def _price(key: str) -> tuple[float, str] | None:
    """벤치마크 CSV 최신 종가 (체결가). (price, date) 또는 None."""
    path = os.path.join(DATA, "benchmark", f"{key}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["close"])
        row = df.iloc[-1]
        return float(row["close"]), str(row["Date"].date())
    except Exception:
        return None


def _load_book(market: str) -> dict:
    path = os.path.join(DATA, f"paper_portfolio_wave_{market}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"created": datetime.now().strftime("%Y-%m-%d"), "initial_capital": INITIAL,
            "cash": INITIAL, "position": None,  # {"target","key","name","qty","entry_price","entry_date"}
            "trades": [], "daily_equity": [], "policy": "V3b(drv<60) BULL=LEV/CAUTION=IDX/BEAR,CRISIS=CASH"}


def _save_book(market: str, book: dict) -> None:
    """원자적 저장 — 중도 kill 시 JSON 파손으로 이후 실행이 조용히 죽는 것 방지."""
    path = os.path.join(DATA, f"paper_portfolio_wave_{market}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_market(market: str, today: str) -> None:
    cfg = MARKETS[market]
    book = _load_book(market)
    if book["daily_equity"] and book["daily_equity"][-1]["date"] == today:
        print(f"[wave:{market}] 오늘({today}) 이미 기록 — 스킵")
        return
    reg, asof = v3b_regime(_index_series(market))
    pos = book.get("position")
    cur = pos["target"] if pos else "CASH"

    # ★fail-safe: 지수 데이터 실패(asof='?') 시 CAUTION 폴백으로 매수하지 않도록 현상 유지
    data_ok = asof != "?"
    if data_ok:
        target = POLICY.get(reg, "CASH")
    else:
        reg, target = "DATA_FAIL", cur
        print(f"[wave:{market}] 지수 데이터 실패 — 매매 스킵(현상 유지)")

    # 체결가 준비 (stale 가드)
    prices = {}
    fresh = True
    for tkey, bmkey in cfg["assets"].items():
        pr = _price(bmkey)
        if pr is None or (datetime.strptime(today, "%Y-%m-%d")
                          - datetime.strptime(pr[1], "%Y-%m-%d")).days > STALE_DAYS:
            fresh = False
        prices[tkey] = pr

    if fresh and data_ok and target != cur:
        # 청산
        if pos:
            px = prices[pos["target"]][0]
            proceeds = pos["qty"] * px * (1 - FEE)
            pnl_pct = (px / pos["entry_price"] - 1) * 100
            book["cash"] += proceeds
            book["trades"].append({"date": today, "side": "SELL", "name": pos["name"],
                                   "price": px, "qty": pos["qty"],
                                   "pnl_pct": round(pnl_pct, 2), "reason": f"regime→{reg}"})
            book["position"] = None
        # 진입
        if target != "CASH":
            px, pdate = prices[target]
            qty = int(book["cash"] * (1 - FEE) / px)
            if qty > 0:
                book["cash"] -= qty * px * (1 + FEE)
                book["position"] = {"target": target, "key": cfg["assets"][target],
                                    "name": cfg["names"][target], "qty": qty,
                                    "entry_price": px, "entry_date": today}
                book["trades"].append({"date": today, "side": "BUY",
                                       "name": cfg["names"][target], "price": px,
                                       "qty": qty, "price_date": pdate})
    elif not fresh:
        print(f"[wave:{market}] 가격 stale(>{STALE_DAYS}일) — 매매 스킵, 평가만")

    # 일일 평가 — 보유 자산 가격이 없으면 왜곡값(현금만) 기록 대신 오늘 기록 스킵
    pos = book.get("position")
    if pos and not prices.get(pos["target"]):
        print(f"[wave:{market}] 보유 {pos['name']} 가격 없음 — 오늘 평가 기록 스킵")
        book["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_book(market, book)
        return
    equity = book["cash"]
    if pos:
        equity += pos["qty"] * prices[pos["target"]][0]
    book["daily_equity"].append({"date": today, "equity": round(equity),
                                 "regime": reg, "asof": asof,
                                 "holding": cfg["names"][pos["target"]] if pos else "현금"})
    book["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_book(market, book)
    ret = (equity / book["initial_capital"] - 1) * 100
    holding = cfg["names"][pos["target"]] if pos else "현금"
    print(f"[wave:{cfg['label']}] {today} 레짐={reg}(asof {asof}) 목표={cfg['names'][target]} | "
          f"보유={holding} | 평가 {equity:,.0f} ({ret:+.2f}%)")


def main() -> int:
    now = datetime.now()
    if now.weekday() >= 5:
        print("[wave] 주말 — 스킵")
        return 0
    today = now.strftime("%Y-%m-%d")
    for market in ("kr", "us"):
        try:
            run_market(market, today)
        except Exception as e:
            print(f"[wave:{market}] 실패(graceful): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
