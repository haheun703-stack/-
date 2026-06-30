"""US 주도주 데이터 수집 — yfinance 주봉 + 분기/연간 영업이익.

주도주 사이클 진단/백테스트(leader_cycle)의 US 입력. yfinance는 로컬(한국)에서 Yahoo
차단으로 실패하므로 VPS(고정IP)에서 실행 권장:
    ssh ... 'cd ~/quantum-master && ./venv/bin/python3.11 -' < scripts/fetch_us_leader_data.py

산출(cwd 기준):
    data/us_market/leader_cycle/{ticker}.parquet      — 주봉 OHLCV (naive index, 소문자)
    data/us_market/leader_cycle/{ticker}_opinc.json   — {"quarterly": {date: 영업이익}, "annual": {...}}

설계: 순수 수집기(분석/판정 없음). 사이클 판정은 leader_cycle_diagnosis가 담당.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# 검증/관측 대상 US 주도주 (AI·반도체·플랫폼 중심)
US_LEADERS = ["NVDA", "META", "AVGO", "PLTR", "SMCI", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL"]

OUT_DIR = Path("data/us_market/leader_cycle")


def _save_price(tk: str) -> str:
    import yfinance as yf

    h = yf.Ticker(tk).history(period="7y", interval="1wk")
    if h is None or h.empty:
        return f"  {tk:6}: 가격 빈값"
    h = h.rename(columns={c: c.lower() for c in h.columns})
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in h.columns]
    h = h[keep].copy()
    if getattr(h.index, "tz", None) is not None:   # tz-aware → naive (엔진 일관)
        h.index = h.index.tz_localize(None)
    h.index.name = "date"
    h = h[h["close"].notna()]
    (OUT_DIR / f"{tk}.parquet").parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(OUT_DIR / f"{tk}.parquet")
    return f"  {tk:6}: 주봉 {len(h):>4}주 {h.index[0].date()}~{h.index[-1].date()}"


def _save_opinc(tk: str) -> str:
    import yfinance as yf

    t = yf.Ticker(tk)
    out: dict = {"quarterly": {}, "annual": {}}
    for kind, df in (("quarterly", t.quarterly_income_stmt), ("annual", t.income_stmt)):
        if df is not None and not df.empty and "Operating Income" in df.index:
            oi = df.loc["Operating Income"]
            for k, v in oi.items():
                if pd.notna(v):
                    out[kind][str(pd.Timestamp(k).date())] = float(v)
    with open(OUT_DIR / f"{tk}_opinc.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return f"opinc(q{len(out['quarterly'])}/a{len(out['annual'])})"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = sys.argv[1:] or US_LEADERS
    print(f"=== US 주도주 데이터 수집: {len(tickers)}종목 → {OUT_DIR} ===")
    for tk in tickers:
        try:
            line = _save_price(tk)
            fin = _save_opinc(tk)
            print(f"{line} | {fin}")
        except Exception as e:  # noqa: BLE001
            print(f"  {tk:6}: ERR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
