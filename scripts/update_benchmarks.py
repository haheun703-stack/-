#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""벤치마크 지수/ETF 시계열 수집 — 지수 Buy&Hold 페이퍼(6번째)용

config/benchmarks.yaml의 20종(국내외 지수·ETF·레버리지·인버스)을 yfinance로 증분 수집
→ data/benchmark/{key}.csv (Date, close). BAT-D에서 update_kospi_index.py 옆에 배선.

사용:
    python -X utf8 scripts/update_benchmarks.py           # 증분 수집
    python -X utf8 scripts/update_benchmarks.py --check   # 현황만 확인
"""
import argparse
import os
import sys

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import yaml

try:
    import yfinance as yf
    YF = True
except ImportError:
    YF = False

CONFIG = os.path.join(ROOT, "config", "benchmarks.yaml")
OUTDIR = os.path.join(ROOT, "data", "benchmark")


def load_config():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["benchmarks"], cfg.get("baseline_start", "2026-06-22")


def _fetch_start(path: str, baseline_start: str) -> str:
    """수집 시작일: 기존 CSV 마지막날짜+1, 없으면 baseline 4일전(직전 거래일 확보)."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if len(df):
                last = str(df["Date"].iloc[-1])[:10]
                return (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            pass
    return (pd.Timestamp(baseline_start) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")


def update_one(key: str, symbol: str, baseline_start: str) -> tuple:
    path = os.path.join(OUTDIR, f"{key}.csv")
    start = _fetch_start(path, baseline_start)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    if start > today:
        return key, "skip(최신)", 0

    try:
        raw = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    except Exception as e:
        return key, f"ERR:{str(e)[:40]}", 0
    if raw is None or len(raw) == 0:
        return key, "no_new", 0

    # yfinance 최신버전 single-symbol MultiIndex 방어 (('Close','SPY')→'Close')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        return key, "no_close", 0

    new = pd.DataFrame({
        "Date": raw.index.strftime("%Y-%m-%d"),
        "close": pd.to_numeric(raw["Close"], errors="coerce").round(4).values,
    }).dropna()
    if new.empty:
        return key, "no_new", 0

    if os.path.exists(path):
        try:
            old = pd.read_csv(path)[["Date", "close"]]
            combined = pd.concat([old, new], ignore_index=True)
        except Exception:
            combined = new
    else:
        combined = new

    combined = (combined.drop_duplicates(subset="Date", keep="last")
                        .sort_values("Date").reset_index(drop=True))
    os.makedirs(OUTDIR, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return key, "ok", len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="현황만 확인")
    args = ap.parse_args()

    if not YF:
        print("[ERROR] yfinance 미설치")
        sys.exit(1)

    benchmarks, baseline = load_config()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.check:
        for key, meta in benchmarks.items():
            path = os.path.join(OUTDIR, f"{key}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                last = str(df["Date"].iloc[-1])[:10] if len(df) else "-"
                print(f"  {key:<16} {meta['symbol']:<10} rows={len(df):<5} last={last}")
            else:
                print(f"  {key:<16} {meta['symbol']:<10} (없음)")
        return

    print(f"[benchmarks] {len(benchmarks)}종 수집 (baseline={baseline})")
    ok = fail = 0
    for key, meta in benchmarks.items():
        k, status, n = update_one(key, meta["symbol"], baseline)
        print(f"  {k:<16} {meta['symbol']:<10} {status:<22} +{n}행")
        if status == "ok":
            ok += 1
        elif status.startswith("ERR") or status == "no_close":
            fail += 1
    print(f"[benchmarks] 완료: ok={ok} / fail={fail} / total={len(benchmarks)}")


if __name__ == "__main__":
    main()
