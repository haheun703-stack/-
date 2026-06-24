#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
밸류-피보나치 포트폴리오 백테스트 v3 — 진입 타이밍 개선(칼받기 방지)

v2 대비 변경: 진입조건에 '반등 확인' 추가 옵션
  v2 진입 = 60일고점 -10%↓ + RSI<40 + 수급>0  ("떨어지는 중" 진입 → 칼받기)
  v3 진입 = 위 조건 + 반등확인(RSI 상승전환 OR SAR반전 OR 당일 양봉)  ("바닥 돌아설 때" 진입)

rebound OFF/ON × 청산 3종(FIB/FIXED/TRAIL) = 6케이스 비교 → 진입개선 효과 직접 측정.
나머지(자본곡선·MDD·벤치마크·미래참조차단)는 v2와 동일.
실행: QM_ROOT=$HOME/quantum-master ./venv/bin/python3.11 -u /tmp/backtest_value_fib_v3.py
"""
import sys
import os
import glob
import json

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd

PROCESSED = os.path.join(ROOT, "data", "processed")
INIT = 100_000_000
MAX_POS = 5
SLOT = INIT / MAX_POS
ENTRY_DROP = -10.0
ENTRY_RSI = 40.0
ADD2_DROP, ADD3_DROP = -7.0, -13.0
F1, F2, F3 = 0.50, 0.30, 0.20
STOP = -20.0
TAKE_FIXED = 18.0
TRAIL = -5.0
MAX_HOLD = 30
COST = 0.004


def load_val():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "data", "valuation_gap_*.json")):
        b = os.path.basename(f).replace("valuation_gap_", "").replace(".json", "")
        if len(b) != 8 or not b.isdigit():
            continue
        date = f"{b[:4]}-{b[4:6]}-{b[6:8]}"
        try:
            d = json.load(open(f))
            out[date] = set(str(x["ticker"]).zfill(6) for x in d if x.get("grade") in ("S", "A"))
        except Exception:
            pass
    return out


def load_data(codes):
    data = {}
    need = {"close", "rsi_14", "high_60", "high", "low", "open"}
    for c in codes:
        p = os.path.join(PROCESSED, f"{c}.parquet")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_parquet(p)
            if need.issubset(df.columns):
                df = df.copy()
                df["ds"] = df.index.astype(str).str[:10]
                df["rsi_prev"] = df["rsi_14"].shift(1)
                data[c] = df.set_index("ds")
        except Exception:
            pass
    return data


def kospi_bench(start, end):
    try:
        k = pd.read_csv(os.path.join(ROOT, "data", "kospi_index.csv"))
        cl = next((c for c in k.columns if c.lower() in ("close", "종가")), None)
        k["d"] = k[k.columns[0]].astype(str).str[:10]
        k = k.set_index("d")
        kk = k.loc[(k.index >= start) & (k.index <= end), cl].dropna()
        if len(kk) > 1:
            return (kk.iloc[-1] / kk.iloc[0] - 1) * 100
    except Exception:
        pass
    return None


def _avg(shares):
    tw = sum(w for _, w in shares)
    return (sum(p * w for p, w in shares) / tw, tw) if tw else (0.0, 0.0)


def run(rule, require_rebound, days, data, val):
    cash = INIT
    pos = {}
    curve = []
    n_trades = 0
    for d in days:
        for c in list(pos):
            df = data[c]
            if d not in df.index:
                continue
            r = df.loc[d]
            hi, lo, cl = float(r["high"]), float(r["low"]), float(r["close"])
            P = pos[c]
            P["peak"] = max(P["peak"], hi)
            entry = P["entry_px"]
            dec = (lo / entry - 1) * 100
            if not P["b2"] and dec <= ADD2_DROP and cash >= SLOT * F2:
                P["shares"].append((entry * (1 + ADD2_DROP / 100), SLOT * F2)); cash -= SLOT * F2; P["b2"] = True
            if not P["b3"] and dec <= ADD3_DROP and cash >= SLOT * F3:
                P["shares"].append((entry * (1 + ADD3_DROP / 100), SLOT * F3)); cash -= SLOT * F3; P["b3"] = True
            avg, tw = _avg(P["shares"])
            P["held"] += 1
            exit_px = None
            if (lo / avg - 1) * 100 <= STOP:
                exit_px = avg * (1 + STOP / 100)
            elif rule == "FIB":
                tgt = avg + (P["high60"] - avg) * 0.618
                if hi >= tgt:
                    exit_px = tgt
            elif rule == "FIXED":
                if (hi / avg - 1) * 100 >= TAKE_FIXED:
                    exit_px = avg * (1 + TAKE_FIXED / 100)
            elif rule == "TRAIL":
                if P["peak"] > avg * 1.05 and (cl / P["peak"] - 1) * 100 <= TRAIL:
                    exit_px = cl
            if exit_px is None and P["held"] >= MAX_HOLD:
                exit_px = cl
            if exit_px is not None:
                cash += tw * (exit_px / avg) * (1 - COST)
                del pos[c]
                n_trades += 1
        slots = MAX_POS - len(pos)
        if slots > 0 and d in val:
            cands = []
            for c in val[d]:
                if c in pos or c not in data:
                    continue
                df = data[c]
                if d not in df.index:
                    continue
                r = df.loc[d]
                h60, cl, rsi = float(r["high_60"]), float(r["close"]), r["rsi_14"]
                if h60 <= 0 or pd.isna(cl) or pd.isna(rsi):
                    continue
                drop = (cl / h60 - 1) * 100
                fs = r.get("foreign_consecutive_buy", 0)
                iss = r.get("inst_consecutive_buy", 0)
                fs = 0 if pd.isna(fs) else fs
                iss = 0 if pd.isna(iss) else iss
                base = drop <= ENTRY_DROP and rsi < ENTRY_RSI and (fs > 0 or iss > 0)
                if not base:
                    continue
                if require_rebound:
                    rsi_pv = r["rsi_prev"]
                    op = float(r["open"])
                    sar = r.get("sar_reversal_up", 0)
                    sar = 0 if pd.isna(sar) else sar
                    rebound = ((not pd.isna(rsi_pv) and rsi > rsi_pv) or sar == 1 or cl > op)
                    if not rebound:
                        continue
                cands.append((drop, c, cl, h60))
            cands.sort()
            for drop, c, cl, h60 in cands[:slots]:
                first = SLOT * F1
                if cash < first:
                    continue
                cash -= first
                pos[c] = {"shares": [(cl, first)], "entry_px": cl, "high60": h60,
                          "peak": cl, "b2": False, "b3": False, "held": 0}
        eq = cash
        for c, P in pos.items():
            df = data[c]
            avg, tw = _avg(P["shares"])
            eq += tw * (float(df.loc[d, "close"]) / avg) if d in df.index else tw
        curve.append(eq)
    total = (curve[-1] / INIT - 1) * 100
    peak = curve[0]
    mdd = 0.0
    for e in curve:
        peak = max(peak, e)
        mdd = min(mdd, (e / peak - 1) * 100)
    return {"total": total, "mdd": mdd, "trades": n_trades}


def main():
    val = load_val()
    if not val:
        print("valuation_gap 없음")
        return
    vdays = sorted(val)
    start, end = vdays[0], vdays[-1]
    data = load_data(set().union(*val.values()))
    alldays = set()
    for df in data.values():
        idx = df.index
        alldays |= set(idx[(idx >= start) & (idx <= end)])
    days = sorted(alldays)
    bench = kospi_bench(start, end)
    print(f"기간 {start}~{end} ({len(days)}일) | KOSPI {bench:+.2f}%" if bench is not None else f"기간 {start}~{end}")
    print("-" * 56)
    for rb in [False, True]:
        tag = "반등확인 OFF(v2)" if not rb else "반등확인 ON (v3)"
        for rule in ["FIB", "FIXED", "TRAIL"]:
            r = run(rule, rb, days, data, val)
            print(f"[{tag} | {rule:5}] 총수익 {r['total']:+7.2f}% | MDD {r['mdd']:+6.2f}% | 거래 {r['trades']:3}건")
        print()


if __name__ == "__main__":
    main()
