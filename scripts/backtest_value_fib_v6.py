#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
밸류-피보나치 백테스트 v6 — 종목선정 필터 비교 (TARGET 보유 고정, Top100)

퐝가님 질문: "종목 선정하는 뭔가" 추가할 게 없나?
  → 진입 기본(과매도-10%↓ + RSI<40 + 수급 + 반등)에 선별필터를 얹어 효과 측정:
      BASE  : 현재 (수급 외인 OR 기관)
      DUAL  : 쌍끌이 (외인 AND 기관 동시) — 상한가학습 수급선행 근거
      SMART : 스마트머니 유입 (smart_z > 0)
      ACCUM : 매집효율 양수 (accumulation_efficiency > 0)
청산 = TARGET(전고점/+25% 익절, -20% 손절) 고정 | 동시보유 10 | 1년
(밸류 grade S/A는 데이터 34일 한계로 백테스트 제외 → 페이퍼에서 실시간 적용)
실행: QM_ROOT=$HOME/quantum-master ./venv/bin/python3.11 -u /tmp/backtest_value_fib_v6.py
"""
import sys
import os

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd

PROCESSED = os.path.join(ROOT, "data", "processed")
INIT = 100_000_000
MAX_POS = 10
ENTRY_DROP = -10.0
ENTRY_RSI = 40.0
ADD2_DROP, ADD3_DROP = -7.0, -13.0
F1, F2, F3 = 0.50, 0.30, 0.20
STOP = -20.0
TAKE_TARGET = 25.0
PERIOD = 240
COST = 0.004
POOL = 100


def top_codes(n):
    u = pd.read_csv(os.path.join(ROOT, "data", "universe.csv"), dtype={"ticker": str})
    u = u.sort_values("market_cap", ascending=False)
    return [str(t).zfill(6) for t in u["ticker"].head(n)]


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


def _avg(shares):
    tw = sum(w for _, w in shares)
    return (sum(p * w for p, w in shares) / tw, tw) if tw else (0.0, 0.0)


def _num(v, default=0.0):
    return default if (v is None or pd.isna(v)) else float(v)


def run(filt, days, data):
    SLOT = INIT / MAX_POS
    cash = INIT
    pos = {}
    curve = []
    trades = []
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
            exit_px = None
            if (lo / avg - 1) * 100 <= STOP:
                exit_px = avg * (1 + STOP / 100)
            elif hi >= P["high60"] or (hi / avg - 1) * 100 >= TAKE_TARGET:
                exit_px = min(hi, max(P["high60"], avg * (1 + TAKE_TARGET / 100)))
            if exit_px is not None:
                cash += tw * (exit_px / avg) * (1 - COST)
                trades.append((exit_px / avg - 1) * 100 - COST * 100)
                del pos[c]
        slots = MAX_POS - len(pos)
        if slots > 0:
            cands = []
            for c in data:
                if c in pos:
                    continue
                df = data[c]
                if d not in df.index:
                    continue
                r = df.loc[d]
                h60, cl, rsi = float(r["high_60"]), float(r["close"]), r["rsi_14"]
                if h60 <= 0 or pd.isna(cl) or pd.isna(rsi):
                    continue
                drop = (cl / h60 - 1) * 100
                fs = _num(r.get("foreign_consecutive_buy"))
                iss = _num(r.get("inst_consecutive_buy"))
                if not (drop <= ENTRY_DROP and rsi < ENTRY_RSI and (fs > 0 or iss > 0)):
                    continue
                rsi_pv = r["rsi_prev"]
                op = float(r["open"])
                sar = _num(r.get("sar_reversal_up"))
                if not ((not pd.isna(rsi_pv) and rsi > rsi_pv) or sar == 1 or cl > op):
                    continue
                # ── 종목선정 필터 ──
                if filt == "DUAL" and not (fs > 0 and iss > 0):
                    continue
                if filt == "SMART" and not (_num(r.get("smart_z"), -9) > 0):
                    continue
                if filt == "ACCUM" and not (_num(r.get("accumulation_efficiency"), -9) > 0):
                    continue
                cands.append((drop, c, cl, h60))
            cands.sort()
            for drop, c, cl, h60 in cands[:slots]:
                first = SLOT * F1
                if cash < first:
                    continue
                cash -= first
                pos[c] = {"shares": [(cl, first)], "entry_px": cl, "high60": h60,
                          "peak": cl, "b2": False, "b3": False}
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
    wins = sum(1 for t in trades if t > 0)
    wr = wins * 100 // len(trades) if trades else 0
    avg_t = sum(trades) / len(trades) if trades else 0
    return {"total": total, "mdd": mdd, "trades": len(trades), "wr": wr, "avg": avg_t}


def main():
    codes = top_codes(POOL)
    data = load_data(codes)
    alld = set()
    for df in data.values():
        alld |= set(df.index)
    days = sorted(alld)[-PERIOD:]
    print(f"시총 Top{POOL} ({len(data)}종목) | {days[0]}~{days[-1]} | TARGET 보유 청산 | 종목선정 필터 비교")
    print("-" * 70)
    for filt in ["BASE", "DUAL", "SMART", "ACCUM"]:
        r = run(filt, days, data)
        print(f"  [{filt:6}] 총수익 {r['total']:+7.2f}% | MDD {r['mdd']:+6.2f}% | 거래 {r['trades']:3}건 | 승률 {r['wr']:2}% | 평균 {r['avg']:+.2f}%")


if __name__ == "__main__":
    main()
