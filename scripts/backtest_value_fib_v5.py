#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
밸류-피보나치 백테스트 v5 — 시총 Top50/Top100, 회전 vs 보유 (삼바 교훈 반영)

퐝가님 지시: 종목풀을 시총 Top50~100으로 넓혀 통계적 유의성 확보.
삼바 교훈: TRAIL-5%(타이트 회전)가 반등 전 손절 → 보유가 답.
  → 청산 3종 비교:
      TRAIL_5  : peak 대비 -5% (기존 회전, 비교 기준선)
      TRAIL_15 : peak 대비 -15% (넓은 보유 — 흔들림 견딤)
      TARGET   : 60일 전고점 회복 OR +25% 익절 / -20% 손절 (목표 보유)
진입 = 60일고점 -10%↓ + RSI<40 + 수급 + 반등확인 | 동시보유 10 | 거래비용 0.4%
출력: 자본곡선 총수익·MDD + 거래별 승률·평균 + 시총풀 Buy&Hold 벤치
실행: QM_ROOT=$HOME/quantum-master ./venv/bin/python3.11 -u /tmp/backtest_value_fib_v5.py
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
PERIOD = 240  # 1년


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


def run(rule, days, data, max_pos):
    SLOT = INIT / max_pos
    COST = 0.004
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
            elif rule == "TRAIL_5":
                if P["peak"] > avg * 1.05 and (cl / P["peak"] - 1) * 100 <= -5:
                    exit_px = cl
            elif rule == "TRAIL_15":
                if P["peak"] > avg * 1.05 and (cl / P["peak"] - 1) * 100 <= -15:
                    exit_px = cl
            elif rule == "TARGET":
                if hi >= P["high60"] or (hi / avg - 1) * 100 >= TAKE_TARGET:
                    exit_px = min(hi, max(P["high60"], avg * (1 + TAKE_TARGET / 100)))
            if exit_px is not None:
                cash += tw * (exit_px / avg) * (1 - COST)
                trades.append((exit_px / avg - 1) * 100 - COST * 100)
                del pos[c]
        slots = max_pos - len(pos)
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
                fs = r.get("foreign_consecutive_buy", 0)
                iss = r.get("inst_consecutive_buy", 0)
                fs = 0 if pd.isna(fs) else fs
                iss = 0 if pd.isna(iss) else iss
                if not (drop <= ENTRY_DROP and rsi < ENTRY_RSI and (fs > 0 or iss > 0)):
                    continue
                rsi_pv = r["rsi_prev"]
                op = float(r["open"])
                sar = r.get("sar_reversal_up", 0)
                sar = 0 if pd.isna(sar) else sar
                if not ((not pd.isna(rsi_pv) and rsi > rsi_pv) or sar == 1 or cl > op):
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


def bh(codes, data, days):
    rets = []
    for c in codes:
        if c not in data:
            continue
        df = data[c]
        idx = [x for x in days if x in df.index]
        if len(idx) > 1:
            rets.append((float(df.loc[idx[-1], "close"]) / float(df.loc[idx[0], "close"]) - 1) * 100)
    return sum(rets) / len(rets) if rets else None


def main():
    for n in [50, 100]:
        codes = top_codes(n)
        data = load_data(codes)
        alld = set()
        for df in data.values():
            alld |= set(df.index)
        days = sorted(alld)[-PERIOD:]
        bhret = bh(codes, data, days)
        print(f"\n========== 시총 Top{n} ({len(data)}종목) | {days[0]}~{days[-1]} ({len(days)}일) ==========")
        print(f"  [벤치] 시총Top{n} 균등 Buy&Hold: {bhret:+.1f}%")
        for rule in ["TRAIL_5", "TRAIL_15", "TARGET"]:
            r = run(rule, days, data, MAX_POS)
            print(f"  [{rule:8}] 총수익 {r['total']:+7.2f}% | MDD {r['mdd']:+6.2f}% | 거래 {r['trades']:3}건 | 승률 {r['wr']:2}% | 평균 {r['avg']:+.2f}%")


if __name__ == "__main__":
    main()
