#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
밸류-피보나치 회전 전략 백테스트 (5/22 퐝가님 인사이트 + 6/24 전략화)

전략:
  종목풀  : valuation_gap grade S/A (저평가 우량 — 흑자/이익증가, PER 적정 = '문제없는 종목')
  진입    : 60일 고점 대비 -10%↓ 되돌림 + RSI<40(과매도) + 외인/기관 순매수(수급 바닥)
  분할매수: 진입가 대비 -7%(2차)·-13%(3차, 0.618 황금비=최대비중) 추가하락 시 추매 → 평단 인하
  청산    : 3종 비교  A)FIB 피보나치목표(0.618 회복)  B)FIXED 고정익절+18%  C)TRAIL 트레일링-5%
  손절    : 평단 -20% | 최대보유 30거래일 | 거래비용 왕복 0.4%

적대적 자기검증 (미래참조 차단):
  - 진입 판단은 d일 '종가까지' 정보(rsi/high_60/수급)만 사용
  - 청산은 d+1부터 일별 high/low로 추적 (진입일 청산 금지)
  - high_60은 진입일 시점값으로 고정 (목표가 미래참조 방지)
  - 거래비용을 수익에서 차감

실행(서버 ad-hoc): QM_ROOT=$HOME/quantum-master ./venv/bin/python3.11 -u /tmp/backtest_value_fib.py
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

# ── 파라미터 ──
ENTRY_DROP = -10.0   # 60일고점 대비 진입 임계(%)
ENTRY_RSI = 40.0     # RSI 과매도
ADD2_DROP = -7.0     # 진입가 대비 2차 추매
ADD3_DROP = -13.0    # 진입가 대비 3차 추매(0.618 최대)
W1, W2, W3 = 1.0, 1.0, 1.5   # 분할 비중(3차=황금비 최대)
STOP = -20.0
TAKE_FIXED = 18.0
TRAIL = -5.0
MAX_HOLD = 30
COST_PCT = 0.4       # 왕복 거래비용(%)


def load_val_grades():
    """날짜별 valuation_gap grade S/A 종목 set."""
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


def _avg(shares):
    tw = sum(w for _, w in shares)
    return sum(p * w for p, w in shares) / tw if tw else 0.0


def simulate(exit_rule, val):
    trades = []
    need = {"close", "rsi_14", "high_60", "high", "low"}
    for f in glob.glob(os.path.join(PROCESSED, "*.parquet")):
        code = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if not need.issubset(df.columns) or len(df) < 65:
            continue
        d_str = df.index.astype(str).str[:10].to_numpy()
        close = df["close"].to_numpy()
        rsi = df["rsi_14"].to_numpy()
        h60 = df["high_60"].to_numpy()
        hi = df["high"].to_numpy()
        lo = df["low"].to_numpy()
        fstreak = (df["foreign_consecutive_buy"].to_numpy() if "foreign_consecutive_buy" in df.columns else [0] * len(df))
        istreak = (df["inst_consecutive_buy"].to_numpy() if "inst_consecutive_buy" in df.columns else [0] * len(df))
        n = len(df)
        held_until = -1
        for i in range(60, n - 1):
            if i <= held_until:
                continue
            d = d_str[i]
            if d not in val or code not in val[d]:
                continue
            if h60[i] <= 0 or pd.isna(close[i]) or pd.isna(rsi[i]):
                continue
            drop = (close[i] / h60[i] - 1) * 100
            fs = fstreak[i] if not pd.isna(fstreak[i]) else 0
            iss = istreak[i] if not pd.isna(istreak[i]) else 0
            if drop > ENTRY_DROP or rsi[i] >= ENTRY_RSI or (fs <= 0 and iss <= 0):
                continue
            # ── 진입 ──
            entry_px = close[i]
            high60_fix = h60[i]
            shares = [(entry_px, W1)]
            b2 = b3 = False
            peak = entry_px
            exit_i = exit_px = reason = None
            for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
                peak = max(peak, hi[j])
                dec_entry = (lo[j] / entry_px - 1) * 100
                if not b2 and dec_entry <= ADD2_DROP:
                    shares.append((entry_px * (1 + ADD2_DROP / 100), W2)); b2 = True
                if not b3 and dec_entry <= ADD3_DROP:
                    shares.append((entry_px * (1 + ADD3_DROP / 100), W3)); b3 = True
                avg = _avg(shares)
                # 손절(평단 기준)
                if (lo[j] / avg - 1) * 100 <= STOP:
                    exit_i, exit_px, reason = j, avg * (1 + STOP / 100), "STOP"; break
                if exit_rule == "FIB":
                    tgt = avg + (high60_fix - avg) * 0.618
                    if hi[j] >= tgt:
                        exit_i, exit_px, reason = j, tgt, "FIB_TARGET"; break
                elif exit_rule == "FIXED":
                    if (hi[j] / avg - 1) * 100 >= TAKE_FIXED:
                        exit_i, exit_px, reason = j, avg * (1 + TAKE_FIXED / 100), "FIXED"; break
                elif exit_rule == "TRAIL":
                    if peak > avg * 1.05 and (close[j] / peak - 1) * 100 <= TRAIL:
                        exit_i, exit_px, reason = j, close[j], "TRAIL"; break
            if exit_i is None:
                exit_i = min(i + MAX_HOLD, n - 1)
                exit_px, reason = close[exit_i], "TIMEOUT"
            avg = _avg(shares)
            ret = (exit_px / avg - 1) * 100 - COST_PCT
            trades.append({"code": code, "entry": d, "exit": d_str[exit_i],
                           "ret": ret, "reason": reason, "hold": exit_i - i, "adds": len(shares) - 1})
            held_until = exit_i
    return trades


def report(trades, label):
    if not trades:
        print(f"=== [{label}] 거래 0건 ===")
        return
    import statistics as st
    from collections import Counter
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    n = len(rets)
    print(f"=== [{label}] 거래 {n}건 ===")
    print(f"  승률 {len(wins) * 100 // n}% | 평균 {st.mean(rets):+.2f}% | 중앙값 {st.median(rets):+.2f}% | 합산 {sum(rets):+.0f}%p")
    print(f"  최대익 {max(rets):+.1f}% | 최대손 {min(rets):+.1f}% | 평균보유 {st.mean(t['hold'] for t in trades):.1f}일 | 평균추매 {st.mean(t['adds'] for t in trades):.1f}회")
    for k, v in Counter(t["reason"] for t in trades).most_common():
        print(f"    {k}: {v}")


def main():
    val = load_val_grades()
    if not val:
        print("valuation_gap 데이터 없음 — 중단")
        return
    days = sorted(val.keys())
    print(f"밸류갭 적용기간: {days[0]} ~ {days[-1]} ({len(days)}일) | grade S/A 종목풀")
    for rule in ["FIB", "FIXED", "TRAIL"]:
        report(simulate(rule, val), rule)


if __name__ == "__main__":
    main()
