"""삼성전자 단일 레버리지 룰 비교 — 사장님룰 vs C60 vs buyhold (6/2).

사장님 지시: 488080 C60 shadow 옆에 삼성 단일레버(0193W0) shadow 병렬 추적.
구현 전 검증: 삼성전자(005930)×2 합성 레버로 룰 비교.
★삼성은 2022 약세장 실데이터 있음 → SOXX/SOXL 대용 불필요, 실제 검증.

룰:
  A 원주 buyhold (005930 1배, 기준선)
  B 레버 buyhold (005930 ×2, 그냥 들기 — decay 노출)
  C C60 (60선 단독 추세추종, strict 재진입)
  D 사장님룰 (진입/보유: 종가>60선 AND 종가>20선 AND 20선상승 / 이탈: 60선 이탈)
1차는 가격신호만 (수급/FIRE/미중헤드라인은 forward shadow에서 kis_investor로 추가).
look-ahead 0(종가 신호→다음날). 비용 0.1%. 시드 2천만.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

SWITCH = 0.001
SEED = 20_000_000


def load_samsung():
    df = fdr.DataReader("005930", "2021-01-01", "2026-06-02")
    close = df["Close"].astype(float); vol = df["Volume"].astype(float)
    return close, vol


def sim(close, vol, mode, start, end):
    """mode: A(원주1배)/B(레버2배buyhold)/C(C60)/D(사장님룰). 레버=일일2배 합성."""
    ma5 = close.rolling(5).mean(); ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    ma20_up = ma20 > ma20.shift(5); ma60_up = ma60 > ma60.shift(5)
    r1 = close.pct_change().fillna(0)
    idx = [d for d in close.index if start <= d <= end]
    v = 1.0; prev_h = None; curve = []; trades = 0; cash = 0
    next_trend = True; eh = None; prev_vol = None
    for d in idx:
        th = next_trend
        if mode == "A":
            holding = True; lev = 1.0
        elif mode == "B":
            holding = True; lev = 2.0
        else:
            holding = th; lev = 2.0
        rd = r1.get(d, 0)
        if holding:
            v *= (1 + lev * rd)
        else:
            cash += 1
        if prev_h is not None and holding != prev_h:
            v *= (1 - SWITCH); trades += 1
        curve.append(v); prev_h = holding
        # 당일 종가로 다음날 추세 (C/D)
        p = close.get(d); m5 = ma5.get(d); m20 = ma20.get(d); m60 = ma60.get(d)
        up20 = bool(ma20_up.get(d, False)); up60 = bool(ma60_up.get(d, False)); vd = vol.get(d)
        if mode in ("C", "D"):
            if pd.isna(m60):
                next_trend = True; prev_vol = vd; continue
            if th:  # 보유 중 → 이탈 체크
                if p < m60:
                    next_trend = False; eh = None
            else:  # 현금 중 → 재진입 체크
                if mode == "C":  # strict
                    volup = prev_vol is not None and not np.isnan(vd) and vd > prev_vol
                    next_trend = (p > m5) and (p > m60) and (rd >= 0.02) and volup and up60
                else:  # D 사장님룰: 60선+20선+20선상승
                    next_trend = (p > m60) and (p > m20) and up20
        prev_vol = vd
    eq = pd.Series(curve, index=idx)
    ret = (eq.iloc[-1] - 1) * 100
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    final = SEED * eq.iloc[-1]
    return dict(ret=ret, mdd=mdd, final=final, trades=trades, cash=cash / len(idx) * 100)


def report(close, vol, label, start, end):
    print(f"\n● {label}  ({start.date()}~{end.date()}, 삼성 원주 {(close.loc[start:end].iloc[-1]/close.loc[start:end].iloc[0]-1)*100:+.0f}%)")
    print(f'  {"룰":<26}{"최종(2천만)":>15}{"수익%":>9}{"MDD%":>7}{"거래":>5}{"현금%":>7}')
    rows = [("A 원주 buyhold(1배)", "A"), ("B 레버 buyhold(2배)", "B"),
            ("C C60(60선단독)", "C"), ("D 사장님룰(60+20+20up)", "D")]
    out = {}
    for nm, m in rows:
        x = sim(close, vol, m, start, end); out[m] = x
        print(f'  {nm:<26}{x["final"]:>14,.0f}{x["ret"]:>+8.0f}%{x["mdd"]:>6.0f}%{x["trades"]:>5}{x["cash"]:>6.0f}%')
    return out


def main() -> int:
    close, vol = load_samsung()
    print("=" * 68)
    print("삼성전자(005930) ×2 합성 레버 — 사장님룰 vs C60 vs buyhold (시드 2천만)")
    print("=" * 68)
    periods = [
        ("강세장", pd.Timestamp("2025-06-02"), pd.Timestamp("2026-06-02")),
        ("2022 약세장 (실데이터!)", pd.Timestamp("2021-12-01"), pd.Timestamp("2022-12-30")),
        ("전체 5년", pd.Timestamp("2021-06-01"), pd.Timestamp("2026-06-02")),
    ]
    results = {}
    for lab, s, e in periods:
        results[lab] = report(close, vol, lab, s, e)

    print("\n" + "=" * 68)
    print("핵심 비교 — C60(60선단독) vs D(사장님 다중조건)")
    for lab in results:
        c, d = results[lab]["C"], results[lab]["D"]
        better = "D 우위" if (d["ret"] > c["ret"] and d["mdd"] >= c["mdd"]) else ("C 우위" if c["ret"] > d["ret"] else "혼재")
        print(f"  [{lab:<22}] C: {c['ret']:+.0f}%/{c['mdd']:.0f}%/{c['trades']}거래  vs  D: {d['ret']:+.0f}%/{d['mdd']:.0f}%/{d['trades']}거래 → {better}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
