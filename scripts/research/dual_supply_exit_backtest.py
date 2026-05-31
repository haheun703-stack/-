"""외인+기관 동반매수 + 능동청산 백테스트 (사장님 5/31 — 마지막 미탐색 차원).

engine_ensemble에서 raw D+10 보유 시 외인+기관 동반매수가 유일 생존(PF 1.08).
실전 청산룰(손절/트레일링)을 붙이면 PF 개선되는지 검증.
신호: foreign_consecutive_buy>=3 AND inst_consecutive_buy>=3 (당일 종가 기준).
entry=다음날 시가(signal≠entry). 거래대금≥10억. 비용 0.5%. 워크포워드(23~24 vs 25~26).
★ 청산 세트는 상식 수준 사전 고정(과최적화 금지). 생존자편향(현재 생존종목, PF 상방).
"""
from __future__ import annotations

import glob
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

COST = 0.005
MIN_TV = 1e9


def signal_mask(d: pd.DataFrame) -> pd.Series:
    return (d["foreign_consecutive_buy"] >= 3) & (d["inst_consecutive_buy"] >= 3)


def backtest(dfs, s, e, stop, trail, holdmax):
    trades = []
    for code, df in dfs:
        d = df[(df.index >= s) & (df.index <= e)]
        # ★ 거래정지/데이터누락(0가격) 제거 — 미적용 시 청산가 0 → -100% 허위손실(5/31 버그)
        d = d[(d["close"] > 0) & (d["open"] > 0) & (d["low"] > 0) & (d["high"] > 0)]
        if len(d) < holdmax + 2:
            continue
        o = d["open"].values
        c = d["close"].values
        hi = d["high"].values
        lo = d["low"].values
        tv = d["trading_value"].values if "trading_value" in d.columns else np.full(len(d), 1e12)
        sig = signal_mask(d).fillna(False).values
        N = len(d)
        i = 0
        while i < N - 1:
            if not sig[i] or tv[i] < MIN_TV:
                i += 1
                continue
            entry = o[i + 1]
            if entry <= 0 or np.isnan(entry):
                i += 1
                continue
            peak = entry
            exitp = None
            ej = None
            for j in range(i + 1, min(i + 1 + holdmax, N)):
                peak = max(peak, hi[j])
                if stop and lo[j] <= entry * (1 - stop):
                    exitp = entry * (1 - stop)
                    ej = j
                    break
                if trail and peak > entry and lo[j] <= peak * (1 - trail):
                    exitp = peak * (1 - trail)
                    ej = j
                    break
            if exitp is None:
                ej = min(i + holdmax, N - 1)
                exitp = c[ej]
            ret = exitp / entry - 1 - COST
            if not np.isnan(ret):
                trades.append((code, ret))
            i = ej + 1
    return trades


def report(trades, label):
    if not trades:
        print(f"{label:<26}{'거래 0':>8}")
        return
    rets = np.array([t[1] for t in trades])
    n = len(rets)
    win = rets[rets > 0]
    loss = rets[rets <= 0]
    pf = win.sum() / (-loss.sum()) if loss.sum() < 0 else 99.0
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    cc = Counter(t[0] for t in trades)
    conc = cc.most_common(1)[0][1] / n * 100
    print(f"{label:<26}{n:>6}{len(win)/n*100:>6.0f}%{rets.mean()*100:>+8.2f}%{pf:>6.2f}{mdd:>8.1f}%{conc:>6.0f}%")


def main() -> int:
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f).sort_index()
            if len(df) >= 30:
                dfs.append((Path(f).stem, df))
        except Exception:
            pass
    print(f"종목 {len(dfs)}개 / 신호=외인+기관 동반 연속매수≥3 / entry=다음날시가 / 거래대금≥10억 / 비용0.5%\n")

    # 청산 세트 사전 고정 (과최적화 금지)
    exits = [
        ("raw 보유 D+10 (비교기준)", 0.0, 0.0, 10),
        ("손절-5% / 트레일-5% / D20", 0.05, 0.05, 20),
        ("손절-8% / 트레일-7% / D20", 0.08, 0.07, 20),
        ("손절-10% / 트레일없음 / D10", 0.10, 0.0, 10),
    ]
    periods = [("2023-06-01", "2024-12-31", "23~24약세"), ("2025-01-01", "2026-05-29", "25~26강세")]
    for name, stop, trail, hold in exits:
        print(f"=== {name} ===")
        print(f'{"구간":<26}{"거래":>6}{"승률":>7}{"평균":>9}{"PF":>6}{"MDD":>8}{"편중":>6}')
        for s, e, lbl in periods:
            report(backtest(dfs, s, e, stop, trail, hold), lbl)
        print()
    print("★ 생존자편향(현재 생존종목만, PF 상방편향). PASS 전까지 가설. 청산세트 사전고정.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
