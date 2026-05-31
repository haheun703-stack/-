"""리버모어 시스템 백테스트 — 무릎 진입 / 어깨 청산 / 손익비 (사장님 5/31).

오늘 우리 백테스트는 'supply_div 아무때나 진입 + D+20 고정 청산 + 평균/승률 평가'였음.
리버모어식으로 재검증:
  무릎 진입 = supply_div>0 + 박스권(20일 고점) 돌파 + 거래량 동반 (휩소 회피)
  어깨 청산 = 이평(sma20) 이탈 시 (추세 꺾임 확인, D+N 고정 X)
  칼손절   = -5% (무릎=추세 확인 후라 타이트)
  평가     = 손익비(평익/평손) · 순익 · 평균보유 (승률 무시)
vs 기존(fixed): supply_div + D+20 고정 + 손절-8%.

★ 검증 구간 1년/6개월 (레짐 혼재 방지). 생존종목(상대 비교 — 편향 동일).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

COST = 0.005


def run(mode, s, e):
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    need = ["supply_divergence", "trading_value", "high", "low", "close", "open", "volume"]
    trades = []
    S, E = pd.Timestamp(s), pd.Timestamp(e)
    for f in files:
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in need):
            continue
        o = df["open"].values; c = df["close"].values
        lo = df["low"].values; hi = df["high"].values
        sma = df["close"].rolling(20).mean().values
        boxhigh = df["high"].rolling(20).max().shift(1).values
        volma = df["volume"].rolling(20).mean().values
        sd = (df["supply_divergence"] > 0).values
        tv = (df["trading_value"] >= 1e9).values
        dates = df.index
        N = len(df)
        if mode == "liver":
            sig = sd & tv & (df["close"].values > boxhigh) & (df["volume"].values > volma)
        else:
            sig = sd & tv
        sig = np.nan_to_num(sig, nan=False).astype(bool)
        maxhold = 60 if mode == "liver" else 20
        stop = 0.05 if mode == "liver" else 0.08
        i = 0
        while i < N - 1:
            if not sig[i] or not (S <= dates[i] <= E):
                i += 1
                continue
            entry = o[i + 1]
            if entry <= 0 or np.isnan(entry):
                i += 1
                continue
            exitp = None; ej = None
            for j in range(i + 1, min(i + 1 + maxhold, N)):
                if c[j] <= 0:
                    continue
                if lo[j] <= entry * (1 - stop):
                    exitp = entry * (1 - stop); ej = j; break
                if mode == "liver" and not np.isnan(sma[j]) and c[j] < sma[j]:
                    exitp = c[j]; ej = j; break   # 어깨 = 이평 이탈
            if exitp is None:
                ej = min(i + maxhold, N - 1); exitp = c[ej]
            ret = exitp / entry - 1 - COST
            if not np.isnan(ret):
                trades.append((ret, ej - (i + 1)))
            i = ej + 1
    return trades


def stats(t):
    if not t:
        return None
    a = np.array([x[0] for x in t]); bars = np.array([x[1] for x in t])
    n = len(a); win = a[a > 0]; loss = a[a <= 0]
    aw = win.mean() * 100 if len(win) else 0
    al = loss.mean() * 100 if len(loss) else 0
    rr = (win.mean() / -loss.mean()) if (len(loss) and loss.mean() < 0) else 99.0
    return n, len(win) / n * 100, a.mean() * 100, aw, al, rr, bars.mean()


def main() -> int:
    print("리버모어(무릎/어깨/손익비) vs 기존(D+20 고정) — supply_div\n")
    for s, e, lbl in [("2025-06-01", "2026-05-29", "최근1년"), ("2025-12-01", "2026-05-29", "최근6개월")]:
        print(f"=== {lbl} ===")
        print(f'{"방식":<14}{"거래":>7}{"승률":>6}{"평균":>8}{"평익":>8}{"평손":>8}{"손익비":>7}{"평균보유":>8}')
        for mode, nm in [("fixed", "기존 D+20"), ("liver", "리버모어 무릎/어깨")]:
            r = stats(run(mode, s, e))
            if r:
                n, wr, avg, aw, al, rr, hold = r
                print(f'{nm:<14}{n:>7}{wr:>5.0f}%{avg:>+7.2f}%{aw:>+7.2f}%{al:>+7.2f}%{rr:>6.2f}{hold:>6.0f}일')
        print()
    print("★ 리버모어 손익비가 기존보다 높고 순익(평균×거래)이 크면 = 무릎/어깨가 답.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
