"""한국형 리버모어 정밀 검증 (사장님 5/31).

베이스: 무릎(저점높이기+박스돌파+거대거래량+supply_div) + sma20 이탈 청산 + 손절.
①손절폭 민감도(-3/5/7/10%) → 특정값만 좋으면 과최적, 다 좋으면 강건.
②6개월 단위 워크포워드 → 매 구간 손익비 일관이면 강건, 한 구간만이면 운.
★ 생존편향: 상폐 종목은 파생지표(sma/박스/거래량ma) 부재로 무릎 계산 불가 → 생존종목만(한계 명시).
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
NEED = ["supply_divergence", "trading_value", "high", "low", "close", "open", "volume",
        "higher_low_5d", "high_20", "volume_ma20"]

_CACHE = []


def load():
    if _CACHE:
        return _CACHE
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in NEED):
            continue
        _CACHE.append(df)
    return _CACHE


def run(stop, s, e):
    S, E = pd.Timestamp(s), pd.Timestamp(e)
    trades = []
    for df in load():
        o = df["open"].values; c = df["close"].values; lo = df["low"].values; hi = df["high"].values
        vol = df["volume"].values; vma20 = df["volume_ma20"].values
        sma = df["close"].rolling(20).mean().values
        h20p = df["high_20"].shift(1).values
        hl5 = df["higher_low_5d"].fillna(0).values
        sd = (df["supply_divergence"] > 0).values
        tv = (df["trading_value"] >= 1e9).values
        dates = df.index; N = len(df)
        sig = np.nan_to_num((hl5 > 0) & (c > h20p) & (vol >= vma20 * 2) & sd & tv, nan=False).astype(bool)
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
            for j in range(i + 1, min(i + 91, N)):
                if c[j] <= 0:
                    continue
                if lo[j] <= entry * (1 - stop):
                    exitp = entry * (1 - stop); ej = j; break
                if not np.isnan(sma[j]) and c[j] < sma[j]:
                    exitp = c[j]; ej = j; break
            if exitp is None:
                ej = min(i + 90, N - 1); exitp = c[ej]
            ret = exitp / entry - 1 - COST
            if not np.isnan(ret):
                trades.append(ret)
            i = ej + 1
    return trades


def stat(t):
    if not t:
        return (0, 0, 0, 0, 0, 0)
    a = np.array(t); n = len(a); win = a[a > 0]; loss = a[a <= 0]
    rr = (win.mean() / -loss.mean()) if (len(loss) and loss.mean() < 0) else 99.0
    eq = np.cumprod(1 + a); peak = np.maximum.accumulate(eq); mdd = ((eq - peak) / peak).min() * 100
    return n, len(win) / n * 100, a.mean() * 100, rr, a.sum() * 100, mdd


def main() -> int:
    print("한국형 리버모어 정밀 검증 (무릎 + sma20 청산)\n")
    print("=== ① 손절폭 민감도 (최근1년, 특정값만 좋으면 과최적) ===")
    print(f'{"손절":>6}{"거래":>7}{"승률":>6}{"평균":>8}{"손익비":>7}{"순익합":>9}{"MDD":>8}')
    for stop in [0.03, 0.05, 0.07, 0.10]:
        n, wr, avg, rr, net, mdd = stat(run(stop, "2025-06-01", "2026-05-29"))
        print(f'{stop*100:>5.0f}%{n:>7}{wr:>5.0f}%{avg:>+7.2f}%{rr:>6.2f}{net:>+8.0f}%{mdd:>7.1f}%')

    print("\n=== ② 6개월 워크포워드 (손절-5%, 매 구간 일관?) ===")
    print(f'{"구간":<14}{"거래":>7}{"승률":>6}{"평균":>8}{"손익비":>7}{"순익합":>9}{"MDD":>8}')
    for s, e, lbl in [("2024-01-01", "2024-06-30", "24상"), ("2024-07-01", "2024-12-31", "24하"),
                       ("2025-01-01", "2025-06-30", "25상"), ("2025-07-01", "2025-12-31", "25하"),
                       ("2026-01-01", "2026-05-29", "26상")]:
        n, wr, avg, rr, net, mdd = stat(run(0.05, s, e))
        print(f'{lbl:<14}{n:>7}{wr:>5.0f}%{avg:>+7.2f}%{rr:>6.2f}{net:>+8.0f}%{mdd:>7.1f}%')
    print("\n★ 손절폭 다 손익비>1 = 강건. 6개월 매 구간 손익비>1 = 운 아님. (생존편향 미보정 — 상대지표)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
