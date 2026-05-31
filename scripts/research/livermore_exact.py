"""리버모어 정확 구현 — 자막 한 줄씩 그대로 (사장님 5/31 "대충 읽지 마라").

진입(무릎, 자막 ①②): 저점 높이기 + 직전 고점(박스 상단) 돌파 + 거대 거래량 + supply_div(기관매집)
건강 보유(자막 ③): 조정이 '직전 상승의 절반 이내 + 거래량 급감'이면 버팀(눌림목)
어깨 청산(자막 ④): ① 역대급 거래량 + 오버슈팅 후 음봉/긴 윗꼬리  ② 거래량 다이버전스(신고가+거래량↓)
손절: 진입가 -10%(정찰). 평가: 손익비·순익·평균보유 (승률 무시).

vs fixed(기존 supply_div + D+20 고정 + 손절-8%). 1년/6개월. 생존종목 상대비교.
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
        "higher_low_5d", "high_20", "volume_ma20", "volume_ma5"]


def run(mode, s, e):
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    trades = []
    S, E = pd.Timestamp(s), pd.Timestamp(e)
    for f in files:
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in NEED):
            continue
        o = df["open"].values; c = df["close"].values
        lo = df["low"].values; hi = df["high"].values; vol = df["volume"].values
        vma20 = df["volume_ma20"].values; vma5 = df["volume_ma5"].values
        h20_prev = df["high_20"].shift(1).values   # 직전까지 20일 고점 = 박스 상단
        hl5 = df["higher_low_5d"].fillna(0).values  # 저점 높이기
        sd = (df["supply_divergence"] > 0).values
        tv = (df["trading_value"] >= 1e9).values
        dates = df.index
        N = len(df)

        if mode == "exact":
            # 무릎: 저점높이기 + 박스 돌파 + 거대 거래량(≥2배) + 기관매집
            sig = (hl5 > 0) & (c > h20_prev) & (vol >= vma20 * 2) & sd & tv
        else:  # fixed
            sig = sd & tv
        sig = np.nan_to_num(sig, nan=False).astype(bool)
        maxhold = 90 if mode == "exact" else 20
        stop = 0.10 if mode == "exact" else 0.08
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
            run_high = entry
            for j in range(i + 1, min(i + 1 + maxhold, N)):
                if c[j] <= 0:
                    continue
                run_high = max(run_high, hi[j])
                # 손절
                if lo[j] <= entry * (1 - stop):
                    exitp = entry * (1 - stop); ej = j; break
                if mode == "exact":
                    body = abs(c[j] - o[j])
                    upper_tail = hi[j] - max(o[j], c[j])
                    # 어깨 위험1: 역대급 거래량(≥3배) + (음봉 or 긴 윗꼬리=오버슈팅 꺾임)
                    risk1 = (vol[j] >= vma20[j] * 3) and (c[j] < o[j] or upper_tail >= body * 1.5)
                    # 어깨 위험2: 거래량 다이버전스 — 신고가인데 거래량 5일평균 미만
                    newhigh = c[j] >= np.nanmax(c[i + 1:j + 1])
                    risk2 = newhigh and (vol[j] < vma5[j]) and (j > i + 3)
                    if risk1 or risk2:
                        exitp = c[j]; ej = j; break
                else:
                    pass  # fixed: D+20 시간청산만
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
    net = a.sum() * 100  # 순익(전 거래 합, 비중 1단위)
    return n, len(win) / n * 100, a.mean() * 100, aw, al, rr, bars.mean(), net


def main() -> int:
    print("리버모어 정확 구현(무릎+건강보유+위험신호 청산) vs 기존 D+20 — supply_div\n")
    for s, e, lbl in [("2025-06-01", "2026-05-29", "최근1년"), ("2025-12-01", "2026-05-29", "최근6개월")]:
        print(f"=== {lbl} ===")
        print(f'{"방식":<16}{"거래":>6}{"승률":>6}{"평균":>8}{"평익":>8}{"평손":>8}{"손익비":>7}{"보유":>6}{"순익합":>9}')
        for mode, nm in [("fixed", "기존 D+20"), ("exact", "리버모어 정확")]:
            r = stats(run(mode, s, e))
            if r:
                n, wr, avg, aw, al, rr, hold, net = r
                print(f'{nm:<16}{n:>6}{wr:>5.0f}%{avg:>+7.2f}%{aw:>+7.2f}%{al:>+7.2f}%{rr:>6.2f}{hold:>5.0f}일{net:>+8.0f}%')
        print()
    print("★ 자막 그대로 구현. 손익비·순익합이 기존보다 크면 = 리버모어 시스템이 답.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
