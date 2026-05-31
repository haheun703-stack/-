"""리버모어 완전체 — 무릎 진입 + 피라미딩 + 위험신호 청산 (사장님 5/31, 자막 그대로).

진입(무릎): 저점높이기 + 박스 돌파 + 거대거래량 + supply_div(기관매집) → 정찰 10%
피라미딩: 진입가 +20%마다 추가 (10%→+20%→+30%→+40%, 총 100%), 역피라미드
손절: 마지막 진입가 -10% → 전량 (정찰만이면 전체 -1%, 자막 핵심)
어깨 청산: ① 역대급 거래량+오버슈팅 후 음봉/긴 윗꼬리  ② 거래량 다이버전스(신고가+거래량↓)
수익 = 비중 가중(자산 대비). 평가: 손익비·순익·평균보유.

vs 기존(supply_div + D+20 고정 100% 단일). 1년/6개월. 생존 상대비교.
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
ADD_W = [0.20, 0.30, 0.40]   # 2·3·4차 비중 (1차 정찰 0.10)


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
        h20p = df["high_20"].shift(1).values
        hl5 = df["higher_low_5d"].fillna(0).values
        sd = (df["supply_divergence"] > 0).values
        tv = (df["trading_value"] >= 1e9).values
        dates = df.index
        N = len(df)
        if mode == "full":
            sig = (hl5 > 0) & (c > h20p) & (vol >= vma20 * 2) & sd & tv
        else:
            sig = sd & tv
        sig = np.nan_to_num(sig, nan=False).astype(bool)
        i = 0
        maxhold = 90 if mode == "full" else 20
        while i < N - 1:
            if not sig[i] or not (S <= dates[i] <= E):
                i += 1
                continue
            e0 = o[i + 1]
            if e0 <= 0 or np.isnan(e0):
                i += 1
                continue
            if mode != "full":
                # 기존: 100% 단일, D+20 또는 -8% 손절
                exitp = None; ej = None
                for j in range(i + 1, min(i + 21, N)):
                    if lo[j] <= e0 * 0.92:
                        exitp = e0 * 0.92; ej = j; break
                if exitp is None:
                    ej = min(i + 20, N - 1); exitp = c[ej]
                ret = exitp / e0 - 1 - COST
                if not np.isnan(ret):
                    trades.append((ret, ej - (i + 1)))
                i = ej + 1
                continue
            # full: 피라미딩
            pos = [(0.10, e0)]            # (비중, 진입가)
            add_idx = 0
            last_entry = e0
            trig = e0 * 1.20             # 다음 추가 트리거 (+20%)
            exitp = None; ej = None
            for j in range(i + 1, min(i + 1 + maxhold, N)):
                if c[j] <= 0:
                    continue
                # 피라미딩 추가 (고가가 트리거 도달)
                while add_idx < len(ADD_W) and hi[j] >= trig:
                    pos.append((ADD_W[add_idx], trig))
                    last_entry = trig
                    add_idx += 1
                    trig = last_entry * 1.20
                # 손절: 마지막 진입가 -10%
                if lo[j] <= last_entry * 0.90:
                    exitp = last_entry * 0.90; ej = j; break
                # 어깨 위험신호
                body = abs(c[j] - o[j]); ut = hi[j] - max(o[j], c[j])
                risk1 = (vol[j] >= vma20[j] * 3) and (c[j] < o[j] or ut >= body * 1.5)
                nh = c[j] >= np.nanmax(c[i + 1:j + 1])
                risk2 = nh and (vol[j] < vma5[j]) and (j > i + 3)
                if risk1 or risk2:
                    exitp = c[j]; ej = j; break
            if exitp is None:
                ej = min(i + maxhold, N - 1); exitp = c[ej]
            tw = sum(w for w, _ in pos)
            ret = sum(w * (exitp / ep - 1) for w, ep in pos) - COST * tw  # 자산 대비
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
    return n, len(win) / n * 100, a.mean() * 100, aw, al, rr, bars.mean(), a.sum() * 100


def main() -> int:
    print("리버모어 완전체(무릎+피라미딩+위험신호) vs 기존 D+20 — 자산 대비 수익\n")
    for s, e, lbl in [("2025-06-01", "2026-05-29", "최근1년"), ("2025-12-01", "2026-05-29", "최근6개월")]:
        print(f"=== {lbl} ===")
        print(f'{"방식":<16}{"거래":>6}{"승률":>6}{"평균":>8}{"평익":>8}{"평손":>8}{"손익비":>7}{"보유":>6}{"순익합":>9}')
        for mode, nm in [("fixed", "기존 D+20(100%)"), ("full", "리버모어 완전체")]:
            r = stats(run(mode, s, e))
            if r:
                n, wr, avg, aw, al, rr, hold, net = r
                print(f'{nm:<16}{n:>6}{wr:>5.0f}%{avg:>+7.2f}%{aw:>+7.2f}%{al:>+7.2f}%{rr:>6.2f}{hold:>5.0f}일{net:>+8.0f}%')
        print()
    print("★ 완전체는 자산 대비(비중가중). 손절 정찰-1% / 추세 시 비중 폭발 → 손익비 확인.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
