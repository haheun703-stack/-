"""한국형 리버모어 포트폴리오 백테스트 — MDD 관리 (사장님 5/31).

단일 순차 베팅은 MDD -98%(비현실). N슬롯 분산 + 현금 재활용으로 실전화.
진입(무릎): 저점높이기+박스돌파+거대거래량+supply_div, 전일신호 → 당일 시가, 슬롯 분배
청산: 손절-5% 또는 sma20 이탈(어깨). 비용 매수/매도 슬리피지 0.15%+세금0.18%.
평가: 최종수익·MDD·거래수·승률. 1년/6개월. ★생존편향 미보정(상대지표).
"""
from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

CAP = 100_000_000
BUY_SLIP, SELL_SLIP, SELL_TAX = 0.0015, 0.0015, 0.0018
NEED = ["supply_divergence", "trading_value", "high", "low", "close", "open", "volume",
        "higher_low_5d", "high_20", "volume_ma20"]


def build():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"])
    cal = list(k.sort_values("date")["date"])
    sig_by = defaultdict(list)
    C, O, L, SMA = {}, {}, {}, {}
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in NEED):
            continue
        code = Path(f).stem
        sma = df["close"].rolling(20).mean()
        h20p = df["high_20"].shift(1)
        knee = (df["higher_low_5d"].fillna(0) > 0) & (df["close"] > h20p) & \
               (df["volume"] >= df["volume_ma20"] * 2) & (df["supply_divergence"] > 0) & \
               (df["trading_value"] >= 1e9)
        C[code] = df["close"]; O[code] = df["open"]; L[code] = df["low"]; SMA[code] = sma
        for t in df.index[knee.fillna(False)]:
            sig_by[t].append(code)
    Cm = pd.DataFrame(C).reindex(cal).ffill()
    Om = pd.DataFrame(O).reindex(cal)
    Lm = pd.DataFrame(L).reindex(cal).ffill()
    SMAm = pd.DataFrame(SMA).reindex(cal).ffill()
    return cal, sig_by, Cm, Om, Lm, SMAm


def sim(cal, sig_by, Cm, Om, Lm, SMAm, N, stop, s, e):
    S, E = pd.Timestamp(s), pd.Timestamp(e)
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}
    peak = CAP; mdd = 0.0; mv = CAP
    entries = 0; wins = 0; closed = 0
    cols = set(Cm.columns)
    for i in idxs:
        d = cal[i]
        for code in list(pos):
            if code not in cols:
                continue
            sh, entry, ei = pos[code]
            lo = Lm.at[d, code]; cl = Cm.at[d, code]; sm = SMAm.at[d, code]
            if pd.isna(cl):
                continue
            ex = None
            if not pd.isna(lo) and lo <= entry * (1 - stop):
                ex = entry * (1 - stop)
            elif not pd.isna(sm) and cl < sm:
                ex = cl
            if ex is not None:
                cash += sh * ex * (1 - SELL_SLIP - SELL_TAX)
                closed += 1
                if ex > entry:
                    wins += 1
                del pos[code]
        if i > 0:
            dp = cal[i - 1]
            free = N - len(pos)
            for code in sig_by.get(dp, []):
                if free <= 0:
                    break
                if code in pos or code not in cols:
                    continue
                eo = Om.at[d, code]
                if pd.isna(eo) or eo <= 0:
                    continue
                alloc = cash / free
                if alloc < 300_000:
                    continue
                sh = int(alloc / (eo * (1 + BUY_SLIP)))
                if sh <= 0:
                    continue
                cash -= sh * eo * (1 + BUY_SLIP)
                pos[code] = (sh, eo, i); free -= 1; entries += 1
        held = 0.0
        for code, (sh, _e, _i) in pos.items():
            if code in cols:
                px = Cm.at[d, code]
                if not pd.isna(px):
                    held += sh * px
        mv = cash + held
        peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd, entries, (wins / closed * 100 if closed else 0)


def main() -> int:
    cal, sig_by, Cm, Om, Lm, SMAm = build()
    print("한국형 리버모어 포트폴리오 (무릎+sma20어깨+손절5%, 현금재활용)\n")
    for s, e, lbl in [("2025-06-01", "2026-05-29", "최근1년"), ("2025-12-01", "2026-05-29", "최근6개월"),
                       ("2024-01-01", "2024-12-31", "2024약세")]:
        print(f"=== {lbl} ===")
        print(f'{"슬롯":>5}{"최종수익":>10}{"MDD":>8}{"진입":>7}{"승률":>6}')
        for N in (5, 10, 20):
            ret, mdd, ent, wr = sim(cal, sig_by, Cm, Om, Lm, SMAm, N, 0.05, s, e)
            print(f'{N:>5}{ret*100:>+9.1f}%{mdd*100:>7.1f}%{ent:>7}{wr:>5.0f}%')
        print()
    print("★ 분산(N슬롯)으로 MDD가 -30%대 이하로 잡히고 수익 +면 = 실전 후보. (생존편향 미보정)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
