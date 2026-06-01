"""포물선 가속점 전략 — "올라가는 걸 잡는다" (사장님 6/1).

추세추종(선형 무릎-어깨)은 약세 방어용. 폭등=포물선(가속)이라 가속 변곡점을 잡아야.
포물선 점 = ① 상승 가속도 양전환(ROC5 > ROC5.shift5) ② 거래량 가속 ③ 상승국면(close>ma20).
청산 = 포물선 꼭대기(거대거래량+윗꼬리/음봉) 또는 가속도 음전환. 손절 -7%.
look-ahead 0(가속도는 과거 ROC만). 포트폴리오 10슬롯. 2025중순~2026 + KOSPI 대비.
★ 사후선택·생존편향 경계. vs 인덱스(KOSPI) / 무릎(어제).
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
STOP = 0.07
NEED = ["trading_value", "high", "low", "close", "open", "volume", "volume_ma20"]


def build(mode):
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    sig = defaultdict(list); data = {}
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in NEED):
            continue
        code = Path(f).stem
        c = df["close"]
        ma20 = c.rolling(20).mean()
        roc5 = c.pct_change(5)
        accel = roc5 - roc5.shift(5)          # 상승 가속도 (look-ahead 0)
        vma5 = df["volume"].rolling(5).mean()
        volaccel = vma5 > df["volume_ma20"]   # 거래량 가속
        if mode == "parabolic":
            s = (accel > 0) & (roc5 > 0) & volaccel & (c > ma20) & (df["trading_value"] >= 1e9)
        else:  # knee (비교)
            h20p = df["high_20"].shift(1) if "high_20" in df.columns else c.rolling(20).max().shift(1)
            s = (c > h20p) & (df["volume"] >= df["volume_ma20"] * 2) & (df.get("supply_divergence", 0) > 0) & (df["trading_value"] >= 1e9)
        data[code] = dict(C=c, O=df["open"], L=df["low"], H=df["high"], MA=ma20, AC=accel,
                          V=df["volume"], VMA=df["volume_ma20"])
        for t in df.index[s.fillna(False)]:
            sig[t].append(code)
    M = {key: pd.DataFrame({c: d[key] for c, d in data.items()}).reindex(cal) for key in
         ["C", "O", "L", "H", "MA", "AC", "V", "VMA"]}
    for key in ["C", "L", "H", "MA", "AC", "V", "VMA"]:
        M[key] = M[key].ffill()
    return cal, k.set_index("date")["close"], sig, M


def sim(cal, sig, M, N, s, e):
    S, E = pd.Timestamp(s), pd.Timestamp(e)
    Cm, Om, Lm, Hm, ACm, Vm, VMAm = M["C"], M["O"], M["L"], M["H"], M["AC"], M["V"], M["VMA"]
    cols = set(Cm.columns)
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP
    for i in idxs:
        d = cal[i]
        for code in list(pos):
            if code not in cols:
                continue
            sh, entry, ph = pos[code]
            lo = Lm.at[d, code]; cl = Cm.at[d, code]; hh = Hm.at[d, code]
            if pd.isna(cl):
                continue
            if not pd.isna(hh):
                ph = max(ph, hh); pos[code] = (sh, entry, ph)
            ex = None
            if not pd.isna(lo) and lo <= entry * (1 - STOP):
                ex = entry * (1 - STOP)
            else:
                # 포물선 꼭대기: 거대거래량 + 윗꼬리/음봉
                body = abs(cl - Om.at[d, code]) if not pd.isna(Om.at[d, code]) else 0
                ut = (hh - max(Om.at[d, code], cl)) if not pd.isna(hh) and not pd.isna(Om.at[d, code]) else 0
                top = (not pd.isna(Vm.at[d, code]) and Vm.at[d, code] >= VMAm.at[d, code] * 3
                       and (cl < Om.at[d, code] or ut >= body * 1.5)) if not pd.isna(Om.at[d, code]) else False
                accel_down = not pd.isna(ACm.at[d, code]) and ACm.at[d, code] < 0
                if top or accel_down:
                    ex = cl
            if ex is not None:
                cash += sh * ex * (1 - SELL_SLIP - SELL_TAX); del pos[code]
        if i > 0:
            free = N - len(pos)
            for code in sig.get(cal[i - 1], []):
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
                cash -= sh * eo * (1 + BUY_SLIP); pos[code] = (sh, eo, eo); free -= 1
        held = sum(sh * Cm.at[d, code] for code, (sh, _e, _p) in pos.items()
                   if code in cols and not pd.isna(Cm.at[d, code]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def main() -> int:
    s, e = "2025-07-01", "2026-05-29"
    out = {}
    for mode in ["parabolic", "knee"]:
        cal, kclose, sig, M = build(mode)
        ret, mdd = sim(cal, sig, M, 10, s, e)
        out[mode] = (ret, mdd)
        ksub = kclose[(kclose.index >= pd.Timestamp(s)) & (kclose.index <= pd.Timestamp(e))]
        kos = (ksub.iloc[-1] / ksub.iloc[0] - 1) * 100
    print(f"포물선 가속점 vs 무릎 vs KOSPI — 2025중순~2026.5 (KOSPI {kos:+.1f}%, 슬롯10)\n")
    print(f'{"전략":<14}{"수익":>9}{"vs KOSPI":>10}{"MDD":>8}')
    for mode, nm in [("parabolic", "포물선 가속점"), ("knee", "무릎(어제)")]:
        ret, mdd = out[mode]
        print(f'{nm:<14}{ret*100:>+8.1f}%{(ret*100-kos):>+9.1f}%p{mdd*100:>7.1f}%')
    print(f'{"KOSPI buy&hold":<14}{kos:>+8.1f}%{0:>+9.1f}%p{"-":>7}')
    print("\n★ 포물선 가속점이 KOSPI 초과(+) + MDD 견딜만 = '올라가는 걸 잡는' 진짜. 생존편향 경계.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
