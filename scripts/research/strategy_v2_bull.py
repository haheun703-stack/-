"""저 전략 v2 — 현재 강세장(2025중순~2026) 추격형 (사장님 6/1).

어제 결론: 무릎(박스돌파)=강세장 늦음(-1.9%p), sma20 청산=추세 일찍 끊어 KOSPI(+214%) 못따라감.
재설계: ①진입 빠르게(거래량급증+돌파, 저점높이기 제거) ②청산 느슨(안 끊기: sma60/트레일-15%).
포트폴리오 N슬롯 + KOSPI 대비 + MDD. ★2024 배제, 2025중순~2026만. 손익비/MDD/KOSPI대비.
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
NEED = ["supply_divergence", "trading_value", "high", "low", "close", "open", "volume",
        "higher_low_5d", "high_20", "volume_ma20"]


def build():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"])
    cal = list(k.sort_values("date")["date"])
    kclose = dict(zip(k["date"], k["close"]))
    data = {}
    sig = {"knee": defaultdict(list), "volbreak": defaultdict(list)}
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in NEED):
            continue
        code = Path(f).stem
        sma20 = df["close"].rolling(20).mean()
        sma60 = df["close"].rolling(60).mean()
        h20p = df["high_20"].shift(1)
        sd = df["supply_divergence"] > 0
        volbig = df["volume"] >= df["volume_ma20"] * 2
        brk = df["close"] > h20p
        knee = (df["higher_low_5d"].fillna(0) > 0) & brk & volbig & sd & (df["trading_value"] >= 1e9)
        volbreak = brk & volbig & sd & (df["trading_value"] >= 1e9)  # 저점높이기 제거 = 더 빠름
        data[code] = dict(C=df["close"], O=df["open"], L=df["low"], H=df["high"], S20=sma20, S60=sma60)
        for t in df.index[knee.fillna(False)]:
            sig["knee"][t].append(code)
        for t in df.index[volbreak.fillna(False)]:
            sig["volbreak"][t].append(code)
    # 매트릭스
    M = {}
    for key in ["C", "O", "L", "H", "S20", "S60"]:
        M[key] = pd.DataFrame({c: d[key] for c, d in data.items()}).reindex(cal)
        if key in ("C", "L", "H", "S20", "S60"):
            M[key] = M[key].ffill()
    return cal, kclose, sig, M


def sim(cal, sig, M, entry_mode, exit_mode, N, s, e):
    S, E = pd.Timestamp(s), pd.Timestamp(e)
    Cm, Om, Lm, Hm, S20, S60 = M["C"], M["O"], M["L"], M["H"], M["S20"], M["S60"]
    cols = set(Cm.columns)
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP
    sg = sig[entry_mode]
    for i in idxs:
        d = cal[i]
        for code in list(pos):
            if code not in cols:
                continue
            sh, entry, ei, ph = pos[code]
            lo = Lm.at[d, code]; cl = Cm.at[d, code]; hh = Hm.at[d, code]
            if pd.isna(cl):
                continue
            if not pd.isna(hh):
                ph = max(ph, hh); pos[code] = (sh, entry, ei, ph)
            ex = None
            if not pd.isna(lo) and lo <= entry * (1 - STOP):
                ex = entry * (1 - STOP)
            elif exit_mode == "sma20" and not pd.isna(S20.at[d, code]) and cl < S20.at[d, code]:
                ex = cl
            elif exit_mode == "sma60" and not pd.isna(S60.at[d, code]) and cl < S60.at[d, code]:
                ex = cl
            elif exit_mode == "trail15" and not pd.isna(lo) and lo <= ph * 0.85:
                ex = ph * 0.85
            if ex is not None:
                cash += sh * ex * (1 - SELL_SLIP - SELL_TAX); del pos[code]
        if i > 0:
            free = N - len(pos)
            for code in sg.get(cal[i - 1], []):
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
                pos[code] = (sh, eo, i, eo); free -= 1
        held = sum(sh * Cm.at[d, code] for code, (sh, _e, _i, _p) in pos.items()
                   if code in cols and not pd.isna(Cm.at[d, code]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def main() -> int:
    cal, kclose, sig, M = build()
    s, e = "2025-07-01", "2026-05-29"
    cl = [kclose[d] for d in cal if pd.Timestamp(s) <= d <= pd.Timestamp(e) and d in kclose]
    kos = (cl[-1] / cl[0] - 1) * 100
    print(f"저 전략 v2 — 2025중순~2026.5 (KOSPI {kos:+.1f}%, 슬롯10)\n")
    print(f'{"진입":<10}{"청산":<10}{"수익":>9}{"vs KOSPI":>10}{"MDD":>8}')
    for em in ["knee", "volbreak"]:
        for xm in ["sma20", "sma60", "trail15"]:
            ret, mdd = sim(cal, sig, M, em, xm, 10, s, e)
            print(f'{em:<10}{xm:<10}{ret*100:>+8.1f}%{(ret*100-kos):>+9.1f}%p{mdd*100:>7.1f}%')
    print("\n★ vs KOSPI + 이고 MDD 견딜만 = 강세장 추격 성공. 어제 무릎+sma20(-140%p)이 출발점.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
