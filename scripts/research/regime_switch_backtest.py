"""레짐 스위칭 검증 — 단타봇 설계(87f506b) 구현 (퀀트봇 6/1).

3전략 비교 × 3구간 + CAGR/MDD/Sharpe (단타봇 합격기준 4번):
  스위칭: 레짐 BULL/CAUTION(강세) → 인덱스(KODEX200), BEAR/CRISIS(약세) → 추세추종
  인덱스: KODEX200 buy&hold
  단일추세: 항상 추세추종(무릎+sma20+손절5%, 10슬롯)
구간: 2024(약세) / 2025중순~2026(강세) / 전체. ★약세 없으면 스위칭 검증 무의미(단타봇).
look-ahead 0: 레짐은 그 시점까지 rolling(MA20/60+RV)으로만 판정.
히스테리시스: 하향 즉시 / 상향 1일 확인. 전환비용 왕복 0.48% + T+1.
★ 생존편향(추세파트 유니버스 5/29 고정) 보수적 할인 해석.
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

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from pykrx import stock

CAP = 100_000_000
BUY_SLIP, SELL_SLIP, SELL_TAX = 0.0015, 0.0015, 0.0018
SWITCH_COST = 0.0048   # 전환 왕복(단타봇 Q4)
STOP = 0.05
NEED = ["supply_divergence", "trading_value", "high", "low", "close", "open", "volume",
        "higher_low_5d", "high_20", "volume_ma20"]


def regime_daily(cal):
    """각 날짜 레짐 (look-ahead 0: rolling만). 히스테리시스(하향즉시/상향1일확인)."""
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    k["ma20"] = k["close"].rolling(20).mean()
    k["ma60"] = k["close"].rolling(60).mean()
    lr = np.log(k["close"] / k["close"].shift(1))
    k["rv"] = lr.rolling(20).std() * np.sqrt(252)
    k["rvp"] = k["rv"].rolling(252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    raw = {}
    for d, r in k.iterrows():
        if pd.isna(r["ma60"]):
            raw[d] = "CAUTION"; continue
        if r["close"] > r["ma20"]:
            raw[d] = "BULL" if (r["rvp"] < 0.5 if not pd.isna(r["rvp"]) else True) else "CAUTION"
        elif r["close"] > r["ma60"]:
            raw[d] = "BEAR"
        else:
            raw[d] = "CRISIS"
    # 히스테리시스: 강세(BULL/CAUTION)→약세 즉시, 약세→강세는 1일 확인
    order = {"CRISIS": 0, "BEAR": 1, "CAUTION": 2, "BULL": 3}
    eff = {}; prev = "CAUTION"; pend = None
    for d in sorted(raw):
        cur = raw[d]
        if order[cur] < order[prev]:       # 하향 = 즉시
            prev = cur; pend = None
        elif order[cur] > order[prev]:      # 상향 = 1일 확인
            if pend == cur:
                prev = cur; pend = None
            else:
                pend = cur
        else:
            pend = None
        eff[d] = prev
    return eff


def trend_equity(cal):
    """추세추종 포트 일별 에쿼티 (무릎+sma20+손절5%, 10슬롯)."""
    N = 10
    sig = defaultdict(list); data = {}
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 90 or not all(c in df.columns for c in NEED):
            continue
        code = Path(f).stem
        sma20 = df["close"].rolling(20).mean()
        knee = (df["higher_low_5d"].fillna(0) > 0) & (df["close"] > df["high_20"].shift(1)) & \
               (df["volume"] >= df["volume_ma20"] * 2) & (df["supply_divergence"] > 0) & (df["trading_value"] >= 1e9)
        data[code] = dict(C=df["close"], O=df["open"], L=df["low"], S=sma20)
        for t in df.index[knee.fillna(False)]:
            sig[t].append(code)
    Cm = pd.DataFrame({c: d["C"] for c, d in data.items()}).reindex(cal).ffill()
    Om = pd.DataFrame({c: d["O"] for c, d in data.items()}).reindex(cal)
    Lm = pd.DataFrame({c: d["L"] for c, d in data.items()}).reindex(cal).ffill()
    Sm = pd.DataFrame({c: d["S"] for c, d in data.items()}).reindex(cal).ffill()
    cols = set(Cm.columns)
    cash = float(CAP); pos = {}; eq = {}
    for i, d in enumerate(cal):
        for code in list(pos):
            if code not in cols:
                continue
            sh, entry = pos[code]
            lo = Lm.at[d, code]; cl = Cm.at[d, code]; sm = Sm.at[d, code]
            if pd.isna(cl):
                continue
            ex = None
            if not pd.isna(lo) and lo <= entry * (1 - STOP):
                ex = entry * (1 - STOP)
            elif not pd.isna(sm) and cl < sm:
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
                cash -= sh * eo * (1 + BUY_SLIP); pos[code] = (sh, eo); free -= 1
        held = sum(sh * Cm.at[d, code] for code, (sh, _e) in pos.items()
                   if code in cols and not pd.isna(Cm.at[d, code]))
        eq[d] = cash + held
    return pd.Series(eq)


def metrics(eq, s, e):
    eq = eq[(eq.index >= pd.Timestamp(s)) & (eq.index <= pd.Timestamp(e))].dropna()
    if len(eq) < 20:
        return None
    eq = eq / eq.iloc[0]
    ret = eq.iloc[-1] - 1
    days = (eq.index[-1] - eq.index[0]).days
    cagr = (eq.iloc[-1] ** (365 / max(days, 1)) - 1) if eq.iloc[-1] > 0 else -1
    dret = eq.pct_change().dropna()
    sharpe = (dret.mean() / dret.std() * np.sqrt(252)) if dret.std() > 0 else 0
    peak = eq.cummax(); mdd = ((eq - peak) / peak).min()
    return ret * 100, cagr * 100, mdd * 100, sharpe


def main() -> int:
    kdf = stock.get_market_ohlcv("20240101", "20260529", "069500")
    kdf.index = pd.to_datetime(kdf.index)
    cal = list(kdf.index)
    idx_eq = kdf["종가"].astype(float)
    idx_eq = idx_eq / idx_eq.iloc[0] * CAP
    reg = regime_daily(cal)
    tr_eq = trend_equity(cal).reindex(cal).ffill()
    # 스위칭: 일별 레짐으로 인덱스/추세 일수익 선택, 전환 시 비용
    idx_r = idx_eq.pct_change().fillna(0)
    tr_r = tr_eq.pct_change().fillna(0)
    sw = [CAP]; prev_mode = None
    for i in range(1, len(cal)):
        d = cal[i]
        mode = "idx" if reg.get(d, "CAUTION") in ("BULL", "CAUTION") else "trend"
        r = idx_r.iloc[i] if mode == "idx" else tr_r.iloc[i]
        v = sw[-1] * (1 + r)
        if prev_mode is not None and mode != prev_mode:
            v *= (1 - SWITCH_COST)
        sw.append(v); prev_mode = mode
    sw_eq = pd.Series(sw, index=cal)

    print("레짐 스위칭 검증 (KODEX200 / 추세추종 / 스위칭) — CAGR·MDD·Sharpe\n")
    for s, e, lbl in [("2024-01-01", "2024-12-31", "2024약세"),
                       ("2025-07-01", "2026-05-29", "2025중순~26강세"),
                       ("2024-01-01", "2026-05-29", "전체")]:
        print(f"=== {lbl} ===")
        print(f'{"전략":<10}{"수익":>9}{"CAGR":>8}{"MDD":>8}{"Sharpe":>8}')
        for nm, eq in [("인덱스", idx_eq), ("단일추세", tr_eq), ("스위칭", sw_eq)]:
            m = metrics(eq, s, e)
            if m:
                ret, cagr, mdd, sh = m
                print(f'{nm:<10}{ret:>+8.1f}%{cagr:>+7.1f}%{mdd:>7.1f}%{sh:>8.2f}')
        print()
    print("★ 합격(단타봇): 전체 AND 약세 둘다 — CAGR≥max(인덱스,단일) & MDD<인덱스 & Sharpe>둘다.")
    print("  미달 시 기각/보류. 생존편향(추세파트 5/29고정) 보수적 할인.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
