"""A(고르는 기계) vs B(올라타되 리스크관리) 정면 비교 — 사장님 6/1.

A는 13번 검증으로 변별력0 확정(참조 1줄). B의 여러 형태를 인덱스와 비교:
  B0 인덱스 buy&hold (순수 베타, 기준)
  B1 대형주10 동일가중 월리밸 (분산 베타)
  B2 인덱스 + 트레일링스탑 -15%/-20% (고점대비 하락 시 현금, 회복=close>ma20 재진입)
  B3 인덱스 + 변동성타겟 (실현변동성 상위40%면 다음날 50%만 노출)
평가: 수익·CAGR·MDD·Sharpe. ★2025.6~2026.5. look-ahead 0. 전환비용 0.48%.
사장님 선택지 = 수익최대(B0) vs MDD최소(B2/B3) 트레이드오프를 숫자로.
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

CAP = 100_000_000
SWITCH = 0.0048
BUY_SLIP, SELL_SLIP, SELL_TAX = 0.0015, 0.0015, 0.0018


def kospi():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    return k


def metrics(eq, s, e):
    eq = eq[(eq.index >= pd.Timestamp(s)) & (eq.index <= pd.Timestamp(e))].dropna()
    if len(eq) < 20:
        return None
    eq = eq / eq.iloc[0]
    ret = (eq.iloc[-1] - 1) * 100
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    cagr = (eq.iloc[-1] ** (365 / days) - 1) * 100 if eq.iloc[-1] > 0 else -100
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return ret, cagr, mdd, sharpe


def b0(k):
    return k["close"] / k["close"].iloc[0]


def b2_trail(k, trail):
    r = k["close"].pct_change().fillna(0)
    ma20 = k["close"].rolling(20).mean()
    invested = True; peak = k["close"].iloc[0]; v = [1.0]; prev = True
    for i in range(1, len(k)):
        px = k["close"].iloc[i]
        if invested:
            peak = max(peak, px)
            if px <= peak * (1 - trail):
                invested = False
        else:
            if px > ma20.iloc[i]:
                invested = True; peak = px
        ret = r.iloc[i] if prev else 0.0
        nv = v[-1] * (1 + ret)
        if invested != prev:
            nv *= (1 - SWITCH)
        v.append(nv); prev = invested
    return pd.Series(v, index=k.index)


def b3_voltarget(k):
    lr = np.log(k["close"] / k["close"].shift(1))
    rv = lr.rolling(20).std()
    rvp = rv.rolling(252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    r = k["close"].pct_change().fillna(0)
    v = [1.0]; prev_w = 1.0
    for i in range(1, len(k)):
        w = 0.5 if (not pd.isna(rvp.iloc[i]) and rvp.iloc[i] >= 0.6) else 1.0
        nv = v[-1] * (1 + r.iloc[i] * prev_w)
        if w != prev_w:
            nv *= (1 - SWITCH * abs(w - prev_w))
        v.append(nv); prev_w = w
    return pd.Series(v, index=k.index)


def b1_largecap(k, cal, n=10):
    avg = {}
    store = {}
    asof = pd.Timestamp("2025-05-30")
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if "trading_value" not in df.columns:
            continue
        sub = df[df.index <= asof].tail(60)
        if len(sub) >= 30:
            avg[Path(f).stem] = sub["trading_value"].mean(); store[Path(f).stem] = df["close"]
    top = [c for c, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:n]]
    C = pd.DataFrame({c: store[c] for c in top}).reindex(cal).ffill()
    # 동일가중 buy&hold (시작 시점 매수, 월 리밸 동일가중)
    rets = C.pct_change().fillna(0)
    eqw = rets.mean(axis=1)  # 매일 동일가중 평균수익 ≈ 월리밸 근사
    return (1 + eqw).cumprod()


def main() -> int:
    k = kospi()
    cal = list(k.index)
    s, e = "2025-06-01", "2026-05-29"
    series = {
        "B0 인덱스 단독": b0(k),
        "B1 대형주10 동일가중": b1_largecap(k, cal, 10),
        "B2a 인덱스+트레일-15%": b2_trail(k, 0.15),
        "B2b 인덱스+트레일-20%": b2_trail(k, 0.20),
        "B3 인덱스+변동성타겟": b3_voltarget(k),
    }
    print("A(고르는 기계) vs B(올라타되 리스크관리) — 2025.6~2026.5\n")
    print(f'{"전략":<22}{"수익":>9}{"CAGR":>8}{"MDD":>8}{"Sharpe":>8}')
    for nm, eq in series.items():
        m = metrics(eq, s, e)
        if m:
            ret, cagr, mdd, sh = m
            print(f'{nm:<22}{ret:>+8.1f}%{cagr:>+7.1f}%{mdd:>7.1f}%{sh:>8.2f}')
    print(f'{"─"*55}')
    print(f'{"A 대형주모멘텀(참조)":<22}{"변별력 IC≈0, t<2 = 인덱스 못이김(13번 확정)":<0}')
    print("\n★ 수익 최대 = ? / MDD 최소 = ? / Sharpe(위험조정) 최고 = ? 를 보고 사장님이 성향 선택.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
