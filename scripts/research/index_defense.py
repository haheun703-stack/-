"""저빈도 방어 — "시장에 길게 올라타되, 진짜 위기 1번에만 빠진다" (사장님 6/1).

다 진 이유 = 자주 사고팔아서(무릎/포물선/스위칭 전부 휩소·전환비용에 깎임).
리버모어 "엉덩이로 돈 번다" = 추세 한 번 잡으면 끝까지. 그걸 시장(인덱스)에 적용.
  인덱스 단독: KOSPI buy&hold (기준)
  저빈도 방어: 기본 인덱스 보유. CRISIS(close<ma60 & 변동성 상위)에만 현금화,
              회복(close>ma20)에 재진입. 조정(CAUTION)엔 안 움직임 = 전환 최소.
look-ahead 0(레짐은 그날까지 rolling). 전환비용 왕복 0.48%.
★ 방어는 약세장(2024)이 있어야 보임 → 전체(2024~2026)로 MDD 검증, 강세는 분리.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

SWITCH_COST = 0.0048


def load_kospi():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    k["ma20"] = k["close"].rolling(20).mean()
    k["ma60"] = k["close"].rolling(60).mean()
    lr = np.log(k["close"] / k["close"].shift(1))
    k["rv"] = lr.rolling(20).std() * np.sqrt(252)
    k["rvp"] = k["rv"].rolling(252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    return k


def in_market_flags(k):
    """일별 시장 참여 여부 (look-ahead 0). CRISIS 진입=탈출, 회복=재진입. 저빈도."""
    invested = True; flags = {}
    for d, r in k.iterrows():
        if pd.isna(r["ma60"]):
            flags[d] = invested; continue
        crisis = (r["close"] < r["ma60"]) and (r["rvp"] >= 0.7 if not pd.isna(r["rvp"]) else False)
        recover = r["close"] > r["ma20"]
        if invested and crisis:
            invested = False
        elif (not invested) and recover:
            invested = True
        flags[d] = invested
    return pd.Series(flags)


def equity(k, flags=None):
    """flags=None → buy&hold. flags 있으면 저빈도 방어(전환 시 비용)."""
    r = k["close"].pct_change().fillna(0)
    if flags is None:
        eq = (1 + r).cumprod()
        return eq / eq.iloc[0]
    flags = flags.reindex(k.index).ffill().fillna(True)
    # 전일 플래그로 당일 참여 (look-ahead 0). 플래그 변할 때 비용.
    part = flags.shift(1).fillna(True)
    v = [1.0]; prev = True
    for i in range(1, len(k)):
        d = k.index[i]
        cur = bool(part.iloc[i])
        ret = r.iloc[i] if cur else 0.0
        nv = v[-1] * (1 + ret)
        if cur != prev:
            nv *= (1 - SWITCH_COST)
        v.append(nv); prev = cur
    return pd.Series(v, index=k.index)


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


def main() -> int:
    k = load_kospi()
    flags = in_market_flags(k)
    bh = equity(k, None)
    df = equity(k, flags)
    switches = int((flags.shift(1) != flags).sum())

    print("저빈도 방어 — 인덱스 길게 올라타되 진짜 위기 1번에만 탈출\n")
    print(f"  전체 기간 전환 횟수: {switches}회 (잦으면 실패)\n")
    for s, e, lbl in [("2024-01-01", "2024-12-31", "2024약세(방어 검증)"),
                       ("2025-07-01", "2026-05-29", "2025중순~26강세(수익)"),
                       ("2024-01-01", "2026-05-29", "전체")]:
        print(f"=== {lbl} ===")
        print(f'{"전략":<14}{"수익":>9}{"CAGR":>8}{"MDD":>8}{"Sharpe":>8}')
        for nm, eq in [("인덱스단독", bh), ("저빈도방어", df)]:
            m = metrics(eq, s, e)
            if m:
                ret, cagr, mdd, sh = m
                print(f'{nm:<14}{ret:>+8.1f}%{cagr:>+7.1f}%{mdd:>7.1f}%{sh:>8.2f}')
        print()
    print("★ 합격: 약세장 MDD를 인덱스보다 확실히 줄이고(방어) + 강세장 수익은 거의 유지(전환 적음).")
    print("  = 실전 봇의 길. 못 줄이거나 강세 수익 까먹으면 → 인덱스 단독이 답.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
