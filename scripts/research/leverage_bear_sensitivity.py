"""(A-2) 약세장 검증 적대적 재검증 — 기각 확정 전 2개 변수 (사장님/메인AI 6/2).

1차 결과: 회복포함 2년 구간서 손절(D)이 buyhold보다 최종·MDD 둘 다 나쁨(휩쏘 120~276%).
기각 확정 전 내 검증을 적대적으로 의심:
  Q1. 기간 cherry — 회복까지 봐서 buyhold 유리. 약세장 바닥에서 끊으면 D가 계좌 지키나?
  Q2. 파라미터 — 재진입 빡빡/트레일 타이트가 휩쏘 키웠나? 완화하면 D 살아나나?
SOXX ×2 합성(488080 대용) 중심. look-ahead 0. 비용 0.1%. 시드 1천만.
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
import FinanceDataReader as fdr

SWITCH = 0.001
SEED = 10_000_000


def load_soxx2():
    df = fdr.DataReader("SOXX", "2021-01-01", "2023-12-31")
    r = 2 * df["Close"].astype(float).pct_change().fillna(0)
    price = 100 * (1 + r).cumprod()
    return price, r, df["Volume"].astype(float)


def run(price, r, vol, mode, start, end, trail=0.08, ma_p=20, reentry="strict", daily=-0.06):
    ma5 = price.rolling(5).mean(); maN = price.rolling(ma_p).mean(); maN_up = maN > maN.shift(5)
    idx = [d for d in price.index if start <= d <= end]
    v = 1.0; prev_h = None; curve = []
    nt = True; eh = None; pv = None
    exits = 0; reent = 0; whip = 0.0; last_ep = None; cash = 0
    avoided = 0.0; missed = 0.0
    for d in idx:
        th = nt
        holding = True if mode == "A" else th
        rd = r.get(d, 0)
        if not holding:
            cash += 1
            if rd < 0: avoided += rd
            else: missed += rd
        v *= (1 + (rd if holding else 0))
        if prev_h is not None and holding != prev_h: v *= (1 - SWITCH)
        curve.append(v); prev_h = holding
        p = price.get(d); m5 = ma5.get(d); mN = maN.get(d); upN = bool(maN_up.get(d, False)); vd = vol.get(d)
        if mode in ("C", "D"):
            if th:
                eh = p if eh is None else max(eh, p)
                ex = p < mN
                if mode == "D":
                    ex = ex or (eh and p <= eh * (1 - trail)) or (rd <= daily)
                if ex:
                    nt = False; eh = None; exits += 1; last_ep = p
            else:
                if reentry == "strict":
                    volup = (pv is not None and not np.isnan(vd) and vd > pv)
                    re = (p > m5) and (p > mN) and (rd >= 0.02) and volup and upN
                else:  # loose: 20선 회복 + 양봉만
                    re = (p > mN) and (rd > 0)
                if re:
                    nt = True; eh = p; reent += 1
                    if last_ep and p > last_ep: whip += (p - last_ep) / last_ep
        pv = vd
    eq = pd.Series(curve, index=idx)
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return dict(final=SEED * eq.iloc[-1], ret=(eq.iloc[-1] - 1) * 100, mdd=mdd,
                exits=exits, whip=whip * 100, cash=cash / len(idx) * 100,
                avoided=avoided * 100, missed=missed * 100)


def main() -> int:
    p, r, vol = load_soxx2()
    print("#" * 70)
    print("# (A-2) 약세장 검증 적대적 재검증 — SOXX×2 합성 (488080 대용)")
    print("#" * 70)

    # ── Q1: 기간 분할 ──────────────────────────────────────
    print("\n[Q1] 기간 cherry 검증 — 회복 제외하고 약세장 바닥에서 끊으면?")
    periods = [
        ("하락만 고점~바닥 21.12~22.10", pd.Timestamp("2021-12-01"), pd.Timestamp("2022-10-31")),
        ("회복포함 전체 21.6~23.6", pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-30")),
    ]
    for nm, s, e in periods:
        print(f"\n  ● {nm}  (자산 {(p.loc[s:e].iloc[-1]/p.loc[s:e].iloc[0]-1)*100:+.0f}%)")
        print(f'    {"전략":<22}{"최종평가금":>13}{"수익%":>8}{"MDD%":>7}{"손절":>5}{"휩쏘%":>7}')
        for mode, lab in [("A", "A buyhold"), ("C", "C 추세추종"), ("D", "D 풀손절")]:
            x = run(p, r, vol, mode, s, e)
            print(f'    {lab:<22}{x["final"]:>13,.0f}{x["ret"]:>+7.0f}%{x["mdd"]:>6.0f}%{x["exits"]:>5}{x["whip"]:>6.0f}%')

    # ── Q2: 파라미터 민감도 (전체기간 D) ───────────────────
    print("\n[Q2] 파라미터 민감도 — 휩쏘 줄이면 D 살아나나? (전체 21.6~23.6, D전략)")
    s, e = pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-30")
    a = run(p, r, vol, "A", s, e)
    print(f'    {"파라미터":<30}{"수익%":>8}{"MDD%":>7}{"손절":>5}{"휩쏘%":>7}{"이탈%":>7}')
    print(f'    {"(A buyhold 기준)":<30}{a["ret"]:>+7.0f}%{a["mdd"]:>6.0f}%{"-":>5}{"-":>7}{"0%":>7}')
    configs = [
        ("base 트레일-8/재진입strict/20선", dict(trail=0.08, ma_p=20, reentry="strict")),
        ("트레일완화 -20%", dict(trail=0.20, ma_p=20, reentry="strict")),
        ("재진입완화 20선+양봉만", dict(trail=0.08, ma_p=20, reentry="loose")),
        ("장기추세 60일선", dict(trail=0.08, ma_p=60, reentry="strict")),
        ("60선+트레일-20%+재진입완화", dict(trail=0.20, ma_p=60, reentry="loose")),
    ]
    for lab, kw in configs:
        x = run(p, r, vol, "D", s, e, **kw)
        win = "★A초과" if x["ret"] > a["ret"] else ""
        print(f'    {lab:<30}{x["ret"]:>+7.0f}%{x["mdd"]:>6.0f}%{x["exits"]:>5}{x["whip"]:>6.0f}%{x["cash"]:>6.0f}% {win}')

    print("\n" + "#" * 70)
    print("# 판정: Q1서 '하락만' 구간 D가 A를 크게 이기면 = 룰은 '버티기 불가능한")
    print("#       폭락 방어용'. Q2서 어떤 파라미터도 전체기간 A 못 넘으면 = 강세복원")
    print("#       장기엔 buyhold 우위 확정. 둘 다 보고 사장님 판단.")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
