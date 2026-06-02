"""(A-3) 60일선 robust 적대검증 — 과최적화/우연인지 3각 검증 (사장님 6/2).

A-2서 발견: SOXX×2 약세장 검증서 20일선→60일선 바꾸면 휩쏘 반감·buyhold 수익초과(+13% vs +5%)·MDD절반(-39%).
"너무 좋은 결과라 의심"([[feedback_adversarial_self_validation]]) → 3각 검증:
  검증1. 강세장(488080 2025.6~26.5) — 60선이 강세장 buyhold(+1768%) 얼마나 포기? (기회비용)
  검증2. SOXL 실3배 — 60선이 3배에서도 방어?
  검증3. SOXX×2 기간 밀기 — +13%/-39%가 위상 바꿔도 유지? (과최적화 경계)
전략 비교: A=buyhold / C20=추세추종20선 / C60=추세추종60선 / D60=60선+트레일-8%+단기-6%.
look-ahead 0. 비용 0.1%. 시드 1천만.
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
from pykrx import stock

SWITCH = 0.001
SEED = 10_000_000


def run(price, r, vol, mode, start, end, trail=0.08, ma_p=20, reentry="strict", daily=-0.06):
    ma5 = price.rolling(5).mean(); maN = price.rolling(ma_p).mean(); maN_up = maN > maN.shift(5)
    idx = [d for d in price.index if start <= d <= end]
    v = 1.0; prev_h = None; curve = []
    nt = True; eh = None; pv = None; exits = 0; whip = 0.0; last_ep = None; cash = 0
    for d in idx:
        th = nt
        holding = True if mode == "A" else th
        rd = r.get(d, 0)
        if not holding:
            cash += 1
        v *= (1 + (rd if holding else 0))
        if prev_h is not None and holding != prev_h:
            v *= (1 - SWITCH)
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
                else:
                    re = (p > mN) and (rd > 0)
                if re:
                    nt = True; eh = p
                    if last_ep and p > last_ep:
                        whip += (p - last_ep) / last_ep
        pv = vd
    eq = pd.Series(curve, index=idx)
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return dict(final=SEED * eq.iloc[-1], ret=(eq.iloc[-1] - 1) * 100, mdd=mdd, exits=exits, whip=whip * 100, cash=cash / len(idx) * 100)


def load_kr(code, s, e):
    raw = stock.get_market_ohlcv(s, e, code)
    price = raw["종가"].astype(float); vol = raw["거래량"].astype(float)
    price.index = pd.to_datetime(price.index); vol.index = pd.to_datetime(vol.index)
    return price, price.pct_change().fillna(0), vol


def load_us(ticker, synth=None):
    df = fdr.DataReader(ticker, "2020-09-01", "2024-03-31")
    r = df["Close"].astype(float).pct_change().fillna(0)
    if synth:
        r = synth * r; price = 100 * (1 + r).cumprod()
    else:
        price = df["Close"].astype(float)
    return price, r, df["Volume"].astype(float)


def table(title, price, r, vol, s, e, rows):
    print(f"\n  ● {title}  (자산 {(price.loc[s:e].iloc[-1]/price.loc[s:e].iloc[0]-1)*100:+.0f}%)")
    print(f'    {"전략":<24}{"최종평가금":>14}{"수익%":>9}{"MDD%":>7}{"손절":>5}{"휩쏘%":>7}')
    out = {}
    for lab, mode, kw in rows:
        x = run(price, r, vol, mode, s, e, **kw); out[lab] = x
        print(f'    {lab:<24}{x["final"]:>14,.0f}{x["ret"]:>+8.0f}%{x["mdd"]:>6.0f}%{x["exits"]:>5}{x["whip"]:>6.0f}%')
    return out


def main() -> int:
    print("#" * 72)
    print("# (A-3) 60일선 robust 적대검증 — 과최적화/우연 여부 3각 검증")
    print("#" * 72)

    ROWS = [
        ("A buyhold", "A", {}),
        ("C20 추세추종20선", "C", dict(ma_p=20)),
        ("C60 추세추종60선", "C", dict(ma_p=60)),
        ("D60 60선+트레일+단기", "D", dict(ma_p=60)),
    ]

    # 검증1: 강세장 488080
    print("\n[검증1] 강세장 488080 (2025.6~2026.5) — 60선의 강세장 기회비용")
    kp, kr, kv = load_kr("488080", "20240723", "20260529")
    r1 = table("488080 강세장", kp, kr, kv, pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29"), ROWS)
    a, c60 = r1["A buyhold"], r1["C60 추세추종60선"]
    print(f"    ▶ 60선은 강세장서 buyhold 대비 수익 {c60['ret']-a['ret']:+.0f}%p 포기 / MDD {c60['mdd']-a['mdd']:+.0f}%p")
    print(f"      = 강세장 기회비용 큼(예상). 핵심은 '약세장 방어가 이 포기를 정당화하나'")

    # 검증2: SOXL 3배
    print("\n[검증2] SOXL 실제 3배 (2021.6~2023.6) — 60선이 3배에서도 방어?")
    sp, sr, sv = load_us("SOXL")
    r2 = table("SOXL 3배", sp, sr, sv, pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-30"), ROWS)
    a2, c60_2 = r2["A buyhold"], r2["C60 추세추종60선"]
    print(f"    ▶ SOXL 60선: buyhold 대비 수익 {c60_2['ret']-a2['ret']:+.0f}%p / MDD {c60_2['mdd']-a2['mdd']:+.0f}%p")

    # 검증3: SOXX×2 기간 밀기
    print("\n[검증3] SOXX×2 기간 밀기 — +13%/-39%가 위상 바꿔도 유지? (과최적화 경계)")
    xp, xr, xv = load_us("SOXX", synth=2)
    print(f'    {"기간":<22}{"A수익":>8}{"A_MDD":>7}{"C60수익":>9}{"C60_MDD":>9}{"C60휩쏘":>8}{"판정":>10}')
    windows = [
        ("21.3~23.9", "2021-03-01", "2023-09-30"),
        ("21.6~23.6(base)", "2021-06-01", "2023-06-30"),
        ("21.9~23.12", "2021-09-01", "2023-12-29"),
        ("21.1~22.12(바닥근처종료)", "2021-01-04", "2022-12-30"),
        ("22.1~23.12", "2022-01-03", "2023-12-29"),
    ]
    for nm, ss, ee in windows:
        s, e = pd.Timestamp(ss), pd.Timestamp(ee)
        A = run(xp, xr, xv, "A", s, e); C = run(xp, xr, xv, "C", s, e, ma_p=60)
        win = "C60>A ★" if C["ret"] > A["ret"] else ("MDD방어" if C["mdd"] > A["mdd"] + 10 else "열위")
        print(f'    {nm:<22}{A["ret"]:>+7.0f}%{A["mdd"]:>6.0f}%{C["ret"]:>+8.0f}%{C["mdd"]:>8.0f}%{C["whip"]:>7.0f}%{win:>12}')

    print("\n" + "#" * 72)
    print("# 종합 판정:")
    print("#  · 검증1 강세장 기회비용 + 검증2 SOXL방어 + 검증3 기간 전반 C60>A 또는 MDD방어")
    print("#    → robust = 60선 추세추종 채택 → shadow")
    print("#  · 기간 밀면 결과 출렁/일부만 우위 → 과최적화, 채택 보류")
    print("#" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
