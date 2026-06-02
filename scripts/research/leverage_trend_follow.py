"""488080(반도체 일일2배 레버리지) 운용전략 4-way 비교 (코덱스 6/2 제안 검증).

코덱스 핵심: 488080은 단타X, '추세추종 봇'. US 극단신호 단독("폭락장만 피한다")은 약세장에서 위험.
추세이탈(20일선) + 트레일링(고점 -8~10%) + 단기(-6%) + 재진입(5/20선 회복+양봉+거래량) 숫자로 박아야.

4-way 비교 (2025.6~2026.5, 강세장):
  A buyhold (기준)
  B US극단신호 exit (메인AI 원래 — 외생신호, VIX2σ|SPY-2%|SOXX-3%일 현금)
  C 코덱스 추세추종 (내생신호 — ETF 20/5일선 + 트레일링 + -6% + 재진입 룰)
  D 결합 (C 추세추종 + B 극단일 추가회피 = 코덱스 룰 전체)
지표: 수익 / MDD / 샤프 / 거래횟수(휩소 측정) / KOSPI대비.
★강세장에선 추세추종(C)이 휩소로 B보다 못할 수 있음 — 손절의 진짜 가치는 약세장(미검증).
look-ahead 0: 당일 종가로 신호 판단 → 다음날 포지션 적용. 전환비용 0.1%.
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
from pykrx import stock

SWITCH = 0.001
TRAIL = 0.08      # 트레일링 고점 대비 -8%
DAILY_STOP = -0.06  # 단기 -6%
REENTRY_UP = 0.02   # 재진입 전일대비 +2%


def us_extreme(kdays):
    us = pd.read_parquet(PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet").sort_index()
    us["vix_z"] = us.get("vix_zscore", (us["vix_close"] - us["vix_close"].rolling(20).mean()) / us["vix_close"].rolling(20).std())
    ext = (us["vix_z"] >= 2.0) | (us["spy_ret_1d"] <= -0.02) | (us["soxx_ret_1d"] <= -0.03)
    sdf = pd.DataFrame({"usdate": us.index, "extreme": ext.values}).sort_values("usdate")
    m = pd.merge_asof(pd.DataFrame({"kdate": kdays}), sdf, left_on="kdate", right_on="usdate", direction="backward")
    return m.set_index("kdate")["extreme"].fillna(False)


def stats(eq, kos=None):
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    dr = eq.pct_change().dropna(); sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return ret, mdd, sh


def sim(days, lev, lev_r, vol, ma5, ma20, ma20_up, extreme, mode):
    """mode: A/B/C/D. extreme(US신호)=당일 회피(전일US라 당일 알 수 있음, look-ahead 0).
    추세신호=당일 종가 판단→다음날 적용(종가매매, look-ahead 0). 시작 보유."""
    v = 1.0; prev_h = None; curve = []; trades = 0
    next_trend = True; entry_high = None; prev_vol = None
    for d in days:
        ext = bool(extreme.get(d, False))
        trend_h = next_trend  # 추세상 보유여부(어제 종가 기준)
        if mode == "A":
            holding = True
        elif mode == "B":
            holding = not ext                 # 당일 극단신호 회피
        elif mode == "C":
            holding = trend_h                 # 순수 추세추종
        else:  # D = 추세 보유 AND 극단신호 아님
            holding = trend_h and not ext
        r = lev_r.get(d, 0) if holding else 0
        v *= (1 + r)
        if prev_h is not None and holding != prev_h:
            v *= (1 - SWITCH); trades += 1
        curve.append(v); prev_h = holding
        # 당일 종가로 추세 next_trend 갱신 (C,D만; entry_high는 추세상태 기준)
        p = lev.get(d, np.nan); m5 = ma5.get(d, np.nan); m20 = ma20.get(d, np.nan)
        up20 = bool(ma20_up.get(d, False)); rd = lev_r.get(d, 0); vd = vol.get(d, np.nan)
        if mode in ("C", "D"):
            if trend_h:
                entry_high = p if entry_high is None else max(entry_high, p)
                exit_sig = (p < m20) or (entry_high and p <= entry_high * (1 - TRAIL)) or (rd <= DAILY_STOP)
                next_trend = not exit_sig
                if exit_sig:
                    entry_high = None
            else:
                volup = (prev_vol is not None and not np.isnan(vd) and vd > prev_vol)
                reentry = (p > m5) and (p > m20) and (rd >= REENTRY_UP) and volup and up20
                next_trend = reentry
                if reentry:
                    entry_high = p
        prev_vol = vd
    return pd.Series(curve, index=days), trades


def main() -> int:
    raw = stock.get_market_ohlcv("20250401", "20260529", "488080")
    lev = raw["종가"].astype(float); vol = raw["거래량"].astype(float)
    lev.index = pd.to_datetime(lev.index); vol.index = pd.to_datetime(vol.index)
    ma5 = lev.rolling(5).mean(); ma20 = lev.rolling(20).mean()
    ma20_up = ma20 > ma20.shift(5)  # 20일선 상승 중 (5일전 대비)
    lev_r = lev.pct_change().fillna(0)

    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    kdays = [d for d in lev.index if S <= d <= E]
    extreme = us_extreme(kdays)

    ks = k[(k.index >= S) & (k.index <= E)]["close"]
    kos = (ks.iloc[-1] / ks.iloc[0] - 1) * 100

    print("=" * 70)
    print(f"488080 레버리지 운용전략 4-way (2025.6~2026.5, KOSPI {kos:+.0f}%, 강세장)")
    print("=" * 70)
    print(f'{"전략":<34}{"수익":>9}{"MDD":>7}{"샤프":>6}{"거래수":>6}')
    res = {}
    for mode, nm in [("A", "A buyhold (기준)"),
                     ("B", "B US극단신호 exit (외생)"),
                     ("C", "C 추세추종 (20/5선+트레일+-6%)"),
                     ("D", "D 결합 (추세추종+극단회피)")]:
        eq, tr = sim(kdays, lev, lev_r, vol, ma5, ma20, ma20_up, extreme, mode)
        st = stats(eq); res[mode] = (st[0], st[1], st[2], tr)
        print(f'{nm:<34}{st[0]:>+8.0f}%{st[1]:>6.0f}%{st[2]:>6.2f}{tr:>6}')

    print(f"\n  (참고) 같은기간 KOSPI {kos:+.0f}%")
    print("\n── 해석 (강세장 한정) ──")
    a, b, c, d = res["A"], res["B"], res["C"], res["D"]
    print(f"  • 추세추종(C) 거래 {c[3]}회 — 강세장 휩소 비용: buyhold 대비 수익 {c[0]-a[0]:+.0f}%p / MDD {c[1]-a[1]:+.0f}%p")
    print(f"  • US극단신호(B) vs 추세추종(C): 수익 {b[0]-c[0]:+.0f}%p, MDD {b[1]-c[1]:+.0f}%p")
    print(f"  • 결합(D): 수익 {d[0]:+.0f}% MDD {d[1]:.0f}% 샤프 {d[2]:.2f} 거래 {d[3]}회")
    best_mdd = min(res, key=lambda m: res[m][1])  # 가장 깊은 MDD
    safe_mdd = max(res, key=lambda m: res[m][1])   # 가장 얕은 MDD
    print(f"  • MDD 가장 얕음(약세장 보험) = {safe_mdd} ({res[safe_mdd][1]:.0f}%), 가장 깊음 = {best_mdd} ({res[best_mdd][1]:.0f}%)")
    print("\n★ 주의: 이 결론은 강세장 한정. 추세추종/손절의 진짜 가치(보험금)는")
    print("  추세하락이 지속되는 약세장에서만 증명됨 — 이 데이터엔 없음(다음 검증 필요).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
