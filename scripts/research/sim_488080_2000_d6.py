"""2000만원 투입 시뮬 — 488080 buyhold vs C60(strict) D-6개월 (사장님 6/2).

2025-12-01 투입 ~ 2026-06-02. C60 strict 룰(검증 v2.1.1과 동일) + 거래비용 0.1%.
look-ahead 0(종가 신호→다음날 적용). 60일선 워밍업 위해 데이터는 2025-08부터.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from pykrx import stock

SEED = 20_000_000
SW = 0.001
START = pd.Timestamp("2025-12-01")
END = pd.Timestamp("2026-06-02")


def main() -> int:
    raw = stock.get_market_ohlcv("20250801", "20260602", "488080")
    close = raw["종가"].astype(float); vol = raw["거래량"].astype(float)
    close.index = pd.to_datetime(close.index); vol.index = pd.to_datetime(vol.index)
    ma5 = close.rolling(5).mean(); ma60 = close.rolling(60).mean(); ma60_up = ma60 > ma60.shift(5)
    r = close.pct_change()
    days = [d for d in close.index if START <= d <= END]

    d0 = days[0]
    state = "HOLD" if close[d0] > ma60[d0] else "CASH"
    c60 = SEED; bh = SEED; prev = close[d0]
    peak_c = SEED; peak_b = SEED; mdd_c = 0.0; mdd_b = 0.0
    exits = 0; reents = 0; cash = 0; last_ep = None; whip = 0
    c60_series = {}; bh_series = {}
    for i, d in enumerate(days):
        if i > 0:
            ret = close[d] / prev - 1.0
            bh *= (1 + ret)
            if state == "HOLD":
                c60 *= (1 + ret)
            else:
                cash += 1
        c60_series[d] = c60; bh_series[d] = bh
        peak_c = max(peak_c, c60); peak_b = max(peak_b, bh)
        mdd_c = min(mdd_c, c60 / peak_c - 1); mdd_b = min(mdd_b, bh / peak_b - 1)
        # 당일 종가로 다음날 상태
        p = close[d]; m5 = ma5[d]; m60 = ma60[d]; up = bool(ma60_up[d]); vd = vol[d]
        pv = vol[days[i - 1]] if i > 0 else None; rd = r[d]
        if state == "HOLD":
            if p <= m60:
                state = "CASH"; exits += 1; last_ep = p; c60 *= (1 - SW)
        else:
            volup = pv is not None and vd > pv
            re = (p > m5) and (p > m60) and (rd >= 0.02) and volup and up
            if re:
                state = "HOLD"; reents += 1; c60 *= (1 - SW)
                if last_ep and p > last_ep:
                    whip += 1
        prev = close[d]

    asset = (close[days[-1]] / close[d0] - 1) * 100
    print("=" * 60)
    print(f"  2,000만원 시뮬 — 488080 (TIGER 반도체TOP10 레버리지)")
    print(f"  투입 {days[0].date()} ~ {days[-1].date()}  (488080 자체 {asset:+.1f}%)")
    print(f"  현재 C60 상태: {state}")
    print("=" * 60)
    print(f'  {"전략":<22}{"최종 평가금":>16}{"수익률":>9}{"최대낙폭":>9}{"최저점 평가금":>16}')
    bh_ret = (bh / SEED - 1) * 100; c_ret = (c60 / SEED - 1) * 100
    bh_low = SEED * peak_b / SEED * (1 + mdd_b)  # = peak_b*(1+mdd_b)
    bh_low = peak_b * (1 + mdd_b); c_low = peak_c * (1 + mdd_c)
    print(f'  {"그냥 들고 있기":<22}{bh:>14,.0f}원{bh_ret:>+8.1f}%{mdd_b*100:>8.1f}%{bh_low:>14,.0f}원')
    print(f'  {"C60 추세추종":<22}{c60:>14,.0f}원{c_ret:>+8.1f}%{mdd_c*100:>8.1f}%{c_low:>14,.0f}원')
    print(f'\n  손절 {exits}회 / 재진입 {reents}회 / 현금대기 {cash}일 / 휩쏘 {whip}회')
    print(f'  거래비용 0.1%/회 반영. C60이 buyhold 대비 수익 {c_ret-bh_ret:+.1f}%p / 낙폭 {(mdd_c-mdd_b)*100:+.1f}%p')

    # 월별 평가금 추이
    print("\n  ── 월말 평가금 추이 ──")
    print(f'  {"시점":<12}{"그냥 들고":>16}{"C60":>16}')
    s_c = pd.Series(c60_series); s_b = pd.Series(bh_series)
    marks = s_b.resample("ME").last()
    for dt in marks.index:
        sub_b = s_b[s_b.index <= dt]; sub_c = s_c[s_c.index <= dt]
        if len(sub_b):
            lab = dt.strftime("%Y-%m")
            print(f'  {lab:<12}{sub_b.iloc[-1]:>14,.0f}원{sub_c.iloc[-1]:>14,.0f}원')
    print(f'  {"최종("+days[-1].strftime("%m/%d")+")":<12}{bh:>14,.0f}원{c60:>14,.0f}원')
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
