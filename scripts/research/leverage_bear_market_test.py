"""(A) 약세장 대용 검증 — 레버리지 손절/재진입 룰이 계좌를 살리나? (사장님 6/2 지시).

488080(2024.7 상장)은 2022 반도체 약세장이 없음 → 대용 자산으로 검증:
  1. SOXX(반도체 ETF) 일일수익률 ×2 합성  ← 488080(2배)에 가장 직접 비교
  2. SOXL(실제 3배)  ← 스트레스 테스트 (3배 상품이라 변동성 큼, 해석 주의)
기간: 2021.6~2023.6 (반도체 고점 → 2022 약세장 바닥 → 회복 일부).

전략 (사장님 정의):
  A = buyhold (무방어)
  C = 추세추종 (20일선 종가이탈 청산 + 재진입). 트레일링/단기손절 없음
  D = 추세추종 + 트레일링(고점-8%) + 단기(-6%) 결합 (풀 다중손절)
재진입(C,D): 5일선&20일선 회복 + 전일 +2% + 거래량 증가 + 20일선 상승.

필수 지표: 최종수익금/MDD/최대손실회피/손절수/재진입수/재진입지연일/휩쏘비용/이탈기간/방어효과.
★핵심: buyhold 최대낙폭 / D가 막은 손실 / D가 놓친 반등 / 보험료 낼 가치 있었나.
look-ahead 0: 종가 신호→다음날 적용. 전환비용 0.1%. 시드 1천만원.
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
TRAIL = 0.08
DAILY_STOP = -0.06
REENTRY_UP = 0.02
START, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-30")


def load(ticker, synth_lev=None):
    df = fdr.DataReader(ticker, "2021-01-01", "2023-12-31")
    close = df["Close"].astype(float); vol = df["Volume"].astype(float)
    r = close.pct_change().fillna(0)
    if synth_lev:
        r = synth_lev * r
        price = 100 * (1 + r).cumprod()
    else:
        price = close.copy()
    return price, r, vol


def run(price, r, vol, mode):
    ma5 = price.rolling(5).mean(); ma20 = price.rolling(20).mean()
    ma20_up = ma20 > ma20.shift(5)
    idx = [d for d in price.index if START <= d <= END]
    v = 1.0; prev_h = None; curve = []
    next_trend = True; entry_high = None; prev_vol = None
    exits = []; reentries = []; cash_days = 0; cost_n = 0
    avoided_loss = 0.0; missed_rally = 0.0
    last_exit = None; reentry_delays = []; whipsaw = 0.0
    for i, d in enumerate(idx):
        trend_h = next_trend
        holding = True if mode == "A" else trend_h
        rd = r.get(d, 0)
        if not holding:
            cash_days += 1
            if rd < 0:
                avoided_loss += rd      # 현금이라 피한 손실(음수 누적)
            else:
                missed_rally += rd      # 현금이라 놓친 상승(양수 누적)
        ret = rd if holding else 0
        v *= (1 + ret)
        if prev_h is not None and holding != prev_h:
            v *= (1 - SWITCH); cost_n += 1
        curve.append(v); prev_h = holding
        p = price.get(d); m5 = ma5.get(d); m20 = ma20.get(d)
        up20 = bool(ma20_up.get(d, False)); vd = vol.get(d)
        if mode in ("C", "D"):
            if trend_h:
                entry_high = p if entry_high is None else max(entry_high, p)
                exit_sig = p < m20
                if mode == "D":
                    exit_sig = exit_sig or (entry_high and p <= entry_high * (1 - TRAIL)) or (rd <= DAILY_STOP)
                if exit_sig:
                    next_trend = False; entry_high = None
                    exits.append((i, d, p)); last_exit = (i, d, p)
            else:
                volup = (prev_vol is not None and not np.isnan(vd) and vd > prev_vol)
                reentry = (p > m5) and (p > m20) and (rd >= REENTRY_UP) and volup and up20
                if reentry:
                    next_trend = True; entry_high = p
                    reentries.append((i, d, p))
                    if last_exit is not None:
                        reentry_delays.append(i - last_exit[0])
                        if p > last_exit[2]:
                            whipsaw += (p - last_exit[2]) / last_exit[2]
        prev_vol = vd
    eq = pd.Series(curve, index=idx)
    final_amt = SEED * eq.iloc[-1]
    ret_pct = (eq.iloc[-1] - 1) * 100
    dd = (eq - eq.cummax()) / eq.cummax()
    mdd_pct = dd.min() * 100
    mdd_amt = (SEED * (eq.cummax() - eq)).max()  # peak→trough 금액
    sh = eq.pct_change().dropna()
    sharpe = sh.mean() / sh.std() * np.sqrt(252) if sh.std() > 0 else 0
    return dict(eq=eq, final=final_amt, ret=ret_pct, mdd=mdd_pct, mdd_amt=mdd_amt,
                sharpe=sharpe, exits=len(exits), reentries=len(reentries),
                delay=np.mean(reentry_delays) if reentry_delays else 0,
                cash_ratio=cash_days / len(idx) * 100, whipsaw=whipsaw * 100,
                avoided=avoided_loss * 100, missed=missed_rally * 100, cost_n=cost_n)


def report(name, price, r, vol, note):
    print("\n" + "=" * 72)
    print(f"  {name}   {note}")
    print(f"  검증기간 {START.date()}~{END.date()}  (자산 자체 {(price.loc[START:END].iloc[-1]/price.loc[START:END].iloc[0]-1)*100:+.0f}%)")
    print("=" * 72)
    R = {m: run(price, r, vol, m) for m in ("A", "C", "D")}
    a, c, d = R["A"], R["C"], R["D"]
    print(f'  {"전략":<26}{"최종평가금":>14}{"수익%":>8}{"MDD%":>7}{"MDD금액":>13}{"샤프":>6}')
    rows = [("A buyhold(무방어)", a), ("C 추세추종(20선)", c), ("D 풀손절(추세+트레일+급락)", d)]
    for nm, x in rows:
        print(f'  {nm:<26}{x["final"]:>14,.0f}{x["ret"]:>+7.0f}%{x["mdd"]:>6.0f}%{x["mdd_amt"]:>13,.0f}{x["sharpe"]:>6.2f}')
    print(f'\n  {"전략":<26}{"손절수":>6}{"재진입":>7}{"평균지연일":>10}{"이탈기간%":>9}{"휩쏘비용%":>9}')
    for nm, x in rows:
        print(f'  {nm:<26}{x["exits"]:>6}{x["reentries"]:>7}{x["delay"]:>10.1f}{x["cash_ratio"]:>8.0f}%{x["whipsaw"]:>8.0f}%')

    print(f"\n  ── ★ 보험료 분석 (사장님 핵심 질문) ──")
    print(f"  • buyhold가 맞은 최대 낙폭: {a['mdd']:.0f}%  ({a['mdd_amt']:,.0f}원 증발)")
    print(f"  • D가 현금구간에서 막은 손실: {d['avoided']:+.0f}%p  (피한 하락 누적)")
    print(f"  • D가 현금구간에서 놓친 반등: {d['missed']:+.0f}%p  (못 먹은 상승 누적)")
    net = d['avoided'] + d['missed']  # 막은손실(음수절댓값 이득) vs 놓친반등(비용)
    print(f"  • 순 타이밍효과(막은손실 - 놓친반등): {-d['avoided'] - d['missed']:+.0f}%p"
          + ("  → 보험가치 있음(손실회피>반등포기)" if (-d['avoided'] - d['missed']) > 0 else "  → ★보험료 손해(반등포기>손실회피)"))
    print(f"  • D vs buyhold:  MDD {d['mdd']-a['mdd']:+.0f}%p 방어 ({a['mdd_amt']-d['mdd_amt']:+,.0f}원 덜 잃음) / 최종수익금 {d['final']-a['final']:+,.0f}원")
    verdict = "채택 후보(MDD 크게 방어 + 수익손상 감내가능)" if (d['mdd'] - a['mdd'] > 10 and d['final'] > a['final'] * 0.5) else "재검토(방어 부족 or 수익손상 과도)"
    print(f"  ▶ 판정: {verdict}")
    return R


def main() -> int:
    soxx_p, soxx_r, soxx_v = load("SOXX", synth_lev=2)
    soxl_p, soxl_r, soxl_v = load("SOXL")
    print("#" * 72)
    print("# (A) 약세장 대용 검증 — 레버리지 손절룰이 계좌를 살리나 (시드 1천만원)")
    print("#" * 72)
    report("SOXX ×2 합성 (488080 대용, 일일2배)", soxx_p, soxx_r, soxx_v, "★메인 비교")
    report("SOXL 실제 3배 (스트레스 테스트)", soxl_p, soxl_r, soxl_v, "⚠️3배 상품, 변동성 큼")
    print("\n" + "#" * 72)
    print("# 종합: D가 buyhold MDD를 크게 줄이고 최종수익 손상이 과도하지 않으면")
    print("#       → 코덱스 룰 채택 → shadow forward → 488080 실전 추적")
    print("#" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
