"""분할 매수표 백테스트 — 일괄 vs 조정분할 vs 시간정액 (× C60 스위치).

사장님 분할 매수표 검증(2026-06-03): thesis(상승) 전제 buyhold 분할 + 약세전환 C60.
풀매수 X(확신100%불가) → 분할 비대칭(맞으면 담고/조정에 평단↓/약세전환 C60 방어).

전략:
  A lump_buyhold : 첫날 전액, 약세 무시 (기준선)
  B lump_c60     : BULL 전액 + 60선 이탈 청산 / 복귀 재매수
  C split_dip    : 조정분할(T1 30% 즉시 / 직전고점 -5,-10,-15% 추가) + C60 청산
  D split_time   : 시간정액(BULL 중 매주 20%×5) + C60 청산

대상: 005930 ×2 합성(삼성 단일레버 대용 — 2022 약세장 실데이터 보유). 488080 보조.
look-ahead 0: 국면(60선)·조정 신호는 전일 종가 → 당일 적용. 비용 0.1%/거래. 시드 1.0.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import FinanceDataReader as fdr

COST = 0.001
TRANCHE_C = [0.30, 0.20, 0.20, 0.30]        # T1~T4 비중
DIP_TRIG = [None, -0.05, -0.10, -0.15]      # T1 즉시 / T2~T4 직전고점 대비 조정
TIME_STEP_DAYS = 5                          # 시간정액: 5거래일마다
TIME_TRANCHE = 0.20                         # 시간정액: 20%씩 5회


def prepare(ticker: str, lev: float):
    df = fdr.DataReader(ticker, "2021-01-01", "2026-06-02")
    close = df["Close"].astype(float)
    ma60 = close.rolling(60).mean()
    r1 = close.pct_change().fillna(0.0)
    lev_price = (1.0 + lev * r1).cumprod()       # 합성 레버 가격 (시작 1.0)
    bull_sig = (close > ma60).shift(1).fillna(False)  # 전일 신호 → 당일 (look-ahead 0)
    return lev_price, bull_sig


def _mdd(curve: list[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return worst


def sim(lev_price: pd.Series, bull_sig: pd.Series, strategy: str, start, end) -> dict:
    idx = [d for d in lev_price.index if start <= d <= end]
    if not idx:
        return {}
    shares = 0.0
    cash = 1.0
    invested_nominal = 0.0   # 누적 투입 명목 (평단/투입률용)
    tranches = 0
    peak_price = None        # 조정 측정용 (BULL 진입 후 최고가)
    last_step_i = None       # 시간정액 직전 매수 인덱스
    curve = []
    trades = 0

    for i, d in enumerate(idx):
        p = float(lev_price[d])
        bull = bool(bull_sig[d])

        def buy(frac: float):
            nonlocal shares, cash, invested_nominal, trades
            amt = min(frac, cash)
            if amt <= 1e-9:
                return
            shares += amt / p
            cash -= amt + amt * COST
            invested_nominal += amt
            trades += 1

        def sell_all():
            nonlocal shares, cash, invested_nominal, tranches, peak_price, last_step_i, trades
            if shares > 1e-12:
                cash += shares * p * (1 - COST)
                shares = 0.0
                trades += 1
            invested_nominal = 0.0
            tranches = 0
            peak_price = None
            last_step_i = None

        if strategy == "A":  # 일괄 buyhold, 약세 무시
            if i == 0:
                buy(1.0)

        elif strategy == "B":  # 일괄 + C60
            if bull and shares <= 1e-12:
                buy(1.0)
                peak_price = p
            elif not bull and shares > 1e-12:
                sell_all()

        elif strategy == "C":  # 조정 분할 + C60
            if bull:
                if tranches == 0:
                    buy(TRANCHE_C[0])
                    tranches = 1
                    peak_price = p
                else:
                    peak_price = max(peak_price, p)
                    # 직전 고점 대비 조정 트리거 (급락 시 여러 트랜치 동시)
                    while tranches < 4 and p <= peak_price * (1.0 + DIP_TRIG[tranches]):
                        buy(TRANCHE_C[tranches])
                        tranches += 1
            else:
                if shares > 1e-12:
                    sell_all()

        elif strategy == "D":  # 시간 정액 + C60
            if bull:
                if tranches < 5 and (last_step_i is None or i - last_step_i >= TIME_STEP_DAYS):
                    buy(TIME_TRANCHE)
                    tranches += 1
                    last_step_i = i
            else:
                if shares > 1e-12:
                    sell_all()

        curve.append(shares * p + cash)

    final = curve[-1]
    avg_invested = invested_nominal  # 종료 시점 미청산 포지션 명목
    return {
        "strategy": strategy,
        "final_return_pct": round((final - 1.0) * 100, 1),
        "mdd_pct": round(_mdd(curve) * 100, 1),
        "trades": trades,
        "end_invested_frac": round(min(avg_invested, 1.0), 2),
    }


STRAT_KR = {
    "A": "일괄buyhold",
    "B": "일괄+C60",
    "C": "조정분할+C60",
    "D": "시간정액+C60",
}


def run_block(title: str, ticker: str, lev: float, start: str, end: str):
    lev_price, bull_sig = prepare(ticker, lev)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    print(f"\n[{title}] {ticker} ×{lev:g}  ({start} ~ {end})")
    print(f"  {'전략':14s} {'수익%':>9} {'MDD%':>8} {'거래':>5} {'종료투입율':>8}")
    for strat in ("A", "B", "C", "D"):
        r = sim(lev_price, bull_sig, strat, s, e)
        if not r:
            continue
        print(f"  {STRAT_KR[strat]:14s} {r['final_return_pct']:>9} {r['mdd_pct']:>8} "
              f"{r['trades']:>5} {r['end_invested_frac']:>8}")


def main():
    print("=" * 64)
    print("분할 매수표 백테스트 (일괄 vs 조정분할 vs 시간정액 × C60)")
    print("  비대칭 가설: 강세장=일괄우위(기회비용) / 약세장=분할·C60 방어")
    print("=" * 64)
    # 삼성 ×2 합성 — 강세장 & 2022 약세장 실데이터
    run_block("강세장 삼성×2", "005930", 2.0, "2025-06-01", "2026-06-02")
    run_block("약세장 삼성×2(2022 실데이터)", "005930", 2.0, "2022-01-01", "2022-12-31")
    # 488080 실가격(이미 2배 ETF) — 강세장만 (2024-10 상장)
    run_block("강세장 488080", "488080", 1.0, "2025-06-01", "2026-06-02")
    print("\n" + "=" * 64)
    print("해석: A=다투입(강세최고/약세최악) / C·D=분할방어 / B=올인+스위치")
    print("=" * 64)


if __name__ == "__main__":
    main()
