"""코덱스 KODEX 레버리지+인버스 헤지 주장 적대검증 (read-only 탐색).

코덱스 60일 "대충" 숫자 정밀 재현 + 2022 약세장 + 동적헤지(C60) 비교.
핵심 질문: 고정 인버스 헤지(30/50/80)가 강세장 착시인가, 아니면 진짜 안전구조인가?
- 고정 인버스 vs C60 동적 인버스(close<ma60일 때만 헤지 on) — 코덱스 본인 통찰
  "언제 붙이고 떼느냐가 핵심"을 실제로 검증.
- 강세장(60일) vs 약세장(2022) — 메모리: 강세장은 검증불가 환경.

별도 트랙(레버리지 운용). 매매 무관·실주문 0·저장 0(콘솔만).
사용: python -u -X utf8 scripts/explore_kodex_leverage_hedge.py
"""
from __future__ import annotations

import contextlib
import io
import logging

import pandas as pd


def load_close(ticker: str, start: str, end: str) -> pd.Series:
    """pykrx ETF/종목 종가. KRX 로그 노이즈 억제."""
    from pykrx import stock as krx
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = krx.get_market_ohlcv(start, end, ticker)
    finally:
        logging.disable(logging.NOTSET)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} empty")
    df.index = pd.to_datetime(df.index)
    return df["종가"].astype(float).rename(ticker)


def perf(close: pd.Series, label: str) -> tuple[float, float]:
    ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
    mdd = ((close / close.cummax() - 1).min()) * 100
    print(f"  {label:<26} {ret:+7.1f}% / MDD {mdd:6.1f}%")
    return round(ret, 1), round(mdd, 1)


def fixed_hedge(lev: pd.Series, inv: pd.Series, w: float) -> tuple[float, float]:
    """레버 100% + 인버스 w% 고정(일별 리밸런싱)."""
    lev_r = lev.pct_change().fillna(0)
    inv_r = inv.pct_change().fillna(0)
    port = lev_r * 1.0 + inv_r * (w / 100.0)
    cum = (1 + port).cumprod()
    ret = (cum.iloc[-1] - 1) * 100
    mdd = ((cum / cum.cummax() - 1).min()) * 100
    print(f"  레버100+인버스{int(w):<3}(고정) {ret:+7.1f}% / MDD {mdd:6.1f}%")
    return round(ret, 1), round(mdd, 1)


# (C60 동적 헤지는 main [2]에서 워밍업 구간 포함 inline 계산 — 워밍업 분리가 필요해 함수화하지 않음)

TICKERS = {"005930": "삼성전자", "000660": "SK하이닉스", "069500": "KODEX200",
           "122630": "KODEX레버리지", "114800": "KODEX인버스1x"}


def main():
    # ── 1) 최근 60거래일 (강세장) — 코덱스 숫자 재현 ──
    print("=" * 60)
    print("[1] 최근 60거래일 (강세장) — 코덱스 숫자 재현")
    d = {}
    for t in TICKERS:
        try:
            d[t] = load_close(t, "20260101", "20260609")
        except Exception as e:
            print(f"  {TICKERS[t]} 로드실패: {e}")
    for t, nm in TICKERS.items():
        if t in d:
            perf(d[t].iloc[-60:], nm)
    if "122630" in d and "114800" in d:
        lev = d["122630"].iloc[-61:]
        inv = d["114800"].iloc[-61:]
        print("  ── 인버스 고정 헤지 ──")
        for w in (30, 50, 80, 100):
            fixed_hedge(lev, inv, w)

    # ── 2) 2022 약세장 — 고정 vs C60 동적 ──
    print("=" * 60)
    print("[2] 2022 약세장 — 고정 인버스 vs C60 동적 (핵심 검증)")
    d22 = {}
    for t in ("122630", "114800", "069500"):
        try:
            d22[t] = load_close(t, "20211001", "20221231")  # ma60 워밍업 위해 10월부터
        except Exception as e:
            print(f"  {t} 로드실패: {e}")
    if "122630" in d22 and "114800" in d22:
        lev22 = d22["122630"].loc["2022-01-01":]
        inv22 = d22["114800"].loc["2022-01-01":]
        # 단독/지수
        perf(lev22, "KODEX레버리지 단독")
        if "069500" in d22:
            perf(d22["069500"].loc["2022-01-01":], "KODEX200(1x)")
        print("  ── 고정 헤지 ──")
        for w in (30, 50, 80, 100):
            fixed_hedge(lev22, inv22, w)
        print("  ── C60 동적 헤지 (ma60 워밍업 포함 계산) ──")
        lev_full = d22["122630"]
        inv_full = d22["114800"]
        for w in (30, 50, 80):
            # 워밍업 구간 포함 계산 후 2022만 평가
            ma60 = lev_full.rolling(60).mean()
            lev_r = lev_full.pct_change().fillna(0)
            inv_r = inv_full.pct_change().fillna(0)
            hedge_on = (lev_full.shift(1) < ma60.shift(1)).astype(float).fillna(0)
            port = (lev_r * 1.0 + inv_r * (w / 100.0) * hedge_on).loc["2022-01-01":]
            cum = (1 + port).cumprod()
            ret = (cum.iloc[-1] - 1) * 100
            mdd = ((cum / cum.cummax() - 1).min()) * 100
            on = int(hedge_on.loc["2022-01-01":].sum())
            print(f"  레버100+인버스{w:<3}(C60동적) {ret:+7.1f}% / MDD {mdd:6.1f}% · 헤지 {on}일 on")

    print("=" * 60)
    print("해석: 강세장(60일)은 인버스가 드래그(수익↓ MDD소폭↓). 약세장(2022)에서")
    print("고정 헤지 vs C60 동적 헤지 차이가 '타이밍이 핵심'인지 가른다.")


if __name__ == "__main__":
    main()
