# -*- coding: utf-8 -*-
"""BRAIN BULL 인정 비대칭 완화 백테스트 — 7/4 진단('방어 유능·공세 무능')의 처방 검증.

진단: 현행 BULL = close>MA20 AND rv_pct<50 이 강세 초입 고변동을 CAUTION으로 묶어
  상단 미참여(-40%p). RV20은 상방/하방 변동성을 구분 못 하는 게 구조 원인.
처방 후보: BULL 인정 완화 변형들. ★방어(BEAR/CRISIS 규칙)는 절대 불변 — 검증된 유능 구간.

방법: 6/30 backtest_regime_gated_allocation.py 방법론 계승(1일 시프트 lookahead 차단·
  합성 2x 일일복리·전환비용 0.1%). 정책 고정: BULL→레버2x, CAUTION→지수1x, BEAR/CRISIS→현금.

★결과 (2026-07-07, 상세: data/backtest/brain_bull_relax_report.md):
  - V0 현행: CAGR 9.8%(2018~) — B&H 14.8%에 대패. 진단 확정(BULL일 26%뿐).
  - V3b 하방변동<60: CAGR 18.9%·MDD -39.2%·Sharpe 0.84 — B&H를 3지표 모두 상회.
    폭락창(-12.3%)·2022 약세년(+4.0%) 방어 V0과 동일 보존. 26상반기 +92.9%(V0 +50.4%).
  - 민감도: rv 50~85·drv 40~70 그리드 매끈(스파이크 없음) — 완화 방향 견고.
  - 채택 권장: V3b (BULL = close>MA20 AND 하방변동 252일 백분위<60). 적용은 퐝가님 결정.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

CSV = PROJECT_ROOT / "data" / "kospi_index.csv"
FEE = 0.001


def load_kospi() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").set_index("Date")
    d = d[["close"]].dropna()
    return d[d["close"] > 0]


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    close = d["close"]
    f = pd.DataFrame(index=d.index)
    f["close"] = close
    f["ma20"] = close.rolling(20).mean()
    f["ma60"] = close.rolling(60).mean()
    ret = close.pct_change()
    f["ret_20d"] = close.pct_change(20) * 100
    rv = ret.rolling(20).std() * np.sqrt(252) * 100
    f["rv_pct"] = (rv.rolling(252, min_periods=60).rank(pct=True) * 100).fillna(50)
    # 하방변동만(상승 변동성은 벌점 제외) — RV의 상/하방 무구분이 비대칭의 구조 원인
    dn = ret.where(ret < 0, 0.0)
    drv = dn.rolling(20).std() * np.sqrt(252) * 100
    f["drv_pct"] = (drv.rolling(252, min_periods=60).rank(pct=True) * 100).fillna(50)
    return f


def classify(f: pd.DataFrame, bull_cond: pd.Series) -> pd.Series:
    """BEAR/CRISIS 불변. close>MA20 구간만 bull_cond로 BULL/CAUTION 분기."""
    reg = pd.Series("CRISIS", index=f.index)
    reg[(f["close"] <= f["ma20"]) & (f["close"] > f["ma60"])] = "BEAR"
    above = f["close"] > f["ma20"]
    reg[above & ~bull_cond] = "CAUTION"
    reg[above & bull_cond] = "BULL"
    reg[f["ma60"].isna()] = "NEUTRAL"
    return reg


def run_policy(d: pd.DataFrame, reg: pd.Series) -> pd.Series:
    ret = d["close"].pct_change().fillna(0.0)
    asset_ret = {"INDEX": ret, "LEV": 2.0 * ret, "CASH": pd.Series(0.0, index=d.index)}
    target = reg.shift(1).fillna("NEUTRAL").map(
        {"BULL": "LEV", "CAUTION": "INDEX", "BEAR": "CASH", "NEUTRAL": "CASH", "CRISIS": "CASH"}
    ).fillna("CASH")
    strat = pd.Series(0.0, index=d.index)
    for a in ("INDEX", "LEV", "CASH"):
        strat[target == a] = asset_ret[a][target == a]
    switch = target != target.shift(1)
    return strat - switch.astype(float) * FEE


def metrics(strat: pd.Series) -> dict:
    eq = (1 + strat).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0,
            "mdd": (eq / eq.cummax() - 1).min(),
            "sharpe": strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0,
            "eq": eq}


def window_ret(strat: pd.Series, a: str, b: str) -> float:
    s = strat.loc[a:b]
    return ((1 + s).prod() - 1) * 100 if len(s) else float("nan")


def _print_variant(name, d, f, cond):
    reg = classify(f, cond(f))
    strat = run_policy(d, reg)
    m = metrics(strat)
    print(f"  {name:28s} CAGR {m['cagr']*100:+6.1f}%  MDD {m['mdd']*100:6.1f}%  "
          f"Sharpe {m['sharpe']:4.2f}  BULL일 {(reg == 'BULL').mean()*100:4.1f}%  "
          f"26상반기 {window_ret(strat, '2026-01-02', '2026-06-19'):+6.1f}%  "
          f"폭락창 {window_ret(strat, '2026-06-22', '2026-07-06'):+5.1f}%  "
          f"22약세 {window_ret(strat, '2022-01-01', '2022-12-31'):+5.1f}%")
    return m


def main() -> int:
    d_full = load_kospi()
    f_full = build_features(d_full)

    variants = {
        "V0 현행(rv<50)": lambda f: f["rv_pct"] < 50,
        "V1 rv<70": lambda f: f["rv_pct"] < 70,
        "V2 rv무시(MA20위=BULL)": lambda f: pd.Series(True, index=f.index),
        "V3 하방변동<50": lambda f: f["drv_pct"] < 50,
        "V3b 하방변동<60 ★권장": lambda f: f["drv_pct"] < 60,
        "V4 rv<50 OR 모멘텀+5%": lambda f: (f["rv_pct"] < 50) | (f["ret_20d"] > 5),
        "V5 정배열(MA20>MA60)": lambda f: f["ma20"] > f["ma60"],
    }

    for start in ["2018-01-01", "2021-01-01"]:
        d = d_full[d_full.index >= start]
        f = f_full.loc[d.index]
        bh = metrics(d["close"].pct_change().fillna(0.0))
        lev = metrics(2.0 * d["close"].pct_change().fillna(0.0))
        print(f"\n===== 구간 {start[:4]}~ =====")
        print(f"  {'기준: Buy&Hold':28s} CAGR {bh['cagr']*100:+6.1f}%  MDD {bh['mdd']*100:6.1f}%  Sharpe {bh['sharpe']:4.2f}")
        print(f"  {'기준: 무지성레버2x':28s} CAGR {lev['cagr']*100:+6.1f}%  MDD {lev['mdd']*100:6.1f}%  Sharpe {lev['sharpe']:4.2f}")
        for name, cond in variants.items():
            _print_variant(name, d, f, cond)

    # 민감도 그리드 (체리피킹 방지)
    print("\n===== 임계 민감도 (CAGR/MDD/Sharpe · 2018~ | 2021~) =====")
    for label, col, ths in [("rv", "rv_pct", [50, 60, 70, 80]), ("drv", "drv_pct", [45, 50, 55, 60, 70])]:
        for th in ths:
            row = f"  {label}<{th:2d} |"
            for start in ["2018-01-01", "2021-01-01"]:
                d = d_full[d_full.index >= start]
                f = f_full.loc[d.index]
                m = metrics(run_policy(d, classify(f, f[col] < th)))
                row += f" {m['cagr']*100:+5.1f}% {m['mdd']*100:5.1f}% {m['sharpe']:.2f} |"
            print(row)

    # 연도별 (B&H vs V0 vs V3b)
    print("\n===== 연도별 (2018~) =====")
    d = d_full[d_full.index >= "2018-01-01"]
    f = f_full.loc[d.index]
    eqs = {"B&H": metrics(d["close"].pct_change().fillna(0.0))["eq"],
           "V0": metrics(run_policy(d, classify(f, f["rv_pct"] < 50)))["eq"],
           "V3b": metrics(run_policy(d, classify(f, f["drv_pct"] < 60)))["eq"]}
    for yr in sorted(set(d.index.year)):
        row = f"  {yr}"
        for n, eq in eqs.items():
            e = eq[eq.index.year == yr]
            if len(e) < 2:
                row += f"  {n}:    — "
                continue
            prev = eq[eq.index < e.index[0]]
            base = prev.iloc[-1] if len(prev) else 1.0
            row += f"  {n}:{(e.iloc[-1] / base - 1) * 100:+6.1f}%"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
