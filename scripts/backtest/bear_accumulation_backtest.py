# -*- coding: utf-8 -*-
"""역행 축적(BEAR Accumulation) 백테스트 — 판정: ❌ 기각 (역방향 유의).

가설: 약세장(KOSPI<MA20)에서 연기금·금투가 담는 종목이 이후 아웃퍼폼한다.
★결과 (2026-07-07, 상세: data/backtest/bear_accumulation_report.md):
  강도 상위20% D+20 초과 -2.22%p(t=-9.54), 2025/-2.48·2026/-1.90 두 구간 일관.
  전 변형(양축적/외인물량받기/금액상위) 전부 음(-). 순매도 하위도 음 → 조용한 종목이 최선.
  해석: 연기금 매수엔 기계적 리밸런싱(하락 시 매수)이 섞여 정보가 아님 — 약세장 기관 매수 = 낙하는 칼.
→ 시나리오 v1 확정 근거: 수급은 '약세장 선행지표' ❌ / '강세장 확인지표' ✅
  (같은 날 limit_up_precursor 백테스트의 '수급선행=강세 전용' 결론과 독립적으로 일치)

방법: 신호일=KOSPI<MA20일 한정. 축적=스마트머니(연기금+금투) 10일 누적(min 8일).
  진입 T+1 시가, D+5/10/20 종가. 대조군=같은 날 수급 커버드 종목(커버리지 교훈).
  유의성=날짜 클러스터 t.
"""
from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
MIN_TVAL = 5e8


def load_flow() -> pd.DataFrame:
    con = sqlite3.connect(DATA / "investor_flow" / "investor_daily.db")
    q = """SELECT date, ticker,
           SUM(CASE WHEN investor IN ('연기금','금융투자') THEN net_val ELSE 0 END)/1e8 AS smart_eok,
           SUM(CASE WHEN investor='외국인' THEN net_val ELSE 0 END)/1e8 AS foreign_eok
           FROM investor_daily GROUP BY date, ticker"""
    flow = pd.read_sql(q, con)
    con.close()
    flow["date"] = pd.to_datetime(flow["date"], format="%Y%m%d")
    flow["ticker"] = flow["ticker"].astype(str).str.zfill(6)
    return flow


def build_flow_features(flow: pd.DataFrame) -> pd.DataFrame:
    sm = flow.pivot_table(index="date", columns="ticker", values="smart_eok")
    fo = flow.pivot_table(index="date", columns="ticker", values="foreign_eok")
    out = pd.concat({
        "acc10": sm.rolling(10, min_periods=8).sum().stack(),
        "f10": fo.rolling(10, min_periods=8).sum().stack(),
    }, axis=1).reset_index()
    out.columns = ["date", "ticker", "acc10", "f10"]
    return out.dropna(subset=["acc10"])


def load_prices(tickers: set) -> pd.DataFrame:
    rows = []
    for p in sorted(PROCESSED.glob("*.parquet")):
        if p.stem not in tickers:
            continue
        try:
            df = pd.read_parquet(p, columns=["open", "close", "volume"])
        except Exception:
            continue
        df = df[df.index >= "2025-04-01"]
        if len(df) < 30:
            continue
        df = df.copy()
        o = df["open"].shift(-1)
        for h in (5, 10, 20):
            df[f"ret_d{h}"] = (df["close"].shift(-h) / o - 1) * 100
        df["tval20"] = (df["close"] * df["volume"]).rolling(20).mean()
        df["code"] = p.stem
        df = df[(df["close"] * df["volume"] >= MIN_TVAL) & (o > 0)]
        df.index.name = "date"
        rows.append(df.reset_index()[["date", "code", "ret_d5", "ret_d10", "ret_d20", "tval20"]])
    return pd.concat(rows, ignore_index=True)


def weak_days() -> pd.DatetimeIndex:
    k = pd.read_csv(DATA / "kospi_index.csv")
    k["Date"] = pd.to_datetime(k["Date"])
    k = k.sort_values("Date").set_index("Date")
    return k.index[k["close"] < k["close"].rolling(20).mean()]


def excess_t(sub: pd.DataFrame, ctrl: pd.DataFrame, col: str):
    ex = (sub.groupby("date")[col].mean() - ctrl.groupby("date")[col].mean()).dropna()
    if len(ex) < 3:
        return (np.nan, np.nan, len(ex))
    m, s = ex.mean(), ex.std(ddof=1)
    return (m, m / (s / np.sqrt(len(ex))) if s > 0 else np.nan, len(ex))


def main() -> int:
    flow = load_flow()
    feats = build_flow_features(flow)
    prices = load_prices(set(feats["ticker"].unique()))
    panel = prices.merge(feats, left_on=["date", "code"], right_on=["date", "ticker"], how="inner")
    panel = panel[panel["date"].isin(weak_days())].dropna(subset=["acc10", "ret_d5"])
    panel["intensity"] = panel["acc10"] * 1e8 / panel["tval20"].clip(lower=1e8)
    panel["int_rank"] = panel.groupby("date")["intensity"].rank(pct=True)
    panel["raw_rank"] = panel.groupby("date")["acc10"].rank(pct=True)
    print(f"약세일 패널: {panel['date'].min().date()} ~ {panel['date'].max().date()} | "
          f"{len(panel):,} 종목일 | 신호일수 {panel['date'].nunique()}")

    sigs = {
        "A 양축적 (acc10>0)": panel["acc10"] > 0,
        "B 강도 상위20% (양축적)": (panel["acc10"] > 0) & (panel["int_rank"] >= 0.8),
        "C 외인물량받기 (acc>0 & f10<0)": (panel["acc10"] > 0) & (panel["f10"] < 0),
        "D 금액 상위20%": panel["raw_rank"] >= 0.8,
        "E 강도+외인받기": (panel["acc10"] > 0) & (panel["int_rank"] >= 0.8) & (panel["f10"] < 0),
        "(반증) 순매도 하위20%": panel["raw_rank"] <= 0.2,
    }
    for name, mask in sigs.items():
        sub = panel[mask]
        out = f"  {name:26s} n={len(sub):6,}"
        for col in ["ret_d5", "ret_d10", "ret_d20"]:
            m, t, _ = excess_t(sub, panel, col)
            out += f" | {col[4:]:>3} {m:+5.2f}%p t={t:+5.2f}"
        print(out)

    print("\n구간 분해 (B 강도 상위20%, D+20 초과):")
    for a, b in [("2025-04-01", "2025-12-31"), ("2026-01-01", "2026-06-30")]:
        w = (panel["date"] >= a) & (panel["date"] <= b)
        sub, ctrl = panel[sigs["B 강도 상위20% (양축적)"] & w], panel[w]
        m, t, _ = excess_t(sub, ctrl, "ret_d20")
        print(f"  {a[:7]}~{b[:7]} n={len(sub):5,} 신호일 {sub['date'].nunique():3d} 초과 {m:+5.2f}%p t={t:+5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
