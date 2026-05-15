"""Phase 9: 인버스 ETF 시그널 백테스트 (과거 약세장 검증)

가설:
- KOSPI 1일 -2%↓ 폭락일 + 외인 5일 누적 매도 -3조원↓ → KODEX 200선물인버스2X 매수
- D+1, D+3, D+5 수익률 측정
- 적중률 60%+ 도달 시 자동매매 활성화 가능

데이터:
- KOSPI 지수 (data/kospi_index.csv)
- 외인 일별 매매 (investor_daily.db)
- KODEX 200선물인버스2X (252670) — FDR

기간: 최근 6개월~1년
출력:
- data/backtest/phase9_inverse_signals.parquet
- data/backtest/phase9_report.md
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sqlite3
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import FinanceDataReader as fdr

OUT_DIR = PROJECT_ROOT / "data" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KOSPI_CSV = PROJECT_ROOT / "data" / "kospi_index.csv"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"

INVERSE_TICKER = "252670"  # KODEX 200선물인버스2X
LOOKBACK_DAYS = 365


def load_kospi() -> pd.DataFrame:
    df = pd.read_csv(KOSPI_CSV, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret_1d"] = df["close"].pct_change() * 100
    df["ret_5d"] = df["close"].pct_change(5) * 100
    return df


def load_foreign_5d_cum() -> pd.DataFrame:
    """시장 전체 외인 5일 누적 매도 추출"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT date, SUM(net_val) as foreign_net
           FROM investor_daily
           WHERE investor='외국인' AND date >= ?
           GROUP BY date ORDER BY date""",
        ((date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d"),),
    ).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["date_compact", "foreign_net"])
    df["Date"] = pd.to_datetime(df["date_compact"]).dt.strftime("%Y-%m-%d")
    df["foreign_eok"] = df["foreign_net"] / 1e8
    df["foreign_5d_eok"] = df["foreign_eok"].rolling(5, min_periods=1).sum()
    return df[["Date", "foreign_eok", "foreign_5d_eok"]]


def load_inverse_price() -> pd.DataFrame:
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    df = fdr.DataReader(INVERSE_TICKER, start, end)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[["Date", "Close", "Volume"]]


def main():
    print("=" * 70)
    print("Phase 9: 인버스 ETF (252670) 시그널 백테스트")
    print("=" * 70)

    kospi = load_kospi()
    foreign = load_foreign_5d_cum()
    inverse = load_inverse_price()
    print(f"[load] KOSPI {len(kospi)}일, 외인 {len(foreign)}일, 인버스 {len(inverse)}일")

    # join
    df = kospi.merge(foreign, on="Date", how="left").merge(
        inverse.rename(columns={"Close": "inv_close", "Volume": "inv_vol"}), on="Date", how="left"
    )
    df["inv_ret_1d"] = df["inv_close"].pct_change() * 100
    df["inv_ret_3d"] = df["inv_close"].pct_change(3) * 100
    df["inv_ret_5d"] = df["inv_close"].pct_change(5) * 100

    # forward returns (매수 시점 D+0 → D+N)
    for n in [1, 3, 5]:
        df[f"fwd_d{n}"] = df["inv_close"].shift(-n) / df["inv_close"] * 100 - 100

    # 시그널 분류
    df["signal_kospi_drop"] = df["ret_1d"] <= -2.0  # KOSPI 1일 -2%↓
    df["signal_kospi_5d"] = df["ret_5d"] <= -3.0     # KOSPI 5일 -3%↓
    df["signal_foreign"] = df["foreign_5d_eok"] <= -30000  # 외인 5일 -3조↓
    df["signal_combined"] = df["signal_kospi_drop"] & df["signal_foreign"]
    df["signal_either"] = df["signal_kospi_drop"] | df["signal_foreign"]

    df.to_parquet(OUT_DIR / "phase9_inverse_signals.parquet", index=False)

    # 측정
    def measure(sub, label):
        m = {"label": label, "n": len(sub)}
        for n in [1, 3, 5]:
            v = sub[f"fwd_d{n}"].dropna()
            if len(v) >= 3:
                m[f"d{n}_avg"] = v.mean()
                m[f"d{n}_hit"] = (v > 0).mean() * 100
                m[f"d{n}_n"] = len(v)
        return m

    results = [
        measure(df.dropna(subset=["fwd_d1"]), "전체 (대조군)"),
        measure(df[df["signal_kospi_drop"]], "KOSPI 1일 -2%↓"),
        measure(df[df["signal_kospi_5d"]], "KOSPI 5일 -3%↓"),
        measure(df[df["signal_foreign"]], "외인 5일 -3조↓"),
        measure(df[df["signal_combined"]], "KOSPI -2% AND 외인 -3조 ⭐"),
        measure(df[df["signal_either"]], "KOSPI -2% OR 외인 -3조"),
    ]

    print()
    print(f"{'시그널':<35} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10} {'D+3 hit':<10}")
    print("-" * 100)
    for r in results:
        if r.get("d1_avg") is None:
            print(f"{r['label']:<35} {r['n']:<8} (부족)")
            continue
        print(
            f"{r['label']:<35} {r['n']:<8} "
            f"{r['d1_avg']:+.2f}%   {r['d1_hit']:.1f}%      "
            f"{r.get('d3_avg', 0):+.2f}%   {r.get('d3_hit', 0):.1f}%"
        )

    # 통과 케이스 (combined)
    cases = df[df["signal_combined"]].dropna(subset=["fwd_d3"])
    print(f"\n=== combined 시그널 발생일 {len(cases)}건 ===")
    if len(cases) > 0:
        sample = cases[["Date", "ret_1d", "foreign_5d_eok", "inv_close", "fwd_d1", "fwd_d3", "fwd_d5"]].copy()
        print(sample.head(20).round(2).to_string(index=False))

    # 보고서
    out = OUT_DIR / "phase9_report.md"
    lines = [
        "# Phase 9: 인버스 ETF (252670) 시그널 백테스트",
        "",
        f"**기간**: 최근 {LOOKBACK_DAYS}일",
        f"**대상**: KODEX 200선물인버스2X (252670)",
        f"**입력**: KOSPI {len(kospi)}일 + 외인 매매 {len(foreign)}일 + 인버스 가격 {len(inverse)}일",
        "",
        "## 시그널 비교",
        "",
        "| 시그널 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 |",
        "|--------|--------|---------|---------|---------|---------|",
    ]
    for r in results:
        if r.get("d1_avg") is None:
            lines.append(f"| {r['label']} | {r['n']} | - | - | - | - |")
            continue
        lines.append(
            f"| {r['label']} | {r['n']} | "
            f"{r['d1_avg']:+.2f}% | {r['d1_hit']:.1f}% | "
            f"{r.get('d3_avg', 0):+.2f}% | {r.get('d3_hit', 0):.1f}% |"
        )

    lines += [
        "",
        "## combined 시그널 발생일 (KOSPI -2% AND 외인 -3조)",
        "",
        f"총 {len(cases)}건",
        "",
        "| 일자 | KOSPI 1일 | 외인5d (억) | 인버스 종가 | D+1 | D+3 | D+5 |",
        "|------|---------|-----------|----------|-----|-----|-----|",
    ]
    if len(cases) > 0:
        for _, r in cases.iterrows():
            d1 = f"{r['fwd_d1']:+.2f}%" if pd.notna(r['fwd_d1']) else "-"
            d3 = f"{r['fwd_d3']:+.2f}%" if pd.notna(r['fwd_d3']) else "-"
            d5 = f"{r['fwd_d5']:+.2f}%" if pd.notna(r['fwd_d5']) else "-"
            lines.append(
                f"| {r['Date']} | {r['ret_1d']:+.2f}% | {r['foreign_5d_eok']:,.0f} | "
                f"{int(r['inv_close']):,} | {d1} | {d3} | {d5} |"
            )

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")


if __name__ == "__main__":
    main()
