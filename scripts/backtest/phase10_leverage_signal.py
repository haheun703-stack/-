"""Phase 10: 레버리지 ETF 시그널 백테스트 (강세장 검증)

가설:
- KOSPI 1일 +2%↑ OR 5일 +3%↑ + 외인 5일 누적 매수 +3조원↑ → KODEX 레버리지(122630) 매수
- D+1, D+3, D+5 수익률 측정
- 적중률 55%+ + D+3 평균 +1%+ 도달 시 자동매매 활성화 검토

데이터:
- KOSPI 지수 (data/kospi_index.csv)
- 외인 일별 매매 (investor_daily.db)
- KODEX 레버리지 (122630) — FDR

기간: 최근 1년
출력:
- data/backtest/phase10_leverage_signals.parquet
- data/backtest/phase10_report.md
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

LEVERAGE_TICKER = "122630"  # KODEX 레버리지
LOOKBACK_DAYS = 365


def load_kospi() -> pd.DataFrame:
    df = pd.read_csv(KOSPI_CSV, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret_1d"] = df["close"].pct_change() * 100
    df["ret_3d"] = df["close"].pct_change(3) * 100
    df["ret_5d"] = df["close"].pct_change(5) * 100
    return df


def load_foreign_5d_cum() -> pd.DataFrame:
    """시장 전체 외인 5일 누적 매수 추출"""
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


def load_leverage_price() -> pd.DataFrame:
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    df = fdr.DataReader(LEVERAGE_TICKER, start, end)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[["Date", "Close", "Volume"]]


def main():
    print("=" * 70)
    print("Phase 10: 레버리지 ETF (122630) 시그널 백테스트")
    print("=" * 70)

    kospi = load_kospi()
    foreign = load_foreign_5d_cum()
    lev = load_leverage_price()
    print(f"[load] KOSPI {len(kospi)}일, 외인 {len(foreign)}일, 레버리지 {len(lev)}일")

    # join
    df = kospi.merge(foreign, on="Date", how="left").merge(
        lev.rename(columns={"Close": "lev_close", "Volume": "lev_vol"}), on="Date", how="left"
    )
    df["lev_ret_1d"] = df["lev_close"].pct_change() * 100
    df["lev_ret_3d"] = df["lev_close"].pct_change(3) * 100
    df["lev_ret_5d"] = df["lev_close"].pct_change(5) * 100

    # forward returns (매수 시점 D+0 → D+N)
    for n in [1, 3, 5]:
        df[f"fwd_d{n}"] = df["lev_close"].shift(-n) / df["lev_close"] * 100 - 100

    # 시그널 분류
    df["signal_kospi_up_1d"] = df["ret_1d"] >= 2.0     # KOSPI 1일 +2%↑
    df["signal_kospi_up_5d"] = df["ret_5d"] >= 3.0     # KOSPI 5일 +3%↑
    df["signal_foreign_buy"] = df["foreign_5d_eok"] >= 30000  # 외인 5일 +3조↑
    df["signal_combined"] = df["signal_kospi_up_5d"] & df["signal_foreign_buy"]  # 강세 확정
    df["signal_either"] = df["signal_kospi_up_1d"] | df["signal_foreign_buy"]
    df["signal_strong"] = df["signal_kospi_up_1d"] & df["signal_foreign_buy"]  # 가장 강한 시그널

    df.to_parquet(OUT_DIR / "phase10_leverage_signals.parquet", index=False)

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
        measure(df[df["signal_kospi_up_1d"]], "KOSPI 1일 +2%↑"),
        measure(df[df["signal_kospi_up_5d"]], "KOSPI 5일 +3%↑"),
        measure(df[df["signal_foreign_buy"]], "외인 5일 +3조↑"),
        measure(df[df["signal_combined"]], "KOSPI 5d +3% AND 외인 +3조"),
        measure(df[df["signal_strong"]], "KOSPI 1d +2% AND 외인 +3조 ⭐"),
        measure(df[df["signal_either"]], "KOSPI 1d +2% OR 외인 +3조"),
    ]

    print()
    print(f"{'시그널':<38} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10} {'D+3 hit':<10}")
    print("-" * 100)
    for r in results:
        if r.get("d1_avg") is None:
            print(f"{r['label']:<38} {r['n']:<8} (부족)")
            continue
        print(
            f"{r['label']:<38} {r['n']:<8} "
            f"{r['d1_avg']:+.2f}%   {r['d1_hit']:.1f}%      "
            f"{r.get('d3_avg', 0):+.2f}%   {r.get('d3_hit', 0):.1f}%"
        )

    # 통과 케이스 (strong)
    cases = df[df["signal_strong"]].dropna(subset=["fwd_d3"])
    print(f"\n=== strong 시그널 발생일 {len(cases)}건 ===")
    if len(cases) > 0:
        sample = cases[["Date", "ret_1d", "foreign_5d_eok", "lev_close", "fwd_d1", "fwd_d3", "fwd_d5"]].copy()
        print(sample.head(20).round(2).to_string(index=False))

    # 보고서
    out = OUT_DIR / "phase10_report.md"
    lines = [
        "# Phase 10: 레버리지 ETF (122630) 시그널 백테스트",
        "",
        f"**기간**: 최근 {LOOKBACK_DAYS}일",
        f"**대상**: KODEX 레버리지 (122630)",
        f"**입력**: KOSPI {len(kospi)}일 + 외인 매매 {len(foreign)}일 + 레버리지 가격 {len(lev)}일",
        "",
        "## 시그널 비교",
        "",
        "| 시그널 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 | D+5 평균 | D+5 적중률 |",
        "|--------|--------|---------|---------|---------|---------|---------|---------|",
    ]
    for r in results:
        if r.get("d1_avg") is None:
            lines.append(f"| {r['label']} | {r['n']} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['label']} | {r['n']} | "
            f"{r['d1_avg']:+.2f}% | {r['d1_hit']:.1f}% | "
            f"{r.get('d3_avg', 0):+.2f}% | {r.get('d3_hit', 0):.1f}% | "
            f"{r.get('d5_avg', 0):+.2f}% | {r.get('d5_hit', 0):.1f}% |"
        )

    lines += [
        "",
        "## strong 시그널 발생일 (KOSPI 1d +2% AND 외인 +3조)",
        "",
        f"총 {len(cases)}건",
        "",
        "| 일자 | KOSPI 1일 | 외인5d (억) | 레버리지 종가 | D+1 | D+3 | D+5 |",
        "|------|---------|-----------|----------|-----|-----|-----|",
    ]
    if len(cases) > 0:
        for _, r in cases.iterrows():
            d1 = f"{r['fwd_d1']:+.2f}%" if pd.notna(r['fwd_d1']) else "-"
            d3 = f"{r['fwd_d3']:+.2f}%" if pd.notna(r['fwd_d3']) else "-"
            d5 = f"{r['fwd_d5']:+.2f}%" if pd.notna(r['fwd_d5']) else "-"
            lines.append(
                f"| {r['Date']} | {r['ret_1d']:+.2f}% | {r['foreign_5d_eok']:,.0f} | "
                f"{int(r['lev_close']):,} | {d1} | {d3} | {d5} |"
            )

    # 결론 자동 판정
    strong_r = next((r for r in results if r["label"].startswith("KOSPI 1d +2% AND")), None)
    if strong_r and strong_r.get("d3_avg") is not None:
        d3_avg = strong_r["d3_avg"]
        d3_hit = strong_r["d3_hit"]
        verdict = "✅ 자동매매 활성화 검토 가능" if (d3_avg >= 1.0 and d3_hit >= 55.0) else "❌ 자동매매 부적합 (조건 미달)"
        lines += [
            "",
            "## 결론",
            "",
            f"- strong 시그널 D+3 평균: **{d3_avg:+.2f}%** (목표 +1%+)",
            f"- strong 시그널 D+3 적중률: **{d3_hit:.1f}%** (목표 55%+)",
            f"- **판정**: {verdict}",
        ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")


if __name__ == "__main__":
    main()
