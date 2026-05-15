"""Phase 11: 인버스 ETF 정교화 백테스트 (단기 1~2일 + 엄격 조건)

Phase 9 결론: D+3 -3.95%, 적중률 28.6% → 부적합
재검증 가설:
- D+1만 노리고 D+2 강제 청산하면 어떤가? (인버스의 음의 복리 회피)
- 시그널 조건 엄격화: KOSPI -2.5%↓ AND 외인 -5조↓ AND 외인 1d -3000억↓
- 더 엄격하면 표본은 적지만 정확도 ↑?

핵심 차이 vs Phase 9:
- Phase 9: D+3 청산 (장기 보유 = 음의 복리 손실)
- Phase 11: D+1 청산 또는 D+2 강제 청산
- Phase 9: 시그널 OR/단순
- Phase 11: 엄격 AND 조합

기간: 최근 1년
출력:
- data/backtest/phase11_inverse_refined.parquet
- data/backtest/phase11_report.md
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
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma20_ret"] = (df["close"] / df["ma20"] - 1) * 100
    return df


def load_foreign() -> pd.DataFrame:
    """시장 전체 외인 일별 + 5일 누적"""
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
    print("Phase 11: 인버스 ETF (252670) 정교화 백테스트")
    print("=" * 70)

    kospi = load_kospi()
    foreign = load_foreign()
    inverse = load_inverse_price()
    print(f"[load] KOSPI {len(kospi)}일, 외인 {len(foreign)}일, 인버스 {len(inverse)}일")

    df = kospi.merge(foreign, on="Date", how="left").merge(
        inverse.rename(columns={"Close": "inv_close", "Volume": "inv_vol"}), on="Date", how="left"
    )

    # forward returns (D+1만 핵심, D+2 강제청산 가정)
    for n in [1, 2, 3]:
        df[f"fwd_d{n}"] = df["inv_close"].shift(-n) / df["inv_close"] * 100 - 100

    # 시그널 후보 (점진 엄격화)
    df["s_loose"] = df["ret_1d"] <= -2.0   # Phase 9와 동일
    df["s_mid"] = (df["ret_1d"] <= -2.0) & (df["foreign_5d_eok"] <= -30000)
    df["s_strict"] = (
        (df["ret_1d"] <= -2.5)
        & (df["foreign_5d_eok"] <= -50000)
        & (df["foreign_eok"] <= -3000)
    )
    # 추가: 시장 분위기 — MA20 -3%↓ (이미 약세장)
    df["s_bear"] = (
        (df["ret_1d"] <= -2.5)
        & (df["ma20_ret"] <= -3.0)
        & (df["foreign_5d_eok"] <= -50000)
    )
    # 추가: 5일 연속 약세 후 단발 폭락 (가속)
    df["s_accel"] = (
        (df["ret_1d"] <= -2.5)
        & (df["ret_5d"] <= -3.0)
        & (df["foreign_5d_eok"] <= -30000)
    )

    df.to_parquet(OUT_DIR / "phase11_inverse_refined.parquet", index=False)

    # 측정: D+1만 + D+2 강제청산 비교
    def measure(sub, label):
        m = {"label": label, "n": len(sub)}
        for n in [1, 2, 3]:
            v = sub[f"fwd_d{n}"].dropna()
            if len(v) >= 2:
                m[f"d{n}_avg"] = v.mean()
                m[f"d{n}_hit"] = (v > 0).mean() * 100
                m[f"d{n}_max"] = v.max()
                m[f"d{n}_min"] = v.min()
                m[f"d{n}_n"] = len(v)
        return m

    results = [
        measure(df.dropna(subset=["fwd_d1"]), "전체 (대조군)"),
        measure(df[df["s_loose"]], "Loose: KOSPI -2%↓ (Phase 9 동일)"),
        measure(df[df["s_mid"]], "Mid: KOSPI -2% AND 외인5d -3조"),
        measure(df[df["s_strict"]], "Strict: -2.5% AND -5조 AND 외인1d -3000억"),
        measure(df[df["s_bear"]], "Bear: -2.5% AND MA20 -3% AND -5조"),
        measure(df[df["s_accel"]], "Accel: -2.5% AND 5d -3% AND -3조"),
    ]

    print()
    print(f"{'시그널':<48} {'n':<6} {'D+1':<10} {'D+1 hit':<10} {'D+2':<10} {'D+2 hit':<10}")
    print("-" * 110)
    for r in results:
        if r.get("d1_avg") is None:
            print(f"{r['label']:<48} {r['n']:<6} (부족)")
            continue
        print(
            f"{r['label']:<48} {r['n']:<6} "
            f"{r['d1_avg']:+.2f}%   {r['d1_hit']:.1f}%      "
            f"{r.get('d2_avg', 0):+.2f}%   {r.get('d2_hit', 0):.1f}%"
        )

    # 최선 시그널 사례
    print()
    for sig_col, label in [("s_strict", "Strict"), ("s_bear", "Bear"), ("s_accel", "Accel")]:
        cases = df[df[sig_col]].dropna(subset=["fwd_d1"])
        print(f"=== {label} 시그널 발생일 {len(cases)}건 ===")
        if len(cases) > 0:
            sample = cases[
                ["Date", "ret_1d", "foreign_eok", "foreign_5d_eok", "inv_close", "fwd_d1", "fwd_d2"]
            ].copy()
            print(sample.head(15).round(2).to_string(index=False))
            print()

    # 보고서
    out = OUT_DIR / "phase11_report.md"
    lines = [
        "# Phase 11: 인버스 ETF (252670) 정교화 백테스트",
        "",
        f"**기간**: 최근 {LOOKBACK_DAYS}일",
        f"**대상**: KODEX 200선물인버스2X (252670)",
        "**전략**: D+1만 청산 또는 D+2 강제 청산 (음의 복리 회피)",
        "",
        "## Phase 9 vs Phase 11 차이",
        "",
        "- Phase 9: D+3 보유 → 음의 복리로 손실 누적",
        "- Phase 11: D+1 청산 (단발 폭락 활용) + D+2 강제 청산 (인버스 보유 최소화)",
        "",
        "## 시그널 비교",
        "",
        "| 시그널 | n | D+1 평균 | D+1 적중률 | D+1 최대 | D+1 최소 | D+2 평균 | D+2 적중률 |",
        "|--------|---|---------|---------|---------|---------|---------|---------|",
    ]
    for r in results:
        if r.get("d1_avg") is None:
            lines.append(f"| {r['label']} | {r['n']} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['label']} | {r['n']} | "
            f"{r['d1_avg']:+.2f}% | {r['d1_hit']:.1f}% | "
            f"{r.get('d1_max', 0):+.2f}% | {r.get('d1_min', 0):+.2f}% | "
            f"{r.get('d2_avg', 0):+.2f}% | {r.get('d2_hit', 0):.1f}% |"
        )

    # 결론
    best = None
    best_score = -999
    for r in results[1:]:  # 대조군 제외
        if r.get("d1_avg") is None:
            continue
        # 적중률 가중 점수: 적중률 * 평균
        score = r["d1_hit"] * 0.5 + r["d1_avg"] * 10
        if score > best_score and r["n"] >= 3:
            best_score = score
            best = r

    lines += ["", "## 결론", ""]
    if best:
        verdict = (
            "✅ 단기 활용 가능"
            if (best["d1_avg"] > 0.5 and best["d1_hit"] >= 55.0)
            else "⚠️ 여전히 부적합 — 자동매매 비권장"
        )
        lines += [
            f"- **최선 시그널**: {best['label']}",
            f"- **D+1 평균**: {best['d1_avg']:+.2f}%, **적중률**: {best['d1_hit']:.1f}%, 표본 {best['n']}건",
            f"- **판정**: {verdict}",
            "",
            "**권장 운영**:",
            "- 인버스 보유 최대 **2일** (음의 복리 회피)",
            "- 손절 **-3%** 즉시 청산",
            "- 시그널 약화 시(외인 매수 전환) 즉시 청산",
        ]
    else:
        lines += ["- 표본 부족 또는 모든 시그널 미달", "- **판정**: ❌ 인버스 자동매매 부적합"]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
