"""Phase 6: ETF 외인+기관 시그널 백테스트

가설 (퐝가님 통찰 응용):
- ETF에 외인+기관 동시 매수 → ETF 가격 D+N 상승
- ETF 자체가 매매 대상 (퀀트봇 자동매매 전략과 일치)

데이터 (Supabase etf_investor_flow):
- 5/6 ~ 5/14 (7일 누적, 262건, 약 106개 ETF × 7일)
- 컬럼: date, ticker, name, foreign_net_amt, institution_net_amt, foreign_streak, institution_streak

분석:
1. ETF별 5일 누적 외인+기관 매수 계산
2. 시그널 분류:
   - stage1 (외+기 둘 다 양수) — 강한 상승 베팅
   - 외인만 매수
   - 기관만 매수
   - 둘 다 매도
3. ETF 가격으로 D+1, D+3, D+5 수익률 (FDR)

출력:
- data/backtest/phase6_etf_signals.parquet
- data/backtest/phase6_report.md
"""

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import FinanceDataReader as fdr
from supabase import create_client

OUT_DIR = PROJECT_ROOT / "data" / "backtest"


def fetch_etf_flow():
    """etf_investor_flow 전체 export"""
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    res = sb.table("etf_investor_flow").select("*").execute()
    df = pd.DataFrame(res.data)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[fetch] {len(df)}건, {df['ticker'].nunique()}개 ETF, {df['date'].min()} ~ {df['date'].max()}")
    return df


def add_5d_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """ticker별 5일 rolling 외+기 누적"""
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    df["fgn_5d"] = df.groupby("ticker")["foreign_net_amt"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["inst_5d"] = df.groupby("ticker")["institution_net_amt"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    return df


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """ETF 가격으로 D+1, D+3, D+5 수익률"""
    df = df.copy()
    tickers = df["ticker"].unique()
    price_cache = {}
    print(f"[FDR] {len(tickers)}개 ETF 가격 fetch...")
    for t in tickers:
        try:
            p = fdr.DataReader(t, "2026-04-20", "2026-05-21")
            if len(p) > 0:
                p = p.reset_index()
                p["Date"] = pd.to_datetime(p["Date"]).dt.strftime("%Y-%m-%d")
                price_cache[t] = p.set_index("Date")["Close"]
        except Exception:
            pass
    print(f"  → {len(price_cache)}개 ETF 가격 캐싱")

    rows = []
    for _, r in df.iterrows():
        t = r["ticker"]
        if t not in price_cache:
            continue
        px = price_cache[t]
        date = r["date"]
        if date not in px.index:
            continue
        c0 = px.loc[date]
        dates_after = px.index[px.index > date].tolist()
        rec = dict(r)
        rec["close_d0"] = c0
        for n in [1, 3, 5]:
            if len(dates_after) >= n:
                cn = px.loc[dates_after[n - 1]]
                rec[f"ret_d{n}"] = (cn - c0) / c0 * 100
            else:
                rec[f"ret_d{n}"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_signal(row):
    """5일 누적 외+기 기준 분류"""
    fgn = row["fgn_5d"]
    inst = row["inst_5d"]
    if fgn > 0 and inst > 0:
        return "dual_buy"
    if fgn > 0 and inst <= 0:
        return "fgn_only"
    if fgn <= 0 and inst > 0:
        return "inst_only"
    return "dual_sell"


def measure(sub, label):
    m = {"label": label, "n": len(sub)}
    for n in [1, 3, 5]:
        v = sub[f"ret_d{n}"].dropna()
        if len(v) >= 3:
            m[f"d{n}_avg"] = v.mean()
            m[f"d{n}_hit"] = (v > 0).mean() * 100
            m[f"d{n}_n"] = len(v)
    return m


def main():
    print("=" * 60)
    print("Phase 6: ETF 외인+기관 시그널 백테스트")
    print("=" * 60)

    df = fetch_etf_flow()
    df = add_5d_cumulative(df)
    df = add_forward_returns(df)
    print(f"[enriched] {len(df)}행 (D+1 valid)")

    df["signal"] = df.apply(classify_signal, axis=1)
    df.to_parquet(OUT_DIR / "phase6_etf_signals.parquet", index=False)

    # 카테고리 + 시그널 결과
    results = []
    for sig in ["dual_buy", "fgn_only", "inst_only", "dual_sell"]:
        results.append(measure(df[df["signal"] == sig], sig))

    print()
    print(f"{'시그널':<15} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10} {'D+5':<10}")
    print("-" * 70)
    for r in results:
        if r.get("d1_avg") is None:
            print(f"{r['label']:<15} {r['n']:<8} N/A")
            continue
        print(
            f"{r['label']:<15} {r['n']:<8} "
            f"{r['d1_avg']:+.2f}%   {r['d1_hit']:.1f}%      "
            f"{r.get('d3_avg', 0):+.2f}%   {r.get('d5_avg', 0):+.2f}%"
        )

    # 카테고리별 dual_buy 성과
    print()
    print("=== 카테고리별 dual_buy (외+기 둘 다 매수) ===")
    print(f"{'카테고리':<20} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10}")
    print("-" * 60)
    for cat in df["category"].dropna().unique():
        sub = df[(df["category"] == cat) & (df["signal"] == "dual_buy")]
        if len(sub) < 3:
            continue
        v1 = sub["ret_d1"].dropna()
        v3 = sub["ret_d3"].dropna()
        if len(v1) >= 3:
            print(
                f"{cat:<20} {len(sub):<8} "
                f"{v1.mean():+.2f}%   {(v1>0).mean()*100:.1f}%      "
                f"{v3.mean():+.2f}%" if len(v3) >= 3 else "..."
            )

    # 보고서 작성
    out = OUT_DIR / "phase6_report.md"
    lines = [
        "# Phase 6: ETF 외인+기관 시그널 백테스트",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**기간**: 2026-05-06 ~ 2026-05-14 (7일)",
        f"**입력**: {len(df):,}건",
        "",
        "## 시그널별 결과 (ETF 자체 D+N 수익률)",
        "",
        "| 시그널 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+5 평균 |",
        "|--------|--------|---------|---------|---------|---------|",
    ]
    for r in results:
        if r.get("d1_avg") is None:
            continue
        lines.append(
            f"| {r['label']} | {r['n']} | "
            f"{r['d1_avg']:+.2f}% | {r['d1_hit']:.1f}% | "
            f"{r.get('d3_avg', 0):+.2f}% | {r.get('d5_avg', 0):+.2f}% |"
        )

    lines += [
        "",
        "## dual_buy (외+기 둘 다 매수) 카테고리별",
        "",
        "| 카테고리 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 |",
        "|---------|--------|---------|---------|---------|",
    ]
    for cat in sorted(df["category"].dropna().unique()):
        sub = df[(df["category"] == cat) & (df["signal"] == "dual_buy")]
        if len(sub) < 3:
            continue
        v1 = sub["ret_d1"].dropna()
        v3 = sub["ret_d3"].dropna()
        if len(v1) >= 3:
            d3a = f"{v3.mean():+.2f}%" if len(v3) >= 3 else "-"
            lines.append(
                f"| {cat} | {len(sub)} | {v1.mean():+.2f}% | {(v1>0).mean()*100:.1f}% | {d3a} |"
            )

    lines += [
        "",
        "## 통과 ETF 샘플 (dual_buy 5/14)",
        "",
        "| ETF | 카테고리 | 외인5d | 기관5d | D+1 | D+3 |",
        "|-----|--------|------|-------|-----|-----|",
    ]
    sample = df[(df["signal"] == "dual_buy") & (df["date"] == "2026-05-14")].sort_values("inst_5d", ascending=False).head(20)
    for _, r in sample.iterrows():
        d1 = f"{r['ret_d1']:+.2f}%" if not pd.isna(r['ret_d1']) else "-"
        d3 = f"{r['ret_d3']:+.2f}%" if not pd.isna(r['ret_d3']) else "-"
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['category']} | "
            f"{r['fgn_5d']:+.0f} | {r['inst_5d']:+.0f} | {d1} | {d3} |"
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")
    print("\n[OK] Phase 6 완료")


if __name__ == "__main__":
    main()
