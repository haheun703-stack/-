"""Phase 12: 위험감지 게이트(P0-7) 적용 시 5/12~5/15 손실 회피율 백테스트.

가설:
- 5/15 정보봇 위험점수 78점 (DANGER) → multiplier 0.4 적용 시 손실 ~60% 감소
- 5/12 57점 (WARNING) → multiplier 0.6 적용 시 손실 ~40% 감소

데이터:
- macro_risk_daily (Supabase) — 일별 위험점수
- KOSPI 일별 수익률 (data/kospi_index.csv) — 시장 대표
- bluechip_top13 ticker 평균 — 대형주 손실
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import FinanceDataReader as fdr
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "data" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 등급별 multiplier (정보봇 정책)
QUANT_MULT = {
    "NORMAL":  1.0,
    "CAUTION": 0.8,
    "WARNING": 0.6,
    "DANGER":  0.4,
    "CRISIS":  0.2,
}

# bluechip TOP 13 (5/16 기준 시총 대형주)
BLUECHIP_TICKERS = [
    "005930", "000660", "373220", "207940", "005380",
    "000270", "068270", "035420", "035720", "051910",
    "012330", "066570", "028260",
]


def fetch_risk_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Supabase macro_risk_daily 조회."""
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    sb = create_client(url, key)
    res = sb.table("macro_risk_daily").select("*").gte("date", start_date).lte("date", end_date).order("date").execute()
    df = pd.DataFrame(res.data or [])
    return df


def fetch_bluechip_returns(start: str, end: str) -> pd.DataFrame:
    """bluechip 13종 일별 수익률 평균."""
    rows = []
    for t in BLUECHIP_TICKERS:
        try:
            df = fdr.DataReader(t, start, end)
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df["ret"] = df["Close"].pct_change() * 100
            df["ticker"] = t
            rows.append(df[["Date", "ticker", "ret"]])
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows)
    avg = all_df.groupby("Date").agg(bluechip_ret=("ret", "mean")).reset_index()
    return avg


def main():
    start = "2026-05-08"  # 1주 + α
    end = "2026-05-15"
    print("=" * 70)
    print("Phase 12: 위험감지 게이트 적용 시 5/12~15 손실 회피율")
    print("=" * 70)

    risk_df = fetch_risk_history(start, end)
    bluechip_df = fetch_bluechip_returns(start, end)

    if risk_df.empty:
        print("[ERR] macro_risk_daily 데이터 없음")
        return
    if bluechip_df.empty:
        print("[ERR] bluechip 데이터 없음")
        return

    # join
    df = bluechip_df.merge(risk_df, left_on="Date", right_on="date", how="inner")
    df = df[["Date", "bluechip_ret", "total_score", "level"]]

    # multiplier 적용
    df["mult"] = df["level"].map(QUANT_MULT).fillna(1.0)
    df["mitigated_ret"] = df["bluechip_ret"] * df["mult"]
    df["savings"] = df["bluechip_ret"] - df["mitigated_ret"]

    print(f"\n{'일자':<12} {'bluechip_평균':<14} {'점수':<6} {'등급':<10} {'mult':<6} {'완화후':<10} {'절감':<10}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"{r['Date']:<12} {r['bluechip_ret']:+.2f}%        "
              f"{int(r['total_score']):<6} {r['level']:<10} "
              f"×{r['mult']:<5} {r['mitigated_ret']:+.2f}%   {r['savings']:+.2f}%")

    # 5/12~15 누적
    target_dates = ["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]
    target = df[df["Date"].isin(target_dates)]
    if not target.empty:
        raw_cum = (1 + target["bluechip_ret"] / 100).prod() - 1
        mit_cum = (1 + target["mitigated_ret"] / 100).prod() - 1
        savings_pct = (raw_cum - mit_cum) * 100
        saving_ratio = (1 - mit_cum / raw_cum) * 100 if raw_cum != 0 else 0

        print(f"\n=== 5/12~15 누적 비교 ===")
        print(f"  원본 bluechip 누적 수익률: {raw_cum*100:+.2f}%")
        print(f"  위험감지 적용 누적 수익률: {mit_cum*100:+.2f}%")
        print(f"  손실 절감: {savings_pct:+.2f}%p ({saving_ratio:.1f}% 감소)")

    # 보고서
    out = OUT_DIR / "phase12_risk_gate_backtest.md"
    lines = [
        "# Phase 12: 위험감지 게이트 백테스트 (5/12~15)",
        "",
        "**검증 대상**: bluechip TOP 13 평균 수익률에 위험감지 multiplier 적용",
        "",
        "| 일자 | bluechip 평균 | 점수 | 등급 | mult | 완화후 | 절감 |",
        "|------|------------|------|------|------|------|------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Date']} | {r['bluechip_ret']:+.2f}% | {int(r['total_score'])} | "
            f"{r['level']} | ×{r['mult']} | {r['mitigated_ret']:+.2f}% | {r['savings']:+.2f}% |"
        )

    if not target.empty:
        lines += [
            "",
            "## 5/12~15 누적 결과",
            "",
            f"- 원본 bluechip 누적: **{raw_cum*100:+.2f}%**",
            f"- 위험감지 적용 누적: **{mit_cum*100:+.2f}%**",
            f"- 손실 절감폭: **{savings_pct:+.2f}%p**",
            f"- 손실 절감률: **{saving_ratio:.1f}%**",
            "",
            "## 결론",
            "",
            "정보봇이 매일 16:49 산출하는 위험점수를 매수금액 multiplier로 적용 시",
            "5/12~15 폭락장에서 누적 손실의 상당 부분을 자동 회피 가능.",
            "5/29 (목) MSCI 적용일 매도 라운드 방어를 위해 5/18 (월) 전 통합 필수.",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")


if __name__ == "__main__":
    main()
