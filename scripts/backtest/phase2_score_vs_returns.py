"""Phase 2: SECTOR FIRE 점수 vs D+N 수익률 상관관계 측정

입력:
- data/backtest/sector_picks_raw.parquet (Phase 1 산출, 1046행)
- stock_data_daily/*_{ticker}.csv (37컬럼 OHLCV)

분석:
1. 각 picks 행마다 D+1, D+3, D+5, D+10 수익률 계산
2. buy_score vs D+N 수익률 Pearson correlation
3. buy_grade별 (BUY/WATCH) 적중률 (D+N > 0 비율)
4. 점수 구간별 평균 수익률
5. 외+기 동시매수 vs 단독매수 비교

출력:
- data/backtest/phase2_returns.parquet
- data/backtest/phase2_report.md
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUT_DIR = PROJECT_ROOT / "data" / "backtest"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"


def load_ohlcv(ticker: str) -> pd.DataFrame | None:
    """ticker로 stock_data_daily CSV 찾아 로드"""
    matches = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df.set_index("Date")


def compute_forward_returns(picks_df: pd.DataFrame) -> pd.DataFrame:
    """각 picks 행에 D+1, D+3, D+5, D+10 수익률 추가"""
    results = []
    ohlcv_cache = {}

    for _, row in picks_df.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        date = row["date"]

        if ticker not in ohlcv_cache:
            ohlcv_cache[ticker] = load_ohlcv(ticker)

        df_o = ohlcv_cache[ticker]
        if df_o is None or date not in df_o.index:
            continue

        try:
            close_d0 = df_o.loc[date, "Close"]
        except KeyError:
            continue

        dates_after = df_o.index[df_o.index > date].tolist()
        if len(dates_after) < 1:
            continue

        record = dict(row)
        record["close_d0"] = close_d0
        for n in [1, 3, 5, 10]:
            if len(dates_after) >= n:
                close_dn = df_o.loc[dates_after[n - 1], "Close"]
                ret = (close_dn - close_d0) / close_d0 * 100
                record[f"ret_d{n}"] = ret
            else:
                record[f"ret_d{n}"] = np.nan
        results.append(record)

    return pd.DataFrame(results)


def analyze(df: pd.DataFrame) -> dict:
    """상관관계 + 적중률 + 점수 구간별 평균"""
    metrics = {}

    # 1. Pearson correlation: buy_score vs ret_dN
    for n in [1, 3, 5, 10]:
        col = f"ret_d{n}"
        sub = df[["buy_score", col]].dropna()
        if len(sub) < 5:
            metrics[f"corr_d{n}"] = None
            continue
        corr = sub["buy_score"].corr(sub[col])
        metrics[f"corr_d{n}"] = corr
        metrics[f"n_d{n}"] = len(sub)

    # 2. buy_grade별 적중률 (D+1, D+3 > 0 비율)
    if "buy_grade" in df.columns:
        for grade in df["buy_grade"].dropna().unique():
            sub = df[df["buy_grade"] == grade]
            for n in [1, 3, 5]:
                col = f"ret_d{n}"
                valid = sub[col].dropna()
                if len(valid) < 3:
                    continue
                hit = (valid > 0).mean() * 100
                avg = valid.mean()
                metrics[f"grade_{grade}_d{n}_hit"] = hit
                metrics[f"grade_{grade}_d{n}_avg"] = avg
                metrics[f"grade_{grade}_d{n}_n"] = len(valid)

    # 3. 점수 구간별 평균 (>=40, 30-39, 20-29, <20)
    for label, lo, hi in [(">=40", 40, 999), ("30-39", 30, 39.99), ("20-29", 20, 29.99), ("<20", 0, 19.99)]:
        sub = df[(df["buy_score"] >= lo) & (df["buy_score"] <= hi)]
        for n in [1, 3, 5]:
            col = f"ret_d{n}"
            valid = sub[col].dropna()
            if len(valid) < 3:
                continue
            metrics[f"score_{label}_d{n}_avg"] = valid.mean()
            metrics[f"score_{label}_d{n}_n"] = len(valid)

    # 4. 외+기 동시매수 vs 단독매수
    if "fgn_5d" in df.columns and "inst_5d" in df.columns:
        dual = df[(df["fgn_5d"] > 0) & (df["inst_5d"] > 0)]
        fgn_only = df[(df["fgn_5d"] > 0) & (df["inst_5d"] <= 0)]
        inst_only = df[(df["fgn_5d"] <= 0) & (df["inst_5d"] > 0)]
        neither = df[(df["fgn_5d"] <= 0) & (df["inst_5d"] <= 0)]
        for label, sub in [("dual", dual), ("fgn_only", fgn_only), ("inst_only", inst_only), ("neither", neither)]:
            for n in [1, 3, 5]:
                col = f"ret_d{n}"
                valid = sub[col].dropna()
                if len(valid) < 3:
                    continue
                metrics[f"flow_{label}_d{n}_avg"] = valid.mean()
                metrics[f"flow_{label}_d{n}_hit"] = (valid > 0).mean() * 100
                metrics[f"flow_{label}_d{n}_n"] = len(valid)

    return metrics


def write_report(df: pd.DataFrame, metrics: dict):
    """phase2_report.md 작성"""
    out = OUT_DIR / "phase2_report.md"

    lines = [
        "# Phase 2: SECTOR FIRE 점수 vs 수익률 상관관계 보고서",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**기간**: 2026-04-25 ~ 2026-05-14 (20일)",
        f"**입력 picks**: {len(df):,}행",
        "",
        "## 1. Pearson Correlation (buy_score vs D+N 수익률)",
        "",
        "| D+N | Correlation | Sample |",
        "|-----|------------|--------|",
    ]
    for n in [1, 3, 5, 10]:
        corr = metrics.get(f"corr_d{n}")
        n_sample = metrics.get(f"n_d{n}", 0)
        corr_str = f"{corr:.4f}" if corr is not None else "N/A"
        lines.append(f"| D+{n} | {corr_str} | {n_sample} |")

    lines += [
        "",
        "**해석**:",
        "- |corr| < 0.1: 거의 무관",
        "- 0.1~0.3: 약한 상관",
        "- 0.3~0.5: 중간 상관",
        "- 0.5+: 강한 상관",
        "",
        "## 2. buy_grade별 적중률",
        "",
        "| Grade | D+1 적중률 | D+1 평균수익 | D+3 적중률 | D+3 평균수익 | Sample |",
        "|-------|---------|------------|---------|------------|--------|",
    ]
    for grade in ["BUY", "WATCH", "KILL"]:
        hit_d1 = metrics.get(f"grade_{grade}_d1_hit")
        avg_d1 = metrics.get(f"grade_{grade}_d1_avg")
        hit_d3 = metrics.get(f"grade_{grade}_d3_hit")
        avg_d3 = metrics.get(f"grade_{grade}_d3_avg")
        n = metrics.get(f"grade_{grade}_d1_n", 0)
        if hit_d1 is None:
            continue
        lines.append(f"| {grade} | {hit_d1:.1f}% | {avg_d1:+.2f}% | {hit_d3:.1f}% | {avg_d3:+.2f}% | {n} |")

    lines += [
        "",
        "## 3. 점수 구간별 평균 수익률",
        "",
        "| 점수 구간 | D+1 평균 | D+3 평균 | D+5 평균 | Sample (D+1) |",
        "|---------|---------|---------|---------|------------|",
    ]
    for label in [">=40", "30-39", "20-29", "<20"]:
        d1 = metrics.get(f"score_{label}_d1_avg")
        d3 = metrics.get(f"score_{label}_d3_avg")
        d5 = metrics.get(f"score_{label}_d5_avg")
        n = metrics.get(f"score_{label}_d1_n", 0)
        if d1 is None:
            continue
        lines.append(f"| {label} | {d1:+.2f}% | {d3:+.2f}% | {d5:+.2f}% | {n} |")

    lines += [
        "",
        "## 4. 외+기 동시매수 vs 단독매수 vs 무매수 ⭐",
        "",
        "| 유형 | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 | D+5 평균 | Sample |",
        "|-----|---------|---------|---------|---------|---------|--------|",
    ]
    for label in ["dual", "fgn_only", "inst_only", "neither"]:
        d1_avg = metrics.get(f"flow_{label}_d1_avg")
        d1_hit = metrics.get(f"flow_{label}_d1_hit")
        d3_avg = metrics.get(f"flow_{label}_d3_avg")
        d3_hit = metrics.get(f"flow_{label}_d3_hit")
        d5_avg = metrics.get(f"flow_{label}_d5_avg")
        n = metrics.get(f"flow_{label}_d1_n", 0)
        if d1_avg is None:
            continue
        label_kr = {"dual": "외+기 둘 다 매수", "fgn_only": "외인만 매수", "inst_only": "기관만 매수", "neither": "둘 다 매도"}[label]
        lines.append(f"| {label_kr} | {d1_avg:+.2f}% | {d1_hit:.1f}% | {d3_avg:+.2f}% | {d3_hit:.1f}% | {d5_avg:+.2f}% | {n} |")

    lines += [
        "",
        "## 5. 결론 + Phase 3 가설",
        "",
        "(데이터 보고 다음 세션에서 결정)",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out}")


def main():
    print("=" * 60)
    print("Phase 2: 점수 vs D+N 수익률 상관관계")
    print("=" * 60)

    picks = pd.read_parquet(OUT_DIR / "sector_picks_raw.parquet")
    print(f"[load] sector_picks_raw {len(picks)}행")

    print("[compute] forward returns (D+1, D+3, D+5, D+10)...")
    df = compute_forward_returns(picks)
    print(f"  결과: {len(df)}행 (D+1 valid)")

    df.to_parquet(OUT_DIR / "phase2_returns.parquet", index=False)
    print(f"[save] {OUT_DIR / 'phase2_returns.parquet'}")

    metrics = analyze(df)
    print()
    print("=== 핵심 지표 ===")
    for k, v in metrics.items():
        if v is not None and isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    write_report(df, metrics)
    print("\n[OK] Phase 2 완료")


if __name__ == "__main__":
    main()
