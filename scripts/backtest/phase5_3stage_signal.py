"""Phase 5: 퐝가님 시그널 우선순위 백테스트 (3단계 스윙)

퐝가님 통찰 (5/14):
- 1단계: ETF 거래대금 + 금투 + 연기금 + 기타법인 (스마트 머니 1차)
- 2단계: + 기관합계 (확인)
- 3단계: + 외인 (완전 스윙 대박)

→ 외인은 *선행*이 아니라 *마지막 확인 시그널*

각 단계별 D+N 수익률 측정:
- 1단계만: 금투_5d > 0 AND 연기금_5d > 0 AND 기타법인_5d > 0
- 1+2단계: 1단계 + 기관합계_5d > 0
- 1+2+3단계: 1+2 + 외인_5d > 0

출력:
- data/backtest/phase5_3stage.parquet
- data/backtest/phase5_report.md
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sqlite3

OUT_DIR = PROJECT_ROOT / "data" / "backtest"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"


def load_5d_cumulative(date_str: str, ticker: str, conn: sqlite3.Connection) -> dict:
    """date_str(YYYY-MM-DD) 기준 직전 5거래일 누적 (금투/연기금/기타법인/기관합계/외인)"""
    # date_str을 YYYYMMDD 변환
    d_compact = date_str.replace("-", "")
    sql = """
    SELECT investor, SUM(net_val) AS net5d
    FROM investor_daily
    WHERE ticker = ?
      AND date <= ?
      AND date >= (SELECT MIN(date) FROM (
          SELECT DISTINCT date FROM investor_daily
          WHERE date <= ?
          ORDER BY date DESC LIMIT 5
      ))
      AND investor IN ('금융투자', '연기금', '기타법인', '기관합계', '외국인')
    GROUP BY investor
    """
    rows = conn.execute(sql, (ticker, d_compact, d_compact)).fetchall()
    result = {r[0]: r[1] / 1e8 for r in rows}  # 억원 단위
    return {
        "fin_inv_5d": result.get("금융투자", 0.0),
        "pension_5d_db": result.get("연기금", 0.0),
        "corp_5d": result.get("기타법인", 0.0),
        "inst_5d_db": result.get("기관합계", 0.0),
        "fgn_5d_db": result.get("외국인", 0.0),
    }


def main():
    print("=" * 60)
    print("Phase 5: 퐝가님 3단계 시그널 백테스트")
    print("=" * 60)

    df = pd.read_parquet(OUT_DIR / "phase2_returns.parquet")
    print(f"[load] phase2_returns {len(df)}행")

    conn = sqlite3.connect(DB_PATH)

    # 각 행에 6유형 5일 누적 추가
    print("[compute] 6유형 5일 누적 계산 (lookup)...")
    enriched = []
    cache = {}
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        date = row["date"]
        cache_key = (ticker, date)
        if cache_key not in cache:
            cache[cache_key] = load_5d_cumulative(date, ticker, conn)
        flows = cache[cache_key]
        rec = dict(row)
        rec.update(flows)
        enriched.append(rec)

    df2 = pd.DataFrame(enriched)
    conn.close()
    print(f"  결과: {len(df2)}행 enriched")

    # 3단계 시그널 분류
    df2["stage1"] = ((df2["fin_inv_5d"] > 0) & (df2["pension_5d_db"] > 0) & (df2["corp_5d"] > 0)).astype(int)
    df2["stage2"] = (df2["stage1"] & (df2["inst_5d_db"] > 0)).astype(int)
    df2["stage3"] = (df2["stage2"] & (df2["fgn_5d_db"] > 0)).astype(int)

    # 분류
    df2["signal_class"] = "none"
    df2.loc[df2["stage1"] == 1, "signal_class"] = "stage1_only"
    df2.loc[df2["stage2"] == 1, "signal_class"] = "stage2_only"
    df2.loc[df2["stage3"] == 1, "signal_class"] = "stage3_full"

    df2.to_parquet(OUT_DIR / "phase5_3stage.parquet", index=False)

    # 측정
    def measure(sub, label):
        m = {"label": label, "n": len(sub)}
        for n in [1, 3, 5]:
            col = f"ret_d{n}"
            v = sub[col].dropna()
            if len(v) >= 3:
                m[f"d{n}_avg"] = v.mean()
                m[f"d{n}_hit"] = (v > 0).mean() * 100
                m[f"d{n}_n"] = len(v)
            else:
                m[f"d{n}_avg"] = None
        return m

    results = []
    results.append(measure(df2[df2["signal_class"] == "none"], "0단계 (시그널 없음, neither)"))
    results.append(measure(df2[df2["signal_class"] == "stage1_only"], "1단계만 (금투+연기금+기타)"))
    results.append(measure(df2[df2["signal_class"] == "stage2_only"], "1+2단계 (+ 기관합계)"))
    results.append(measure(df2[df2["signal_class"] == "stage3_full"], "1+2+3단계 (+ 외인) ⭐ 스윙"))

    print()
    print("=" * 60)
    print(f"{'시그널':<35} {'Sample':<8} {'D+1':<10} {'D+1 hit':<10} {'D+3':<10} {'D+5':<10}")
    print("-" * 100)
    for r in results:
        d1a = f"{r.get('d1_avg', 0):+.2f}%" if r.get('d1_avg') is not None else "N/A"
        d1h = f"{r.get('d1_hit', 0):.1f}%" if r.get('d1_hit') is not None else "N/A"
        d3a = f"{r.get('d3_avg', 0):+.2f}%" if r.get('d3_avg') is not None else "N/A"
        d5a = f"{r.get('d5_avg', 0):+.2f}%" if r.get('d5_avg') is not None else "N/A"
        print(f"{r['label']:<35} {r['n']:<8} {d1a:<10} {d1h:<10} {d3a:<10} {d5a:<10}")

    # 보고서
    out = OUT_DIR / "phase5_report.md"
    lines = [
        "# Phase 5: 퐝가님 3단계 시그널 백테스트",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**기간**: 2026-04-25 ~ 2026-05-14 (20일)",
        f"**입력**: {len(df2):,}행",
        "",
        "## 가설 (퐝가님 통찰 5/14)",
        "",
        "- **1단계**: ETF 거래대금 + 금투 + 연기금 + 기타법인 → 스마트 머니 진입",
        "- **2단계**: + 기관합계 → 확인",
        "- **3단계**: + 외인 → **완전 스윙 대박**",
        "",
        "**핵심**: 외인은 *선행*이 아니라 *마지막 확인 시그널*",
        "",
        "## 결과",
        "",
        "| 단계 | Sample | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 | D+5 평균 | D+5 적중률 |",
        "|------|--------|---------|---------|---------|---------|---------|---------|",
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
        "## 통과 종목 (1+2+3단계, 스윙 시그널)",
        "",
        "| 일자 | 종목 | 섹터 | 금투5d | 연5d | 기타5d | 기관5d | 외인5d | D+1 | D+3 | D+5 |",
        "|------|-----|------|------|------|-------|-------|-------|-----|-----|-----|",
    ]
    sample = df2[df2["signal_class"] == "stage3_full"].sort_values("date", ascending=False)
    for _, r in sample.iterrows():
        d1 = f"{r['ret_d1']:+.2f}%" if not pd.isna(r.get('ret_d1')) else "-"
        d3 = f"{r['ret_d3']:+.2f}%" if not pd.isna(r.get('ret_d3')) else "-"
        d5 = f"{r['ret_d5']:+.2f}%" if not pd.isna(r.get('ret_d5')) else "-"
        ticker_s = str(r['ticker']).zfill(6)
        lines.append(
            f"| {r['date']} | {r['name']}({ticker_s}) | {r.get('sector', '?')} | "
            f"{r['fin_inv_5d']:+.0f} | {r['pension_5d_db']:+.0f} | {r['corp_5d']:+.0f} | "
            f"{r['inst_5d_db']:+.0f} | {r['fgn_5d_db']:+.0f} | "
            f"{d1} | {d3} | {d5} |"
        )

    lines += [
        "",
        "## 결론",
        "",
        "(데이터 결과 보고 다음 단계 결정)",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {out}")
    print(f"\n[OK] Phase 5 완료")


if __name__ == "__main__":
    main()
