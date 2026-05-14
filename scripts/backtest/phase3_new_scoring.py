"""Phase 3: 새 점수 산식 (기관 단독 매수 가중) 백테스트

Phase 2 발견:
- 외+기 동시매수: D+1 -0.43% (역시그널)
- 외인만 매수: D+1 -0.50% (역시그널)
- 기관만 매수: D+1 +0.28% (양의 시그널!)
- 둘 다 매도: D+1 -0.59%

새 가설:
- INST_PURE_BUY: fgn_5d <= 0 AND inst_5d > 0 → 강력 매수 시그널
- 위 조건 + buy_score 30-39 → 추가 필터 (Phase 2의 양수 구간)
- 우량주 + 대형주 한정 (vol_ratio > 1.0 또는 시총 상위)

검증:
1. 새 시그널 적중률 vs 기존 BUY
2. 새 시그널 평균 수익률 vs 기존 BUY
3. 점수 구간 결합 효과

출력:
- data/backtest/phase3_new_signals.parquet
- data/backtest/phase3_report.md
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUT_DIR = PROJECT_ROOT / "data" / "backtest"


def apply_new_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """새 시그널 컬럼 추가"""
    df = df.copy()

    # 시그널 분류
    df["inst_pure_buy"] = ((df["fgn_5d"] <= 0) & (df["inst_5d"] > 0)).astype(int)
    df["fgn_pure_buy"] = ((df["fgn_5d"] > 0) & (df["inst_5d"] <= 0)).astype(int)
    df["dual_buy"] = ((df["fgn_5d"] > 0) & (df["inst_5d"] > 0)).astype(int)
    df["dual_sell"] = ((df["fgn_5d"] <= 0) & (df["inst_5d"] <= 0)).astype(int)

    # 강도 (절대값 큰 기관 매수일수록 강력)
    df["inst_strength"] = df["inst_5d"].abs()  # 억 단위
    df["fgn_strength"] = df["fgn_5d"].abs()

    # 새 점수 (Phase 2 통계 기반)
    df["new_score"] = (
        df["inst_pure_buy"] * 15.0  # 핵심: 기관 단독 매수
        + df["dual_buy"] * 5.0       # 외+기 동시 (기관 영향)
        + df["fgn_pure_buy"] * (-5.0)  # 외인 단독 (역시그널)
        + df["dual_sell"] * (-10.0)   # 둘 다 매도
    )

    # 점수 구간 결합 (Phase 2의 양수 구간 30-39)
    df["score_in_sweet_zone"] = ((df["buy_score"] >= 30) & (df["buy_score"] <= 39)).astype(int)

    # 강력 매수 시그널 = inst_pure_buy + sweet_zone
    df["new_strong_buy"] = ((df["inst_pure_buy"] == 1) & (df["score_in_sweet_zone"] == 1)).astype(int)

    # 단순 매수 시그널 = inst_pure_buy만
    df["new_simple_buy"] = (df["inst_pure_buy"] == 1).astype(int)

    return df


def analyze(df: pd.DataFrame) -> dict:
    """새 시그널 성과 측정"""
    m = {}

    # 기존 BUY (대조군)
    if "buy_grade" in df.columns:
        bench = df[df["buy_grade"] == "BUY"]
        for n in [1, 3, 5]:
            col = f"ret_d{n}"
            v = bench[col].dropna()
            if len(v) >= 3:
                m[f"bench_BUY_d{n}_avg"] = v.mean()
                m[f"bench_BUY_d{n}_hit"] = (v > 0).mean() * 100
                m[f"bench_BUY_d{n}_n"] = len(v)

    # 새 시그널: new_simple_buy (기관 단독 매수)
    sub = df[df["new_simple_buy"] == 1]
    for n in [1, 3, 5]:
        col = f"ret_d{n}"
        v = sub[col].dropna()
        if len(v) >= 3:
            m[f"new_simple_d{n}_avg"] = v.mean()
            m[f"new_simple_d{n}_hit"] = (v > 0).mean() * 100
            m[f"new_simple_d{n}_n"] = len(v)

    # 새 시그널: new_strong_buy (inst_pure + sweet_zone)
    sub = df[df["new_strong_buy"] == 1]
    for n in [1, 3, 5]:
        col = f"ret_d{n}"
        v = sub[col].dropna()
        if len(v) >= 3:
            m[f"new_strong_d{n}_avg"] = v.mean()
            m[f"new_strong_d{n}_hit"] = (v > 0).mean() * 100
            m[f"new_strong_d{n}_n"] = len(v)

    # 새 점수 vs 수익률 correlation
    for n in [1, 3, 5]:
        col = f"ret_d{n}"
        sub = df[["new_score", col]].dropna()
        if len(sub) >= 5:
            m[f"new_score_corr_d{n}"] = sub["new_score"].corr(sub[col])

    # 추가: 기관 매수 강도 구간별 (inst_5d > 0 인 종목들)
    inst_pos = df[df["inst_5d"] > 0].copy()
    if len(inst_pos) > 0:
        # 5분위
        quantiles = pd.qcut(inst_pos["inst_5d"], q=4, labels=["Q1_낮음", "Q2", "Q3", "Q4_높음"], duplicates="drop")
        inst_pos["inst_q"] = quantiles
        for q in ["Q1_낮음", "Q2", "Q3", "Q4_높음"]:
            sub = inst_pos[inst_pos["inst_q"] == q]
            for n in [1, 3]:
                col = f"ret_d{n}"
                v = sub[col].dropna()
                if len(v) >= 3:
                    m[f"inst_q_{q}_d{n}_avg"] = v.mean()
                    m[f"inst_q_{q}_d{n}_hit"] = (v > 0).mean() * 100
                    m[f"inst_q_{q}_d{n}_n"] = len(v)

    return m


def write_report(df: pd.DataFrame, metrics: dict):
    out = OUT_DIR / "phase3_report.md"
    lines = [
        "# Phase 3: 새 점수 산식 (기관 단독 매수 가중) 백테스트",
        "",
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**기간**: 2026-04-25 ~ 2026-05-14 (20일)",
        f"**입력**: {len(df):,}행",
        "",
        "## 1. 기존 vs 새 시그널 비교",
        "",
        "| 시그널 | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 | D+5 평균 | Sample |",
        "|--------|---------|---------|---------|---------|---------|--------|",
    ]

    # 기존 BUY
    if "bench_BUY_d1_avg" in metrics:
        lines.append(
            f"| 기존 BUY (대조군) | "
            f"{metrics['bench_BUY_d1_avg']:+.2f}% | "
            f"{metrics['bench_BUY_d1_hit']:.1f}% | "
            f"{metrics.get('bench_BUY_d3_avg', 0):+.2f}% | "
            f"{metrics.get('bench_BUY_d3_hit', 0):.1f}% | "
            f"{metrics.get('bench_BUY_d5_avg', 0):+.2f}% | "
            f"{metrics['bench_BUY_d1_n']} |"
        )

    # new_simple
    if "new_simple_d1_avg" in metrics:
        lines.append(
            f"| **new_simple (기관 단독 매수)** | "
            f"**{metrics['new_simple_d1_avg']:+.2f}%** | "
            f"**{metrics['new_simple_d1_hit']:.1f}%** | "
            f"{metrics.get('new_simple_d3_avg', 0):+.2f}% | "
            f"{metrics.get('new_simple_d3_hit', 0):.1f}% | "
            f"{metrics.get('new_simple_d5_avg', 0):+.2f}% | "
            f"{metrics['new_simple_d1_n']} |"
        )

    # new_strong
    if "new_strong_d1_avg" in metrics:
        lines.append(
            f"| **new_strong (기관 + sweet_zone)** | "
            f"**{metrics['new_strong_d1_avg']:+.2f}%** | "
            f"**{metrics['new_strong_d1_hit']:.1f}%** | "
            f"{metrics.get('new_strong_d3_avg', 0):+.2f}% | "
            f"{metrics.get('new_strong_d3_hit', 0):.1f}% | "
            f"{metrics.get('new_strong_d5_avg', 0):+.2f}% | "
            f"{metrics['new_strong_d1_n']} |"
        )

    lines += [
        "",
        "## 2. 새 점수 (new_score) vs 수익률 correlation",
        "",
        "| D+N | Correlation | 의미 |",
        "|-----|----------|------|",
    ]
    for n in [1, 3, 5]:
        c = metrics.get(f"new_score_corr_d{n}")
        if c is None:
            continue
        meaning = "강한 상관" if abs(c) > 0.3 else "약한 상관" if abs(c) > 0.1 else "거의 무관"
        lines.append(f"| D+{n} | {c:+.4f} | {meaning} |")

    lines += [
        "",
        "## 3. 기관 매수 강도 (inst_5d) 분위별",
        "",
        "기관 순매수 0 이상인 종목만 4분위로 나눠 분석.",
        "",
        "| 분위 | D+1 평균 | D+1 적중률 | D+3 평균 | D+3 적중률 | Sample (D+1) |",
        "|------|---------|---------|---------|---------|------------|",
    ]
    for q in ["Q1_낮음", "Q2", "Q3", "Q4_높음"]:
        d1a = metrics.get(f"inst_q_{q}_d1_avg")
        d1h = metrics.get(f"inst_q_{q}_d1_hit")
        d3a = metrics.get(f"inst_q_{q}_d3_avg")
        d3h = metrics.get(f"inst_q_{q}_d3_hit")
        n = metrics.get(f"inst_q_{q}_d1_n", 0)
        if d1a is None:
            continue
        lines.append(f"| {q} | {d1a:+.2f}% | {d1h:.1f}% | {d3a:+.2f}% | {d3h:.1f}% | {n} |")

    lines += [
        "",
        "## 4. 결론 및 Phase 4 제안",
        "",
        "**핵심 검증**:",
        "- new_simple > bench_BUY (적중률 + 평균 수익률)인가?",
        "- new_strong > new_simple (sweet_zone 필터 효과)인가?",
        "- 기관 매수 강도가 클수록 수익률 ↑ (Q1 < Q4)인가?",
        "",
        "**다음 액션 (Phase 4 후보)**:",
        "1. 결과 양호 → scan_sector_fire.py 점수 산식 패치 (환경변수 토글)",
        "2. 결과 미흡 → 추가 가설 (시장 레짐 보정, 거래량 필터, 시총 필터 등)",
        "3. 데이터 부족 → 1~3개월 누적 후 재검증",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out}")


def main():
    print("=" * 60)
    print("Phase 3: 새 점수 산식 (기관 단독 매수) 백테스트")
    print("=" * 60)

    df = pd.read_parquet(OUT_DIR / "phase2_returns.parquet")
    print(f"[load] phase2_returns {len(df)}행")

    df = apply_new_scoring(df)
    df.to_parquet(OUT_DIR / "phase3_new_signals.parquet", index=False)

    metrics = analyze(df)
    print()
    print("=== 핵심 지표 ===")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, (int, float)) and not np.isnan(v):
            print(f"  {k}: {v:.4f}")

    write_report(df, metrics)
    print("\n[OK] Phase 3 완료")


if __name__ == "__main__":
    main()
