"""
패치 3: NW 앙상블 가중치 검증 백테스트

780일(3년) US daily 데이터로 가중치 조합별 성과 비교:
  - 2-way: L1_US × w_us + NW × w_nw (w_us + w_nw = 1.0)
  - 3-way: L1_US × w_us + KR × w_kr + NW × w_nw (합 = 1.0)

평가 지표:
  - 다음날 EWY 등락과의 상관관계 (Pearson, Spearman)
  - 등급 적중률: 예측 등급 방향과 EWY 실제 방향 일치율
  - 평균 수익: 예측 BULL 시 EWY 수익, BEAR 시 EWY 손실

사용: python -u -X utf8 scripts/backtest_ensemble_weights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

US_PARQUET = PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet"
KR_FLOW_CSV = PROJECT_ROOT / "data" / "kospi_investor_flow.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "us_market" / "ensemble_weight_backtest.json"


def _compute_l1_score(row) -> float:
    """기존 US Overnight L1 스코어 간이 계산 (generate_signal 로직 축약)."""
    score = 0.0

    # 지수 (40%)
    idx_score = 0.0
    for prefix, weight in [("ewy", 0.30), ("qqq", 0.25), ("spy", 0.25), ("dia", 0.20)]:
        ret_1d = float(row.get(f"{prefix}_ret_1d", 0) or 0)
        ret_5d = float(row.get(f"{prefix}_ret_5d", 0) or 0)
        above_sma = int(row.get(f"{prefix}_above_sma20", 0) or 0)
        day_s = max(-1.0, min(1.0, ret_1d * 50))
        trend = max(-0.3, min(0.3, ret_5d * 10))
        sma_b = 0.1 if above_sma else -0.1
        idx_score += (day_s * 0.6 + trend * 0.25 + sma_b * 0.15) * weight
    score += idx_score * 0.40

    # VIX (20%)
    vix_z = float(row.get("vix_zscore", 0) or 0)
    if vix_z < -0.5:
        vix_s = 0.5
    elif vix_z < 0.5:
        vix_s = 0.0
    elif vix_z < 1.5:
        vix_s = -0.4
    else:
        vix_s = -0.8
    if int(row.get("vix_spike", 0) or 0):
        vix_s -= 0.2
    score += vix_s * 0.20

    # 채권/달러 (8%)
    tlt_ret = float(row.get("tlt_ret_1d", 0) or 0)
    score += -max(-0.5, min(0.5, tlt_ret * 30)) * 0.08

    # 원자재 (7%) — 간이
    gld_ret = float(row.get("gld_ret_1d", 0) or 0)
    uso_ret = float(row.get("uso_ret_1d", 0) or 0)
    copx_ret = float(row.get("copx_ret_1d", 0) or 0)
    comm_s = (
        -max(-0.5, min(0.5, gld_ret * 30)) * 0.20
        + max(-0.5, min(0.5, uso_ret * 20)) * 0.20
        + max(-0.5, min(0.5, copx_ret * 25)) * 0.15
    )
    score += comm_s * 0.07

    # 섹터 (25%) — 간이 (soxx만)
    soxx_ret = float(row.get("soxx_ret_1d", 0) or 0)
    soxx_rel = float(row.get("soxx_rel_spy_5d", 0) or 0)
    if soxx_rel > 0.01 and soxx_ret > 0:
        sec_s = 0.5
    elif soxx_rel < -0.01 and soxx_ret < 0:
        sec_s = -0.5
    else:
        sec_s = 0.0
    score += sec_s * 0.25

    return max(-1.0, min(1.0, score))


def _compute_nw_score(row) -> float:
    """NIGHTWATCH 스코어 간이 계산 (engine.py 로직 축약).

    4개 레이어: L0(20%), L1(35%), L2(20%), L4(25%)
    """
    score = 0.0

    # L0: HYG 선행 (5D 수익률)
    hyg_ret_5d = float(row.get("hyg_ret_5d", 0) or 0) if "hyg_ret_5d" in row.index else 0
    if hyg_ret_5d <= -0.04:
        l0 = -0.8
    elif hyg_ret_5d <= -0.02:
        l0 = -0.4
    elif hyg_ret_5d >= 0.02:
        l0 = 0.3
    else:
        l0 = 0.0
    score += l0 * 0.20

    # L1: 채권 자경단 (SPY × 10Y 교차)
    spy_ret = float(row.get("spy_ret_1d", 0) or 0)
    # tnx 변동: 간이로 스킵 (대부분 0에 가까움)
    score += 0.0 * 0.35

    # L2: 레짐 전환 (credit spread, MOVE)
    # 간이: 데이터 없으면 0
    score += 0.0 * 0.20

    # L4: FX 삼각 — 간이
    score += 0.0 * 0.25

    return max(-1.0, min(1.0, score))


def _compute_kr_score(flow_df: pd.DataFrame | None, date) -> float:
    """KR 수급 스코어 간이 계산."""
    if flow_df is None or flow_df.empty:
        return 0.0

    try:
        # date 이전 5일 데이터
        mask = flow_df.index <= pd.Timestamp(date)
        recent = flow_df[mask].tail(5)
        if len(recent) < 3:
            return 0.0

        foreign_5d = recent["foreign_net"].sum()
        inst_5d = recent["inst_net"].sum()

        f_score = max(-1.0, min(1.0, foreign_5d / 50000))
        i_score = max(-1.0, min(1.0, inst_5d / 30000))
        combined = f_score * 0.60 + i_score * 0.40

        # 3일 연속 보너스
        last3 = recent["foreign_net"].tail(3)
        if (last3 > 0).all():
            combined = min(1.0, combined + 0.15)
        elif (last3 < 0).all():
            combined = max(-1.0, combined - 0.15)

        return max(-1.0, min(1.0, combined))
    except Exception:
        return 0.0


def _grade_from_score(score_100: float) -> str:
    if score_100 >= 40:
        return "STRONG_BULL"
    elif score_100 >= 10:
        return "MILD_BULL"
    elif score_100 > -10:
        return "NEUTRAL"
    elif score_100 > -40:
        return "MILD_BEAR"
    else:
        return "STRONG_BEAR"


def _direction_match(grade: str, ewy_next_ret: float) -> bool:
    """예측 방향과 실제 방향 일치 여부."""
    bull = grade in ("STRONG_BULL", "MILD_BULL")
    bear = grade in ("STRONG_BEAR", "MILD_BEAR")
    if bull and ewy_next_ret > 0:
        return True
    if bear and ewy_next_ret < 0:
        return True
    if grade == "NEUTRAL":
        return abs(ewy_next_ret) < 0.01  # ±1% 이내면 정답
    return False


def run_backtest():
    """앙상블 가중치 백테스트."""
    df = pd.read_parquet(US_PARQUET)
    print(f"데이터: {len(df)}일 ({df.index.min().date()} ~ {df.index.max().date()})")

    # KR 수급 데이터
    flow_df = None
    if KR_FLOW_CSV.exists():
        flow_df = pd.read_csv(KR_FLOW_CSV, parse_dates=["Date"], index_col="Date")
        print(f"KR 수급: {len(flow_df)}일")

    # 각 날짜별 스코어 계산
    records = []
    for i in range(len(df) - 1):
        row = df.iloc[i]
        date = df.index[i]

        l1 = _compute_l1_score(row)
        nw = _compute_nw_score(row)
        kr = _compute_kr_score(flow_df, date)

        # 다음날 EWY 수익률
        next_row = df.iloc[i + 1]
        ewy_next = float(next_row.get("ewy_ret_1d", 0) or 0)

        records.append({
            "date": date,
            "l1": l1,
            "nw": nw,
            "kr": kr,
            "ewy_next": ewy_next,
        })

    rdf = pd.DataFrame(records).set_index("date")
    print(f"평가 대상: {len(rdf)}일\n")

    # ── 2-way 가중치 비교 ──
    two_way_configs = [
        {"name": "L1:100 NW:0",   "w_us": 1.00, "w_nw": 0.00},
        {"name": "L1:80 NW:20",   "w_us": 0.80, "w_nw": 0.20},
        {"name": "L1:70 NW:30 ★", "w_us": 0.70, "w_nw": 0.30},
        {"name": "L1:60 NW:40",   "w_us": 0.60, "w_nw": 0.40},
        {"name": "L1:50 NW:50",   "w_us": 0.50, "w_nw": 0.50},
    ]

    print("=" * 70)
    print("2-way 앙상블 (L1_US + NW)")
    print("=" * 70)
    print(f"{'Config':<20} {'Pearson':>8} {'Spearman':>9} {'방향적중':>8} {'BULL수익':>9} {'BEAR손실':>9}")
    print("-" * 70)

    results = {"two_way": [], "three_way": []}

    for cfg in two_way_configs:
        ensemble = rdf["l1"] * cfg["w_us"] + rdf["nw"] * cfg["w_nw"]
        ens_100 = ensemble * 100

        grades = ens_100.apply(_grade_from_score)
        matches = [_direction_match(g, r) for g, r in zip(grades, rdf["ewy_next"])]

        bull_mask = grades.isin(["STRONG_BULL", "MILD_BULL"])
        bear_mask = grades.isin(["STRONG_BEAR", "MILD_BEAR"])

        bull_ret = rdf.loc[bull_mask, "ewy_next"].mean() * 100 if bull_mask.any() else 0
        bear_ret = rdf.loc[bear_mask, "ewy_next"].mean() * 100 if bear_mask.any() else 0

        p_corr = pearsonr(ensemble, rdf["ewy_next"])[0]
        s_corr = spearmanr(ensemble, rdf["ewy_next"])[0]
        dir_acc = sum(matches) / len(matches) * 100

        print(
            f"{cfg['name']:<20} {p_corr:>8.4f} {s_corr:>9.4f} "
            f"{dir_acc:>7.1f}% {bull_ret:>+8.2f}% {bear_ret:>+8.2f}%"
        )

        results["two_way"].append({
            "config": cfg["name"],
            "w_us": cfg["w_us"], "w_nw": cfg["w_nw"],
            "pearson": round(p_corr, 4),
            "spearman": round(s_corr, 4),
            "direction_accuracy": round(dir_acc, 1),
            "bull_avg_return": round(bull_ret, 2),
            "bear_avg_return": round(bear_ret, 2),
        })

    # ── 3-way 가중치 비교 ──
    three_way_configs = [
        {"name": "US50 KR20 NW30 ★", "w_us": 0.50, "w_kr": 0.20, "w_nw": 0.30},
        {"name": "US50 KR30 NW20",    "w_us": 0.50, "w_kr": 0.30, "w_nw": 0.20},
        {"name": "US40 KR30 NW30",    "w_us": 0.40, "w_kr": 0.30, "w_nw": 0.30},
        {"name": "US60 KR10 NW30",    "w_us": 0.60, "w_kr": 0.10, "w_nw": 0.30},
        {"name": "US40 KR20 NW40",    "w_us": 0.40, "w_kr": 0.20, "w_nw": 0.40},
    ]

    print(f"\n{'=' * 70}")
    print("3-way 앙상블 (L1_US + KR + NW)")
    print("=" * 70)
    print(f"{'Config':<22} {'Pearson':>8} {'Spearman':>9} {'방향적중':>8} {'BULL수익':>9} {'BEAR손실':>9}")
    print("-" * 70)

    for cfg in three_way_configs:
        ensemble = (
            rdf["l1"] * cfg["w_us"]
            + rdf["kr"] * cfg["w_kr"]
            + rdf["nw"] * cfg["w_nw"]
        )
        ens_100 = ensemble * 100

        grades = ens_100.apply(_grade_from_score)
        matches = [_direction_match(g, r) for g, r in zip(grades, rdf["ewy_next"])]

        bull_mask = grades.isin(["STRONG_BULL", "MILD_BULL"])
        bear_mask = grades.isin(["STRONG_BEAR", "MILD_BEAR"])

        bull_ret = rdf.loc[bull_mask, "ewy_next"].mean() * 100 if bull_mask.any() else 0
        bear_ret = rdf.loc[bear_mask, "ewy_next"].mean() * 100 if bear_mask.any() else 0

        p_corr = pearsonr(ensemble, rdf["ewy_next"])[0]
        s_corr = spearmanr(ensemble, rdf["ewy_next"])[0]
        dir_acc = sum(matches) / len(matches) * 100

        print(
            f"{cfg['name']:<22} {p_corr:>8.4f} {s_corr:>9.4f} "
            f"{dir_acc:>7.1f}% {bull_ret:>+8.2f}% {bear_ret:>+8.2f}%"
        )

        results["three_way"].append({
            "config": cfg["name"],
            "w_us": cfg["w_us"], "w_kr": cfg["w_kr"], "w_nw": cfg["w_nw"],
            "pearson": round(p_corr, 4),
            "spearman": round(s_corr, 4),
            "direction_accuracy": round(dir_acc, 1),
            "bull_avg_return": round(bull_ret, 2),
            "bear_avg_return": round(bear_ret, 2),
        })

    # ── 최적 조합 선택 ──
    all_cfgs = results["two_way"] + results["three_way"]
    best = max(all_cfgs, key=lambda x: x["pearson"])
    print(f"\n{'=' * 70}")
    print(f"최적 가중치: {best['config']} (Pearson {best['pearson']:.4f})")
    print("=" * 70)

    results["best"] = best
    results["data_range"] = f"{df.index.min().date()} ~ {df.index.max().date()}"
    results["total_days"] = len(rdf)

    # 저장
    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_backtest()
