#!/usr/bin/env python3
"""
US Overnight Signal 백테스트 — Walk-Forward 방식

us_kr_history.db의 466건 데이터로 시스템 성과를 사후 검증한다.
Walk-Forward: 과거 데이터만으로 예측 → 다음날 실제 결과와 비교.

비교 실험:
  A) EWY 포함 Score vs EWY 미포함 Score
  B) L1 단독 vs L1+L2 결합
  C) Kill Signal 발동일 vs 미발동일

사용법:
  python scripts/backtest_us_signal.py [--warmup 60]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_us_kr_history import _clamp, PatternMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "us_market" / "us_kr_history.db"
REPORT_PATH = PROJECT_ROOT / "data" / "us_market" / "backtest_us_signal_report.json"

# ── Kill 설정 (us_overnight_signal.py와 동일) ──
SECTOR_KILL_CONFIG = {
    "반도체":   {"kill_col": "us_soxx_chg",  "threshold": -3.0, "sensitivity": 0.95},
    "전자부품": {"kill_col": "us_soxx_chg",  "threshold": -3.0, "sensitivity": 0.95},
    "디스플레이":{"kill_col": "us_soxx_chg",  "threshold": -3.0, "sensitivity": 0.95},
    "IT":       {"kill_col": "us_nasdaq_chg", "threshold": -3.5, "sensitivity": 0.70},
    "소프트웨어":{"kill_col": "us_nasdaq_chg", "threshold": -3.5, "sensitivity": 0.70},
    "에너지":   {"kill_col": "us_sp500_chg",  "threshold": -5.0, "sensitivity": 0.80},
    "정유":     {"kill_col": "us_sp500_chg",  "threshold": -5.0, "sensitivity": 0.80},
    "화학":     {"kill_col": "us_sp500_chg",  "threshold": -5.0, "sensitivity": 0.80},
    "은행":     {"kill_col": "us_sp500_chg",  "threshold": -5.0, "sensitivity": 0.50},
    "증권":     {"kill_col": "us_sp500_chg",  "threshold": -5.0, "sensitivity": 0.50},
    "금융":     {"kill_col": "us_sp500_chg",  "threshold": -5.0, "sensitivity": 0.50},
    "제약":     {"kill_col": "us_nasdaq_chg", "threshold": -5.0, "sensitivity": 0.45},
    "바이오":   {"kill_col": "us_nasdaq_chg", "threshold": -5.0, "sensitivity": 0.45},
    "의료기기": {"kill_col": "us_nasdaq_chg", "threshold": -5.0, "sensitivity": 0.45},
    "헬스케어": {"kill_col": "us_nasdaq_chg", "threshold": -5.0, "sensitivity": 0.45},
    "조선":     {"kill_col": "us_sp500_chg",  "threshold": -7.0, "sensitivity": 0.25},
    "기계":     {"kill_col": "us_sp500_chg",  "threshold": -7.0, "sensitivity": 0.25},
    "건설":     {"kill_col": "us_sp500_chg",  "threshold": -7.0, "sensitivity": 0.25},
    "자동차":   {"kill_col": "us_sp500_chg",  "threshold": -7.0, "sensitivity": 0.25},
    "운송":     {"kill_col": "us_sp500_chg",  "threshold": -7.0, "sensitivity": 0.25},
}

# ── 섹터 매핑 (DB 컬럼 → 한국 섹터명) ──
KR_SECTOR_MAP = {
    "kr_semi_chg":     "반도체",
    "kr_ev_chg":       "2차전지",
    "kr_bio_chg":      "바이오",
    "kr_bank_chg":     "은행",
    "kr_steel_chg":    "철강",
    "kr_it_chg":       "IT",
    "kr_oil_chg":      "에너지",
    "kr_domestic_chg": "내수",
}


# ================================================================
# Score 계산 함수들
# ================================================================

def calc_score_with_ewy(record: dict) -> float:
    """EWY 포함 Score (현재 v8.7 가중치)."""
    score = 0.0
    v = record.get("us_ewy_chg")
    if v is not None:
        score += _clamp(v * 12.5, -25, 25)
    v = record.get("us_nasdaq_chg")
    if v is not None:
        score += _clamp(v * 10, -20, 20)
    v = record.get("us_sp500_chg")
    if v is not None:
        score += _clamp(v * 7.5, -15, 15)
    v = record.get("us_vix_chg")
    if v is not None:
        score += _clamp(v * -3.75, -15, 15)
    v = record.get("us_soxx_chg")
    if v is not None:
        score += _clamp(v * 10, -15, 15)
    v = record.get("us_dollar_chg")
    if v is not None:
        score += _clamp(v * -7, -10, 10)
    return round(_clamp(score, -100, 100), 1)


def calc_score_without_ewy(record: dict) -> float:
    """EWY 미포함 Score (이전 v8.6 가중치)."""
    score = 0.0
    v = record.get("us_nasdaq_chg")
    if v is not None:
        score += _clamp(v * 15, -30, 30)
    v = record.get("us_sp500_chg")
    if v is not None:
        score += _clamp(v * 10, -20, 20)
    v = record.get("us_vix_chg")
    if v is not None:
        score += _clamp(v * -5, -20, 20)
    v = record.get("us_dollar_chg")
    if v is not None:
        score += _clamp(v * -10, -15, 15)
    v = record.get("us_soxx_chg")
    if v is not None:
        score += _clamp(v * 10, -15, 15)
    return round(_clamp(score, -100, 100), 1)


def classify_grade(score: float) -> str:
    """Score → 5등급."""
    if score >= 50:
        return "STRONG_BULL"
    elif score >= 20:
        return "MILD_BULL"
    elif score > -20:
        return "NEUTRAL"
    elif score > -50:
        return "MILD_BEAR"
    else:
        return "STRONG_BEAR"


def check_kills(record: dict) -> dict[str, bool]:
    """섹터 Kill 판정. DB 컬럼값은 % 단위."""
    kills = {}
    for sector, cfg in SECTOR_KILL_CONFIG.items():
        val = record.get(cfg["kill_col"])
        if val is not None:
            kills[sector] = val <= cfg["threshold"]
        else:
            kills[sector] = False
    return kills


def check_special_rules(record: dict) -> list[str]:
    """특수 룰 체크."""
    triggered = []
    vix_chg = record.get("us_vix_chg") or 0
    vix_level = record.get("us_vix_level") or 20
    qqq = record.get("us_nasdaq_chg") or 0
    spy = record.get("us_sp500_chg") or 0
    soxx = record.get("us_soxx_chg") or 0

    if vix_chg > 20:
        triggered.append("VIX_SPIKE")
    if vix_level >= 30:
        triggered.append("VIX_HIGH")
    if soxx <= -5:
        triggered.append("SOXX_CRASH")
    if qqq <= -3:
        triggered.append("NASDAQ_CIRCUIT")
    if qqq >= 2 and spy >= 2 and soxx >= 2:
        triggered.append("TRIPLE_BULL")
    if spy <= -3:
        triggered.append("MARKET_CRASH")
    return triggered


# ================================================================
# Walk-Forward 패턴매칭 (L2)
# ================================================================

def walk_forward_l2(df_history: pd.DataFrame, today_idx: int) -> float:
    """Walk-Forward L2: today_idx 이전 데이터만 사용하여 패턴 보정값 산출."""
    if today_idx < 30:
        return 0.0

    past = df_history.iloc[:today_idx].copy()
    today = df_history.iloc[today_idx]

    features = [
        "us_nasdaq_chg", "us_sp500_chg", "us_vix_chg",
        "us_soxx_chg", "us_dollar_chg", "us_ewy_chg",
    ]

    today_vec = np.array([today.get(f, 0) or 0 for f in features])
    feature_df = past[features].fillna(0)

    if len(feature_df) < 30:
        return 0.0

    means = feature_df.mean()
    stds = feature_df.std().replace(0, 1)

    norm_today = (today_vec - means.values) / stds.values
    norm_hist = (feature_df - means) / stds

    distances = np.sqrt(((norm_hist - norm_today) ** 2).sum(axis=1))
    threshold = np.percentile(distances.dropna(), 20)
    similar = past.loc[distances <= threshold]

    if len(similar) < 10:
        return 0.0

    kospi = similar["kr_kospi_chg"].dropna()
    if len(kospi) < 10:
        return 0.0

    mean_chg = kospi.mean()
    adjustment = _clamp(mean_chg * 5, -15, 15)
    return round(adjustment, 1)


# ================================================================
# 메인 백테스트
# ================================================================

def run_backtest(warmup: int = 60) -> dict:
    """Walk-Forward 백테스트 실행."""

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM us_kr_history ORDER BY date ASC", conn)
    conn.close()

    logger.info(f"DB 로드: {len(df)}건 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    logger.info(f"워밍업: {warmup}일 → 테스트: {len(df) - warmup}일")

    results = []

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        record = row.to_dict()

        # ── Score 계산 ──
        score_ewy = calc_score_with_ewy(record)
        score_no_ewy = calc_score_without_ewy(record)

        # ── L2 Walk-Forward ──
        l2_adj = walk_forward_l2(df, i)
        combined_ewy = round(_clamp(score_ewy + l2_adj, -100, 100), 1)
        combined_no_ewy = round(_clamp(score_no_ewy + l2_adj, -100, 100), 1)

        # ── Kill / Rules ──
        kills = check_kills(record)
        rules = check_special_rules(record)
        any_kill = any(kills.values())
        any_rule = len(rules) > 0

        # ── 실제 결과 ──
        kospi_actual = record.get("kr_kospi_chg") or 0
        gap_actual = record.get("kr_kospi_open_gap") or 0

        results.append({
            "date": record["date"],
            "us_date": record["us_date"],
            "score_ewy": score_ewy,
            "score_no_ewy": score_no_ewy,
            "l2_adj": l2_adj,
            "combined_ewy": combined_ewy,
            "combined_no_ewy": combined_no_ewy,
            "grade_ewy": classify_grade(score_ewy),
            "grade_no_ewy": classify_grade(score_no_ewy),
            "any_kill": any_kill,
            "any_rule": any_rule,
            "rules": rules,
            "kills": {k: v for k, v in kills.items() if v},
            "kospi_actual": kospi_actual,
            "gap_actual": gap_actual,
            # 섹터 실적
            "kr_semi": record.get("kr_semi_chg") or 0,
            "kr_ev": record.get("kr_ev_chg") or 0,
            "kr_bio": record.get("kr_bio_chg") or 0,
            "kr_bank": record.get("kr_bank_chg") or 0,
            "kr_steel": record.get("kr_steel_chg") or 0,
            "kr_it": record.get("kr_it_chg") or 0,
            "kr_oil": record.get("kr_oil_chg") or 0,
            "kr_domestic": record.get("kr_domestic_chg") or 0,
            # EWY
            "us_ewy_chg": record.get("us_ewy_chg") or 0,
        })

    logger.info(f"백테스트 완료: {len(results)}건")
    return analyze_results(results)


# ================================================================
# 분석
# ================================================================

def analyze_results(results: list[dict]) -> dict:
    """백테스트 결과 종합 분석."""
    df = pd.DataFrame(results)
    report = {"test_days": len(df), "period": f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"}

    # ─── 실험 A: EWY 포함 vs 미포함 ───
    report["experiment_A"] = exp_a_ewy_comparison(df)

    # ─── 실험 B: L1 단독 vs L1+L2 ───
    report["experiment_B"] = exp_b_l2_effect(df)

    # ─── 실험 C: Kill Signal 효과 ───
    report["experiment_C"] = exp_c_kill_effect(df)

    # ─── Score 구간별 분석 ───
    report["grade_analysis"] = grade_analysis(df)

    # ─── 섹터별 상관 분석 ───
    report["sector_analysis"] = sector_analysis(df)

    # ─── 특수 룰 분석 ───
    report["special_rules_analysis"] = rules_analysis(df)

    # ─── 극단 상황 분석 ───
    report["extreme_analysis"] = extreme_analysis(df)

    return report


def _direction_accuracy(scores: pd.Series, actuals: pd.Series) -> dict:
    """방향 적중률 계산."""
    positive_pred = scores > 0
    negative_pred = scores < 0
    neutral_pred = scores == 0

    tp = ((positive_pred) & (actuals > 0)).sum()  # 양수 예측, 실제 상승
    tn = ((negative_pred) & (actuals < 0)).sum()  # 음수 예측, 실제 하락
    total_pred = (positive_pred | negative_pred).sum()

    acc = (tp + tn) / total_pred * 100 if total_pred > 0 else 0

    # 양수 예측 적중률
    pos_count = positive_pred.sum()
    pos_hit = tp / pos_count * 100 if pos_count > 0 else 0

    # 음수 예측 적중률
    neg_count = negative_pred.sum()
    neg_hit = tn / neg_count * 100 if neg_count > 0 else 0

    return {
        "overall_accuracy": round(acc, 1),
        "bull_accuracy": round(pos_hit, 1),
        "bull_count": int(pos_count),
        "bear_accuracy": round(neg_hit, 1),
        "bear_count": int(neg_count),
        "neutral_count": int(neutral_pred.sum()),
    }


def _rmse(predicted: pd.Series, actual: pd.Series) -> float:
    """Score(-100~100)를 -2%~+2% 스케일로 변환 후 RMSE."""
    pred_pct = predicted / 50  # 100 → 2%, -100 → -2%
    return round(np.sqrt(((pred_pct - actual) ** 2).mean()), 4)


def exp_a_ewy_comparison(df: pd.DataFrame) -> dict:
    """실험 A: EWY 포함 vs 미포함."""
    acc_ewy = _direction_accuracy(df["score_ewy"], df["kospi_actual"])
    acc_no_ewy = _direction_accuracy(df["score_no_ewy"], df["kospi_actual"])

    rmse_ewy = _rmse(df["score_ewy"], df["kospi_actual"])
    rmse_no_ewy = _rmse(df["score_no_ewy"], df["kospi_actual"])

    # 상관계수
    corr_ewy = round(df["score_ewy"].corr(df["kospi_actual"]), 4)
    corr_no_ewy = round(df["score_no_ewy"].corr(df["kospi_actual"]), 4)

    # EWY vs KOSPI 직접 상관
    ewy_kospi_corr = round(df["us_ewy_chg"].corr(df["kospi_actual"]), 4)

    improvement = round(acc_ewy["overall_accuracy"] - acc_no_ewy["overall_accuracy"], 1)

    return {
        "with_ewy": {
            "accuracy": acc_ewy,
            "rmse": rmse_ewy,
            "correlation": corr_ewy,
        },
        "without_ewy": {
            "accuracy": acc_no_ewy,
            "rmse": rmse_no_ewy,
            "correlation": corr_no_ewy,
        },
        "ewy_kospi_direct_corr": ewy_kospi_corr,
        "accuracy_improvement": improvement,
        "rmse_improvement": round(rmse_no_ewy - rmse_ewy, 4),
        "summary": f"EWY 추가로 방향 적중률 {improvement:+.1f}%p, "
                   f"상관계수 {corr_ewy - corr_no_ewy:+.4f} 개선",
    }


def exp_b_l2_effect(df: pd.DataFrame) -> dict:
    """실험 B: L1 단독 vs L1+L2."""
    # L1 단독
    acc_l1 = _direction_accuracy(df["score_ewy"], df["kospi_actual"])
    rmse_l1 = _rmse(df["score_ewy"], df["kospi_actual"])
    corr_l1 = round(df["score_ewy"].corr(df["kospi_actual"]), 4)

    # L1+L2
    acc_l1l2 = _direction_accuracy(df["combined_ewy"], df["kospi_actual"])
    rmse_l1l2 = _rmse(df["combined_ewy"], df["kospi_actual"])
    corr_l1l2 = round(df["combined_ewy"].corr(df["kospi_actual"]), 4)

    # L2 보정값이 있는 날만
    l2_active = df[df["l2_adj"] != 0]
    l2_correct = ((l2_active["l2_adj"] > 0) & (l2_active["kospi_actual"] > 0)) | \
                 ((l2_active["l2_adj"] < 0) & (l2_active["kospi_actual"] < 0))

    l2_dir_acc = round(l2_correct.mean() * 100, 1) if len(l2_active) > 0 else 0

    return {
        "l1_only": {
            "accuracy": acc_l1,
            "rmse": rmse_l1,
            "correlation": corr_l1,
        },
        "l1_plus_l2": {
            "accuracy": acc_l1l2,
            "rmse": rmse_l1l2,
            "correlation": corr_l1l2,
        },
        "l2_active_days": len(l2_active),
        "l2_direction_accuracy": l2_dir_acc,
        "accuracy_improvement": round(
            acc_l1l2["overall_accuracy"] - acc_l1["overall_accuracy"], 1
        ),
        "summary": f"L2 패턴매칭으로 적중률 "
                   f"{acc_l1l2['overall_accuracy'] - acc_l1['overall_accuracy']:+.1f}%p, "
                   f"L2 자체 방향 적중률 {l2_dir_acc:.1f}%",
    }


def exp_c_kill_effect(df: pd.DataFrame) -> dict:
    """실험 C: Kill Signal 효과."""
    kill_days = df[df["any_kill"]]
    normal_days = df[~df["any_kill"]]

    # Kill 발동일의 KOSPI 평균
    kill_kospi = kill_days["kospi_actual"].mean() if len(kill_days) > 0 else 0
    normal_kospi = normal_days["kospi_actual"].mean() if len(normal_days) > 0 else 0

    # Kill 발동일에 실제로 빠졌는지 (방어 성공률)
    kill_defense = (kill_days["kospi_actual"] < 0).mean() * 100 if len(kill_days) > 0 else 0

    # 섹터별 Kill 효과
    sector_kill_effect = {}
    for _, row in kill_days.iterrows():
        for sector, killed in row.get("kills", {}).items():
            if killed:
                # 해당 섹터의 DB 컬럼 찾기
                col_map = {
                    "반도체": "kr_semi", "IT": "kr_it", "에너지": "kr_oil",
                    "은행": "kr_bank", "바이오": "kr_bio",
                }
                col = col_map.get(sector)
                if col and col in row:
                    if sector not in sector_kill_effect:
                        sector_kill_effect[sector] = []
                    sector_kill_effect[sector].append(row[col])

    sector_summary = {}
    for sector, vals in sector_kill_effect.items():
        arr = np.array(vals)
        sector_summary[sector] = {
            "count": len(arr),
            "mean_chg": round(arr.mean(), 3),
            "defense_rate": round((arr < 0).mean() * 100, 1),
        }

    # 특수 룰 발동일
    rule_days = df[df["any_rule"]]
    rule_kospi = rule_days["kospi_actual"].mean() if len(rule_days) > 0 else 0

    return {
        "kill_days": len(kill_days),
        "normal_days": len(normal_days),
        "kill_kospi_mean": round(kill_kospi, 3),
        "normal_kospi_mean": round(normal_kospi, 3),
        "kill_defense_rate": round(kill_defense, 1),
        "sector_kill_detail": sector_summary,
        "rule_days": len(rule_days),
        "rule_kospi_mean": round(rule_kospi, 3),
        "summary": f"Kill 발동 {len(kill_days)}일: KOSPI 평균 {kill_kospi:+.3f}% "
                   f"(방어율 {kill_defense:.0f}%) | "
                   f"정상 {len(normal_days)}일: {normal_kospi:+.3f}%",
    }


def grade_analysis(df: pd.DataFrame) -> dict:
    """Score 구간별 KOSPI 반응 분석."""
    grades = {}
    for grade in ["STRONG_BULL", "MILD_BULL", "NEUTRAL", "MILD_BEAR", "STRONG_BEAR"]:
        subset = df[df["grade_ewy"] == grade]
        if len(subset) == 0:
            continue
        kospi = subset["kospi_actual"]
        gap = subset["gap_actual"]
        grades[grade] = {
            "count": len(subset),
            "kospi_mean": round(kospi.mean(), 3),
            "kospi_median": round(kospi.median(), 3),
            "kospi_std": round(kospi.std(), 3),
            "positive_rate": round((kospi > 0).mean() * 100, 1),
            "best": round(kospi.max(), 3),
            "worst": round(kospi.min(), 3),
            "gap_mean": round(gap.mean(), 3),
        }
    return grades


def sector_analysis(df: pd.DataFrame) -> dict:
    """섹터별 상관 분석."""
    sectors = {}
    for col, name in KR_SECTOR_MAP.items():
        kr_col = col.replace("_chg", "")  # kr_semi_chg → kr_semi
        if kr_col in df.columns:
            corr_score = round(df["score_ewy"].corr(df[kr_col]), 4)
            corr_combined = round(df["combined_ewy"].corr(df[kr_col]), 4)
            actual_mean = round(df[kr_col].mean(), 3)
            sectors[name] = {
                "corr_with_score": corr_score,
                "corr_with_combined": corr_combined,
                "actual_mean": actual_mean,
                "days": len(df[kr_col].dropna()),
            }
    return sectors


def rules_analysis(df: pd.DataFrame) -> dict:
    """특수 룰 발동 분석."""
    rule_names = ["VIX_SPIKE", "VIX_HIGH", "SOXX_CRASH", "NASDAQ_CIRCUIT",
                  "TRIPLE_BULL", "MARKET_CRASH"]
    analysis = {}
    for rule in rule_names:
        triggered = df[df["rules"].apply(lambda x: rule in x)]
        if len(triggered) == 0:
            continue
        kospi = triggered["kospi_actual"]
        analysis[rule] = {
            "count": len(triggered),
            "kospi_mean": round(kospi.mean(), 3),
            "kospi_worst": round(kospi.min(), 3),
            "positive_rate": round((kospi > 0).mean() * 100, 1),
            "dates": triggered["date"].tolist()[:5],
        }
    return analysis


def extreme_analysis(df: pd.DataFrame) -> dict:
    """극단 상황 분석."""
    strong_bear = df[df["score_ewy"] <= -50]
    strong_bull = df[df["score_ewy"] >= 50]

    result = {}
    if len(strong_bear) > 0:
        result["strong_bear"] = {
            "count": len(strong_bear),
            "kospi_mean": round(strong_bear["kospi_actual"].mean(), 3),
            "kospi_worst": round(strong_bear["kospi_actual"].min(), 3),
            "defense_rate": round((strong_bear["kospi_actual"] < 0).mean() * 100, 1),
        }
    if len(strong_bull) > 0:
        result["strong_bull"] = {
            "count": len(strong_bull),
            "kospi_mean": round(strong_bull["kospi_actual"].mean(), 3),
            "kospi_best": round(strong_bull["kospi_actual"].max(), 3),
            "hit_rate": round((strong_bull["kospi_actual"] > 0).mean() * 100, 1),
        }
    return result


# ================================================================
# 출력
# ================================================================

def print_report(report: dict) -> None:
    """콘솔 리포트 출력."""
    print()
    print("=" * 65)
    print("  US Overnight Signal 백테스트 결과")
    print(f"  기간: {report['period']} | 테스트: {report['test_days']}일")
    print("=" * 65)

    # ─── 실험 A: EWY ───
    a = report["experiment_A"]
    print("\n┌─ 실험 A: EWY 포함 vs 미포함 ─────────────────────┐")
    print(f"│  {'항목':12s} {'EWY 포함':>12s} {'EWY 미포함':>12s} {'차이':>10s} │")
    print(f"│  {'방향적중률':12s} "
          f"{a['with_ewy']['accuracy']['overall_accuracy']:>11.1f}% "
          f"{a['without_ewy']['accuracy']['overall_accuracy']:>11.1f}% "
          f"{a['accuracy_improvement']:>+9.1f}%p│")
    print(f"│  {'상승예측적중':12s} "
          f"{a['with_ewy']['accuracy']['bull_accuracy']:>11.1f}% "
          f"{a['without_ewy']['accuracy']['bull_accuracy']:>11.1f}% "
          f"{a['with_ewy']['accuracy']['bull_accuracy'] - a['without_ewy']['accuracy']['bull_accuracy']:>+9.1f}%p│")
    print(f"│  {'하락예측적중':12s} "
          f"{a['with_ewy']['accuracy']['bear_accuracy']:>11.1f}% "
          f"{a['without_ewy']['accuracy']['bear_accuracy']:>11.1f}% "
          f"{a['with_ewy']['accuracy']['bear_accuracy'] - a['without_ewy']['accuracy']['bear_accuracy']:>+9.1f}%p│")
    print(f"│  {'상관계수':12s} "
          f"{a['with_ewy']['correlation']:>12.4f} "
          f"{a['without_ewy']['correlation']:>12.4f} "
          f"{a['with_ewy']['correlation'] - a['without_ewy']['correlation']:>+10.4f}│")
    print(f"│  {'RMSE':12s} "
          f"{a['with_ewy']['rmse']:>12.4f} "
          f"{a['without_ewy']['rmse']:>12.4f} "
          f"{a['rmse_improvement']:>+10.4f}│")
    print(f"│  EWY↔KOSPI 직접 상관: {a['ewy_kospi_direct_corr']:.4f}           │")
    print(f"└──────────────────────────────────────────────────┘")

    # ─── 실험 B: L2 ───
    b = report["experiment_B"]
    print(f"\n┌─ 실험 B: L1 단독 vs L1+L2 결합 ──────────────────┐")
    print(f"│  {'항목':12s} {'L1 단독':>12s} {'L1+L2':>12s} {'차이':>10s} │")
    print(f"│  {'방향적중률':12s} "
          f"{b['l1_only']['accuracy']['overall_accuracy']:>11.1f}% "
          f"{b['l1_plus_l2']['accuracy']['overall_accuracy']:>11.1f}% "
          f"{b['accuracy_improvement']:>+9.1f}%p│")
    print(f"│  {'상관계수':12s} "
          f"{b['l1_only']['correlation']:>12.4f} "
          f"{b['l1_plus_l2']['correlation']:>12.4f} "
          f"{b['l1_plus_l2']['correlation'] - b['l1_only']['correlation']:>+10.4f}│")
    print(f"│  L2 보정 활성 {b['l2_active_days']}일 | L2 자체 적중률 {b['l2_direction_accuracy']:.1f}%     │")
    print(f"└──────────────────────────────────────────────────┘")

    # ─── 실험 C: Kill ───
    c = report["experiment_C"]
    print(f"\n┌─ 실험 C: Kill Signal 효과 ────────────────────────┐")
    print(f"│  Kill 발동일:  {c['kill_days']:>3d}일 | KOSPI 평균 {c['kill_kospi_mean']:>+.3f}%         │")
    print(f"│  정상일:       {c['normal_days']:>3d}일 | KOSPI 평균 {c['normal_kospi_mean']:>+.3f}%         │")
    print(f"│  Kill 방어율:  {c['kill_defense_rate']:.0f}% (실제 하락한 비율)              │")
    if c.get("sector_kill_detail"):
        for s, info in c["sector_kill_detail"].items():
            print(f"│    {s:6s}: {info['count']}회, 평균 {info['mean_chg']:+.3f}%, "
                  f"방어 {info['defense_rate']:.0f}%          │")
    print(f"│  특수룰 발동:  {c['rule_days']:>3d}일 | KOSPI 평균 {c['rule_kospi_mean']:>+.3f}%         │")
    print(f"└──────────────────────────────────────────────────┘")

    # ─── Score 구간별 ───
    g = report["grade_analysis"]
    print(f"\n┌─ Score 구간별 KOSPI 반응 ─────────────────────────┐")
    print(f"│  {'구간':12s} {'건수':>5s} {'평균':>7s} {'상승률':>7s} {'최선':>7s} {'최악':>7s}│")
    for grade_name in ["STRONG_BULL", "MILD_BULL", "NEUTRAL", "MILD_BEAR", "STRONG_BEAR"]:
        info = g.get(grade_name)
        if info:
            print(f"│  {grade_name:12s} {info['count']:>5d} "
                  f"{info['kospi_mean']:>+6.3f}% "
                  f"{info['positive_rate']:>5.1f}% "
                  f"{info['best']:>+6.3f}% "
                  f"{info['worst']:>+6.3f}%│")
    print(f"└──────────────────────────────────────────────────┘")

    # ─── 섹터 상관 ───
    s = report["sector_analysis"]
    print(f"\n┌─ 섹터별 Score↔실적 상관계수 ───────────────────────┐")
    print(f"│  {'섹터':8s} {'Score상관':>10s} {'Combined':>10s} {'평균등락':>8s}      │")
    for name, info in sorted(s.items(), key=lambda x: abs(x[1]["corr_with_score"]), reverse=True):
        print(f"│  {name:8s} {info['corr_with_score']:>+10.4f} "
              f"{info['corr_with_combined']:>+10.4f} "
              f"{info['actual_mean']:>+7.3f}%     │")
    print(f"└──────────────────────────────────────────────────┘")

    # ─── 특수 룰 ───
    r = report.get("special_rules_analysis", {})
    if r:
        print(f"\n┌─ 특수 룰 발동 분석 ──────────────────────────────┐")
        for rule, info in r.items():
            print(f"│  {rule:16s}: {info['count']}회, KOSPI 평균 {info['kospi_mean']:+.3f}%, "
                  f"최악 {info['kospi_worst']:+.3f}%│")
        print(f"└──────────────────────────────────────────────────┘")

    # ─── 극단 상황 ───
    e = report.get("extreme_analysis", {})
    if e:
        print(f"\n┌─ 극단 상황 분석 ────────────────────────────────┐")
        if "strong_bear" in e:
            sb = e["strong_bear"]
            print(f"│  STRONG_BEAR(≤-50): {sb['count']}일, KOSPI {sb['kospi_mean']:+.3f}%, "
                  f"방어율 {sb['defense_rate']:.0f}%   │")
        if "strong_bull" in e:
            sb = e["strong_bull"]
            print(f"│  STRONG_BULL(≥+50): {sb['count']}일, KOSPI {sb['kospi_mean']:+.3f}%, "
                  f"적중률 {sb['hit_rate']:.0f}%   │")
        print(f"└──────────────────────────────────────────────────┘")

    # ─── 핵심 결론 ───
    print(f"\n{'='*65}")
    print("  핵심 결론")
    print(f"{'='*65}")
    print(f"  {a['summary']}")
    print(f"  {b['summary']}")
    print(f"  {c['summary']}")
    print(f"{'='*65}")


# ================================================================
# 메인
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="US Overnight Signal 백테스트")
    parser.add_argument("--warmup", type=int, default=60,
                        help="워밍업 기간 (기본: 60일)")
    args = parser.parse_args()

    report = run_backtest(warmup=args.warmup)

    # 콘솔 출력
    print_report(report)

    # JSON 저장
    # kills 딕셔너리를 직렬화 가능하게 정리
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"리포트 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
