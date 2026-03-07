"""
블라인드 테스트 — 4대 전략 업그레이드 수익률 검증

기존 v10.3 백테스트 결과에 전략1(레짐 부스트), 전략3(DART AVOID),
전략4(8소스 앙상블) 효과를 역산하여 성과 비교.

테스트 항목:
  A. 레짐 부스트 효과: position_multiplier 적용 vs 미적용
  B. DART AVOID 필터 효과: 유상증자/관리종목 제거 시 MDD 개선
  C. 8소스 앙상블 vs 5소스: 다중시그널 종목 집중 시 승률 차이
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
KOSPI_CSV = DATA_DIR / "kospi_index.csv"
PROCESSED_DIR = DATA_DIR / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  유틸리티
# ══════════════════════════════════════════════

def classify_regime(close: float, ma20: float, ma60: float, rv_pct: float) -> tuple[str, int]:
    """v10.3 레짐 판정 재현."""
    if close > ma20:
        return ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
    elif close > ma60:
        return ("BEAR", 2)
    else:
        return ("CRISIS", 0)


def calc_macro_score(
    ma20_slope: float, ma60_slope: float, rv_declining: bool,
    us_grade: str, vix_level: float, regime: str, next_regime_direction: int,
) -> tuple[int, float]:
    """매크로 점수 → position_multiplier 간이 계산."""
    score = 0

    # 레짐 전환 방향 (40점)
    if next_regime_direction > 0:
        score += 40
    elif next_regime_direction == 0:
        score += 20

    # MA 기울기 (20점)
    if ma20_slope > 0:
        score += min(10, ma20_slope * 5)
    if ma60_slope > 0:
        score += min(10, ma60_slope * 5)

    # RV 안정 (15점)
    if rv_declining:
        score += 15

    # US 정렬 (15점)
    us_map = {"STRONG_BULL": 15, "MILD_BULL": 12, "NEUTRAL": 7, "MILD_BEAR": 3, "STRONG_BEAR": 0}
    score += us_map.get(us_grade, 7)

    # VIX 안정 (10점)
    if vix_level < 15:
        score += 10
    elif vix_level < 20:
        score += 7
    elif vix_level < 25:
        score += 3

    score = min(100, max(0, int(score)))

    # 등급 → multiplier
    if score >= 75:
        mult = 1.3
    elif score >= 60:
        mult = 1.2
    elif score >= 45:
        mult = 1.0
    elif score >= 30:
        mult = 0.7
    else:
        mult = 0.5

    return score, mult


def load_kospi_with_regime() -> pd.DataFrame:
    """KOSPI 일봉 + 레짐 + RV 계산."""
    df = pd.read_csv(KOSPI_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    log_ret = np.log(df["close"] / df["close"].shift(1))
    rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["rv_pct"] = rv20.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["daily_ret"] = df["close"].pct_change()

    # 레짐 판정
    regimes, slots = [], []
    for _, row in df.iterrows():
        if pd.isna(row["ma20"]) or pd.isna(row["ma60"]) or pd.isna(row["rv_pct"]):
            regimes.append(None)
            slots.append(0)
        else:
            r, s = classify_regime(row["close"], row["ma20"], row["ma60"], row["rv_pct"])
            regimes.append(r)
            slots.append(s)
    df["regime"] = regimes
    df["slots"] = slots

    # MA 기울기
    df["ma20_slope"] = (df["ma20"] / df["ma20"].shift(5) - 1) * 100
    df["ma60_slope"] = (df["ma60"] / df["ma60"].shift(10) - 1) * 100

    # RV 변화 (5일전 대비)
    df["rv_declining"] = df["rv_pct"] < df["rv_pct"].shift(5)

    return df


# ══════════════════════════════════════════════
#  테스트 A: 레짐 부스트 효과
# ══════════════════════════════════════════════

def test_regime_boost(df: pd.DataFrame):
    """
    v10.3 백테스트 시뮬레이션:
    - 기존: 슬롯 수만큼 균등 배분 (1.0x 고정)
    - 신규: 슬롯 수 × position_multiplier로 배분 비율 조절

    시뮬: 매일 KOSPI 수익률 × (슬롯비율) → 누적수익률 비교
    레짐 부스트가 상승장에서 공격적, 하락장에서 방어적이면 알파 생성
    """
    logger.info("=" * 60)
    logger.info("  테스트 A: 레짐 부스트 효과 (3년)")
    logger.info("=" * 60)

    # 유효 데이터 (레짐 판정 가능한 구간)
    valid = df.dropna(subset=["regime", "ma20_slope", "rv_pct"]).copy()
    valid = valid[valid["Date"] >= "2023-06-01"]  # 충분한 히스토리 확보 후

    # 매일 매크로 점수 + multiplier 계산
    scores, mults = [], []
    for _, row in valid.iterrows():
        regime = row["regime"]
        regime_order = {"BULL": 3, "CAUTION": 2, "BEAR": 1, "CRISIS": 0}
        cur_order = regime_order.get(regime, 2)

        # 다음 5일 평균 레짐 방향
        ma20_s = row["ma20_slope"] if not pd.isna(row["ma20_slope"]) else 0
        ma60_s = row["ma60_slope"] if not pd.isna(row["ma60_slope"]) else 0

        direction = 1 if ma20_s > 0 and ma60_s > 0 else (-1 if ma20_s < 0 and ma60_s < 0 else 0)

        rv_dec = bool(row["rv_declining"]) if not pd.isna(row["rv_declining"]) else False
        rv_pct = float(row["rv_pct"]) if not pd.isna(row["rv_pct"]) else 0.5

        s, m = calc_macro_score(ma20_s, ma60_s, rv_dec, "NEUTRAL", 18, regime, direction)
        scores.append(s)
        mults.append(m)

    valid = valid.copy()
    valid["macro_score"] = scores
    valid["multiplier"] = mults
    valid["next_ret"] = valid["daily_ret"].shift(-1)  # 다음날 수익률

    valid = valid.dropna(subset=["next_ret"])

    # 시뮬레이션: 매일 슬롯 비율만큼 투자
    max_slots = 5

    # 기존: 슬롯/5 비율로 투자 (부스트 없음)
    valid["exposure_base"] = valid["slots"] / max_slots
    valid["pnl_base"] = valid["next_ret"] * valid["exposure_base"]

    # 신규: (슬롯/5) × multiplier (상한 1.0)
    valid["exposure_new"] = np.minimum(valid["slots"] / max_slots * valid["multiplier"], 1.0)
    valid["pnl_new"] = valid["next_ret"] * valid["exposure_new"]

    # 누적 수익률
    cum_base = (1 + valid["pnl_base"]).cumprod()
    cum_new = (1 + valid["pnl_new"]).cumprod()

    ret_base = (cum_base.iloc[-1] - 1) * 100
    ret_new = (cum_new.iloc[-1] - 1) * 100

    # MDD 계산
    def calc_mdd(cum_series):
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        return dd.min() * 100

    mdd_base = calc_mdd(cum_base)
    mdd_new = calc_mdd(cum_new)

    # 레짐별 통계
    logger.info("")
    logger.info("  %-10s %6s %8s %8s %8s", "레짐", "일수", "평균mult", "기존노출", "신규노출")
    logger.info("  " + "-" * 50)
    for regime in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
        subset = valid[valid["regime"] == regime]
        if len(subset) == 0:
            continue
        logger.info("  %-10s %6d %8.2f %8.1f%% %8.1f%%",
                     regime, len(subset),
                     subset["multiplier"].mean(),
                     subset["exposure_base"].mean() * 100,
                     subset["exposure_new"].mean() * 100)

    # multiplier 분포
    logger.info("")
    logger.info("  Multiplier 분포:")
    for m in [0.5, 0.7, 1.0, 1.2, 1.3]:
        cnt = (valid["multiplier"] == m).sum()
        pct = cnt / len(valid) * 100
        logger.info("    x%.1f: %4d일 (%.1f%%)", m, cnt, pct)

    logger.info("")
    logger.info("  ┌─────────────────────────────────────────┐")
    logger.info("  │  결과 비교 (KOSPI 연동 시뮬레이션)      │")
    logger.info("  ├─────────────────────────────────────────┤")
    logger.info("  │  기존(1.0x 고정): 수익률 %+.1f%%, MDD %.1f%% │", ret_base, mdd_base)
    logger.info("  │  신규(부스트):     수익률 %+.1f%%, MDD %.1f%% │", ret_new, mdd_new)
    logger.info("  │  알파:            %+.1f%%p              │", ret_new - ret_base)
    logger.info("  │  MDD 개선:        %.1f%%p              │", mdd_new - mdd_base)
    logger.info("  └─────────────────────────────────────────┘")

    return {
        "base_return": round(ret_base, 2),
        "new_return": round(ret_new, 2),
        "alpha": round(ret_new - ret_base, 2),
        "base_mdd": round(mdd_base, 2),
        "new_mdd": round(mdd_new, 2),
        "mdd_improvement": round(mdd_new - mdd_base, 2),
        "trading_days": len(valid),
    }


# ══════════════════════════════════════════════
#  테스트 B: DART AVOID 필터 효과
# ══════════════════════════════════════════════

def test_avoid_filter():
    """
    유상증자/관리종목/거래정지 종목의 과거 수익률을 측정.
    이런 종목을 사전 제거했다면 얼마나 손실을 피했는지 계산.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  테스트 B: DART AVOID 필터 효과")
    logger.info("=" * 60)

    # 현재 AVOID 목록 로드
    avoid_path = DATA_DIR / "dart_event_signals.json"
    if not avoid_path.exists():
        logger.info("  dart_event_signals.json 없음 — 스킵")
        return None

    with open(avoid_path, encoding="utf-8") as f:
        de = json.load(f)

    avoid_list = de.get("avoid_list", [])
    avoid_tickers = set()
    avoid_details = {}
    for item in avoid_list:
        t = item.get("ticker", "")
        if t:
            avoid_tickers.add(t)
            avoid_details[t] = item.get("event", "")

    logger.info("  AVOID 종목: %d개", len(avoid_tickers))

    # 각 AVOID 종목의 최근 수익률 측정
    results = []
    for ticker in avoid_tickers:
        parquet = PROCESSED_DIR / f"{ticker}.parquet"
        if not parquet.exists():
            continue
        try:
            df = pd.read_parquet(parquet)
            if len(df) < 20:
                continue
            # 최근 20일 수익률
            ret_20d = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
            # 최근 5일 수익률
            ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100
            results.append({
                "ticker": ticker,
                "event": avoid_details.get(ticker, ""),
                "ret_5d": round(ret_5d, 2),
                "ret_20d": round(ret_20d, 2),
                "close": float(df["close"].iloc[-1]),
            })
        except Exception:
            continue

    if not results:
        logger.info("  AVOID 종목 parquet 데이터 없음")
        return None

    logger.info("")
    logger.info("  %-8s %-14s %8s %8s", "티커", "이벤트", "5일수익", "20일수익")
    logger.info("  " + "-" * 42)

    total_5d = []
    total_20d = []
    for r in sorted(results, key=lambda x: x["ret_20d"]):
        logger.info("  %-8s %-14s %+7.1f%% %+7.1f%%",
                     r["ticker"], r["event"][:14], r["ret_5d"], r["ret_20d"])
        total_5d.append(r["ret_5d"])
        total_20d.append(r["ret_20d"])

    avg_5d = np.mean(total_5d) if total_5d else 0
    avg_20d = np.mean(total_20d) if total_20d else 0
    negative_count = sum(1 for x in total_20d if x < 0)

    logger.info("")
    logger.info("  AVOID 종목 평균 수익률: 5일 %+.1f%%, 20일 %+.1f%%", avg_5d, avg_20d)
    logger.info("  20일 손실 종목: %d/%d (%.0f%%)", negative_count, len(total_20d),
                negative_count / len(total_20d) * 100 if total_20d else 0)

    verdict = "효과적" if avg_20d < 0 else "무효 (해당 종목들이 오히려 상승)"
    logger.info("  판정: AVOID 필터 → %s", verdict)

    return {
        "avoid_count": len(avoid_tickers),
        "measured_count": len(results),
        "avg_5d": round(avg_5d, 2),
        "avg_20d": round(avg_20d, 2),
        "negative_pct": round(negative_count / len(total_20d) * 100, 1) if total_20d else 0,
    }


# ══════════════════════════════════════════════
#  테스트 C: 8소스 앙상블 vs 5소스
# ══════════════════════════════════════════════

def test_ensemble_expansion():
    """
    현재 8소스 스캔 결과에서:
    - 2소스 이상 교차 종목 vs 1소스 단독 종목의 점수/품질 비교
    - 세력감지+이벤트 소스 추가로 교차 확인 강화 효과 측정
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  테스트 C: 8소스 앙상블 vs 5소스 (교차검증 효과)")
    logger.info("=" * 60)

    picks_path = DATA_DIR / "tomorrow_picks.json"
    if not picks_path.exists():
        logger.info("  tomorrow_picks.json 없음 — 스킵")
        return None

    with open(picks_path, encoding="utf-8") as f:
        picks = json.load(f)

    all_picks = picks.get("picks", [])
    if not all_picks:
        logger.info("  추천 데이터 없음")
        return None

    # score 키 통일 (total_score → score)
    for p in all_picks:
        if "total_score" in p and "score" not in p:
            p["score"] = p["total_score"]

    # 소스 개수별 분류
    multi_source = []  # 2개 이상 소스
    single_source = []  # 1개 소스
    new_source_hits = []  # 세력감지 or 이벤트가 포함된 종목

    for p in all_picks:
        sources = p.get("sources", [])
        n_sources = p.get("n_sources", len(sources))
        score = p.get("score", p.get("total_score", 0))

        if n_sources >= 2:
            multi_source.append(p)
        else:
            single_source.append(p)

        # 신규 소스(세력감지, 이벤트) 포함 여부
        new_sources = {"세력감지", "이벤트"}
        if any(s in new_sources for s in sources):
            new_source_hits.append(p)

    # 통계
    avg_multi = np.mean([p["score"] for p in multi_source]) if multi_source else 0
    avg_single = np.mean([p["score"] for p in single_source]) if single_source else 0

    logger.info("")
    logger.info("  다중소스(2+): %d종목, 평균 %.1f점", len(multi_source), avg_multi)
    logger.info("  단독소스(1):  %d종목, 평균 %.1f점", len(single_source), avg_single)
    logger.info("  점수 차이:    %+.1f점 (다중소스 우위)", avg_multi - avg_single)

    logger.info("")
    logger.info("  신규 소스(세력감지+이벤트) 기여:")
    logger.info("    히트 종목: %d개 / 전체 %d개 (%.1f%%)",
                len(new_source_hits), len(all_picks),
                len(new_source_hits) / len(all_picks) * 100 if all_picks else 0)

    # 신규 소스 종목의 등급 분포
    if new_source_hits:
        grades = {}
        for p in new_source_hits:
            g = p.get("grade", "미분류")
            grades[g] = grades.get(g, 0) + 1
        logger.info("    등급 분포: %s", ", ".join(f"{k}:{v}" for k, v in sorted(grades.items())))

    # 다중소스 종목 중 TOP5 분석
    logger.info("")
    logger.info("  다중소스 종목 TOP5:")
    for p in sorted(multi_source, key=lambda x: -x["score"])[:5]:
        sources_str = "+".join(p.get("sources", []))
        logger.info("    %s(%s) %.1f점 [%s]",
                     p.get("name", ""), p.get("ticker", ""),
                     p["score"], sources_str)

    # 기존 5소스에서 잡히지 않았을 종목 (세력감지/이벤트 단독)
    new_only = [p for p in all_picks
                if all(s in {"세력감지", "이벤트"} for s in p.get("sources", []))]

    logger.info("")
    logger.info("  신규 소스 단독 종목 (기존 5소스에서 미탐지): %d개", len(new_only))
    for p in sorted(new_only, key=lambda x: -x["score"])[:5]:
        logger.info("    %s(%s) %.1f점 [%s]",
                     p.get("name", ""), p.get("ticker", ""),
                     p["score"], "+".join(p.get("sources", [])))

    # 과거 수익률 비교 (다중소스 vs 단독소스)
    logger.info("")
    logger.info("  과거 5일 수익률 비교 (parquet 기반):")

    def measure_group_returns(group, label):
        rets = []
        for p in group:
            ticker = p.get("ticker", "")
            pq = PROCESSED_DIR / f"{ticker}.parquet"
            if not pq.exists():
                continue
            try:
                df = pd.read_parquet(pq)
                if len(df) < 10:
                    continue
                # 최근 5일 수익률 (과거 데이터이므로 forward가 아닌 backward 측정)
                ret = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
                rets.append(ret)
            except Exception:
                continue
        if rets:
            avg = np.mean(rets)
            win = sum(1 for r in rets if r > 0)
            logger.info("    %s: %d종목, 평균 %+.1f%%, 양수 %d/%d (%.0f%%)",
                         label, len(rets), avg, win, len(rets), win / len(rets) * 100)
            return {"count": len(rets), "avg_ret": round(avg, 2), "win_rate": round(win / len(rets) * 100, 1)}
        return None

    multi_ret = measure_group_returns(multi_source, "다중소스(2+)")
    single_ret = measure_group_returns(single_source, "단독소스(1) ")

    return {
        "multi_count": len(multi_source),
        "single_count": len(single_source),
        "avg_multi_score": round(avg_multi, 1),
        "avg_single_score": round(avg_single, 1),
        "new_source_hits": len(new_source_hits),
        "new_only": len(new_only),
        "multi_ret": multi_ret,
        "single_ret": single_ret,
    }


# ══════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════

def main():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  블라인드 테스트 — 4대 전략 업그레이드 수익률 검증      ║")
    logger.info("║  데이터: KOSPI 3년 + 662종목 parquet                   ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # KOSPI 데이터 로드 + 레짐 계산
    logger.info("\nKOSPI 데이터 로드 중...")
    df = load_kospi_with_regime()
    logger.info("  %d일 로드 완료 (레짐: %s)",
                len(df), df["regime"].value_counts().to_dict())

    results = {}

    # 테스트 A: 레짐 부스트
    results["A"] = test_regime_boost(df)

    # 테스트 B: AVOID 필터
    results["B"] = test_avoid_filter()

    # 테스트 C: 앙상블 확대
    results["C"] = test_ensemble_expansion()

    # 종합 판정
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  종합 판정                                             ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")

    if results["A"]:
        a = results["A"]
        verdict_a = "PASS" if a["alpha"] > 0 else "FAIL"
        logger.info("║  A. 레짐 부스트:  알파 %+.1f%%p, MDD %.1f%%p → %s      ║",
                     a["alpha"], a["mdd_improvement"], verdict_a)

    if results["B"]:
        b = results["B"]
        verdict_b = "PASS" if b["avg_20d"] < 0 else "NEUTRAL"
        logger.info("║  B. AVOID 필터:   20일 평균 %+.1f%%, 손실비 %.0f%% → %s ║",
                     b["avg_20d"], b["negative_pct"], verdict_b)

    if results["C"]:
        c = results["C"]
        score_diff = c["avg_multi_score"] - c["avg_single_score"]
        verdict_c = "PASS" if score_diff > 0 else "NEUTRAL"
        logger.info("║  C. 앙상블 확대:  다중소스 %+.1f점 우위 → %s            ║",
                     score_diff, verdict_c)

    logger.info("╚══════════════════════════════════════════════════════════╝")

    # JSON 저장
    output_path = DATA_DIR / "blind_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("\n결과 저장: %s", output_path)


if __name__ == "__main__":
    main()
