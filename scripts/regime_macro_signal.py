"""레짐 전환 매크로 시그널 — Timefolio 벤치마크 전략 1

KOSPI 레짐(BULL/CAUTION/BEAR/CRISIS) + US Overnight 시그널을 결합하여
레짐 전환 방향과 매크로 방향 시그널을 생성한다.

기존 kospi_regime.json은 현재 상태만 제공하지만,
이 스크립트는 전환 방향, 속도, 포지션 배수를 추가 판단한다.

입력:
  - data/kospi_index.csv (KOSPI 일봉)
  - data/kospi_regime.json (현재 레짐)
  - data/us_market/overnight_signal.json (US 시그널)

출력:
  - data/regime_macro_signal.json

Usage:
    python scripts/regime_macro_signal.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
KOSPI_CSV = DATA_DIR / "kospi_index.csv"
REGIME_JSON = DATA_DIR / "kospi_regime.json"
US_SIGNAL_JSON = DATA_DIR / "us_market" / "overnight_signal.json"
OUTPUT_PATH = DATA_DIR / "regime_macro_signal.json"

# 레짐 순서 (상위→하위)
REGIME_ORDER = {"BULL": 3, "CAUTION": 2, "BEAR": 1, "CRISIS": 0}
REGIME_NAMES = {3: "BULL", 2: "CAUTION", 1: "BEAR", 0: "CRISIS"}

# 등급 체계
GRADE_MAP = [
    (75, "공격적 매수", 1.3),
    (60, "적극 매수", 1.2),
    (45, "보통", 1.0),
    (30, "방어적", 0.7),
    (0, "현금비중 확대", 0.5),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_regime(close: float, ma20: float, ma60: float,
                    rv_pct: float) -> str:
    """단일 시점 레짐 분류 (kospi_regime.json 로직 재현)"""
    if close > ma20:
        return "BULL" if rv_pct < 50 else "CAUTION"
    elif close > ma60:
        return "BEAR"
    else:
        return "CRISIS"


def calc_rv_percentile(returns: pd.Series, window: int = 20) -> pd.Series:
    """실현변동성의 롤링 백분위"""
    rv = returns.rolling(window).std() * np.sqrt(252) * 100
    rv_rank = rv.rolling(252, min_periods=60).rank(pct=True) * 100
    return rv_rank.fillna(50)


def calc_slope(series: pd.Series, window: int) -> float:
    """최근 window일 기울기 (% 변화율/일)"""
    if len(series) < window + 1:
        return 0.0
    recent = series.iloc[-window:]
    if recent.iloc[0] == 0:
        return 0.0
    return (recent.iloc[-1] / recent.iloc[0] - 1) * 100


def main():
    logger.info("=" * 60)
    logger.info("  레짐 전환 매크로 시그널 — 전략 1")
    logger.info("=" * 60)

    # ── 1. KOSPI 데이터 로드 ──
    if not KOSPI_CSV.exists():
        logger.error("KOSPI CSV 없음: %s", KOSPI_CSV)
        return

    df = pd.read_csv(KOSPI_CSV, parse_dates=["Date"]).sort_values("Date")
    df = df.tail(120)  # 최근 120일

    if len(df) < 60:
        logger.error("KOSPI 데이터 부족 (%d행)", len(df))
        return

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["returns"] = df["close"].pct_change()
    df["rv_pct"] = calc_rv_percentile(df["returns"])

    # ── 2. 레짐 히스토리 계산 (최근 60일) ──
    regime_history = []
    for _, row in df.tail(60).iterrows():
        if pd.isna(row["ma20"]) or pd.isna(row["ma60"]):
            continue
        r = classify_regime(row["close"], row["ma20"], row["ma60"], row["rv_pct"])
        regime_history.append(r)

    # 현재 레짐
    regime_data = load_json(REGIME_JSON)
    current_regime = regime_data.get("regime", "CAUTION")
    current_close = regime_data.get("close", df["close"].iloc[-1])
    current_ma20 = regime_data.get("ma20", df["ma20"].iloc[-1])
    current_ma60 = regime_data.get("ma60", df["ma60"].iloc[-1])

    logger.info("  현재 레짐: %s (KOSPI %.1f)", current_regime, current_close)

    # ── 3. 전환 방향 판단 (6개 시그널, 각 100점 만점) ──
    scores = {}

    # 3-1. MA20 기울기 (5일)
    ma20_slope = calc_slope(df["ma20"].dropna(), 5)
    scores["ma20_slope"] = round(ma20_slope, 2)

    # 3-2. MA60 기울기 (10일)
    ma60_slope = calc_slope(df["ma60"].dropna(), 10)
    scores["ma60_slope"] = round(ma60_slope, 2)

    # 3-3. close-MA20 갭 (레짐 경계 접근 감지)
    if current_ma20 > 0:
        gap_ma20_pct = (current_close / current_ma20 - 1) * 100
    else:
        gap_ma20_pct = 0
    scores["gap_ma20_pct"] = round(gap_ma20_pct, 1)

    # 3-4. RV20 추이 (변동성 안정화)
    rv_current = df["rv_pct"].iloc[-1] if not pd.isna(df["rv_pct"].iloc[-1]) else 50
    rv_5d_ago = df["rv_pct"].iloc[-6] if len(df) >= 6 and not pd.isna(df["rv_pct"].iloc[-6]) else 50
    rv_declining = rv_current < rv_5d_ago
    scores["rv_current"] = round(float(rv_current), 1)
    scores["rv_declining"] = bool(rv_declining)

    # 3-5. US Overnight 시그널
    us_signal = load_json(US_SIGNAL_JSON)
    us_grade = us_signal.get("final_grade", "NEUTRAL")
    vix_level = us_signal.get("vix", {}).get("level", 20)
    ewy_data = us_signal.get("index_direction", {}).get("EWY", {})
    ewy_5d = ewy_data.get("ret_5d", 0)
    scores["vix_level"] = round(float(vix_level), 1)
    scores["us_grade"] = us_grade
    scores["ewy_5d"] = round(float(ewy_5d), 2)

    # ── 4. 전환 방향 판정 ──
    current_order = REGIME_ORDER.get(current_regime, 2)

    # 최근 5일 레짐 히스토리
    recent_5 = regime_history[-5:] if len(regime_history) >= 5 else regime_history
    recent_orders = [REGIME_ORDER.get(r, 2) for r in recent_5]

    # 상위 레짐 접근 여부
    if len(recent_orders) >= 3:
        avg_recent = np.mean(recent_orders[-3:])
        trend_up = avg_recent >= current_order
    else:
        trend_up = ma20_slope > 0

    # 전환 방향 텍스트
    next_regime_order = min(current_order + 1, 3)
    next_regime = REGIME_NAMES.get(next_regime_order, "BULL")
    prev_regime_order = max(current_order - 1, 0)
    prev_regime = REGIME_NAMES.get(prev_regime_order, "CRISIS")

    if trend_up and ma20_slope > 0:
        transition_dir = f"{next_regime} 접근"
    elif not trend_up and ma20_slope < 0:
        transition_dir = f"{prev_regime} 접근"
    else:
        transition_dir = f"{current_regime} 유지"

    # ── 5. 매크로 점수 계산 (100점) ──
    macro_score = 0

    # 축1: 레짐 전환 방향 (40점)
    if trend_up:
        if ma20_slope > 1.0 and ma60_slope > 0:
            macro_score += 40  # 강한 상승 전환
        elif ma20_slope > 0.5:
            macro_score += 30  # 완만한 상승
        elif ma20_slope > 0:
            macro_score += 20  # 미약한 상승
        else:
            macro_score += 10  # 하락 둔화
    else:
        if ma20_slope > 0:
            macro_score += 15  # 하락 둔화
        else:
            macro_score += 0

    # 축2: MA 기울기 합산 (20점)
    if ma20_slope > 0 and ma60_slope > 0:
        macro_score += 20
    elif ma20_slope > 0:
        macro_score += 12
    elif ma60_slope > 0:
        macro_score += 8
    else:
        macro_score += 0

    # 축3: RV 안정성 (15점)
    if rv_declining and rv_current < 60:
        macro_score += 15  # 변동성 감소 + 낮은 수준
    elif rv_declining:
        macro_score += 10  # 변동성 감소
    elif rv_current < 50:
        macro_score += 8   # 낮은 변동성
    else:
        macro_score += 0

    # 축4: US 정렬 (15점)
    us_bullish = us_grade in ("STRONG_BULL", "MILD_BULL")
    us_bearish = us_grade in ("STRONG_BEAR", "MILD_BEAR")
    if us_bullish and trend_up:
        macro_score += 15
    elif us_bullish:
        macro_score += 10
    elif us_grade == "NEUTRAL":
        macro_score += 7
    elif us_bearish and not trend_up:
        macro_score += 3
    else:
        macro_score += 0

    # 축5: VIX 안정 (10점)
    if vix_level < 15:
        macro_score += 10
    elif vix_level < 20:
        macro_score += 8
    elif vix_level < 25:
        macro_score += 4
    else:
        macro_score += 0

    macro_score = min(macro_score, 100)

    # ── 6. 전환 확률 추정 ──
    # 간단 휴리스틱: 매크로 점수 기반
    transition_prob = min(macro_score + 5, 95) if trend_up else max(100 - macro_score - 5, 5)

    # ── 7. 등급 + 포지션 배수 ──
    macro_grade = "현금비중 확대"
    position_mult = 0.5
    for threshold, grade, mult in GRADE_MAP:
        if macro_score >= threshold:
            macro_grade = grade
            position_mult = mult
            break

    # 레짐 갭 설명
    regime_gap_desc = (
        f"MA20 위 {gap_ma20_pct:.1f}%, RV {rv_current:.0f}%ile → {current_regime}"
    )
    scores["regime_gap"] = regime_gap_desc

    # 추천 메시지
    if macro_score >= 70:
        recommendation = f"{current_regime}→{next_regime} 전환 가능성 높음. 포지션 확대 유효."
    elif macro_score >= 50:
        recommendation = f"{current_regime} 유지 가능성. 기본 포지션 유지."
    elif macro_score >= 30:
        recommendation = f"하락 리스크 존재. 신규 진입 신중."
    else:
        recommendation = f"{current_regime}→{prev_regime} 전환 우려. 현금비중 확대 권장."

    # ── 8. 출력 ──
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "current_regime": current_regime,
        "transition_direction": transition_dir,
        "transition_probability": round(transition_prob),
        "macro_score": macro_score,
        "macro_grade": macro_grade,
        "position_multiplier": position_mult,
        "signals": scores,
        "recommendation": recommendation,
        "regime_history_5d": recent_5,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("── 결과 ──")
    logger.info("  레짐: %s → %s", current_regime, transition_dir)
    logger.info("  매크로 점수: %d/100 → %s (x%.1f)", macro_score, macro_grade, position_mult)
    logger.info("  전환 확률: %d%%", transition_prob)
    logger.info("  MA20 기울기(5d): %.2f%%, MA60 기울기(10d): %.2f%%", ma20_slope, ma60_slope)
    logger.info("  VIX: %.1f, US: %s, EWY 5d: %.1f%%", vix_level, us_grade, ewy_5d)
    logger.info("  추천: %s", recommendation)
    logger.info("  저장: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
