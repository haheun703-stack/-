"""JARVIS D+1 방향 예측 엔진 — 메타-앙상블 + 시크릿 팩터 + 피보나치

기존 3개 시스템(눈치엔진, US Overnight, 레짐)의 출력을 메타-앙상블하고,
시크릿 팩터 3개 + 피보나치 레벨 + 이벤트 임팩트를 종합하여
D+1 KOSPI 방향을 예측한다.

입력 (BAT-D G3.5 시점에 이미 생성된 파일들):
  - data/market_sense.json          (눈치엔진)
  - data/us_market/overnight_signal.json  (US Overnight)
  - data/regime_macro_signal.json   (레짐)
  - data/kospi_index.csv            (KOSPI 일봉)
  - data/kospi_investor_flow.csv    (투자자 수급)

출력:
  - data/jarvis_direction.json

Usage:
    python scripts/jarvis_direction_engine.py
    python scripts/jarvis_direction_engine.py --send   # 텔레그램 발송
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date, datetime, timedelta
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
OUTPUT_PATH = DATA_DIR / "jarvis_direction.json"
LOG_PATH = DATA_DIR / "jarvis_direction_log.json"
KOSPI_CSV = DATA_DIR / "kospi_index.csv"
FLOW_CSV = DATA_DIR / "kospi_investor_flow.csv"

# ── 피보나치 상수 ──
FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]
FIB_EXTENSIONS = [1.0, 1.272, 1.618, 2.0, 2.618]

# ── ETF 추천 매핑 ──
ETF_LONG = [
    {"code": "122630", "name": "KODEX 레버리지", "desc": "KOSPI200 2배"},
    {"code": "069500", "name": "KODEX 200", "desc": "KOSPI200 추종"},
]
ETF_SHORT = [
    {"code": "252670", "name": "KODEX 200선물인버스2X", "desc": "KOSPI200 인버스 2배"},
    {"code": "114800", "name": "KODEX 인버스", "desc": "KOSPI200 인버스"},
]
ETF_SAFE = [
    {"code": "132030", "name": "KODEX 골드선물(H)", "desc": "금"},
    {"code": "148070", "name": "KODEX 국고채10년", "desc": "채권"},
]

# ── 기본 가중치 ──
DEFAULT_WEIGHTS = {
    "sense": 0.25,
    "overnight": 0.30,
    "regime": 0.15,
    "secret": 0.30,
}


# ══════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════

def load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _classify_direction(score: float) -> str:
    if score >= 40:
        return "UP"
    elif score >= 15:
        return "UP"
    elif score <= -40:
        return "DOWN"
    elif score <= -15:
        return "DOWN"
    return "FLAT"


# ══════════════════════════════════════════
# 1. 시크릿 팩터 3개
# ══════════════════════════════════════════

def secret_flow_reversal() -> dict:
    """Secret 1: 외국인 수급 반전 감지.

    kospi_investor_flow.csv에서 외국인 N일 연속 매도 후 매수 전환을 감지.
    기존 시스템은 5일 합산만 사용하여 추세 반전을 놓침.
    """
    if not FLOW_CSV.exists():
        return {"score": 0, "detail": "데이터 없음"}

    try:
        df = pd.read_csv(FLOW_CSV, parse_dates=["Date"]).sort_values("Date")
    except Exception:
        return {"score": 0, "detail": "파싱 실패"}

    if len(df) < 7:
        return {"score": 0, "detail": "데이터 부족"}

    recent = df.tail(10)
    foreign = recent["foreign_net"].values
    inst = recent["inst_net"].values

    # 최근 5일 중 4일+ 매도 후, 오늘 매수 전환
    if len(foreign) >= 6:
        last5_sell = sum(1 for x in foreign[-6:-1] if x < 0)
        today_buy = foreign[-1] > 0
        if last5_sell >= 4 and today_buy:
            return {"score": 30, "detail": f"외인 반전 매수 (직전5일 중 {last5_sell}일 매도 후 전환)"}

        # 반대: 5일 매수 후 매도 전환
        last5_buy = sum(1 for x in foreign[-6:-1] if x > 0)
        today_sell = foreign[-1] < 0
        if last5_buy >= 4 and today_sell:
            return {"score": -30, "detail": f"외인 반전 매도 (직전5일 중 {last5_buy}일 매수 후 전환)"}

    # 외인+기관 동시 방향
    if len(foreign) >= 1 and len(inst) >= 1:
        f_today = float(foreign[-1])
        i_today = float(inst[-1])
        if f_today > 0 and i_today > 0:
            return {"score": 15, "detail": "외인+기관 동시 순매수"}
        if f_today < 0 and i_today < 0:
            return {"score": -15, "detail": "외인+기관 동시 순매도"}

    return {"score": 0, "detail": "특이 반전 없음"}


def secret_range_squeeze() -> dict:
    """Secret 2: KOSPI ATR 스퀴즈/확대 패턴.

    일중 고가-저가 Range의 축소/확대를 감지.
    레짐의 실현변동성(20일 종가 기준)과는 별개 정보.
    """
    if not KOSPI_CSV.exists():
        return {"score": 0, "detail": "데이터 없음"}

    try:
        df = pd.read_csv(KOSPI_CSV, parse_dates=["Date"]).sort_values("Date")
    except Exception:
        return {"score": 0, "detail": "파싱 실패"}

    if len(df) < 20:
        return {"score": 0, "detail": "데이터 부족"}

    recent = df.tail(20).copy()
    recent["range_pct"] = (recent["high"] - recent["low"]) / recent["close"] * 100

    atr_5 = recent["range_pct"].tail(5).mean()
    atr_20 = recent["range_pct"].mean()
    squeeze_ratio = atr_5 / atr_20 if atr_20 > 0 else 1.0

    if squeeze_ratio < 0.6:
        # 스퀴즈 → 폭발 임박
        last_2_bullish = (
            recent["close"].iloc[-1] > recent["open"].iloc[-1]
            and recent["close"].iloc[-2] > recent["open"].iloc[-2]
        )
        last_2_bearish = (
            recent["close"].iloc[-1] < recent["open"].iloc[-1]
            and recent["close"].iloc[-2] < recent["open"].iloc[-2]
        )
        if last_2_bullish:
            return {"score": 25, "detail": f"스퀴즈 상방 (ratio {squeeze_ratio:.2f})"}
        elif last_2_bearish:
            return {"score": -25, "detail": f"스퀴즈 하방 (ratio {squeeze_ratio:.2f})"}
        return {"score": 0, "detail": f"스퀴즈 감지 (ratio {squeeze_ratio:.2f}), 방향 불확실"}

    if squeeze_ratio > 1.3:
        # 변동폭 확대 → 추세 지속
        today_up = recent["close"].iloc[-1] > recent["close"].iloc[-2]
        score = 10 if today_up else -10
        direction = "상승" if today_up else "하락"
        return {"score": score, "detail": f"변동폭 확대 {direction} 지속 (ratio {squeeze_ratio:.2f})"}

    return {"score": 0, "detail": f"정상 범위 (ratio {squeeze_ratio:.2f})"}


def secret_decoupling() -> dict:
    """Secret 3: US-KR 디커플링 캐치업/조정.

    미국이 5일간 상승했는데 한국이 따라가지 못했으면
    D+1~D+3에 캐치업 상승 확률이 높음. 반대도 마찬가지.
    """
    us = load_json("us_market/overnight_signal.json")
    if not us:
        return {"score": 0, "detail": "US 데이터 없음"}

    if not KOSPI_CSV.exists():
        return {"score": 0, "detail": "KOSPI 데이터 없음"}

    # US 5일 수익률 (ret_5d는 이미 % 단위: 1.66 = 1.66%)
    idx_dir = us.get("index_direction", {})
    spy_5d = idx_dir.get("SPY", {}).get("ret_5d", 0)
    qqq_5d = idx_dir.get("QQQ", {}).get("ret_5d", 0)
    us_avg = (spy_5d + qqq_5d) / 2  # 이미 %

    # KOSPI 5일 수익률
    try:
        df = pd.read_csv(KOSPI_CSV, parse_dates=["Date"]).sort_values("Date")
    except Exception:
        return {"score": 0, "detail": "KOSPI 파싱 실패"}

    if len(df) < 6:
        return {"score": 0, "detail": "KOSPI 데이터 부족"}

    kr_close = df["close"].values
    kr_5d = (kr_close[-1] / kr_close[-6] - 1) * 100

    gap = us_avg - kr_5d

    if gap > 3.0:
        return {"score": 20, "detail": f"US 앞서감 (gap {gap:+.1f}%p) → KR 캐치업 기대"}
    elif gap > 1.5:
        return {"score": 10, "detail": f"US 소폭 앞서감 (gap {gap:+.1f}%p)"}
    elif gap < -3.0:
        return {"score": -20, "detail": f"KR 과열 (gap {gap:+.1f}%p) → 조정 기대"}
    elif gap < -1.5:
        return {"score": -10, "detail": f"KR 소폭 과열 (gap {gap:+.1f}%p)"}

    return {"score": 0, "detail": f"US-KR 동조 (gap {gap:+.1f}%p)"}


def compute_secret_factors() -> dict:
    """시크릿 팩터 3개 통합."""
    s1 = secret_flow_reversal()
    s2 = secret_range_squeeze()
    s3 = secret_decoupling()

    total = s1["score"] + s2["score"] + s3["score"]
    total = max(-100, min(100, total))

    return {
        "score": total,
        "flow_reversal": s1,
        "range_squeeze": s2,
        "decoupling": s3,
    }


# ══════════════════════════════════════════
# 2. KOSPI 피보나치 레벨
# ══════════════════════════════════════════

def compute_kospi_fibonacci() -> dict:
    """KOSPI 지수 피보나치 되돌림/확장 레벨 계산."""
    if not KOSPI_CSV.exists():
        return {}

    try:
        df = pd.read_csv(KOSPI_CSV, parse_dates=["Date"]).sort_values("Date")
    except Exception:
        return {}

    if len(df) < 20:
        return {}

    current = float(df["close"].iloc[-1])
    results = {}

    for label, window in [("swing_20d", 20), ("swing_60d", 60), ("swing_120d", 120)]:
        if len(df) < window:
            continue

        recent = df.tail(window)
        high = float(recent["high"].max())
        low = float(recent["low"].min())

        if high == low:
            continue

        retracements = {}
        for r in FIB_RATIOS:
            retracements[f"fib_{r}"] = round(high - (high - low) * r, 2)

        extensions = {}
        for r in FIB_EXTENSIONS:
            extensions[f"ext_{r}"] = round(low + (high - low) * r, 2)

        all_levels = sorted(list(retracements.values()) + list(extensions.values()))
        nearest_support = max([l for l in all_levels if l < current], default=None)
        nearest_resistance = min([l for l in all_levels if l > current], default=None)

        position_pct = round((current - low) / (high - low) * 100, 1)

        results[label] = {
            "swing_high": round(high, 2),
            "swing_low": round(low, 2),
            "current": round(current, 2),
            "position_pct": position_pct,
            "retracements": retracements,
            "extensions": extensions,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
        }

    return results


# ══════════════════════════════════════════
# 3. 이벤트 임팩트 스코어
# ══════════════════════════════════════════

def compute_event_impact() -> dict:
    """향후 7일 이벤트 → 방향 영향도."""
    IMPACT_SCORE = {"HIGH": 15, "MEDIUM": 8, "LOW": 3}

    try:
        from src.alpha.event_calendar import EventCalendar
        cal = EventCalendar()
        upcoming = cal.get_upcoming(days=7)
    except Exception:
        upcoming = []

    # market_intelligence의 이벤트도 보조 활용
    intel = load_json("market_intelligence.json")
    key_events = intel.get("key_events", [])

    total = 0
    events_out = []
    today = date.today()

    for ev in upcoming:
        try:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue

        days_until = (ev_date - today).days
        if days_until < 0:
            continue

        impact = IMPACT_SCORE.get(ev.get("impact", "LOW"), 3)
        # 가까울수록 영향 큼
        proximity_mult = max(0.5, 1.5 - days_until * 0.15)
        score = round(impact * proximity_mult)

        events_out.append({
            "name": ev.get("name", ""),
            "date": ev["date"],
            "days_until": days_until,
            "impact_score": score,
        })
        total += score

    # key_events에서 추가 보정 (긍정/부정 이벤트)
    for ke in key_events:
        kr_impact = ke.get("kr_impact_score", 0)
        urgency = ke.get("urgency", "NORMAL")
        if abs(kr_impact) >= 3:
            boost = kr_impact * 3
            if urgency == "BREAKING":
                boost = int(boost * 1.3)
            total += boost

    total = max(-30, min(30, total))

    return {"score": total, "events": events_out}


# ══════════════════════════════════════════
# 4. 적중률 로그 + 자동 검증 + 가중치 조정
# ══════════════════════════════════════════

def update_jarvis_log(result: dict):
    """예측 기록 저장."""
    log = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []

    today = date.today().isoformat()

    existing = next((e for e in log if e["date"] == today), None)
    entry_data = {
        "date": today,
        "meta_score": result["meta_score"],
        "direction": result["direction"],
        "components": {
            "sense": result["components"]["sense"]["score"],
            "overnight": result["components"]["overnight"]["score"],
            "regime": result["components"]["regime"]["score"],
            "secret": result["components"]["secret"]["score"],
        },
        "actual_change_pct": None,
        "hit": None,
    }

    if existing:
        existing.update(entry_data)
    else:
        log.append(entry_data)

    # 최근 60일만 유지
    log = log[-60:]

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def auto_verify() -> int:
    """kospi_index.csv로 미검증 기록 자동 검증. 검증 건수 반환."""
    if not LOG_PATH.exists() or not KOSPI_CSV.exists():
        return 0

    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        return 0

    try:
        kdf = pd.read_csv(KOSPI_CSV, index_col="Date", parse_dates=True).sort_index()
    except Exception:
        return 0

    close_col = "Close" if "Close" in kdf.columns else "close"
    if close_col not in kdf.columns:
        return 0

    verified_count = 0

    for entry in log:
        if entry.get("actual_change_pct") is not None:
            continue

        pred_date = entry["date"]
        try:
            pred_dt = pd.Timestamp(pred_date)
        except Exception:
            continue

        # pred_date 다음 거래일 찾기
        future_dates = kdf.index[kdf.index > pred_dt]
        if len(future_dates) == 0:
            continue

        next_td = future_dates[0]

        # pred_date 당일 또는 직전 거래일
        past_dates = kdf.index[kdf.index <= pred_dt]
        if len(past_dates) == 0:
            continue

        base_td = past_dates[-1]

        base_close = float(kdf.loc[base_td, close_col])
        next_close = float(kdf.loc[next_td, close_col])

        if base_close <= 0:
            continue

        change_pct = round((next_close / base_close - 1) * 100, 2)
        entry["actual_change_pct"] = change_pct

        # 방향 적중 판단
        predicted_up = entry["meta_score"] > 0
        actual_up = change_pct > 0
        entry["hit"] = predicted_up == actual_up

        verified_count += 1

    if verified_count > 0:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    return verified_count


def load_adaptive_weights() -> dict:
    """최근 20일 적중률 기반 동적 가중치."""
    if not LOG_PATH.exists():
        return dict(DEFAULT_WEIGHTS)

    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        return dict(DEFAULT_WEIGHTS)

    verified = [e for e in log if e.get("actual_change_pct") is not None][-20:]
    if len(verified) < 10:
        return dict(DEFAULT_WEIGHTS)

    # 각 컴포넌트별 적중률 계산
    hits = {"sense": 0, "overnight": 0, "regime": 0, "secret": 0}
    for entry in verified:
        actual = entry["actual_change_pct"]
        actual_up = actual > 0
        components = entry.get("components", {})
        for comp_name in hits:
            comp_score = components.get(comp_name, 0)
            predicted_up = comp_score > 0
            if predicted_up == actual_up:
                hits[comp_name] += 1

    # 적중률 비례 가중치 (최소 0.1 보장)
    total_hits = sum(hits.values())
    if total_hits == 0:
        return dict(DEFAULT_WEIGHTS)

    weights = {}
    for k, v in hits.items():
        weights[k] = max(0.1, round(v / total_hits, 3))

    # 합이 1이 되도록 정규화
    w_sum = sum(weights.values())
    if w_sum > 0:
        weights = {k: round(v / w_sum, 3) for k, v in weights.items()}

    return weights


# ══════════════════════════════════════════
# 5. 메타-앙상블 + 종합 판단
# ══════════════════════════════════════════

def compute_direction() -> dict:
    """3개 기존 시스템 + 시크릿 팩터 메타-앙상블."""

    # ── 기존 3시스템 점수 로드 ──
    sense = load_json("market_sense.json")
    overnight = load_json("us_market/overnight_signal.json")
    regime = load_json("regime_macro_signal.json")

    # 정규화 (-100 ~ +100)
    sense_score = float(sense.get("score", 0))
    overnight_score = float(overnight.get("combined_score_100", 0))
    # regime.macro_score는 0~100 → -100~+100으로 변환
    regime_raw = float(regime.get("macro_score", 50))
    regime_score = (regime_raw - 50) * 2

    # 클램프
    sense_score = max(-100, min(100, sense_score))
    overnight_score = max(-100, min(100, overnight_score))
    regime_score = max(-100, min(100, regime_score))

    # ── 시크릿 팩터 ──
    secret = compute_secret_factors()
    secret_score = secret["score"]

    # ── 피보나치 ──
    fibonacci = compute_kospi_fibonacci()

    # ── 이벤트 임팩트 ──
    event_impact = compute_event_impact()

    # ── 적중률 기반 동적 가중치 ──
    weights = load_adaptive_weights()

    # ── 가중 합산 ──
    meta_score = (
        sense_score * weights["sense"]
        + overnight_score * weights["overnight"]
        + regime_score * weights["regime"]
        + secret_score * weights["secret"]
    )

    # 이벤트 보정 (보조)
    meta_score += event_impact["score"]

    # ── 합의도 보정 ──
    directions = [
        _classify_direction(sense_score),
        _classify_direction(overnight_score),
        _classify_direction(regime_score),
    ]
    agreement_count = max(Counter(directions).values())

    if agreement_count == 3:
        meta_score *= 1.2
        agreement_str = "3/3 만장일치"
    elif agreement_count == 2:
        meta_score *= 1.0
        agreement_str = "2/3 다수결"
    else:
        meta_score *= 0.6
        agreement_str = "1/3 삼파전"

    meta_score = round(max(-100, min(100, meta_score)), 1)

    # ── 방향 판정 ──
    if meta_score >= 40:
        direction = "STRONG_LONG"
        direction_kr = "강한 상승"
        etfs = ETF_LONG
    elif meta_score >= 15:
        direction = "LONG"
        direction_kr = "상승"
        etfs = ETF_LONG
    elif meta_score > -15:
        direction = "NEUTRAL"
        direction_kr = "중립"
        etfs = ETF_SAFE
    elif meta_score > -40:
        direction = "SHORT"
        direction_kr = "하락"
        etfs = ETF_SHORT
    else:
        direction = "STRONG_SHORT"
        direction_kr = "강한 하락"
        etfs = ETF_SHORT

    confidence = min(int(abs(meta_score)), 100)

    # ── ETF 추천 ──
    if abs(meta_score) >= 40 and len(etfs) > 0:
        pick = etfs[0]  # 레버리지/인버스2X
    elif len(etfs) > 1:
        pick = etfs[1]  # 1배
    elif len(etfs) > 0:
        pick = etfs[0]
    else:
        pick = None

    # ── 피보나치 기반 진입/목표/손절 ──
    action = _compute_action(direction, pick, fibonacci, meta_score)

    # ── 컴포넌트 요약 ──
    sense_dir = sense.get("direction", "NEUTRAL")
    overnight_grade = overnight.get("grade", overnight.get("final_grade", "NEUTRAL"))
    regime_current = regime.get("current_regime", "CAUTION")
    regime_transition = regime.get("transition_direction", "")

    components = {
        "sense": {"score": round(sense_score, 1), "direction": sense_dir},
        "overnight": {"score": round(overnight_score, 1), "direction": overnight_grade},
        "regime": {
            "score": round(regime_score, 1),
            "direction": f"{regime_current} ({regime_transition})" if regime_transition else regime_current,
        },
        "secret": {
            "score": secret_score,
            "details": {
                "flow_reversal": secret["flow_reversal"]["score"],
                "range_squeeze": secret["range_squeeze"]["score"],
                "decoupling": secret["decoupling"]["score"],
            },
        },
    }

    return {
        "date": date.today().isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "direction": direction,
        "direction_kr": direction_kr,
        "meta_score": meta_score,
        "confidence": confidence,
        "agreement": agreement_str,
        "weights": weights,
        "components": components,
        "secret_details": {
            "flow_reversal": secret["flow_reversal"],
            "range_squeeze": secret["range_squeeze"],
            "decoupling": secret["decoupling"],
        },
        "fibonacci": fibonacci,
        "event_impact": event_impact,
        "action": action,
        "recommended_etf": pick,
    }


def _compute_action(direction: str, etf: dict | None, fibonacci: dict, meta_score: float) -> dict:
    """피보나치 기반 진입/목표/손절 추천."""
    action = {
        "recommendation": "관망",
        "etf": etf,
        "entry_level": None,
        "target_level": None,
        "stop_level": None,
    }

    if direction == "NEUTRAL":
        return action

    fib_20d = fibonacci.get("swing_20d", {})
    if not fib_20d:
        # 피보나치 없으면 기본 추천만
        if direction in ("STRONG_LONG", "LONG"):
            action["recommendation"] = "KOSPI ETF 매수"
        elif direction in ("STRONG_SHORT", "SHORT"):
            action["recommendation"] = "인버스 ETF 매수"
        return action

    current = fib_20d.get("current", 0)
    support = fib_20d.get("nearest_support")
    resistance = fib_20d.get("nearest_resistance")

    if direction in ("STRONG_LONG", "LONG"):
        action["recommendation"] = "KOSPI ETF 매수"
        action["entry_level"] = support  # 피보나치 지지선에서 진입
        action["target_level"] = resistance  # 피보나치 저항선이 목표
        # 손절은 지지선 아래 1%
        if support:
            action["stop_level"] = round(support * 0.99, 2)
    elif direction in ("STRONG_SHORT", "SHORT"):
        action["recommendation"] = "인버스 ETF 매수"
        action["entry_level"] = resistance  # 저항선에서 숏 진입
        action["target_level"] = support  # 지지선까지 하락 목표
        if resistance:
            action["stop_level"] = round(resistance * 1.01, 2)

    return action


# ══════════════════════════════════════════
# 6. 텔레그램 메시지
# ══════════════════════════════════════════

def build_telegram_message(result: dict) -> str:
    """텔레그램 발송용 HTML 메시지 생성."""
    comp = result["components"]
    secret = result["secret_details"]
    fib = result.get("fibonacci", {}).get("swing_20d", {})
    events = result.get("event_impact", {}).get("events", [])
    action = result["action"]

    # 방향 아이콘
    dir_icon = {"STRONG_LONG": "🔴", "LONG": "🟢", "NEUTRAL": "⚪", "SHORT": "🔵", "STRONG_SHORT": "🔵"}
    icon = dir_icon.get(result["direction"], "⚪")

    # 컴포넌트 방향 표시
    def _arrow(score):
        if score > 15:
            return "↑"
        elif score < -15:
            return "↓"
        return "→"

    lines = [
        f"{icon} <b>JARVIS D+1 방향 예측</b>",
        "",
        f"📊 판정: {result['direction_kr']} ({result['direction']})",
        f"🎯 메타점수: {result['meta_score']:+.1f} (신뢰도 {result['confidence']}%)",
        f"✅ 합의: {result['agreement']}",
        f"   눈치{_arrow(comp['sense']['score'])} · 야간{_arrow(comp['overnight']['score'])} · 레짐{_arrow(comp['regime']['score'])}",
    ]

    # 시크릿 팩터
    if comp["secret"]["score"] != 0:
        lines.append("")
        lines.append(f"🔮 시크릿 팩터: {comp['secret']['score']:+d}")
        for key, detail in secret.items():
            if detail["score"] != 0:
                lines.append(f"   {detail['detail']}")

    # 피보나치
    if fib:
        lines.append("")
        lines.append(f"📈 피보나치 (20일)")
        lines.append(f"   현재: {fib['current']:,.0f}")
        if fib.get("nearest_support"):
            lines.append(f"   지지: {fib['nearest_support']:,.0f}")
        if fib.get("nearest_resistance"):
            lines.append(f"   저항: {fib['nearest_resistance']:,.0f}")
        lines.append(f"   위치: {fib['position_pct']}% (고점 대비)")

    # 이벤트
    if events:
        lines.append("")
        lines.append("📅 이벤트:")
        for ev in events[:3]:
            lines.append(f"   {ev['name']} D-{ev['days_until']} (+{ev['impact_score']}pt)")

    # 추천
    if action["recommendation"] != "관망":
        lines.append("")
        etf = action.get("etf", {})
        if etf:
            lines.append(f"💡 추천: {etf.get('name', '')} ({etf.get('code', '')})")
        else:
            lines.append(f"💡 추천: {action['recommendation']}")
        if action.get("entry_level"):
            lines.append(f"   진입: {action['entry_level']:,.0f}")
        if action.get("target_level"):
            lines.append(f"   목표: {action['target_level']:,.0f}")
        if action.get("stop_level"):
            lines.append(f"   손절: {action['stop_level']:,.0f}")

    return "\n".join(lines)


# ══════════════════════════════════════════
# 7. main
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="JARVIS D+1 방향 예측 엔진")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  JARVIS D+1 방향 예측 엔진")
    logger.info("=" * 60)

    # ── 미검증 기록 자동 검증 ──
    verified = auto_verify()
    if verified > 0:
        logger.info("  자동 검증: %d건", verified)

    # ── 적중률 표시 ──
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                log_data = json.load(f)
            verified_entries = [e for e in log_data if e.get("hit") is not None]
            if verified_entries:
                hits = sum(1 for e in verified_entries if e["hit"])
                logger.info("  누적 적중률: %d/%d = %.1f%%",
                            hits, len(verified_entries),
                            hits / len(verified_entries) * 100)
        except Exception:
            pass

    # ── 방향 예측 ──
    result = compute_direction()

    logger.info("")
    logger.info("  ★ 방향: %s (%s)", result["direction_kr"], result["direction"])
    logger.info("  ★ 메타점수: %+.1f / 신뢰도: %d%%", result["meta_score"], result["confidence"])
    logger.info("  ★ 합의: %s", result["agreement"])
    logger.info("")

    # 컴포넌트 출력
    comp = result["components"]
    logger.info("  [눈치엔진]  %+.1f (%s)", comp["sense"]["score"], comp["sense"]["direction"])
    logger.info("  [US야간]    %+.1f (%s)", comp["overnight"]["score"], comp["overnight"]["direction"])
    logger.info("  [레짐]      %+.1f (%s)", comp["regime"]["score"], comp["regime"]["direction"])
    logger.info("  [시크릿]    %+d", comp["secret"]["score"])

    # 시크릿 상세
    for key in ("flow_reversal", "range_squeeze", "decoupling"):
        detail = result["secret_details"][key]
        if detail["score"] != 0:
            logger.info("    %s: %+d (%s)", key, detail["score"], detail["detail"])

    # 가중치
    w = result["weights"]
    logger.info("")
    logger.info("  가중치: 눈치%.0f%% · 야간%.0f%% · 레짐%.0f%% · 시크릿%.0f%%",
                w["sense"] * 100, w["overnight"] * 100, w["regime"] * 100, w["secret"] * 100)

    # 피보나치
    fib = result.get("fibonacci", {}).get("swing_20d", {})
    if fib:
        logger.info("")
        logger.info("  피보나치 (20일): 고점 %.0f / 저점 %.0f / 현재 %.0f (%.1f%%)",
                    fib["swing_high"], fib["swing_low"], fib["current"], fib["position_pct"])
        if fib.get("nearest_support"):
            logger.info("    지지: %.0f", fib["nearest_support"])
        if fib.get("nearest_resistance"):
            logger.info("    저항: %.0f", fib["nearest_resistance"])

    # 이벤트
    events = result.get("event_impact", {}).get("events", [])
    if events:
        logger.info("")
        logger.info("  이벤트 임팩트: %+d", result["event_impact"]["score"])
        for ev in events[:3]:
            logger.info("    %s D-%d (+%d)", ev["name"], ev["days_until"], ev["impact_score"])

    # 추천
    action = result["action"]
    if action["recommendation"] != "관망":
        logger.info("")
        etf = action.get("etf", {})
        if etf:
            logger.info("  추천: %s (%s)", etf.get("name", ""), etf.get("code", ""))
        if action.get("entry_level"):
            logger.info("  진입: %.0f / 목표: %.0f / 손절: %.0f",
                        action["entry_level"],
                        action.get("target_level", 0),
                        action.get("stop_level", 0))

    # ── 적중률 기록 ──
    update_jarvis_log(result)

    # ── 출력 저장 ──
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("")
    logger.info("  저장: %s", OUTPUT_PATH)

    # ── 텔레그램 ──
    if args.send:
        try:
            from src.telegram_sender import send_message
            msg = build_telegram_message(result)
            send_message(msg)
            logger.info("  텔레그램 발송 완료")
        except Exception as e:
            logger.warning("  텔레그램 발송 실패: %s", e)

    logger.info("")
    logger.info("  JARVIS 완료")


if __name__ == "__main__":
    main()
