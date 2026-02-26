"""
내일 추천 종목 통합 스캐너 — 9개 시그널 교차 검증

9개 시그널 소스를 통합하여 최종 매수 추천 종목을 산출합니다.

소스:
  1. 섹터릴레이 picks (relay_trading_signal.json)
  2. 그룹순환 waiting_subsidiaries (group_relay_today.json)
  3. 눌림목 반등임박/매수대기 (pullback_scan.json)
  4. 퀀텀시그널 survivors + killed (scan_cache.json)
  5. 동반매수 S/A등급 + core_watch (dual_buying_watch.json)
  6. 세력감지 세력포착/매집의심 (force_hybrid.json)
  7. DART 이벤트 BUY (dart_event_signals.json)
  8. 레짐 부스트 (regime_macro_signal.json)
  9. 수급폭발→조정 매수 (volume_spike_watchlist.json) — v7 NEW

통합 점수 (100점, 5축 + 과열패널티):
  다중 시그널 (25): 2소스 +12, 3소스 +20, 4+ +25
  개별 점수  (20): 각 소스 점수 정규화 평균
  기술적 지지 (25): RSI(8) + MA(5) + MACD(4) + TRIX(4) + Stoch(4) + MACD 3중 보너스
  수급       (20): 외인(8) + 기관(5) + 동시매수(2) + 연속매수(2)
  안전       (10): BB(4) + ADX(3) + 낙폭(3)
  과열 패널티: RSI/Stoch/BB/급등 최대 -25점
  레짐 부스트: 매크로 점수에 따라 최종 점수 × position_multiplier (0.5~1.3x)
  US섹터 부스트: 전략C — US ETF 모멘텀 → KR 섹터 가산 (최대 ±5점)
  DART AVOID: 유상증자/관리종목 등 자동 제외

v7 변경: 전략A(수급폭발 소스) + 전략B(MACD 3중 필터) + 전략C(US섹터 부스트)
v8 변경: 전략D(매집추적 소스) — 거래량폭발 이후 매집 진행 중 종목
v9 변경: 전략E(Perplexity 인텔리전스) — 미국장 이벤트 → 한국 섹터/종목 파급 보정

Usage:
    python scripts/scan_tomorrow_picks.py
"""

from __future__ import annotations

import json
import logging
import sys
import calendar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = DATA_DIR / "tomorrow_picks.json"


def _sf(val, default=0):
    """NaN/Inf/None/str 안전 변환"""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else round(v, 2)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    """NaN-safe int 변환"""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else int(v)
    except (TypeError, ValueError):
        return default


def load_json(rel_path: str) -> dict | list:
    fp = DATA_DIR / rel_path
    if not fp.exists():
        return {}
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def build_name_map() -> dict[str, str]:
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


# ──────────────────────────────────────────
# 소스별 종목 수집
# ──────────────────────────────────────────

def collect_relay() -> dict[str, dict]:
    """소스1: 섹터릴레이 picks"""
    relay = load_json("sector_rotation/relay_trading_signal.json")
    result = {}
    for sig in relay.get("signals", []):
        lead = sig.get("lead", "")
        follow = sig.get("follow", "")
        for p in sig.get("picks", []):
            ticker = p.get("ticker", "")
            if not ticker:
                continue
            result[ticker] = {
                "source": "릴레이",
                "score": p.get("score", 0),
                "name": p.get("name", ""),
                "detail": f"{lead}→{follow}",
            }
    return result


def collect_group_relay() -> dict[str, dict]:
    """소스2: 그룹순환 대기 종목"""
    gr = load_json("group_relay/group_relay_today.json")
    result = {}
    for g in gr.get("fired_groups", []):
        group_name = g.get("group_name", "")
        for w in g.get("waiting_subsidiaries", []):
            ticker = w.get("ticker", "")
            if not ticker:
                continue
            result[ticker] = {
                "source": "그룹순환",
                "score": w.get("score", 0) or w.get("composite_score", 0),
                "name": w.get("name", ""),
                "rsi": w.get("rsi", 50),
                "foreign_5d": w.get("foreign_5d", 0),
                "detail": f"{group_name} 계열",
            }
    return result


def collect_pullback() -> dict[str, dict]:
    """소스3: 눌림목 반등임박/매수대기"""
    pb = load_json("pullback_scan.json")
    result = {}
    # pullback_scan.json의 키: "candidates" (상위 30) + "all_uptrend" (전체)
    candidates = pb.get("candidates", pb.get("items", []))
    for item in candidates:
        grade = item.get("grade", "")
        if grade not in ("반등임박", "매수대기"):
            continue
        ticker = item.get("ticker", "")
        if not ticker:
            continue
        result[ticker] = {
            "source": "눌림목",
            "score": item.get("score", 0),
            "name": item.get("name", ""),
            "grade": grade,
            "detail": grade,
        }
    return result


def collect_quantum() -> dict[str, dict]:
    """소스4: 퀀텀시그널 (survivors + killed 중 유망)"""
    q = load_json("scan_cache.json")
    result = {}

    # 최종 통과
    for c in q.get("candidates", []):
        ticker = c.get("ticker", "")
        if not ticker:
            continue
        result[ticker] = {
            "source": "퀀텀",
            "score": 90,  # 최종 통과 = 높은 기본점수
            "name": c.get("name", ""),
            "rr": c.get("risk_reward", 0),
            "entry": c.get("entry_price", 0),
            "target": c.get("target_price", 0),
            "stop": c.get("stop_loss", 0),
            "detail": f"v9통과 R:R {c.get('risk_reward',0):.1f}",
        }

    # Kill된 종목 중 R:R >= 1.5이고 기술적 지표 양호한 것
    stats = q.get("stats", {})
    for k in stats.get("v9_killed_list", []):
        ticker = k.get("ticker", "")
        if not ticker or ticker in result:
            continue
        rr = k.get("risk_reward", 0)
        rsi = k.get("rsi", 50)
        if rr < 1.5 or rsi > 60:
            continue
        result[ticker] = {
            "source": "퀀텀",
            "score": 50 + min(rr * 10, 30),  # 50~80
            "name": k.get("name", ""),
            "rr": rr,
            "entry": k.get("entry_price", 0),
            "target": k.get("target_price", 0),
            "stop": k.get("stop_loss", 0),
            "detail": f"Kill(R:R {rr:.1f})",
        }

    return result


def collect_dual_buying() -> dict[str, dict]:
    """소스5: 동반매수 S/A등급 + core_watch"""
    db = load_json("dual_buying_watch.json")
    result = {}

    for grade, label, base_score in [
        ("s_grade", "S등급", 65),
        ("a_grade", "A등급", 50),
        ("core_watch", "핵심관찰", 40),
    ]:
        for item in db.get(grade, []):
            ticker = item.get("ticker", "")
            if not ticker:
                continue
            bonus = min(int(item.get("dual_days", 0) or 0) * 3, 15)
            result[ticker] = {
                "source": "동반매수",
                "score": base_score + bonus,
                "name": item.get("name", ""),
                "dual_days": item.get("dual_days", 0),
                "f_streak": item.get("f_streak", 0),
                "i_streak": item.get("i_streak", 0),
                "detail": f"{label} 동반{item.get('dual_days',0)}일",
            }

    return result


def collect_force_hybrid() -> dict[str, dict]:
    """소스6: 세력감지 하이브리드 (세력포착/매집의심)"""
    fh = load_json("force_hybrid.json")
    result = {}
    for item in fh.get("anomaly", {}).get("items", []):
        grade = item.get("grade", "")
        if grade not in ("세력포착", "매집의심"):
            continue
        ticker = item.get("ticker", "")
        if not ticker:
            continue
        patterns = item.get("patterns", [])
        pattern_names = [p.get("pattern", "") for p in patterns[:2]]
        result[ticker] = {
            "source": "세력감지",
            "score": 70 if grade == "세력포착" else 55,
            "name": item.get("name", ""),
            "detail": f"{grade} {','.join(pattern_names)}",
        }
    return result


def collect_dart_event() -> dict[str, dict]:
    """소스7: DART 이벤트 BUY 시그널"""
    de = load_json("dart_event_signals.json")
    result = {}
    for sig in de.get("signals", []):
        if sig.get("action") != "BUY":
            continue
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        result[ticker] = {
            "source": "이벤트",
            "score": sig.get("event_score", 0),
            "name": sig.get("name", ""),
            "detail": sig.get("event", "DART"),
        }
    return result


def collect_accumulation_tracker() -> dict[str, dict]:
    """소스10: 세력 매집 추적 (전략 D) — 거래량폭발 이후 매집 진행 중 종목.

    phase: 매집/재돌파/가속
    score: 40~100
    """
    at = load_json("accumulation_tracker.json")
    result = {}
    for item in at.get("items", []):
        ticker = item.get("ticker", "")
        if not ticker:
            continue
        phase = item.get("phase", "")
        # 폭발 직후(Phase1)는 이미 수급폭발 소스에서 커버 → 매집/재돌파/가속만 사용
        if phase == "폭발":
            continue
        score = item.get("total_score", 0)
        if score < 50:
            continue  # 50점 이하는 시그널로 부적합

        phase_icon = {"매집": "🔄", "재돌파": "🚀", "가속": "⚡"}.get(phase, "")
        result[ticker] = {
            "source": "매집추적",
            "score": score,
            "name": item.get("name", ""),
            "detail": f"{phase_icon}{phase} {item.get('days_since_spike',0)}일전폭발 수익:{item.get('return_since_spike',0):+.1f}%",
            "phase": phase,
            "days_since_spike": item.get("days_since_spike", 0),
            "return_since_spike": item.get("return_since_spike", 0),
        }
    return result


def collect_volume_spike() -> dict[str, dict]:
    """소스9: 수급 폭발 → 조정 매수 시그널 (전략 A)"""
    vs = load_json("volume_spike_watchlist.json")
    result = {}
    for sig in vs.get("signals", []):
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        score = sig.get("score", 50)
        result[ticker] = {
            "source": "수급폭발",
            "score": score,
            "name": sig.get("name", ""),
            "detail": f"폭발→조정{sig.get('pullback_pct', 0):+.1f}% {sig.get('days_since_spike', 0)}일전",
        }
    return result


def load_dart_avoid_tickers() -> set[str]:
    """DART AVOID 종목 티커 세트 (유상증자/관리종목 등)"""
    de = load_json("dart_event_signals.json")
    avoid = set()
    for item in de.get("avoid_list", []):
        ticker = item.get("ticker", "")
        if ticker:
            avoid.add(ticker)
    return avoid


# ── 전략 C: US→KR 섹터 모멘텀 부스트 ──
SECTOR_BRIDGE = {
    "에너지화학": ["에너지", "화학"],
    "헬스케어": ["헬스케어", "제약", "의료기기"],
    "반도체": ["반도체", "전자부품"],
    "IT": ["IT", "소프트웨어"],
    "금융": ["금융", "은행"],
    "은행": ["은행"],
    "증권": ["증권"],
    "건설": ["건설"],
    "조선": ["조선"],
    "바이오": ["바이오", "제약"],
    "소프트웨어": ["소프트웨어", "IT"],
    "2차전지": ["반도체", "전자부품"],
    "인터넷": ["IT", "소프트웨어"],
    "철강소재": ["에너지", "화학"],
}

_STOCK_SECTOR_CACHE: dict | None = None


def _load_stock_to_sector() -> dict:
    """stock_to_sector.json 캐시 로드."""
    global _STOCK_SECTOR_CACHE
    if _STOCK_SECTOR_CACHE is None:
        fp = DATA_DIR / "sector_rotation" / "stock_to_sector.json"
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                _STOCK_SECTOR_CACHE = json.load(f)
        else:
            _STOCK_SECTOR_CACHE = {}
    return _STOCK_SECTOR_CACHE


def load_sector_momentum_boost() -> dict[str, float]:
    """overnight_signal.json의 sector_momentum → {섹터명: boost} 딕셔너리."""
    sig = load_json("us_market/overnight_signal.json")
    sm = sig.get("sector_momentum", {})
    return {k: v.get("boost", 0) for k, v in sm.items()}


def get_ticker_sector_boost(ticker: str, boost_map: dict[str, float]) -> float:
    """종목 → stock_to_sector → SECTOR_BRIDGE → sector_momentum boost (절대값 최대)."""
    if not boost_map:
        return 0.0
    sts = _load_stock_to_sector()
    sectors = sts.get(ticker, [])
    best_boost = 0.0
    for sec in sectors:
        # 직접 매칭
        if sec in boost_map:
            b = boost_map[sec]
            if abs(b) > abs(best_boost):
                best_boost = b
        # SECTOR_BRIDGE 매칭
        bridge_keys = SECTOR_BRIDGE.get(sec, [])
        for bk in bridge_keys:
            if bk in boost_map:
                b = boost_map[bk]
                if abs(b) > abs(best_boost):
                    best_boost = b
    return round(best_boost, 1)


def load_regime_boost() -> float:
    """레짐 매크로 시그널에서 position_multiplier 로드"""
    macro = load_json("regime_macro_signal.json")
    return macro.get("position_multiplier", 1.0)


def load_institutional_targets() -> dict:
    """기관 추정 목표가 데이터 로드."""
    data = load_json("institutional_targets.json")
    return data.get("targets", {})


def load_market_intelligence() -> dict:
    """Perplexity 시장 인텔리전스 데이터 로드."""
    return load_json("market_intelligence.json")


def get_target_zone_bonus(ticker: str, targets: dict) -> tuple[float, dict]:
    """D존 기반 점수 보정 + 부가 정보 반환.

    Returns:
        (bonus_points, info_dict)
        D-3: +5, D-2: +3, D-1: +1, 도달: 0, 초과: -3
        confidence < 0.5이면 보정 절반.
    """
    t = targets.get(ticker)
    if not t:
        return 0.0, {}

    zone = t.get("zone", "")
    zone_bonus = {
        "D-3": 5.0,
        "D-2": 3.0,
        "D-1": 1.0,
        "도달": 0.0,
        "초과": -3.0,
    }

    bonus = zone_bonus.get(zone, 0.0)

    confidence = t.get("confidence", 0.5)
    if confidence < 0.5:
        bonus *= 0.5

    # Velocity 보정: RISING +2, FALLING -2
    direction = t.get("target_direction", "")
    if direction == "RISING":
        bonus += 2.0
    elif direction == "FALLING":
        bonus -= 2.0

    info = {
        "estimated_target": t.get("estimated_target", 0),
        "gap_pct": t.get("gap_pct", 0),
        "zone": zone,
        "confidence": confidence,
        "direction": direction,
        "delta_5d": t.get("target_delta_5d"),
    }

    return bonus, info


# ──────────────────────────────────────────
# 통합 점수 계산
# ──────────────────────────────────────────

def _calc_macd_triple_bonus(pq: dict | None) -> int:
    """전략 B: MACD 0선 근처 + 골든크로스 + 외인매수 + 거래량surge → 보너스

    4중 충족: +6점, 3중: +4점, 2중+MACD상승: +2점
    """
    if not pq:
        return 0

    hits = 0

    # 조건1: MACD 0선 근처 (close의 ±2% 이내)
    macd_val = pq.get("macd", 0)
    close = pq.get("close", 0)
    if close > 0 and abs(macd_val) < close * 0.02:
        hits += 1

    # 조건2: MACD 골든크로스 (histogram 음→양)
    hist = pq.get("macd_histogram", 0)
    hist_prev = pq.get("macd_histogram_prev", 0)
    if hist > 0 and hist_prev <= 0:
        hits += 1

    # 조건3: 외인 5일 순매수
    if pq.get("foreign_5d", 0) > 0:
        hits += 1

    # 조건4: 거래량 서지 (vol_z >= 2.0 또는 vsr >= 2.0)
    if pq.get("vol_z", 0) >= 2.0 or pq.get("volume_surge_ratio", 0) >= 2.0:
        hits += 1

    if hits >= 4:
        return 6
    if hits >= 3:
        return 4
    if hits >= 2 and pq.get("macd_rising", False):
        return 2
    return 0


def calc_integrated_score(
    ticker: str,
    sources: list[dict],
    parquet_data: dict | None,
) -> dict:
    """5축 100점 + 과열패널티 통합 점수 계산 (v5)

    기본 100점 배분:
      다중시그널(25) + 개별점수(20) + 기술적(25) + 수급(20) + 안전(10)
    v5: 동반매수 단독 보장 삭제, 다른 소스와 겹칠 때만 소액 가산
    과열 패널티: 최대 -25점
    """

    # ── 축1: 다중 시그널 (25점) ──
    n_sources = len(sources)
    if n_sources >= 4:
        multi_score = 25
    elif n_sources >= 3:
        multi_score = 20
    elif n_sources >= 2:
        multi_score = 12
    else:
        multi_score = 0

    # 동반매수 연속일: 다중소스 점수에 소액 가산만 (단독 보장 삭제)
    # v4 교훈: 동반매수 단독 15점 보장 → 1소스만으로도 관심매수 → 결과 부진
    dual_days = 0
    for s in sources:
        dd = s.get("dual_days", 0) or s.get("f_streak", 0) or 0
        dual_days = max(dual_days, int(dd))
    if n_sources >= 2 and dual_days >= 3:
        multi_score += 3  # 다른 소스와 겹칠 때만 소액 보너스

    # ── 축2: 개별 점수 평균 (20점) ──
    avg_src_score = np.mean([s["score"] for s in sources]) if sources else 0
    individual_score = min(avg_src_score / 100 * 20, 20)

    # ── parquet 기반 기술적 지표 ──
    rsi = 50; adx = 20; above_ma60 = False; above_ma20 = False
    bb_pos = 50; drawdown = 0; foreign_5d = 0; inst_5d = 0
    close = 0; price_change = 0; ma20 = 0; ma60 = 0
    stoch_k = 50; stoch_d = 50; trix_gx = False; macd_rising = False
    ret_5d = 0; ret_20d = 0; low_20d = 0
    trix = 0; trix_signal = 0

    if parquet_data:
        rsi = parquet_data.get("rsi", 50)
        adx = parquet_data.get("adx", 20)
        above_ma60 = parquet_data.get("above_ma60", False)
        above_ma20 = parquet_data.get("above_ma20", False)
        bb_pos = parquet_data.get("bb_pos", 50)
        drawdown = parquet_data.get("drawdown", 0)
        foreign_5d = parquet_data.get("foreign_5d", 0)
        inst_5d = parquet_data.get("inst_5d", 0)
        close = parquet_data.get("close", 0)
        price_change = parquet_data.get("price_change", 0)
        ma20 = parquet_data.get("ma20", 0)
        ma60 = parquet_data.get("ma60", 0)
        stoch_k = parquet_data.get("stoch_k", 50)
        stoch_d = parquet_data.get("stoch_d", 50)
        trix = parquet_data.get("trix", 0)
        trix_signal = parquet_data.get("trix_signal", 0)
        trix_gx = parquet_data.get("trix_gx", False)
        macd_rising = parquet_data.get("macd_rising", False)
        ret_5d = parquet_data.get("ret_5d", 0)
        ret_20d = parquet_data.get("ret_20d", 0)
        low_20d = parquet_data.get("low_20d", 0)

    # ── 축3: 기술적 지지 (25점) ──
    tech_score = 0
    # RSI 적정대 (0~8점) — 수급 동반 시 55~65도 유효
    if 35 <= rsi <= 60:
        tech_score += 8
    elif 30 <= rsi <= 70:
        tech_score += 4
    # 이동평균 (0~5점)
    if above_ma60:
        tech_score += 3
    if above_ma20:
        tech_score += 2
    # MACD 히스토그램 상승 (0~4점)
    if macd_rising:
        tech_score += 4
    # TRIX 골든크로스 또는 상향추세 (0~4점)
    if trix_gx:
        tech_score += 4
    elif trix > trix_signal:
        tech_score += 2
    # Stochastic 적정대 (0~4점) — 40~65가 매수 최적
    if 30 <= stoch_k <= 65:
        tech_score += 4
    elif 20 <= stoch_k <= 75:
        tech_score += 2

    # 전략 B: MACD 3중 필터 가산
    tech_score += _calc_macd_triple_bonus(parquet_data)

    tech_score = min(tech_score, 25)

    # ── 축4: 수급 (20점, 기존 15→20 상향) ──
    flow_score = 0
    if foreign_5d > 0:
        flow_score += 8
    elif foreign_5d > -1e6:
        flow_score += 2
    if inst_5d > 0:
        flow_score += 5
    # 외인+기관 동시매수
    if foreign_5d > 0 and inst_5d > 0:
        flow_score += 2
    # 연속 동반매수: 소액 보너스만 (v4 축소)
    if dual_days >= 4:
        flow_score += 2
    elif dual_days >= 3:
        flow_score += 1

    flow_score = min(flow_score, 20)

    # ── 축5: 안전 (10점, 기존 15→10) ──
    safety_score = 0
    if bb_pos < 80:
        safety_score += 4
    elif bb_pos < 95:
        safety_score += 2
    if 15 <= adx <= 35:
        safety_score += 3
    elif adx <= 45:
        safety_score += 2
    if abs(drawdown) < 15:
        safety_score += 3
    elif abs(drawdown) < 25:
        safety_score += 1

    safety_score = min(safety_score, 10)

    # ── 과열 패널티 (최대 -25점) ── NEW
    overheat_penalty = 0
    overheat_flags = []

    if rsi > 75:
        overheat_penalty += 8
        overheat_flags.append(f"RSI {rsi:.0f} 과매수")
    elif rsi > 70:
        overheat_penalty += 4
        overheat_flags.append(f"RSI {rsi:.0f} 주의")

    if stoch_k > 90:
        overheat_penalty += 7
        overheat_flags.append(f"Stoch {stoch_k:.0f} 극과열")
    elif stoch_k > 80:
        overheat_penalty += 4
        overheat_flags.append(f"Stoch {stoch_k:.0f} 과열")

    if bb_pos > 110:
        overheat_penalty += 6
        overheat_flags.append(f"BB {bb_pos:.0f}% 상단이탈")
    elif bb_pos > 95:
        overheat_penalty += 3
        overheat_flags.append(f"BB {bb_pos:.0f}% 상단근접")

    if ret_5d > 15:
        overheat_penalty += 4
        overheat_flags.append(f"5일 +{ret_5d:.0f}% 급등")
    elif ret_5d > 10:
        overheat_penalty += 2
        overheat_flags.append(f"5일 +{ret_5d:.0f}% 급등주의")

    overheat_penalty = min(overheat_penalty, 25)

    base_total = multi_score + individual_score + tech_score + flow_score + safety_score
    total = max(base_total - overheat_penalty, 0)

    # ── 진입가 / 손절가 / 목표가 자동 생성 ──
    ma5 = parquet_data.get("ma5", 0) if parquet_data else 0
    ma7 = parquet_data.get("ma7", 0) if parquet_data else 0
    entry_info = _calc_entry_stop(close, ma20, ma60, low_20d, rsi, stoch_k, bb_pos, ma5, ma7)

    # ── 핵심 근거 생성 ──
    reasons = _build_reasons(
        n_sources, rsi, stoch_k, bb_pos, adx, above_ma20, above_ma60,
        trix_gx, macd_rising, foreign_5d, inst_5d, ret_5d, overheat_flags,
    )

    return {
        "total": round(min(total, 100), 1),
        "multi": multi_score,
        "individual": round(individual_score, 1),
        "tech": tech_score,
        "flow": flow_score,
        "safety": safety_score,
        "overheat": overheat_penalty,
        "overheat_flags": overheat_flags,
        "rsi": _sf(rsi),
        "adx": _sf(adx),
        "stoch_k": _sf(stoch_k),
        "above_ma60": above_ma60,
        "above_ma20": above_ma20,
        "bb_position": _sf(bb_pos),
        "drawdown": _sf(drawdown),
        "foreign_5d": _sf(foreign_5d),
        "inst_5d": _sf(inst_5d),
        "ret_5d": _sf(ret_5d),
        "close": _safe_int(close),
        "price_change": _sf(price_change),
        "entry_info": entry_info,
        "reasons": reasons,
    }


def _calc_entry_stop(
    close: float, ma20: float, ma60: float,
    low_20d: float, rsi: float, stoch_k: float, bb_pos: float,
    ma5: float = 0, ma7: float = 0,
) -> dict:
    """진입가/손절가/진입조건 자동 생성 (MA5~MA7 진입 전략 반영)"""
    if close <= 0:
        return {"entry": 0, "stop": 0, "target": 0, "condition": "데이터 부족",
                "ma5_entry": ""}

    # 손절가: 20일 저점 또는 MA20*0.98 중 현재가 아래에 있는 지지선 활용
    # (현재가 이상인 후보는 제외 — 한화에어로 버그 방지)
    stop_candidates = [v for v in [low_20d, ma20 * 0.98] if 0 < v < close]
    stop = max(stop_candidates) if stop_candidates else close * 0.95
    stop = max(stop, close * 0.93)   # 손절폭 최대 -7%
    stop = min(stop, close * 0.97)   # 손절폭 최소 -3% (너무 가까우면 의미 없음)

    # ── MA5~MA7 진입 전략 ──
    # 핵심: 5일선~7일선 근처에서 진입해야 승률이 높다
    ma5_gap = ((close / ma5) - 1) * 100 if ma5 > 0 else 0
    ma7_gap = ((close / ma7) - 1) * 100 if ma7 > 0 else 0
    # MA5와 MA7의 중간값을 기준선으로 사용
    ma_mid = (ma5 + ma7) / 2 if ma5 > 0 and ma7 > 0 else ma5 or ma7
    ma_mid_gap = ((close / ma_mid) - 1) * 100 if ma_mid > 0 else 0
    ma5_entry = ""  # MA5 진입 판정 태그

    if ma_mid > 0:
        if ma_mid_gap < -3.0:
            # MA5 하향 이탈 → 반등 확인 필요
            ma5_entry = "하향이탈"
            condition = f"MA5 하향이탈 {ma5_gap:+.1f}%→반등확인 후"
            entry = int(round(ma5 * 0.995, -1))  # MA5 -0.5%
        elif ma_mid_gap < 0:
            # -3% ~ 0%: MA5 아래 소폭 → 반등 대기
            ma5_entry = "반등대기"
            condition = f"MA5 소폭하회 {ma5_gap:+.1f}%→반등 확인"
            entry = _safe_int(close)
        elif ma_mid_gap <= 3.0:
            # 최적 진입 구간: MA5~7 이내 (0%~+3%)
            ma5_entry = "5·7선접근"
            condition = f"MA5~7 부근 진입적기 ({ma5_gap:+.1f}%)"
            entry = _safe_int(close)
        elif ma_mid_gap <= 5.0:
            # 약간 이격: 눌림 대기 권장
            ma5_entry = "눌림대기"
            condition = f"MA5 대비 +{ma5_gap:.1f}% 이격→5일선 눌림 대기"
            entry = int(round(ma_mid * 1.005, -1))  # MA중간 +0.5% 수준
        else:
            # 과이격 (+5% 초과): MA5 복귀 대기
            ma5_entry = "이격과대"
            condition = f"MA5 대비 +{ma5_gap:.1f}% 과이격→5일선 복귀 대기"
            entry = int(round(ma5 * 1.01, -1))  # MA5 +1% 수준
    else:
        # MA5 데이터 없을 때 기존 로직 폴백
        ma5_entry = ""
        if stoch_k > 85 or bb_pos > 100:
            if stoch_k > 85:
                condition = f"Stoch {stoch_k:.0f}→70 이하 냉각 시"
                entry = round(close * 0.97, -1)
            else:
                condition = f"BB {bb_pos:.0f}%→85% 이하 복귀 시"
                entry = round(close * 0.96, -1)
        elif rsi > 70:
            condition = f"RSI {rsi:.0f}→65 이하 조정 시"
            entry = round(close * 0.97, -1)
        elif rsi < 35:
            condition = "RSI 과매도 반등 확인 후"
            entry = round(close * 1.01, -1)
        else:
            condition = "현재가 부근 매수 가능"
            entry = _safe_int(close)

    # 과열 상태에서는 MA5 접근이어도 과열 경고 추가
    if ma5_entry and (stoch_k > 85 or rsi > 75):
        condition += f" (⚠ 과열: RSI {rsi:.0f}/Stoch {stoch_k:.0f})"

    # ── 손절가를 entry 기준으로 재조정 ──
    # entry가 MA5 수준으로 낮아졌을 때 stop이 entry 근처면 R:R 무의미
    stop = min(stop, entry * 0.97)   # entry 대비 최소 -3%
    stop = max(stop, entry * 0.93)   # entry 대비 최대 -7%

    # 목표가: R:R 2:1 기준
    risk = entry - stop
    target = int(entry + risk * 2) if risk > 0 else int(entry * 1.07)

    # 가격 반올림 (10원 단위)
    stop = int(round(stop, -1))
    target = int(round(target, -1))

    return {
        "entry": int(entry),
        "stop": stop,
        "target": target,
        "condition": condition,
        "risk_pct": round((entry - stop) / entry * 100, 1) if entry > 0 else 0,
        "ma5_entry": ma5_entry,
    }


def _build_reasons(
    n_sources, rsi, stoch_k, bb_pos, adx, above_ma20, above_ma60,
    trix_gx, macd_rising, foreign_5d, inst_5d, ret_5d, overheat_flags,
) -> list[str]:
    """핵심 근거 리스트 생성 (장점 + 주의사항)"""
    pros = []
    cons = []

    # 장점
    if n_sources >= 3:
        pros.append(f"{n_sources}중 시그널 교차")
    elif n_sources >= 2:
        pros.append(f"{n_sources}중 시그널")

    if 35 <= rsi <= 60:
        pros.append(f"RSI {rsi:.0f} 최적")
    elif 30 <= rsi <= 70:
        pros.append(f"RSI {rsi:.0f} 적정")

    if above_ma20 and above_ma60:
        pros.append("추세 만점")
    elif above_ma60:
        pros.append("MA60 위")

    if trix_gx:
        pros.append("TRIX 골든크로스")
    if macd_rising:
        pros.append("MACD 상승전환")

    if 30 <= stoch_k <= 60:
        pros.append(f"Stoch {stoch_k:.0f} 안전")

    if foreign_5d > 0 and inst_5d > 0:
        pros.append("외인+기관 동시매수")
    elif foreign_5d > 0:
        pros.append("외인 순매수")
    elif inst_5d > 0:
        pros.append("기관 순매수")

    if 20 <= adx <= 35:
        pros.append(f"ADX {adx:.0f} 강추세")

    # 주의사항 (과열 플래그에서)
    cons = [f"⚠ {f}" for f in overheat_flags]

    if foreign_5d < 0 and inst_5d < 0:
        cons.append("⚠ 외인+기관 동시매도")

    return pros + cons


def get_parquet_data(ticker: str) -> dict | None:
    """parquet에서 최신 기술적 지표 추출 (확장판)"""
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path).tail(25)
        if len(df) < 5:
            return None
        last = df.iloc[-1]
        close = float(last.get("close", 0))
        ma60 = float(last.get("sma_60", 0))
        ma20 = float(last.get("sma_20", 0))
        ma5 = float(last.get("sma_5", 0))
        # MA7은 parquet에 없으므로 직접 계산
        ma7 = float(df["close"].tail(7).mean()) if len(df) >= 7 else 0

        # 외인/기관 5일 합산
        f5 = float(np.nansum(df.tail(5)["외국인합계"].values)) if "외국인합계" in df.columns else 0
        i5 = float(np.nansum(df.tail(5)["기관합계"].values)) if "기관합계" in df.columns else 0

        # CSV fallback for foreign/inst
        if f5 == 0 and i5 == 0:
            csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
            if csvs:
                cdf = pd.read_csv(csvs[0], parse_dates=["Date"]).sort_values("Date").tail(5)
                if "Foreign_Net" in cdf.columns:
                    f5 = float(cdf["Foreign_Net"].sum())
                    i5 = float(cdf["Inst_Net"].sum())

        high_52 = float(last.get("high_252", close))
        dd = ((close / high_52) - 1) * 100 if high_52 > 0 else 0

        # 수익률
        closes = df["close"].values
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0

        # Stochastic
        stoch_k = float(last.get("stoch_slow_k", 50))
        stoch_d = float(last.get("stoch_slow_d", 50))

        # TRIX
        trix = float(last.get("trix", 0))
        trix_signal = float(last.get("trix_signal", 0))
        trix_gx = bool(last.get("trix_golden_cross", 0))

        # MACD
        macd_hist = float(last.get("macd_histogram", 0))
        macd_hist_prev = float(last.get("macd_histogram_prev", 0))
        macd_rising = macd_hist > macd_hist_prev

        # 손절가 = 최근 20일 최저가
        low_20d = float(df.tail(20)["low"].min()) if "low" in df.columns else close * 0.93

        # MA5 이격도 (%)
        ma5_gap_pct = round((close / ma5 - 1) * 100, 2) if ma5 > 0 else 0
        ma7_gap_pct = round((close / ma7 - 1) * 100, 2) if ma7 > 0 else 0

        return {
            "close": close,
            "price_change": float(last.get("price_change", 0)),
            "rsi": float(last.get("rsi_14", 50)),
            "adx": float(last.get("adx_14", 20)),
            "above_ma60": close > ma60 if ma60 > 0 else False,
            "above_ma20": close > ma20 if ma20 > 0 else False,
            "bb_pos": float(last.get("bb_position", 50)),
            "drawdown": dd,
            "foreign_5d": f5,
            "inst_5d": i5,
            "ma5": ma5,
            "ma7": ma7,
            "ma5_gap_pct": ma5_gap_pct,
            "ma7_gap_pct": ma7_gap_pct,
            "ma20": ma20,
            "ma60": ma60,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "trix": trix,
            "trix_signal": trix_signal,
            "trix_gx": trix_gx,
            "macd_rising": macd_rising,
            "low_20d": low_20d,
            "vol_z": float(last.get("vol_z", 0) or 0),
            "volume_surge_ratio": float(last.get("volume_surge_ratio", 0) or 0),
            "macd": float(last.get("macd", 0) or 0),
            "macd_signal": float(last.get("macd_signal", 0) or 0),
            "macd_histogram": macd_hist,
            "macd_histogram_prev": macd_hist_prev,
        }
    except Exception as e:
        logger.warning("parquet 읽기 실패 %s: %s", ticker, e)
        return None


# ──────────────────────────────────────────
# 등급 분류
# ──────────────────────────────────────────

def classify_pick(
    total_score: float, n_sources: int, rsi: float,
    has_data: bool = True, stoch_k: float = 50, ret_5d: float = 0,
) -> str:
    """등급 분류 — 하드 필터 포함 (v4)

    하드 디스퀄:
      - parquet 데이터 없음 → 데이터부족
      - Stoch >= 90 (극과열) → 최대 관찰
      - 5일 수익률 >= 15% (추격매수) → 최대 관찰
      - RSI >= 78 (과매수 극단) → 최대 관찰
    """
    if not has_data:
        return "데이터부족"

    # 하드 디스퀄: 극과열/추격매수는 관찰 이상 불가
    is_disqualified = stoch_k >= 90 or ret_5d >= 15 or rsi >= 78

    if is_disqualified:
        return "관찰" if total_score >= 40 else "보류"

    if total_score >= 70 and n_sources >= 2:
        return "적극매수"
    if total_score >= 60 and n_sources >= 2:
        return "매수"
    if total_score >= 55:
        return "관심매수"
    if total_score >= 40:
        return "관찰"
    return "보류"


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    name_map = build_name_map()

    # 10개 소스 수집 (v8: 9→10 앙상블, 전략D 매집추적 추가)
    src1 = collect_relay()
    src2 = collect_group_relay()
    src3 = collect_pullback()
    src4 = collect_quantum()
    src5 = collect_dual_buying()
    src6 = collect_force_hybrid()
    src7 = collect_dart_event()
    src9 = collect_volume_spike()
    src10 = collect_accumulation_tracker()

    print(f"[소스 수집] 릴레이:{len(src1)} 그룹순환:{len(src2)} "
          f"눌림목:{len(src3)} 퀀텀:{len(src4)} 동반매수:{len(src5)} "
          f"세력감지:{len(src6)} 이벤트:{len(src7)} 수급폭발:{len(src9)} "
          f"매집추적:{len(src10)}")

    # DART AVOID 필터 + 레짐 부스트 + 섹터 부스트 + 기관목표가 + 시장 인텔리전스
    avoid_tickers = load_dart_avoid_tickers()
    regime_mult = load_regime_boost()
    sector_boost_map = load_sector_momentum_boost()
    inst_targets = load_institutional_targets()
    intel = load_market_intelligence()
    if inst_targets:
        print(f"[기관목표가] {len(inst_targets)}종목 로드됨")
    if avoid_tickers:
        print(f"[DART AVOID] {len(avoid_tickers)}종목 자동 제외")
    print(f"[레짐 부스트] x{regime_mult:.1f}")
    active_boosts = {k: v for k, v in sector_boost_map.items() if v != 0}
    if active_boosts:
        print(f"[US섹터 부스트] {len(active_boosts)}섹터 활성: {active_boosts}")

    # 전략 E: Perplexity 인텔리전스
    intel_sector_boost = intel.get("sector_boost", {})
    intel_beneficiary = set(intel.get("beneficiary_stocks", []))
    intel_risk = set(intel.get("risk_stocks", []))
    intel_mood = intel.get("us_market_mood", "")
    intel_themes = intel.get("hot_themes", [])
    if intel_sector_boost:
        print(f"[인텔리전스] {intel_mood} | 섹터부스트: {intel_sector_boost}")
        print(f"  수혜종목: {len(intel_beneficiary)}개 | 주의종목: {len(intel_risk)}개")
        if intel_themes:
            print(f"  핫테마: {' | '.join(intel_themes)}")

    # 전체 종목 티커 수집
    all_tickers = set()
    for src in [src1, src2, src3, src4, src5, src6, src7, src9, src10]:
        all_tickers.update(src.keys())

    # AVOID 종목 제외
    all_tickers -= avoid_tickers

    print(f"[통합] 고유 종목: {len(all_tickers)}개")

    # 종목별 통합
    results = []
    for ticker in all_tickers:
        sources = []
        source_names = []
        for src, label in [(src1, "릴레이"), (src2, "그룹순환"), (src3, "눌림목"),
                           (src4, "퀀텀"), (src5, "동반매수"), (src6, "세력감지"),
                           (src7, "이벤트"), (src9, "수급폭발"), (src10, "매집추적")]:
            if ticker in src:
                sources.append(src[ticker])
                source_names.append(label)

        # parquet 기술적 데이터
        pq_data = get_parquet_data(ticker)

        # 통합 점수 계산
        score_detail = calc_integrated_score(ticker, sources, pq_data)

        # 레짐 부스트 적용 (v6): 매크로 점수에 따라 최종 점수 보정
        if regime_mult != 1.0:
            boosted = min(score_detail["total"] * regime_mult, 100)
            score_detail["total"] = round(boosted, 1)

        # 전략 C: US섹터 모멘텀 부스트 (가산, 최대 ±5점)
        ticker_boost = get_ticker_sector_boost(ticker, sector_boost_map)
        if ticker_boost != 0:
            boosted = max(min(score_detail["total"] + ticker_boost, 100), 0)
            score_detail["total"] = round(boosted, 1)
            if ticker_boost > 0:
                source_names.append("US모멘텀")

        # 기관 추정 목표가 D존 보정
        target_bonus, target_info = get_target_zone_bonus(ticker, inst_targets)
        if target_bonus != 0:
            boosted = max(min(score_detail["total"] + target_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)

        # 전략 E: Perplexity 인텔리전스 보정 (최대 ±5점)
        intel_bonus = 0.0
        intel_tag = ""
        # E-1: 수혜/피해 종목 직접 매칭 (종목명 기반)
        cur_name = ""
        for s in sources:
            if s.get("name"):
                cur_name = s["name"]
                break
        if not cur_name:
            cur_name = name_map.get(ticker, "")
        if cur_name in intel_beneficiary:
            intel_bonus += 3.0
            intel_tag = "수혜"
        elif cur_name in intel_risk:
            intel_bonus -= 3.0
            intel_tag = "피해"
        # E-2: 섹터 부스트 (stock_to_sector → intel_sector_boost)
        if intel_sector_boost:
            sts = _load_stock_to_sector()
            stock_sectors = sts.get(ticker, [])
            for sec in stock_sectors:
                if sec in intel_sector_boost:
                    sb = intel_sector_boost[sec]
                    intel_bonus += min(max(sb * 0.5, -2), 2)  # 섹터당 ±2 한도
                    if not intel_tag:
                        intel_tag = f"{sec}{'수혜' if sb > 0 else '피해'}"
                    break  # 첫 매칭만

        intel_bonus = round(max(min(intel_bonus, 5), -5), 1)
        if intel_bonus != 0:
            boosted = max(min(score_detail["total"] + intel_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            if intel_bonus > 0:
                source_names.append("인텔리전스")

        # 이름 결정
        name = ""
        for s in sources:
            if s.get("name"):
                name = s["name"]
                break
        if not name:
            name = name_map.get(ticker, ticker)

        has_data = pq_data is not None
        grade = classify_pick(
            score_detail["total"], len(sources), score_detail["rsi"],
            has_data=has_data,
            stoch_k=score_detail.get("stoch_k", 50),
            ret_5d=score_detail.get("ret_5d", 0),
        )

        entry_info = score_detail.get("entry_info", {})
        reasons = score_detail.get("reasons", [])

        rec = {
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "total_score": score_detail["total"],
            "n_sources": len(sources),
            "sources": source_names,
            "source_details": [s.get("detail", s["source"]) for s in sources],
            "score_breakdown": {
                "multi": score_detail["multi"],
                "individual": score_detail["individual"],
                "tech": score_detail["tech"],
                "flow": score_detail["flow"],
                "safety": score_detail["safety"],
                "overheat": score_detail.get("overheat", 0),
            },
            "close": score_detail["close"],
            "price_change": score_detail["price_change"],
            "rsi": score_detail["rsi"],
            "adx": score_detail["adx"],
            "stoch_k": score_detail.get("stoch_k", 50),
            "above_ma60": score_detail["above_ma60"],
            "above_ma20": score_detail["above_ma20"],
            "bb_position": score_detail["bb_position"],
            "foreign_5d": score_detail["foreign_5d"],
            "inst_5d": score_detail.get("inst_5d", 0),
            "ret_5d": score_detail.get("ret_5d", 0),
            "drawdown": score_detail["drawdown"],
            "entry_price": entry_info.get("entry", 0),
            "stop_loss": entry_info.get("stop", 0),
            "target_price": entry_info.get("target", 0),
            "entry_condition": entry_info.get("condition", ""),
            "risk_pct": entry_info.get("risk_pct", 0),
            "reasons": reasons,
            "overheat_flags": score_detail.get("overheat_flags", []),
            "estimated_target": target_info.get("estimated_target", 0),
            "target_gap_pct": target_info.get("gap_pct", 0),
            "target_zone": target_info.get("zone", ""),
            "target_confidence": target_info.get("confidence", 0),
            "target_direction": target_info.get("direction", ""),
            "target_delta_5d": target_info.get("delta_5d"),
            "accum_phase": src10.get(ticker, {}).get("phase", ""),
            "accum_days": src10.get(ticker, {}).get("days_since_spike", 0),
            "accum_return": src10.get(ticker, {}).get("return_since_spike", 0),
            "intel_bonus": intel_bonus,
            "intel_tag": intel_tag,
            "ma5_gap_pct": pq_data.get("ma5_gap_pct", 0) if pq_data else 0,
            "ma7_gap_pct": pq_data.get("ma7_gap_pct", 0) if pq_data else 0,
            "ma5_entry": entry_info.get("ma5_entry", ""),
        }

        results.append(rec)

    # 정렬: 등급 → 점수
    grade_order = {"적극매수": 0, "매수": 1, "관심매수": 2, "관찰": 3, "보류": 4, "데이터부족": 5}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["total_score"]))

    # ── TOP 5 선별: 매수 등급 이상에서 점수순 top 5 ──
    buyable_grades = {"적극매수", "매수", "관심매수"}
    buyable = [r for r in results if r["grade"] in buyable_grades]
    top5 = buyable[:5]
    for r in top5:
        r["is_top5"] = True

    # 통계
    grade_stats = {}
    for r in results:
        g = r["grade"]
        grade_stats[g] = grade_stats.get(g, 0) + 1

    print(f"\n{'='*60}")
    print(f"[내일 추천] 총 {len(results)}건 (TOP5: {len(top5)}건)")
    for g in ["적극매수", "매수", "관심매수", "관찰", "보류", "데이터부족"]:
        cnt = grade_stats.get(g, 0)
        if cnt:
            print(f"  {g}: {cnt}건")
    print(f"{'='*60}")

    # TOP 5 출력
    if top5:
        print(f"\n{'─'*60}")
        print(f"  ★ TOP 5 내일 매수 추천 ★")
        print(f"{'─'*60}")
        for i, r in enumerate(top5, 1):
            srcs = "+".join(r["sources"])
            oh = f" 🔥-{r['score_breakdown']['overheat']}p" if r["score_breakdown"]["overheat"] > 0 else ""
            cond = r.get("entry_condition", "")
            reasons_str = ", ".join(r.get("reasons", [])[:3])
            zone_tag = f" [{r['target_zone']}]" if r.get("target_zone") else ""
            print(f"  {i}. [{r['grade']}]{zone_tag} {r['name']}({r['ticker']}) "
                  f"{r['total_score']}점{oh} ({r['n_sources']}개 소스: {srcs})")
            if r.get("estimated_target"):
                dir_icon = {"RISING": "▲", "FALLING": "▼", "STABLE": "─", "NEW": "★"}.get(r.get("target_direction", ""), "")
                print(f"     기관목표:{r['estimated_target']:,} (갭:{r.get('target_gap_pct',0):+.1f}%) {dir_icon} "
                      f"| 진입:{r.get('entry_price',0):,}  손절:{r.get('stop_loss',0):,}  "
                      f"목표:{r.get('target_price',0):,}")
            else:
                print(f"     진입:{r.get('entry_price',0):,}  손절:{r.get('stop_loss',0):,}  "
                      f"목표:{r.get('target_price',0):,} | {cond}")
            ma5g = r.get("ma5_gap_pct", 0)
            ma5e = r.get("ma5_entry", "")
            ma5_str = f"  📐 MA5 {ma5g:+.1f}% [{ma5e}]" if ma5e else ""
            intel_str = f"  🌐{r['intel_tag']}" if r.get("intel_tag") else ""
            print(f"     근거: {reasons_str}{ma5_str}{intel_str}")
        print(f"{'─'*60}")
    else:
        print("\n  ⚠ 매수 적합 종목 없음 — 전체 관망 추천")

    # 나머지 관찰 종목 간략 출력
    rest = [r for r in results if r["grade"] in buyable_grades][5:]
    if rest:
        print(f"\n  [기타 관심종목]")
        for r in rest:
            print(f"    - {r['name']}({r['ticker']}) {r['total_score']}점 [{r['grade']}]")

    # 날짜 기입 + JSON 저장
    now = datetime.now()
    # 내일 날짜 (금→월, 토→월, 일→월)
    wd = now.weekday()
    if wd == 4:      # 금 → 월
        target = now + timedelta(days=3)
    elif wd == 5:    # 토 → 월
        target = now + timedelta(days=2)
    elif wd == 6:    # 일 → 월
        target = now + timedelta(days=1)
    else:
        target = now + timedelta(days=1)

    output = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "target_date": target.strftime("%Y-%m-%d"),
        "target_date_label": f"{target.month}/{target.day}({calendar.day_abbr[target.weekday()]})",
        "total_candidates": len(results),
        "stats": grade_stats,
        "top5": [r["ticker"] for r in top5],
        "picks": results,
        "market_intel": {
            "mood": intel_mood,
            "forecast": intel.get("kr_open_forecast", ""),
            "forecast_reason": intel.get("kr_forecast_reason", ""),
            "hot_themes": intel_themes,
            "summary": intel.get("us_market_summary", ""),
        } if intel_mood else {},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUTPUT_PATH}")
    print(f"[대상일] {output['target_date_label']} ({output['target_date']})")


if __name__ == "__main__":
    main()
