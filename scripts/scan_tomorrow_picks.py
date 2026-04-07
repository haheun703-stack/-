"""
내일 추천 종목 통합 스캐너 — 12개 시그널 교차 검증

12개 시그널 소스를 통합하여 최종 매수 추천 종목을 산출합니다.

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
v10 변경: 전략L(국적별 수급 7 Secrets) — 킬필터 + 부스트 + 소스

Usage:
    python scripts/scan_tomorrow_picks.py            # 기본 모드
    python scripts/scan_tomorrow_picks.py --mode war  # 공포탐욕 모드
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import calendar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = DATA_DIR / "tomorrow_picks.json"

# ── 학습 피드백 루프: 시그널 소스별 적중률 → 가중치 ──
_SOURCE_NAME_MAP = {
    "눌림목": "pullback_scan",
    "세력감지": "whale_detect",
    "매집추적": "accumulation_tracker",
    "이벤트": "dart_event",
    "수급폭발": "volume_spike",
    "동반매수": "dual_buying",
    "릴레이": "relay_signal",
    "그룹순환": "group_relay",
    "퀀텀": "quantum_signal",
    "이벤트촉매": "dart_event",
    "밸류체인": "value_chain",
    "기술촉매": "technical_trigger",
    "이벤트섹터": "event_sector",
}


_learning_weights_cache: dict | None = None


def _load_learning_weights() -> dict:
    """learning_weights.json 로드 (모듈 레벨 캐싱). 실패 시 빈 dict."""
    global _learning_weights_cache
    if _learning_weights_cache is not None:
        return _learning_weights_cache
    path = DATA_DIR / "market_learning" / "learning_weights.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            _learning_weights_cache = data.get("weights", {})
            return _learning_weights_cache
    except Exception:
        pass
    _learning_weights_cache = {}
    return _learning_weights_cache


# ──────────────────────────────────────────
# 공포탐욕(war) 모드 오버라이드 설정
# ──────────────────────────────────────────
WAR_MODE_OVERRIDES = {
    "min_sources": 1,            # 1소스만으로도 추천 (기본 3)
    "min_sources_alt_score": 55, # 대체 점수 기준 완화 (기본 70)
    "veto_avoid": False,         # AI AVOID 거부권 비활성
    "pool_only": True,           # 컨센서스 풀 내 종목만 추천
    "dynamic_slots": False,      # 슬롯 고정 (bearish에서 축소 방지)
    "label": "[공포탐욕]",
}

# ──────────────────────────────────────────
# FLOWX 공개추천 모드 오버라이드 설정
# 주린이(개인투자자) 대상 — 시장 상황과 무관하게 3~5일 상승 종목 추천
# ──────────────────────────────────────────
FLOWX_MODE_OVERRIDES = {
    "min_sources": 1,            # 1소스만으로도 추천 (기본 3)
    "min_sources_alt_score": 50, # 대체 점수 기준 완화 (기본 70)
    "veto_avoid": False,         # AI AVOID 거부권 비활성
    "pool_only": False,          # 컨센서스 풀 제한 없음
    "dynamic_slots": False,      # 슬롯 고정 (bearish에서 축소 방지)
    "regime_boost_floor": 0.8,   # 레짐 부스트 최소값 보장 (BEAR에서도 0.8x 이상)
    "accumulation_boost": 1.3,   # 매집추적 소스 추가 가산 (적중률 1위 소스 강화)
    "min_top_count": 5,          # TOP 최소 5개 보장 (관찰 등급까지 승격)
    "label": "[FLOWX]",
}

# ──────────────────────────────────────────
# 전략 그룹 정의: 스윙(3~7일) vs 단타(1~3일)
# ──────────────────────────────────────────
STRATEGY_GROUPS = {
    "swing": {
        "label": "스윙(3~7일)",
        "slots": 5,
        "sources": {"릴레이", "그룹순환", "눌림목", "퀀텀", "동반매수", "이벤트촉매", "이벤트", "밸류체인", "국적수급", "기술촉매", "이벤트섹터"},
        "overlap_pairs": [("릴레이", "그룹순환"), ("기술촉매", "이벤트섹터")],
    },
    "short": {
        "label": "단타(1~3일)",
        "slots": 5,
        "sources": {"수급폭발", "세력감지", "매집추적"},
        "overlap_pairs": [("세력감지", "매집추적")],
    },
}


def calc_effective_source_count(source_names: list[str], group_key: str) -> float:
    """그룹 내 유효 소스 수 계산 — 겹치는 소스쌍 보정.

    릴레이+그룹순환 동시 → 2가 아닌 1.5소스로 카운트.
    세력감지+매집추적 동시 → 마찬가지.
    """
    group = STRATEGY_GROUPS[group_key]
    group_sources = [s for s in source_names if s in group["sources"]]
    n = float(len(group_sources))
    for a, b in group.get("overlap_pairs", []):
        if a in group_sources and b in group_sources:
            n -= 0.5
    return max(n, 0)


def classify_strategy_group(source_names: list[str]) -> str:
    """소스 분포 기반 전략 그룹 판별 → "swing" / "short" / "both" """
    swing_cnt = sum(1 for s in source_names if s in STRATEGY_GROUPS["swing"]["sources"])
    short_cnt = sum(1 for s in source_names if s in STRATEGY_GROUPS["short"]["sources"])
    if swing_cnt > 0 and short_cnt > 0:
        return "both"
    if short_cnt > 0:
        return "short"
    return "swing"


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
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def build_name_map() -> dict[str, str]:
    """종목코드 → 종목명 매핑 (universe.csv 최우선 → CSV 폴백 → pykrx 폴백)."""
    name_map = {}
    # 1) universe.csv (최우선 — VPS에서도 안정적)
    uni_path = DATA_DIR / "universe.csv"
    if uni_path.exists():
        try:
            import csv as csv_mod
            with open(uni_path, encoding="utf-8-sig") as uf:
                reader = csv_mod.DictReader(uf)
                for row in reader:
                    t = row.get("ticker", row.get("code", "")).strip()
                    n = row.get("name", row.get("종목명", "")).strip()
                    if t and n:
                        name_map[t] = n
        except Exception:
            pass
    if name_map:
        logger.info("[picks] universe.csv 종목명 %d건 로드", len(name_map))
        return name_map
    # 2) stock_data_daily/ CSV 파일명
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    if name_map:
        return name_map
    # VPS에 CSV 없을 경우 pykrx 폴백
    try:
        from pykrx import stock as krx
        from datetime import timedelta
        dt = datetime.now()
        for _ in range(7):
            ds = dt.strftime("%Y%m%d")
            try:
                for mkt in ("KOSPI", "KOSDAQ"):
                    for t in krx.get_market_ticker_list(ds, market=mkt):
                        if t not in name_map:
                            name_map[t] = krx.get_market_ticker_name(t)
                if name_map:
                    logger.info("[picks] pykrx 종목명 %d건 로드", len(name_map))
                    return name_map
            except Exception:
                pass
            dt -= timedelta(days=1)
    except ImportError:
        pass
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


def collect_event_catalyst() -> dict[str, dict]:
    """소스11: 이벤트 촉매 — 정책/협력/합병 등 외부 이벤트 수혜종목 (전략 F)

    data/event_catalyst.json에서 로드. expires 지나면 자동 무시.
    """
    ec = load_json("event_catalyst.json")
    if not ec:
        return {}

    # 만료일 체크
    expires = ec.get("expires", "")
    if expires:
        try:
            exp_date = datetime.strptime(expires, "%Y-%m-%d").date()
            if datetime.now().date() > exp_date:
                logger.info("[이벤트촉매] 만료됨 (%s) — 무시", expires)
                return {}
        except ValueError:
            pass

    event_name = ec.get("event", "이벤트")
    result = {}
    for item in ec.get("stocks", []):
        ticker = item.get("ticker", "")
        if not ticker:
            continue
        result[ticker] = {
            "source": "이벤트촉매",
            "score": item.get("score", 60),
            "name": item.get("name", ""),
            "detail": f"[{item.get('sector','')}] {item.get('detail','')} ({event_name})",
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


def collect_value_chain() -> dict[str, dict]:
    """소스12: 밸류체인 릴레이 (대장주→소부장)"""
    vc = load_json("value_chain_relay.json")
    result = {}
    for sector_data in vc.get("fired_sectors", []):
        sector = sector_data.get("sector", "")
        leader_names = [l["name"] for l in sector_data.get("leaders", [])]
        for item in sector_data.get("candidates", []):
            ticker = item.get("ticker", "")
            if not ticker:
                continue
            result[ticker] = {
                "source": "밸류체인",
                "score": item.get("score", 0),
                "name": item.get("name", ""),
                "detail": f"{sector} 소부장 ({'+'.join(leader_names)}↑)",
            }
    return result


def collect_technical_trigger() -> dict[str, dict]:
    """소스13: 기술촉매 — processed/*.parquet 전수 스캔으로 급등 전조 패턴 감지.

    3가지 트리거 (백테스트 입증 팩터 기반):
      T1 PULLBACK_15: 고점 대비 -15%+, RSI<40, 외인 or 기관 매수전환 → 70점
      T2 VOL_STOCH_REV: vol_z≥2.0, stoch<35 상승전환, close>MA60 → 65점
      T3 BREAKOUT_60D: 60일 신고가, 외인+기관 동시 순매수 → 60점
    """
    result = {}
    parquet_files = list(PROCESSED_DIR.glob("*.parquet"))
    if not parquet_files:
        return result

    trigger_counts = {"T1": 0, "T2": 0, "T3": 0}

    for pf in parquet_files:
        ticker = pf.stem
        try:
            df = pd.read_parquet(pf).tail(25)
            if len(df) < 10:
                continue
            last = df.iloc[-1]
            close = float(last.get("close", 0))
            if close <= 0:
                continue

            # 공통 지표
            rsi = float(last.get("rsi_14", 50))
            stoch_k = float(last.get("stoch_slow_k", 50))
            stoch_d = float(last.get("stoch_slow_d", 50))
            ma60 = float(last.get("sma_60", 0))
            vol_z = float(last.get("vol_z", 0) or 0)
            high_252 = float(last.get("high_252", close))
            drawdown = ((close / high_252) - 1) * 100 if high_252 > 0 else 0

            # 수급: 외인/기관 5일 합산
            f5 = float(np.nansum(df.tail(5)["외국인합계"].values)) if "외국인합계" in df.columns else 0
            i5 = float(np.nansum(df.tail(5)["기관합계"].values)) if "기관합계" in df.columns else 0

            # 60일 최고가 (processed에 high_60이 없으면 직접 계산)
            if "high" in df.columns and len(df) >= 20:
                high_60d = float(df.tail(min(len(df), 25))["high"].max())
            else:
                high_60d = close

            triggers = []

            # T1: PULLBACK_15 — 고점대비 -15%+, RSI<40, 수급 전환
            if drawdown <= -15 and rsi < 40 and (f5 > 0 or i5 > 0):
                triggers.append(("PULLBACK_15", 70, f"낙폭{drawdown:.0f}% RSI{rsi:.0f}"))
                trigger_counts["T1"] += 1

            # T2: VOL_STOCH_REV — 거래량 폭발 + 스토캐스틱 바닥 반전 + MA60 위
            if vol_z >= 2.0 and stoch_k < 35 and stoch_k > stoch_d and close > ma60 > 0:
                triggers.append(("VOL_STOCH_REV", 65, f"vol_z{vol_z:.1f} Stoch{stoch_k:.0f}"))
                trigger_counts["T2"] += 1

            # T3: BREAKOUT_60D — 60일 신고가 + 외인+기관 동시 순매수
            if close >= high_60d * 0.99 and f5 > 0 and i5 > 0:
                triggers.append(("BREAKOUT_60D", 60, f"60일고가 외인+기관"))
                trigger_counts["T3"] += 1

            if triggers:
                # 최고 점수 트리거 선택
                best = max(triggers, key=lambda x: x[1])
                result[ticker] = {
                    "source": "기술촉매",
                    "score": best[1],
                    "name": "",  # name_map에서 나중에 채워짐
                    "detail": f"{best[0]}: {best[2]}",
                    "trigger": best[0],
                }
        except Exception:
            continue

    if result:
        logger.info("[기술촉매] %d종목 감지 (T1:%d T2:%d T3:%d)",
                    len(result), trigger_counts["T1"], trigger_counts["T2"], trigger_counts["T3"])
        if len(result) > 100:
            logger.warning("[기술촉매] 100종목 초과 (%d) — 트리거 조건 검토 필요", len(result))
    return result


def collect_event_sector_source() -> dict[str, dict]:
    """소스14: 이벤트섹터 — market_intelligence의 sector_boost + key_events에서
    relay_sectors 종목을 독립 소스로 추가.

    sector_boost score≥3 이거나, key_events의 abs(kr_impact_score)≥3인 수혜 섹터의
    relay_sectors 종목들을 후보풀에 진입시킨다.
    """
    result = {}
    intel = load_json("market_intelligence.json")
    if not intel:
        return result

    # relay_sectors.yaml 로드
    relay_path = PROJECT_ROOT / "config" / "relay_sectors.yaml"
    if not relay_path.exists():
        return result
    try:
        with open(relay_path, encoding="utf-8") as f:
            relay_cfg = yaml.safe_load(f) or {}
    except Exception:
        return result

    sectors_cfg = relay_cfg.get("relay_engine", {}).get("sectors", {})

    # 수혜 섹터 수집: key_events에서 kr_impact_score >= 3 (수혜 방향)인 섹터
    boosted_sectors = {}  # sector_name → max_score
    for evt in intel.get("key_events", []):
        score = evt.get("kr_impact_score", 0)
        direction = evt.get("kr_impact_direction", "")
        if score >= 3 and direction == "수혜":
            for sec in evt.get("kr_sectors_affected", []):
                if sec not in boosted_sectors or score > boosted_sectors[sec]:
                    boosted_sectors[sec] = score

    # sector_boost에서도 추가 (양수 3 이상)
    for sec, sb in intel.get("sector_boost", {}).items():
        if sb >= 3:
            if sec not in boosted_sectors or sb > boosted_sectors[sec]:
                boosted_sectors[sec] = sb

    if not boosted_sectors:
        return result

    # 섹터명 → relay_sectors 키 브릿지
    SECTOR_RELAY_BRIDGE = {
        "반도체": "ai_semiconductor",
        "AI 반도체": "ai_semiconductor",
        "IT/소프트웨어": "ai_semiconductor",
        "방산": "defense",
        "조선": "shipbuilding",
        "화학/에너지": "energy",
        "에너지": "energy",
        "2차전지": "ev_battery",
        "자동차": "ev_battery",
        "바이오": "bio_pharma",
        "원전": "nuclear",
        "로봇": "robotics",
    }

    for sec_name, boost_score in boosted_sectors.items():
        relay_key = SECTOR_RELAY_BRIDGE.get(sec_name)
        if not relay_key or relay_key not in sectors_cfg:
            continue

        sector_data = sectors_cfg[relay_key]
        # kr_leaders + kr_secondaries 종목 수집
        all_stocks = []
        for s in sector_data.get("kr_leaders", []):
            all_stocks.append(s)
        for s in sector_data.get("kr_secondaries", []):
            all_stocks.append(s)

        for s in all_stocks:
            ticker = s.get("ticker", "")
            if not ticker or ticker in result:
                continue
            # 점수: boost_score × 12 (상한 70)
            score = min(boost_score * 12, 70)
            result[ticker] = {
                "source": "이벤트섹터",
                "score": score,
                "name": s.get("name", ""),
                "detail": f"{sec_name} 수혜 (강도{boost_score})",
            }

    if result:
        logger.info("[이벤트섹터] %d종목 | 수혜섹터: %s",
                    len(result), ", ".join(f"{k}({v})" for k, v in boosted_sectors.items()))
    return result


# ──────────────────────────────────────────
# 전략 L: 국적별 수급 7 Secrets (nationality_signal.json)
# ──────────────────────────────────────────

def load_nationality_signals() -> dict:
    """국적별 수급 시그널 로드 → {ticker: signal_dict}.

    Returns:
        {
            ticker: {
                "signal": "STRONG_BUY" | "BUY" | "NEUTRAL" | "CAUTION" | "SELL",
                "score": float (-28 ~ +53),
                "pattern": str,         # QUIET_ACCUM, ABSORPTION 등
                "inst_trend": float,    # 기관 추세
                "hedge_trend": float,   # 헤지 추세
                "retail_pattern": str,  # RETAIL_SUPPORT 등
                "name": str,
            }
        }
    """
    data = load_json("krx_nationality/nationality_signal.json")
    if not data:
        return {}
    result = {}
    for sig in data.get("signals", []):
        ticker = sig.get("ticker", "")
        if ticker:
            result[ticker] = sig
    return result


def get_nationality_kill_tickers(nat_signals: dict) -> set[str]:
    """국적별 수급 SELL 시그널 종목 → 킬 필터.

    기관장기 전면 이탈(SELL) 패턴이면 매수 후보에서 제외.
    조건: signal == SELL AND score <= -10
    """
    kills = set()
    for ticker, sig in nat_signals.items():
        signal = sig.get("signal", "")
        score = sig.get("score", 0)
        if signal == "SELL" and score <= -10:
            kills.add(ticker)
    return kills


def load_institutional_accumulation() -> dict[str, dict]:
    """기관매집 조기감지 데이터 로드 → {ticker: alert_dict}.

    EARLY_SURGE/EARLY_ACCEL/EARLY_DUAL/STRONG/MODERATE 등급별
    scan_tomorrow_picks 점수 부스트에 사용.
    """
    data = load_json("institutional_flow/accumulation_alert.json")
    if not data:
        return {}
    result = {}
    for alert in data.get("stock_alerts", []):
        ticker = alert.get("ticker", "")
        if ticker:
            result[ticker] = alert
    return result


def load_foreign_exhaustion() -> dict[str, dict]:
    """외인소진율 데이터 로드 → {ticker: signal_dict}.

    FE1(90%+한도임박), FE2(+5%p급등), FE3(70~89%+상승) 신호
    scan_tomorrow_picks 점수 부스트에 사용.
    """
    data = load_json("foreign_exhaustion/daily_exhaustion.json")
    if not data:
        return {}
    result = {}
    # fe2 (소진율 급등) — 가장 유용한 조기 신호
    for item in data.get("fe2_signals", []):
        ticker = item.get("ticker", "")
        if ticker:
            result[ticker] = {**item, "fe_type": "FE2"}
    # fe1 (한도 임박) — 추가 매수 제한 우려 → 오히려 주의
    for item in data.get("fe1_signals", []):
        ticker = item.get("ticker", "")
        if ticker and ticker not in result:
            result[ticker] = {**item, "fe_type": "FE1"}
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


# AI Brain 섹터명 → stock_to_sector 섹터명 브릿지
AI_SECTOR_BRIDGE = {
    "반도체": ["반도체", "IT"],
    "IT/소프트웨어": ["IT", "소프트웨어"],
    "금융": ["금융", "은행", "보험", "증권"],
    "에너지(원전/정유)": ["에너지화학", "에너지"],
    "방산": ["조선", "기계", "방산"],
    "2차전지": ["2차전지", "전지", "전자부품"],
    "자동차": ["자동차", "자동차부품"],
    "바이오": ["바이오", "제약"],
    "건설": ["건설"],
    "유통": ["유통"],
}


def _match_ai_sector(stock_sector: str, ai_sector_outlook: dict) -> dict | None:
    """stock_to_sector의 섹터명 → AI brain sector_outlook 매칭.

    직접 매칭 시도 후, AI_SECTOR_BRIDGE 역방향 매칭.
    """
    # 직접 매칭
    if stock_sector in ai_sector_outlook:
        return ai_sector_outlook[stock_sector]
    # 브릿지: AI 섹터명 → stock_to_sector 섹터명 리스트에 stock_sector 포함?
    for ai_name, bridge_names in AI_SECTOR_BRIDGE.items():
        if stock_sector in bridge_names and ai_name in ai_sector_outlook:
            return ai_sector_outlook[ai_name]
    return None


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


def load_regime_name() -> str:
    """레짐 매크로 시그널에서 current_regime 문자열 로드 (BULL/NORMAL/BEAR/CRISIS)"""
    macro = load_json("regime_macro_signal.json")
    return macro.get("current_regime", "NORMAL")


def load_vix_level() -> float:
    """brain_decision.json에서 VIX 로드 → 동적 손절폭 결정에 사용."""
    brain = load_json("brain_decision.json")
    return brain.get("vix_level", 20.0) if isinstance(brain, dict) else 20.0


def calc_dynamic_stop_range(vix: float) -> tuple[float, float]:
    """VIX 기반 동적 손절 범위 (최소%, 최대%) 반환.

    VIX 낮음(<15) → 타이트 손절 (-3%~-5%) : 변동성 작으니 빨리 컷
    VIX 보통(15~25) → 기본 (-4%~-6%)
    VIX 높음(25~35) → 넓은 손절 (-5%~-8%) : 변동성 크니 여유
    VIX 극단(>35) → 최대 (-6%~-10%) : 패닉 변동성 허용
    """
    if vix < 15:
        return (0.97, 0.95)   # -3% ~ -5%
    elif vix < 25:
        return (0.96, 0.94)   # -4% ~ -6%
    elif vix < 35:
        return (0.95, 0.92)   # -5% ~ -8%
    else:
        return (0.94, 0.90)   # -6% ~ -10%


def load_institutional_targets() -> dict:
    """기관 추정 목표가 데이터 로드."""
    data = load_json("institutional_targets.json")
    return data.get("targets", {})


def load_market_intelligence() -> dict:
    """Perplexity 시장 인텔리전스 데이터 로드."""
    return load_json("market_intelligence.json")


# ── 전략 M: 시나리오 뉴스 부스트 (원자재 원가 갭 + 시나리오 체인) ──

# 시나리오별 관련 원자재 매핑
_SCENARIO_COMMODITY_MAP = {
    "WAR_MIDDLE_EAST": ["wti", "gold"],
    "WAR_UKRAINE_ESCALATION": ["natural_gas", "gold"],
    "OIL_SPIKE": ["wti", "natural_gas"],
    "COMMODITY_SUPERCYCLE": ["copper", "tio2", "aluminum", "uranium"],
    "AI_POWER_DEMAND": ["uranium", "natural_gas"],
    "SPACEX_IPO": [],
    "SEMICONDUCTOR_CYCLE_UP": [],
    "FED_RATE_CUT": [],
    "FED_RATE_HIKE": [],
    "CHINA_STIMULUS": ["copper", "aluminum"],
    "USD_KRW_SPIKE": [],
}


def load_scenario_data() -> dict:
    """활성 시나리오 + 원자재 원가 갭 + 시나리오 체인 로드."""
    scenarios = load_json("scenarios/active_scenarios.json")
    commodities = load_json("commodity_prices.json")
    chains = load_json("scenarios/scenario_chains.json")
    return {
        "active": scenarios.get("scenarios", {}),
        "commodities": commodities.get("commodities", {}),
        "chains": chains.get("scenarios", []),
    }


def build_scenario_ticker_map(scenario_data: dict) -> dict[str, list[dict]]:
    """시나리오 체인 → 티커별 시나리오 매핑 구축.

    Returns: {ticker: [{scenario_id, scenario_name, phase, logic, ...}, ...]}
    """
    ticker_map: dict[str, list[dict]] = {}
    active = scenario_data["active"]
    chains = scenario_data["chains"]

    for chain in chains:
        sc_id = chain.get("id", "")
        sc_name = chain.get("name", "")

        # 활성 시나리오가 아니면 스킵
        if sc_id not in active:
            continue
        sc_info = active[sc_id]
        current_phase = sc_info.get("current_phase", 0)
        sc_score = sc_info.get("score", 0)

        # score 10 미만이면 의미 없음 → 스킵
        if sc_score < 10:
            continue

        for phase_info in chain.get("chain", []):
            phase_num = phase_info.get("phase", 0)
            # 현재 phase ±1 범위까지 유효 (약간 선행 허용)
            if abs(phase_num - current_phase) > 1:
                continue

            for ticker_str in phase_info.get("hot_tickers", []):
                # "012450:한화에어로스페이스" 형태 파싱
                parts = ticker_str.split(":", 1)
                ticker = parts[0]

                if ticker not in ticker_map:
                    ticker_map[ticker] = []

                ticker_map[ticker].append({
                    "scenario_id": sc_id,
                    "scenario_name": sc_name,
                    "scenario_score": sc_score,
                    "phase": phase_num,
                    "phase_name": phase_info.get("name", ""),
                    "logic": phase_info.get("logic", ""),
                    "hot_sectors": phase_info.get("hot_sectors", []),
                })

    return ticker_map


def build_scenario_narrative(
    scenarios: list[dict], commodities: dict
) -> tuple[str, str, dict]:
    """종목의 시나리오 기반 narrative 생성.

    Returns: (scenario_tag, narrative, risk_reward)
    """
    if not scenarios:
        return "", "", {}

    # 가장 점수 높은 시나리오 선택
    best = max(scenarios, key=lambda x: x["scenario_score"])
    tag = best["scenario_id"]

    # 1-line narrative
    narrative = f"{best['scenario_name']} → {best['phase_name']}: {best['logic']}"

    # risk_reward: 관련 원자재 cost gap 기반
    risk_reward: dict = {}
    related = _SCENARIO_COMMODITY_MAP.get(tag, [])
    for comm_key in related:
        if comm_key in commodities:
            comm = commodities[comm_key]
            gap = comm.get("cost_gap", {})
            if gap:
                risk_reward = {
                    "commodity": comm.get("name", comm_key),
                    "price": comm.get("price", 0),
                    "cost_gap_pct": gap.get("gap_pct", 0),
                    "zone": gap.get("zone", ""),
                    "downside_limited": gap.get("zone") in ("buy", "watch"),
                }
                break

    return tag, narrative, risk_reward


def load_ai_brain_data() -> dict:
    """AI 두뇌 판단 결과 로드 (전략 H)."""
    return load_json("ai_brain_judgment.json")


def load_v3_picks() -> dict:
    """v3 AI Brain 최종 picks 로드 (전략 I)."""
    data = load_json("ai_v3_picks.json")
    if not data:
        return {}
    # 당일 데이터만 유효
    today = datetime.now().strftime("%Y-%m-%d")
    if data.get("decision_date", data.get("analysis_date", "")) != today:
        return {}
    return data


def load_morning_reports() -> dict:
    """장전 리포트 스캔 결과 로드 (전략 G)."""
    data = load_json("morning_reports.json")
    if not data:
        return {}
    # 당일 또는 전일 데이터만 유효
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if data.get("date") not in (today, yesterday):
        return {}
    return data


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
    group_source_count: float | None = None,
) -> dict:
    """5축 100점 + 과열패널티 통합 점수 계산 (v5)

    기본 100점 배분:
      다중시그널(25) + 개별점수(20) + 기술적(25) + 수급(20) + 안전(10)
    v5: 동반매수 단독 보장 삭제, 다른 소스와 겹칠 때만 소액 가산
    과열 패널티: 최대 -25점

    group_source_count: 그룹별 유효 소스 수 (겹침 보정 적용). None이면 len(sources) 사용.
    """

    # ── 축1: 다중 시그널 (25점) ──
    n_eff = group_source_count if group_source_count is not None else len(sources)
    n_sources = len(sources)  # 전체 소스 수 (동반매수 보너스 등에서 사용)
    if n_eff >= 4:
        multi_score = 25
    elif n_eff >= 3:
        multi_score = 20
    elif n_eff >= 2:
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

    # ── 학습 가중치 적용 (축2 계산 전, 복사본 사용) ──
    learning_w = _load_learning_weights()
    if learning_w:
        sources = [dict(s) for s in sources]  # shallow copy로 원본 보호
        for s in sources:
            key = _SOURCE_NAME_MAP.get(s.get("source", ""))
            if key and key in learning_w:
                m = learning_w[key].get("multiplier", 1.0)
                s["score"] = s["score"] * m

    # ── 승자 소스 교차 보너스 ──
    src_names = {s.get("source", "") for s in sources}
    has_pullback = "눌림목" in src_names
    has_accumulation = "매집추적" in src_names
    if has_pullback and has_accumulation:
        multi_score += 5  # 적중률 TOP2 동시 발생 → 강한 시그널
    elif has_pullback or has_accumulation:
        multi_score += 2  # 단독이라도 승자 소스 가산

    # ── 축2: 개별 점수 평균 (20점) ──
    avg_src_score = np.mean([s["score"] for s in sources]) if sources else 0
    individual_score = min(avg_src_score / 100 * 20, 20)

    # ── parquet 기반 기술적 지표 ──
    rsi = 50; adx = 20; above_ma60 = False; above_ma20 = False
    bb_pos = 50; drawdown = 0; foreign_5d = 0; inst_5d = 0
    close = 0; price_change = 0; ma20 = 0; ma60 = 0
    stoch_k = 50; stoch_d = 50; trix_gx = False; macd_rising = False
    ret_5d = 0; ret_20d = 0; low_20d = 0
    trix = 0; trix_signal = 0; volume_surge = False

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
        volume_surge = (parquet_data.get("vol_z", 0) >= 1.5
                        or parquet_data.get("volume_surge_ratio", 0) >= 1.5)

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

    # SAR 추세 (0~3점)
    sar_trend = parquet_data.get("sar_trend", 0) if parquet_data else 0
    sar_val = parquet_data.get("sar", 0) if parquet_data else 0
    if sar_trend == 1 and close > sar_val > 0:
        tech_score += 3   # SAR 상향 + 가격 위
    elif sar_trend == 1:
        tech_score += 1   # SAR 상향이지만 근접

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

    # 수급 반전 보너스 (v11): 10일 매도→2일 매수전환 시 가산
    if parquet_data:
        if parquet_data.get("foreign_reversal"):
            flow_score += 3
        if parquet_data.get("inst_reversal"):
            flow_score += 2

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

    # SAR 하향 + 가격 < SAR → 하락추세 페널티 (반전 근접 시 감면)
    if sar_trend == -1 and 0 < sar_val and close < sar_val:
        sar_gap_pct = (sar_val - close) / close * 100 if close > 0 else 99
        if sar_gap_pct <= 2.0 and adx >= 25:
            # SAR 반전 근접(2% 이내) + 추세 강도 있음 → 감면
            overheat_penalty += 1
            overheat_flags.append(f"SAR↓근접({sar_gap_pct:.1f}%)")
        elif sar_gap_pct <= 3.0 and volume_surge:
            # 거래량 급증 시에도 감면
            overheat_penalty += 1
            overheat_flags.append(f"SAR↓+거래량({sar_gap_pct:.1f}%)")
        else:
            overheat_penalty += 3
            overheat_flags.append("SAR↓")

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

    # 손절가: VIX 기반 동적 범위 + 20일 저점/MA20 지지선 활용
    # (현재가 이상인 후보는 제외 — 한화에어로 버그 방지)
    vix = load_vix_level()
    stop_min_ratio, stop_max_ratio = calc_dynamic_stop_range(vix)
    # stop_min_ratio: 최소 손절폭 (예: 0.97 = -3%), stop_max_ratio: 최대 손절폭 (예: 0.95 = -5%)
    stop_candidates = [v for v in [low_20d, ma20 * 0.98] if 0 < v < close]
    stop = max(stop_candidates) if stop_candidates else close * stop_min_ratio
    stop = max(stop, close * stop_max_ratio)  # 손절폭 최대 제한
    stop = min(stop, close * stop_min_ratio)  # 손절폭 최소 제한

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

    # ── 손절가를 entry 기준으로 재조정 (VIX 동적) ──
    # entry가 MA5 수준으로 낮아졌을 때 stop이 entry 근처면 R:R 무의미
    stop = min(stop, entry * stop_min_ratio)   # entry 대비 최소 손절
    stop = max(stop, entry * stop_max_ratio)   # entry 대비 최대 손절

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


_FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]


def compute_stock_fibonacci(ticker: str) -> dict:
    """개별 종목 피보나치 되돌림 레벨 계산 (20일/60일 스윙)."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        return {}
    try:
        df = pd.read_parquet(pq, columns=["high", "low", "close"])
        if len(df) < 20:
            return {}
        close = float(df["close"].iloc[-1])
        result = {}
        for label, window in [("20d", 20), ("60d", 60)]:
            if len(df) < window:
                continue
            recent = df.tail(window)
            high = float(recent["high"].max())
            low = float(recent["low"].min())
            if high <= low:
                continue
            levels = sorted(round(high - (high - low) * r) for r in _FIB_RATIOS)
            support = max((l for l in levels if l < close), default=None)
            resistance = min((l for l in levels if l > close), default=None)
            result[label] = {
                "high": round(high),
                "low": round(low),
                "support": support,
                "resistance": resistance,
                "position_pct": round((close - low) / (high - low) * 100, 1),
            }
        return result
    except Exception:
        return {}


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


def get_parquet_raw_df(ticker: str, min_rows: int = 80) -> pd.DataFrame | None:
    """parquet에서 raw DataFrame 반환 (alpha_scoring v3용).

    alpha_scoring.calc_alpha_score()에 필요한 최소 80일 데이터 반환.
    columns: close, high, low, volume, 기관합계, 외국인합계
    """
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path)
        if len(df) < min_rows:
            return None
        # 필수 컬럼 확인
        required = ["close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                return None
        return df
    except Exception as e:
        logger.warning("parquet raw 읽기 실패 %s: %s", ticker, e)
        return None


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

        # 수급 반전 감지: 직전 10일 순매도 → 최근 2일 순매수 전환
        foreign_reversal = False
        inst_reversal = False
        if "외국인합계" in df.columns and len(df) >= 12:
            f_old = float(np.nansum(df.iloc[-12:-2]["외국인합계"].values))
            f_new = float(np.nansum(df.tail(2)["외국인합계"].values))
            foreign_reversal = (f_old < 0 and f_new > 0)
        if "기관합계" in df.columns and len(df) >= 12:
            i_old = float(np.nansum(df.iloc[-12:-2]["기관합계"].values))
            i_new = float(np.nansum(df.tail(2)["기관합계"].values))
            inst_reversal = (i_old < 0 and i_new > 0)

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
            "vol_ratio": float(last["volume"]) / float(df["volume"].mean()) if "volume" in df.columns and df["volume"].mean() > 0 else 0,
            "volume_surge_ratio": float(last.get("volume_surge_ratio", 0) or 0),
            "macd": float(last.get("macd", 0) or 0),
            "macd_signal": float(last.get("macd_signal", 0) or 0),
            "macd_histogram": macd_hist,
            "macd_histogram_prev": macd_hist_prev,
            "sar": float(last.get("sar", 0) or 0),
            "sar_trend": int(last.get("sar_trend", 0) or 0),
            "foreign_reversal": foreign_reversal,
            "inst_reversal": inst_reversal,
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
    foreign_5d: float = 0, inst_5d: float = 0,
    foreign_reversal: bool = False, inst_reversal: bool = False,
) -> str:
    """등급 분류 — 수급 게이트 + 하드 필터 (v5.1)

    수급 게이트 (v5 신규):
      - 외인+기관 동시 대량매도 → 최대 "보류" (어떤 테마도 수급 역행 불가)
      - 외인+기관 동시 순매도 → 최대 "관찰" (수급 확인 전까지 대기)
    v5.1: 수급 반전 감지 시 게이트 면제

    하드 디스퀄:
      - parquet 데이터 없음 → 데이터부족
      - Stoch >= 90 (극과열) → 최대 관찰
      - 5일 수익률 >= 15% (추격매수) → 최대 관찰
      - RSI >= 78 (과매수 극단) → 최대 관찰
    """
    if not has_data:
        return "데이터부족"

    # ── v5 수급 게이트: 수급이 1순위 필터 ──
    # 외인+기관 동시 대량매도 (각 -500억 이상) → 즉시 보류
    HEAVY_SELL = -50_000_000_000  # -500억
    if foreign_5d < HEAVY_SELL and inst_5d < HEAVY_SELL:
        return "보류"

    # 외인+기관 동시 순매도 → 최대 관찰 (수급 역행)
    # v5.1: 수급 반전(10일 매도→2일 매수전환) 감지 시 캡 면제
    flow_cap = None
    if foreign_5d < 0 and inst_5d < 0:
        if not (foreign_reversal or inst_reversal):
            flow_cap = "관찰"

    # 하드 디스퀄: 극과열/추격매수는 관찰 이상 불가
    is_disqualified = stoch_k >= 90 or ret_5d >= 15 or rsi >= 78

    if is_disqualified:
        return "관찰" if total_score >= 40 else "보류"

    if total_score >= 70 and n_sources >= 2:
        grade = "적극매수"
    elif total_score >= 60 and n_sources >= 2:
        grade = "매수"
    elif total_score >= 55:
        grade = "관심매수"
    elif total_score >= 40:
        grade = "관찰"
    else:
        grade = "보류"

    # 수급 캡 적용: 수급 역행이면 관찰까지만
    if flow_cap:
        grade_order = {"보류": 0, "데이터부족": 0, "관찰": 1, "관심매수": 2, "매수": 3, "적극매수": 4}
        cap_level = grade_order.get(flow_cap, 1)
        if grade_order.get(grade, 0) > cap_level:
            grade = flow_cap

    return grade


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="내일 추천 종목 통합 스캐너")
    parser.add_argument("--mode", choices=["normal", "war"], default="normal",
                        help="실행 모드: normal(기본), war(공포탐욕)")
    parser.add_argument("--flowx", action="store_true",
                        help="FLOWX 공개추천 모드 (주린이 대상, 필터 완화)")
    parser.add_argument("--no-send", action="store_true",
                        help="텔레그램 전송 생략 (테스트용)")
    args = parser.parse_args()

    is_war_mode = args.mode == "war"
    is_flowx_mode = args.flowx
    no_send = args.no_send

    # FLOWX 모드가 war 모드보다 우선
    if is_flowx_mode:
        mode_label = FLOWX_MODE_OVERRIDES["label"]
    elif is_war_mode:
        mode_label = WAR_MODE_OVERRIDES["label"]
    else:
        mode_label = "[기본]"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if is_flowx_mode:
        print(f"\n{'='*60}")
        print(f"  📡 FLOWX 공개추천 모드 — 주린이 대상 공격적 추천")
        print(f"  min_sources=1 | veto=off | regime_floor=0.8 | 매집부스트=1.3x")
        print(f"{'='*60}\n")
    elif is_war_mode:
        print(f"\n{'='*60}")
        print(f"  🔥 공포탐욕 모드 활성화 — 바닥잡이 설정 적용")
        print(f"  min_sources=1 | pool_only=true | veto_avoid=false")
        print(f"{'='*60}\n")

    name_map = build_name_map()

    # 11개 소스 수집 (v8: 9→10 앙상블, 전략D 매집추적, 전략F 이벤트촉매)
    src1 = collect_relay()
    src2 = collect_group_relay()
    src3 = collect_pullback()
    src4 = collect_quantum()
    src5 = collect_dual_buying()
    src6 = collect_force_hybrid()
    src7 = collect_dart_event()
    src9 = collect_volume_spike()
    src10 = collect_accumulation_tracker()
    src11 = collect_event_catalyst()
    src12 = collect_value_chain()
    src13 = collect_technical_trigger()
    src14 = collect_event_sector_source()

    print(f"[소스 수집] 릴레이:{len(src1)} 그룹순환:{len(src2)} "
          f"눌림목:{len(src3)} 퀀텀:{len(src4)} 동반매수:{len(src5)} "
          f"세력감지:{len(src6)} 이벤트:{len(src7)} 수급폭발:{len(src9)} "
          f"매집추적:{len(src10)} 이벤트촉매:{len(src11)} 밸류체인:{len(src12)} "
          f"기술촉매:{len(src13)} 이벤트섹터:{len(src14)}")

    # 전략 L: 국적별 수급 7 Secrets
    nat_signals = load_nationality_signals()
    nat_kill_tickers = set()
    if nat_signals:
        nat_kill_tickers = get_nationality_kill_tickers(nat_signals)
        nat_buy = sum(1 for s in nat_signals.values() if s.get("signal") in ("STRONG_BUY", "BUY"))
        nat_sell = sum(1 for s in nat_signals.values() if s.get("signal") in ("CAUTION", "SELL"))
        print(f"[국적수급 7S] {len(nat_signals)}종목 | "
              f"매수시그널:{nat_buy} 주의/매도:{nat_sell} 킬:{len(nat_kill_tickers)}")
    else:
        print("[국적수급 7S] 데이터 없음 — 전략L 스킵")

    # ── 전략 N: 기관매집 조기감지 부스트 (accumulation_alert) ──
    inst_accum = load_institutional_accumulation()
    if inst_accum:
        early_cnt = sum(1 for a in inst_accum.values() if a.get("grade", "").startswith("EARLY"))
        strong_cnt = sum(1 for a in inst_accum.values() if a.get("grade") == "STRONG")
        print(f"[기관매집] {len(inst_accum)}종목 | STRONG:{strong_cnt} 조기감지:{early_cnt}")
    else:
        print("[기관매집] 데이터 없음 — 전략N 스킵")

    # ── 전략 O: 외인소진율 부스트 (daily_exhaustion) ──
    fe_signals = load_foreign_exhaustion()
    if fe_signals:
        fe2_cnt = sum(1 for f in fe_signals.values() if f.get("fe_type") == "FE2")
        fe1_cnt = sum(1 for f in fe_signals.values() if f.get("fe_type") == "FE1")
        print(f"[외인소진율] {len(fe_signals)}종목 | 급등(FE2):{fe2_cnt} 한도임박(FE1):{fe1_cnt}")
    else:
        print("[외인소진율] 데이터 없음 — 전략O 스킵")

    # ── 공매도 체제 프로파일 ──
    try:
        from src.use_cases.short_regime_manager import get_short_regime_manager
        short_mgr = get_short_regime_manager()
        short_summary = short_mgr.summary()
        if short_summary["enabled"]:
            print(f"[공매도 체제] {short_summary['label']} | "
                  f"포지션×{short_summary['profile'].get('position_scale_mult', 1.0)} | "
                  f"슬롯×{short_summary['profile'].get('max_positions_scale', 1.0)} | "
                  f"SA floor={short_summary['profile'].get('sa_floor', 0.55)}")
        else:
            short_summary = {"enabled": False, "profile": {}}
    except Exception as e:
        logger.debug("공매도 체제 로드 실패: %s", e)
        short_summary = {"enabled": False, "profile": {}}

    # DART AVOID 필터 + 레짐 부스트 + 섹터 부스트 + 기관목표가 + 시장 인텔리전스
    avoid_tickers = load_dart_avoid_tickers()
    regime_mult = load_regime_boost()
    regime_name = load_regime_name()  # v3용: BULL/NORMAL/BEAR/CRISIS
    sector_boost_map = load_sector_momentum_boost()
    inst_targets = load_institutional_targets()
    intel = load_market_intelligence()

    # 전략 M: 시나리오 뉴스 부스트 (원자재 원가 갭 + 시나리오 체인)
    scenario_data = load_scenario_data()
    scenario_ticker_map = build_scenario_ticker_map(scenario_data)
    if scenario_ticker_map:
        active_sc = {s_id: s["score"] for s_id, s in scenario_data["active"].items() if s.get("score", 0) >= 10}
        print(f"[시나리오] {len(active_sc)}개 활성, {len(scenario_ticker_map)}종목 매핑 "
              f"({', '.join(f'{k}:{v}점' for k, v in sorted(active_sc.items(), key=lambda x: -x[1])[:3])})")
    else:
        print("[시나리오] 활성 시나리오 없음 — 전략M 스킵")

    if inst_targets:
        print(f"[기관목표가] {len(inst_targets)}종목 로드됨")
    if avoid_tickers:
        print(f"[DART AVOID] {len(avoid_tickers)}종목 자동 제외")
    print(f"[레짐 부스트] x{regime_mult:.1f}")
    active_boosts = {k: v for k, v in sector_boost_map.items() if v != 0}
    if active_boosts:
        print(f"[US섹터 부스트] {len(active_boosts)}섹터 활성: {active_boosts}")

    # 전략 G: 장전 리포트 부스트
    morning = load_morning_reports()
    report_boost_map = morning.get("report_boost_map", {})
    pplx_themes = morning.get("perplexity_themes", {})
    news_boost_map = morning.get("news_boost_map", {})
    if report_boost_map:
        print(f"[리포트 부스트] {len(report_boost_map)}종목 활성")
    if pplx_themes.get("hot_themes"):
        themes_str = " | ".join(t["theme"] for t in pplx_themes["hot_themes"][:3])
        print(f"[장전 테마] {themes_str}")

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

    # settings.yaml 로드 (AI 독립추천 등 설정용)
    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    yaml_config = {}
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

    # ── Alpha Scoring v3 (표시 전용) ──
    # v3는 점수/등급에 영향 없이, 백테스트 팩터 시그널을 보조 태그로만 표시
    alpha_v3_display = yaml_config.get("alpha_v3", {}).get("enabled", False)
    _v3_score = None
    if alpha_v3_display:
        try:
            from src.use_cases.alpha_scoring import calc_alpha_score as _v3_score
            print(f"  [v3 보조태그] 백테스트 팩터 시그널 표시 활성")
        except ImportError as e:
            logger.warning("alpha_scoring 모듈 임포트 실패: %s", e)
            alpha_v3_display = False

    # 전략 J: 컨센서스 풀 (consensus_screening.json)
    consensus_pool_cfg = yaml_config.get("consensus_pool", {})
    consensus_pool_enabled = consensus_pool_cfg.get("enabled", False)
    consensus_pool_only = consensus_pool_cfg.get("pool_only", False)
    # 공포탐욕/FLOWX 모드: pool_only 오버라이드
    if is_flowx_mode:
        consensus_pool_only = FLOWX_MODE_OVERRIDES["pool_only"]
    elif is_war_mode:
        consensus_pool_enabled = True
        consensus_pool_only = WAR_MODE_OVERRIDES["pool_only"]
    consensus_pool = {}  # ticker → {upside_pct, composite_score, grade, forward_per, ...}
    if consensus_pool_enabled:
        cs_data = load_json("consensus_screening.json")
        for p in cs_data.get("top_picks", []):
            t = p.get("ticker", "")
            if t:
                consensus_pool[t] = p
        # 풀 확장: 전체 필터 통과 종목 (all_picks 필드가 있으면)
        for p in cs_data.get("all_picks", []):
            t = p.get("ticker", "")
            if t and t not in consensus_pool:
                consensus_pool[t] = p
        if consensus_pool:
            print(f"[컨센서스 풀] {len(consensus_pool)}종목 로드 | "
                  f"pool_only={consensus_pool_only}")

    # 전략 I: v3 AI Brain picks
    v3_picks_data = load_v3_picks()
    v3_buy_map = {}  # ticker → {conviction, size_pct, strategy, ...}
    if v3_picks_data:
        for b in v3_picks_data.get("buys", []):
            t = b.get("ticker", "")
            if t:
                v3_buy_map[t] = b
        if v3_buy_map:
            print(f"[v3 Brain] {len(v3_buy_map)}종목 매수 결정 로드")

    # 전략 H: AI 두뇌 판단
    ai_brain_data = load_ai_brain_data()
    ai_brain_judgments = {}  # ticker → judgment dict
    if ai_brain_data and ai_brain_data.get("stock_judgments"):
        for j in ai_brain_data["stock_judgments"]:
            t = j.get("ticker", "")
            if t:
                ai_brain_judgments[t] = j
        ai_sentiment = ai_brain_data.get("market_sentiment", "")
        ai_themes = ai_brain_data.get("key_themes", [])
        print(f"[AI 두뇌] 센티먼트: {ai_sentiment} | 판단종목: {len(ai_brain_judgments)}개")
        if ai_themes:
            print(f"  테마: {' | '.join(ai_themes[:3])}")

    # 전체 종목 티커 수집
    all_tickers = set()
    for src in [src1, src2, src3, src4, src5, src6, src7, src9, src10, src11, src12, src13, src14]:
        all_tickers.update(src.keys())

    # AVOID 종목 제외
    all_tickers -= avoid_tickers

    # 국적수급 킬 필터: 기관 전면 이탈(SELL, score ≤ -10) 종목 제외
    if nat_kill_tickers:
        before_nat = len(all_tickers)
        all_tickers -= nat_kill_tickers
        killed_names = [nat_signals[t].get("name", t) for t in nat_kill_tickers if t in nat_signals]
        if killed_names:
            print(f"[국적수급 킬] {before_nat} → {len(all_tickers)}종목 | "
                  f"제외: {', '.join(killed_names[:5])}")

    # 컨센서스 풀 필터 (pool_only 모드)
    if consensus_pool_enabled and consensus_pool_only and consensus_pool:
        before_cnt = len(all_tickers)
        all_tickers &= set(consensus_pool.keys())
        print(f"[컨센서스 풀 필터] {before_cnt} → {len(all_tickers)}종목 (풀 내 종목만)")

    # ── 채널 1: AI 독립 추천 슬롯 (12개 소스에 없어도 AI BUY면 추천 가능) ──
    ai_ind_cfg = yaml_config.get("ai_brain", {}).get("independent_pick", {})
    ai_ind_enabled = ai_ind_cfg.get("enabled", False)
    ai_ind_max = ai_ind_cfg.get("max_count", 5)
    ai_ind_min_conf = ai_ind_cfg.get("min_confidence", 0.70)

    ai_independent_tickers = set()
    if ai_ind_enabled and ai_brain_judgments:
        for t, j in ai_brain_judgments.items():
            if j.get("action") != "BUY" or j.get("confidence", 0) < ai_ind_min_conf:
                continue
            if t in all_tickers or t in avoid_tickers:
                continue
            pq_path = PROCESSED_DIR / f"{t}.parquet"
            if not pq_path.exists():
                continue
            ai_independent_tickers.add(t)
            if len(ai_independent_tickers) >= ai_ind_max:
                break
        if ai_independent_tickers:
            all_tickers.update(ai_independent_tickers)
            print(f"[AI 독립추천] {len(ai_independent_tickers)}종목 추가: "
                  f"{', '.join(ai_independent_tickers)}")

    print(f"[통합] 고유 종목: {len(all_tickers)}개")

    # 안전마진 컨센서스 풀 사전 로드
    _safety_pool = {}
    try:
        from src.safety_margin import _load_consensus_pool
        _safety_pool = _load_consensus_pool()
        print(f"[안전마진] 컨센서스 풀 {len(_safety_pool)}종목 로드")
    except Exception as e:
        logger.warning("[안전마진] 풀 로드 실패: %s", e)

    # 공매도 프로파일 사전 추출 (루프 내 참조용)
    short_prof = short_summary.get("profile", {})
    short_pos_scale = short_prof.get("position_scale_mult", 1.0)
    short_slot_scale = short_prof.get("max_positions_scale", 1.0)

    # 종목별 통합
    results = []
    for ticker in all_tickers:
        sources = []
        source_names = []
        for src, label in [(src1, "릴레이"), (src2, "그룹순환"), (src3, "눌림목"),
                           (src4, "퀀텀"), (src5, "동반매수"), (src6, "세력감지"),
                           (src7, "이벤트"), (src9, "수급폭발"), (src10, "매집추적"),
                           (src11, "이벤트촉매"), (src12, "밸류체인"),
                           (src13, "기술촉매"), (src14, "이벤트섹터")]:
            if ticker in src:
                sources.append(src[ticker])
                source_names.append(label)

        # AI 독립추천 소스 (12개 소스에 없던 종목)
        if ai_ind_enabled and ticker in ai_independent_tickers:
            j = ai_brain_judgments[ticker]
            sources.append({
                "source": "AI두뇌", "name": j.get("name", ""),
                "detail": f"AI독립:conf={j.get('confidence', 0):.0%}",
                "score": j.get("confidence", 0) * 100,
            })
            source_names.append("AI두뇌독립")

        # ── 눌림목 단독 추천 차단 (v12) ──
        # 눌림목이 유일한 소스면 교차검증 없음 → 승률 28% 문제
        # 최소 2개 소스 필수. 눌림목 단독이면 스킵
        if source_names == ["눌림목"]:
            continue

        # FLOWX 모드: 매집추적 소스 부스트 (적중률 1위 소스 강화)
        if is_flowx_mode and "매집추적" in source_names:
            acc_boost = FLOWX_MODE_OVERRIDES["accumulation_boost"]
            for s in sources:
                if s.get("source") == "매집추적":
                    s["score"] = s["score"] * acc_boost

        # parquet 기술적 데이터
        pq_data = get_parquet_data(ticker)

        # 전략 그룹 판별 + 유효 소스 수 (겹침 보정)
        strategy = classify_strategy_group(source_names)
        if strategy == "short":
            grp_src_cnt = calc_effective_source_count(source_names, "short")
        elif strategy == "swing":
            grp_src_cnt = calc_effective_source_count(source_names, "swing")
        else:
            # both: 더 많은 쪽 기준
            sw = calc_effective_source_count(source_names, "swing")
            sh = calc_effective_source_count(source_names, "short")
            grp_src_cnt = max(sw, sh)

        # 통합 점수 계산 (그룹별 유효 소스 수 반영)
        score_detail = calc_integrated_score(ticker, sources, pq_data, grp_src_cnt)

        # 레짐 부스트 적용 (v7): 교차검증 시그널 바이패스
        # - 2개 이상 독립 소스 교차검증 + 거래량 2x → 레짐 감쇠 면제
        # - BEAR 모드 = "진입 금지"가 아니라 "조건 강화"
        effective_regime = regime_mult
        if is_flowx_mode and effective_regime < FLOWX_MODE_OVERRIDES["regime_boost_floor"]:
            effective_regime = FLOWX_MODE_OVERRIDES["regime_boost_floor"]
        if short_summary.get("enabled") and short_pos_scale < 1.0:
            effective_regime *= short_pos_scale

        # v7 교차검증 바이패스: 2소스 이상 + 거래량 2배 이상 → 레짐 감쇠 면제
        vol_ratio = pq_data.get("vol_ratio", 0) if pq_data else 0
        cross_validated = len(source_names) >= 2 and vol_ratio >= 2.0
        if cross_validated and effective_regime < 1.0:
            effective_regime = 1.0  # 감쇠 면제 (부스트는 아님)

        if effective_regime != 1.0:
            boosted = min(score_detail["total"] * effective_regime, 100)
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

        # 전략 E: Perplexity 인텔리전스 보정 (이벤트 강도 비례 동적 부스트)
        # v11: ±5 고정 → 이벤트 강도(kr_impact_score) 비례 최대 +20/-15
        intel_bonus = 0.0
        intel_tag = ""

        # E-1: 수혜/피해 종목 직접 매칭 (종목명 기반) — 가장 강한 이벤트 점수 적용
        cur_name = ""
        for s in sources:
            if s.get("name"):
                cur_name = s["name"]
                break
        if not cur_name:
            cur_name = name_map.get(ticker, "")

        # key_events에서 이 종목과 관련된 최대 이벤트 강도 계산
        max_event_score = 0
        max_event_urgency = ""
        for evt in intel.get("key_events", []):
            evt_score = abs(evt.get("kr_impact_score", 0))
            if evt_score > abs(max_event_score):
                # 종목 직접 매칭 확인
                if cur_name in intel_beneficiary:
                    max_event_score = evt_score
                    max_event_urgency = evt.get("urgency", "")
                elif cur_name in intel_risk:
                    max_event_score = -evt_score
                    max_event_urgency = evt.get("urgency", "")

        if cur_name in intel_beneficiary:
            # 이벤트 강도 비례 부스트: score 4~5→+15, 2~3→+8, 1→+5
            abs_score = abs(max_event_score) if max_event_score != 0 else 3
            if abs_score >= 4:
                intel_bonus = 15.0
            elif abs_score >= 2:
                intel_bonus = 8.0
            else:
                intel_bonus = 5.0
            intel_tag = "수혜"
        elif cur_name in intel_risk:
            abs_score = abs(max_event_score) if max_event_score != 0 else 3
            if abs_score >= 4:
                intel_bonus = -12.0
            elif abs_score >= 2:
                intel_bonus = -6.0
            else:
                intel_bonus = -3.0
            intel_tag = "피해"

        # E-2: 섹터 부스트 (stock_to_sector → key_events 섹터 매칭)
        if not intel_tag:
            sts = _load_stock_to_sector()
            stock_sectors = sts.get(ticker, [])
            for evt in intel.get("key_events", []):
                evt_score = evt.get("kr_impact_score", 0)
                evt_dir = evt.get("kr_impact_direction", "")
                evt_urg = evt.get("urgency", "")
                for sec in evt.get("kr_sectors_affected", []):
                    if sec in stock_sectors:
                        # 강도 비례 부스트
                        abs_s = abs(evt_score)
                        if evt_dir == "수혜" and evt_score > 0:
                            if abs_s >= 4:
                                bonus = 12.0
                            elif abs_s >= 2:
                                bonus = 6.0
                            else:
                                bonus = 3.0
                        elif evt_dir == "피해" and evt_score < 0:
                            if abs_s >= 4:
                                bonus = -10.0
                            elif abs_s >= 2:
                                bonus = -5.0
                            else:
                                bonus = -2.0
                        else:
                            continue
                        if abs(bonus) > abs(intel_bonus):
                            intel_bonus = bonus
                            intel_tag = f"{sec}{'수혜' if bonus > 0 else '피해'}"
                        break  # 첫 섹터 매칭만

        # E-2b: sector_boost fallback (key_events에 안 걸리면)
        if not intel_tag and intel_sector_boost:
            sts = _load_stock_to_sector()
            stock_sectors = sts.get(ticker, [])
            for sec in stock_sectors:
                if sec in intel_sector_boost:
                    sb = intel_sector_boost[sec]
                    intel_bonus = min(max(sb * 2, -8), 8)
                    if not intel_tag:
                        intel_tag = f"{sec}{'수혜' if sb > 0 else '피해'}"
                    break

        # BREAKING urgency → ×1.3 배율
        if max_event_urgency == "BREAKING" and intel_bonus > 0:
            intel_bonus = round(intel_bonus * 1.3, 1)

        intel_bonus = round(max(min(intel_bonus, 20), -15), 1)
        if intel_bonus != 0:
            boosted = max(min(score_detail["total"] + intel_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            if intel_bonus > 0:
                source_names.append("인텔리전스")

        # 전략 G: 리포트 + 뉴스 부스트 (최대 ±10점, intel과 독립)
        report_bonus = 0.0
        report_tag = ""
        # G-1: 증권사 리포트 (티커 기반)
        if ticker in report_boost_map:
            rb = report_boost_map[ticker]
            report_bonus += rb.get("boost", 0)
            report_tag = f"리포트:{rb.get('tag', '')}"
        # G-2: Perplexity 개별 촉매 (종목명 기반)
        cur_name = name_map.get(ticker, "")
        if pplx_themes:
            for cat in pplx_themes.get("breaking_catalysts", []):
                if cat.get("stock_name") == cur_name and cat.get("impact") == "positive":
                    report_bonus += 3.0
                    if not report_tag:
                        report_tag = f"촉매:{cat.get('catalyst', '')[:15]}"
                    break
        # G-3: 뉴스 부스트 (종목명 기반)
        if cur_name in news_boost_map:
            nb = news_boost_map[cur_name]
            report_bonus += nb.get("boost", 0)
            if not report_tag:
                report_tag = f"뉴스:{nb.get('reason', '')[:15]}"
        report_bonus = round(max(min(report_bonus, 10), -10), 1)
        if report_bonus != 0:
            boosted = max(min(score_detail["total"] + report_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            if report_bonus > 0:
                source_names.append("리포트")

        # 전략 H: AI 두뇌 보너스 (참고 수준, settings.yaml bonus_max 적용)
        ai_bonus_max = yaml_config.get("ai_brain", {}).get("bonus_max", 5)
        ai_bonus = 0.0
        ai_tag = ""
        ai_action = ""
        ai_urgency = ""
        if ticker in ai_brain_judgments:
            j = ai_brain_judgments[ticker]
            ai_action = j.get("action", "")
            confidence = j.get("confidence", 0)
            ai_urgency = j.get("urgency", "")
            if ai_action == "BUY":
                # confidence 0.70→+2, 0.85→+4, 0.95→+5 (참고 수준)
                ai_bonus = round(confidence * ai_bonus_max - 1, 1)
                ai_tag = f"AI:BUY({confidence:.0%})"
            elif ai_action == "WATCH":
                ai_bonus = round(confidence * 1.5 - 0.5, 1)
                ai_tag = "AI:WATCH"
            elif ai_action == "AVOID":
                ai_bonus = round(-confidence * ai_bonus_max + 1, 1)
                ai_tag = f"AI:AVOID({confidence:.0%})"
        ai_bonus = round(max(min(ai_bonus, ai_bonus_max), -ai_bonus_max), 1)
        if ai_bonus != 0:
            boosted = max(min(score_detail["total"] + ai_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            if ai_bonus > 0:
                source_names.append("AI두뇌")

        # 전략 H-2: AI 섹터 전망 보정 (positive +10, negative -15)
        ai_sector_bonus = 0.0
        ai_sector_tag = ""
        ai_sel_cfg_h2 = yaml_config.get("ai_brain", {}).get("stock_selection", {})
        if ai_sel_cfg_h2.get("sector_boost", False) and ai_brain_data and ai_brain_data.get("sector_outlook"):
            sts = _load_stock_to_sector()
            stock_sectors = sts.get(ticker, [])
            for sec in stock_sectors:
                matched_outlook = _match_ai_sector(sec, ai_brain_data["sector_outlook"])
                if matched_outlook:
                    direction = matched_outlook.get("direction", "")
                    if direction == "positive":
                        ai_sector_bonus = max(ai_sector_bonus, 10.0)
                        ai_sector_tag = f"섹터↑{sec}"
                    elif direction == "negative":
                        ai_sector_bonus = min(ai_sector_bonus, -15.0)
                        ai_sector_tag = f"섹터↓{sec}"
                    break  # 첫 매칭만
        if ai_sector_bonus != 0:
            boosted = max(min(score_detail["total"] + ai_sector_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            if ai_sector_bonus > 0:
                source_names.append("AI섹터")

        # 전략 I: v3 AI Brain conviction 부스트 (최대 +15점)
        v3_bonus = 0.0
        v3_tag = ""
        if ticker in v3_buy_map:
            v3 = v3_buy_map[ticker]
            conv = v3.get("conviction", 0)
            # conviction 5→+5, 7→+10, 9→+14, 10→+15
            v3_bonus = round(min(conv * 1.67 - 3.35, 15), 1)
            v3_tag = f"v3:{v3.get('strategy', '?')}(c{conv})"
        v3_bonus = round(max(min(v3_bonus, 15), 0), 1)
        if v3_bonus > 0:
            boosted = max(min(score_detail["total"] + v3_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            source_names.append("v3Brain")

        # 전략 J: 컨센서스 풀 보너스 (최대 +10점)
        consensus_bonus = 0.0
        consensus_tag = ""
        consensus_upside = 0.0
        consensus_score = 0.0
        consensus_fper = 0.0
        if consensus_pool_enabled and ticker in consensus_pool:
            cp = consensus_pool[ticker]
            consensus_upside = cp.get("upside_pct", 0)
            consensus_score = cp.get("composite_score", 0)
            consensus_fper = cp.get("forward_per", 0) or 0
            cp_grade = cp.get("grade", "D")
            # 등급별 보너스: S→+10, A→+8, B→+5, C→+2
            grade_bonus = {"S": 10, "A": 8, "B": 5, "C": 2}.get(cp_grade, 0)
            consensus_bonus = float(grade_bonus)
            if consensus_bonus > 0:
                consensus_tag = f"컨센서스:{cp_grade}({consensus_score:.0f}점,↑{consensus_upside:.0f}%)"
        if consensus_bonus > 0:
            boosted = max(min(score_detail["total"] + consensus_bonus, 100), 0)
            score_detail["total"] = round(boosted, 1)
            source_names.append("컨센서스")

        # ── 전략K: 네 마녀의 날 감점 ──
        witching_penalty = 0
        witching_tag = ""
        try:
            from src.use_cases.market_calendar import check_witching_proximity
            witching = check_witching_proximity()
            if witching["warning_level"] in ("CRITICAL", "HIGH", "MODERATE"):
                penalty_map = {"CRITICAL": -10, "HIGH": -5, "MODERATE": -3}
                witching_penalty = penalty_map[witching["warning_level"]]
                boosted = max(score_detail["total"] + witching_penalty, 0)
                score_detail["total"] = round(boosted, 1)
                label = {"CRITICAL": "만기당일", "HIGH": "만기D-1", "MODERATE": "만기주간"}
                witching_tag = f"⚠️{label[witching['warning_level']]}({witching_penalty})"
        except Exception:
            pass

        # ── 전략L: 국적별 수급 7 Secrets (부스트/감점 + 소스) ──
        nat_bonus = 0.0
        nat_tag = ""
        nat_signal_str = ""
        nat_score_val = 0.0
        nat_pattern = ""
        if nat_signals and ticker in nat_signals:
            ns = nat_signals[ticker]
            nat_signal_str = ns.get("signal", "NEUTRAL")
            nat_score_val = ns.get("score", 0)
            nat_pattern = ns.get("pattern", "")
            retail_pat = ns.get("retail_pattern", "")

            # 부스트/감점 매핑
            bonus_map = {
                "STRONG_BUY": 8,   # 기관 매집 + 가속도 → 최고 가산
                "BUY": 5,          # 기관 확인 매수
                "NEUTRAL": 0,
                "CAUTION": -3,     # 이탈 징후
                "SELL": -5,        # 전면 이탈 (킬에서 걸러졌을 수도 있지만 잔여 감점)
            }
            nat_bonus = float(bonus_map.get(nat_signal_str, 0))

            # 패턴 보너스: 조용한 매집(+3), 흡수 매집(+2)
            if nat_pattern == "QUIET_ACCUM":
                nat_bonus += 3
            elif nat_pattern == "ABSORPTION":
                nat_bonus += 2

            # 개인 역류 보정: 지지(+1), 추격(-2)
            if retail_pat == "RETAIL_SUPPORT":
                nat_bonus += 1
            elif retail_pat == "RETAIL_FOMO":
                nat_bonus -= 2

            nat_bonus = round(nat_bonus, 1)

            if nat_bonus != 0:
                boosted = max(min(score_detail["total"] + nat_bonus, 100), 0)
                score_detail["total"] = round(boosted, 1)

            # 태그 생성
            pat_str = f",{nat_pattern}" if nat_pattern and nat_pattern != "MIXED" else ""
            retail_str = f",{retail_pat}" if retail_pat else ""
            nat_tag = f"국적:{nat_signal_str}({nat_score_val:+.0f}{pat_str}{retail_str})"

            # BUY/STRONG_BUY면 소스로 추가
            if nat_signal_str in ("STRONG_BUY", "BUY"):
                source_names.append("국적수급")

        # ── 전략N: 기관매집 조기감지 부스트 ──
        # EARLY_SURGE(오늘 3배 대량) → +10, EARLY_ACCEL(가속) → +7
        # EARLY_DUAL(기관+외인 동시) → +5, STRONG(장기 쌍끌이) → +8
        # 기관 연속매수일이 길수록 추가 보너스
        inst_accum_tag = ""
        if ticker in inst_accum:
            ia = inst_accum[ticker]
            ia_grade = ia.get("grade", "")
            inst_bonus_map = {
                "EARLY_SURGE": 10,  # 오늘 대량 진입 (평균 3배+) — 가장 긴급
                "STRONG": 8,        # 장기 쌍끌이 확인 — 확신도 높음
                "EARLY_ACCEL": 7,   # 가속 매수 (2일+) — 추세 확인
                "EARLY_DUAL": 5,    # 기관+외인 동시 진입 — 초기 신호
                "MODERATE": 4,      # 쌍끌이 (단기)
                "NOTABLE": 3,       # 한쪽 강함
                "WATCH": 1,         # 관심
            }
            ia_bonus = float(inst_bonus_map.get(ia_grade, 0))

            # 연속매수일 보너스: 기관 3일+이면 +2, 5일+이면 +3
            inst_consec = ia.get("inst_consecutive", 0)
            frgn_consec = ia.get("foreign_consecutive", 0)
            max_consec = max(inst_consec, frgn_consec)
            if max_consec >= 5:
                ia_bonus += 3
            elif max_consec >= 3:
                ia_bonus += 2

            # 쌍끌이(기관+외인) 동시 누적 보너스
            if ia.get("dual_buying"):
                ia_bonus += 2

            if ia_bonus > 0:
                boosted = max(min(score_detail["total"] + ia_bonus, 100), 0)
                score_detail["total"] = round(boosted, 1)
                inst_accum_tag = f"기관:{ia_grade}(+{ia_bonus:.0f})"
                if ia_grade.startswith("EARLY") or ia_grade == "STRONG":
                    source_names.append("기관매집")

        # ── 전략O: 외인소진율 부스트 ──
        # FE2(급등): 5일간 소진율 +5%p 이상 급등 → +6점 (집중 매수 중)
        # FE1(한도임박): 90%+ → +3점 (주의 but 인기 확인)
        fe_tag = ""
        if ticker in fe_signals:
            fe = fe_signals[ticker]
            fe_type = fe.get("fe_type", "")
            fe_score = fe.get("score", 0)
            if fe_type == "FE2":
                fe_bonus = 6  # 소진율 급등 = 외인 집중 매수 중
                fe_tag = f"외인급등(FE2,{fe_score}점)"
            elif fe_type == "FE1":
                fe_bonus = 3  # 한도 임박 = 인기는 확인
                fe_tag = f"외인한도(FE1,{fe_score}점)"
            else:
                fe_bonus = 0
            if fe_bonus > 0:
                boosted = max(min(score_detail["total"] + fe_bonus, 100), 0)
                score_detail["total"] = round(boosted, 1)
                source_names.append("외인소진")

        # ── 전략M: 시나리오 뉴스 부스트 (원자재 원가 갭 + 시나리오 체인) ──
        scenario_bonus = 0.0
        scenario_tag = ""
        scenario_narrative = ""
        scenario_risk_reward: dict = {}
        if ticker in scenario_ticker_map:
            matched_scenarios = scenario_ticker_map[ticker]
            best_sc = max(matched_scenarios, key=lambda x: x["scenario_score"])

            # 시나리오 점수 기반 부스트 (최대 +10점)
            # score 30→+3, 50→+5, 70→+7, 100→+10
            scenario_bonus = round(min(best_sc["scenario_score"] * 0.1, 10), 1)

            # narrative 생성
            scenario_tag, scenario_narrative, scenario_risk_reward = \
                build_scenario_narrative(matched_scenarios, scenario_data["commodities"])

            if scenario_bonus > 0:
                boosted = max(min(score_detail["total"] + scenario_bonus, 100), 0)
                score_detail["total"] = round(boosted, 1)
                source_names.append("시나리오")

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
            foreign_5d=score_detail.get("foreign_5d", 0),
            inst_5d=score_detail.get("inst_5d", 0),
            foreign_reversal=pq_data.get("foreign_reversal", False) if pq_data else False,
            inst_reversal=pq_data.get("inst_reversal", False) if pq_data else False,
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
            "report_bonus": report_bonus,
            "report_tag": report_tag,
            "ai_bonus": ai_bonus,
            "ai_tag": ai_tag,
            "ai_action": ai_action,
            "ai_urgency": ai_urgency,
            "ai_sector_bonus": ai_sector_bonus,
            "ai_sector_tag": ai_sector_tag,
            "consensus_bonus": consensus_bonus,
            "consensus_tag": consensus_tag,
            "consensus_upside": consensus_upside,
            "consensus_score": consensus_score,
            "consensus_fper": consensus_fper,
            "witching_penalty": witching_penalty,
            "witching_tag": witching_tag,
            "nat_bonus": nat_bonus,
            "nat_tag": nat_tag,
            "nat_signal": nat_signal_str,
            "nat_score": nat_score_val,
            "nat_pattern": nat_pattern,
            "inst_accum_tag": inst_accum_tag,
            "fe_tag": fe_tag,
            "scenario_bonus": scenario_bonus,
            "scenario_tag": scenario_tag,
            "scenario_narrative": scenario_narrative,
            "scenario_risk_reward": scenario_risk_reward,
            "ma5_gap_pct": pq_data.get("ma5_gap_pct", 0) if pq_data else 0,
            "ma7_gap_pct": pq_data.get("ma7_gap_pct", 0) if pq_data else 0,
            "ma5_entry": entry_info.get("ma5_entry", ""),
            "strategy": strategy,
            "group_source_count": grp_src_cnt,
            "sar_trend": pq_data.get("sar_trend", 0) if pq_data else 0,
            "fibonacci": compute_stock_fibonacci(ticker),
        }

        # ── Alpha Scoring v3 태그 + STRONG_ALPHA 점수 반영 ──
        # v3 시그널 중 검증된 STRONG_ALPHA → 점수 부스트 + 등급 재분류
        # 실제 alpha_scoring.py Tier1 시그널명 사용
        _STRONG_ALPHA_SIGS = {"PULLBACK15_VOL3x", "PULLBACK15_DUAL", "BREAKOUT60_VOL3x_DUAL", "MEGA_VOL_8x"}
        _MODERATE_ALPHA_SIGS = {"PULLBACK10_SUPPLY", "PULLBACK7_SUPPLY"}  # PF~1.5
        _ALPHA_SCORE_BONUS = 15  # STRONG 1개당 +15점
        _MODERATE_SCORE_BONUS = 8  # MODERATE 1개당 +8점
        if alpha_v3_display and _v3_score is not None:
            raw_df = get_parquet_raw_df(ticker)
            if raw_df is not None:
                ia_v3 = inst_accum.get(ticker, {})
                fe_v3 = fe_signals.get(ticker, {})
                v3_result = _v3_score(
                    df=raw_df,
                    ticker=ticker,
                    name=name,
                    regime=regime_name,
                    inst_accum_days=ia_v3.get("inst_consecutive", ia_v3.get("days", 0)),
                    inst_accum_grade=ia_v3.get("grade", ""),
                    fe_grade=fe_v3.get("fe_type", ""),
                )
                fired = [s.name for s in v3_result.signals if s.fired]
                if fired:
                    rec["alpha_v3_tag"] = True
                    rec["alpha_signals"] = fired
                    rec["alpha_tier1"] = v3_result.tier1_count
                    rec["alpha_tier2"] = v3_result.tier2_count
                    rec["alpha_v3_score"] = v3_result.total_score

                    # STRONG/MODERATE ALPHA → 점수 부스트 + 등급 재분류
                    strong_match = set(fired) & _STRONG_ALPHA_SIGS
                    moderate_match = set(fired) & _MODERATE_ALPHA_SIGS
                    alpha_bonus = (_ALPHA_SCORE_BONUS * len(strong_match)
                                   + _MODERATE_SCORE_BONUS * len(moderate_match))
                    if alpha_bonus > 0:
                        old_score = rec["total_score"]
                        new_score = round(min(old_score + alpha_bonus, 100), 1)
                        rec["total_score"] = new_score
                        rec["alpha_bonus"] = alpha_bonus
                        score_detail["total"] = new_score
                        # 등급 재분류 (부스트된 점수로)
                        new_grade = classify_pick(
                            new_score, len(sources), score_detail["rsi"],
                            has_data=has_data,
                            stoch_k=score_detail.get("stoch_k", 50),
                            ret_5d=score_detail.get("ret_5d", 0),
                            foreign_5d=score_detail.get("foreign_5d", 0),
                            inst_5d=score_detail.get("inst_5d", 0),
                            foreign_reversal=pq_data.get("foreign_reversal", False) if pq_data else False,
                            inst_reversal=pq_data.get("inst_reversal", False) if pq_data else False,
                        )
                        matched = list(strong_match | moderate_match)
                        if new_grade != rec["grade"]:
                            logger.info("[ALPHA] %s: %s→%s (%.1f→%.1f +%d) [%s]",
                                        name, rec["grade"], new_grade,
                                        old_score, new_score, alpha_bonus,
                                        ",".join(matched))
                            rec["grade"] = new_grade

        # 안전마진 플래그
        try:
            from src.safety_margin import calc_safety_margin
            sm = calc_safety_margin(
                ticker, name, score_detail["close"],
                consensus=_safety_pool.get(ticker),
            )
            rec["safety_signal"] = sm.signal
            rec["safety_label"] = sm.signal_label
            rec["safety_floor"] = sm.floor_price
            rec["safety_floor_pct"] = sm.floor_margin_pct
            rec["safety_reason"] = sm.reason
        except Exception:
            rec["safety_signal"] = "NO_DATA"
            rec["safety_label"] = "데이터없음"
            rec["safety_floor"] = 0
            rec["safety_floor_pct"] = 0.0
            rec["safety_reason"] = ""

        results.append(rec)

    # 정렬: 등급 → 점수
    grade_order = {"적극매수": 0, "매수": 1, "관심매수": 2, "관찰": 3, "보류": 4, "데이터부족": 5}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["total_score"]))

    # ── TOP 선별: 전략 그룹별 슬롯 분리 ──
    buyable_grades = {"적극매수", "매수", "관심매수"}
    buyable = [r for r in results if r["grade"] in buyable_grades]

    # AI 거부권: AVOID 종목은 최종 추천에서 제외
    ai_veto_cfg = yaml_config.get("ai_brain", {}).get("stock_selection", {})
    veto_avoid_active = ai_veto_cfg.get("veto_avoid", False)
    if is_flowx_mode:
        veto_avoid_active = FLOWX_MODE_OVERRIDES["veto_avoid"]
    elif is_war_mode:
        veto_avoid_active = WAR_MODE_OVERRIDES["veto_avoid"]
    if veto_avoid_active:
        ai_vetoed = [r for r in buyable if r.get("ai_action") == "AVOID"]
        if ai_vetoed:
            buyable = [r for r in buyable if r.get("ai_action") != "AVOID"]
            print(f"[AI 거부권] {len(ai_vetoed)}종목 제외: "
                  f"{', '.join(r['name'] for r in ai_vetoed)}")

    # 고가주 추천 제외 (v12): 1주 80만원 초과 종목은 시드 제약으로 추천에서만 제외
    MAX_PRICE_FOR_PICK = 800_000

    # 동일 이벤트촉매 섹터 2개 제한 (v12): UAE 방산 3개 등 편중 방지
    _ec = load_json("event_catalyst.json") or {}
    _ec_sector_map = {s["ticker"]: s.get("sector", "") for s in _ec.get("stocks", [])}

    # AI 센티먼트 → 추천 수량 동적 조절 (최소 10개 보장)
    ai_sel_cfg = yaml_config.get("ai_brain", {}).get("stock_selection", {})
    base_swing = STRATEGY_GROUPS["swing"]["slots"]  # 5
    base_short = STRATEGY_GROUPS["short"]["slots"]  # 5
    dynamic_slots_active = ai_sel_cfg.get("dynamic_slots", False)
    if is_flowx_mode:
        dynamic_slots_active = FLOWX_MODE_OVERRIDES["dynamic_slots"]
    elif is_war_mode:
        dynamic_slots_active = WAR_MODE_OVERRIDES["dynamic_slots"]
    if dynamic_slots_active and ai_brain_data:
        sentiment = ai_brain_data.get("market_sentiment", "neutral")
        # bullish: +2/+2 (총14), neutral: 기본(총10), bearish: -1/-1 (총8→10보장)
        slot_adj = {"bullish": 2, "neutral": 0, "bearish": -1}.get(sentiment, 0)
        swing_slots = max(base_swing + slot_adj, 3)
        short_slots = max(base_short + slot_adj, 3)
        # 최소 총 10개 보장
        if swing_slots + short_slots < 10:
            swing_slots = 5
            short_slots = 5
        print(f"[AI 슬롯] {sentiment} → 스윙 {swing_slots} + 단타 {short_slots} = {swing_slots + short_slots}개")
    else:
        swing_slots = base_swing
        short_slots = base_short

    # 공매도 체제 슬롯 축소 (reopened/active → max_positions_scale 적용)
    if short_summary.get("enabled") and short_slot_scale < 1.0:
        orig_swing, orig_short = swing_slots, short_slots
        swing_slots = max(2, int(swing_slots * short_slot_scale))
        short_slots = max(2, int(short_slots * short_slot_scale))
        print(f"[공매도 슬롯] ×{short_slot_scale} → "
              f"스윙 {orig_swing}→{swing_slots} + 단타 {orig_short}→{short_slots}")

    # 최소 소스 수 필터 (복합 조건: 소스 N개+ OR 소스 (N-1)개 + 고점수)
    min_sources = ai_sel_cfg.get("min_sources", 0)
    alt_score = ai_sel_cfg.get("min_sources_alt_score", 70)
    if is_flowx_mode:
        min_sources = FLOWX_MODE_OVERRIDES["min_sources"]
        alt_score = FLOWX_MODE_OVERRIDES["min_sources_alt_score"]
    elif is_war_mode:
        min_sources = WAR_MODE_OVERRIDES["min_sources"]
        alt_score = WAR_MODE_OVERRIDES["min_sources_alt_score"]
    if min_sources >= 2:
        before_cnt = len(buyable)
        buyable = [
            r for r in buyable
            if r.get("n_sources", 0) >= min_sources
            or (r.get("n_sources", 0) >= min_sources - 1
                and r.get("total_score", 0) >= alt_score)
        ]
        filtered_cnt = before_cnt - len(buyable)
        if filtered_cnt:
            print(f"[소스 필터] {min_sources}소스+ OR ({min_sources-1}소스+{alt_score}점+) → "
                  f"{filtered_cnt}종목 제외, 잔여 {len(buyable)}종목")

    # 그룹별 풀 분리 (both는 양쪽 모두에 포함)
    swing_pool = [r for r in buyable if r.get("strategy") in ("swing", "both")]
    short_pool = [r for r in buyable if r.get("strategy") in ("short", "both")]

    def _select_top_n(pool, n, used_tickers, ec_sector_cnt):
        """풀에서 상위 n개 선별 (고가주/이벤트섹터 제한 적용)."""
        selected = []
        for r in pool:
            if len(selected) >= n:
                break
            if r["ticker"] in used_tickers:
                continue
            if r.get("close", 0) > MAX_PRICE_FOR_PICK:
                continue
            ec_sector = _ec_sector_map.get(r["ticker"], "")
            if ec_sector:
                cnt = ec_sector_cnt.get(ec_sector, 0)
                if cnt >= 2:
                    continue
                ec_sector_cnt[ec_sector] = cnt + 1
            selected.append(r)
            used_tickers.add(r["ticker"])
        return selected

    used = set()
    ec_cnt: dict[str, int] = {}

    top5_swing = _select_top_n(swing_pool, swing_slots, used, ec_cnt)
    top5_short = _select_top_n(short_pool, short_slots, used, ec_cnt)

    # 부족 시 상호 보충
    if len(top5_swing) < swing_slots:
        need = swing_slots - len(top5_swing)
        extras = _select_top_n(short_pool, need, used, ec_cnt)
        top5_swing.extend(extras)
    if len(top5_short) < short_slots:
        need = short_slots - len(top5_short)
        extras = _select_top_n(swing_pool, need, used, ec_cnt)
        top5_short.extend(extras)

    top5 = top5_swing + top5_short

    # FLOWX 모드: TOP 최소 보장 — buyable이 부족하면 관찰 등급까지 승격
    if is_flowx_mode and len(top5) < FLOWX_MODE_OVERRIDES["min_top_count"]:
        need = FLOWX_MODE_OVERRIDES["min_top_count"] - len(top5)
        top5_tickers_tmp = {r["ticker"] for r in top5}
        # 관찰 등급까지 포함하여 점수순 승격
        fallback_pool = [r for r in results
                         if r["grade"] in ("관찰", "보류")
                         and r["ticker"] not in top5_tickers_tmp
                         and r.get("close", 0) <= MAX_PRICE_FOR_PICK]
        fallback_pool.sort(key=lambda x: -x["total_score"])
        promoted = fallback_pool[:need]
        for r in promoted:
            r["flowx_promoted"] = True
            r["original_grade"] = r["grade"]
            r["grade"] = "관심매수"
        top5.extend(promoted)
        if promoted:
            print(f"[FLOWX] TOP 부족 → {len(promoted)}종목 승격 (관찰→관심매수)")

    for r in top5:
        r["is_top5"] = True

    # ── 관심종목 5개 (v12): TOP5 제외, 소스 3개 이상 우선 ──
    top5_tickers = {r["ticker"] for r in top5}
    watchlist_pool = [r for r in buyable
                      if r["ticker"] not in top5_tickers
                      and r.get("close", 0) <= MAX_PRICE_FOR_PICK]
    # 소스 3개+ 우선, 그 다음 점수순
    watchlist_pool.sort(key=lambda x: (-min(x["n_sources"], 3), -x["total_score"]))
    watchlist5 = watchlist_pool[:5]
    for r in watchlist5:
        r["is_watchlist"] = True

    # 통계
    grade_stats = {}
    for r in results:
        g = r["grade"]
        grade_stats[g] = grade_stats.get(g, 0) + 1

    print(f"\n{'='*60}")
    print(f"{mode_label} [내일 추천] 총 {len(results)}건 (TOP10: {len(top5)}건)")
    for g in ["적극매수", "매수", "관심매수", "관찰", "보류", "데이터부족"]:
        cnt = grade_stats.get(g, 0)
        if cnt:
            print(f"  {g}: {cnt}건")
    print(f"{'='*60}")

    # TOP 5 출력 — 전략 그룹별 구분
    def _print_pick(idx, r):
        """종목 1건 상세 출력."""
        srcs = "+".join(r["sources"])
        oh = f" 🔥-{r['score_breakdown']['overheat']}p" if r["score_breakdown"]["overheat"] > 0 else ""
        cond = r.get("entry_condition", "")
        reasons_str = ", ".join(r.get("reasons", [])[:3])
        zone_tag = f" [{r['target_zone']}]" if r.get("target_zone") else ""
        print(f"  {idx}. [{r['grade']}]{zone_tag} {r['name']}({r['ticker']}) "
              f"{r['total_score']}점{oh} ({r['n_sources']}개 소스: {srcs})")
        if r.get("estimated_target"):
            dir_icon = {"RISING": "▲", "FALLING": "▼", "STABLE": "─", "NEW": "★"}.get(r.get("target_direction", ""), "")
            print(f"     기관목표:{r['estimated_target']:,} (갭:{r.get('target_gap_pct',0):+.1f}%) {dir_icon} "
                  f"| 진입:{r.get('entry_price',0):,}  손절:{r.get('stop_loss',0):,}  "
                  f"목표:{r.get('target_price',0):,}")
        else:
            print(f"     진입:{r.get('entry_price',0):,}  손절:{r.get('stop_loss',0):,}  "
                  f"목표:{r.get('target_price',0):,} | {cond}")
        fib = r.get("fibonacci", {}).get("20d", {})
        fib_str = ""
        if fib:
            s = f"{fib['support']:,}" if fib.get("support") else "?"
            rs = f"{fib['resistance']:,}" if fib.get("resistance") else "?"
            fib_str = f"\n     📐 피보나치(20d): 지지 {s} | 저항 {rs} | 위치 {fib.get('position_pct', 0)}%"
        ma5g = r.get("ma5_gap_pct", 0)
        ma5e = r.get("ma5_entry", "")
        ma5_str = f"  📐 MA5 {ma5g:+.1f}% [{ma5e}]" if ma5e else ""
        intel_str = f"  🌐{r['intel_tag']}" if r.get("intel_tag") else ""
        report_str = f"  📋{r['report_tag']}" if r.get("report_tag") else ""
        cons_str = f"  📊{r['consensus_tag']}" if r.get("consensus_tag") else ""
        nat_str = f"  🌍{r['nat_tag']}" if r.get("nat_tag") else ""
        ia_str = f"  🏛{r['inst_accum_tag']}" if r.get("inst_accum_tag") else ""
        fe_str = f"  🔥{r['fe_tag']}" if r.get("fe_tag") else ""
        sc_str = ""
        if r.get("scenario_tag"):
            rr = r.get("scenario_risk_reward", {})
            rr_str = f" [{rr['commodity']} {rr['zone']}]" if rr.get("zone") else ""
            sc_str = f"\n     📰 {r['scenario_narrative']}{rr_str}"
        # v3 알파 시그널 보조 표시
        alpha_str = ""
        if r.get("alpha_v3_tag"):
            sigs = r.get("alpha_signals", [])
            t1 = r.get("alpha_tier1", 0)
            t2 = r.get("alpha_tier2", 0)
            v3s = r.get("alpha_v3_score", 0)
            alpha_str = f"\n     🧬 v3({v3s:.0f}): T1={t1} T2={t2} [{'+'.join(sigs)}]"
        print(f"     근거: {reasons_str}{ma5_str}{intel_str}{report_str}{cons_str}{nat_str}{ia_str}{fe_str}{sc_str}{alpha_str}{fib_str}")

    if top5:
        print(f"\n{'─'*60}")
        print(f"  ★ {mode_label} TOP 10 내일 매수 추천 (스윙 {len(top5_swing)} + 단타 {len(top5_short)}) ★")
        print(f"{'─'*60}")
        idx = 1
        if top5_swing:
            print(f"  [{STRATEGY_GROUPS['swing']['label']}]")
            for r in top5_swing:
                _print_pick(idx, r)
                idx += 1
        if top5_short:
            print(f"\n  [{STRATEGY_GROUPS['short']['label']}]")
            for r in top5_short:
                _print_pick(idx, r)
                idx += 1
        print(f"{'─'*60}")
    else:
        print("\n  ⚠ 매수 적합 종목 없음 — 전체 관망 추천")

    # 관심종목 5개 출력 (v12)
    if watchlist5:
        print(f"\n{'─'*60}")
        print(f"  👀 관심종목 5 (소스 다양성 우선)")
        print(f"{'─'*60}")
        for i, r in enumerate(watchlist5, 1):
            srcs = "+".join(r["sources"])
            print(f"  {i}. {r['name']}({r['ticker']}) "
                  f"{r['total_score']}점 [{r['grade']}] ({r['n_sources']}소스: {srcs})")
        print(f"{'─'*60}")

    # AI 대형주 참고 섹션 제거 — 시그널 기반 추천이 주체 (v6)
    ai_largecap = []  # 하위 호환용 빈 리스트

    # 나머지 관찰 종목 간략 출력
    rest = [r for r in results if r["grade"] in buyable_grades
            and r["ticker"] not in top5_tickers
            and r["ticker"] not in {w["ticker"] for w in watchlist5}]
    if rest:
        print(f"\n  [기타 관심종목]")
        for r in rest[:10]:
            print(f"    - {r['name']}({r['ticker']}) {r['total_score']}점 [{r['grade']}]")

    # ── 딥다이브 분석: TOP5 수급 + 급등패턴 + 이격 ──
    deep_dive_data = []
    dd_results = []
    try:
        from src.deep_dive import deep_dive_batch, format_deep_dive_telegram
        dd_targets = (top5_swing + top5_short)[:5]
        dd_results = deep_dive_batch(dd_targets, top_n=5)
        if dd_results:
            deep_dive_data = [r.to_dict() for r in dd_results]
            dd_msg = format_deep_dive_telegram(dd_results)
            print(f"\n{dd_msg}")
    except Exception as e:
        logger.warning("[딥다이브] 실행 실패: %s", e)
        print(f"\n  ⚠ 딥다이브 분석 실패: {e}")

    # ── 딥다이브 HTML 보고서 + 텔레그램 전송 ──
    dd_png_path = None
    if dd_results:
        try:
            from src.html_report import generate_deep_dive_report, send_report_to_telegram
            dd_html, dd_png_path = generate_deep_dive_report(dd_results)
            if dd_html.exists():
                print(f"  딥다이브 HTML: {dd_html}")
            if dd_png_path and dd_png_path.exists():
                print(f"  딥다이브 PNG: {dd_png_path}")
                if not no_send:
                    caption = f"[딥다이브] TOP{len(dd_results)} 수급+급등패턴+이격 분석"
                    img_ok = send_report_to_telegram(dd_png_path, caption)
                    print(f"  딥다이브 이미지 전송: {'OK' if img_ok else 'FAIL'}")
        except Exception as e:
            logger.warning("[딥다이브 HTML] 실패: %s", e)
            print(f"  ⚠ 딥다이브 HTML/전송 실패: {e}")

    # 딥다이브 텍스트도 텔레그램 전송
    if dd_results and deep_dive_data and not no_send:
        try:
            from src.telegram_sender import send_message
            dd_text_msg = format_deep_dive_telegram(dd_results)
            if dd_text_msg:
                send_message(dd_text_msg)
                print("  딥다이브 텍스트 전송: OK")
        except Exception as e:
            logger.warning("[딥다이브 텍스트 전송] 실패: %s", e)
            print(f"  ⚠ 딥다이브 텍스트 전송 실패: {e}")

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
        "mode": "flowx" if is_flowx_mode else ("war" if is_war_mode else "normal"),
        "mode_label": mode_label,
        "total_candidates": len(results),
        "stats": grade_stats,
        "top5": [r["ticker"] for r in top5],
        "top5_swing": [r["ticker"] for r in top5_swing],
        "top5_short": [r["ticker"] for r in top5_short],
        "watchlist5": [r["ticker"] for r in watchlist5],
        "ai_largecap": ai_largecap,
        "picks": results,
        "market_intel": {
            "mood": intel_mood,
            "forecast": intel.get("kr_open_forecast", ""),
            "forecast_reason": intel.get("kr_forecast_reason", ""),
            "hot_themes": intel_themes,
            "summary": intel.get("us_market_summary", ""),
        } if intel_mood else {},
        "morning_report": {
            "date": morning.get("date", ""),
            "report_count": len(report_boost_map),
            "themes": [t["theme"] for t in pplx_themes.get("hot_themes", [])][:5],
            "boosted_tickers": list(report_boost_map.keys()),
        } if morning else {},
        "consensus_pool": {
            "enabled": consensus_pool_enabled,
            "pool_only": consensus_pool_only,
            "pool_size": len(consensus_pool),
            "matched": sum(1 for r in results if r.get("consensus_bonus", 0) > 0),
        } if consensus_pool_enabled else {},
        "nationality_signal": {
            "enabled": bool(nat_signals),
            "total_stocks": len(nat_signals),
            "kill_count": len(nat_kill_tickers),
            "boosted": sum(1 for r in results if r.get("nat_bonus", 0) > 0),
            "penalized": sum(1 for r in results if r.get("nat_bonus", 0) < 0),
            "as_source": sum(1 for r in results if "국적수급" in r.get("sources", [])),
        } if nat_signals else {},
        "scenario_info": {
            "active_count": len(scenario_data["active"]),
            "matched_stocks": sum(1 for r in results if r.get("scenario_bonus", 0) > 0),
            "top_scenarios": [
                {"id": s_id, "score": s["score"], "phase": s["current_phase"]}
                for s_id, s in sorted(
                    scenario_data["active"].items(),
                    key=lambda x: -x[1].get("score", 0)
                )[:3]
            ],
        } if scenario_data.get("active") else {},
        "deep_dive": deep_dive_data,
        "short_selling_regime": {
            "enabled": short_summary.get("enabled", False),
            "status": short_summary.get("status", ""),
            "label": short_summary.get("label", ""),
            "slot_scale": short_slot_scale,
            "position_scale": short_pos_scale,
        } if short_summary.get("enabled") else {},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if is_flowx_mode:
        # FLOWX 모드: 별도 파일 저장 (기본 모드 결과를 덮어쓰지 않음)
        flowx_path = DATA_DIR / "tomorrow_picks_flowx.json"
        with open(flowx_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {flowx_path}")
    elif is_war_mode:
        # 공포탐욕 모드: 별도 파일 저장 (기본 모드 결과를 덮어쓰지 않음)
        war_path = DATA_DIR / "tomorrow_picks_war.json"
        with open(war_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {war_path}")
    else:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {OUTPUT_PATH}")
    print(f"[대상일] {output['target_date_label']} ({output['target_date']})")


if __name__ == "__main__":
    main()
