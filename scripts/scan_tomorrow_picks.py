"""
내일 추천 종목 통합 스캐너 — 5개 시그널 교차 검증

5개 시그널 소스를 통합하여 최종 매수 추천 종목을 산출합니다.

소스:
  1. 섹터릴레이 picks (relay_trading_signal.json)
  2. 그룹순환 waiting_subsidiaries (group_relay_today.json)
  3. 눌림목 반등임박/매수대기 (pullback_scan.json)
  4. 퀀텀시그널 survivors + killed (scan_cache.json)
  5. 동반매수 S/A등급 + core_watch (dual_buying_watch.json)

통합 점수 (100점, 5축 + 과열패널티):
  다중 시그널 (25): 2소스 +12, 3소스 +20, 4+ +25, 동반매수3일+ 부스트
  개별 점수  (20): 각 소스 점수 정규화 평균
  기술적 지지 (25): RSI 적정(8) + MA(5) + MACD(4) + TRIX(4) + Stoch(4)
  수급       (20): 외인(8) + 기관(5) + 동시매수(2) + 연속매수(5)
  안전       (10): BB(4) + ADX(3) + 낙폭(3)
  과열 패널티: RSI/Stoch/BB/급등 최대 -25점

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
    for item in pb.get("items", []):
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
        ("s_grade", "S등급", 85),
        ("a_grade", "A등급", 70),
        ("core_watch", "핵심관찰", 60),
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


# ──────────────────────────────────────────
# 통합 점수 계산
# ──────────────────────────────────────────

def calc_integrated_score(
    ticker: str,
    sources: list[dict],
    parquet_data: dict | None,
) -> dict:
    """5축 100점 + 과열패널티 통합 점수 계산 (v3)

    기본 100점 배분:
      다중시그널(25) + 개별점수(20) + 기술적(25) + 수급(20) + 안전(10)
    동반매수 부스트: 3일+ 연속 동반매수 시 멀티시그널 + 수급 가점
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

    # 동반매수 연속일 부스트: 3일+ 지속 매수는 그 자체가 확인 시그널
    dual_days = 0
    for s in sources:
        dd = s.get("dual_days", 0) or s.get("f_streak", 0) or 0
        dual_days = max(dual_days, int(dd))
    if dual_days >= 5:
        multi_score = max(multi_score, 15)  # 5일+ → 15점 보장
    elif dual_days >= 4:
        multi_score = max(multi_score, 12)  # 4일 → 12점 보장
    elif dual_days >= 3:
        multi_score = max(multi_score, 8)   # 3일 → 8점 보장

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
    # 연속 동반매수 보너스 (3일+ 지속 = 스마트머니 확인)
    if dual_days >= 4:
        flow_score += 5
    elif dual_days >= 3:
        flow_score += 3

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
    entry_info = _calc_entry_stop(close, ma20, ma60, low_20d, rsi, stoch_k, bb_pos)

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
) -> dict:
    """진입가/손절가/진입조건 자동 생성"""
    if close <= 0:
        return {"entry": 0, "stop": 0, "target": 0, "condition": "데이터 부족"}

    # 손절가: 20일 저점 또는 MA20*0.98 중 더 높은 값 (최대 -7%)
    stop_candidates = [v for v in [low_20d, ma20 * 0.98] if v > 0]
    stop = max(stop_candidates) if stop_candidates else close * 0.93
    stop = max(stop, close * 0.93)  # 손절폭 -7% 이내로 제한

    # 진입 조건 판단
    if stoch_k > 85 or bb_pos > 100:
        # 과열 → 조정 대기
        if stoch_k > 85:
            condition = f"Stoch {stoch_k:.0f}→70 이하 냉각 시"
            entry = round(close * 0.97, -1)  # -3% 수준
        else:
            condition = f"BB {bb_pos:.0f}%→85% 이하 복귀 시"
            entry = round(close * 0.96, -1)
    elif rsi > 70:
        condition = f"RSI {rsi:.0f}→65 이하 조정 시"
        entry = round(close * 0.97, -1)
    elif rsi < 35:
        condition = "RSI 과매도 반등 확인 후"
        entry = round(close * 1.01, -1)  # 반등 확인 후
    else:
        condition = "현재가 부근 매수 가능"
        entry = _safe_int(close)

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
        }
    except Exception as e:
        logger.warning("parquet 읽기 실패 %s: %s", ticker, e)
        return None


# ──────────────────────────────────────────
# 등급 분류
# ──────────────────────────────────────────

def classify_pick(total_score: float, n_sources: int, rsi: float) -> str:
    if total_score >= 70 and n_sources >= 2:
        return "강력매수"
    if total_score >= 55 and n_sources >= 2:
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

    # 5개 소스 수집
    src1 = collect_relay()
    src2 = collect_group_relay()
    src3 = collect_pullback()
    src4 = collect_quantum()
    src5 = collect_dual_buying()

    print(f"[소스 수집] 릴레이:{len(src1)} 그룹순환:{len(src2)} "
          f"눌림목:{len(src3)} 퀀텀:{len(src4)} 동반매수:{len(src5)}")

    # 전체 종목 티커 수집
    all_tickers = set()
    for src in [src1, src2, src3, src4, src5]:
        all_tickers.update(src.keys())

    print(f"[통합] 고유 종목: {len(all_tickers)}개")

    # 종목별 통합
    results = []
    for ticker in all_tickers:
        sources = []
        source_names = []
        for src, label in [(src1, "릴레이"), (src2, "그룹순환"), (src3, "눌림목"),
                           (src4, "퀀텀"), (src5, "동반매수")]:
            if ticker in src:
                sources.append(src[ticker])
                source_names.append(label)

        # parquet 기술적 데이터
        pq_data = get_parquet_data(ticker)

        # 통합 점수 계산
        score_detail = calc_integrated_score(ticker, sources, pq_data)

        # 이름 결정
        name = ""
        for s in sources:
            if s.get("name"):
                name = s["name"]
                break
        if not name:
            name = name_map.get(ticker, ticker)

        grade = classify_pick(score_detail["total"], len(sources), score_detail["rsi"])

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
        }

        results.append(rec)

    # 정렬: 등급 → 점수
    grade_order = {"강력매수": 0, "매수": 1, "관심매수": 2, "관찰": 3, "보류": 4}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["total_score"]))

    # 통계
    grade_stats = {}
    for r in results:
        g = r["grade"]
        grade_stats[g] = grade_stats.get(g, 0) + 1

    print(f"\n{'='*60}")
    print(f"[내일 추천] 총 {len(results)}건")
    for g in ["강력매수", "매수", "관심매수", "관찰", "보류"]:
        cnt = grade_stats.get(g, 0)
        if cnt:
            print(f"  {g}: {cnt}건")
    print(f"{'='*60}\n")

    # 상위 종목 출력
    for i, r in enumerate(results[:15], 1):
        srcs = "+".join(r["sources"])
        oh = f" 🔥-{r['score_breakdown']['overheat']}p" if r["score_breakdown"]["overheat"] > 0 else ""
        cond = f" | {r['entry_condition']}" if r.get("entry_condition") else ""
        reasons_str = ", ".join(r.get("reasons", [])[:3])
        print(f"  {i:2d}. [{r['grade']}] {r['name']}({r['ticker']}) "
              f"{r['total_score']}점{oh} ({r['n_sources']}개 소스: {srcs})")
        print(f"      진입:{r.get('entry_price',0):,}  손절:{r.get('stop_loss',0):,}  "
              f"목표:{r.get('target_price',0):,}{cond}")
        print(f"      근거: {reasons_str}")

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
        "picks": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUTPUT_PATH}")
    print(f"[대상일] {output['target_date_label']} ({output['target_date']})")


if __name__ == "__main__":
    main()
