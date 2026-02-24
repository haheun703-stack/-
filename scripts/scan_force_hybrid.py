"""세력감지 하이브리드 스캐너 — 3층 통합 분석

3개 시스템을 통합하여 세력감지 탭에 공급합니다:

  Layer 1. 수급건전성 (매크로) — 시장 전체 자금 흐름 건전성
    - 외인/기관 순매수 동향 (최근 5일)
    - KOSPI 대비 수급 온도
    - 종합 경보 등급: 안전 / 주의 / 위험

  Layer 2. 이상거래 탐지 (마이크로) — 기존 5패턴 + VWAP 강화
    - P1 거래량폭발, P2 수급반전, P3 BB스퀴즈
    - P4 OBV다이버전스, P5 수급이탈
    - + P6 VWAP 이탈 (당일 VWAP 대비 종가 괴리)
    - + P7 연속 기관/외인 매집 (5일+ 연속 순매수)

  Layer 3. 이벤트 레이더 (메조) — RSS 뉴스 + DART 공시 기반 이벤트 감지
    - crawl_market_news의 high/medium 임팩트 뉴스
    - DART 전자공시 (뉴스 대비 30분~수시간 선행)
    - theme_dictionary 키워드 매칭 → 수혜종목 연결

출력: data/force_hybrid.json

Usage:
    python scripts/scan_force_hybrid.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
KOSPI_PATH = PROJECT_ROOT / "data" / "kospi_index.csv"
MARKET_NEWS_PATH = PROJECT_ROOT / "data" / "market_news.json"
DART_PATH = PROJECT_ROOT / "data" / "dart_disclosures.json"
THEME_DICT_PATH = PROJECT_ROOT / "config" / "theme_dictionary.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "force_hybrid.json"


# ══════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════

def _sf(val, default=0):
    """NaN/Inf 안전 변환"""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else round(v, 2)
    except (TypeError, ValueError):
        return default


def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드 → 종목명 매핑"""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name, ticker = parts
            name_map[ticker] = name
    return name_map


def get_flow_from_csv(ticker: str) -> dict:
    """CSV에서 최근 수급 상세 데이터 추출"""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return {}
    try:
        df = pd.read_csv(csvs[0], parse_dates=["Date"])
        df = df.dropna(subset=["Close"]).sort_values("Date")
        if len(df) < 10:
            return {}
        tail = df.tail(10)
        f_net = tail["Foreign_Net"].values if "Foreign_Net" in tail.columns else np.zeros(10)
        i_net = tail["Inst_Net"].values if "Inst_Net" in tail.columns else np.zeros(10)
        return {
            "f_prev5": float(np.nansum(f_net[:5])),
            "f_last5": float(np.nansum(f_net[5:])),
            "i_prev5": float(np.nansum(i_net[:5])),
            "i_last5": float(np.nansum(i_net[5:])),
            "f_today": float(f_net[-1]) if len(f_net) > 0 else 0,
            "i_today": float(i_net[-1]) if len(i_net) > 0 else 0,
            # 연속 순매수 일수
            "f_streak": _count_streak(f_net),
            "i_streak": _count_streak(i_net),
        }
    except Exception:
        return {}


def _count_streak(arr) -> int:
    """최근부터 연속 양수(순매수) 일수"""
    streak = 0
    for v in reversed(arr):
        if np.isnan(v) or v <= 0:
            break
        streak += 1
    return streak


# ══════════════════════════════════════════
# Layer 1: 수급건전성 (시장 매크로)
# ══════════════════════════════════════════

def analyze_supply_demand_health() -> dict:
    """시장 전체 수급 건전성 분석"""
    # 전 종목 CSV에서 외인/기관 수급 집계
    total_foreign_5d = 0
    total_inst_5d = 0
    total_foreign_today = 0
    total_inst_today = 0
    foreign_buying_count = 0  # 외인 순매수 종목 수
    inst_buying_count = 0     # 기관 순매수 종목 수
    total_stocks = 0

    for pq in PROCESSED_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq, columns=["close", "volume", "외국인합계", "기관합계"])
            if len(df) < 5:
                continue
            tail = df.tail(5)
            close = float(tail["close"].iloc[-1])
            vol = float(tail["volume"].iloc[-1])
            if close < 1000 or vol < 1000:
                continue

            total_stocks += 1
            f_5d = float(np.nansum(tail["외국인합계"].values)) if "외국인합계" in tail.columns else 0
            i_5d = float(np.nansum(tail["기관합계"].values)) if "기관합계" in tail.columns else 0
            f_today = float(tail["외국인합계"].iloc[-1]) if "외국인합계" in tail.columns else 0
            i_today = float(tail["기관합계"].iloc[-1]) if "기관합계" in tail.columns else 0

            total_foreign_5d += f_5d
            total_inst_5d += i_5d
            total_foreign_today += f_today
            total_inst_today += i_today

            if f_5d > 0:
                foreign_buying_count += 1
            if i_5d > 0:
                inst_buying_count += 1

        except Exception:
            continue

    # KOSPI 동향
    kospi_info = _get_kospi_trend()

    # 건전성 점수 계산 (0~100)
    score = 50  # 기본 중립

    # 외인 수급 (±20)
    if total_stocks > 0:
        f_buy_ratio = foreign_buying_count / total_stocks
        score += (f_buy_ratio - 0.5) * 40  # 50% 기준 ±20

    # 기관 수급 (±15)
    if total_stocks > 0:
        i_buy_ratio = inst_buying_count / total_stocks
        score += (i_buy_ratio - 0.5) * 30  # 50% 기준 ±15

    # KOSPI 추세 (±15)
    if kospi_info.get("above_ma20"):
        score += 10
    if kospi_info.get("above_ma60"):
        score += 5
    if kospi_info.get("pct_5d", 0) > 1:
        score += 5
    elif kospi_info.get("pct_5d", 0) < -1:
        score -= 5

    score = max(0, min(100, score))

    # 경보 등급
    if score >= 65:
        alert = "안전"
        alert_color = "#10B981"
        alert_desc = "수급 양호 — 외인/기관 유입 우세"
    elif score >= 40:
        alert = "주의"
        alert_color = "#F59E0B"
        alert_desc = "수급 혼조 — 매수/매도 혼재"
    else:
        alert = "위험"
        alert_color = "#EF4444"
        alert_desc = "수급 악화 — 외인/기관 이탈 우세"

    return {
        "score": round(score, 1),
        "alert": alert,
        "alert_color": alert_color,
        "alert_desc": alert_desc,
        "total_stocks": total_stocks,
        "foreign": {
            "net_5d": round(total_foreign_5d),
            "net_today": round(total_foreign_today),
            "buying_count": foreign_buying_count,
            "buying_ratio": round(foreign_buying_count / max(total_stocks, 1) * 100, 1),
        },
        "institution": {
            "net_5d": round(total_inst_5d),
            "net_today": round(total_inst_today),
            "buying_count": inst_buying_count,
            "buying_ratio": round(inst_buying_count / max(total_stocks, 1) * 100, 1),
        },
        "kospi": kospi_info,
    }


def _get_kospi_trend() -> dict:
    """KOSPI 지수 추세"""
    if not KOSPI_PATH.exists():
        return {}
    try:
        df = pd.read_csv(KOSPI_PATH, parse_dates=["Date"])
        df = df.sort_values("Date").tail(60)
        if len(df) < 20:
            return {}
        last = df.iloc[-1]
        close = float(last["Close"])
        ma20 = float(df["Close"].tail(20).mean())
        ma60 = float(df["Close"].tail(60).mean()) if len(df) >= 60 else ma20

        pct_1d = (close / float(df.iloc[-2]["Close"]) - 1) * 100 if len(df) >= 2 else 0
        pct_5d = (close / float(df.iloc[-6]["Close"]) - 1) * 100 if len(df) >= 6 else 0

        return {
            "close": round(close, 2),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "above_ma20": close > ma20,
            "above_ma60": close > ma60,
            "pct_1d": round(pct_1d, 2),
            "pct_5d": round(pct_5d, 2),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════
# Layer 2: 이상거래 탐지 (기존 확장)
# ══════════════════════════════════════════

def detect_volume_explosion(df: pd.DataFrame) -> dict | None:
    """P1: 거래량 폭발"""
    if "volume_surge_ratio" not in df.columns:
        return None
    last = df.iloc[-1]
    vsr = float(last["volume_surge_ratio"])
    if np.isnan(vsr) or vsr < 2.5:
        return None
    pct = float(last.get("price_change", 0)) if "price_change" in df.columns else 0
    direction = "상승폭발" if pct > 1.0 else ("하락폭발" if pct < -1.0 else "횡보폭발")
    return {
        "pattern": "P1_거래량폭발",
        "strength": min(vsr / 3.0, 3.0),
        "vsr": round(vsr, 1),
        "direction": direction,
        "price_change": _sf(pct),
        "desc": f"거래량 {vsr:.1f}배 폭발 ({direction})",
    }


def detect_flow_reversal(df: pd.DataFrame, csv_flow: dict) -> dict | None:
    """P2: 수급 반전"""
    if not csv_flow:
        if "외국인합계" not in df.columns:
            return None
        tail = df.tail(10)
        if len(tail) < 10:
            return None
        f_vals = tail["외국인합계"].values
        i_vals = tail["기관합계"].values if "기관합계" in tail.columns else np.zeros(10)
        f_prev5, f_last5 = float(np.nansum(f_vals[:5])), float(np.nansum(f_vals[5:]))
        i_prev5, i_last5 = float(np.nansum(i_vals[:5])), float(np.nansum(i_vals[5:]))
        f_today = float(f_vals[-1]) if not np.isnan(f_vals[-1]) else 0
        i_today = float(i_vals[-1]) if not np.isnan(i_vals[-1]) else 0
    else:
        f_prev5, f_last5 = csv_flow["f_prev5"], csv_flow["f_last5"]
        i_prev5, i_last5 = csv_flow["i_prev5"], csv_flow["i_last5"]
        f_today, i_today = csv_flow["f_today"], csv_flow["i_today"]

    reasons, strength = [], 0
    if f_prev5 < 0 and f_last5 > 0 and abs(f_last5) > abs(f_prev5) * 0.5:
        reasons.append(f"외인 매도→매수 반전 ({f_prev5:+,.0f}→{f_last5:+,.0f})")
        strength += 1.5
    if i_prev5 < 0 and i_last5 > 0 and abs(i_last5) > abs(i_prev5) * 0.5:
        reasons.append(f"기관 매도→매수 반전 ({i_prev5:+,.0f}→{i_last5:+,.0f})")
        strength += 1.5
    if f_today > 0 and i_today > 0:
        reasons.append(f"금일 외인({f_today:+,.0f})+기관({i_today:+,.0f}) 동시매수")
        strength += 0.5
    if strength < 1.0:
        return None
    return {
        "pattern": "P2_수급반전", "strength": min(strength, 3.0),
        "reasons": reasons, "desc": " / ".join(reasons),
    }


def detect_bb_squeeze(df: pd.DataFrame) -> dict | None:
    """P3: BB Squeeze → 폭발"""
    if "bb_width" not in df.columns or len(df) < 30:
        return None
    bw = df["bb_width"].dropna()
    if len(bw) < 30:
        return None
    recent_20 = bw.iloc[-20:]
    min_bw, curr_bw = float(recent_20.min()), float(bw.iloc[-1])
    prev_bw = float(bw.iloc[-2]) if len(bw) >= 2 else curr_bw
    avg_bw = float(recent_20.mean())
    if np.isnan(min_bw) or np.isnan(curr_bw):
        return None
    was_squeezed = min_bw < avg_bw * 0.7
    is_expanding = curr_bw > min_bw * 1.5 and curr_bw > prev_bw * 1.1
    if not (was_squeezed and is_expanding):
        return None
    vsr = float(df.iloc[-1].get("volume_surge_ratio", 1.0))
    vol_confirm = vsr > 1.5
    expansion = curr_bw / min_bw if min_bw > 0 else 1.0
    strength = min(expansion / 2.0, 2.5) + (0.5 if vol_confirm else 0)
    return {
        "pattern": "P3_BB스퀴즈", "strength": min(strength, 3.0),
        "expansion": _sf(expansion), "vol_confirm": vol_confirm,
        "desc": f"BB 수축→폭발 ({expansion:.1f}배)" + (" +거래량" if vol_confirm else ""),
    }


def detect_obv_divergence(df: pd.DataFrame) -> dict | None:
    """P4: OBV 다이버전스 — 은밀 매집"""
    if "obv" not in df.columns or len(df) < 20:
        return None
    tail = df.tail(20)
    close, obv = tail["close"].values, tail["obv"].values
    if np.any(np.isnan(close[-5:])) or np.any(np.isnan(obv[-5:])):
        return None
    price_chg_10 = (close[-1] / close[-10] - 1) * 100 if close[-10] > 0 else 0
    obv_start = obv[-10]
    if obv_start == 0:
        return None
    obv_chg_pct = (obv[-1] / obv_start - 1) * 100
    if not (-5 < price_chg_10 < 3 and obv_chg_pct > 3):
        return None
    vsr = float(df.iloc[-1].get("volume_surge_ratio", 1.0))
    vol_contracted = vsr < 0.7
    strength = min(obv_chg_pct / 10.0, 2.5) + (0.3 if vol_contracted else 0)
    return {
        "pattern": "P4_OBV다이버전스", "strength": min(strength, 3.0),
        "price_chg_10d": _sf(price_chg_10), "obv_chg_10d": _sf(obv_chg_pct),
        "vol_contracted": vol_contracted,
        "desc": f"가격 {price_chg_10:+.1f}% vs OBV +{obv_chg_pct:.1f}% (은밀매집"
                + (" +저거래량)" if vol_contracted else ")"),
    }


def detect_flow_exit(df: pd.DataFrame, csv_flow: dict) -> dict | None:
    """P5: 수급 이탈 경고"""
    if not csv_flow:
        if "외국인합계" not in df.columns:
            return None
        tail = df.tail(5)
        if len(tail) < 5:
            return None
        f_5d = float(np.nansum(tail["외국인합계"].values))
        i_5d = float(np.nansum(tail["기관합계"].values)) if "기관합계" in tail.columns else 0
        f_today = float(tail["외국인합계"].iloc[-1]) if not np.isnan(tail["외국인합계"].iloc[-1]) else 0
        i_today = float(tail["기관합계"].iloc[-1]) if "기관합계" in tail.columns and not np.isnan(tail["기관합계"].iloc[-1]) else 0
    else:
        f_5d, i_5d = csv_flow["f_last5"], csv_flow["i_last5"]
        f_today, i_today = csv_flow["f_today"], csv_flow["i_today"]

    if not (f_5d < 0 and i_5d < 0 and f_today < 0 and i_today < 0):
        return None
    total = abs(f_5d) + abs(i_5d)
    strength = min(1.0 + (total / 1e9), 3.0)
    return {
        "pattern": "P5_수급이탈", "strength": min(strength, 3.0),
        "f_5d": _sf(f_5d), "i_5d": _sf(i_5d),
        "desc": f"외인({f_5d:+,.0f})+기관({i_5d:+,.0f}) 동시매도 5일 지속",
    }


def detect_consecutive_accumulation(csv_flow: dict) -> dict | None:
    """P6: 연속 매집 — 외인 또는 기관 5일+ 연속 순매수"""
    if not csv_flow:
        return None
    f_streak = csv_flow.get("f_streak", 0)
    i_streak = csv_flow.get("i_streak", 0)

    if f_streak < 5 and i_streak < 5:
        return None

    reasons = []
    strength = 0
    if f_streak >= 5:
        reasons.append(f"외인 {f_streak}일 연속 순매수")
        strength += min(f_streak / 5.0, 2.0)
    if i_streak >= 5:
        reasons.append(f"기관 {i_streak}일 연속 순매수")
        strength += min(i_streak / 5.0, 2.0)

    return {
        "pattern": "P6_연속매집", "strength": min(strength, 3.0),
        "f_streak": f_streak, "i_streak": i_streak,
        "desc": " / ".join(reasons),
    }


def detect_vwap_breakout(df: pd.DataFrame) -> dict | None:
    """P7: VWAP 이탈 — 종가가 추정 VWAP 대비 크게 괴리"""
    if len(df) < 20:
        return None
    # 간이 VWAP 추정: 최근 20일 (close * volume 합) / volume 합
    tail = df.tail(20)
    close_arr = tail["close"].values
    vol_arr = tail["volume"].values
    if np.any(np.isnan(close_arr)) or np.any(np.isnan(vol_arr)):
        return None
    total_vol = np.nansum(vol_arr)
    if total_vol <= 0:
        return None
    vwap_20 = float(np.nansum(close_arr * vol_arr) / total_vol)
    last_close = float(close_arr[-1])
    if vwap_20 <= 0:
        return None

    gap_pct = (last_close / vwap_20 - 1) * 100

    # 상방 이탈 (매집 후 돌파) 또는 하방 이탈 (투매)
    if abs(gap_pct) < 3:
        return None

    if gap_pct > 0:
        direction = "상방돌파"
        strength = min(gap_pct / 5.0, 2.5)
    else:
        direction = "하방이탈"
        strength = min(abs(gap_pct) / 5.0, 2.5)

    return {
        "pattern": "P7_VWAP이탈", "strength": min(strength, 3.0),
        "vwap_20": round(vwap_20), "gap_pct": _sf(gap_pct),
        "direction": direction,
        "desc": f"VWAP20 대비 {gap_pct:+.1f}% ({direction})",
    }


def classify_whale(patterns: list[dict]) -> str:
    """패턴 조합으로 핵심필터 등급 분류"""
    p_names = [p["pattern"] for p in patterns]
    has_exit = "P5_수급이탈" in p_names
    positive_patterns = [p for p in patterns if p["pattern"] != "P5_수급이탈"]

    if has_exit and not positive_patterns:
        return "이탈경고"
    obv_p = next((p for p in patterns if p["pattern"] == "P4_OBV다이버전스"), None)
    if obv_p and obv_p.get("vol_contracted"):
        return "매집의심"
    # P6 연속매집 포함 시 매집의심
    if "P6_연속매집" in p_names:
        if len(positive_patterns) >= 2:
            return "세력포착"
        return "매집의심"
    if len(positive_patterns) >= 2:
        return "세력포착"
    if len(positive_patterns) == 1:
        return "이상감지"
    if has_exit and positive_patterns:
        return "혼합시그널"
    return "이상감지"


def scan_anomaly() -> list[dict]:
    """Layer 2: 전종목 이상거래 스캔 (7패턴)"""
    name_map = build_name_map()
    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    print(f"[Layer2] {len(parquets)}개 종목 이상거래 스캔...")

    results = []
    for pq in parquets:
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 30:
                continue
            df = df.tail(60)
            last = df.iloc[-1]

            close = float(last.get("close", 0))
            vol = float(last.get("volume", 0))
            if close < 1000 or vol < 1000 or close * vol < 5e8:
                continue

            csv_flow = get_flow_from_csv(ticker)

            detected = []
            for fn in [
                lambda: detect_volume_explosion(df),
                lambda: detect_flow_reversal(df, csv_flow),
                lambda: detect_bb_squeeze(df),
                lambda: detect_obv_divergence(df),
                lambda: detect_flow_exit(df, csv_flow),
                lambda: detect_consecutive_accumulation(csv_flow),
                lambda: detect_vwap_breakout(df),
            ]:
                p = fn()
                if p:
                    detected.append(p)

            if not detected:
                continue

            grade = classify_whale(detected)
            total_strength = sum(p["strength"] for p in detected)
            name = name_map.get(ticker, ticker)

            results.append({
                "ticker": ticker,
                "name": name,
                "close": int(close),
                "price_change": _sf(last.get("price_change", 0)),
                "volume": int(vol),
                "rsi": _sf(last.get("rsi_14", 50)),
                "volume_surge_ratio": _sf(last.get("volume_surge_ratio", 1)),
                "above_ma20": close > float(last.get("sma_20", 0)) if last.get("sma_20", 0) else False,
                "above_ma60": close > float(last.get("sma_60", 0)) if last.get("sma_60", 0) else False,
                "foreign_5d": _sf(last.get("foreign_net_5d", 0)),
                "grade": grade,
                "strength": _sf(total_strength),
                "pattern_count": len(detected),
                "patterns": detected,
            })
        except Exception as e:
            logger.debug("종목 %s 처리 실패: %s", ticker, e)

    grade_order = {"세력포착": 0, "매집의심": 1, "이상감지": 2, "혼합시그널": 3, "이탈경고": 4}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["strength"]))
    return results


# ══════════════════════════════════════════
# Layer 3: 이벤트 레이더 (뉴스 기반)
# ══════════════════════════════════════════

def scan_event_radar(surging_stocks: list[dict] | None = None) -> dict:
    """Layer 3: 뉴스 기반 이벤트 레이더

    Args:
        surging_stocks: Layer 2에서 감지된 급등주 리스트
            [{name, ticker, price_change, ...}, ...]
    """
    events = []
    theme_hits = []
    surging_news = []  # 급등주 자동 뉴스

    # 1) market_news.json에서 high/medium 임팩트 뉴스 가져오기
    if MARKET_NEWS_PATH.exists():
        try:
            with open(MARKET_NEWS_PATH, "r", encoding="utf-8") as f:
                news_data = json.load(f)
            articles = news_data.get("articles", [])
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            for art in articles:
                if art.get("impact") in ("high", "medium"):
                    # 최근 2일 뉴스만
                    if art.get("date", "") >= yesterday:
                        events.append({
                            "title": art["title"],
                            "source": art.get("source", ""),
                            "date": art.get("date", ""),
                            "impact": art["impact"],
                            "url": art.get("url", ""),
                        })
        except Exception as e:
            logger.warning("market_news 로드 실패: %s", e)

    # 2) theme_dictionary 키워드로 뉴스 → 수혜종목 매칭
    theme_dict = _load_theme_dict()
    all_news_titles = [e["title"] for e in events]

    if theme_dict and events:
        for event in events:
            title = event["title"]
            for theme_name, theme_data in theme_dict.items():
                keywords = theme_data.get("keywords", [])
                matched_kw = None
                for kw in keywords:
                    if kw.lower() in title.lower():
                        matched_kw = kw
                        break
                if matched_kw:
                    stocks = theme_data.get("stocks", [])
                    theme_hits.append({
                        "theme": theme_name,
                        "keyword": matched_kw,
                        "news_title": title,
                        "news_date": event.get("date", ""),
                        "impact": event["impact"],
                        "stocks": stocks[:5],  # 상위 5종목
                    })

    # 3) DART 공시 연동 (뉴스 대비 30분~수시간 선행)
    dart_disclosures = []
    if DART_PATH.exists():
        try:
            with open(DART_PATH, "r", encoding="utf-8") as f:
                dart_data = json.load(f)
            # tier1 + tier2 공시를 이벤트로 추가
            for tier_key in ("tier1", "tier2"):
                for d in dart_data.get(tier_key, []):
                    dart_disclosures.append({
                        "title": f"[공시] {d['corp_name']} — {d['report_nm'][:50]}",
                        "source": "DART",
                        "date": d.get("rcept_dt", "")[:4] + "-" + d.get("rcept_dt", "")[4:6] + "-" + d.get("rcept_dt", "")[6:8] if len(d.get("rcept_dt", "")) == 8 else "",
                        "impact": "high" if "tier1" in d.get("tier", "") else "medium",
                        "tier": d.get("tier", ""),
                        "keyword": d.get("keyword", ""),
                        "corp_name": d.get("corp_name", ""),
                        "stock_code": d.get("stock_code", ""),
                        "url": d.get("url", ""),
                    })
            # 유니버스 관련 공시 별도 수집
            universe_dart = dart_data.get("universe_hits", [])
            logger.info("  DART 공시: tier1+2 %d건, 유니버스 관련 %d건",
                        len(dart_disclosures), len(universe_dart))
        except Exception as e:
            logger.warning("DART 공시 로드 실패: %s", e)
            universe_dart = []
    else:
        universe_dart = []

    # DART 공시도 테마 매칭
    if theme_dict and dart_disclosures:
        for disc in dart_disclosures:
            title = disc["title"]
            for theme_name, theme_data in theme_dict.items():
                for kw in theme_data.get("keywords", []):
                    if kw.lower() in title.lower():
                        theme_hits.append({
                            "theme": theme_name,
                            "keyword": kw,
                            "news_title": title,
                            "news_date": disc.get("date", ""),
                            "impact": disc["impact"],
                            "stocks": theme_data.get("stocks", [])[:5],
                            "source": "DART공시",
                        })
                        break

    # 4) 급등주 자동 뉴스 검색 (P1 거래량폭발 + 가격변동 >=8%)
    if surging_stocks:
        stock_names = [s["name"] for s in surging_stocks[:10]]  # 최대 10종목
        try:
            from scripts.crawl_market_news import crawl_stock_news
            raw_news = crawl_stock_news(stock_names, days=3)
            for n in raw_news:
                surging_news.append({
                    "stock_name": n["stock_name"],
                    "title": n["title"],
                    "source": n.get("source", ""),
                    "date": n.get("date", ""),
                    "impact": n.get("impact", "medium"),
                    "url": n.get("url", ""),
                })
                # 급등주 뉴스도 테마 매칭 시도
                if theme_dict:
                    for theme_name, theme_data in theme_dict.items():
                        for kw in theme_data.get("keywords", []):
                            if kw.lower() in n["title"].lower():
                                theme_hits.append({
                                    "theme": theme_name,
                                    "keyword": kw,
                                    "news_title": n["title"],
                                    "news_date": n.get("date", ""),
                                    "impact": "high",
                                    "stocks": [{"name": n["stock_name"]}],
                                    "source": "급등주뉴스",
                                })
                                break
            logger.info(f"  급등주 뉴스: {len(surging_news)}건 ({len(stock_names)}종목)")
        except Exception as e:
            logger.warning("급등주 뉴스 크롤 실패: %s", e)

    # 이벤트 요약 통계
    high_count = sum(1 for e in events if e["impact"] == "high")
    med_count = sum(1 for e in events if e["impact"] == "medium")

    # 시장 분위기 판정
    if high_count >= 3:
        mood = "긴장"
        mood_desc = f"HIGH 임팩트 뉴스 {high_count}건 — 변동성 주의"
    elif high_count >= 1:
        mood = "경계"
        mood_desc = f"HIGH {high_count}건 + MED {med_count}건 — 주요 이슈 모니터링"
    elif med_count >= 3:
        mood = "관심"
        mood_desc = f"중요 뉴스 {med_count}건 — 테마/이벤트 주시"
    elif surging_news:
        mood = "관심"
        mood_desc = f"급등주 {len(surging_stocks)}개 뉴스 {len(surging_news)}건 — 이벤트 확인"
    else:
        mood = "평온"
        mood_desc = "특별한 이벤트 없음 — 기술적 분석 우선"

    # theme_hits 중복 제거
    seen_theme = set()
    unique_themes = []
    for th in theme_hits:
        key = (th["theme"], th["news_title"][:30])
        if key not in seen_theme:
            seen_theme.add(key)
            unique_themes.append(th)

    return {
        "mood": mood,
        "mood_desc": mood_desc,
        "event_count": len(events),
        "high_impact": high_count,
        "medium_impact": med_count,
        "events": events[:15],  # 최대 15건
        "theme_hits": unique_themes[:15],  # 최대 15건
        "surging_stock_news": surging_news[:20],  # 급등주 뉴스 최대 20건
        "dart_disclosures": dart_disclosures[:20],  # DART 공시 최대 20건
        "dart_universe": universe_dart[:15],  # 유니버스 관련 DART 공시
    }


def _load_theme_dict() -> dict:
    """theme_dictionary.yaml 로드"""
    if not THEME_DICT_PATH.exists():
        return {}
    try:
        import yaml
        with open(THEME_DICT_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("themes", {})
    except Exception:
        return {}


# ══════════════════════════════════════════
# 메인 통합
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="세력감지 하이브리드 스캐너")
    parser.add_argument("--top", type=int, default=30, help="이상거래 출력 종목 수")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  세력감지 하이브리드 스캐너 (3-Layer)")
    print("=" * 60)

    # Layer 1: 수급건전성
    print("\n[Layer 1] 수급건전성 분석 중...")
    health = analyze_supply_demand_health()
    alert = health["alert"]
    print(f"  → 종합 경보: {alert} (점수 {health['score']})")
    print(f"  → 외인 순매수 비율: {health['foreign']['buying_ratio']}%")
    print(f"  → 기관 순매수 비율: {health['institution']['buying_ratio']}%")

    # Layer 2: 이상거래
    anomaly_items = scan_anomaly()
    anomaly_stats = {}
    for r in anomaly_items:
        g = r["grade"]
        anomaly_stats[g] = anomaly_stats.get(g, 0) + 1

    print(f"\n[Layer2] 총 {len(anomaly_items)}건 이상거래 탐지")
    for g in ["세력포착", "매집의심", "이상감지", "혼합시그널", "이탈경고"]:
        cnt = anomaly_stats.get(g, 0)
        if cnt:
            print(f"  {g}: {cnt}건")

    # Layer 2 → Layer 3 연계: P1 거래량폭발 + 가격변동 >=8% → 급등주
    surging_stocks = []
    for item in anomaly_items:
        has_p1 = any(p["pattern"] == "P1_거래량폭발" for p in item["patterns"])
        pct = abs(item.get("price_change", 0))
        if has_p1 and pct >= 8:
            surging_stocks.append(item)
    if surging_stocks:
        print(f"\n[Layer2→3] 급등주 {len(surging_stocks)}개 뉴스 자동검색 대상:")
        for s in surging_stocks[:5]:
            print(f"  → {s['name']}({s['ticker']}) {s['price_change']:+.1f}%")

    # Layer 3: 이벤트 레이더
    print("\n[Layer 3] 이벤트 레이더 스캔 중...")
    radar = scan_event_radar(surging_stocks=surging_stocks if surging_stocks else None)
    print(f"  → 시장 분위기: {radar['mood']} ({radar['mood_desc']})")
    print(f"  → 이벤트 {radar['event_count']}건 (HIGH:{radar['high_impact']} MED:{radar['medium_impact']})")
    if radar["theme_hits"]:
        print(f"  → 테마 매칭: {len(radar['theme_hits'])}건")
        for th in radar["theme_hits"][:5]:
            print(f"    [{th['theme']}] {th['keyword']} — {th['news_title'][:40]}...")
    if radar.get("surging_stock_news"):
        print(f"  → 급등주 뉴스: {len(radar['surging_stock_news'])}건")
        for sn in radar["surging_stock_news"][:5]:
            print(f"    [{sn['stock_name']}] {sn['title'][:50]}...")
    if radar.get("dart_disclosures"):
        print(f"  → DART 공시: {len(radar['dart_disclosures'])}건 (tier1+2)")
        for dc in radar["dart_disclosures"][:5]:
            icon = "🔴" if dc.get("impact") == "high" else "🟡"
            print(f"    {icon} {dc['corp_name']}({dc.get('stock_code','')}) [{dc.get('keyword','')}]")
    if radar.get("dart_universe"):
        print(f"  → 유니버스 DART: {len(radar['dart_universe'])}건")
        for du in radar["dart_universe"][:5]:
            print(f"    🎯 {du['corp_name']}({du['stock_code']}) — {du['report_nm'][:40]}")

    # 크로스 분석: 수급건전성 × 이상거래 맥락 해석
    cross_insights = []
    if health["alert"] == "위험":
        exit_count = anomaly_stats.get("이탈경고", 0)
        if exit_count > 0:
            cross_insights.append(f"수급 위험 + 이탈경고 {exit_count}건 → 시장 리스크 상승")
        accum_count = anomaly_stats.get("매집의심", 0) + anomaly_stats.get("세력포착", 0)
        if accum_count > 0:
            cross_insights.append(f"수급 위험 속 매집 {accum_count}건 → 역발상 매수 후보 (주의)")
    elif health["alert"] == "안전":
        accum_count = anomaly_stats.get("매집의심", 0) + anomaly_stats.get("세력포착", 0)
        if accum_count > 0:
            cross_insights.append(f"수급 안전 + 매집 {accum_count}건 → 강한 매수 신호")

    if cross_insights:
        print(f"\n[크로스 분석]")
        for insight in cross_insights:
            print(f"  → {insight}")

    # 상위 종목 출력
    print(f"\n{'─'*60}")
    print(f"[이상거래 TOP {min(args.top, len(anomaly_items))}]")
    for i, r in enumerate(anomaly_items[:args.top], 1):
        pats = " + ".join(p["pattern"].split("_")[1] for p in r["patterns"])
        print(f"  {i:2d}. [{r['grade']}] {r['name']}({r['ticker']}) "
              f"종가 {r['close']:,} ({r['price_change']:+.1f}%) "
              f"강도 {r['strength']:.1f} — {pats}")

    # JSON 저장
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "supply_demand_health": health,
        "anomaly": {
            "total_detected": len(anomaly_items),
            "stats": anomaly_stats,
            "items": anomaly_items[:args.top],
        },
        "event_radar": radar,
        "cross_insights": cross_insights,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
