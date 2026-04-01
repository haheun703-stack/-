"""눈치 엔진 — 시장 방향 감지 + ETF 추천

US 야간 시그널 + VIX + 파생 + 레짐 방향 + NXT를 종합하여
가장 빠르게 시장 방향을 판단하고, 레버리지/인버스 ETF를 추천한다.

핵심 원칙:
  1. 느린 레짐(MA20/60) 대신 빠른 시그널(US야간, VIX변화, 파생)로 판단
  2. 방향이 명확하면 적극 추천, 불명확하면 관망
  3. NXT 프리/애프터마켓 데이터로 확신도 보정

실행 시점:
  - BAT-A (06:10 KST): US 장 마감 직후 → 한국 장전 방향 판단
  - BAT-D (16:30 KST): 장마감 후 → 내일 방향 판단

출력:
  - data/market_sense.json

Usage:
    python scripts/market_sense_engine.py
    python scripts/market_sense_engine.py --send   # 텔레그램 발송
"""

from __future__ import annotations

import json
import logging
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

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
OUTPUT_PATH = DATA_DIR / "market_sense.json"

# ── ETF 매핑 ──
ETF_LONG = [
    {"code": "122630", "name": "KODEX 레버리지", "mult": 2.0, "desc": "KOSPI200 2배"},
    {"code": "069500", "name": "KODEX 200", "mult": 1.0, "desc": "KOSPI200 추종"},
]
ETF_SHORT = [
    {"code": "252670", "name": "KODEX 200선물인버스2X", "mult": -2.0, "desc": "KOSPI200 인버스 2배"},
    {"code": "114800", "name": "KODEX 인버스", "mult": -1.0, "desc": "KOSPI200 인버스"},
]
ETF_SAFE = [
    {"code": "132030", "name": "KODEX 골드선물(H)", "mult": 0, "desc": "금"},
    {"code": "148070", "name": "KODEX 국고채10년", "mult": 0, "desc": "채권"},
]

# ── 원자재 ETF 유니버스 ──
# US 원자재 시그널(GLD/USO/COPX/UNG/URA/SLV)과 한국 ETF 매핑
COMMODITY_ETF = {
    "oil": {
        "us_ticker": "USO",
        "label": "원유(WTI)",
        "etfs": [
            {"code": "261220", "name": "KODEX WTI원유선물(H)", "mult": 1.0},
            {"code": "130680", "name": "TIGER 원유선물Enhanced(H)", "mult": 1.0},
        ],
    },
    "gold": {
        "us_ticker": "GLD",
        "label": "금",
        "etfs": [
            {"code": "225130", "name": "ACE 골드선물레버리지", "mult": 2.0},
            {"code": "132030", "name": "KODEX 골드선물(H)", "mult": 1.0},
        ],
    },
    "silver": {
        "us_ticker": "SLV",
        "label": "은",
        "etfs": [
            {"code": "144600", "name": "KODEX 은선물(H)", "mult": 1.0},
        ],
    },
    "copper": {
        "us_ticker": "COPX",
        "label": "구리",
        "etfs": [
            {"code": "160580", "name": "TIGER 구리실물", "mult": 1.0},
        ],
    },
    "natgas": {
        "us_ticker": "UNG",
        "label": "천연가스",
        "etfs": [
            {"code": "217770", "name": "TIGER 천연가스선물Enhanced(H)", "mult": 1.0},
        ],
    },
    "uranium": {
        "us_ticker": "URA",
        "label": "우라늄/원전",
        "etfs": [
            {"code": "457990", "name": "HANARO 원자력iSelect", "mult": 1.0},
        ],
    },
}

# 시나리오 → 원자재 매핑 (활성 시나리오가 있으면 관련 원자재 부스트)
SCENARIO_COMMODITY_MAP = {
    "WAR_MIDDLE_EAST": ["oil", "gold"],
    "OIL_SPIKE": ["oil", "natgas"],
    "COMMODITY_SUPERCYCLE": ["copper", "oil", "gold", "silver"],
    "CHINA_STIMULUS": ["copper", "silver"],
    "NUCLEAR_RENAISSANCE": ["uranium"],
    "INFLATION_SPIKE": ["gold", "silver", "oil"],
    "GEOPOLITICAL_RISK": ["gold", "oil"],
}


def load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_nested_json(*parts: str) -> dict:
    path = DATA_DIR / Path(*parts)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════
# 1. 시그널 수집 (5축 눈치)
# ══════════════════════════════════════════

def gather_signals() -> dict:
    """모든 시그널 소스를 수집하여 방향 점수화."""
    signals = {}

    # ── 축1: US 야간 시그널 (가중치 35%) ──
    us = load_nested_json("us_market", "overnight_signal.json")
    us_grade = us.get("final_grade", "NEUTRAL")
    us_score_raw = us.get("final_score", 0)

    # US 등급 → 방향 점수 (-100 ~ +100)
    us_direction_map = {
        "STRONG_BULL": 100,
        "MILD_BULL": 50,
        "NEUTRAL": 0,
        "MILD_BEAR": -50,
        "STRONG_BEAR": -100,
    }
    us_direction = us_direction_map.get(us_grade, 0)

    # EWY (한국 프록시) 5일 수익률 반영
    ewy = us.get("index_direction", {}).get("EWY", {})
    ewy_5d = ewy.get("ret_5d", 0)
    ewy_bonus = min(max(ewy_5d * 10, -30), 30)  # ±3% → ±30점

    signals["us"] = {
        "grade": us_grade,
        "direction": us_direction,
        "ewy_5d": round(ewy_5d, 2),
        "ewy_bonus": round(ewy_bonus, 1),
        "score": round(us_direction + ewy_bonus, 1),
        "weight": 0.25,
    }

    # ── 축2: VIX 변화 (가중치 20%) ──
    vix_data = us.get("vix", {})
    vix_level = vix_data.get("level", 20)
    vix_z = vix_data.get("z_score", 0)

    # VIX 높으면 부정적, 낮으면 긍정적
    if vix_level < 15:
        vix_score = 80  # 매우 안정 → 강한 롱
    elif vix_level < 20:
        vix_score = 40  # 안정 → 약한 롱
    elif vix_level < 25:
        vix_score = -10  # 불안 → 약한 숏
    elif vix_level < 30:
        vix_score = -50  # 공포 → 숏
    else:
        vix_score = -90  # 극단 공포 → 강한 숏

    # VIX Z-score 추가 보정 (급변 감지)
    if vix_z > 1.5:
        vix_score -= 30  # VIX 급등 → 추가 부정
    elif vix_z < -1.0:
        vix_score += 20  # VIX 급락 → 추가 긍정

    vix_score = max(-100, min(100, vix_score))

    signals["vix"] = {
        "level": round(vix_level, 1),
        "z_score": round(vix_z, 2),
        "score": vix_score,
        "weight": 0.15,
    }

    # ── 축3: 파생 시그널 (가중치 15%) ──
    deriv = load_nested_json("derivatives", "derivatives_signal.json")
    deriv_composite = deriv.get("composite", {})
    deriv_grade = deriv_composite.get("grade", "NEUTRAL")
    deriv_score_raw = deriv_composite.get("score", 0)

    deriv_map = {
        "STRONG_BULL": 80,
        "MILD_BULL": 40,
        "NEUTRAL": 0,
        "MILD_BEAR": -40,
        "STRONG_BEAR": -80,
    }
    deriv_score = deriv_map.get(deriv_grade, 0)

    # 풋콜 반전 시그널 (극단 공포 후 반등 감지)
    pc_reversal = deriv.get("put_call_proxy", {}).get("reversal", "")
    if pc_reversal == "CALL_REVERSAL":
        deriv_score += 30  # 콜 반전 → 반등 시그널
    elif pc_reversal == "PUT_REVERSAL":
        deriv_score -= 30  # 풋 반전 → 추가 하락

    deriv_score = max(-100, min(100, deriv_score))

    signals["derivatives"] = {
        "grade": deriv_grade,
        "reversal": pc_reversal or "없음",
        "score": deriv_score,
        "weight": 0.15,
    }

    # ── 축4: 레짐 전환 방향 (가중치 15%) ──
    regime = load_json("regime_macro_signal.json")
    current_regime = regime.get("current_regime", "CAUTION")
    transition = regime.get("transition_direction", "")
    macro_score = regime.get("macro_score", 50)

    # 레짐 기본 방향
    regime_base = {"BULL": 80, "CAUTION": 20, "BEAR": -40, "CRISIS": -80}
    regime_score = regime_base.get(current_regime, 0)

    # 전환 방향 보정 (핵심 "눈치")
    if "접근" in transition:
        if "BULL" in transition:
            regime_score += 30  # BULL 접근 → 강한 롱 시그널
        elif "CRISIS" in transition:
            regime_score -= 30  # CRISIS 접근 → 강한 숏 시그널

    regime_score = max(-100, min(100, regime_score))

    signals["regime"] = {
        "current": current_regime,
        "transition": transition,
        "macro_score": macro_score,
        "score": regime_score,
        "weight": 0.15,
    }

    # ── 축5: NXT 프리/애프터마켓 (가중치 15%) ──
    today = date.today().isoformat()
    nxt_after = load_nested_json("nxt", f"nxt_after_{today}.json")
    nxt_pre = load_nested_json("nxt", f"nxt_pre_{today}.json")
    nxt_signal = load_json("nxt_signal.json") if (DATA_DIR / "nxt_signal.json").exists() else {}

    nxt_score = 0
    nxt_detail = "데이터 없음"

    if nxt_after or nxt_pre:
        # 애프터마켓: 전체 종목의 평균 프리미엄/갭
        summary = (nxt_after or nxt_pre).get("summary", {})
        if summary:
            gaps = [v.get("gap_pct", 0) for v in summary.values() if isinstance(v, dict)]
            buys = [v.get("net_buy_ratio", 0.5) for v in summary.values() if isinstance(v, dict)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            avg_buy = sum(buys) / len(buys) if buys else 0.5

            # 갭 방향 + 매수비율로 점수
            nxt_score = int(avg_gap * 20 + (avg_buy - 0.5) * 100)
            nxt_score = max(-100, min(100, nxt_score))
            nxt_detail = f"갭 {avg_gap:+.1f}%, 순매수비 {avg_buy:.0%}"

    # NXT 시그널 파일이 있으면 우선 사용
    if nxt_signal:
        after_picks = nxt_signal.get("aftermarket_picks", [])
        if after_picks:
            strong_buys = sum(1 for p in after_picks if p.get("signal") == "STRONG_BUY")
            sells = sum(1 for p in after_picks if p.get("signal") == "SELL")
            nxt_score = (strong_buys - sells) * 15
            nxt_score = max(-100, min(100, nxt_score))
            nxt_detail = f"STRONG_BUY {strong_buys}개, SELL {sells}개"

    signals["nxt"] = {
        "score": nxt_score,
        "detail": nxt_detail,
        "weight": 0.10,
    }

    # ── 축6: 원자재 시그널 (가중치 15%) ──
    commodity_scores = {}
    commodity_alerts = []
    # US overnight의 commodities 섹션에서 원자재 데이터 로드
    us_commodities = us.get("commodities", {})
    # US ticker → commodities 키 매핑
    _US_COMM_KEY = {
        "USO": "oil", "GLD": "gold", "COPX": "copper",
        "UNG": "natgas", "URA": "uranium", "SLV": "silver",
    }

    for comm_key, comm_info in COMMODITY_ETF.items():
        us_ticker = comm_info["us_ticker"]
        comm_data_key = _US_COMM_KEY.get(us_ticker, comm_key)
        ticker_data = us_commodities.get(comm_data_key, {})
        ret_1d = ticker_data.get("ret_1d", 0)
        ret_5d = ticker_data.get("ret_5d", 0)

        # 급등/급락 감지
        if abs(ret_1d) >= 3:  # 하루 ±3% 이상 → 강한 시그널
            commodity_alerts.append({
                "commodity": comm_info["label"],
                "change_1d": round(ret_1d, 2),
                "change_5d": round(ret_5d, 2),
                "etfs": comm_info["etfs"],
                "direction": "LONG" if ret_1d > 0 else "SHORT",
            })

        # 점수 계산 (5일 추세 + 1일 모멘텀)
        score = ret_1d * 10 + ret_5d * 5
        commodity_scores[comm_key] = round(max(-100, min(100, score)), 1)

    # 원자재 종합 점수 (전체 평균)
    if commodity_scores:
        avg_comm = sum(commodity_scores.values()) / len(commodity_scores)
    else:
        avg_comm = 0

    signals["commodity"] = {
        "scores": commodity_scores,
        "alerts": commodity_alerts,
        "score": round(avg_comm),
        "weight": 0.15,
    }

    # ── 축7: 시나리오 연동 (가중치 10%) ──
    scenario_data = load_json("scenario_supply_conflicts.json")
    active_scenarios = scenario_data.get("active_scenarios", [])

    scenario_score = 0
    scenario_commodity_boost = {}  # 시나리오가 부스트하는 원자재
    scenario_detail = "없음"

    if active_scenarios:
        # 활성 시나리오 점수 합산
        total_scenario = sum(s.get("score", 0) for s in active_scenarios)
        scenario_score = min(total_scenario // 2, 100)  # 최대 100

        # 시나리오별 관련 원자재 매핑
        for sc in active_scenarios:
            sc_name = sc.get("name", "")
            boost_commodities = SCENARIO_COMMODITY_MAP.get(sc_name, [])
            for comm in boost_commodities:
                if comm not in scenario_commodity_boost:
                    scenario_commodity_boost[comm] = 0
                scenario_commodity_boost[comm] += sc.get("score", 0)

        top_scenarios = [f"{s['name']}({s.get('score',0)})" for s in active_scenarios[:3]]
        scenario_detail = ", ".join(top_scenarios)

        # 시나리오 방향: 지정학 리스크는 보통 KOSPI에 부정적
        if any("WAR" in s.get("name", "") for s in active_scenarios):
            scenario_score = -scenario_score  # 전쟁 → KOSPI 부정적

    signals["scenario"] = {
        "active": [s.get("name") for s in active_scenarios],
        "commodity_boost": scenario_commodity_boost,
        "detail": scenario_detail,
        "score": scenario_score,
        "weight": 0.10,
    }

    return signals


# ══════════════════════════════════════════
# 2. 종합 판단
# ══════════════════════════════════════════

def calculate_direction(signals: dict) -> dict:
    """5축 시그널을 가중 합산하여 방향 판단."""

    # 가중 합산 (-100 ~ +100)
    total = 0
    for key, sig in signals.items():
        total += sig["score"] * sig["weight"]

    total = round(total, 1)

    # 방향 판정
    if total >= 40:
        direction = "STRONG_LONG"
        direction_kr = "강한 상승"
        etfs = ETF_LONG
    elif total >= 15:
        direction = "LONG"
        direction_kr = "상승"
        etfs = ETF_LONG
    elif total > -15:
        direction = "NEUTRAL"
        direction_kr = "중립"
        etfs = ETF_SAFE
    elif total > -40:
        direction = "SHORT"
        direction_kr = "하락"
        etfs = ETF_SHORT
    else:
        direction = "STRONG_SHORT"
        direction_kr = "강한 하락"
        etfs = ETF_SHORT

    # 신뢰도 (절대값 기준)
    confidence = min(abs(total), 100)

    # 추천 ETF 선택 (강도에 따라)
    if abs(total) >= 40 and len(etfs) > 0:
        pick = etfs[0]  # 레버리지/인버스2X
    elif len(etfs) > 1:
        pick = etfs[1]  # 1배짜리
    elif len(etfs) > 0:
        pick = etfs[0]
    else:
        pick = None

    # 동적 손절/목표
    vix = signals.get("vix", {}).get("level", 20)
    if vix < 15:
        stop_pct, target_pct = -2.0, 4.0
    elif vix < 20:
        stop_pct, target_pct = -2.5, 3.5
    elif vix < 25:
        stop_pct, target_pct = -3.0, 4.0
    elif vix < 30:
        stop_pct, target_pct = -4.0, 5.0
    else:
        stop_pct, target_pct = -5.0, 7.0

    # ── 원자재 ETF 추천 (급등/급락 감지 시) ──
    commodity_picks = []
    commodity_alerts = signals.get("commodity", {}).get("alerts", [])
    scenario_boost = signals.get("scenario", {}).get("commodity_boost", {})

    for alert in commodity_alerts:
        comm_etfs = alert["etfs"]
        if comm_etfs:
            best_etf = comm_etfs[0]  # 레버리지 우선
            # 시나리오 부스트가 있으면 신뢰도 증가
            comm_key = next(
                (k for k, v in COMMODITY_ETF.items() if v["label"] == alert["commodity"]),
                None,
            )
            boost = scenario_boost.get(comm_key, 0)

            commodity_picks.append({
                "commodity": alert["commodity"],
                "direction": alert["direction"],
                "change_1d": alert["change_1d"],
                "etf": best_etf,
                "scenario_boost": boost,
                "confidence": min(abs(alert["change_1d"]) * 15 + boost // 2, 95),
            })

    # 시나리오 활성 + 원자재 급등 없어도, 시나리오 부스트 높은 원자재 추천
    for comm_key, boost_score in scenario_boost.items():
        if boost_score >= 60 and comm_key in COMMODITY_ETF:
            # 이미 alerts에 있으면 스킵
            already = any(
                p["commodity"] == COMMODITY_ETF[comm_key]["label"]
                for p in commodity_picks
            )
            if not already:
                comm = COMMODITY_ETF[comm_key]
                if comm["etfs"]:
                    commodity_picks.append({
                        "commodity": comm["label"],
                        "direction": "LONG",
                        "change_1d": 0,
                        "etf": comm["etfs"][0],
                        "scenario_boost": boost_score,
                        "confidence": min(boost_score // 2, 80),
                    })

    return {
        "direction": direction,
        "direction_kr": direction_kr,
        "score": total,
        "confidence": round(confidence),
        "recommended_etf": pick,
        "commodity_picks": commodity_picks,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "etf_candidates": etfs,
    }


# ══════════════════════════════════════════
# 3. 적중률 기록
# ══════════════════════════════════════════

def update_accuracy_log(direction: str, score: float):
    """방향 예측 기록 → 다음날 검증용."""
    log_path = DATA_DIR / "market_sense_log.json"
    log = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)

    today = date.today().isoformat()

    # 이미 오늘 기록 있으면 업데이트
    existing = next((e for e in log if e["date"] == today), None)
    if existing:
        existing["direction"] = direction
        existing["score"] = score
    else:
        log.append({
            "date": today,
            "direction": direction,
            "score": score,
            "actual_change_pct": None,  # 다음날 검증 시 채움
            "hit": None,
        })

    # 최근 60일만 유지
    log = log[-60:]

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def verify_yesterday(kospi_change_pct: float):
    """전날 예측 vs 실제 결과 검증."""
    log_path = DATA_DIR / "market_sense_log.json"
    if not log_path.exists():
        return None

    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)

    if not log:
        return None

    # 가장 최근 미검증 기록 찾기
    for entry in reversed(log):
        if entry.get("actual_change_pct") is None:
            entry["actual_change_pct"] = round(kospi_change_pct, 2)
            predicted_up = entry["direction"] in ("STRONG_LONG", "LONG")
            predicted_down = entry["direction"] in ("STRONG_SHORT", "SHORT")
            actual_up = kospi_change_pct > 0.3
            actual_down = kospi_change_pct < -0.3

            if predicted_up and actual_up:
                entry["hit"] = True
            elif predicted_down and actual_down:
                entry["hit"] = True
            elif entry["direction"] == "NEUTRAL" and abs(kospi_change_pct) < 0.5:
                entry["hit"] = True
            else:
                entry["hit"] = False
            break

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # 적중률 계산
    verified = [e for e in log if e.get("hit") is not None]
    if verified:
        hits = sum(1 for e in verified if e["hit"])
        return {"total": len(verified), "hits": hits, "rate": round(hits / len(verified) * 100, 1)}
    return None


# ══════════════════════════════════════════
# 4. 메인
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="눈치 엔진 — 시장 방향 감지")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    parser.add_argument("--verify", type=float, default=None, help="전날 KOSPI 등락률(%%)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  눈치 엔진 — 시장 방향 감지 + ETF 추천")
    logger.info("=" * 60)

    # 전날 검증
    if args.verify is not None:
        result = verify_yesterday(args.verify)
        if result:
            logger.info("  적중률: %d/%d = %.1f%%", result["hits"], result["total"], result["rate"])

    # ── 시그널 수집 ──
    signals = gather_signals()

    for name, sig in signals.items():
        logger.info("  [%s] 점수: %+d (가중치 %.0f%%)", name, sig["score"], sig["weight"] * 100)

    # ── 종합 판단 ──
    result = calculate_direction(signals)

    logger.info("")
    logger.info("  ★ 방향: %s (%s)", result["direction_kr"], result["direction"])
    logger.info("  ★ 종합 점수: %+.1f / 신뢰도: %d%%", result["score"], result["confidence"])

    if result["recommended_etf"]:
        etf = result["recommended_etf"]
        logger.info("  ★ 추천 ETF: %s (%s) — %s", etf["name"], etf["code"], etf["desc"])
        logger.info("  ★ 손절: %.1f%% / 목표: +%.1f%%", result["stop_pct"], result["target_pct"])
    else:
        logger.info("  ★ 추천: 관망")

    # ── 원자재 ETF 추천 ──
    commodity_picks = result.get("commodity_picks", [])
    if commodity_picks:
        logger.info("")
        logger.info("  ── 원자재 ETF 추천 ──")
        for cp in commodity_picks:
            etf = cp["etf"]
            sc_boost = f" (시나리오 +{cp['scenario_boost']})" if cp["scenario_boost"] else ""
            logger.info("  %s %s: %s (%s) — 신뢰도 %d%%%s",
                        "▲" if cp["direction"] == "LONG" else "▼",
                        cp["commodity"],
                        etf["name"], etf["code"],
                        cp["confidence"], sc_boost)
            if cp["change_1d"]:
                logger.info("    1일 %+.1f%% 변동", cp["change_1d"])

    # ── 적중률 기록 ──
    update_accuracy_log(result["direction"], result["score"])

    # ── 적중률 표시 ──
    log_path = DATA_DIR / "market_sense_log.json"
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log_data = json.load(f)
        verified = [e for e in log_data if e.get("hit") is not None]
        if verified:
            hits = sum(1 for e in verified if e["hit"])
            logger.info("  적중률: %d/%d = %.1f%%", hits, len(verified), hits / len(verified) * 100)

    # ── 출력 ──
    output = {
        "date": date.today().isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "direction": result["direction"],
        "direction_kr": result["direction_kr"],
        "score": result["score"],
        "confidence": result["confidence"],
        "recommended_etf": result["recommended_etf"],
        "commodity_picks": result.get("commodity_picks", []),
        "stop_pct": result["stop_pct"],
        "target_pct": result["target_pct"],
        "signals": signals,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("  저장: %s", OUTPUT_PATH)

    # ── 텔레그램 ──
    if args.send and (result["direction"] != "NEUTRAL" or commodity_picks):
        try:
            from src.telegram_sender import send_message

            etf = result["recommended_etf"]
            etf_line = f"  추천: {etf['name']} ({etf['code']})" if etf else "  추천: 관망"

            # 원자재 추천 라인
            comm_lines = ""
            for cp in commodity_picks:
                arrow = "▲" if cp["direction"] == "LONG" else "▼"
                comm_lines += f"\n  {arrow} {cp['commodity']}: {cp['etf']['name']} (신뢰 {cp['confidence']}%)"

            msg = (
                f"🎯 <b>눈치 엔진</b> — {result['direction_kr']}\n\n"
                f"  종합 점수: {result['score']:+.1f} (신뢰도 {result['confidence']}%)\n"
                f"{etf_line}\n"
                f"  손절: {result['stop_pct']}% / 목표: +{result['target_pct']}%\n\n"
                f"  US야간: {signals['us']['grade']}\n"
                f"  VIX: {signals['vix']['level']}\n"
                f"  파생: {signals['derivatives']['grade']}\n"
                f"  레짐: {signals['regime']['current']} ({signals['regime']['transition']})\n"
                f"  NXT: {signals['nxt']['detail']}"
            )
            if comm_lines:
                msg += f"\n\n📦 <b>원자재 ETF</b>{comm_lines}"
            if signals.get("scenario", {}).get("detail", "없음") != "없음":
                msg += f"\n\n🎭 시나리오: {signals['scenario']['detail']}"

            send_message(msg)
            logger.info("  텔레그램 발송 완료")
        except Exception as e:
            logger.warning("  텔레그램 발송 실패: %s", e)


if __name__ == "__main__":
    main()
