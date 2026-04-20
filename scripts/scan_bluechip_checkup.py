#!/usr/bin/env python
"""대형주 체크업 + 소형주 테마 수혜주 — 한 페이지 통합 스캐너

주린이를 위한 "건강검진표":
  상단: 대형주 TOP 30 — 5축 스코어 + 등급 + 한줄 코멘트
  하단: 테마별 소형주 수혜주 — 대형주 산업에서 파생

5축 스코어 (각 20점, 총 100점):
  1) 밸류에이션: PER/PBR vs 업종 평균
  2) 실적 모멘텀: 컨센서스 상향/하향, forward PER
  3) 수주/이벤트: DART 공시 (수주, 계약, 자사주 등)
  4) 수급 흐름: 외인/기관 연속매수, 쌍끌이
  5) 기술적 위치: 피보나치 zone, 52주 대비 위치

종합 등급: A(80+) / B(60+) / C(40+) / D(<40)

스케줄: BAT-D G4.5 (장후)
출력: data/bluechip_checkup.json → quant_bluechip_checkup 테이블

Usage:
    python -u -X utf8 scripts/scan_bluechip_checkup.py
    python -u -X utf8 scripts/scan_bluechip_checkup.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "bluechip_checkup.json"


# ═══════════════════════════════════════════════════
# 데이터 로더
# ═══════════════════════════════════════════════════

def load_json(path: Path) -> dict | list | None:
    """JSON 파일 안전 로드."""
    if not path.exists():
        logger.warning("파일 없음: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("JSON 로드 실패: %s (%s)", path, e)
        return None


def load_fib_leaders() -> dict[str, dict]:
    """fib_scanner.json → 대형주 피보나치 데이터 (code → dict)."""
    data = load_json(DATA_DIR / "fib_scanner.json")
    if not data:
        return {}
    result = {}
    for s in data.get("fib_leaders", []):
        result[s["code"]] = s
    logger.info("피보나치 대형주: %d종목", len(result))
    return result


def load_consensus() -> dict[str, dict]:
    """consensus_screening.json → 컨센서스 데이터 (ticker → dict)."""
    data = load_json(DATA_DIR / "consensus_screening.json")
    if not data:
        return {}
    result = {}
    for pick in data.get("all_picks", data.get("top_picks", [])):
        ticker = pick.get("ticker", "")
        if ticker:
            result[ticker] = pick
    logger.info("컨센서스: %d종목", len(result))
    return result


def load_sector_map() -> dict[str, str]:
    """sector_map.json → 종목코드→섹터명 매핑.

    sector_map 구조: { "방산": {"etf_code": "...", "stocks": [{"code": "...", "name": "...", "weight": ...}]}, ... }
    한 종목이 여러 섹터에 속할 수 있음 → 구체적 업종 우선, 지수(KRX300 등) 후순위.
    """
    # 지수/그룹 이름 → 업종이 아닌 것들 (후순위)
    INDEX_SECTORS = {"코리아TOP10", "KRX300", "삼성그룹", "현대차그룹"}

    data = load_json(DATA_DIR / "sector_rotation" / "sector_map.json")
    if not data:
        return {}
    result = {}
    for sector_name, info in data.items():
        if not isinstance(info, dict):
            continue
        is_index = sector_name in INDEX_SECTORS
        for stock in info.get("stocks", []):
            code = stock.get("code", "")
            if not code:
                continue
            # 구체적 업종 우선: 이미 구체적 업종이 있으면 지수로 덮어쓰지 않음
            if code in result and is_index:
                continue
            result[code] = sector_name
    logger.info("섹터 매핑: %d종목 (%d개 섹터)", len(result), len(data))
    return result


def load_institutional_flow() -> dict[str, dict]:
    """accumulation_alert.json → 기관/외인 매집 시그널 (code → dict).

    핵심 필드: inst_consecutive, foreign_consecutive, dual_buying, sector, grade
    """
    data = load_json(DATA_DIR / "institutional_flow" / "accumulation_alert.json")
    if not data:
        return {}
    result = {}
    items = data.get("stock_alerts", []) if isinstance(data, dict) else data
    for item in items:
        code = item.get("ticker", item.get("code", ""))
        if code:
            result[code] = item
    logger.info("수급 매집: %d종목", len(result))
    return result


def load_dart_events() -> dict[str, list]:
    """dart_event_signals.json → DART 공시 이벤트 (code → [events])."""
    data = load_json(DATA_DIR / "dart_event_signals.json")
    if not data:
        return {}
    result: dict[str, list] = {}
    items = data if isinstance(data, list) else data.get("signals", data.get("events", []))
    for item in items:
        code = item.get("ticker", item.get("stock_code", ""))
        if code:
            result.setdefault(code, []).append(item)
    logger.info("DART 이벤트: %d종목", len(result))
    return result


def load_sector_rotation() -> list[dict]:
    """fib_scanner.json → 섹터 로테이션 데이터."""
    data = load_json(DATA_DIR / "fib_scanner.json")
    if not data:
        return []
    return data.get("sector_rotation", [])


# ═══════════════════════════════════════════════════
# 5축 스코어링 엔진
# ═══════════════════════════════════════════════════

def score_valuation(fib: dict, consensus: dict) -> tuple[int, str]:
    """밸류에이션 점수 (0~20)."""
    score = 10  # 기본점수
    reasons = []

    per = fib.get("per") or consensus.get("forward_per") or 0
    pbr = fib.get("pbr") or consensus.get("forward_pbr") or 0

    # PER 기반 (낮을수록 좋음)
    if 0 < per <= 8:
        score += 6
        reasons.append(f"PER {per:.1f} 저평가")
    elif 0 < per <= 12:
        score += 3
        reasons.append(f"PER {per:.1f} 적정")
    elif per > 25:
        score -= 4
        reasons.append(f"PER {per:.1f} 고평가")
    elif per > 15:
        score -= 1
        reasons.append(f"PER {per:.1f}")

    # PBR 기반
    if 0 < pbr <= 0.7:
        score += 4
        reasons.append(f"PBR {pbr:.2f} 저평가")
    elif 0 < pbr <= 1.0:
        score += 2
        reasons.append(f"PBR {pbr:.2f}")
    elif pbr > 3.0:
        score -= 2
        reasons.append(f"PBR {pbr:.2f} 고평가")

    return max(0, min(20, score)), " | ".join(reasons) if reasons else "데이터 부족"


def score_earnings(consensus: dict) -> tuple[int, str]:
    """실적 모멘텀 점수 (0~20)."""
    score = 10
    reasons = []

    upside = consensus.get("upside_pct", 0)
    analyst_count = consensus.get("analyst_count", 0)
    opinion = consensus.get("opinion_score", 0)

    # 컨센서스 상승여력
    if upside >= 30:
        score += 6
        reasons.append(f"목표가 괴리 +{upside:.0f}%")
    elif upside >= 15:
        score += 3
        reasons.append(f"목표가 괴리 +{upside:.0f}%")
    elif upside <= -10:
        score -= 4
        reasons.append(f"목표가 하회 {upside:.0f}%")
    elif upside <= 0:
        score -= 1
        reasons.append(f"목표가 근접 {upside:.0f}%")

    # 애널리스트 커버리지
    if analyst_count >= 20:
        score += 2
        reasons.append(f"애널 {analyst_count}명 (두터운 커버)")
    elif analyst_count >= 10:
        score += 1
    elif analyst_count <= 2 and analyst_count > 0:
        reasons.append(f"애널 {analyst_count}명 (관심 부족)")

    # 의견 점수
    if opinion >= 4.5:
        score += 2
        reasons.append("Strong Buy 의견")
    elif opinion >= 3.5:
        score += 1

    return max(0, min(20, score)), " | ".join(reasons) if reasons else "컨센서스 없음"


def score_events(dart_events: list) -> tuple[int, str, list[dict]]:
    """수주/이벤트 점수 (0~20).

    Returns:
        (score, reason_str, major_events): major_events는 UI 표시용 주요 이벤트 리스트
    """
    score = 10
    reasons = []
    major_events = []  # UI 아코디언 펼쳤을 때 표시

    # tier1 = 핵심(수주/자사주 등), tier2 = 관심, tier3 = 참고
    TIER_WEIGHT = {"tier1_즉시": 1.0, "tier2_관심": 0.5, "tier3_참고": 0.2}

    buy_score_sum = 0
    avoid_count = 0

    for evt in dart_events:
        action = evt.get("action", "")
        keyword = evt.get("keyword", evt.get("event", ""))
        tier = evt.get("tier", "tier3_참고")
        evt_score = evt.get("event_score", 10)
        weight = TIER_WEIGHT.get(tier, 0.2)

        if action == "BUY":
            buy_score_sum += evt_score * weight
            major_events.append({
                "event": keyword,
                "tier": tier,
                "score": evt_score,
                "url": evt.get("url", ""),
                "report": evt.get("report_nm", ""),
            })
        elif action == "AVOID":
            avoid_count += 1
            major_events.append({
                "event": keyword,
                "tier": tier,
                "score": -evt_score,
                "url": evt.get("url", ""),
                "report": evt.get("report_nm", ""),
            })

    # 가중 점수 → 0~10 범위로 변환 (tier1 수주 1건=15~20점 기대)
    if buy_score_sum >= 30:
        score += 8
        reasons.append(f"호재 공시 (가중 {buy_score_sum:.0f})")
    elif buy_score_sum >= 15:
        score += 5
        top_evt = major_events[0]["event"] if major_events else ""
        reasons.append(f"호재: {top_evt}")
    elif buy_score_sum >= 5:
        score += 2
        reasons.append("공시 관심")

    if avoid_count >= 2:
        score -= 6
        reasons.append(f"악재 공시 {avoid_count}건")
    elif avoid_count >= 1:
        score -= 3
        reasons.append("악재 공시 주의")

    if not dart_events:
        reasons.append("최근 공시 없음")

    # 주요 이벤트만 상위 3건 (tier1 우선)
    major_events.sort(key=lambda x: abs(x["score"]), reverse=True)
    major_events = major_events[:3]

    return max(0, min(20, score)), " | ".join(reasons) if reasons else "공시 없음", major_events


def score_supply_demand(flow: dict) -> tuple[int, str]:
    """수급 흐름 점수 (0~20).

    flow: accumulation_alert.json stock_alerts 항목
      - foreign_consecutive: 외인 연속매수일 (음수=매도)
      - inst_consecutive: 기관 연속매수일
      - dual_buying: True/False
    """
    score = 10
    reasons = []

    # 외국인 연속매수
    foreign_days = flow.get("foreign_consecutive", 0) or 0
    if foreign_days >= 10:
        score += 5
        reasons.append(f"외인 {foreign_days}일 연속매수")
    elif foreign_days >= 5:
        score += 3
        reasons.append(f"외인 {foreign_days}일 매수")
    elif foreign_days <= -5:
        score -= 3
        reasons.append(f"외인 {abs(foreign_days)}일 매도")

    # 기관 매집
    inst_days = flow.get("inst_consecutive", 0) or 0
    if inst_days >= 10:
        score += 5
        reasons.append(f"기관 {inst_days}일 연속매수")
    elif inst_days >= 5:
        score += 3
        reasons.append(f"기관 {inst_days}일 매수")
    elif inst_days <= -5:
        score -= 3
        reasons.append(f"기관 {abs(inst_days)}일 매도")

    # 쌍끌이 보너스
    if flow.get("dual_buying"):
        score += 2
        reasons.append("쌍끌이 매수!")
    elif foreign_days >= 3 and inst_days >= 3:
        score += 2
        reasons.append("쌍끌이 매수!")

    if not reasons:
        reasons.append("수급 데이터 부족")

    return max(0, min(20, score)), " | ".join(reasons)


def score_technical(fib: dict) -> tuple[int, str]:
    """기술적 위치 점수 (0~20)."""
    score = 10
    reasons = []

    zone = fib.get("fib_zone", "")
    drop = fib.get("drop", 0)
    position = fib.get("position_pct", 50)
    upside = fib.get("upside", 0)

    # 피보나치 zone
    if zone == "DEEP":
        score += 5
        reasons.append(f"DEEP 구간 (하락 {drop:+.1f}%)")
    elif zone == "MID":
        score += 3
        reasons.append(f"MID 구간 (하락 {drop:+.1f}%)")
    elif zone == "MILD":
        score += 1
        reasons.append(f"MILD 구간 ({drop:+.1f}%)")
    elif zone == "NEAR_HIGH":
        score -= 3
        reasons.append(f"고점 근접 ({drop:+.1f}%)")

    # 위치 백분율 (낮을수록 바닥 근접)
    if position <= 25:
        score += 3
        reasons.append(f"52주 바닥권 ({position:.0f}%)")
    elif position >= 80:
        score -= 2
        reasons.append(f"52주 고점권 ({position:.0f}%)")

    # 상승여력
    if upside >= 50:
        score += 2
        reasons.append(f"목표 상승여력 +{upside:.0f}%")

    return max(0, min(20, score)), " | ".join(reasons) if reasons else "피보나치 데이터 없음"


# ═══════════════════════════════════════════════════
# 등급 판정 + 한줄 코멘트
# ═══════════════════════════════════════════════════

def classify_grade(total: int) -> str:
    """총점 → 등급."""
    if total >= 80:
        return "A"
    elif total >= 60:
        return "B"
    elif total >= 40:
        return "C"
    else:
        return "D"


def generate_comment(grade: str, scores: dict, fib: dict, consensus: dict) -> str:
    """한줄 코멘트 생성."""
    parts = []

    # 가장 높은 축
    axis_names = {"valuation": "저평가", "earnings": "실적 호조", "events": "호재 공시",
                  "supply_demand": "수급 양호", "technical": "기술적 매력"}
    best_axis = max(scores, key=lambda k: scores[k]["score"])
    if scores[best_axis]["score"] >= 15:
        parts.append(axis_names.get(best_axis, ""))

    # 가장 낮은 축 (경고)
    worst_axis = min(scores, key=lambda k: scores[k]["score"])
    axis_warnings = {"valuation": "밸류 부담", "earnings": "실적 불안", "events": "공시 리스크",
                     "supply_demand": "수급 이탈", "technical": "차트 약세"}
    if scores[worst_axis]["score"] <= 8:
        parts.append(f"{axis_warnings.get(worst_axis, '')} 주의")

    # 상승여력
    upside = consensus.get("upside_pct", fib.get("upside", 0))
    if upside >= 20:
        parts.append(f"목표가 +{upside:.0f}%")

    if not parts:
        if grade in ("A", "B"):
            parts.append("전반적 양호")
        else:
            parts.append("관망 추천")

    return ", ".join(parts)


# ═══════════════════════════════════════════════════
# 메인 스캔
# ═══════════════════════════════════════════════════

def scan_bluechip(top_n: int = 30) -> list[dict]:
    """대형주 TOP N 체크업 스캔."""
    fib_leaders = load_fib_leaders()
    consensus_data = load_consensus()
    institutional_data = load_institutional_flow()
    dart_data = load_dart_events()
    sector_map = load_sector_map()

    if not fib_leaders:
        logger.error("피보나치 대형주 데이터 없음")
        return []

    results = []

    for code, fib in fib_leaders.items():
        cons = consensus_data.get(code, {})
        flow = institutional_data.get(code, {})
        dart = dart_data.get(code, [])

        # 5축 스코어링
        val_score, val_reason = score_valuation(fib, cons)
        ear_score, ear_reason = score_earnings(cons)
        evt_score, evt_reason, major_events = score_events(dart)
        sd_score, sd_reason = score_supply_demand(flow)
        tech_score, tech_reason = score_technical(fib)

        total = val_score + ear_score + evt_score + sd_score + tech_score
        grade = classify_grade(total)

        scores = {
            "valuation": {"score": val_score, "reason": val_reason},
            "earnings": {"score": ear_score, "reason": ear_reason},
            "events": {"score": evt_score, "reason": evt_reason},
            "supply_demand": {"score": sd_score, "reason": sd_reason},
            "technical": {"score": tech_score, "reason": tech_reason},
        }

        comment = generate_comment(grade, scores, fib, cons)

        # 산업 태그: sector_map → accumulation_alert → fib (우선순위)
        sector = sector_map.get(code, "") or flow.get("sector", "") or fib.get("sector", "")

        results.append({
            "code": code,
            "name": fib.get("name", ""),
            "sector": sector,
            "cap": fib.get("cap", 0),
            "price": fib.get("price", 0),
            "grade": grade,
            "total_score": total,
            "comment": comment,
            # 5축 상세
            "scores": scores,
            # 핵심 지표 (접힌 상태 표시용)
            "target_price": cons.get("target_price", fib.get("target", 0)),
            "upside_pct": cons.get("upside_pct", fib.get("upside", 0)),
            "per": fib.get("per", 0) or cons.get("forward_per", 0),
            "pbr": fib.get("pbr", 0) or cons.get("forward_pbr", 0),
            "fib_zone": fib.get("fib_zone", ""),
            "drop_52w": fib.get("drop", 0),
            "position_pct": fib.get("position_pct", 0),
            # DART 주요 공시 (아코디언 펼칠 때 표시)
            "major_events": major_events,
            # 해외 시각 (P2에서 추가)
            "global_view": None,
        })

    # 등급순 → 총점순 정렬
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["total_score"]))
    logger.info("대형주 체크업: %d종목", len(results))
    return results[:top_n]


def scan_theme_smallcap(bluechips: list[dict], sector_rotation: list[dict]) -> list[dict]:
    """대형주 산업에서 파생된 소형주 테마 수혜주.

    1단계: 기존 데이터(fib_stocks 눌림목 + sector_map)로 소형주 추출
    2단계(P3): Perplexity로 테마→종목 자동 매핑 추가
    """
    # 대형주에서 핫 섹터 추출
    hot_sectors = set()
    for bc in bluechips:
        if bc["grade"] in ("A", "B") and bc["sector"]:
            hot_sectors.add(bc["sector"])

    # 섹터 로테이션에서 선도/추격 섹터 추가
    for sr in sector_rotation:
        if sr.get("quadrant") in ("선도", "추격"):
            hot_sectors.add(sr.get("sector", ""))

    # 지수(코리아TOP10, KRX300 등)는 업종이 아니므로 제외
    INDEX_NAMES = {"코리아TOP10", "KRX300", "삼성그룹", "현대차그룹"}
    hot_sectors -= INDEX_NAMES
    hot_sectors.discard("")
    logger.info("핫 섹터: %s", hot_sectors)

    if not hot_sectors:
        logger.warning("핫 섹터 없음 — 섹터 매핑 확인 필요")
        return []

    # sector_map으로 종목별 섹터 확인
    sector_map = load_sector_map()
    institutional_data = load_institutional_flow()

    # fib_stocks (눌림목)에서 소형주 필터링
    fib_data = load_json(DATA_DIR / "fib_scanner.json")
    if not fib_data:
        return []

    consensus_data = load_consensus()

    smallcaps = []
    for s in fib_data.get("fib_stocks", []):
        cap = s.get("cap", 0)
        code = s.get("code", "")

        # 소형주: 시총 1,000억 ~ 30,000억 (중소형)
        if cap < 1000 or cap > 30000:
            continue

        # 섹터: sector_map → accumulation_alert → fib (우선순위)
        flow = institutional_data.get(code, {})
        sector = sector_map.get(code, "") or flow.get("sector", "") or s.get("sector", "")

        # 핫 섹터 매칭
        if sector not in hot_sectors:
            continue

        cons = consensus_data.get(code, {})

        # 수급 시그널
        foreign_days = flow.get("foreign_consecutive", 0) or 0
        inst_days = flow.get("inst_consecutive", 0) or 0
        if flow.get("dual_buying"):
            supply_signal = "쌍끌이"
        elif foreign_days >= 3 or inst_days >= 3:
            supply_signal = "매집"
        else:
            supply_signal = "관망"

        smallcaps.append({
            "code": code,
            "name": s.get("name", ""),
            "sector": sector,
            "theme": sector,
            "cap": cap,
            "price": s.get("price", 0),
            "drop_52w": s.get("drop", 0),
            "fib_zone": s.get("fib_zone", ""),
            "upside_pct": cons.get("upside_pct", s.get("upside", 0)),
            "target_price": cons.get("target_price", s.get("target", 0)),
            "supply_signal": supply_signal,
            "per": s.get("per", 0),
            "pbr": s.get("pbr", 0),
        })

    # 테마별 그룹핑 + upside 순 정렬
    smallcaps.sort(key=lambda x: (x["theme"], -x.get("upside_pct", 0)))
    logger.info("테마 소형주: %d종목 (%d개 섹터)", len(smallcaps), len(hot_sectors))
    return smallcaps


def load_universe_caps() -> dict[str, float]:
    """universe.csv → {ticker: 시총(억원)}."""
    csv_path = DATA_DIR / "universe.csv"
    if not csv_path.exists():
        return {}
    result = {}
    import csv
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "")
            cap_raw = row.get("market_cap", "0")
            try:
                result[ticker] = float(cap_raw) / 1e8  # 원→억원
            except (ValueError, TypeError):
                pass
    return result


def fetch_theme_smallcaps(hot_sectors: set[str], existing_codes: set[str]) -> list[dict]:
    """Perplexity로 핫 섹터의 테마 수혜 소형주를 추가 발굴.

    Args:
        hot_sectors: 대형주에서 도출된 핫 섹터 (방산, 2차전지 등)
        existing_codes: 이미 추출된 소형주 코드 (중복 방지, 대형주 코드 포함)
    Returns:
        추가 소형주 리스트 (시총 검증 통과 종목만)
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        logger.warning("[테마매핑] PERPLEXITY_API_KEY 미설정")
        return []

    if not hot_sectors:
        return []

    sectors_str = ", ".join(sorted(hot_sectors))
    prompt = f"""한국 주식시장에서 아래 핫 섹터별로 수혜가 예상되는 **소형주** (시총 1000~30000억)를 각 섹터별 3종목씩 추천해주세요.

핫 섹터: {sectors_str}

조건:
- 대형주가 아닌 중소형주만 (코스피200 제외)
- 해당 섹터 대형주의 밸류체인, 부품/소재 공급사, 또는 기술 수혜주
- 최근 산업 트렌드에서 수혜 가능한 종목

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "themes": [
    {{
      "sector": "방산",
      "stocks": [
        {{
          "code": "종목코드 6자리",
          "name": "종목명",
          "reason": "수혜 이유 1줄"
        }}
      ]
    }}
  ]
}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "한국 주식시장 섹터별 소형주 전문가입니다. "
                    "밸류체인 분석으로 대형주 산업의 수혜 소형주를 찾습니다. "
                    "반드시 JSON 형식으로만 응답하세요."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }

    try:
        import requests
        resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # JSON 추출: 첫 번째 { ... } 블록만 파싱
        start = content.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i, ch in enumerate(content[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            content = content[start:end]

        data = json.loads(content)

        # 시총 검증용 데이터 로드
        universe_caps = load_universe_caps()
        MAX_CAP = 30000  # 억원 (소형주 상한)
        MIN_CAP = 500    # 억원 (너무 작은 종목 제외)

        result = []
        skipped = 0
        for theme in data.get("themes", []):
            sector = theme.get("sector", "")
            for s in theme.get("stocks", []):
                code = s.get("code", "")
                if not code or code in existing_codes:
                    continue
                # 시총 검증: universe.csv에 있으면 시총 확인
                cap = universe_caps.get(code, 0)
                if cap > MAX_CAP:
                    logger.info("[테마매핑] 시총 초과 제외: %s(%s) 시총 %.0f억",
                                s.get("name", ""), code, cap)
                    skipped += 1
                    continue
                result.append({
                    "code": code,
                    "name": s.get("name", ""),
                    "sector": sector,
                    "theme": sector,
                    "cap": cap if cap > 0 else 0,
                    "price": 0,
                    "drop_52w": 0,
                    "fib_zone": "",
                    "upside_pct": 0,
                    "target_price": 0,
                    "supply_signal": "AI추천",
                    "per": 0,
                    "pbr": 0,
                    "ai_reason": s.get("reason", ""),
                })
        logger.info("[테마매핑] Perplexity %d종목 추가 (%d종목 시총초과 제외)", len(result), skipped)
        return result

    except Exception as e:
        logger.error("[테마매핑] Perplexity 호출 실패: %s", e)
        return []


# ═══════════════════════════════════════════════════
# 해외 시각 수집 (Perplexity Sonar)
# ═══════════════════════════════════════════════════

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


# 주요 대형주 영문 종목명 매핑 (Perplexity 검색 정확도 향상)
KR_TO_EN = {
    "삼성전자": "Samsung Electronics",
    "SK하이닉스": "SK Hynix",
    "현대차": "Hyundai Motor",
    "기아": "Kia Corporation",
    "삼성바이오로직스": "Samsung Biologics",
    "셀트리온": "Celltrion",
    "LG에너지솔루션": "LG Energy Solution",
    "POSCO홀딩스": "POSCO Holdings",
    "KB금융": "KB Financial Group",
    "신한지주": "Shinhan Financial Group",
    "하나금융지주": "Hana Financial Group",
    "삼성SDI": "Samsung SDI",
    "LG화학": "LG Chem",
    "현대모비스": "Hyundai Mobis",
    "NAVER": "NAVER Corp",
    "카카오": "Kakao Corp",
    "삼성물산": "Samsung C&T",
    "한국전력": "KEPCO",
    "SK이노베이션": "SK Innovation",
    "S-Oil": "S-Oil Corporation",
    "한화에어로스페이스": "Hanwha Aerospace",
    "HD현대중공업": "HD Hyundai Heavy Industries",
    "현대로템": "Hyundai Rotem",
    "한화오션": "Hanwha Ocean",
    "두산에너빌리티": "Doosan Enerbility",
    "HD한국조선해양": "HD Korea Shipbuilding",
    "LG전자": "LG Electronics",
    "SK텔레콤": "SK Telecom",
    "KT": "KT Corp",
    "크래프톤": "Krafton",
    "삼성전기": "Samsung Electro-Mechanics",
    "삼성생명": "Samsung Life Insurance",
    "HMM": "HMM Co Ltd",
    "LS에코에너지": "LS Eco Energy",
    "HD현대일렉트릭": "HD Hyundai Electric",
    "한미반도체": "Hanmi Semiconductor",
    "리노공업": "LEENO Industrial",
}


def fetch_global_views(bluechips: list[dict], top_n: int = 10) -> dict[str, dict]:
    """Perplexity Sonar로 상위 대형주의 해외 시각 수집.

    영문 종목명 포함 → 해외 소스 검색 정확도 향상.
    Returns: {code: {"summary": "...", "sentiment": "positive/neutral/negative", "source": "..."}}
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        logger.warning("[해외시각] PERPLEXITY_API_KEY 미설정")
        return {}

    # 상위 N종목만 (등급순 정렬된 상태)
    targets = bluechips[:top_n]
    # 영문명 포함: "삼성전자/Samsung Electronics(005930)"
    stock_lines = []
    for b in targets:
        en_name = KR_TO_EN.get(b["name"], "")
        if en_name:
            stock_lines.append(f"{b['name']} / {en_name} ({b['code']})")
        else:
            stock_lines.append(f"{b['name']} ({b['code']})")
    stock_list = "\n".join(f"- {line}" for line in stock_lines)

    prompt = f"""Analyze recent overseas (English-language) investment views on these Korean large-cap stocks.
For each stock, summarize analyst/institutional/media opinions in 1-2 sentences (respond in Korean).

Stocks:
{stock_list}

Respond ONLY in this JSON format:
{{
  "stocks": [
    {{
      "code": "005930",
      "name": "삼성전자",
      "summary": "해외 시각 1~2줄 요약 (한국어로)",
      "sentiment": "positive 또는 neutral 또는 negative",
      "source": "출처 (Bloomberg, Reuters 등)"
    }}
  ]
}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a global equity analyst specializing in Korean stocks. "
                    "Search for recent English-language analyst reports, news, and opinions "
                    "about the given Korean stocks. Use Bloomberg, Reuters, Seeking Alpha, "
                    "Goldman Sachs, Morgan Stanley, JP Morgan sources when available. "
                    "Summarize findings in Korean. Output pure JSON only, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 3000,
    }

    try:
        import requests
        resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # JSON 파싱 (코드블록 제거 + 안전 추출)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # 안전한 JSON 추출: 첫 번째 { ... } 블록
        start = content.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i, ch in enumerate(content[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            content = content[start:end]

        data = json.loads(content)
        result = {}
        for s in data.get("stocks", []):
            code = s.get("code", "")
            if code:
                result[code] = {
                    "summary": s.get("summary", ""),
                    "sentiment": s.get("sentiment", "neutral"),
                    "source": s.get("source", ""),
                }
        logger.info("[해외시각] %d종목 수집 완료", len(result))
        return result

    except Exception as e:
        logger.error("[해외시각] Perplexity 호출 실패: %s", e)
        return {}


# ═══════════════════════════════════════════════════
# 업로드 + 출력
# ═══════════════════════════════════════════════════

def upload_checkup(bluechips: list, smallcaps: list, date_str: str = "") -> bool:
    """quant_bluechip_checkup 테이블에 업로드."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "bluechips": bluechips,
        "smallcaps": smallcaps,
        "summary": {
            "bluechip_count": len(bluechips),
            "smallcap_count": len(smallcaps),
            "grades": {
                g: sum(1 for b in bluechips if b["grade"] == g)
                for g in ["A", "B", "C", "D"]
            },
            "themes": list({s["theme"] for s in smallcaps}),
        },
    }

    try:
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        if not uploader.is_active:
            logger.warning("[체크업] Supabase 미연결")
            return False
        row = {"date": date_str, "data": payload}
        uploader.client.table("quant_bluechip_checkup").upsert(
            row, on_conflict="date"
        ).execute()
        logger.info("[체크업] 업로드 완료: %s (대형주 %d, 소형주 %d)",
                    date_str, len(bluechips), len(smallcaps))
        return True
    except Exception as e:
        logger.error("[체크업] 업로드 오류: %s", e)
        return False


def print_report(bluechips: list, smallcaps: list):
    """콘솔 출력."""
    print(f"\n{'='*70}")
    print(f"  대형주 체크업 + 소형주 테마 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # 등급 요약
    grades = {}
    for b in bluechips:
        g = b["grade"]
        grades[g] = grades.get(g, 0) + 1
    grade_str = " | ".join(f"{g}: {c}종목" for g, c in sorted(grades.items()))
    print(f"\n  [대형주 TOP {len(bluechips)}] {grade_str}")

    grade_emoji = {"A": "🟢", "B": "🟡", "C": "🔴", "D": "⚫"}

    for b in bluechips:
        emoji = grade_emoji.get(b["grade"], "⚪")
        cap_조 = b["cap"] / 10000
        target_str = f"목표 {b['target_price']:,}" if b["target_price"] else ""
        print(
            f"    {emoji}{b['grade']} {b['name']:12s} {b['price']:>8,}원 "
            f"({b['total_score']:2d}점) {target_str}"
        )
        print(f"       → {b['comment']}")

        # 5축 바 표시
        axes = ["valuation", "earnings", "events", "supply_demand", "technical"]
        labels = ["밸류", "실적", "수주", "수급", "위치"]
        bars = []
        for ax, lb in zip(axes, labels):
            s = b["scores"][ax]["score"]
            dots = "●" * (s // 4) + "○" * (5 - s // 4)
            bars.append(f"{lb}{dots}")
        print(f"       {' '.join(bars)}")

        # 해외 시각
        gv = b.get("global_view")
        if gv:
            sent_icon = {"positive": "+", "negative": "-", "neutral": "="}.get(gv["sentiment"], "?")
            print(f"       [{sent_icon}해외] {gv['summary'][:60]}")

    # 소형주
    if smallcaps:
        themes = {}
        for s in smallcaps:
            themes.setdefault(s["theme"], []).append(s)

        print(f"\n  [테마 소형주] {len(smallcaps)}종목")
        for theme, stocks in themes.items():
            print(f"\n    💎 {theme}")
            for s in stocks[:5]:
                if s.get("ai_reason"):
                    # Perplexity AI 추���
                    print(f"      {s['name']:14s} [{s['supply_signal']}] {s['ai_reason'][:50]}")
                else:
                    print(
                        f"      {s['name']:14s} {s['price']:>8,}원 "
                        f"시총{s['cap']:,.0f}억 하락{s['drop_52w']:+.1f}% "
                        f"{s['fib_zone']} {s['supply_signal']}"
                    )
            if len(stocks) > 5:
                print(f"      ... 외 {len(stocks) - 5}종목")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="대형주 체크업 + 소형주 테마")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 출력만")
    parser.add_argument("--no-global", action="store_true", help="해외 시각 수집 건너뜀")
    parser.add_argument("--top", type=int, default=30, help="대형주 수 (기본 30)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n[체크업] 스캔 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 대형주 체크업
    bluechips = scan_bluechip(top_n=args.top)
    if not bluechips:
        print("[체크업] 대형주 데이터 없음 — fib_scanner.json 확인 필요")
        return

    # 섹터 로테이션
    sector_rotation = load_sector_rotation()

    # 소형주 테마 수혜주
    smallcaps = scan_theme_smallcap(bluechips, sector_rotation)

    # Perplexity 테마 매핑 (기존 소형주에 추가)
    if not args.no_global:
        hot_sectors = {b["sector"] for b in bluechips if b["grade"] in ("A", "B") and b["sector"]}
        for sr in sector_rotation:
            if sr.get("quadrant") in ("선도", "추격"):
                hot_sectors.add(sr.get("sector", ""))
        INDEX_NAMES = {"코리아TOP10", "KRX300", "삼성그룹", "현대차그룹"}
        hot_sectors -= INDEX_NAMES
        hot_sectors.discard("")

        # 대형주 코드 + 이미 추출된 소형주 코드 → 중복/대형주 오추천 방지
        existing_codes = {s["code"] for s in smallcaps} | {b["code"] for b in bluechips}
        ai_smallcaps = fetch_theme_smallcaps(hot_sectors, existing_codes)
        if ai_smallcaps:
            smallcaps.extend(ai_smallcaps)
            print(f"\n[체크업] AI 테마 소형주: {len(ai_smallcaps)}종목 추가")

    # 해외 시각 수집 (Perplexity) — 상위 10종목
    if not args.no_global:
        global_views = fetch_global_views(bluechips, top_n=10)
        for b in bluechips:
            view = global_views.get(b["code"])
            if view:
                b["global_view"] = view
        if global_views:
            print(f"\n[체크업] 해외 시각: {len(global_views)}종목 수집")

    # 콘솔 출력
    print_report(bluechips, smallcaps)

    # JSON 저장
    date_str = datetime.now().strftime("%Y-%m-%d")
    report = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "bluechips": bluechips,
        "smallcaps": smallcaps,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[체크업] JSON 저장: {OUTPUT_PATH}")

    # 업로드
    if not args.dry_run:
        upload_checkup(bluechips, smallcaps, date_str)

    print(f"\n[체크업] 완료 — 대형주 {len(bluechips)} + 소형주 {len(smallcaps)}")


if __name__ == "__main__":
    main()
