"""SD V2 수급 패턴 전환 추적기

일별 패턴을 저장하고, 주간 전환을 분석한다.
핵심 전환:
  A→F: 매집 포기 → 위험 (스마트머니가 이탈)
  F→D: 바닥 탈출 초기 신호 (방향 전환)
  D→A: 매집 본격화 (최고 매수 기회)

데이터: data/sd_pattern_daily/{YYYY-MM-DD}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATTERN_DIR = PROJECT_ROOT / "data" / "sd_pattern_daily"


def save_daily_patterns(
    date_str: str,
    patterns: dict[str, dict],
) -> Path:
    """일별 패턴을 저장한다.

    Args:
        date_str: "YYYY-MM-DD"
        patterns: {ticker: {"name": ..., "pattern": "A", "sd_score": 0.78,
                            "foreign_net_20d": 123, "inst_net_20d": -45, ...}}
    Returns:
        저장된 파일 경로
    """
    PATTERN_DIR.mkdir(parents=True, exist_ok=True)
    path = PATTERN_DIR / f"{date_str}.json"

    data = {
        "date": date_str,
        "count": len(patterns),
        "distribution": _count_distribution(patterns),
        "patterns": patterns,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("[SD Tracker] %s: %d종목 저장 → %s", date_str, len(patterns), path)
    return path


def _count_distribution(patterns: dict) -> dict[str, int]:
    """패턴 분포 카운트."""
    dist: dict[str, int] = {}
    for p in patterns.values():
        pat = p.get("pattern", "X")
        dist[pat] = dist.get(pat, 0) + 1
    return dist


def get_transitions(
    today_str: str,
    days_back: int = 7,
) -> dict:
    """오늘 vs N일 전 패턴을 비교하여 전환 목록을 반환한다.

    Returns:
        {
            "compared_dates": {"today": "2026-03-23", "past": "2026-03-16"},
            "transitions": [
                {"ticker": "005930", "name": "삼성전자",
                 "from": "C", "to": "F", "type": "danger",
                 "from_name": "추세확인", "to_name": "물림"},
                ...
            ],
            "summary": {
                "danger": [...],   # A/B/C/D → F
                "recovery": [...], # F → D/A/B
                "accumulation": [...],  # D/C/X → A/B
                "retreat": [...],  # A/B → C/D
            }
        }
    """
    today_path = PATTERN_DIR / f"{today_str}.json"
    if not today_path.exists():
        logger.warning("[SD Tracker] 오늘 패턴 없음: %s", today_str)
        return {"compared_dates": {}, "transitions": [], "summary": {}}

    # N일 전 데이터 찾기 (주말/공휴일 고려: 가장 가까운 과거 데이터)
    past_data = None
    past_date = None
    for d in range(days_back, days_back + 5):
        target = datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=d)
        target_str = target.strftime("%Y-%m-%d")
        target_path = PATTERN_DIR / f"{target_str}.json"
        if target_path.exists():
            with open(target_path, encoding="utf-8") as f:
                past_data = json.load(f)
            past_date = target_str
            break

    if past_data is None:
        logger.info("[SD Tracker] %d일 전 데이터 없음 — 전환 분석 불가", days_back)
        return {"compared_dates": {}, "transitions": [], "summary": {}}

    with open(today_path, encoding="utf-8") as f:
        today_data = json.load(f)

    today_patterns = today_data.get("patterns", {})
    past_patterns = past_data.get("patterns", {})

    # 전환 감지
    transitions = []
    for ticker, cur in today_patterns.items():
        prev = past_patterns.get(ticker)
        if prev is None:
            continue
        from_pat = prev.get("pattern", "X")
        to_pat = cur.get("pattern", "X")
        if from_pat == to_pat or from_pat == "X" or to_pat == "X":
            continue
        transitions.append({
            "ticker": ticker,
            "name": cur.get("name", ticker),
            "from": from_pat,
            "to": to_pat,
            "from_name": prev.get("pattern_name", ""),
            "to_name": cur.get("pattern_name", ""),
            "type": _classify_transition(from_pat, to_pat),
            "foreign_net_20d": cur.get("foreign_net_20d", 0),
            "inst_net_20d": cur.get("inst_net_20d", 0),
        })

    # 분류
    summary: dict[str, list] = {
        "danger": [],
        "recovery": [],
        "accumulation": [],
        "retreat": [],
    }
    for t in transitions:
        cat = t["type"]
        if cat in summary:
            summary[cat].append(t)

    return {
        "compared_dates": {"today": today_str, "past": past_date},
        "transitions": transitions,
        "summary": summary,
        "distribution_today": today_data.get("distribution", {}),
        "distribution_past": past_data.get("distribution", {}),
    }


def _classify_transition(from_pat: str, to_pat: str) -> str:
    """전환 유형 분류."""
    if to_pat == "F":
        return "danger"      # → 물림 (위험)
    if from_pat == "F" and to_pat in ("A", "B", "D"):
        return "recovery"    # 물림 → 탈출
    if to_pat in ("A", "B") and from_pat not in ("A", "B"):
        return "accumulation"  # → 매집 시작
    if from_pat in ("A", "B") and to_pat not in ("A", "B"):
        return "retreat"     # 매집 → 후퇴
    return "other"


def format_transition_report(result: dict) -> str:
    """전환 분석 결과를 텔레그램 메시지로 포맷."""
    dates = result.get("compared_dates", {})
    if not dates:
        return ""

    summary = result.get("summary", {})
    dist_today = result.get("distribution_today", {})
    dist_past = result.get("distribution_past", {})

    lines = [
        f"\U0001f4ca [주간 수급 전환 리포트]",
        f"기간: {dates.get('past', '?')} → {dates.get('today', '?')}",
        "━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    # 분포 변화
    lines.append("📊 패턴 분포 변화:")
    for pat in ["A", "B", "C", "D", "F"]:
        prev_cnt = dist_past.get(pat, 0)
        cur_cnt = dist_today.get(pat, 0)
        diff = cur_cnt - prev_cnt
        diff_str = f"({diff:+d})" if diff != 0 else ""
        lines.append(f"  {pat}: {prev_cnt} → {cur_cnt} {diff_str}")
    lines.append("")

    # 위험 전환 (→ F)
    danger = summary.get("danger", [])
    if danger:
        lines.append(f"\U0001f534 물림 전환 ({len(danger)}건) — 매도 검토!")
        for t in danger[:10]:
            lines.append(f"  {t['name']}: {t['from']}→F 외{t['foreign_net_20d']:+,.0f}억 기{t['inst_net_20d']:+,.0f}억")
        if len(danger) > 10:
            lines.append(f"  ... 외 {len(danger) - 10}건")
        lines.append("")

    # 탈출 (F → A/B/D)
    recovery = summary.get("recovery", [])
    if recovery:
        lines.append(f"\U0001f7e2 물림 탈출 ({len(recovery)}건) — 관심 종목!")
        for t in recovery[:10]:
            lines.append(f"  {t['name']}: F→{t['to']}({t['to_name']})")
        lines.append("")

    # 매집 시작 (→ A/B)
    accum = summary.get("accumulation", [])
    if accum:
        lines.append(f"\U0001f535 매집 시작 ({len(accum)}건)")
        for t in accum[:10]:
            lines.append(f"  {t['name']}: {t['from']}→{t['to']}({t['to_name']})")
        lines.append("")

    # 매집 후퇴 (A/B → C/D)
    retreat = summary.get("retreat", [])
    if retreat:
        lines.append(f"\U0001f7e0 매집 후퇴 ({len(retreat)}건)")
        for t in retreat[:10]:
            lines.append(f"  {t['name']}: {t['from']}→{t['to']}({t['to_name']})")
        lines.append("")

    total = len(result.get("transitions", []))
    if total == 0:
        lines.append("전환 없음 (패턴 안정)")
    else:
        lines.append(f"총 {total}건 전환 감지")

    return "\n".join(lines)
