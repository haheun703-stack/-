"""DART 공시 통합 + 우리 자체 수급 교차 검증.

배경 (5/22 퐝가님 인사이트):
  하나마이크론 5%룰 사례:
  - 정보봇 DART 캡처 → 5%룰 공시 (보고자 X, 방향 미상)
  - 단순 공시만 보면 매수/매도 미상 → 잘못된 시그널 가능
  - 우리 자체 외인+기관 5d로 진짜 수급 변동 검증

교차 검증:
  1. 정보봇 intelligence_disclosures 조회 (오늘~7일 내)
  2. 종목별 severity 가중 (CRITICAL +2, WARNING +1, INFO 0)
  3. 우리 자체 외인+기관 5d 매수 검증:
     - 외인+기관 둘 다 매수 + DART 공시 = 매수 보고자 추정 → 점수 ×1.5
     - 외인 또는 기관 매수 + DART 공시 = 일부 매수 → 점수 ×1.0
     - 외인+기관 매도 + DART 공시 = 매도 보고자 추정 → 점수 ×-1 (악재)

DART 공시 sentiment + tags 활용:
  - sentiment_score ≥ +0.5: 호재 추정
  - sentiment_score ≤ -0.5: 악재 추정
  - tags에 'PPA수혜주' 또는 '메모리' 포함: 가중치 +1

사용:
  from src.use_cases.dart_signal_scorer import calculate_dart_score
  result = calculate_dart_score("067310")  # 하나마이크론
  # → {score: +2, severity: "WARNING", verified: True, reason: "..."}
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JGIS_OHLCV_DIR = Path("/home/ubuntu/jgis/stock_data_daily")

# 임계
DART_LOOKBACK_DAYS = 7        # 최근 7일 공시 추적
SEVERITY_SCORES = {
    "CRITICAL": 2,
    "WARNING": 1,
    "INFO": 0,
}
SENTIMENT_BOOST = 0.5         # ± 0.5 임계로 호재/악재 판정


def _fetch_recent_disclosures(ticker: str, days: int = DART_LOOKBACK_DAYS) -> list[dict]:
    """정보봇 intelligence_disclosures 조회 (최근 N일)."""
    try:
        from src.adapters.quant_supabase_reader import _get_client
        client = _get_client()
        if not client:
            return []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        res = (
            client.table("intelligence_disclosures")
            .select(
                "ticker,date,severity,sentiment_score,tags,"
                "ai_summary,ai_impact,disclosed_at,original_url"
            )
            .eq("ticker", str(ticker).zfill(6))
            .gte("date", cutoff)
            .order("date", desc=True)
            .limit(5)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.debug("DART 공시 조회 실패 %s: %s", ticker, e)
        return []


def _verify_supply_via_jgis(ticker: str) -> tuple[float, dict]:
    """jgis OHLCV로 외인+기관 5일 누적 수급 검증.

    Returns:
        (multiplier, details)
        multiplier:
            둘 다 매수: +1.5 (매수 보고자 추정)
            한쪽 매수: +1.0 (일부 매수)
            둘 다 매도: -1.0 (매도 보고자 추정, 악재로 전환)
            데이터 없음: 0
    """
    import csv
    matches = list(JGIS_OHLCV_DIR.glob(f"*_{str(ticker).zfill(6)}.csv"))
    if not matches:
        return 0, {"reason": "jgis 데이터 없음"}
    try:
        with open(matches[0], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 5:
            return 0, {"reason": "OHLCV 데이터 부족"}
        recent_5 = rows[-5:]
        fg_5d = sum(float(r.get("Foreign_Net", 0) or 0) for r in recent_5)
        inst_5d = sum(float(r.get("Inst_Net", 0) or 0) for r in recent_5)

        if fg_5d > 0 and inst_5d > 0:
            return 1.5, {"foreign_5d": fg_5d, "inst_5d": inst_5d, "verdict": "외인+기관 매수"}
        elif fg_5d > 0 or inst_5d > 0:
            return 1.0, {"foreign_5d": fg_5d, "inst_5d": inst_5d, "verdict": "일부 매수"}
        elif fg_5d < 0 and inst_5d < 0:
            return -1.0, {"foreign_5d": fg_5d, "inst_5d": inst_5d, "verdict": "외인+기관 매도 (악재)"}
        return 0, {"foreign_5d": fg_5d, "inst_5d": inst_5d, "verdict": "중립"}
    except Exception as e:
        return 0, {"error": str(e)}


def calculate_dart_score(ticker: str) -> dict[str, Any]:
    """DART 공시 점수 + 우리 자체 수급 교차 검증.

    Returns:
        {
            "score": int,                       # -2 ~ +3 (realtime_score 통합)
            "n_disclosures": int,               # 최근 7일 공시 수
            "latest_severity": str | None,
            "latest_tags": list,
            "supply_verified": bool,
            "supply_multiplier": float,
            "reason": str,
        }
    """
    disclosures = _fetch_recent_disclosures(ticker)
    if not disclosures:
        return {
            "score": 0,
            "n_disclosures": 0,
            "latest_severity": None,
            "latest_tags": [],
            "supply_verified": False,
            "supply_multiplier": 0,
            "reason": "DART 공시 없음 (최근 7일)",
        }

    # 최신 공시 기준
    latest = disclosures[0]
    severity = latest.get("severity", "INFO")
    sentiment_score = float(latest.get("sentiment_score", 0) or 0)
    tags = latest.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    # severity 기본 점수
    base = SEVERITY_SCORES.get(severity, 0)

    # sentiment 보너스
    sent_bonus = 0
    if sentiment_score >= SENTIMENT_BOOST:
        sent_bonus = 1
    elif sentiment_score <= -SENTIMENT_BOOST:
        sent_bonus = -1

    # tags 보너스 (PPA수혜주 / 메모리 등)
    tag_bonus = 0
    priority_tags = {"PPA수혜주", "메모리", "AI수혜", "반도체"}
    if any(t in priority_tags for t in tags):
        tag_bonus = 1

    # 우리 자체 수급 검증 (5%룰 같은 방향 미상 공시 보강)
    multiplier, supply_info = _verify_supply_via_jgis(ticker)

    raw_score = base + sent_bonus + tag_bonus
    if multiplier != 0:
        # 수급 검증 통과/실패 반영
        final_score = int(raw_score * multiplier) if raw_score > 0 else int(raw_score)
        if multiplier < 0 and raw_score > 0:
            # 외인+기관 매도 + 공시 = 악재 전환 (매도 보고자)
            final_score = -abs(raw_score)
    else:
        final_score = raw_score

    # 점수 클램프 (-2 ~ +3)
    final_score = max(-2, min(3, final_score))

    reason_parts = [f"공시 {len(disclosures)}건 ({severity})"]
    if sent_bonus:
        reason_parts.append(f"sentiment {sentiment_score:+.2f}")
    if tag_bonus:
        reason_parts.append(f"태그 {[t for t in tags if t in priority_tags]}")
    if multiplier:
        reason_parts.append(f"수급 검증 {supply_info.get('verdict', '?')}")
    reason = " | ".join(reason_parts) + f" → {final_score:+d}점"

    return {
        "score": final_score,
        "n_disclosures": len(disclosures),
        "latest_severity": severity,
        "latest_tags": tags,
        "latest_sentiment": sentiment_score,
        "supply_verified": multiplier != 0,
        "supply_multiplier": multiplier,
        "supply_info": supply_info,
        "reason": reason,
    }
