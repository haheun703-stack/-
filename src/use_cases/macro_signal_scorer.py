"""정보봇 intelligence_macro 매크로 시그널 통합 점수.

배경 (5/22 퐝가님 인사이트, 5/23 옵션 C):
  realtime_score는 기존 KOSPI regime만 매크로로 사용했음. 그러나 진짜 매크로 시그널은:
    - 기재부 AI 데이터센터 정책 (#8) → AI 인프라 섹터 호재
    - 젠슨황 슈퍼사이클 발언 (#6) → 반도체 호재
    - FOMC 금리 결정 → 시장 전체
    - 환율 급변 → 수출주 / 내수주 차별 영향
  정보봇 intelligence_macro 테이블이 이 시그널들을 CRITICAL/WARNING 분류 + sentiment_score
  + tags + affected_sectors / affected_tickers 형태로 제공.

교차 검증 패턴 (dart_signal_scorer와 동일 철학):
  1. 정보봇 시그널 단독 신뢰 X
  2. affected_tickers 직접 매칭 → ×1.5 (가장 강한 매칭)
  3. affected_sectors 매칭 → ×1.0 (섹터 단위 영향)
  4. 글로벌 매크로만 (둘 다 매칭 X) → ×0.5 (모든 종목에 약하게)

점수 산정 (-3 ~ +3):
  CRITICAL severity:
    호재 (sentiment ≥ +30 or tags 호재): base +2
    악재 (sentiment ≤ -30):             base -2
  WARNING severity:
    호재: base +1
    악재: base -1
  INFO: base 0

신선도 (3일 윈도):
  - 최근 24시간 이내: 가중치 1.0
  - 24~72시간: 가중치 0.5 (1단계 감쇠)
  - 72시간 초과: 0

사용:
  from src.use_cases.macro_signal_scorer import calculate_macro_signal_score
  result = calculate_macro_signal_score("005930", sector="반도체")
  # → {score: +2, severity: "CRITICAL", matched_by: "ticker", reason: "..."}
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 신선도 윈도 (시간)
FRESH_WINDOW_H = 24
STALE_WINDOW_H = 72

# severity 기본 점수
SEVERITY_BASE = {
    "CRITICAL": 2,
    "WARNING": 1,
    "INFO": 0,
}

# sentiment_score 임계 (정보봇 scale -100~+100, dart_signal_scorer와 동일)
SENTIMENT_GOOD = 30
SENTIMENT_BAD = -30

# 매칭 강도 multiplier
MATCH_TICKER = 1.5
MATCH_SECTOR = 1.0
MATCH_GLOBAL = 0.5

# 호재 키워드 (sentiment 부족 시 보조)
POSITIVE_TAGS = {
    "AI수혜", "데이터센터수혜", "PPA수혜주", "수출호재",
    "반도체호재", "정책수혜", "감세", "유동성확대",
}
NEGATIVE_TAGS = {
    "금리인상", "긴축", "유가급등", "환율급등",
    "전쟁리스크", "규제강화", "공급망충격",
}


def _fetch_recent_macro(days: int = 3) -> list[dict]:
    """정보봇 intelligence_macro 조회 (최근 N일).

    스키마 가정:
        date / severity / category / title / sentiment_score / tags(JSONB) /
        affected_sectors(JSONB) / affected_tickers(JSONB) / source / published_at
    """
    try:
        from src.adapters.quant_supabase_reader import _get_client
        client = _get_client()
        if not client:
            return []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        res = (
            client.table("intelligence_macro")
            .select(
                "date,severity,category,title,sentiment_score,tags,"
                "affected_sectors,affected_tickers,source,published_at"
            )
            .gte("date", cutoff)
            .order("published_at", desc=True)
            .limit(20)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.debug("intelligence_macro 조회 실패: %s", e)
        return []


def _normalize_list(field) -> list[str]:
    """JSONB list / CSV string 모두 list[str]로 정규화."""
    if not field:
        return []
    if isinstance(field, list):
        return [str(x).strip() for x in field if x]
    if isinstance(field, str):
        return [t.strip() for t in field.split(",") if t.strip()]
    return []


def _freshness_weight(published_at: Optional[str]) -> float:
    """공시 시각 기준 신선도 가중치 (0~1)."""
    if not published_at:
        return 1.0
    try:
        if isinstance(published_at, str):
            ts = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        else:
            ts = published_at
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed_h = (datetime.now() - ts).total_seconds() / 3600
        if elapsed_h < 0:
            return 1.0
        if elapsed_h <= FRESH_WINDOW_H:
            return 1.0
        if elapsed_h <= STALE_WINDOW_H:
            return 0.5
        return 0.0
    except Exception:
        return 1.0


def _score_single_macro(row: dict, ticker: str, sector: str) -> tuple[int, str, str]:
    """매크로 시그널 1건 → (score, matched_by, reason)."""
    severity = (row.get("severity") or "INFO").upper()
    sentiment = float(row.get("sentiment_score") or 0)
    tags = _normalize_list(row.get("tags"))
    sectors = _normalize_list(row.get("affected_sectors"))
    tickers = _normalize_list(row.get("affected_tickers"))

    # 매칭 강도 결정
    ticker_padded = str(ticker).zfill(6)
    if ticker_padded in tickers:
        multiplier = MATCH_TICKER
        matched_by = "ticker"
    elif sector and sector in sectors:
        multiplier = MATCH_SECTOR
        matched_by = "sector"
    else:
        multiplier = MATCH_GLOBAL
        matched_by = "global"

    # base 점수 (severity + sentiment + tags)
    base = SEVERITY_BASE.get(severity, 0)
    sign = 0
    if sentiment >= SENTIMENT_GOOD:
        sign = +1
    elif sentiment <= SENTIMENT_BAD:
        sign = -1

    # tags 부호 (sentiment가 약한 경우 보조)
    if sign == 0:
        if any(t in POSITIVE_TAGS for t in tags):
            sign = +1
        elif any(t in NEGATIVE_TAGS for t in tags):
            sign = -1

    raw = base * sign  # base 자체는 항상 양수 → sign으로 호재/악재 결정

    # 신선도 가중
    fresh = _freshness_weight(row.get("published_at"))
    weighted = raw * multiplier * fresh

    # 정수 라운드 + 클램프 (-3 ~ +3)
    score = int(round(weighted))
    score = max(-3, min(3, score))

    title = (row.get("title") or row.get("category") or "")[:30]
    reason = (
        f"{severity} '{title}' "
        f"(sent {sentiment:+.0f}, {matched_by}, fresh {fresh:.1f}) → {score:+d}"
    )
    return score, matched_by, reason


def calculate_macro_signal_score(
    ticker: str,
    sector: str = "",
) -> dict[str, Any]:
    """intelligence_macro 시그널 통합 점수 산정.

    여러 매크로 시그널 누적 시:
        - 같은 방향(호재/악재) 시그널 N건 → 합산하되 -3 ~ +3 클램프
        - 호재/악재 혼재 → 자연 상쇄
        - ticker 매칭 우선, 없으면 sector, 없으면 글로벌

    Returns:
        {
            "score": int,                  # -3 ~ +3
            "n_signals": int,              # 최근 3일 매크로 시그널 수
            "matched_by": str,             # ticker / sector / global / none
            "top_signal": dict | None,     # 가장 강한 시그널 (raw)
            "reason": str,
            "breakdown": list[str],
        }
    """
    rows = _fetch_recent_macro(days=3)
    if not rows:
        return {
            "score": 0,
            "n_signals": 0,
            "matched_by": "none",
            "top_signal": None,
            "reason": "최근 3일 매크로 시그널 없음",
            "breakdown": [],
        }

    total = 0
    breakdown = []
    matched_modes = []
    abs_max = 0
    top_signal_row = None

    for row in rows:
        s, mode, reason = _score_single_macro(row, ticker, sector)
        if s == 0:
            continue
        total += s
        breakdown.append(reason)
        matched_modes.append(mode)
        if abs(s) > abs_max:
            abs_max = abs(s)
            top_signal_row = row

    # 클램프 (-3 ~ +3)
    final = max(-3, min(3, total))

    # 대표 matched_by (우선순위: ticker > sector > global)
    if "ticker" in matched_modes:
        matched_by = "ticker"
    elif "sector" in matched_modes:
        matched_by = "sector"
    elif "global" in matched_modes:
        matched_by = "global"
    else:
        matched_by = "none"

    if final == 0 and not breakdown:
        reason_top = f"매크로 {len(rows)}건 모두 영향 없음 (sentiment/tags 중립)"
    else:
        reason_top = (
            f"매크로 {len(breakdown)}/{len(rows)}건 영향 ({matched_by}) "
            f"→ 합계 {total:+d}, 클램프 {final:+d}"
        )

    return {
        "score": final,
        "n_signals": len(rows),
        "n_effective": len(breakdown),
        "matched_by": matched_by,
        "top_signal": top_signal_row,
        "reason": reason_top,
        "breakdown": breakdown,
    }
