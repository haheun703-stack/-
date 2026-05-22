"""C2 매수 후보 필터 — 12시그널 통합 패턴 분석 (5/22 백테스트 결과 반영).

배경 (5/22 picks_history 256건 백테스트):
  단순 'grade=강력 포착' 필터: D+1 평균 +12.01%, 승률 71.9%, 폭등 35%
  → C2 필터 적용 후: D+1 평균 +20.60%, 승률 84.2%, 폭등 46.5% (101건)

C2 필터 명세 (≥+71% 수익 개선 검증):
  필수 1. sources ⊃ {AI섹터, 밸류체인, US모멘텀, 인텔리전스} 최소 2개
  필수 2. entry_condition 에 'MA5 하향이탈' 미포함
  필수 3. score >= 80
  추천 (선택). n_sources >= 3 (다중 합의)

폭등 시그널 (4핵심, 폭등 그룹에 +20%↑ 더 많이 등장):
  AI섹터:    폭등 50.0% vs 손실 21.7% (+28.3%)
  밸류체인:  폭등 37.2% vs 손실 10.1% (+27.1%)
  US모멘텀:  폭등 48.8% vs 손실 24.6% (+24.2%)
  인텔리전스: 폭등 47.7% vs 손실 27.5% (+20.1%)

회피 시그널 (손실 그룹에 더 많음):
  수급폭발: 폭등 22.1% vs 손실 33.3% (-11.2%)
  매집추적: 폭등 83.7% vs 손실 98.6% (-14.8%) — 단독 의존 위험

사용:
  from src.use_cases.signal_filter import passes_c2_filter
  result = passes_c2_filter(pick_record)
  if result['passed']:
      매수 후보 진입
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# C2 필터 핵심 시그널 (4개 중 2개 이상 요구)
CORE_SIGNALS = frozenset({"AI섹터", "밸류체인", "US모멘텀", "인텔리전스"})
CORE_MIN_COUNT = 2  # 4핵심 중 최소 2개

# 회피 시그널 (단독 의존 시 위험)
RISK_SIGNALS = frozenset({"수급폭발", "매집추적"})

# 임계
MIN_SCORE = 80               # 통합 점수 (90→80 완화, picks_history 246건 정상 분포)
MIN_N_SOURCES = 2             # 최소 시그널 개수 (보수: 3)
MA5_BLOCKED_KEYWORD = "MA5 하향이탈"


def passes_c2_filter(pick: dict[str, Any]) -> dict[str, Any]:
    """C2 필터 통과 여부.

    Args:
        pick: tomorrow_picks/picks_history record dict
              필수 키: sources (list), score (float), entry_condition (str|None)

    Returns:
        {
            "passed": bool,
            "score": float,                  # 입력 score
            "n_core_hit": int,                # 4핵심 시그널 중 매칭 개수
            "core_hit_list": list[str],       # 매칭된 4핵심 시그널
            "n_sources": int,                  # 전체 시그널 개수
            "risk_only": bool,                 # 위험 시그널 단독 의존 여부
            "blocks": list[str],               # 차단 사유
            "reason": str,                      # 종합 사유
        }
    """
    sources = pick.get("sources") or []
    sources_set = set(s for s in sources if isinstance(s, str))
    score = float(pick.get("score", 0) or 0)
    n_sources = int(pick.get("n_sources", len(sources_set)) or len(sources_set))
    entry_condition = (pick.get("entry_condition") or "")

    blocks: list[str] = []
    core_hits = sources_set & CORE_SIGNALS
    n_core_hit = len(core_hits)

    # 필수 1: 4핵심 ≥ 2개
    if n_core_hit < CORE_MIN_COUNT:
        blocks.append(
            f"4핵심 시그널 {n_core_hit}/{CORE_MIN_COUNT}개 ({sorted(core_hits) or '없음'})"
        )

    # 필수 2: MA5 하향이탈 차단
    if MA5_BLOCKED_KEYWORD in entry_condition:
        blocks.append(f"MA5 하향이탈 차단: {entry_condition[:50]}")

    # 필수 3: score >= 80
    if score < MIN_SCORE:
        blocks.append(f"score {score:.1f} < {MIN_SCORE}")

    # 위험 시그널 단독 의존 (참고용, 차단은 아니지만 표기)
    risk_only = sources_set and sources_set.issubset(RISK_SIGNALS)
    if risk_only:
        blocks.append(
            f"위험 시그널 단독 의존 ({sorted(sources_set)}) — 회피"
        )

    # n_sources 최소 (보수)
    if n_sources < MIN_N_SOURCES:
        blocks.append(f"n_sources {n_sources} < {MIN_N_SOURCES}")

    passed = len(blocks) == 0

    if passed:
        reason = (
            f"C2 통과: 4핵심 {n_core_hit}/{len(CORE_SIGNALS)}개 ({sorted(core_hits)}), "
            f"score {score:.1f}, n_sources {n_sources}"
        )
    else:
        reason = f"C2 차단: {'; '.join(blocks)}"

    return {
        "passed": passed,
        "score": score,
        "n_core_hit": n_core_hit,
        "core_hit_list": sorted(core_hits),
        "n_sources": n_sources,
        "risk_only": risk_only,
        "blocks": blocks,
        "reason": reason,
    }


def filter_picks(picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """tomorrow_picks 리스트 일괄 C2 필터링.

    Args:
        picks: tomorrow_picks.json의 picks 리스트

    Returns:
        C2 통과 종목만 (각 pick에 'c2_filter' 키 추가)
    """
    result = []
    for p in picks:
        filt = passes_c2_filter(p)
        p["c2_filter"] = filt
        if filt["passed"]:
            result.append(p)
    logger.info(
        "[C2 필터] 입력 %d건 → 통과 %d건 (%.1f%%)",
        len(picks), len(result),
        len(result) / len(picks) * 100 if picks else 0,
    )
    return result
