"""동시호가 강도 게이트 — 갭 #3 (5/26 19:45 신규).

배경:
- 5/22 풀세트 잔여 ❌ 4종 중 "동시호가" = 시각만 인식 (08:30~09:00), 매수 결정 X
- 동시호가 강도 = 매수 잔량 / 매도 잔량 비율 = 시초가 방향성 예측
- 사용자 5/26 캡쳐 사례: 산일전기 -3% 출발 (약세 동시호가) → 우리 시스템 무시함

룰:
- 08:30~09:00 동시호가 시간대만 적용 (그 외 시각은 PASS)
- 매수/매도 잔량 비율 확인
  - STRONG_OPEN (>= 1.5): 매수세 우세 = 매수 우대
  - BALANCED (0.67~1.5): 정상 = 통과
  - WEAK_OPEN (< 0.67): 매도세 우세 = 차단
- 잔량 부족 시 (총합 < 1000주) → fail-open

활용:
- 본 게이트는 동시호가 시간(08:30~09:00) 매수 시도 시만 작동
- 정규 시장(09:00~15:30)에는 PASS — 기존 H5 호가 게이트가 담당
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, time as dtime

logger = logging.getLogger(__name__)

OPENING_CALL_START = dtime(8, 30)
OPENING_CALL_END = dtime(9, 0)
STRONG_RATIO = float(os.getenv("OPENING_CALL_STRONG_RATIO", "1.5"))
WEAK_RATIO = float(os.getenv("OPENING_CALL_WEAK_RATIO", "0.67"))
MIN_TOTAL_VOLUME = int(os.getenv("OPENING_CALL_MIN_VOLUME", "1000"))


@dataclass
class OpeningCallGate:
    """동시호가 게이트 결과."""
    allow: bool
    reason: str           # 'OUT_OF_HOURS' / 'STRONG_OPEN' / 'BALANCED' / 'WEAK_OPEN' / 'DATA_MISSING'
    bid_total: int        # 매수 잔량 합계
    ask_total: int        # 매도 잔량 합계
    ratio: float          # bid/ask
    is_strong: bool       # 강세 동시호가


def _is_opening_call_time(now: datetime | None = None) -> bool:
    """현재 시각이 동시호가 시간대인지."""
    now = now or datetime.now()
    return OPENING_CALL_START <= now.time() < OPENING_CALL_END


def check_opening_call_gate(
    orderbook: dict | None,
    now: datetime | None = None,
    strong_ratio: float = STRONG_RATIO,
    weak_ratio: float = WEAK_RATIO,
    min_volume: int = MIN_TOTAL_VOLUME,
) -> OpeningCallGate:
    """동시호가 강도 게이트.

    Args:
        orderbook: kis_intraday_adapter.fetch_orderbook() 반환 dict
        now: 평가 시각 (None=현재)
    """
    # 1. 시각 검사 — 동시호가 시간 외는 PASS (정규 시장은 H5가 담당)
    if not _is_opening_call_time(now):
        return OpeningCallGate(
            allow=True, reason="OUT_OF_HOURS",
            bid_total=0, ask_total=0, ratio=0.0, is_strong=False,
        )

    if not orderbook:
        return OpeningCallGate(
            allow=True, reason="DATA_MISSING",
            bid_total=0, ask_total=0, ratio=0.0, is_strong=False,
        )

    bid_total = int(orderbook.get("total_bid_vol", 0) or 0)
    ask_total = int(orderbook.get("total_ask_vol", 0) or 0)
    total = bid_total + ask_total

    if total < min_volume:
        return OpeningCallGate(
            allow=True, reason="DATA_MISSING",
            bid_total=bid_total, ask_total=ask_total, ratio=0.0, is_strong=False,
        )

    ratio = bid_total / ask_total if ask_total > 0 else 999.0

    if ratio >= strong_ratio:
        return OpeningCallGate(
            allow=True, reason="STRONG_OPEN",
            bid_total=bid_total, ask_total=ask_total, ratio=ratio, is_strong=True,
        )
    if ratio < weak_ratio:
        return OpeningCallGate(
            allow=False, reason="WEAK_OPEN",
            bid_total=bid_total, ask_total=ask_total, ratio=ratio, is_strong=False,
        )
    return OpeningCallGate(
        allow=True, reason="BALANCED",
        bid_total=bid_total, ask_total=ask_total, ratio=ratio, is_strong=False,
    )
