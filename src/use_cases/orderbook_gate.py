"""호가창 매수 게이트 — H5 (5/26 PDCA flexible-pullback-buy).

배경:
- 5/22 풀세트 D 학습 11종 중 호가 API: "어댑터에 있음, 매수 게이트 X"
- kis_intraday_adapter.fetch_orderbook()는 10호가 잔량 조회 가능
- 매수 직전 슬리피지 검증 + 매수1 잔량 + 매수/매도 비율로 진입 정밀도 향상

룰 (백테스트 검증 전 잠정치 — 5/27 백테스트 후 보정):
1. SLIPPAGE_OK: 매수1 호가 <= 목표가 × 1.005 (0.5% 슬리피지 이내) → 통과
2. SLIPPAGE_TOO_WIDE: 매수1 - 매도1 > 1% → 차단 (호가 갭 큼 = 변동성 위험)
3. BID_THIN: 매수 총잔량 < 매도 총잔량 × 0.5 → 차단 (매수세 약함)
4. BID_STRONG: 매수/매도 잔량 비율 >= 2.0 → 통과 + 우대 마커 (강한 매수세)
5. NORMAL: 정상 범위 → 통과

활용:
- src/use_cases/adaptive_buy_queue.py execute_auto_buy() 직전 호출
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SLIPPAGE_MAX_PCT = float(os.getenv("ORDERBOOK_SLIPPAGE_MAX_PCT", "0.5"))
SPREAD_MAX_PCT = float(os.getenv("ORDERBOOK_SPREAD_MAX_PCT", "1.0"))
BID_ASK_RATIO_MIN = float(os.getenv("ORDERBOOK_BID_ASK_RATIO_MIN", "0.5"))
BID_ASK_RATIO_STRONG = float(os.getenv("ORDERBOOK_BID_ASK_RATIO_STRONG", "2.0"))


@dataclass
class OrderbookGate:
    """호가 게이트 결과."""
    allow: bool
    reason: str                    # 'SLIPPAGE_OK' / 'SLIPPAGE_TOO_WIDE' / 'SPREAD_WIDE' / 'BID_THIN' / 'BID_STRONG' / 'NORMAL' / 'DATA_MISSING'
    best_ask: int                  # 매도1
    best_bid: int                  # 매수1
    spread_pct: float              # (매도1 - 매수1) / 매수1 × 100
    slippage_pct: float            # (매도1 - 목표가) / 목표가 × 100
    bid_ask_ratio: float           # 매수 총잔량 / 매도 총잔량
    is_strong_bid: bool


def check_orderbook_buy_gate(
    orderbook: dict,
    target_price: int,
    slippage_max_pct: float = SLIPPAGE_MAX_PCT,
    spread_max_pct: float = SPREAD_MAX_PCT,
    ratio_min: float = BID_ASK_RATIO_MIN,
    ratio_strong: float = BID_ASK_RATIO_STRONG,
) -> OrderbookGate:
    """매수 직전 호가 게이트.

    Args:
        orderbook: kis_intraday_adapter.fetch_orderbook() 반환 dict.
            구조: {"asks": [{"price","volume"},...], "bids": [...],
                   "total_ask_vol": int, "total_bid_vol": int, "bid_ask_ratio": float}
        target_price: 매수 희망가 (큐의 stage["target_price"])
    """
    if not orderbook or not orderbook.get("asks") or not orderbook.get("bids"):
        logger.warning("[Orderbook gate] 호가 데이터 미수신 — fail-open")
        return OrderbookGate(
            allow=True, reason="DATA_MISSING",
            best_ask=0, best_bid=0,
            spread_pct=0.0, slippage_pct=0.0,
            bid_ask_ratio=0.0, is_strong_bid=False,
        )

    asks = orderbook.get("asks", [])
    bids = orderbook.get("bids", [])
    best_ask = int(asks[0]["price"]) if asks else 0
    best_bid = int(bids[0]["price"]) if bids else 0

    if best_ask <= 0 or best_bid <= 0:
        return OrderbookGate(
            allow=True, reason="DATA_MISSING",
            best_ask=best_ask, best_bid=best_bid,
            spread_pct=0.0, slippage_pct=0.0,
            bid_ask_ratio=0.0, is_strong_bid=False,
        )

    spread_pct = (best_ask - best_bid) / best_bid * 100
    slippage_pct = (best_ask - target_price) / target_price * 100 if target_price > 0 else 0.0
    ratio = orderbook.get("bid_ask_ratio", 0)
    if not ratio:
        total_ask = orderbook.get("total_ask_vol", 0) or 1
        total_bid = orderbook.get("total_bid_vol", 0)
        ratio = total_bid / total_ask if total_ask > 0 else 0
    is_strong = ratio >= ratio_strong

    # 차단 룰 우선 검사
    # 1. 슬리피지 과도 — 매도1 > 목표가 × (1 + slippage_max%)
    if slippage_pct > slippage_max_pct:
        return OrderbookGate(
            allow=False, reason="SLIPPAGE_TOO_WIDE",
            best_ask=best_ask, best_bid=best_bid,
            spread_pct=spread_pct, slippage_pct=slippage_pct,
            bid_ask_ratio=ratio, is_strong_bid=is_strong,
        )
    # 2. 스프레드 과도
    if spread_pct > spread_max_pct:
        return OrderbookGate(
            allow=False, reason="SPREAD_WIDE",
            best_ask=best_ask, best_bid=best_bid,
            spread_pct=spread_pct, slippage_pct=slippage_pct,
            bid_ask_ratio=ratio, is_strong_bid=is_strong,
        )
    # 3. 매수세 약함
    if ratio > 0 and ratio < ratio_min:
        return OrderbookGate(
            allow=False, reason="BID_THIN",
            best_ask=best_ask, best_bid=best_bid,
            spread_pct=spread_pct, slippage_pct=slippage_pct,
            bid_ask_ratio=ratio, is_strong_bid=False,
        )

    # 통과 — 강한 매수세 여부 표기
    if is_strong:
        return OrderbookGate(
            allow=True, reason="BID_STRONG",
            best_ask=best_ask, best_bid=best_bid,
            spread_pct=spread_pct, slippage_pct=slippage_pct,
            bid_ask_ratio=ratio, is_strong_bid=True,
        )
    return OrderbookGate(
        allow=True, reason="NORMAL",
        best_ask=best_ask, best_bid=best_bid,
        spread_pct=spread_pct, slippage_pct=slippage_pct,
        bid_ask_ratio=ratio, is_strong_bid=False,
    )
