"""AI 밸류체인 동조 발화 시 분할매수 큐 자동 등록 — 5/27 실매매 진입 핵심.

배경 (5/26 퐝가님 지시):
- "내일부터 제대로 실매매로 들어갈 수 있게 진행"
- 현재 큐 (MVP-2) = 가격 -10/-20/-30% 고정 → 강세장에서 발화 X
- AI 동조 발화 종목을 강세장 적응형 큐로 자동 등록 → 눌림목 매수

차이점 (기존 MVP-2 큐 vs 본 모듈):
| 항목 | MVP-2 큐 (peak 기반) | AI 동조 큐 (강세장) |
|------|------------------|-----------------|
| peak 기준 | 30일 천장 | 현재가 (오늘 강세 시점) |
| L1 | -10% (peak × 0.9) | **-3%** (현재가 × 0.97) |
| L2 | -20% (peak × 0.8) | **-7%** (현재가 × 0.93) |
| L3 | -30% (peak × 0.7) | **-12%** (현재가 × 0.88) |
| 발동 조건 | 보유 종목 천장 -3% 매도 후 | **AI 동조 발화 + 워치리스트 추가 후** |
| 만료 | 14일 | **3일** (강세 모멘텀 짧음) |

룰:
1. AI 동조 (4섹터 중 3섹터 동시 발화) + 폭등 종목 (+5% 이상) 발견
2. 보호종목 / 보유종목 / 기존 큐 등록 제외
3. peak = 현재가 (강세 시점)
4. L1 -3% / L2 -7% / L3 -12% PENDING 큐 등록 (각 alloc 30%)
5. 만료 3일 (강세 모멘텀 종료 가정)
6. H4~H7 entry_gates 통과 시에만 buy_limit (안전망 보존)

환경변수:
- AI_CHAIN_QUEUE_AUTO_REGISTER=1 (활성)
- AI_CHAIN_QUEUE_ALLOC_AMOUNT (기본 100만원)
- AI_CHAIN_QUEUE_EXPIRY_DAYS (기본 3일)

활용:
- run_adaptive_cycle.py MVP-5 (AI 동조) 발화 시 자동 호출
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ENABLED = os.getenv("AI_CHAIN_QUEUE_AUTO_REGISTER", "0") == "1"
ALLOC_AMOUNT = int(os.getenv("AI_CHAIN_QUEUE_ALLOC_AMOUNT", "1000000"))
EXPIRY_DAYS = int(os.getenv("AI_CHAIN_QUEUE_EXPIRY_DAYS", "3"))

# 강세장 적응 단계 (기존 -10/-20/-30 → -3/-7/-12)
# ★ C3 fix (5/26 검수): alloc_ratio 합계 1.00 (이전 0.90 = 10% 사장 문제)
DEFAULT_STAGES = [
    {"level": 1, "pullback_pct": 3.0, "alloc_ratio": 0.34},
    {"level": 2, "pullback_pct": 7.0, "alloc_ratio": 0.33},
    {"level": 3, "pullback_pct": 12.0, "alloc_ratio": 0.33},
]


@dataclass
class QueueRegistrationResult:
    """큐 자동 등록 결과."""
    registered: list[dict]    # 새로 등록된 종목 [{ticker, name, peak, stages}]
    skipped: list[dict]       # 제외된 종목 [{ticker, reason}]
    total_queued: int


def _adjust_to_tick(price: int) -> int:
    """KRX 호가 단위 보정 (KisOrderAdapter._adjust_to_tick 동일 로직)."""
    if price <= 0:
        return price
    if price < 1_000: tick = 1
    elif price < 5_000: tick = 5
    elif price < 20_000: tick = 10
    elif price < 50_000: tick = 50
    elif price < 200_000: tick = 100
    elif price < 500_000: tick = 500
    else: tick = 1_000
    return (price // tick) * tick


def register_ai_chain_queues(
    surge_stocks: list[dict],
    protected_tickers: set[str] | None = None,
    held_tickers: set[str] | None = None,
    queue_state: dict | None = None,
    alloc_amount: int = ALLOC_AMOUNT,
) -> QueueRegistrationResult:
    """AI 동조 폭등 종목들을 강세장 적응형 큐로 자동 등록.

    Args:
        surge_stocks: ai_chain_detector.detect_ai_chain_sync() 결과의 surge_stocks
        protected_tickers: 보호 종목 (사용자 수동 보유)
        held_tickers: 자동매매 보유 종목
        queue_state: 현재 adaptive_buy_queue.json 내용 (중복 등록 회피)
        alloc_amount: 종목당 총 할당 금액 (기본 100만)

    Returns:
        QueueRegistrationResult — 등록된 stage 목록 + 스킵 사유
    """
    protected_tickers = protected_tickers or set()
    held_tickers = held_tickers or set()
    queue_state = queue_state or {"queues": {}}

    if not ENABLED:
        return QueueRegistrationResult(
            registered=[], skipped=[{"reason": "AI_CHAIN_QUEUE_AUTO_REGISTER=0"}],
            total_queued=0,
        )

    existing = set(queue_state.get("queues", {}).keys())
    registered = []
    skipped = []

    for s in surge_stocks:
        tk = str(s.get("ticker", "")).zfill(6)
        if not tk:
            continue
        # 제외 사유
        if tk in protected_tickers:
            skipped.append({"ticker": tk, "reason": "PROTECTED"})
            continue
        if tk in held_tickers:
            skipped.append({"ticker": tk, "reason": "HELD"})
            continue
        if tk in existing:
            skipped.append({"ticker": tk, "reason": "ALREADY_QUEUED"})
            continue

        peak = int(s.get("current_price", 0))
        if peak <= 0:
            skipped.append({"ticker": tk, "reason": "INVALID_PRICE"})
            continue

        # 강세장 적응 stages 생성
        stages = []
        for cfg in DEFAULT_STAGES:
            target_price = _adjust_to_tick(int(peak * (1 - cfg["pullback_pct"] / 100)))
            qty = max(1, int(alloc_amount * cfg["alloc_ratio"] / max(target_price, 1)))
            stages.append({
                "level": cfg["level"],
                "target_pct": 1 - cfg["pullback_pct"] / 100,
                "target_price": target_price,
                "alloc_ratio": cfg["alloc_ratio"],
                "alloc_amount": int(alloc_amount * cfg["alloc_ratio"]),
                "qty": qty,
                "status": "PENDING",
                "triggered_at": None,
                "order_id": None,
                "actual_price": 0,
                "actual_qty": 0,
                "error": None,
                "quick_profit_target": 0,
                "quick_profit_order_id": None,
                "quick_profit_sold_at": None,
                "quick_profit_sold_price": 0,
                "trailing_peak": 0,
                "trailing_armed_at": None,
                "trailing_peak_updated_at": None,
            })

        now = datetime.now()
        entry = {
            "ticker": tk,
            "name": s.get("name", "") or tk,
            "peak_price": peak,
            "peak_date": now.date().isoformat(),
            "available_cash": alloc_amount,
            "registered_at": now.isoformat(timespec="seconds"),
            "stages": stages,
            "source": "AI_CHAIN_SYNC",   # 추적용 마커
            "sector": s.get("sector", ""),
            "expiry_days": EXPIRY_DAYS,   # MVP-2 기본 14일 vs 본 3일
        }
        registered.append(entry)

    logger.info(
        "[AI chain queue] %d종목 등록, %d종목 스킵 (할당 %d원/종목)",
        len(registered), len(skipped), alloc_amount,
    )
    return QueueRegistrationResult(
        registered=registered, skipped=skipped,
        total_queued=len(existing) + len(registered),
    )


def merge_into_queue_state(queue_state: dict, registrations: list[dict]) -> dict:
    """등록 결과를 queue_state에 병합. 호출자가 file write 책임."""
    queues = queue_state.setdefault("queues", {})
    for entry in registrations:
        queues[entry["ticker"]] = entry
    return queue_state


def format_registration_for_telegram(result: QueueRegistrationResult) -> str:
    """텔레그램 알림 포맷."""
    if not result.registered:
        return "📭 AI 동조 큐 자동 등록 없음"

    lines = [
        f"⚡ [AI 동조 강세장 큐 자동 등록] {len(result.registered)}종목",
        f"  단계: L1 -3% / L2 -7% / L3 -12% (강세장 적응)",
        f"  만료: {EXPIRY_DAYS}일 (강세 모멘텀 종료)",
        f"  할당: {ALLOC_AMOUNT:,}원/종목",
        "",
    ]
    for r in result.registered[:8]:
        peak = r["peak_price"]
        l1 = r["stages"][0]["target_price"]
        lines.append(
            f"  {r['ticker']} {r['name']:14s} peak {peak:,} → L1 {l1:,} (-3%)"
        )
    return "\n".join(lines)
