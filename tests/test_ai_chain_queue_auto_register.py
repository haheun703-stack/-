"""AI 동조 자동 큐 등록 단위 테스트 (5/26)."""
from __future__ import annotations

import pytest

from src.use_cases.ai_chain_queue_auto_register import (
    register_ai_chain_queues,
    merge_into_queue_state,
    format_registration_for_telegram,
    QueueRegistrationResult,
    _adjust_to_tick,
    DEFAULT_STAGES,
)


@pytest.fixture(autouse=True)
def enable_register(monkeypatch):
    """모든 테스트에서 ENABLED=True 강제."""
    monkeypatch.setattr(
        "src.use_cases.ai_chain_queue_auto_register.ENABLED", True
    )


def test_register_simple():
    """단일 폭등 종목 큐 등록 — L1 -3%/L2 -7%/L3 -12%."""
    surge = [
        {"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
         "current_price": 250000, "change_pct": 16.9},
    ]
    r = register_ai_chain_queues(surge)
    assert len(r.registered) == 1
    entry = r.registered[0]
    assert entry["ticker"] == "095340"
    assert entry["peak_price"] == 250000
    stages = entry["stages"]
    assert len(stages) == 3
    # L1 = peak × 0.97 = 242,500 → tick 500원 단위 (200K~500K 구간)
    assert stages[0]["level"] == 1
    assert stages[0]["target_price"] == 242500
    # L2 = peak × 0.93 = 232,499.999 (부동소수) → int 232499 → tick 500 → 232000
    assert stages[1]["target_price"] == 232000
    # L3 = peak × 0.88 = 220,000 (tick 500 그대로)
    assert stages[2]["target_price"] == 220000
    assert all(s["status"] == "PENDING" for s in stages)
    assert entry["source"] == "AI_CHAIN_SYNC"


def test_disabled_returns_empty(monkeypatch):
    """ENABLED=False 시 등록 X."""
    monkeypatch.setattr(
        "src.use_cases.ai_chain_queue_auto_register.ENABLED", False
    )
    surge = [{"ticker": "095340", "current_price": 250000, "change_pct": 16.9}]
    r = register_ai_chain_queues(surge)
    assert len(r.registered) == 0


def test_skip_protected():
    """보호 종목 스킵."""
    surge = [
        {"ticker": "010120", "name": "LS ELECTRIC", "current_price": 290000, "change_pct": 10.0},
        {"ticker": "095340", "name": "ISC", "current_price": 250000, "change_pct": 16.9},
    ]
    r = register_ai_chain_queues(surge, protected_tickers={"010120"})
    assert len(r.registered) == 1
    assert r.registered[0]["ticker"] == "095340"
    assert any(s["reason"] == "PROTECTED" for s in r.skipped)


def test_skip_held():
    """보유 종목 스킵."""
    surge = [{"ticker": "067310", "current_price": 50000, "change_pct": 8.0}]
    r = register_ai_chain_queues(surge, held_tickers={"067310"})
    assert len(r.registered) == 0
    assert r.skipped[0]["reason"] == "HELD"


def test_skip_already_queued():
    """이미 큐에 있으면 중복 X."""
    surge = [{"ticker": "095340", "current_price": 250000, "change_pct": 16.9}]
    existing_queue = {"queues": {"095340": {"stages": []}}}
    r = register_ai_chain_queues(surge, queue_state=existing_queue)
    assert len(r.registered) == 0
    assert r.skipped[0]["reason"] == "ALREADY_QUEUED"


def test_invalid_price_skipped():
    """current_price=0 → 스킵."""
    surge = [{"ticker": "095340", "current_price": 0, "change_pct": 16.9}]
    r = register_ai_chain_queues(surge)
    assert len(r.registered) == 0
    assert r.skipped[0]["reason"] == "INVALID_PRICE"


def test_ticker_zfill():
    """5자리 ticker → 6자리 정규화."""
    surge = [{"ticker": "67310", "current_price": 50000, "change_pct": 10.0}]
    r = register_ai_chain_queues(surge)
    assert r.registered[0]["ticker"] == "067310"


def test_tick_size_adjustment_high_price():
    """50만 이상은 1,000원 단위 호가."""
    surge = [{"ticker": "000150", "name": "두산",
              "current_price": 1759000, "change_pct": 10.4}]
    r = register_ai_chain_queues(surge)
    stages = r.registered[0]["stages"]
    # L1 = 1,759,000 × 0.97 = 1,706,230 → tick 1,000 → 1,706,000
    assert stages[0]["target_price"] == 1706000
    # L2 = 1,759,000 × 0.93 = 1,635,870 → 1,635,000
    assert stages[1]["target_price"] == 1635000


def test_tick_size_low_price():
    """1,000원 미만은 1원 단위."""
    surge = [{"ticker": "432320", "name": "KB스타리츠",
              "current_price": 1664, "change_pct": 6.9}]
    r = register_ai_chain_queues(surge)
    stages = r.registered[0]["stages"]
    # 1,664 × 0.97 = 1,614.08 → tick 10 (1000~5000 구간) → 1,610
    assert stages[0]["target_price"] == 1610


def test_merge_into_queue_state():
    """등록 결과를 queue_state에 병합."""
    initial = {"queues": {"OTHER": {"stages": []}}}
    surge = [{"ticker": "095340", "current_price": 250000, "change_pct": 16.9}]
    r = register_ai_chain_queues(surge, queue_state=initial)
    merged = merge_into_queue_state(initial, r.registered)
    assert "095340" in merged["queues"]
    assert "OTHER" in merged["queues"]  # 기존 보존
    assert merged["queues"]["095340"]["source"] == "AI_CHAIN_SYNC"


def test_qty_calculation():
    """alloc_amount 100만 / target 30% / target_price = qty 계산."""
    surge = [{"ticker": "095340", "current_price": 250000, "change_pct": 16.9}]
    r = register_ai_chain_queues(surge, alloc_amount=1000000)
    stages = r.registered[0]["stages"]
    # L1: 100만 × 30% = 30만 / 242,500 = 1.23 → 1주
    assert stages[0]["qty"] == 1
    assert stages[0]["alloc_amount"] == 300000


def test_full_5_surge_scenario():
    """5/26 실제 시나리오: 5종 동시 폭등 모두 큐 등록."""
    surge = [
        {"ticker": "095340", "name": "ISC", "current_price": 245500, "change_pct": 16.9},
        {"ticker": "064290", "name": "인텍플러스", "current_price": 42000, "change_pct": 18.9},
        {"ticker": "007810", "name": "코리아써키트", "current_price": 113000, "change_pct": 12.4},
        {"ticker": "000150", "name": "두산", "current_price": 1759000, "change_pct": 10.1},
        {"ticker": "005290", "name": "동진쎄미켐", "current_price": 68500, "change_pct": 10.2},
    ]
    r = register_ai_chain_queues(surge)
    assert len(r.registered) == 5
    # 5종 모두 stages 3개씩
    for entry in r.registered:
        assert len(entry["stages"]) == 3


def test_format_telegram():
    """텔레그램 포맷에 종목명 + L1 가격 포함."""
    surge = [{"ticker": "095340", "name": "ISC", "current_price": 250000, "change_pct": 16.9}]
    r = register_ai_chain_queues(surge)
    msg = format_registration_for_telegram(r)
    assert "ISC" in msg
    assert "L1 -3%" in msg
    assert "242,500" in msg or "242500" in msg


def test_format_empty():
    """등록 0건 → 짧은 메시지."""
    r = QueueRegistrationResult(registered=[], skipped=[], total_queued=0)
    msg = format_registration_for_telegram(r)
    assert "없음" in msg


def test_adjust_to_tick_boundary():
    """KRX 호가 단위 보정 경계값.

    1원 단위 (<1000) / 5원 단위 (1000~5000) / 10원 단위 (5000~20000)
    / 50원 단위 (20000~50000) / 100원 단위 (50000~200000)
    / 500원 단위 (200000~500000) / 1000원 단위 (>=500000)
    """
    assert _adjust_to_tick(999) == 999     # 1원 단위
    assert _adjust_to_tick(1234) == 1230   # 5원 단위: 1234//5*5 = 1230
    assert _adjust_to_tick(4999) == 4995   # 5원 단위
    assert _adjust_to_tick(15000) == 15000 # 10원 단위 정수배
    assert _adjust_to_tick(25000) == 25000 # 50원 단위 정수배
    assert _adjust_to_tick(25049) == 25000 # 50원 단위 (25049//50*50)
    assert _adjust_to_tick(232499) == 232000  # 500원 단위 (200K~500K)
