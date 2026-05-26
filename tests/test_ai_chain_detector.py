"""AI 밸류체인 동조화 검출 단위 테스트 (5/26)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.use_cases.ai_chain_detector import (
    load_ai_chain_tickers,
    detect_ai_chain_sync,
    format_ai_chain_signal_for_telegram,
    AIChainSignal,
    AI_CHAIN_SECTORS,
)


def test_load_ai_chain_4_sectors():
    """sector_fire_map.yaml에서 4개 AI 세부 섹터 로드."""
    chain = load_ai_chain_tickers()
    assert len(chain) == 4
    assert "AI반도체검사" in chain
    assert "AI반도체PCB" in chain
    assert "AI반도체소재" in chain
    assert "AI산업소재" in chain
    # ISC가 검사 섹터에 있는지
    assert "095340" in chain["AI반도체검사"]


def _make_broker_for_surge(surge_map: dict):
    """ticker → (price, change_pct) map.

    검색되지 않은 ticker는 change_pct=0 반환.
    """
    def fetch(tk):
        if tk in surge_map:
            p, c = surge_map[tk]
            return {"output": {"stck_prpr": str(p), "prdy_ctrt": str(c),
                              "hts_kor_isnm": f"NAME_{tk}"}}
        return {"output": {"stck_prpr": "10000", "prdy_ctrt": "0.0"}}
    broker = MagicMock()
    broker.fetch_price = fetch
    return broker


def test_no_surge_no_sync():
    """폭등 0건 → 동조 X."""
    broker = _make_broker_for_surge({})
    sig = detect_ai_chain_sync(broker)
    assert sig.triggered is False
    assert sig.fire_sector_count == 0


def test_one_sector_fire_no_sync():
    """1개 섹터만 폭등 → 동조 X (min 3 미달)."""
    # AI반도체검사 ISC만 +10%
    broker = _make_broker_for_surge({"095340": (250000, 10.0)})
    sig = detect_ai_chain_sync(broker)
    assert sig.triggered is False
    assert sig.fire_sector_count == 1


def test_three_sectors_fire_triggers_sync():
    """3개 섹터 동시 폭등 → 동조 발화."""
    surge = {
        "095340": (250000, 16.9),    # AI반도체검사 (ISC)
        "007810": (113000, 12.4),    # AI반도체PCB (코리아써키트)
        "005290": (68500, 10.2),     # AI반도체소재 (동진쎄미켐)
    }
    broker = _make_broker_for_surge(surge)
    sig = detect_ai_chain_sync(broker)
    assert sig.triggered is True
    assert sig.fire_sector_count == 3
    assert len(sig.surge_stocks) == 3


def test_four_sectors_fire_full_sync():
    """4개 섹터 모두 폭등 — 5/26 실제 시나리오."""
    surge = {
        "095340": (250000, 16.9),    # 검사
        "064290": (42000, 18.9),     # 검사
        "007810": (113000, 12.4),    # PCB
        "005290": (68500, 10.2),     # 소재
        "000150": (1750000, 10.1),   # 산업소재 (두산)
    }
    broker = _make_broker_for_surge(surge)
    sig = detect_ai_chain_sync(broker)
    assert sig.triggered is True
    assert sig.fire_sector_count == 4
    assert len(sig.surge_stocks) == 5
    # 정렬: 가장 큰 폭등 먼저
    assert sig.surge_stocks[0]["change_pct"] == 18.9


def test_custom_min_sectors():
    """min_fire_sectors=2로 변경 시 2개만 발화해도 동조."""
    surge = {
        "095340": (250000, 16.9),
        "007810": (113000, 12.4),
    }
    broker = _make_broker_for_surge(surge)
    sig = detect_ai_chain_sync(broker, min_fire_sectors=2)
    assert sig.triggered is True
    assert sig.fire_sector_count == 2


def test_custom_surge_pct():
    """surge_pct=3.0 시 +5% 미만도 포착."""
    surge = {
        "095340": (250000, 3.5),
        "007810": (113000, 4.0),
        "005290": (68500, 3.2),
    }
    broker = _make_broker_for_surge(surge)
    # 기본 +5% 임계로는 발화 안 함
    sig_default = detect_ai_chain_sync(broker)
    assert sig_default.triggered is False
    # +3% 임계로 발화
    sig_low = detect_ai_chain_sync(broker, surge_pct=3.0)
    assert sig_low.triggered is True


def test_fetch_error_does_not_break():
    """fetch_price 일부 실패 → 다른 종목은 정상 처리."""
    def fetch(tk):
        if tk == "095340":
            raise Exception("API timeout")
        return {"output": {"stck_prpr": "100000", "prdy_ctrt": "12.0"}}
    broker = MagicMock()
    broker.fetch_price = fetch
    sig = detect_ai_chain_sync(broker)
    # 095340 1개 실패해도 다른 종목 +12% → 동조
    assert sig.fire_sector_count >= 1


def test_format_triggered():
    """발화 시 텔레그램 포맷."""
    sig = AIChainSignal(
        triggered=True,
        fire_sectors=["AI반도체검사", "AI반도체PCB", "AI반도체소재"],
        surge_stocks=[
            {"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
             "current_price": 250000, "change_pct": 16.9},
        ],
        fire_sector_count=3,
        reason="3/4 섹터 발화",
    )
    msg = format_ai_chain_signal_for_telegram(sig)
    assert "AI 밸류체인 동조" in msg
    assert "ISC" in msg
    assert "16.90%" in msg or "16.9" in msg


def test_format_not_triggered():
    """미발화 시 짧은 포맷."""
    sig = AIChainSignal(triggered=False, reason="1/4 섹터 발화")
    msg = format_ai_chain_signal_for_telegram(sig)
    assert "⏸️" in msg
