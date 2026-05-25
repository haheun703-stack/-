"""MVP-2.6 자동 손절 -5% 테스트 (5/25 백테스트 R2 기반)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_queue(tmp_path, monkeypatch):
    """가짜 큐 파일 + adaptive_buy_queue.QUEUE_PATH 임시 변경."""
    queue_path = tmp_path / "queue.json"
    from src.use_cases import adaptive_buy_queue as bq
    monkeypatch.setattr(bq, "QUEUE_PATH", queue_path)
    return queue_path


def _setup_queue(queue_path: Path, ticker: str, actual_price: int, status: str = "FILLED"):
    """단일 종목 FILLED stage 큐 작성."""
    data = {
        "queues": {
            ticker: {
                "ticker": ticker,
                "name": f"테스트{ticker}",
                "peak_price": actual_price * 2,
                "stages": [
                    {
                        "level": 1,
                        "target_price": actual_price,
                        "actual_price": actual_price,
                        "actual_qty": 1,
                        "qty": 1,
                        "status": status,
                    }
                ],
            }
        }
    }
    queue_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _mock_broker(current_price: int):
    """현재가 mock broker."""
    broker = MagicMock(name="MockBroker")
    broker.fetch_price = lambda ticker: {"output": {"stck_prpr": str(current_price)}}
    # sell_market 호출 mock
    order = MagicMock()
    order.order_id = "TEST_ORDER_001"
    broker.sell_market = lambda t, q: order
    return broker


# === 1. 정상 손절 (-5% 도달) ===

def test_stop_loss_triggered_at_minus_5pct(fake_queue, monkeypatch):
    """매수가 100,000 → 현재가 94,000 (-6%) → 손절 매도."""
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "1")
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "5")

    _setup_queue(fake_queue, "TEST001", actual_price=100_000, status="FILLED")
    broker = _mock_broker(current_price=94_000)  # -6% (임계 -5% 초과)

    # 모듈 재import (env 반영)
    import importlib
    from src.use_cases import adaptive_stop_loss as sl
    importlib.reload(sl)

    triggers = sl.check_stop_loss_triggers(broker)
    assert len(triggers) == 1
    assert triggers[0]["ticker"] == "TEST001"
    assert triggers[0]["status"] == "STOP_LOSS_SOLD"
    assert triggers[0]["actual_buy"] == 100_000
    assert triggers[0]["current_price"] == 94_000
    assert triggers[0]["loss_pct"] < -5  # 약 -6%


def test_stop_loss_not_triggered_above_threshold(fake_queue, monkeypatch):
    """매수가 100,000 → 현재가 96,000 (-4%) → 손절 미발동 (보유 유지)."""
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "1")
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "5")

    _setup_queue(fake_queue, "TEST002", actual_price=100_000, status="FILLED")
    broker = _mock_broker(current_price=96_000)  # -4% (임계 미달)

    import importlib
    from src.use_cases import adaptive_stop_loss as sl
    importlib.reload(sl)

    triggers = sl.check_stop_loss_triggers(broker)
    assert len(triggers) == 0


def test_stop_loss_exactly_at_threshold(fake_queue, monkeypatch):
    """매수가 100,000 → 현재가 95,000 (정확히 -5%) → 손절 발동 (≤ 조건)."""
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "1")
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "5")

    _setup_queue(fake_queue, "TEST003", actual_price=100_000, status="FILLED")
    broker = _mock_broker(current_price=95_000)  # 정확히 -5%

    import importlib
    from src.use_cases import adaptive_stop_loss as sl
    importlib.reload(sl)

    triggers = sl.check_stop_loss_triggers(broker)
    assert len(triggers) == 1
    assert abs(triggers[0]["loss_pct"] - (-5.0)) < 0.001


# === 2. KILL_SWITCH P0-4 보장 (매도 계속) ===

def test_stop_loss_continues_under_kill_switch(fake_queue, monkeypatch):
    """KILL_SWITCH 활성 시에도 손절 매도 계속 진행 (P0-4 보장)."""
    from src.use_cases.adaptive_stop_loss import KILL_SWITCH_PATH

    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if not KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.write_text("test_p0_4_stop_loss")
        created = True

    try:
        monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "1")
        monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "5")

        _setup_queue(fake_queue, "TEST004", actual_price=100_000, status="FILLED")
        broker = _mock_broker(current_price=93_000)  # -7% (임계 초과)

        import importlib
        from src.use_cases import adaptive_stop_loss as sl
        importlib.reload(sl)

        triggers = sl.check_stop_loss_triggers(broker)
        # P0-4 보장: KILL_SWITCH 있어도 손절 매도 발동
        assert len(triggers) == 1, "P0-4 위반: KILL_SWITCH 시 손절 매도 차단됨"
    finally:
        if created and KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()


# === 3. STOP_LOSS_ENABLED=0 비활성 ===

def test_stop_loss_disabled_noop(fake_queue, monkeypatch):
    """ADAPTIVE_STOP_LOSS_ENABLED=0 → no-op (매도 X)."""
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "0")
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "5")

    _setup_queue(fake_queue, "TEST005", actual_price=100_000, status="FILLED")
    broker = _mock_broker(current_price=90_000)  # -10% (정상이면 손절)

    import importlib
    from src.use_cases import adaptive_stop_loss as sl
    importlib.reload(sl)

    triggers = sl.check_stop_loss_triggers(broker)
    assert len(triggers) == 0, "STOP_LOSS_ENABLED=0인데 손절 발동됨"


# === 4. QUICK_ARMED 상태는 손절 평가 X (Quick Profit이 보호) ===

def test_stop_loss_skips_quick_armed_stages(fake_queue, monkeypatch):
    """QUICK_ARMED 상태는 손절 평가 미실시 (Trailing이 보호)."""
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "1")
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "5")

    _setup_queue(fake_queue, "TEST006", actual_price=100_000, status="QUICK_ARMED")
    broker = _mock_broker(current_price=90_000)  # -10%

    import importlib
    from src.use_cases import adaptive_stop_loss as sl
    importlib.reload(sl)

    triggers = sl.check_stop_loss_triggers(broker)
    assert len(triggers) == 0, "QUICK_ARMED 상태에서 손절 발동됨 (Trailing 충돌)"


# === 5. 포맷터 ===

def test_format_stop_loss_for_telegram():
    """텔레그램 포맷 동작."""
    from src.use_cases.adaptive_stop_loss import format_stop_loss_for_telegram

    trigger = {
        "ticker": "110990",
        "name": "디아이티",
        "level": 1,
        "actual_buy": 25_245,
        "current_price": 23_982,
        "loss_pct": -5.0,
        "qty": 1,
        "order_id": "ORDER_TEST",
    }
    msg = format_stop_loss_for_telegram(trigger)
    assert "🛑 손절 매도" in msg
    assert "디아이티" in msg
    assert "25,245" in msg
    assert "23,982" in msg
    assert "-5.00%" in msg
    assert "MVP-2.6" in msg


# === 6. 임계값 동적 변경 (env override) ===

def test_stop_loss_custom_threshold_3pct(fake_queue, monkeypatch):
    """ADAPTIVE_STOP_LOSS_PCT=3 → -3% 손절."""
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_ENABLED", "1")
    monkeypatch.setenv("ADAPTIVE_STOP_LOSS_PCT", "3")

    _setup_queue(fake_queue, "TEST007", actual_price=100_000, status="FILLED")
    broker = _mock_broker(current_price=96_500)  # -3.5% (3% 임계 초과)

    import importlib
    from src.use_cases import adaptive_stop_loss as sl
    importlib.reload(sl)

    triggers = sl.check_stop_loss_triggers(broker)
    assert len(triggers) == 1
    assert triggers[0]["loss_pct"] < -3
