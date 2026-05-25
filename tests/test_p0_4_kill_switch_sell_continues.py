"""P0-4 회귀 테스트 — KILL_SWITCH 활성 시에도 매도는 계속 진행.

배경 (5/24 bkit:code-analyzer 검수 P0-4):
  "KILL_SWITCH 발동 시 매도까지 정지 → 꺾이는 순간 매도 안 되어 손실 확대"

수정 (5/25):
  - src/use_cases/adaptive_position_manager.py: detect_peak_signal 천장 감지 계속
  - src/use_cases/adaptive_quick_profit.py: check_quick_profit_triggers 매도 계속

원칙:
  KILL_SWITCH는 "매수 차단"이 목적 (시장 패닉 추격 매수 방지).
  매도는 손실 차단이므로 KILL_SWITCH와 무관하게 항상 진행되어야 함.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


@pytest.fixture
def kill_switch_active():
    """KILL_SWITCH 강제 활성화 컨텍스트."""
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if not KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.write_text("test_p0_4", encoding="utf-8")
        created = True
    yield
    if created and KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()


def _fake_broker_with_peak(current: int, peak: int, peak_days_ago: int = 1):
    """현재가/천장가 mock broker."""
    from datetime import date, timedelta

    broker = MagicMock(name="MockBroker")

    # _fetch_current_price 우회 (직접 current_price 인자로 주입할 것이므로 broker는 OHLCV만)
    peak_date = (date.today() - timedelta(days=peak_days_ago)).strftime("%Y%m%d")
    today_str = date.today().strftime("%Y%m%d")

    rows = [
        {"stck_bsop_date": peak_date, "stck_hgpr": str(peak)},
        {"stck_bsop_date": today_str, "stck_hgpr": str(current)},
    ]
    # 5건 채우기 (detect_peak_signal min 5 요구)
    for i in range(2, 7):
        d = (date.today() - timedelta(days=i + 5)).strftime("%Y%m%d")
        rows.append({"stck_bsop_date": d, "stck_hgpr": str(int(peak * 0.85))})

    return broker, rows


# === 1. detect_peak_signal: KILL_SWITCH 활성 시 천장 감지 계속 ===

def test_detect_peak_signal_kill_switch_active_still_triggers(kill_switch_active, monkeypatch):
    """KILL_SWITCH 활성 + 천장 -3% 진입 → trigger=True 유지."""
    from src.use_cases import adaptive_position_manager as pm

    broker, rows = _fake_broker_with_peak(current=9700, peak=10000, peak_days_ago=2)
    monkeypatch.setattr(pm, "_fetch_recent_ohlcv", lambda b, t, d: rows)

    sig = pm.detect_peak_signal(broker, "TEST001", current_price=9700)

    # P0-4 보장: KILL_SWITCH 있어도 천장 감지 + trigger 발생
    assert sig.peak_price == 10000, f"천장 추출 실패: {sig.peak_price}"
    assert sig.trigger is True, (
        f"KILL_SWITCH 활성 시 trigger=False (매도 차단됨!) "
        f"reasons_fail={sig.reasons_fail}"
    )
    # KILL_SWITCH 메시지는 reasons_fail에 들어가지 않아야 함 (수정 전엔 들어갔었음)
    kill_sw_in_fail = any("KILL_SWITCH" in r for r in sig.reasons_fail)
    assert not kill_sw_in_fail, (
        f"KILL_SWITCH가 reasons_fail에 잘못 추가됨 (매도 차단됨): {sig.reasons_fail}"
    )


def test_detect_peak_signal_kill_switch_inactive_normal(monkeypatch):
    """KILL_SWITCH 비활성 + 동일 조건 → 정상 trigger (회귀)."""
    from src.use_cases import adaptive_position_manager as pm

    # KILL_SWITCH 확실 제거
    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()

    broker, rows = _fake_broker_with_peak(current=9700, peak=10000, peak_days_ago=2)
    monkeypatch.setattr(pm, "_fetch_recent_ohlcv", lambda b, t, d: rows)

    sig = pm.detect_peak_signal(broker, "TEST001", current_price=9700)
    assert sig.trigger is True


# === 2. check_quick_profit_triggers: KILL_SWITCH 활성 시 매도 계속 ===

def test_quick_profit_kill_switch_active_still_evaluates(kill_switch_active, tmp_path, monkeypatch):
    """KILL_SWITCH 활성 + FILLED stage + 현재가 ≥ +7% target → ARMED 전환됨 (매도 평가 계속)."""
    from src.use_cases import adaptive_quick_profit as qp
    from src.use_cases import adaptive_buy_queue as bq

    # 임시 queue 파일
    fake_queue = tmp_path / "queue.json"
    monkeypatch.setattr(bq, "QUEUE_PATH", fake_queue)

    # 가짜 큐 데이터: FILLED stage + +7% target
    queue_data = {
        "queues": {
            "TEST002": {
                "ticker": "TEST002",
                "stages": [
                    {
                        "level": 1,
                        "status": "FILLED",
                        "actual_price": 10000,
                        "quick_profit_target": 10700,  # +7%
                    }
                ],
            }
        }
    }
    import json
    fake_queue.write_text(json.dumps(queue_data), encoding="utf-8")

    # 현재가 11000 (target 10700 초과)
    monkeypatch.setattr(qp, "_fetch_current_price", lambda b, t: 11000)

    broker = MagicMock(name="MockBroker")
    triggers = qp.check_quick_profit_triggers(broker)

    # P0-4 보장: KILL_SWITCH 있어도 매도 평가 (ARMED 전환) 발생
    # triggers는 ARMED 전환된 stage 정보를 반환할 것
    assert len(triggers) > 0, (
        "KILL_SWITCH 활성 시 Quick Profit이 차단됨 (P0-4 위반) — "
        "매도 평가가 멈춰 ARMED 전환 0건"
    )


def test_quick_profit_kill_switch_inactive_normal(tmp_path, monkeypatch):
    """KILL_SWITCH 비활성 + 동일 조건 → 정상 ARMED 전환 (회귀)."""
    from src.use_cases import adaptive_quick_profit as qp
    from src.use_cases import adaptive_buy_queue as bq

    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()

    fake_queue = tmp_path / "queue.json"
    monkeypatch.setattr(bq, "QUEUE_PATH", fake_queue)

    queue_data = {
        "queues": {
            "TEST003": {
                "ticker": "TEST003",
                "stages": [
                    {
                        "level": 1,
                        "status": "FILLED",
                        "actual_price": 10000,
                        "quick_profit_target": 10700,
                    }
                ],
            }
        }
    }
    import json
    fake_queue.write_text(json.dumps(queue_data), encoding="utf-8")

    monkeypatch.setattr(qp, "_fetch_current_price", lambda b, t: 11000)

    broker = MagicMock(name="MockBroker")
    triggers = qp.check_quick_profit_triggers(broker)
    assert len(triggers) > 0
