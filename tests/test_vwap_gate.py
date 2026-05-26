"""VWAP 매수 게이트 단위 테스트 — H4 (5/26 작성).

검증 시나리오:
1. OVERHEAT (+2.5%) → 차단
2. DIP (-2.0%) → 통과 + is_dip=True
3. NORMAL (+0.5%) → 통과
4. DATA_MISSING (vwap_monitor.json에 없음) → fail-open
5. 임계값 정확히 경계
6. ticker zfill(6) 정규화
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.use_cases.vwap_gate import (
    check_vwap_buy_gate,
    format_gate_result,
)


@pytest.fixture
def mock_vwap_state(tmp_path, monkeypatch):
    """vwap_monitor.json 모의 데이터."""
    state = {
        "updated_at": "2026-05-26T10:30:00",
        "stocks": {
            "110990": {  # 디아이티 — DIP
                "name": "디아이티",
                "current_price": 27000,
                "vwap": 27500,
                "vwap_dev_pct": -1.82,
                "day_high": 28000,
                "day_low": 26800,
            },
            "062040": {  # 산일전기 — OVERHEAT
                "name": "산일전기",
                "current_price": 290000,
                "vwap": 282000,
                "vwap_dev_pct": 2.84,
                "day_high": 292000,
                "day_low": 280000,
            },
            "240810": {  # 원익IPS — NORMAL
                "name": "원익IPS",
                "current_price": 122000,
                "vwap": 121500,
                "vwap_dev_pct": 0.41,
            },
            "267270": {  # 경계 +2.0% 정확히
                "vwap_dev_pct": 2.0,
                "current_price": 1020,
                "vwap": 1000,
            },
            "272210": {  # 경계 -1.5% 정확히
                "vwap_dev_pct": -1.5,
                "current_price": 985,
                "vwap": 1000,
            },
        },
    }
    p = tmp_path / "vwap_monitor.json"
    p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("src.use_cases.vwap_gate.VWAP_STATE_PATH", p)
    return state


def test_overheat_block(mock_vwap_state):
    """OVERHEAT (+2.84%): 매수 차단."""
    r = check_vwap_buy_gate("062040")
    assert r["allow"] is False
    assert r["reason"] == "OVERHEAT_BLOCK"
    assert r["vwap_dev_pct"] == 2.84
    assert r["is_dip"] is False


def test_dip_bonus(mock_vwap_state):
    """DIP (-1.82%): 매수 허용 + 눌림목 우대."""
    r = check_vwap_buy_gate("110990")
    assert r["allow"] is True
    assert r["reason"] == "DIP_BONUS"
    assert r["is_dip"] is True


def test_normal(mock_vwap_state):
    """NORMAL (+0.41%): 매수 허용 정상."""
    r = check_vwap_buy_gate("240810")
    assert r["allow"] is True
    assert r["reason"] == "NORMAL"
    assert r["is_dip"] is False


def test_data_missing_fail_open(mock_vwap_state):
    """vwap_monitor.json에 없는 종목 → fail-open (allow=True)."""
    r = check_vwap_buy_gate("999999")  # 등록 X
    assert r["allow"] is True
    assert r["reason"] == "DATA_MISSING"


def test_no_vwap_file(monkeypatch, tmp_path):
    """vwap_monitor.json 자체가 없으면 → fail-open."""
    monkeypatch.setattr(
        "src.use_cases.vwap_gate.VWAP_STATE_PATH",
        tmp_path / "nonexistent.json",
    )
    r = check_vwap_buy_gate("110990")
    assert r["allow"] is True
    assert r["reason"] == "DATA_MISSING"


def test_boundary_overheat_inclusive(mock_vwap_state):
    """경계 +2.0%은 OVERHEAT_BLOCK (>=).

    267270은 dev=+2.0% 정확. >= 임계라 차단.
    """
    r = check_vwap_buy_gate("267270", overheat_threshold=2.0)
    assert r["allow"] is False
    assert r["reason"] == "OVERHEAT_BLOCK"


def test_boundary_dip_inclusive(mock_vwap_state):
    """경계 -1.5%은 DIP_BONUS (<=)."""
    r = check_vwap_buy_gate("272210", dip_threshold=-1.5)
    assert r["allow"] is True
    assert r["reason"] == "DIP_BONUS"
    assert r["is_dip"] is True


def test_custom_threshold(mock_vwap_state):
    """엄격 임계 (+1.0%)로 변경 시 NORMAL이었던 0.41%도 통과 유지."""
    r = check_vwap_buy_gate("240810", overheat_threshold=1.0)
    assert r["allow"] is True
    assert r["reason"] == "NORMAL"  # 0.41 < 1.0


def test_custom_threshold_strict_blocks(mock_vwap_state):
    """엄격 임계 (+0.3%)로 변경 시 NORMAL이었던 0.41%가 OVERHEAT."""
    r = check_vwap_buy_gate("240810", overheat_threshold=0.3)
    assert r["allow"] is False
    assert r["reason"] == "OVERHEAT_BLOCK"


def test_format_overheat(mock_vwap_state):
    """포맷 텍스트에 OVERHEAT 표시."""
    r = check_vwap_buy_gate("062040")
    msg = format_gate_result(r, "062040")
    assert "🛑" in msg
    assert "OVERHEAT" in msg
    assert "062040" in msg


def test_format_dip(mock_vwap_state):
    """포맷 텍스트에 DIP 표시."""
    r = check_vwap_buy_gate("110990")
    msg = format_gate_result(r, "110990")
    assert "💚" in msg
    assert "DIP" in msg


def test_ticker_normalization(mock_vwap_state):
    """5자리 ticker → zfill(6) 정규화."""
    # 062040은 6자리 그대로
    # 5자리로 들어와도 6자리로 normalize
    r = check_vwap_buy_gate("62040")  # 5자리
    assert r["reason"] == "OVERHEAT_BLOCK"  # 062040 hit
