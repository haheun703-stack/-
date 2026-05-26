"""adaptive_entry_gates 통합 단위 테스트 (5/26)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.use_cases.adaptive_entry_gates import (
    check_all_entry_gates,
    EntryGateResult,
    _fetch_ohlcv_safe,
)


@pytest.fixture
def mock_vwap_normal(tmp_path, monkeypatch):
    """vwap_monitor.json 정상 (NORMAL — 통과)."""
    state = {
        "stocks": {
            "067310": {
                "current_price": 25000,
                "vwap": 25000,
                "vwap_dev_pct": 0.0,
            }
        }
    }
    p = tmp_path / "vwap_monitor.json"
    p.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("src.use_cases.vwap_gate.VWAP_STATE_PATH", p)


@pytest.fixture
def mock_vwap_overheat(tmp_path, monkeypatch):
    """VWAP OVERHEAT (차단)."""
    state = {
        "stocks": {
            "067310": {
                "current_price": 26000,
                "vwap": 25000,
                "vwap_dev_pct": 4.0,  # +4% > +2% 임계 → 차단
            }
        }
    }
    p = tmp_path / "vwap_monitor.json"
    p.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("src.use_cases.vwap_gate.VWAP_STATE_PATH", p)


def _make_broker(orderbook=None, ohlcv=None):
    """mock broker."""
    broker = MagicMock()
    if ohlcv is not None:
        broker.fetch_ohlcv = MagicMock(return_value={"output2": ohlcv})
    else:
        broker.fetch_ohlcv = MagicMock(side_effect=AttributeError)
    return broker


def _make_intraday(orderbook=None):
    if orderbook is None:
        return None
    a = MagicMock()
    a.fetch_orderbook = MagicMock(return_value=orderbook)
    return a


def test_all_pass_no_atr(mock_vwap_normal, monkeypatch):
    """모든 게이트 통과 + ATR 비활성 → allow=True."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.USE_ATR_STOPS", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VOLUME_POWER_ENABLED", False)
    # 매물대용 OHLCV (넓은 가격대 — 목표 25000이 VAH 이내)
    # POC가 25000 근처 + VAH 26000 이상 되도록 분포
    ohlcv = [
        {"stck_bsop_date": f"2026020{i%10}",
         "stck_oprc": 24000 + (i % 20) * 100,
         "stck_hgpr": 24500 + (i % 20) * 100,
         "stck_lwpr": 23500 + (i % 20) * 100,
         "stck_clpr": 24000 + (i % 20) * 100,
         "acml_vol": 100000 if i % 5 == 0 else 200000}  # 25000 근처 거래량 많음
        for i in range(60)
    ]
    broker = _make_broker(ohlcv=ohlcv)
    # 호가창: 통과
    ob = {
        "asks": [{"price": 25050, "volume": 100}],
        "bids": [{"price": 25000, "volume": 100}],
        "total_ask_vol": 1000, "total_bid_vol": 1500, "bid_ask_ratio": 1.5,
    }
    intraday = _make_intraday(orderbook=ob)
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=intraday)
    assert r.allow
    assert r.block_reason == ""


def test_vwap_overheat_blocks(mock_vwap_overheat, monkeypatch):
    """VWAP OVERHEAT → allow=False."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=26000,
                                broker=broker, intraday_adapter=None)
    assert r.allow is False
    assert "VWAP" in r.block_reason
    assert r.vwap_reason == "OVERHEAT_BLOCK"


def test_orderbook_blocks_slippage(mock_vwap_normal, monkeypatch):
    """호가 슬리피지 +2% → 차단."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VOLUME_POWER_ENABLED", False)
    ob = {
        "asks": [{"price": 25500, "volume": 100}],  # +2% 슬리피지
        "bids": [{"price": 24800, "volume": 100}],
        "total_ask_vol": 1000, "total_bid_vol": 800, "bid_ask_ratio": 0.8,
    }
    intraday = _make_intraday(orderbook=ob)
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=intraday)
    assert r.allow is False
    assert "ORDERBOOK" in r.block_reason


def test_supply_zone_overheat_blocks(mock_vwap_normal, monkeypatch):
    """매물대 VAH 과열 → 차단."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VOLUME_POWER_ENABLED", False)
    # 9000~10000 분포에서 11000 매수 시도 → VAH 과열
    ohlcv = [
        {"stck_bsop_date": f"2026020{i%10}", "stck_oprc": 9000 + i*10,
         "stck_hgpr": 9100 + i*10, "stck_lwpr": 8900 + i*10,
         "stck_clpr": 9000 + i*10, "acml_vol": 100000}
        for i in range(60)
    ]
    broker = _make_broker(ohlcv=ohlcv)
    r = check_all_entry_gates("067310", target_price=11000,
                                broker=broker, intraday_adapter=None)
    assert r.allow is False
    assert "SUPPLY" in r.block_reason


def test_intraday_adapter_none_skips_orderbook(mock_vwap_normal, monkeypatch):
    """intraday_adapter=None → H5 호가 게이트 skip."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VOLUME_POWER_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.USE_ATR_STOPS", False)
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=None)
    assert r.allow
    assert r.orderbook_reason == "DISABLED"


def test_disabled_gates(mock_vwap_normal, monkeypatch):
    """게이트 전부 비활성 → allow=True (skip all)."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VWAP_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VOLUME_POWER_ENABLED", False)
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=None)
    assert r.allow
    assert r.vwap_reason == "DISABLED"


def test_ohlcv_fetch_failure_skips_supply(mock_vwap_normal, monkeypatch):
    """OHLCV 조회 실패 → 매물대 skip (fail-open)."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_VOLUME_POWER_ENABLED", False)
    broker = MagicMock()
    broker.fetch_ohlcv = MagicMock(side_effect=Exception("API down"))
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=None)
    assert r.allow  # fail-open
    # supply_zone_reason은 "ERROR_SKIP" 또는 "DATA_MISSING"


def test_fetch_ohlcv_safe_returns_empty_on_no_attr():
    """broker에 fetch_ohlcv 없으면 빈 리스트."""
    broker = MagicMock(spec=[])  # 메서드 없음
    bars = _fetch_ohlcv_safe(broker, "067310", 60)
    assert bars == []


def test_volume_power_weak_blocks(mock_vwap_normal, monkeypatch):
    """체결강도 100 미만 → WEAK_BUY 차단."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    # _fetch_volume_power mock: 50 반환
    monkeypatch.setattr(
        "src.use_cases.entry_gates._fetch_volume_power",
        lambda b, t: (50.0, "ccnl_tday_rltv"),
    )
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=None)
    assert r.allow is False
    assert "VOLUME_POWER" in r.block_reason
    assert r.volume_power == 50.0


def test_volume_power_strong_passes(mock_vwap_normal, monkeypatch):
    """체결강도 150 → STRONG_BUY 통과."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    monkeypatch.setattr(
        "src.use_cases.entry_gates._fetch_volume_power",
        lambda b, t: (150.0, "ccnl_tday_rltv"),
    )
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=None)
    assert r.allow
    assert r.volume_power_reason == "STRONG_BUY"
    assert r.volume_power == 150.0


def test_volume_power_fetch_fail_fail_open(mock_vwap_normal, monkeypatch):
    """체결강도 fetch 실패 (vp=0) → fail-open (allow=True)."""
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_SUPPLY_ZONE_ENABLED", False)
    monkeypatch.setattr("src.use_cases.adaptive_entry_gates.GATE_ORDERBOOK_ENABLED", False)
    monkeypatch.setattr(
        "src.use_cases.entry_gates._fetch_volume_power",
        lambda b, t: (0.0, "none"),
    )
    broker = MagicMock()
    r = check_all_entry_gates("067310", target_price=25000,
                                broker=broker, intraday_adapter=None)
    assert r.allow  # fail-open
    assert r.volume_power_reason == "FETCH_FAILED"


def test_fetch_ohlcv_safe_parses_kis_format():
    """KIS output2 형식 정상 파싱."""
    raw = {
        "output2": [
            {"stck_bsop_date": "20260301", "stck_oprc": "24000",
             "stck_hgpr": "24500", "stck_lwpr": "23500", "stck_clpr": "24000",
             "acml_vol": "100000"},
        ] * 70
    }
    broker = MagicMock()
    broker.fetch_ohlcv = MagicMock(return_value=raw)
    bars = _fetch_ohlcv_safe(broker, "067310", 60)
    assert len(bars) == 60  # lookback_days 만큼
    assert bars[0]["open"] == 24000.0
    assert bars[0]["volume"] == 100000
