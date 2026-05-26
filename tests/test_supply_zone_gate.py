"""H6 매물대 게이트 단위 테스트 (5/26)."""
from __future__ import annotations

import pytest

from src.use_cases.supply_zone_gate import (
    calc_supply_zones,
    check_supply_zone_buy_gate,
    SupplyZoneProfile,
    SupplyZoneGate,
)


def _bars(prices_volumes):
    """간단한 bar 빌더. (close_price, volume) → bar dict."""
    return [
        {"open": p, "high": p + 100, "low": p - 100, "close": p, "volume": v}
        for p, v in prices_volumes
    ]


# ─────────────────────────────────────────
# calc_supply_zones
# ─────────────────────────────────────────
def test_poc_at_max_volume_bin():
    """최대 거래량 가격대가 POC."""
    # 10000원에 가장 많은 거래량 → POC = 10000 근처
    bars = _bars([
        (9000, 100), (9500, 200), (10000, 1000), (10500, 200), (11000, 100),
    ])
    profile = calc_supply_zones(bars, bins=10)
    assert profile is not None
    assert 9800 <= profile.poc_price <= 10200


def test_value_area_70pct():
    """VAH-VAL 사이가 70% 거래량."""
    bars = _bars([(p, 100) for p in range(9000, 11000, 100)])  # 균등 분포
    profile = calc_supply_zones(bars, bins=20, value_area_pct=70)
    assert profile is not None
    assert profile.val_price < profile.poc_price <= profile.vah_price


def test_insufficient_data():
    """OHLCV 5개 미만 → None."""
    bars = _bars([(10000, 100), (10100, 100)])
    profile = calc_supply_zones(bars)
    assert profile is None


def test_empty_data():
    """빈 리스트 → None."""
    assert calc_supply_zones([]) is None


def test_zero_volume_skipped():
    """volume=0 bar는 무시."""
    bars = [
        {"open": 10000, "high": 10100, "low": 9900, "close": 10000, "volume": 0},
        {"open": 10100, "high": 10200, "low": 10000, "close": 10100, "volume": 1000},
        {"open": 10000, "high": 10100, "low": 9900, "close": 10000, "volume": 500},
        {"open": 10000, "high": 10100, "low": 9900, "close": 10000, "volume": 500},
        {"open": 10000, "high": 10100, "low": 9900, "close": 10000, "volume": 500},
    ]
    profile = calc_supply_zones(bars, bins=10)
    assert profile is not None
    assert profile.total_volume > 0


def test_zones_top5_returned():
    """zones 리스트는 상위 5개."""
    bars = _bars([(9000 + i * 100, 100 + i) for i in range(20)])
    profile = calc_supply_zones(bars, bins=20)
    assert profile is not None
    assert len(profile.zones) == 5


# ─────────────────────────────────────────
# check_supply_zone_buy_gate
# ─────────────────────────────────────────
def _make_profile(poc=10000, vah=10500, val=9500):
    return SupplyZoneProfile(
        poc_price=poc, vah_price=vah, val_price=val,
        total_volume=10000, bin_count=20, zones=[],
    )


def test_poc_breakout():
    """현재가 > POC × 1.005 → POC_BREAKOUT 우대."""
    profile = _make_profile(poc=10000, vah=10500, val=9500)
    r = check_supply_zone_buy_gate(current_price=10100, profile=profile)
    assert r.allow
    assert r.reason == "POC_BREAKOUT"
    assert r.is_breakout


def test_poc_support():
    """현재가 POC 부근 (±0.5%) → POC_SUPPORT."""
    profile = _make_profile(poc=10000)
    r = check_supply_zone_buy_gate(current_price=10010, profile=profile)
    assert r.allow
    assert r.reason == "POC_SUPPORT"


def test_inside_va():
    """현재가 VAL~VAH 사이 (POC 외) → INSIDE_VA."""
    profile = _make_profile(poc=10000, vah=10500, val=9500)
    r = check_supply_zone_buy_gate(current_price=9700, profile=profile)
    assert r.allow
    assert r.reason == "INSIDE_VA"


def test_vah_overheated():
    """현재가 > VAH × 1.02 → 차단."""
    profile = _make_profile(poc=10000, vah=10500, val=9500)
    r = check_supply_zone_buy_gate(current_price=10800, profile=profile)
    assert r.allow is False
    assert r.reason == "VAH_OVERHEATED"


def test_val_breakdown():
    """현재가 < VAL × 0.98 → 차단."""
    profile = _make_profile(poc=10000, vah=10500, val=9500)
    r = check_supply_zone_buy_gate(current_price=9200, profile=profile)
    assert r.allow is False
    assert r.reason == "VAL_BREAKDOWN"


def test_data_missing_fail_open():
    """profile=None → fail-open."""
    r = check_supply_zone_buy_gate(current_price=10000, profile=None)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_zero_current_price():
    """현재가 0 → DATA_MISSING."""
    profile = _make_profile()
    r = check_supply_zone_buy_gate(current_price=0, profile=profile)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_custom_breakout_threshold():
    """엄격 임계 +0.1% → +0.5% 종목도 POC_BREAKOUT."""
    profile = _make_profile(poc=10000)
    r = check_supply_zone_buy_gate(
        current_price=10050, profile=profile, breakout_pct=0.1,
    )
    assert r.reason == "POC_BREAKOUT"


def test_integration_with_calc_zones():
    """calc_supply_zones + check_supply_zone_buy_gate 통합."""
    # 9000~11000 분포 + 10000에 집중
    bars = _bars([
        (9000, 50), (9500, 100), (10000, 2000), (10500, 100), (11000, 50),
    ])
    profile = calc_supply_zones(bars, bins=10)
    assert profile is not None

    # 현재가 = POC 근처
    r = check_supply_zone_buy_gate(current_price=int(profile.poc_price), profile=profile)
    assert r.allow
    assert r.reason in ("POC_SUPPORT", "POC_BREAKOUT", "INSIDE_VA")
