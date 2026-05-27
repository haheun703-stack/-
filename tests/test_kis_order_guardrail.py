"""P0 가드레일 단위 테스트 (kis_order_adapter._guard + auto_trading_volume).

12 케이스:
  1. ENABLED=0 매수 차단
  2. MAX_QTY 초과 차단
  3. WHITELIST 외 종목 차단
  4. 거래시간 외 차단
  5. 주말 차단
  6. 석가탄신일 (공휴일) 차단
  7. 부처님오신날 대체공휴일 (5/25) 차단
  8. 일일 매수 금액 한도 초과 차단
  9. 일일 매수 횟수 한도 초과 차단
  10. 지정가 현재가 ±5% 초과 차단
  11. 정상 케이스 통과
  12. record_buy → daily_volume.json 갱신
"""

from __future__ import annotations

import json
from datetime import date, datetime, time as dtime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.auto_trading_volume import (
    VOLUME_DIR,
    VOLUME_FILE,
    check_daily_limits,
    get_today_volume,
    record_buy,
)


# ──────────────────────────────────────────
# 고정 픽스처
# ──────────────────────────────────────────

@pytest.fixture
def clean_volume(tmp_path, monkeypatch):
    """data/auto_trading/daily_volume.json을 tmp_path로 격리."""
    test_dir = tmp_path / "auto_trading"
    test_file = test_dir / "daily_volume.json"
    monkeypatch.setattr("src.utils.auto_trading_volume.VOLUME_DIR", test_dir)
    monkeypatch.setattr("src.utils.auto_trading_volume.VOLUME_FILE", test_file)
    yield test_dir, test_file


@pytest.fixture
def trading_hour(monkeypatch):
    """now()를 거래시간 (5/26 화 10:30)으로 고정 + 거래일(5/26) 통과 보장."""
    fixed_dt = datetime(2026, 5, 26, 10, 30, 0)
    fixed_date = date(2026, 5, 26)
    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_dt
    class FakeDate(date):
        @classmethod
        def today(cls):
            return fixed_date
    monkeypatch.setattr("src.adapters.kis_order_adapter.datetime", FakeDateTime)
    monkeypatch.setattr("src.adapters.kis_order_adapter.date", FakeDate)
    # is_kr_trading_day가 내부적으로 date.today() 호출하므로 trading_calendar도 fake
    monkeypatch.setattr("src.trading_calendar.date", FakeDate)
    yield fixed_dt


@pytest.fixture
def adapter(monkeypatch, clean_volume, tmp_path):
    """KisOrderAdapter mock (mojito API 호출 차단)."""
    monkeypatch.setattr(
        "src.utils.trade_runtime_safety.KILL_SWITCH_PATHS",
        (tmp_path / "missing_KILL_SWITCH", tmp_path / "missing_kill_switch.flag"),
    )
    monkeypatch.setattr("mojito.KoreaInvestment", MagicMock())
    monkeypatch.setenv("MODEL", "REAL")
    from src.adapters.kis_order_adapter import KisOrderAdapter
    adp = KisOrderAdapter()
    # current_price는 종목 기준 mock (지정가 검증용)
    adp.fetch_current_price = MagicMock(return_value={"current_price": 10000})
    yield adp


# ──────────────────────────────────────────
# 1. ENABLED=0 차단
# ──────────────────────────────────────────

def test_disabled_blocks_order(adapter, monkeypatch, trading_hour):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "0")
    with pytest.raises(PermissionError, match="AUTO_TRADING_ENABLED=1 필수"):
        adapter._guard("487240", 1, price=10000, side="BUY")


# ──────────────────────────────────────────
# 2. MAX_QTY 초과 차단
# ──────────────────────────────────────────

def test_max_qty_exceeded(adapter, monkeypatch, trading_hour):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("AUTO_TRADING_MAX_QTY", "1")
    with pytest.raises(ValueError, match="수량 한도 초과"):
        adapter._guard("487240", 2, price=10000, side="BUY")


# ──────────────────────────────────────────
# 3. WHITELIST 외 종목 차단
# ──────────────────────────────────────────

def test_whitelist_blocks_outside(adapter, monkeypatch, trading_hour):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("AUTO_TRADING_WHITELIST_ONLY", "1")
    with pytest.raises(PermissionError, match="화이트리스트 외 종목"):
        adapter._guard("005930", 1, price=70000, side="BUY")  # 삼성전자 (whitelist에 없음)


# ──────────────────────────────────────────
# 4. 거래시간 외 차단
# ──────────────────────────────────────────

def test_outside_trading_hours(adapter, monkeypatch):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    fixed = datetime(2026, 5, 26, 18, 0, 0)  # 18시
    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed
    monkeypatch.setattr("src.adapters.kis_order_adapter.datetime", FakeDateTime)
    with pytest.raises(RuntimeError, match="거래시간 외"):
        adapter._guard("487240", 1, price=10000, side="BUY")


# ──────────────────────────────────────────
# 5-7. 휴장일 차단 (주말 / 공휴일 / 대체공휴일)
# ──────────────────────────────────────────

@pytest.mark.parametrize("test_date,label", [
    (date(2026, 5, 23), "토요일"),
    (date(2026, 5, 24), "석가탄신일 (일)"),
    (date(2026, 5, 25), "대체공휴일 (월)"),
])
def test_non_trading_day_blocked(adapter, monkeypatch, test_date, label):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    fixed = datetime.combine(test_date, dtime(10, 30, 0))
    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed
    class FakeDate(date):
        @classmethod
        def today(cls):
            return test_date
    monkeypatch.setattr("src.adapters.kis_order_adapter.datetime", FakeDateTime)
    monkeypatch.setattr("src.adapters.kis_order_adapter.date", FakeDate)
    monkeypatch.setattr("src.adapters.kis_order_adapter.is_kr_trading_day", lambda: False)
    with pytest.raises(RuntimeError, match="KRX 휴장일"):
        adapter._guard("487240", 1, price=10000, side="BUY")


# ──────────────────────────────────────────
# 8. 일일 매수 금액 한도 초과
# ──────────────────────────────────────────

def test_daily_amount_exceeded(adapter, monkeypatch, trading_hour, clean_volume):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("AUTO_TRADING_MAX_QTY", "10")  # 수량 한도는 충분
    monkeypatch.setenv("AUTO_TRADING_MAX_AMOUNT", "100000")
    # 사전에 90,000원 누적 (9주 × 10,000원)
    record_buy("487240", 9, 10000)
    # 추가 20,000원 시도 (2주 × 10,000원) → 누적 110,000 > 한도 100,000
    with pytest.raises(ValueError, match="일일 금액 한도 초과"):
        adapter._guard("487240", 2, price=10000, side="BUY")


# ──────────────────────────────────────────
# 9. 일일 매수 횟수 한도 초과
# ──────────────────────────────────────────

def test_daily_trades_exceeded(adapter, monkeypatch, trading_hour, clean_volume):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("AUTO_TRADING_MAX_AMOUNT", "10000000")  # 금액은 충분
    monkeypatch.setenv("AUTO_TRADING_MAX_TRADES_PER_DAY", "2")
    record_buy("487240", 1, 10000)
    record_buy("395160", 1, 10000)
    with pytest.raises(ValueError, match="일일 횟수 한도 초과"):
        adapter._guard("487240", 1, price=10000, side="BUY")


# ──────────────────────────────────────────
# 10. 지정가 현재가 ±5% 초과
# ──────────────────────────────────────────

def test_price_range_exceeded(adapter, monkeypatch, trading_hour):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("AUTO_TRADING_PRICE_RANGE_PCT", "5")
    # 현재가 10,000원, 지정가 11,000원 (편차 10%)
    with pytest.raises(ValueError, match="지정가 현재가"):
        adapter._guard("487240", 1, price=11000, side="BUY")


# ──────────────────────────────────────────
# 11. 정상 케이스 — 모든 가드 통과
# ──────────────────────────────────────────

def test_normal_pass(adapter, monkeypatch, trading_hour, clean_volume):
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("AUTO_TRADING_MAX_QTY", "1")
    monkeypatch.setenv("AUTO_TRADING_WHITELIST_ONLY", "1")
    monkeypatch.setenv("AUTO_TRADING_MAX_AMOUNT", "300000")
    monkeypatch.setenv("AUTO_TRADING_MAX_TRADES_PER_DAY", "5")
    # 화이트리스트 종목 (487240 KODEX AI전력), 수량 1, 가격 10,000 (현재가와 동일)
    adapter._guard("487240", 1, price=10000, side="BUY")  # 예외 없으면 통과


# ──────────────────────────────────────────
# 12. record_buy + get_today_volume 연동
# ──────────────────────────────────────────

def test_record_buy_updates_volume(clean_volume):
    test_dir, test_file = clean_volume
    record_buy("487240", 1, 12500)
    record_buy("395160", 1, 187500)
    v = get_today_volume()
    assert v["total_trades"] == 2
    assert v["total_amount"] == 200000
    assert len(v["buys"]) == 2
    assert v["buys"][0]["ticker"] == "487240"
    assert v["buys"][1]["amount"] == 187500
    # 파일 존재 확인
    assert test_file.exists()
    raw = json.loads(test_file.read_text(encoding="utf-8"))
    assert raw["total_amount"] == 200000


def test_check_daily_limits_pass_and_fail():
    # tmp_path 없이도 동작 (기본 file이 없는 케이스)
    ok, msg = check_daily_limits(50000, 300000, 5, today=date(2099, 1, 1))
    assert ok, msg
    ok, msg = check_daily_limits(400000, 300000, 5, today=date(2099, 1, 1))
    assert not ok
    assert "한도 초과" in msg
