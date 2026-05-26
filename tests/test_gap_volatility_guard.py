"""갭/변동성 가드 단위 테스트 (P0-3, 5/26 23:45)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.use_cases.gap_volatility_guard import (
    check_kospi_volatility_surge,
    check_foreign_dump,
    check_gap_down_held,
    evaluate_market_guard,
    format_guard_for_telegram,
    GuardResult,
)


# ─────────────────────────────────────────
# check_kospi_volatility_surge
# ─────────────────────────────────────────
def test_volatility_normal():
    """변동성 1% (정상) → 미발화."""
    adapter = MagicMock()
    adapter.fetch_minute_candles = lambda t, period: [
        {"open": 2500, "high": 2520, "low": 2495, "close": 2510, "volume": 1000}
    ]
    surge, pct = check_kospi_volatility_surge(adapter)
    assert surge is False
    assert 0.9 < pct < 1.1


def test_volatility_surge_3pct():
    """변동성 3.5% → 발화."""
    adapter = MagicMock()
    adapter.fetch_minute_candles = lambda t, period: [
        {"open": 2500, "high": 2545, "low": 2460, "close": 2510, "volume": 1000}
    ]
    # (2545 - 2460) / 2500 = 3.4%
    surge, pct = check_kospi_volatility_surge(adapter)
    assert surge is True
    assert pct >= 3.0


def test_volatility_no_candles():
    """캔들 없음 → False."""
    adapter = MagicMock()
    adapter.fetch_minute_candles = lambda t, period: []
    surge, pct = check_kospi_volatility_surge(adapter)
    assert surge is False
    assert pct == 0.0


def test_volatility_none_adapter():
    """adapter None → False."""
    surge, pct = check_kospi_volatility_surge(None)
    assert surge is False


# ─────────────────────────────────────────
# check_foreign_dump
# ─────────────────────────────────────────
def test_foreign_dump_threshold():
    """외인 -15,000주 (임계 -10,000) → 발화."""
    adapter = MagicMock()
    adapter.fetch_investor_flow = lambda t: {"foreign_net_buy": -15000}
    dump, foreign = check_foreign_dump(adapter)
    assert dump is True
    assert foreign == -15000


def test_foreign_normal():
    """외인 -3,000주 (정상) → 미발화."""
    adapter = MagicMock()
    adapter.fetch_investor_flow = lambda t: {"foreign_net_buy": -3000}
    dump, foreign = check_foreign_dump(adapter)
    assert dump is False


def test_foreign_buying():
    """외인 +5,000주 (매수) → 미발화."""
    adapter = MagicMock()
    adapter.fetch_investor_flow = lambda t: {"foreign_net_buy": 5000}
    dump, foreign = check_foreign_dump(adapter)
    assert dump is False
    assert foreign == 5000


# ─────────────────────────────────────────
# check_gap_down_held
# ─────────────────────────────────────────
def test_gap_down_held():
    """평단 120,900 / 시초가 110,000 → 갭 -9% 발화."""
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_oprc": "110000"}}
    held = [{"ticker": "240810", "avg_price": 120900}]
    has, ticker, pct = check_gap_down_held(broker, held)
    assert has is True
    assert ticker == "240810"
    assert pct < -5


def test_gap_down_safe():
    """평단 120,900 / 시초가 119,000 → -1.6% (정상)."""
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_oprc": "119000"}}
    held = [{"ticker": "240810", "avg_price": 120900}]
    has, ticker, pct = check_gap_down_held(broker, held)
    assert has is False


def test_gap_down_no_holdings():
    """보유 0 → 발화 X."""
    broker = MagicMock()
    has, ticker, pct = check_gap_down_held(broker, [])
    assert has is False


def test_gap_down_worst_selected():
    """여러 보유 종목 중 가장 큰 갭 하락 선택. ticker zfill 6 처리."""
    broker = MagicMock()
    # ticker zfill 후 (000A0F → 6자리)
    prices = {"00000A": 95000, "00000B": 80000, "00000C": 99000}
    broker.fetch_price = lambda t: {"output": {"stck_oprc": str(prices.get(t, 0))}}
    held = [
        {"ticker": "A", "avg_price": 100000},  # -5%
        {"ticker": "B", "avg_price": 100000},  # -20% ★
        {"ticker": "C", "avg_price": 100000},  # -1%
    ]
    has, ticker, pct = check_gap_down_held(broker, held)
    assert has is True
    assert ticker == "00000B"
    assert pct == -20.0


# ─────────────────────────────────────────
# evaluate_market_guard
# ─────────────────────────────────────────
def test_evaluate_disabled(monkeypatch):
    """ENABLED=False → 모든 가드 통과."""
    monkeypatch.setattr("src.use_cases.gap_volatility_guard.ENABLED", False)
    r = evaluate_market_guard(MagicMock(), MagicMock(), [])
    assert r.block_new_buy is False
    assert r.force_sell_held is False


def test_evaluate_all_clear(monkeypatch):
    """모든 가드 통과 → 정상."""
    monkeypatch.setattr("src.use_cases.gap_volatility_guard.ENABLED", True)
    intraday = MagicMock()
    intraday.fetch_minute_candles = lambda t, period: [
        {"open": 2500, "high": 2510, "low": 2495, "close": 2505, "volume": 100}
    ]
    intraday.fetch_investor_flow = lambda t: {"foreign_net_buy": 1000}
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_oprc": "120000"}}
    r = evaluate_market_guard(broker, intraday, [
        {"ticker": "240810", "avg_price": 120900}
    ])
    assert r.block_new_buy is False
    assert r.force_sell_held is False
    assert r.reason == "정상"


def test_evaluate_gap_down(monkeypatch):
    """갭 하락 발화."""
    monkeypatch.setattr("src.use_cases.gap_volatility_guard.ENABLED", True)
    intraday = MagicMock()
    intraday.fetch_minute_candles = lambda t, period: [
        {"open": 2500, "high": 2510, "low": 2495, "close": 2505, "volume": 100}
    ]
    intraday.fetch_investor_flow = lambda t: {"foreign_net_buy": 1000}
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_oprc": "110000"}}
    r = evaluate_market_guard(broker, intraday, [
        {"ticker": "240810", "avg_price": 120900}
    ])
    assert r.force_sell_held is True
    assert "갭 하락" in r.reason


def test_evaluate_volatility_surge(monkeypatch):
    """KOSPI 변동성 폭증 → 신규 매수 차단."""
    monkeypatch.setattr("src.use_cases.gap_volatility_guard.ENABLED", True)
    intraday = MagicMock()
    intraday.fetch_minute_candles = lambda t, period: [
        {"open": 2500, "high": 2580, "low": 2420, "close": 2510, "volume": 100}
    ]  # 변동성 6.4%
    intraday.fetch_investor_flow = lambda t: {"foreign_net_buy": 1000}
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_oprc": "120500"}}
    r = evaluate_market_guard(broker, intraday, [
        {"ticker": "240810", "avg_price": 120900}
    ])
    assert r.block_new_buy is True
    assert "변동성" in r.reason


def test_evaluate_foreign_dump(monkeypatch):
    """외인 대량 매도 → 차단."""
    monkeypatch.setattr("src.use_cases.gap_volatility_guard.ENABLED", True)
    intraday = MagicMock()
    intraday.fetch_minute_candles = lambda t, period: [
        {"open": 2500, "high": 2510, "low": 2495, "close": 2505, "volume": 100}
    ]
    intraday.fetch_investor_flow = lambda t: {"foreign_net_buy": -50000}
    broker = MagicMock()
    broker.fetch_price = lambda t: {"output": {"stck_oprc": "120500"}}
    r = evaluate_market_guard(broker, intraday, [
        {"ticker": "240810", "avg_price": 120900}
    ])
    assert r.block_new_buy is True
    assert "외인" in r.reason


# ─────────────────────────────────────────
# format
# ─────────────────────────────────────────
def test_format_no_guard():
    """가드 미발화 → 빈 문자열."""
    r = GuardResult()
    msg = format_guard_for_telegram(r)
    assert msg == ""


def test_format_gap_down():
    """갭 하락 텔레그램."""
    r = GuardResult(
        force_sell_held=True, gap_down_ticker="240810", gap_down_pct=-8.5,
        reason="갭 하락 240810 -8.50%",
    )
    msg = format_guard_for_telegram(r)
    assert "🚨" in msg
    assert "240810" in msg
    assert "-8.5" in msg
