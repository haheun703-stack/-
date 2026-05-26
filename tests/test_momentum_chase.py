"""모멘텀 추격 매수 단위 테스트 (5/26 MVP-6)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.use_cases.momentum_chase import (
    evaluate_momentum_signal,
    scan_momentum_candidates,
    format_momentum_for_telegram,
    MomentumSignal,
    _adjust_to_tick,
)


def _candle(open_p, close_p, vol, high=None, low=None):
    return {
        "open": open_p,
        "close": close_p,
        "high": high or max(open_p, close_p),
        "low": low or min(open_p, close_p),
        "volume": vol,
    }


# ─────────────────────────────────────────
# evaluate_momentum_signal
# ─────────────────────────────────────────
def test_fire_when_all_conditions_met():
    """모든 조건 만족 → fire=True."""
    # 평균 거래량 100 × 20봉
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    # 마지막 봉: 1000 → 1050 (+5%), 거래량 500 (5배)
    candles.append(_candle(1000, 1050, 500))

    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=5.0, current_price=1050,
    )
    assert sig.fire is True
    assert sig.five_min_pct == 5.0
    assert sig.vol_ratio == 5.0
    assert sig.target_price == 1050  # 1050 × 1.003 = 1053.15 → tick 5 = 1050
    assert sig.stop_price == 1015  # 1050 × 0.97 = 1018.5 → tick 5 = 1015
    assert sig.profit_target == 1100  # 1050 × 1.05 = 1102.5 → tick 5 = 1100


def test_no_fire_5min_pct_low():
    """5분봉 +2% (< 3%) → 미발화."""
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    candles.append(_candle(1000, 1020, 500))  # +2%
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=5.0, current_price=1020,
    )
    assert sig.fire is False
    assert "5분봉" in sig.reason


def test_no_fire_volume_low():
    """거래량 2배 (< 3배) → 미발화."""
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    candles.append(_candle(1000, 1050, 200))  # 거래량 2배만
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=5.0, current_price=1050,
    )
    assert sig.fire is False
    assert "거래량" in sig.reason


def test_no_fire_daily_overheated():
    """종일 +25% → 과열 회피."""
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    candles.append(_candle(1000, 1050, 500))
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=25.0, current_price=1050,
    )
    assert sig.fire is False
    assert "과열" in sig.reason


def test_no_fire_daily_too_weak():
    """종일 +0.5% (< 1%) → 약세 회피."""
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    candles.append(_candle(1000, 1050, 500))
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=0.5, current_price=1050,
    )
    assert sig.fire is False
    assert "약세" in sig.reason


def test_no_fire_negative_candle():
    """음봉 (close < open) → 미발화."""
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    candles.append(_candle(1050, 1020, 500))  # 음봉
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=5.0, current_price=1020,
    )
    assert sig.fire is False


def test_insufficient_candles():
    """5분봉 1개 → 데이터 부족."""
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=[_candle(1000, 1050, 500)],
        daily_pct=5.0, current_price=1050,
    )
    assert sig.fire is False
    assert "부족" in sig.reason


def test_zero_current_price():
    """현재가 0 → 부적합."""
    candles = [_candle(1000, 1000, 100) for _ in range(20)]
    candles.append(_candle(1000, 1050, 500))
    sig = evaluate_momentum_signal(
        ticker="067310", five_min_candles=candles,
        daily_pct=5.0, current_price=0,
    )
    assert sig.fire is False


def test_real_world_isc_scenario():
    """5/26 ISC 시나리오 시뮬: 5분봉 +4% + 거래량 5배 + 종일 +12%."""
    candles = [_candle(240000, 240000, 1000) for _ in range(20)]
    candles.append(_candle(240000, 249600, 5000))  # +4%, 5배
    sig = evaluate_momentum_signal(
        ticker="095340", name="ISC", five_min_candles=candles,
        daily_pct=12.0, current_price=249600,
    )
    assert sig.fire is True
    assert sig.five_min_pct == 4.0
    assert sig.vol_ratio == 5.0


# ─────────────────────────────────────────
# scan_momentum_candidates
# ─────────────────────────────────────────
def test_scan_disabled_returns_empty(monkeypatch):
    """ENABLED=False → 빈 리스트."""
    monkeypatch.setattr("src.use_cases.momentum_chase.ENABLED", False)
    adapter = MagicMock()
    sigs = scan_momentum_candidates(adapter, ["067310"])
    assert sigs == []


def test_scan_skips_protected(monkeypatch):
    """보호 종목은 평가 안 함."""
    monkeypatch.setattr("src.use_cases.momentum_chase.ENABLED", True)
    called_tickers = []
    def mock_fetch(t, period=5):
        called_tickers.append(t)
        return []
    adapter = MagicMock()
    adapter.fetch_minute_candles = mock_fetch
    sigs = scan_momentum_candidates(
        adapter, ["067310", "010120"],
        protected_tickers={"010120"},
    )
    assert sigs == []
    # 010120은 fetch_minute_candles 호출 안 됨
    assert "010120" not in called_tickers
    assert "067310" in called_tickers


def test_scan_fire_detected(monkeypatch):
    """모멘텀 발화 종목 정상 반환 (15분봉 추세 확인 포함)."""
    monkeypatch.setattr("src.use_cases.momentum_chase.ENABLED", True)
    candles_5min = [{"open": 1000, "close": 1000, "high": 1000, "low": 1000, "volume": 100}
                    for _ in range(20)]
    candles_5min.append({"open": 1000, "close": 1050, "high": 1050, "low": 1000, "volume": 500})

    # 15분봉: 우상향 양봉 3개 (P0-2 추세 검증 통과)
    candles_15min = [
        {"open": 1000, "close": 1020, "high": 1025, "low": 998, "volume": 300},
        {"open": 1020, "close": 1040, "high": 1045, "low": 1015, "volume": 350},
        {"open": 1040, "close": 1060, "high": 1065, "low": 1035, "volume": 400},
    ]

    def fetch_candles(t, period):
        if period == 5:
            return candles_5min
        elif period == 15:
            return candles_15min
        return []

    adapter = MagicMock()
    adapter.fetch_minute_candles = fetch_candles
    adapter.fetch_tick = lambda t: {"current_price": 1050, "change_pct": 5.0}

    sigs = scan_momentum_candidates(adapter, ["067310"])
    assert len(sigs) == 1
    assert sigs[0].fire
    assert sigs[0].ticker == "067310"


def test_scan_held_excluded(monkeypatch):
    """보유 종목 제외."""
    monkeypatch.setattr("src.use_cases.momentum_chase.ENABLED", True)
    adapter = MagicMock()
    adapter.fetch_minute_candles = lambda t, period: []
    sigs = scan_momentum_candidates(
        adapter, ["067310"], held_tickers={"067310"},
    )
    assert sigs == []


# ─────────────────────────────────────────
# format_momentum_for_telegram
# ─────────────────────────────────────────
def test_format_telegram():
    """텔레그램 포맷에 핵심 정보 포함."""
    sig = MomentumSignal(
        ticker="095340", name="ISC", fire=True,
        five_min_pct=4.0, vol_ratio=5.0, daily_pct=12.0,
        current_price=249600, target_price=250000,
        stop_price=242000, profit_target=262000,
        reason="test",
    )
    msg = format_momentum_for_telegram(sig)
    assert "ISC" in msg
    assert "095340" in msg
    assert "+4.00%" in msg or "+4.0" in msg
    assert "5.0x" in msg
    assert "+5%" in msg
    assert "-3%" in msg


def test_adjust_to_tick():
    """호가 단위 보정."""
    assert _adjust_to_tick(1050) == 1050  # 5원 단위 정수배
    assert _adjust_to_tick(1053) == 1050  # 1050 (round down)
    assert _adjust_to_tick(249600) == 249500  # 500원 단위 (200K~500K)
