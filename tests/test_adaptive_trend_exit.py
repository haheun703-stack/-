"""추세 이탈 매도 단위 테스트 — MVP-2.8 옵션 C (5/26 23:00)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.use_cases.adaptive_trend_exit import (
    evaluate_trend_exit,
    update_rsi_peak_state,
    scan_queue_for_trend_exit,
    execute_trend_exit,
    format_trend_exit_for_telegram,
    TrendExitSignal,
)


# ─────────────────────────────────────────
# 룰 A: MA 이탈
# ─────────────────────────────────────────
def test_ma_full_bear_immediate_exit():
    """역배열 (MA5<MA20<MA60) → 평단 무관 즉시 매도."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=9800, qty=1,
        ma5=9000, ma20=9500, ma60=10000, rsi=40,
    )
    assert sig.triggered
    assert sig.exit_type == "MA_FULL_BEAR"
    assert sig.pnl_pct == -2.0  # 손실 중이어도 매도


def test_ma_bear_long_with_profit():
    """대세 끝 (MA20<MA60) + 수익 0%+ → 매도."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10100, qty=1,
        ma5=10100, ma20=9500, ma60=10000, rsi=50,
    )
    assert sig.triggered
    assert sig.exit_type == "MA_BEAR_LONG"


def test_ma_bear_long_with_loss_no_exit():
    """대세 끝 + 손실 → 매도 X (MA 역배열까지 대기)."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=9500, qty=1,
        ma5=10000, ma20=9500, ma60=10000, rsi=40,
    )
    # MA5(10000) > MA20(9500) 이므로 역배열 X. MA20<MA60 but pnl -5% → 매도 X
    assert sig.triggered is False


def test_ma_bear_short_with_profit_5pct():
    """단기 추세 끝 (MA5<MA20) + 수익 +5%+ → 매도."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10500, qty=1,
        ma5=10400, ma20=10500, ma60=9800, rsi=60,
    )
    # ma5 < ma20 (10400 < 10500), ma20 > ma60 (10500 > 9800) — MA_BEAR_LONG 안 됨
    # 평단 +5% 이상 + MA5<MA20 → MA_BEAR_SHORT
    assert sig.triggered
    assert sig.exit_type == "MA_BEAR_SHORT"


def test_ma_bear_short_with_profit_3pct_no_exit():
    """단기 추세 끝 + 수익 +3% (5% 미달) → 매도 X."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10300, qty=1,
        ma5=10250, ma20=10400, ma60=9900, rsi=55,
    )
    # MA5<MA20, MA20>MA60, pnl +3% < 5% → 매도 X
    assert sig.triggered is False


# ─────────────────────────────────────────
# 룰 B: RSI 모멘텀 끝
# ─────────────────────────────────────────
def test_rsi_overheat_with_profit():
    """RSI 80+ 과열 + 수익 +3%+ → 매도."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10400, qty=1,
        ma5=10400, ma20=10300, ma60=10100, rsi=82,
    )
    assert sig.triggered
    assert sig.exit_type == "RSI_OVERHEATED"


def test_rsi_overheat_with_profit_2pct_no_exit():
    """RSI 85 but 수익 +2% (3% 미달) → 매도 X."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10200, qty=1,
        ma5=10200, ma20=10100, ma60=10000, rsi=85,
    )
    assert sig.triggered is False


def test_rsi_momentum_end_after_peak():
    """RSI 70+ 도달 기록 + 현재 RSI < 50 → 매도."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10200, qty=1,
        ma5=10200, ma20=10100, ma60=10000, rsi=45,
        rsi_peak_reached=True,  # 과거 70+ 도달
    )
    assert sig.triggered
    assert sig.exit_type == "RSI_MOMENTUM_END"


def test_rsi_no_peak_no_exit():
    """RSI 45 but peak 도달 X → 매도 X (조정 중일 뿐)."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10200, qty=1,
        ma5=10200, ma20=10100, ma60=10000, rsi=45,
        rsi_peak_reached=False,
    )
    assert sig.triggered is False


# ─────────────────────────────────────────
# update_rsi_peak_state
# ─────────────────────────────────────────
def test_rsi_peak_state_false_to_true():
    """RSI 70+ 도달 → True."""
    assert update_rsi_peak_state(75.0, False) is True


def test_rsi_peak_state_stays_true():
    """이미 True면 RSI 떨어져도 True 유지."""
    assert update_rsi_peak_state(45.0, True) is True


def test_rsi_peak_state_below_threshold():
    """RSI 65 (70 미달) + 이전 False → False 유지."""
    assert update_rsi_peak_state(65.0, False) is False


# ─────────────────────────────────────────
# 우선순위 (룰 충돌 시)
# ─────────────────────────────────────────
def test_priority_ma_full_bear_over_rsi():
    """역배열 + RSI 과열 동시 → MA_FULL_BEAR 우선."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10500, qty=1,
        ma5=9000, ma20=9500, ma60=10000, rsi=85,
    )
    assert sig.exit_type == "MA_FULL_BEAR"


def test_priority_ma_bear_long_over_short():
    """대세 끝 + 단기 끝 동시 → MA_BEAR_LONG 우선 (수익 0% 이상)."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10100, qty=1,
        ma5=9500, ma20=9700, ma60=10000, rsi=50,
    )
    # MA5(9500)<MA20(9700)<MA60(10000) → MA_FULL_BEAR (가장 우선)
    assert sig.exit_type == "MA_FULL_BEAR"


# ─────────────────────────────────────────
# 가드 케이스
# ─────────────────────────────────────────
def test_invalid_prices():
    """entry/current 0 → 매도 X."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=0, current_price=10000, qty=1,
        ma5=0, ma20=0, ma60=0, rsi=0,
    )
    assert sig.triggered is False


def test_zero_ma_skipped():
    """MA 데이터 0 → 룰 무시 (안전)."""
    sig = evaluate_trend_exit(
        ticker="067310", entry_price=10000, current_price=10500, qty=1,
        ma5=0, ma20=0, ma60=0, rsi=85,
    )
    # MA 룰 무시, RSI 과열 + 수익 5% → RSI_OVERHEATED
    assert sig.triggered
    assert sig.exit_type == "RSI_OVERHEATED"


# ─────────────────────────────────────────
# scan_queue_for_trend_exit
# ─────────────────────────────────────────
def test_scan_disabled_returns_empty(monkeypatch):
    """ENABLED=False → 빈 리스트."""
    monkeypatch.setattr("src.use_cases.adaptive_trend_exit.ENABLED", False)
    broker = MagicMock()
    sigs = scan_queue_for_trend_exit({"queues": {}}, broker)
    assert sigs == []


def test_scan_skips_non_filled(monkeypatch):
    """FILLED 상태만 평가."""
    monkeypatch.setattr("src.use_cases.adaptive_trend_exit.ENABLED", True)
    queue = {
        "queues": {
            "067310": {
                "stages": [
                    {"status": "PENDING", "actual_price": 0},
                    {"status": "FAILED", "actual_price": 0},
                ]
            }
        }
    }
    broker = MagicMock()
    sigs = scan_queue_for_trend_exit(queue, broker)
    assert sigs == []


# ─────────────────────────────────────────
# execute_trend_exit
# ─────────────────────────────────────────
def test_execute_sell_limit_success(monkeypatch):
    """sell_limit 성공."""
    monkeypatch.setenv("ADAPTIVE_SELL_USE_LIMIT", "1")
    sig = TrendExitSignal(
        ticker="067310", triggered=True, exit_type="MA_BEAR_SHORT",
        entry_price=10000, current_price=10500, qty=1, pnl_pct=5.0,
    )
    broker = MagicMock()
    mock_order = MagicMock()
    mock_order.order_id = "TREND001"
    broker.sell_limit = lambda t, p, q: mock_order
    r = execute_trend_exit(broker, sig)
    assert r["success"]
    # slippage 0.3% → 10500 × 0.997 = 10468
    assert r["limit_price"] == 10468


def test_execute_not_triggered():
    """trigger=False → 실패."""
    sig = TrendExitSignal(ticker="067310", triggered=False)
    r = execute_trend_exit(MagicMock(), sig)
    assert r["success"] is False


# ─────────────────────────────────────────
# format
# ─────────────────────────────────────────
def test_format_telegram():
    """텔레그램 포맷."""
    sig = TrendExitSignal(
        ticker="067310", triggered=True, exit_type="MA_FULL_BEAR",
        entry_price=10000, current_price=9500, qty=1, pnl_pct=-5.0,
        ma5=9000, ma20=9500, ma60=10000, rsi=40,
        reason="역배열 매도",
    )
    msg = format_trend_exit_for_telegram(sig)
    assert "🔴" in msg
    assert "MA_FULL_BEAR" in msg
    assert "067310" in msg
    assert "-5.00%" in msg
    assert "RSI=40" in msg or "RSI=40.0" in msg
