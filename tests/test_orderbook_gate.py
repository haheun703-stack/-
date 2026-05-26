"""H5 호가 게이트 단위 테스트 (5/26)."""
from __future__ import annotations

import pytest

from src.use_cases.orderbook_gate import (
    check_orderbook_buy_gate,
    OrderbookGate,
)


def _book(best_ask, best_bid, total_ask=1000, total_bid=1000):
    """간단한 orderbook 빌더."""
    return {
        "asks": [{"price": best_ask, "volume": total_ask // 10}],
        "bids": [{"price": best_bid, "volume": total_bid // 10}],
        "total_ask_vol": total_ask,
        "total_bid_vol": total_bid,
        "bid_ask_ratio": total_bid / total_ask if total_ask else 0,
    }


def test_normal_pass():
    """일반 매수: 매수1 9990 / 매도1 10010 / 목표 10000."""
    ob = _book(best_ask=10010, best_bid=9990)
    r = check_orderbook_buy_gate(ob, target_price=10000)
    assert r.allow
    assert r.reason == "NORMAL"
    assert r.spread_pct < 1.0


def test_slippage_too_wide_blocks():
    """매도1 10100 / 목표 10000 → 슬리피지 +1% > 0.5% → 차단."""
    ob = _book(best_ask=10100, best_bid=10000)
    r = check_orderbook_buy_gate(ob, target_price=10000)
    assert r.allow is False
    assert r.reason == "SLIPPAGE_TOO_WIDE"


def test_spread_wide_blocks():
    """스프레드 매수 9000 → 매도 9200 = +2.2% > 1.0% → 차단."""
    ob = _book(best_ask=9200, best_bid=9000)
    r = check_orderbook_buy_gate(ob, target_price=9100)
    # slippage = (9200-9100)/9100 = +1.1% > 0.5% → SLIPPAGE_TOO_WIDE 먼저
    assert r.allow is False


def test_spread_wide_blocks_with_low_target():
    """슬리피지는 OK인데 스프레드만 큰 케이스."""
    # 목표가를 매도1과 정확히 같게 설정 — 슬리피지 0%, 스프레드만 검증
    ob = _book(best_ask=10000, best_bid=9890)
    # spread = (10000-9890)/9890 = +1.11% > 1.0% → SPREAD_WIDE
    r = check_orderbook_buy_gate(ob, target_price=10000)
    assert r.allow is False
    assert r.reason == "SPREAD_WIDE"


def test_bid_thin_blocks():
    """매수 잔량 500 / 매도 2000 → ratio 0.25 < 0.5 → 차단."""
    ob = _book(best_ask=10010, best_bid=9990, total_ask=2000, total_bid=500)
    r = check_orderbook_buy_gate(ob, target_price=10000)
    assert r.allow is False
    assert r.reason == "BID_THIN"


def test_bid_strong_marker():
    """매수 잔량 5000 / 매도 1500 → ratio 3.33 >= 2.0 → 통과 + BID_STRONG."""
    ob = _book(best_ask=10010, best_bid=9990, total_ask=1500, total_bid=5000)
    r = check_orderbook_buy_gate(ob, target_price=10000)
    assert r.allow
    assert r.reason == "BID_STRONG"
    assert r.is_strong_bid


def test_data_missing_fail_open():
    """빈 orderbook → fail-open (allow=True)."""
    r = check_orderbook_buy_gate({"asks": [], "bids": []}, target_price=10000)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_none_orderbook():
    """None 입력 → fail-open."""
    r = check_orderbook_buy_gate(None, target_price=10000)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_zero_best_prices():
    """매수1=0/매도1=0 → DATA_MISSING."""
    ob = {"asks": [{"price": 0, "volume": 0}], "bids": [{"price": 0, "volume": 0}]}
    r = check_orderbook_buy_gate(ob, target_price=10000)
    assert r.allow
    assert r.reason == "DATA_MISSING"


def test_custom_thresholds():
    """엄격한 슬리피지 임계 0.1%로 변경."""
    ob = _book(best_ask=10020, best_bid=9990)
    r = check_orderbook_buy_gate(ob, target_price=10000, slippage_max_pct=0.1)
    # slippage = +0.2% > 0.1% → 차단
    assert r.allow is False
    assert r.reason == "SLIPPAGE_TOO_WIDE"


def test_real_world_dia_iti():
    """디아이티 시나리오: 목표 25,245 / 매수1 25,200 / 매도1 25,300."""
    ob = _book(best_ask=25300, best_bid=25200, total_ask=5000, total_bid=8000)
    r = check_orderbook_buy_gate(ob, target_price=25245)
    # slippage = (25300-25245)/25245 = +0.218% < 0.5% → 통과
    # spread = (25300-25200)/25200 = +0.397% < 1.0%
    # ratio = 8000/5000 = 1.6 → NORMAL (1.6 < 2.0)
    assert r.allow
    assert r.reason == "NORMAL"
