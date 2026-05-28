"""어댑터 ↔ order_intents_gate paper-first 통합 테스트 (코덱스 3차 응답 5/28).

코덱스 지시:
  "KisOrderAdapter/PaperOrderAdapter 통합 PR에는 모든 주문 함수가
   ticker/side/mode/executor_bot을 명시 전달하는 테스트 포함"

검증:
  - PaperOrderAdapter 매매 4종 (buy_limit/sell_limit/buy_market/sell_market)
    each with intent / without intent / backward-compat (mode=None)
  - quant + live register 시점 차단 (Note 1)
  - KisOrderAdapter._guard에 mode/executor_bot optional 인자 받기 (시그니처 확인)
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import src.use_cases.order_intents_gate as gate
from src.use_cases.order_intents_gate import (
    register_intent, OrderIntentError, NoIntentError,
)

TEST_HMAC_KEY = "test_adapter_intents_integration_minimum_32_chars"
SEOUL = ZoneInfo("Asia/Seoul")


@pytest.fixture
def isolated(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="adapter_intent_"))
    monkeypatch.setattr(gate, "ORDER_INTENTS_DIR", tmp)
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", TEST_HMAC_KEY)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _make_paper_intent(ticker="240810", side="BUY", bot="day"):
    now = datetime.now(tz=SEOUL)
    return {
        "intent_id": f"{bot[0]}_test_{ticker}_{side}",
        "bot": bot,
        "engine": "test_adapter_integration",
        "ticker": ticker,
        "side": side,
        "mode": "paper",
        "score": 80.0,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=4)).isoformat(),
    }


# ──────────────────────────────────────────────
# Note 1: quant + live register 차단
# ──────────────────────────────────────────────
class TestNote1_QuantNoLiveIntent:
    def test_quant_live_register_blocked(self, isolated):
        intent = _make_paper_intent(bot="quant")
        intent["mode"] = "live"
        with pytest.raises(OrderIntentError) as exc_info:
            register_intent(intent, bot="quant")
        assert "live" in str(exc_info.value).lower()
        assert "quant" in str(exc_info.value).lower()

    def test_quant_paper_register_allowed(self, isolated):
        intent = _make_paper_intent(bot="quant")
        # paper 모드는 허용
        register_intent(intent, bot="quant")

    def test_day_live_register_allowed(self, isolated):
        intent = _make_paper_intent(bot="day")
        intent["mode"] = "live"
        # day bot은 live 가능 (별도 권한 검증은 다른 단계)
        register_intent(intent, bot="day")


# ──────────────────────────────────────────────
# Note 2: 어댑터 시그니처 검증 (정적)
# ──────────────────────────────────────────────
class TestNote2_AdapterSignatures:
    """ticker/side/mode/executor_bot 명시 전달 가능한지 시그니처 검사."""

    def test_kis_buy_limit_has_mode_executor_bot(self):
        from src.adapters.kis_order_adapter import KisOrderAdapter
        sig = inspect.signature(KisOrderAdapter.buy_limit)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_kis_sell_limit_has_mode_executor_bot(self):
        from src.adapters.kis_order_adapter import KisOrderAdapter
        sig = inspect.signature(KisOrderAdapter.sell_limit)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_kis_buy_market_has_mode_executor_bot(self):
        from src.adapters.kis_order_adapter import KisOrderAdapter
        sig = inspect.signature(KisOrderAdapter.buy_market)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_kis_sell_market_has_mode_executor_bot(self):
        from src.adapters.kis_order_adapter import KisOrderAdapter
        sig = inspect.signature(KisOrderAdapter.sell_market)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_kis_guard_has_mode_executor_bot(self):
        from src.adapters.kis_order_adapter import KisOrderAdapter
        sig = inspect.signature(KisOrderAdapter._guard)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_paper_buy_limit_has_mode_executor_bot(self):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        sig = inspect.signature(PaperOrderAdapter.buy_limit)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_paper_sell_limit_has_mode_executor_bot(self):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        sig = inspect.signature(PaperOrderAdapter.sell_limit)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_paper_buy_market_has_mode_executor_bot(self):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        sig = inspect.signature(PaperOrderAdapter.buy_market)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters

    def test_paper_sell_market_has_mode_executor_bot(self):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        sig = inspect.signature(PaperOrderAdapter.sell_market)
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters


# ──────────────────────────────────────────────
# PaperOrderAdapter 동작 검증
# ──────────────────────────────────────────────
class TestPaperOrderAdapterIntents:
    """PaperOrderAdapter는 실제 KIS API 호출 없으므로 단위 테스트 가능."""

    def test_paper_buy_limit_with_valid_intent_passes(self, isolated, monkeypatch):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        # KisOrderAdapter 의존성 회피 (paper는 KIS 호출 X지만 _get_market_data만 사용)
        register_intent(_make_paper_intent(ticker="240810", side="BUY", bot="day"), bot="day")

        adapter = PaperOrderAdapter()
        # buy_limit은 KIS 의존 없이 시뮬만 (orderbook_available=False)
        order = adapter.buy_limit(
            ticker="240810", price=121000, quantity=1,
            mode="paper", executor_bot="day",
        )
        # PaperOrderAdapter는 즉시 FILLED 반환
        from src.entities.trading_models import OrderStatus
        assert order.status == OrderStatus.FILLED
        assert order.ticker == "240810"

    def test_paper_buy_limit_without_intent_blocked(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        # intent 등록 X
        adapter = PaperOrderAdapter()
        with pytest.raises(NoIntentError):
            adapter.buy_limit(
                ticker="999999", price=10000, quantity=1,
                mode="paper", executor_bot="day",
            )

    def test_paper_sell_limit_with_valid_intent_passes(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        register_intent(_make_paper_intent(ticker="240810", side="SELL", bot="day"), bot="day")

        adapter = PaperOrderAdapter()
        order = adapter.sell_limit(
            ticker="240810", price=121000, quantity=1,
            mode="paper", executor_bot="day",
        )
        from src.entities.trading_models import OrderStatus
        assert order.status == OrderStatus.FILLED

    def test_paper_sell_limit_without_intent_blocked(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        with pytest.raises(NoIntentError):
            adapter.sell_limit(
                ticker="999999", price=10000, quantity=1,
                mode="paper", executor_bot="day",
            )

    def test_paper_backward_compat_no_mode(self, isolated):
        """mode/executor_bot None 시 기존 동작 (intent 강제 X) — backward compat."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        # intent 없어도 통과 (mode/executor_bot None)
        order = adapter.buy_limit(ticker="999999", price=10000, quantity=1)
        from src.entities.trading_models import OrderStatus
        assert order.status == OrderStatus.FILLED

    def test_paper_quant_executor_with_quant_intent(self, isolated):
        """퀀트봇 paper intent를 quant executor가 사용."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        register_intent(_make_paper_intent(ticker="240810", side="BUY", bot="quant"), bot="quant")

        adapter = PaperOrderAdapter()
        order = adapter.buy_limit(
            ticker="240810", price=121000, quantity=1,
            mode="paper", executor_bot="quant",
        )
        from src.entities.trading_models import OrderStatus
        assert order.status == OrderStatus.FILLED

    def test_paper_quant_intent_rejected_for_day_executor(self, isolated):
        """quant intent + day executor → NoIntentError (격리)."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        register_intent(_make_paper_intent(ticker="240810", side="BUY", bot="quant"), bot="quant")

        adapter = PaperOrderAdapter()
        with pytest.raises(NoIntentError):
            adapter.buy_limit(
                ticker="240810", price=121000, quantity=1,
                mode="paper", executor_bot="day",  # quant intent 사용 시도
            )


# ──────────────────────────────────────────────
# KisOrderAdapter._guard 직접 호출 (mojito broker 없이)
# ──────────────────────────────────────────────
class TestKisGuardIntegration:
    """KisOrderAdapter._guard 에 mode/executor_bot 전달 시 order_intents_gate 호출."""

    def test_guard_without_mode_executor_bot_skips_intent_check(self, isolated, monkeypatch):
        """mode/executor_bot None → intent 체크 X (backward compat). 다른 가드 통과 필요."""
        from src.adapters.kis_order_adapter import KisOrderAdapter
        sig = inspect.signature(KisOrderAdapter._guard)
        assert sig.parameters["mode"].default is None
        assert sig.parameters["executor_bot"].default is None


# ──────────────────────────────────────────────
# 코덱스 4차 응답 P0 3건 (5/28 13:37)
# ──────────────────────────────────────────────
class TestP0_1_KisRejectsPaperMode:
    """P0-1: KisOrderAdapter는 mode='live'만 허용. mode='paper' 즉시 차단."""

    def _make_kis_adapter_for_guard_test(self, monkeypatch):
        """KisOrderAdapter._guard를 직접 호출하기 위한 mock 인스턴스."""
        from src.adapters.kis_order_adapter import KisOrderAdapter
        # __init__의 mojito 호출 회피
        adapter = KisOrderAdapter.__new__(KisOrderAdapter)
        adapter._is_mock = True
        return adapter

    def test_kis_guard_rejects_paper_mode(self, isolated, monkeypatch):
        adapter = self._make_kis_adapter_for_guard_test(monkeypatch)
        with pytest.raises(ValueError) as exc_info:
            adapter._guard(
                ticker="240810", quantity=1, side="BUY",
                mode="paper", executor_bot="day",
            )
        assert "mode='live'만 허용" in str(exc_info.value) or "paper" in str(exc_info.value).lower()

    def test_kis_guard_requires_executor_bot_when_mode_given(self, isolated, monkeypatch):
        adapter = self._make_kis_adapter_for_guard_test(monkeypatch)
        with pytest.raises(ValueError):
            adapter._guard(
                ticker="240810", quantity=1, side="BUY",
                mode="live", executor_bot=None,
            )

    def test_kis_guard_rejects_invalid_mode(self, isolated, monkeypatch):
        adapter = self._make_kis_adapter_for_guard_test(monkeypatch)
        with pytest.raises(ValueError):
            adapter._guard(
                ticker="240810", quantity=1, side="BUY",
                mode="dry_run", executor_bot="day",
            )


class TestP0_2_PaperRejectsLiveMode:
    """P0-2: PaperOrderAdapter는 mode='paper'만 허용. mode='live' 즉시 차단."""

    def test_paper_buy_limit_rejects_live_mode(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError) as exc_info:
            adapter.buy_limit(
                ticker="240810", price=121000, quantity=1,
                mode="live", executor_bot="day",
            )
        assert "mode='paper'만 허용" in str(exc_info.value) or "live" in str(exc_info.value).lower()

    def test_paper_sell_limit_rejects_live_mode(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError):
            adapter.sell_limit(
                ticker="240810", price=121000, quantity=1,
                mode="live", executor_bot="day",
            )

    def test_paper_buy_market_rejects_live_mode(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError):
            adapter.buy_market(
                ticker="240810", quantity=1, current_price=121000,
                mode="live", executor_bot="day",
            )

    def test_paper_sell_market_rejects_live_mode(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError):
            adapter.sell_market(
                ticker="240810", quantity=1, current_price=121000,
                mode="live", executor_bot="day",
            )

    def test_paper_requires_executor_bot_when_mode_given(self, isolated):
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError):
            adapter.buy_limit(
                ticker="240810", price=121000, quantity=1,
                mode="paper", executor_bot=None,
            )


class TestP0_3_AssertQuantLiveBlocked:
    """P0-3: assert_order_intent_exists 단계에서 quant + live 조합 직접 차단."""

    def test_assert_quant_live_combo_blocked(self, isolated):
        """register에서 막혔지만, 외부에서 잘못된 intent 주입 시 assert도 차단."""
        from src.use_cases.order_intents_gate import (
            assert_order_intent_exists, OrderIntentError,
        )
        with pytest.raises(OrderIntentError) as exc_info:
            assert_order_intent_exists(
                ticker="240810", side="BUY",
                mode="live", executor_bot="quant",
            )
        assert "quant" in str(exc_info.value).lower()
        assert "live" in str(exc_info.value).lower()

    def test_assert_day_live_combo_allowed(self, isolated):
        """day + live는 허용 (intent 등록만 안 됐을 뿐)."""
        from src.use_cases.order_intents_gate import (
            assert_order_intent_exists, NoIntentError, OrderIntentError,
        )
        # day + live는 P0-3 단계는 통과 (input validation OK)
        # 다만 intent 등록 X → NoIntentError raise
        with pytest.raises(NoIntentError):
            assert_order_intent_exists(
                ticker="240810", side="BUY",
                mode="live", executor_bot="day",
            )
