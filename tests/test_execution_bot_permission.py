"""B⑤ (5/30) execution bot 권한 모델 — selector(quant)/executor(execution) 분리 검증.

opt-in 하위호환: allowed_executors 필드가 없으면 기존 동작 100% 동일.
execution이 live 실매수하려면 quant가 allowed_executors=["execution"]를 서명해 위임한
intent만 소비 가능. 위조(필드 수동 추가)는 HMAC 서명 검증으로 차단.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import src.use_cases.order_intents_gate as gate
from src.use_cases.order_intents_gate import (
    IntentSignatureError,
    NoIntentError,
    OrderIntentError,
    assert_order_intent_exists,
    register_intent,
)

TEST_HMAC_KEY = "x" * 64
KST = timezone(timedelta(hours=9))


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    d = tmp_path / "intents"
    d.mkdir()
    monkeypatch.setattr(gate, "ORDER_INTENTS_DIR", d)
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", TEST_HMAC_KEY)
    yield d


def _intent(ticker="240810", mode="paper", allowed=None):
    now = datetime.now(tz=KST)
    i = {
        "intent_id": f"q_{ticker}_{now.strftime('%H%M%S%f')}",
        "bot": "quant",
        "engine": "test",
        "ticker": ticker,
        "side": "BUY",
        "mode": mode,
        "score": 80.0,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=4)).isoformat(),
    }
    if allowed is not None:
        i["allowed_executors"] = allowed
    return i


class TestExecutionBotPermission:
    def test_backward_compat_quant_paper(self, isolated):
        # 하위호환: allowed_executors 없는 기존 quant paper intent → quant 소비 정상
        register_intent(_intent(mode="paper"), bot="quant")
        got = assert_order_intent_exists(
            ticker="240810", side="BUY", mode="paper", executor_bot="quant"
        )
        assert got["ticker"] == "240810"

    def test_execution_cannot_consume_plain_quant_intent(self, isolated):
        # allowed_executors 없으면 execution이 quant intent 소비 불가 (하위호환 격리)
        register_intent(_intent(mode="paper"), bot="quant")
        with pytest.raises(NoIntentError):
            assert_order_intent_exists(
                ticker="240810", side="BUY", mode="paper", executor_bot="execution"
            )

    def test_execution_live_requires_delegation(self, isolated):
        # quant가 allowed_executors=["execution"] 위임한 live intent → execution 소비 가능
        register_intent(_intent(mode="live", allowed=["execution"]), bot="quant")
        got = assert_order_intent_exists(
            ticker="240810", side="BUY", mode="live", executor_bot="execution"
        )
        assert got["mode"] == "live"

    def test_quant_live_without_delegation_blocked_at_register(self, isolated):
        # 위임 없는 quant live intent는 등록 자체 차단
        with pytest.raises(OrderIntentError):
            register_intent(_intent(mode="live"), bot="quant")

    def test_quant_executor_live_still_blocked(self, isolated):
        # quant가 직접 live 실행은 위임 intent가 있어도 여전히 차단 (L284 입력검증)
        register_intent(_intent(mode="live", allowed=["execution"]), bot="quant")
        with pytest.raises(OrderIntentError):
            assert_order_intent_exists(
                ticker="240810", side="BUY", mode="live", executor_bot="quant"
            )

    def test_forged_allowed_executors_breaks_hmac(self, isolated):
        # 등록된 paper intent에 allowed_executors를 수동 추가(위조) → HMAC 서명 깨짐
        register_intent(_intent(mode="paper"), bot="quant")
        files = list(isolated.glob("*.jsonl"))
        assert files
        obj = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[0])
        obj["allowed_executors"] = ["execution"]  # 서명 재계산 없이 필드 추가 = 위조
        files[0].write_text(
            json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with pytest.raises(IntentSignatureError):
            assert_order_intent_exists(
                ticker="240810", side="BUY", mode="paper", executor_bot="execution"
            )
