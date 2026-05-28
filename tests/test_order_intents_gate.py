"""order_intents_gate 회귀 테스트 (Trading Factory v1, 5/28 코덱스 2차 응답 반영)

검증 P0-1~5:
  P0-1: ORDER_INTENTS_GATE_DISABLED 환경변수 우회 영구 제거
  P0-2: mode 명시 강제 (paper/live만, 기본값 X)
  P0-3: executor_bot 매치 검증 (quant↔day 격리)
  P0-4: expires_at timezone-aware 강제 + 만료 검증
  P0-5: HMAC 서명 위조 차단 + 키 미설정 차단

연결:
  - src/use_cases/order_intents_gate.py
  - ops/codex_outbox/20260528T131057_..._changes-requested.md
"""

from __future__ import annotations

import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import src.use_cases.order_intents_gate as gate
from src.use_cases.order_intents_gate import (
    register_intent,
    assert_order_intent_exists,
    list_today_intents,
    OrderIntentError,
    NoIntentError,
    IntentSignatureError,
    IntentExpiredError,
    IntentSchemaError,
    _compute_signature,
)

# HMAC 키 — 테스트 전용 (운영 키 X)
TEST_HMAC_KEY = "test_key_for_pytest_minimum_32_chars_xxxxxxxx"
SEOUL = ZoneInfo("Asia/Seoul")


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def isolated_intents_dir(monkeypatch):
    """매 테스트마다 임시 디렉토리 + HMAC 키 격리."""
    tmp = Path(tempfile.mkdtemp(prefix="oig_test_"))
    monkeypatch.setattr(gate, "ORDER_INTENTS_DIR", tmp)
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", TEST_HMAC_KEY)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _make_intent(
    intent_id="q_test_001",
    bot="quant",
    ticker="240810",
    side="BUY",
    mode="paper",
    score=80.0,
    expires_in_hours=4,
    tz=SEOUL,
):
    """timezone-aware intent dict 빌더."""
    now = datetime.now(tz=tz)
    return {
        "intent_id": intent_id,
        "bot": bot,
        "engine": "test_engine",
        "ticker": ticker,
        "name": "테스트종목",
        "side": side,
        "mode": mode,
        "score": score,
        "confidence": "strong",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=expires_in_hours)).isoformat(),
    }


# ──────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────
class TestHappyPath:
    def test_register_then_assert_passes(self, isolated_intents_dir):
        intent = _make_intent()
        register_intent(intent, bot="quant")
        result = assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")
        assert result["intent_id"] == "q_test_001"

    def test_signature_is_added_on_register(self, isolated_intents_dir):
        intent = _make_intent()
        register_intent(intent, bot="quant")
        # 파일에서 다시 읽어 서명 필드 확인
        intents = list_today_intents()
        assert len(intents) == 1
        assert "hmac_signature" in intents[0]
        assert len(intents[0]["hmac_signature"]) == 64  # SHA256 hex

    def test_list_filter_by_side_mode_executor(self, isolated_intents_dir):
        register_intent(_make_intent("q_001", side="BUY"), bot="quant")
        register_intent(_make_intent("q_002", side="SELL", ticker="005930"), bot="quant")
        register_intent(_make_intent("d_001", bot="day", ticker="123456"), bot="day")

        all_intents = list_today_intents()
        assert len(all_intents) == 3

        buys = list_today_intents(side="BUY")
        assert len(buys) == 2

        day_only = list_today_intents(executor_bot="day")
        assert len(day_only) == 1
        assert day_only[0]["intent_id"] == "d_001"


# ──────────────────────────────────────────────
# P0-1: ORDER_INTENTS_GATE_DISABLED 우회 영구 제거
# ──────────────────────────────────────────────
class TestP0_1_RuntimeBypassRemoved:
    def test_env_disabled_does_not_bypass(self, isolated_intents_dir, monkeypatch):
        # 환경변수로 우회 시도 → 효과 없음
        monkeypatch.setenv("ORDER_INTENTS_GATE_DISABLED", "1")
        with pytest.raises(NoIntentError):
            assert_order_intent_exists("999999", "BUY", "paper", executor_bot="quant")

    def test_env_true_does_not_bypass(self, isolated_intents_dir, monkeypatch):
        monkeypatch.setenv("ORDER_INTENTS_GATE_DISABLED", "true")
        with pytest.raises(NoIntentError):
            assert_order_intent_exists("999999", "BUY", "paper", executor_bot="quant")


# ──────────────────────────────────────────────
# P0-2: mode 명시 강제 + 기본값 제거
# ──────────────────────────────────────────────
class TestP0_2_ModeExplicit:
    def test_invalid_mode_rejected(self, isolated_intents_dir):
        register_intent(_make_intent(), bot="quant")
        with pytest.raises(OrderIntentError) as exc_info:
            assert_order_intent_exists("240810", "BUY", "invalid_mode", executor_bot="quant")
        assert "mode" in str(exc_info.value)

    def test_empty_mode_rejected(self, isolated_intents_dir):
        register_intent(_make_intent(), bot="quant")
        with pytest.raises(OrderIntentError):
            assert_order_intent_exists("240810", "BUY", "", executor_bot="quant")

    def test_paper_mode_intent_not_accepted_for_live_call(self, isolated_intents_dir):
        # paper intent 등록 → live 호출 시 OrderIntentError (P0-3 코덱스 4차: quant+live 직접 차단)
        # NoIntentError 도달 전에 quant+live 조합 자체가 거부됨 (defense in depth)
        register_intent(_make_intent(mode="paper"), bot="quant")
        with pytest.raises(OrderIntentError):
            assert_order_intent_exists("240810", "BUY", "live", executor_bot="quant")

    def test_register_invalid_mode_rejected(self, isolated_intents_dir):
        bad = _make_intent(mode="dry_run")  # paper/live 외
        with pytest.raises(OrderIntentError):
            register_intent(bad, bot="quant")


# ──────────────────────────────────────────────
# P0-3: executor_bot 매치
# ──────────────────────────────────────────────
class TestP0_3_ExecutorBotMatch:
    def test_quant_intent_rejected_for_day_executor(self, isolated_intents_dir):
        register_intent(_make_intent(bot="quant"), bot="quant")
        with pytest.raises(NoIntentError):
            assert_order_intent_exists("240810", "BUY", "paper", executor_bot="day")

    def test_day_intent_rejected_for_quant_executor(self, isolated_intents_dir):
        intent = _make_intent(intent_id="d_001", bot="day", ticker="005930")
        register_intent(intent, bot="day")
        with pytest.raises(NoIntentError):
            assert_order_intent_exists("005930", "BUY", "paper", executor_bot="quant")

    def test_invalid_executor_bot_rejected(self, isolated_intents_dir):
        register_intent(_make_intent(), bot="quant")
        with pytest.raises(OrderIntentError):
            assert_order_intent_exists("240810", "BUY", "paper", executor_bot="info")


# ──────────────────────────────────────────────
# P0-4: expires_at timezone-aware 강제 (코덱스 2차 응답)
# ──────────────────────────────────────────────
class TestP0_4_ExpiresAtTimezoneAware:
    def test_naive_expires_at_rejected_on_register(self, isolated_intents_dir):
        # timezone-naive isoformat → register 시점에 거부
        now_naive = datetime.now()  # tzinfo X
        intent = _make_intent()
        intent["expires_at"] = (now_naive + timedelta(hours=4)).isoformat()
        with pytest.raises(IntentSchemaError) as exc_info:
            register_intent(intent, bot="quant")
        assert "timezone-naive" in str(exc_info.value)

    def test_seoul_timezone_accepted(self, isolated_intents_dir):
        intent = _make_intent(tz=SEOUL)
        register_intent(intent, bot="quant")
        result = assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")
        assert result["intent_id"] == "q_test_001"

    def test_utc_timezone_accepted(self, isolated_intents_dir):
        intent = _make_intent(tz=timezone.utc)
        register_intent(intent, bot="quant")
        result = assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")
        assert result["intent_id"] == "q_test_001"

    def test_expired_intent_rejected(self, isolated_intents_dir):
        # 과거 expires_at으로 등록 시 register는 통과 (스키마 OK)하지만 assert 시 IntentExpiredError
        past_intent = _make_intent(expires_in_hours=-1, tz=SEOUL)
        register_intent(past_intent, bot="quant")
        with pytest.raises(IntentExpiredError):
            assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")

    def test_naive_intent_in_jsonl_rejected_on_assert(self, isolated_intents_dir):
        # 누군가 직접 파일에 naive expires_at + 위조 서명까지 추가했다고 가정
        # → assert 시점 IntentSchemaError raise (timezone-naive 거부)
        # 단, HMAC 검증을 먼저 통과해야 schema 검증까지 도달
        # 정상 등록 후 expires_at만 수동 변경 시 서명 깨짐 → IntentSignatureError가 먼저
        # 따라서 naive expires_at 차단의 진정한 보장은 register 시점에 (위 첫 테스트)
        intent = _make_intent(tz=SEOUL)
        register_intent(intent, bot="quant")
        # 파일을 수동으로 변조하여 expires_at을 naive로 + 같은 서명 유지 (시뮬)
        today_str = datetime.now().strftime("%Y%m%d")
        fpath = isolated_intents_dir / f"quant_intents_{today_str}.jsonl"
        # 직접 jsonl 추가: naive expires_at, 서명 없이
        forged = _make_intent(intent_id="forged_naive", ticker="999999")
        forged["expires_at"] = (datetime.now() + timedelta(hours=4)).isoformat()  # naive
        with fpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(forged) + "\n")
        # 서명 없으므로 IntentSignatureError가 먼저 raise (HMAC 우선 검증)
        with pytest.raises(IntentSignatureError):
            assert_order_intent_exists("999999", "BUY", "paper", executor_bot="quant")


# ──────────────────────────────────────────────
# P0-5: HMAC 서명
# ──────────────────────────────────────────────
class TestP0_5_HmacSignature:
    def test_forged_intent_rejected(self, isolated_intents_dir):
        # 서명 없이 jsonl에 직접 추가
        today_str = datetime.now().strftime("%Y%m%d")
        fpath = isolated_intents_dir / f"quant_intents_{today_str}.jsonl"
        forged = _make_intent(intent_id="forged_001", ticker="999999")
        with fpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(forged) + "\n")
        with pytest.raises(IntentSignatureError):
            assert_order_intent_exists("999999", "BUY", "paper", executor_bot="quant")

    def test_tampered_intent_rejected(self, isolated_intents_dir):
        # 정상 등록 → 서명 추가됨 → 파일에서 score만 변조 → 서명 깨짐
        intent = _make_intent()
        register_intent(intent, bot="quant")
        today_str = datetime.now().strftime("%Y%m%d")
        fpath = isolated_intents_dir / f"quant_intents_{today_str}.jsonl"
        lines = fpath.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[0])
        tampered["score"] = 99.9  # 위조: 서명 깨짐
        fpath.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
        with pytest.raises(IntentSignatureError):
            assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")

    def test_hmac_key_missing_blocks_register(self, isolated_intents_dir, monkeypatch):
        monkeypatch.delenv("ORDER_INTENTS_HMAC_KEY", raising=False)
        with pytest.raises(IntentSignatureError):
            register_intent(_make_intent(), bot="quant")

    def test_hmac_key_too_short_blocks(self, isolated_intents_dir, monkeypatch):
        monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", "short")
        with pytest.raises(IntentSignatureError):
            register_intent(_make_intent(), bot="quant")

    def test_different_key_invalidates_signature(self, isolated_intents_dir, monkeypatch):
        # 키 A로 register
        register_intent(_make_intent(), bot="quant")
        # 키 B로 assert (다른 키) → 서명 검증 실패
        monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", "another_key_32_chars_xxxxxxxxxxxxxxxx")
        with pytest.raises(IntentSignatureError):
            assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")


# ──────────────────────────────────────────────
# 입력 validation
# ──────────────────────────────────────────────
class TestInputValidation:
    def test_invalid_side_rejected(self, isolated_intents_dir):
        register_intent(_make_intent(), bot="quant")
        with pytest.raises(OrderIntentError):
            assert_order_intent_exists("240810", "HOLD", "paper", executor_bot="quant")

    def test_register_missing_required_field(self, isolated_intents_dir):
        bad = _make_intent()
        del bad["score"]
        with pytest.raises(OrderIntentError):
            register_intent(bad, bot="quant")

    def test_register_bot_mismatch(self, isolated_intents_dir):
        intent = _make_intent(bot="quant")
        with pytest.raises(OrderIntentError):
            register_intent(intent, bot="day")


# ──────────────────────────────────────────────
# 누락 시나리오
# ──────────────────────────────────────────────
class TestNoIntent:
    def test_no_file_raises(self, isolated_intents_dir):
        # 파일 자체가 없을 때
        with pytest.raises(NoIntentError):
            assert_order_intent_exists("240810", "BUY", "paper", executor_bot="quant")

    def test_ticker_not_registered(self, isolated_intents_dir):
        register_intent(_make_intent(), bot="quant")
        with pytest.raises(NoIntentError):
            assert_order_intent_exists("999999", "BUY", "paper", executor_bot="quant")
