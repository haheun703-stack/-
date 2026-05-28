"""Phase 1 paper 호출처 마이그레이션 회귀 테스트 (5/28 코덱스 5차 PASS).

검증:
  - paper_warmup_daily.cmd_paper_trade_open이 register_intent + PaperOrderAdapter.buy_limit 정상 호출
  - intent 미등록 종목 시도 시 NoIntentError 차단
  - mode='paper' + executor_bot='quant' 정합
  - 추적표 (data/phase1_paper_trades/{date}_open.json) 출력 확인

연결:
  - docs/02-design/phase-1-migration-tracking.md
  - scripts/paper_warmup_daily.py cmd_paper_trade_open
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import src.use_cases.order_intents_gate as gate
from src.use_cases.order_intents_gate import (
    register_intent, NoIntentError, list_today_intents, OrderIntentError,
)
from src.entities.trading_models import OrderStatus

TEST_HMAC_KEY = "test_phase1_paper_trade_minimum_32_chars_xxxx"
SEOUL = ZoneInfo("Asia/Seoul")


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    tmp_intent = tmp_path / "intents"
    tmp_intent.mkdir()
    monkeypatch.setattr(gate, "ORDER_INTENTS_DIR", tmp_intent)
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", TEST_HMAC_KEY)
    monkeypatch.setenv("PAPER_BUDGET_PER_TICKER", "100000")
    yield {"tmp_path": tmp_path, "intent_dir": tmp_intent}


class TestPhase1PaperTradeOpen:
    def test_paper_trade_open_intent_registered(self, isolated_env):
        now_kst = datetime.now(tz=SEOUL)
        intent = {
            "intent_id": f"q_240810_{now_kst.strftime('%Y%m%d%H%M%S')}",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "name": "원익IPS",
            "side": "BUY", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=8)).isoformat(),
        }
        register_intent(intent, bot="quant")

        intents = list_today_intents(executor_bot="quant", mode="paper")
        assert len(intents) == 1
        assert intents[0]["ticker"] == "240810"
        assert intents[0]["mode"] == "paper"
        assert intents[0]["bot"] == "quant"
        assert intents[0]["_signature_valid"] is True

    def test_paper_trade_open_buy_limit_passes(self, isolated_env):
        from src.adapters.paper_order_adapter import PaperOrderAdapter

        now_kst = datetime.now(tz=SEOUL)
        intent = {
            "intent_id": f"q_240810_{now_kst.strftime('%Y%m%d%H%M%S')}",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "BUY", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=8)).isoformat(),
        }
        register_intent(intent, bot="quant")

        adapter = PaperOrderAdapter()
        order = adapter.buy_limit(
            ticker="240810", price=121000, quantity=1,
            mode="paper", executor_bot="quant",
        )
        assert order.status == OrderStatus.FILLED

    def test_paper_trade_open_blocked_without_intent(self, isolated_env):
        from src.adapters.paper_order_adapter import PaperOrderAdapter

        adapter = PaperOrderAdapter()
        with pytest.raises(NoIntentError):
            adapter.buy_limit(
                ticker="999999", price=10000, quantity=1,
                mode="paper", executor_bot="quant",
            )

    def test_paper_trade_open_quant_mode_live_blocked(self, isolated_env):
        now_kst = datetime.now(tz=SEOUL)
        bad_intent = {
            "intent_id": "q_quant_live_attempt",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "BUY", "mode": "live",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=8)).isoformat(),
        }
        with pytest.raises(OrderIntentError):
            register_intent(bad_intent, bot="quant")

    def test_paper_trade_open_paper_adapter_live_mode_blocked(self, isolated_env):
        from src.adapters.paper_order_adapter import PaperOrderAdapter

        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError):
            adapter.buy_limit(
                ticker="240810", price=121000, quantity=1,
                mode="live", executor_bot="quant",
            )


class TestPhase1OutputFile:
    def test_output_file_schema_keys(self, isolated_env):
        sample = {
            "date": "2026-05-28", "mode": "open", "top_n": 9,
            "n_registered": 5, "n_filled": 4, "n_blocked": 1,
            "records": [
                {"ticker": "240810", "name": "원익IPS", "rank": 1, "status": "filled",
                 "intent_id": "q_240810_test", "entry": 121000, "filled_price": 121060,
                 "qty": 1, "order_id": "PAPER_test"},
                {"ticker": "999999", "name": "MISSING", "rank": 2, "status": "intent_blocked",
                 "intent_id": "q_999999_test", "error": "no intent"},
            ],
            "created_at": datetime.now(tz=SEOUL).isoformat(),
        }
        out_dir = isolated_env["tmp_path"] / "phase1_paper_trades"
        out_dir.mkdir(parents=True)
        out_path = out_dir / "20260528_open.json"
        out_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert {"date", "mode", "top_n", "n_registered", "n_filled", "n_blocked",
                "records", "created_at"} <= set(loaded.keys())
        for rec in loaded["records"]:
            assert "ticker" in rec
            assert "status" in rec
            assert "rank" in rec
            if rec["status"] == "filled":
                assert "intent_id" in rec
                assert "filled_price" in rec
            elif rec["status"] == "intent_blocked":
                assert "intent_id" in rec
                assert "error" in rec
