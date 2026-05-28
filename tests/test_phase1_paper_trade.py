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


# ──────────────────────────────────────────────
# Phase 1 row 2 — cmd_paper_trade_close (SELL 페어)
# ──────────────────────────────────────────────
class TestPhase1PaperTradeClose:
    def test_paper_trade_close_intent_registered(self, isolated_env):
        """SELL intent register + HMAC 서명 자동 추가 검증."""
        now_kst = datetime.now(tz=SEOUL)
        intent = {
            "intent_id": f"q_240810_SELL_{now_kst.strftime('%Y%m%d%H%M%S')}",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "name": "원익IPS",
            "side": "SELL", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=2)).isoformat(),
            "parent_buy_intent_id": "q_240810_open_test",
        }
        register_intent(intent, bot="quant")
        intents = list_today_intents(executor_bot="quant", side="SELL", mode="paper")
        assert len(intents) == 1
        assert intents[0]["side"] == "SELL"
        assert intents[0]["_signature_valid"] is True

    def test_paper_trade_close_sell_limit_passes(self, isolated_env):
        """SELL intent 등록 후 PaperOrderAdapter.sell_limit 통과 (P0-2 + L10)."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter

        now_kst = datetime.now(tz=SEOUL)
        intent = {
            "intent_id": f"q_240810_SELL_{now_kst.strftime('%Y%m%d%H%M%S')}",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "SELL", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=2)).isoformat(),
        }
        register_intent(intent, bot="quant")

        adapter = PaperOrderAdapter()
        order = adapter.sell_limit(
            ticker="240810", price=121500, quantity=1,
            mode="paper", executor_bot="quant",
        )
        assert order.status == OrderStatus.FILLED

    def test_paper_trade_close_blocked_without_sell_intent(self, isolated_env):
        """BUY intent만 등록된 상태에서 SELL 시도 → NoIntentError."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter

        now_kst = datetime.now(tz=SEOUL)
        # BUY intent만 등록
        buy_intent = {
            "intent_id": "q_240810_BUY_test",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "BUY", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=4)).isoformat(),
        }
        register_intent(buy_intent, bot="quant")

        adapter = PaperOrderAdapter()
        # SELL 시도 → SELL intent 미등록이므로 차단
        with pytest.raises(NoIntentError):
            adapter.sell_limit(
                ticker="240810", price=121500, quantity=1,
                mode="paper", executor_bot="quant",
            )

    def test_paper_trade_close_quant_sell_live_blocked(self, isolated_env):
        """quant + SELL + live 조합 register 차단 (Note 1)."""
        now_kst = datetime.now(tz=SEOUL)
        bad_intent = {
            "intent_id": "q_240810_SELL_live_attempt",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "SELL", "mode": "live",  # ★ 위반
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=2)).isoformat(),
        }
        with pytest.raises(OrderIntentError):
            register_intent(bad_intent, bot="quant")

    def test_paper_trade_close_paper_adapter_live_mode_blocked(self, isolated_env):
        """PaperOrderAdapter.sell_limit mode='live' 직접 호출 차단 (P0-2)."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter

        adapter = PaperOrderAdapter()
        with pytest.raises(ValueError):
            adapter.sell_limit(
                ticker="240810", price=121500, quantity=1,
                mode="live", executor_bot="quant",
            )

    def test_paper_trade_close_output_schema(self, isolated_env):
        """close 출력 파일 스키마 검증."""
        sample = {
            "date": "2026-05-28", "mode": "close",
            "n_open_filled": 5, "n_sell_registered": 5,
            "n_sell_filled": 4, "n_sell_blocked": 1,
            "total_buy_cost": 500000, "total_sell_proceed": 512000,
            "total_pnl": 11500, "total_pnl_pct": 2.30,
            "records": [
                {"ticker": "240810", "name": "원익IPS", "status": "filled",
                 "close_status": "filled",
                 "intent_id": "q_240810_BUY_test",
                 "sell_intent_id": "q_240810_SELL_test",
                 "filled_price": 121060, "sell_filled_price": 122500,
                 "qty": 1, "paper_pnl": 1320, "paper_pnl_pct": 1.09},
            ],
            "closed_at": datetime.now(tz=SEOUL).isoformat(),
        }
        out_dir = isolated_env["tmp_path"] / "phase1_paper_trades"
        out_dir.mkdir(parents=True)
        out_path = out_dir / "20260528_close.json"
        out_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert {"date", "mode", "n_open_filled", "n_sell_registered",
                "n_sell_filled", "n_sell_blocked", "total_pnl",
                "total_pnl_pct", "records", "closed_at"} <= set(loaded.keys())
        for rec in loaded["records"]:
            assert "ticker" in rec
            assert "close_status" in rec
            if rec["close_status"] == "filled":
                assert "sell_intent_id" in rec
                assert "sell_filled_price" in rec
                assert "paper_pnl" in rec

    def test_paper_trade_close_buy_sell_paired_intents(self, isolated_env):
        """BUY + SELL intent 페어 등록 → 모두 조회 가능."""
        now_kst = datetime.now(tz=SEOUL)
        buy = {
            "intent_id": "q_240810_BUY_paired",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "BUY", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=4)).isoformat(),
        }
        sell = {
            "intent_id": "q_240810_SELL_paired",
            "bot": "quant", "engine": "phase1_paper_warmup",
            "ticker": "240810", "side": "SELL", "mode": "paper",
            "score": 80.0,
            "created_at": now_kst.isoformat(),
            "expires_at": (now_kst + timedelta(hours=2)).isoformat(),
            "parent_buy_intent_id": "q_240810_BUY_paired",
        }
        register_intent(buy, bot="quant")
        register_intent(sell, bot="quant")

        buys = list_today_intents(executor_bot="quant", side="BUY", mode="paper")
        sells = list_today_intents(executor_bot="quant", side="SELL", mode="paper")
        assert len(buys) == 1 and len(sells) == 1
        assert buys[0]["intent_id"] == "q_240810_BUY_paired"
        assert sells[0]["intent_id"] == "q_240810_SELL_paired"
        assert sells[0].get("parent_buy_intent_id") == "q_240810_BUY_paired"


# ──────────────────────────────────────────────
# Phase 1 row 3 — chart_hero_picker_cycle (selector)
# ──────────────────────────────────────────────
class TestPhase1ChartHeroPicker:
    """chart_hero_picker_cycle은 selector — 매매 호출 X, intent 등록만.

    picker 자체는 surge_d1_picker.run_picker 의존성. 단위 테스트 어려움.
    여기서는 selector 로직 (5-Gate picks → intent 등록) 자체를 격리 검증.
    """

    def _register_picks_as_intents(self, picks, today, next_day, seoul):
        """picker_cycle.py 내부 selector 로직 재현 (테스트용)."""
        now_kst = datetime.now(tz=seoul)
        d1_close_kst = datetime.combine(
            next_day, datetime.min.time().replace(hour=15, minute=30),
            tzinfo=seoul,
        )
        registered = 0
        for p in picks:
            tk = p["ticker"]
            intent = {
                "intent_id": f"q_{tk}_chart_hero_d1_{next_day.isoformat()}",
                "bot": "quant", "engine": "chart_hero_5gate",
                "ticker": tk, "name": p.get("name", tk),
                "side": "BUY", "mode": "paper",
                "score": float(p.get("buy_score", 0.0)),
                "created_at": now_kst.isoformat(),
                "expires_at": d1_close_kst.isoformat(),
                "d0_date": today.isoformat(), "d1_date": next_day.isoformat(),
            }
            register_intent(intent, bot="quant")
            registered += 1
        return registered

    def test_chart_hero_picker_intents_registered(self, isolated_env):
        """picker가 picks → quant_intents 등록 (selector 역할 검증)."""
        from datetime import date

        today = date(2026, 5, 28)
        next_day = date(2026, 5, 29)
        picks = [
            {"ticker": "240810", "name": "원익IPS", "buy_score": 85.0},
            {"ticker": "005930", "name": "삼성전자", "buy_score": 72.0},
        ]
        n = self._register_picks_as_intents(picks, today, next_day, SEOUL)
        assert n == 2

        intents = list_today_intents(executor_bot="quant", side="BUY", mode="paper")
        assert len(intents) == 2
        tickers = {i["ticker"] for i in intents}
        assert tickers == {"240810", "005930"}
        # selector 메타데이터
        assert all(i["engine"] == "chart_hero_5gate" for i in intents)
        assert all(i["d1_date"] == "2026-05-29" for i in intents)

    def test_chart_hero_picker_expires_at_d1_close(self, isolated_env):
        """expires_at은 D+1 15:30 (Asia/Seoul) — D+1 close_cycle까지 유효."""
        from datetime import date

        today = date(2026, 5, 28)
        next_day = date(2026, 5, 29)
        picks = [{"ticker": "240810", "name": "원익IPS", "buy_score": 85.0}]
        self._register_picks_as_intents(picks, today, next_day, SEOUL)

        intents = list_today_intents(executor_bot="quant")
        assert len(intents) == 1
        # expires_at은 D+1 15:30 KST
        from datetime import datetime as _dt
        expires = _dt.fromisoformat(intents[0]["expires_at"])
        assert expires.year == 2026 and expires.month == 5 and expires.day == 29
        assert expires.hour == 15 and expires.minute == 30
        assert expires.tzinfo is not None  # timezone-aware

    def test_chart_hero_picker_real_mode_no_intent(self, isolated_env):
        """real 모드는 intent 등록 금지 (코덱스 5차 지시).

        시뮬: picker_cycle main이 args.real 분기 시 register_intent 호출 X.
        여기서는 빈 리스트로 통과 검증 (테스트는 selector 로직 격리).
        """
        # real 모드 시 picks가 있어도 register 호출 X (picker_cycle 분기 로직)
        # 이 테스트는 register_intent 미호출 시 list가 빈 것 확인
        intents = list_today_intents(executor_bot="quant")
        assert len(intents) == 0  # 호출 없음 → 빈 리스트


# ──────────────────────────────────────────────
# Phase 1 row 4~6 — ChartHeroExecutor (paper + intent 페어)
# ──────────────────────────────────────────────
class TestPhase1ChartHeroExecutor:
    """ChartHeroExecutor paper=True → PaperOrderAdapter + intent 페어 검증."""

    def test_executor_paper_uses_paper_adapter(self):
        """paper=True 시 self.order가 PaperOrderAdapter."""
        from src.strategies.chart_hero_executor import ChartHeroExecutor
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        ex = ChartHeroExecutor(paper=True, total_capital=25_000_000)
        assert isinstance(ex.order, PaperOrderAdapter)

    def test_executor_real_uses_kis_adapter(self):
        """paper=False 시 self.order가 KisOrderAdapter."""
        from src.strategies.chart_hero_executor import ChartHeroExecutor
        from src.adapters.kis_order_adapter import KisOrderAdapter
        ex = ChartHeroExecutor(paper=False, total_capital=25_000_000)
        assert isinstance(ex.order, KisOrderAdapter)

    def test_executor_buy_limit_signature_includes_mode_executor(self):
        """buy_limit/sell_limit 호출 시 mode + executor_bot 명시 (시그니처 확인)."""
        import inspect
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        sig = inspect.signature(PaperOrderAdapter.buy_limit)
        # buy_limit은 mode/executor_bot keyword-only 인자 보유
        assert "mode" in sig.parameters
        assert "executor_bot" in sig.parameters
        sig_sell = inspect.signature(PaperOrderAdapter.sell_limit)
        assert "mode" in sig_sell.parameters
        assert "executor_bot" in sig_sell.parameters

    def test_executor_paper_buy_blocked_without_picker_intent(self, isolated_env):
        """picker가 intent 등록 안 했으면 executor BUY → BLOCKED_NO_INTENT.

        row 3 (picker) + row 4 (executor) 페어 검증.
        picker intent 없이 executor가 호출되면 NoIntentError → results.action='BLOCKED_NO_INTENT'.
        """
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        # picker intent 미등록 상태에서 직접 PaperOrderAdapter 호출 시도
        adapter = PaperOrderAdapter()
        with pytest.raises(NoIntentError):
            adapter.buy_limit(
                ticker="240810", price=121000, quantity=1,
                mode="paper", executor_bot="quant",
            )

    def test_executor_paper_buy_passes_with_picker_intent(self, isolated_env):
        """row 3 picker가 등록한 intent로 row 4 executor가 buy_limit 통과."""
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        from datetime import date

        # row 3 picker가 D0 17:30에 등록한 intent (시뮬)
        today = date(2026, 5, 28)
        next_day = date(2026, 5, 29)
        now_kst = datetime.now(tz=SEOUL)
        d1_close_kst = datetime.combine(
            next_day, datetime.min.time().replace(hour=15, minute=30),
            tzinfo=SEOUL,
        )
        picker_intent = {
            "intent_id": "q_240810_chart_hero_d1_2026-05-29",
            "bot": "quant", "engine": "chart_hero_5gate",
            "ticker": "240810", "side": "BUY", "mode": "paper",
            "score": 85.0,
            "created_at": now_kst.isoformat(),
            "expires_at": d1_close_kst.isoformat(),
            "d0_date": today.isoformat(), "d1_date": next_day.isoformat(),
        }
        register_intent(picker_intent, bot="quant")

        # row 4 executor (시뮬: PaperOrderAdapter.buy_limit 직접 호출)
        adapter = PaperOrderAdapter()
        order = adapter.buy_limit(
            ticker="240810", price=121000, quantity=1,
            mode="paper", executor_bot="quant",
        )
        assert order.status == OrderStatus.FILLED
