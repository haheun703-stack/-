"""
Phase 1: 장중 데이터 수집 파이프라인 단위 테스트

테스트 대상:
  1. 엔티티 모델 (intraday_models)
  2. SQLite 저장소 (sqlite_intraday_store)
  3. 수집 오케스트레이터 (intraday_collector) — Mock 데이터 포트
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.entities.intraday_models import (
    AIJudgment,
    AlertLevel,
    Candle5Min,
    InvestorFlowIntraday,
    MarketContext,
    NewsEvent,
    SectorPrice,
    TickData,
)
from src.adapters.sqlite_intraday_store import SqliteIntradayStore
from src.use_cases.intraday_collector import IntradayCollector


# ═══════════════════════════════════════════════════
# 1. 엔티티 모델 테스트
# ═══════════════════════════════════════════════════

class TestIntradayModels:
    def test_tick_data_to_dict(self):
        tick = TickData(
            ticker="005930",
            timestamp="2026-02-15 09:01:00",
            current_price=70000,
            volume=1000,
            change_pct=1.5,
        )
        d = tick.to_dict()
        assert d["ticker"] == "005930"
        assert d["current_price"] == 70000
        assert d["change_pct"] == 1.5

    def test_candle_5min_to_dict(self):
        candle = Candle5Min(
            ticker="000660",
            timestamp="2026-02-15 09:05:00",
            open=120000,
            high=121000,
            low=119500,
            close=120500,
            volume=50000,
        )
        d = candle.to_dict()
        assert d["open"] == 120000
        assert d["close"] == 120500

    def test_market_context_to_dict(self):
        ctx = MarketContext(
            timestamp="2026-02-15 09:05:00",
            kospi=2850.5,
            kospi_change_pct=0.35,
            kosdaq=910.2,
            kosdaq_change_pct=-0.12,
        )
        d = ctx.to_dict()
        assert d["kospi"] == 2850.5
        assert d["kosdaq_change_pct"] == -0.12

    def test_alert_level_enum(self):
        assert AlertLevel.GREEN.value == "GREEN"
        assert AlertLevel.RED.value == "RED"
        assert AlertLevel.BLUE.value == "BLUE"

    def test_ai_judgment_to_dict(self):
        j = AIJudgment(
            id="test-001",
            ticker="005930",
            alert_level="YELLOW",
            action="tighten_stop",
            confidence=0.75,
        )
        d = j.to_dict()
        assert d["action"] == "tighten_stop"
        assert d["confidence"] == 0.75


# ═══════════════════════════════════════════════════
# 2. SQLite 저장소 테스트
# ═══════════════════════════════════════════════════

class TestSqliteStore:
    @pytest.fixture
    def store(self, tmp_path):
        """임시 DB로 테스트"""
        db_path = tmp_path / "test_intraday.db"
        return SqliteIntradayStore(db_path)

    def test_init_creates_tables(self, store):
        stats = store.get_db_stats()
        assert "tick_data" in stats
        assert "candle_5min" in stats
        assert "investor_flow" in stats
        assert "market_context" in stats
        assert "sector_price" in stats
        assert "news_events" in stats
        assert "ai_judgments" in stats

    def test_save_and_get_tick(self, store):
        tick = {
            "ticker": "005930",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:00"),
            "current_price": 70000,
            "volume": 1000,
            "change_pct": 1.5,
        }
        store.save_tick(tick)

        ticks = store.get_recent_ticks("005930", minutes=5)
        assert len(ticks) == 1
        assert ticks[0]["current_price"] == 70000

    def test_save_ticks_batch(self, store):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:00")
        ticks = [
            {"ticker": "005930", "timestamp": now, "current_price": 70000},
            {"ticker": "000660", "timestamp": now, "current_price": 120000},
        ]
        store.save_ticks_batch(ticks)

        t1 = store.get_recent_ticks("005930", minutes=5)
        t2 = store.get_recent_ticks("000660", minutes=5)
        assert len(t1) == 1
        assert len(t2) == 1

    def test_save_and_get_candle(self, store):
        candle = {
            "ticker": "005930",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:00"),
            "open": 70000,
            "high": 71000,
            "low": 69500,
            "close": 70500,
            "volume": 50000,
            "vwap": 70250.0,
        }
        store.save_candle(candle)

        candles = store.get_today_candles("005930")
        assert len(candles) == 1
        assert candles[0]["close"] == 70500

    def test_save_market_context(self, store):
        ctx = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:00"),
            "kospi": 2850.5,
            "kospi_change_pct": 0.35,
            "kosdaq": 910.2,
            "kosdaq_change_pct": -0.12,
        }
        store.save_market_context(ctx)

        latest = store.get_latest_market_context()
        assert latest is not None
        assert latest["kospi"] == 2850.5

    def test_save_investor_flow(self, store):
        flow = {
            "ticker": "005930",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:00"),
            "foreign_net_buy": 5000,
            "inst_net_buy": -2000,
            "individual_net_buy": -3000,
        }
        store.save_investor_flow(flow)

        flows = store.get_today_investor_flow("005930")
        assert len(flows) == 1
        assert flows[0]["foreign_net_buy"] == 5000

    def test_save_sector_price(self, store):
        sector = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:00"),
            "sector_code": "0003",
            "sector_name": "반도체",
            "index_value": 3500.0,
            "change_pct": 1.2,
        }
        store.save_sector_price(sector)

        sectors = store.get_today_sector_prices()
        assert len(sectors) == 1
        assert sectors[0]["sector_name"] == "반도체"

    def test_cleanup_old_data(self, store):
        # 오래된 데이터 삽입
        old_ts = "2020-01-01 09:00:00"
        store.save_tick({"ticker": "005930", "timestamp": old_ts, "current_price": 50000})

        deleted = store.cleanup_old_data(days=1)
        assert deleted >= 1

    def test_tick_count_today(self, store):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:00")
        store.save_tick({"ticker": "005930", "timestamp": now, "current_price": 70000})
        store.save_tick({"ticker": "005930", "timestamp": now, "current_price": 70100})

        count = store.get_tick_count_today("005930")
        assert count == 2

    def test_db_stats(self, store):
        stats = store.get_db_stats()
        assert all(v == 0 for v in stats.values())

        store.save_tick({
            "ticker": "005930",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:00"),
            "current_price": 70000,
        })
        stats = store.get_db_stats()
        assert stats["tick_data"] == 1


# ═══════════════════════════════════════════════════
# 3. IntradayCollector 테스트 (Mock 포트)
# ═══════════════════════════════════════════════════

class TestIntradayCollector:
    @pytest.fixture
    def mock_data_port(self):
        port = MagicMock()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:00")

        port.fetch_ticks_batch.return_value = [
            {"ticker": "005930", "timestamp": now, "current_price": 70000},
            {"ticker": "000660", "timestamp": now, "current_price": 120000},
        ]
        port.fetch_minute_candles.return_value = [
            {"ticker": "005930", "timestamp": now, "open": 70000,
             "high": 71000, "low": 69500, "close": 70500,
             "volume": 50000, "vwap": 70250.0},
        ]
        port.fetch_investor_flow.return_value = {
            "ticker": "005930", "timestamp": now,
            "foreign_net_buy": 5000, "inst_net_buy": -2000,
            "individual_net_buy": -3000, "foreign_cum_net": 5000,
            "inst_cum_net": -2000, "program_net_buy": 0,
        }
        port.fetch_market_index.return_value = {
            "timestamp": now, "kospi": 2850.5, "kospi_change_pct": 0.35,
            "kosdaq": 910.2, "kosdaq_change_pct": -0.12,
        }
        port.fetch_sector_prices.return_value = [
            {"timestamp": now, "sector_code": "0003", "sector_name": "반도체",
             "index_value": 3500.0, "change_pct": 1.2},
        ]
        return port

    @pytest.fixture
    def store(self, tmp_path):
        return SqliteIntradayStore(tmp_path / "test.db")

    @pytest.fixture
    def collector(self, mock_data_port, store):
        config = {
            "intraday_monitor": {
                "tick_interval_sec": 60,
                "candle_interval_min": 5,
                "flow_interval_min": 10,
                "cleanup_days": 30,
            }
        }
        return IntradayCollector(
            config=config,
            data_port=mock_data_port,
            store_port=store,
            holdings=["005930", "000660"],
        )

    def test_collect_once(self, collector):
        result = collector.collect_once()
        assert result["ticks"] == 2
        assert result["candles"] == 2  # 2 holdings
        assert result["flows"] == 2
        assert result["market"] is True
        assert result["sectors"] == 1

    def test_update_holdings(self, collector):
        collector.update_holdings(["005930", "035420"])
        assert "035420" in collector.holdings
        assert "000660" not in collector.holdings

    def test_get_status(self, collector):
        status = collector.get_status()
        assert status["running"] is False
        assert status["holdings_count"] == 2
        assert "tick" in status["collect_count"]

    def test_collect_stores_data(self, collector, store):
        collector.collect_once()

        # 틱 확인
        ticks = store.get_recent_ticks("005930", minutes=5)
        assert len(ticks) >= 1

        # 시장 컨텍스트 확인
        ctx = store.get_latest_market_context()
        assert ctx is not None
        assert ctx["kospi"] == 2850.5

        # DB 통계
        stats = store.get_db_stats()
        assert stats["tick_data"] >= 2
        assert stats["market_context"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
