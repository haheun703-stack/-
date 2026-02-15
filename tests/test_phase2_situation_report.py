"""
Phase 2: 상황보고서 생성기 단위 테스트

테스트 대상:
  1. SituationReport / StockSituation 엔티티 모델
  2. SituationReporter — Mock 데이터로 보고서 생성
  3. 장중 추세 분석 (intraday_trend)
  4. 수급 분류 (flow_direction)
  5. 경고 / 기회 감지
  6. Claude API 프롬프트 텍스트 생성
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.entities.intraday_models import (
    SituationReport,
    StockSituation,
)
from src.use_cases.situation_reporter import SituationReporter

# ═══════════════════════════════════════════════════
# 1. 엔티티 모델 테스트
# ═══════════════════════════════════════════════════

class TestSituationEntities:
    def test_stock_situation_to_dict(self):
        sit = StockSituation(
            ticker="005930",
            name="삼성전자",
            current_price=70000,
            change_pct=1.5,
            rsi_14=42.3,
            ou_z=-1.2,
        )
        d = sit.to_dict()
        assert d["ticker"] == "005930"
        assert d["name"] == "삼성전자"
        assert d["current_price"] == 70000
        assert d["rsi_14"] == 42.3

    def test_stock_situation_defaults(self):
        sit = StockSituation()
        assert sit.ticker == ""
        assert sit.current_price == 0
        assert sit.alerts == []

    def test_situation_report_to_dict(self):
        report = SituationReport(
            timestamp="2026-02-15 10:30:00",
            report_type="regular",
            kospi=2850.5,
            kospi_change_pct=0.35,
            kosdaq=910.2,
            kosdaq_change_pct=-0.12,
            market_regime="소폭상승",
            holdings_count=2,
        )
        d = report.to_dict()
        assert d["timestamp"] == "2026-02-15 10:30:00"
        assert d["market"]["kospi"] == 2850.5
        assert d["market"]["market_regime"] == "소폭상승"
        assert d["holdings_count"] == 2

    def test_situation_report_to_prompt_text(self):
        stock_dict = StockSituation(
            ticker="005930",
            name="삼성전자",
            current_price=70000,
            change_pct=1.5,
            shares=100,
            entry_price=65000,
            pnl_pct=7.69,
            hold_days=5,
            stop_loss=62000,
            target_price=80000,
            intraday_trend="상승",
            price_from_open_pct=0.75,
            foreign_net_buy=5000,
            inst_net_buy=-2000,
            flow_direction="외인 매수, 기관 매도",
            rsi_14=42.3,
            adx_14=25.1,
            bb_position=0.35,
            ou_z=-0.8,
            macd_histogram=0.0015,
            trix_signal="매수세",
            volume_ratio=1.2,
            smart_z=0.5,
            v8_grade="B",
            v8_total_score=0.62,
            v8_gate_passed=True,
            alerts=["거래량 급증 (2.5x)"],
        ).to_dict()

        report = SituationReport(
            timestamp="2026-02-15 10:30:00",
            report_type="regular",
            kospi=2850.5,
            kospi_change_pct=0.35,
            kosdaq=910.2,
            kosdaq_change_pct=-0.12,
            market_regime="소폭상승",
            top_sectors=[{"name": "반도체", "change_pct": 2.1}],
            bottom_sectors=[{"name": "건설", "change_pct": -1.5}],
            stocks=[stock_dict],
            summary_alerts=["KOSPI 소폭 상승"],
            opportunities=["삼성전자 TRIX 매수세"],
            holdings_count=1,
        )

        text = report.to_prompt_text()
        assert "=== 상황보고서" in text
        assert "삼성전자" in text
        assert "KOSPI: 2,850.5" in text
        assert "70,000원" in text
        assert "RSI=42.3" in text
        assert "OU_z=-0.80" in text
        assert "반도체" in text

    def test_empty_report_prompt_text(self):
        report = SituationReport(
            timestamp="2026-02-15 10:30:00",
            kospi=0.0,
        )
        text = report.to_prompt_text()
        assert "=== 상황보고서" in text


# ═══════════════════════════════════════════════════
# 2. SituationReporter 테스트 (Mock Store)
# ═══════════════════════════════════════════════════

class TestSituationReporter:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:00")

        # 틱 데이터
        store.get_recent_ticks.return_value = [
            {
                "ticker": "005930",
                "timestamp": now,
                "current_price": 70000,
                "change_pct": 1.5,
                "open_price": 69500,
                "high_price": 70200,
                "low_price": 69000,
                "cum_volume": 5000000,
            }
        ]

        # 5분봉
        store.get_today_candles.return_value = [
            {"timestamp": "2026-02-15 09:05:00", "close": 69500, "high": 69700, "low": 69200},
            {"timestamp": "2026-02-15 09:10:00", "close": 69800, "high": 70000, "low": 69500},
            {"timestamp": "2026-02-15 09:15:00", "close": 69600, "high": 69900, "low": 69300},
            {"timestamp": "2026-02-15 09:20:00", "close": 69900, "high": 70100, "low": 69400},
            {"timestamp": "2026-02-15 09:25:00", "close": 70000, "high": 70200, "low": 69800},
            {"timestamp": "2026-02-15 09:30:00", "close": 70100, "high": 70300, "low": 69900},
        ]

        # 투자자 수급
        store.get_today_investor_flow.return_value = [
            {
                "ticker": "005930",
                "timestamp": now,
                "foreign_net_buy": 5000,
                "inst_net_buy": -2000,
                "individual_net_buy": -3000,
            }
        ]

        # 시장 컨텍스트
        store.get_latest_market_context.return_value = {
            "timestamp": now,
            "kospi": 2850.5,
            "kospi_change_pct": 0.35,
            "kosdaq": 910.2,
            "kosdaq_change_pct": -0.12,
        }

        # 업종 시세
        store.get_today_sector_prices.return_value = [
            {"sector_name": "반도체", "change_pct": 2.1},
            {"sector_name": "자동차", "change_pct": 0.5},
            {"sector_name": "화학", "change_pct": -0.3},
            {"sector_name": "은행", "change_pct": -0.8},
            {"sector_name": "건설", "change_pct": -1.5},
        ]

        return store

    @pytest.fixture
    def reporter(self, mock_store, tmp_path):
        # 임시 positions.json
        positions = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "shares": 100,
                "entry_price": 65000,
                "stop_loss": 62000,
                "target_price": 80000,
                "hold_days": 5,
            }
        ]
        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps(positions, ensure_ascii=False), encoding="utf-8")

        # 임시 parquet dir (비어있음)
        parquet_dir = tmp_path / "parquet"
        parquet_dir.mkdir()

        config = {
            "intraday_monitor": {
                "situation_report": {
                    "interval_min": 30,
                    "emergency_threshold": 0.03,
                }
            }
        }

        return SituationReporter(
            config=config,
            store_port=mock_store,
            positions_file=pos_file,
            parquet_dir=parquet_dir,
        )

    def test_generate_regular_report(self, reporter):
        report = reporter.generate(
            holdings=["005930"],
            report_type="regular",
        )
        assert report.report_type == "regular"
        assert report.holdings_count == 1
        assert report.kospi == 2850.5
        assert report.market_regime == "횡보"  # avg(0.35, -0.12)=0.115 < 0.3
        assert len(report.stocks) == 1
        assert report.stocks[0]["ticker"] == "005930"
        assert report.stocks[0]["current_price"] == 70000
        assert report.generation_ms >= 0

    def test_generate_with_position_info(self, reporter):
        report = reporter.generate(holdings=["005930"])
        stock = report.stocks[0]
        assert stock["shares"] == 100
        assert stock["entry_price"] == 65000
        assert stock["pnl_pct"] > 0  # 70000/65000 ≈ +7.7%
        assert stock["hold_days"] == 5

    def test_market_regime_classification(self, reporter):
        # 시장 레짐 분류
        assert reporter._classify_market_regime({"kospi_change_pct": 2.0, "kosdaq_change_pct": 1.5}) == "강세"
        assert reporter._classify_market_regime({"kospi_change_pct": 0.5, "kosdaq_change_pct": 0.3}) == "소폭상승"
        assert reporter._classify_market_regime({"kospi_change_pct": 0.0, "kosdaq_change_pct": 0.0}) == "횡보"
        assert reporter._classify_market_regime({"kospi_change_pct": -0.5, "kosdaq_change_pct": -0.5}) == "소폭하락"
        assert reporter._classify_market_regime({"kospi_change_pct": -2.0, "kosdaq_change_pct": -2.0}) == "약세"

    def test_intraday_trend_rising(self, reporter):
        # 상승 추세
        candles = [
            {"close": 100}, {"close": 101}, {"close": 102},
            {"close": 103}, {"close": 104}, {"close": 105},
        ]
        assert reporter._analyze_intraday_trend(candles) == "상승"

    def test_intraday_trend_falling(self, reporter):
        candles = [
            {"close": 105}, {"close": 104}, {"close": 103},
            {"close": 102}, {"close": 101}, {"close": 100},
        ]
        assert reporter._analyze_intraday_trend(candles) == "하락"

    def test_intraday_trend_v_shape(self, reporter):
        candles = [
            {"close": 105}, {"close": 103}, {"close": 100},
            {"close": 101}, {"close": 104}, {"close": 106},
        ]
        assert reporter._analyze_intraday_trend(candles) == "V반등"

    def test_intraday_trend_inverted_v(self, reporter):
        candles = [
            {"close": 100}, {"close": 103}, {"close": 106},
            {"close": 105}, {"close": 102}, {"close": 99},
        ]
        assert reporter._analyze_intraday_trend(candles) == "역V"

    def test_intraday_trend_sideways(self, reporter):
        candles = [
            {"close": 100}, {"close": 100}, {"close": 100},
            {"close": 100}, {"close": 100}, {"close": 100},
        ]
        assert reporter._analyze_intraday_trend(candles) == "횡보"

    def test_classify_flow(self, reporter):
        assert reporter._classify_flow(5000, 3000) == "외인+기관 동반매수"
        assert reporter._classify_flow(-5000, -3000) == "외인+기관 동반매도"
        assert reporter._classify_flow(5000, -3000) == "외인 매수, 기관 매도"
        assert reporter._classify_flow(-5000, 3000) == "외인 매도, 기관 매수"

    def test_detect_alerts_sharp_drop(self, reporter):
        sit = StockSituation(ticker="005930", change_pct=-3.5)
        alerts = reporter._detect_stock_alerts(sit)
        assert any("급락" in a for a in alerts)

    def test_detect_alerts_stop_loss_breach(self, reporter):
        sit = StockSituation(
            ticker="005930",
            current_price=61000,
            stop_loss=62000,
        )
        alerts = reporter._detect_stock_alerts(sit)
        assert any("손절선 이탈" in a for a in alerts)

    def test_detect_alerts_volume_surge(self, reporter):
        sit = StockSituation(ticker="005930", volume_ratio=3.5)
        alerts = reporter._detect_stock_alerts(sit)
        assert any("거래량 폭발" in a for a in alerts)

    def test_detect_alerts_rsi_extreme(self, reporter):
        sit_high = StockSituation(ticker="005930", rsi_14=75)
        alerts_high = reporter._detect_stock_alerts(sit_high)
        assert any("과매수" in a for a in alerts_high)

        sit_low = StockSituation(ticker="005930", rsi_14=25)
        alerts_low = reporter._detect_stock_alerts(sit_low)
        assert any("과매도" in a for a in alerts_low)

    def test_generate_opportunities(self, reporter):
        stocks = [
            StockSituation(
                ticker="005930",
                name="삼성전자",
                ou_z=-1.8,
                rsi_14=40,
            ),
        ]
        opps = reporter._generate_opportunities(stocks)
        assert any("OU 반등" in o for o in opps)

    def test_generate_opportunities_trix_golden(self, reporter):
        stocks = [
            StockSituation(
                ticker="005930",
                name="삼성전자",
                trix_signal="골든크로스",
            ),
        ]
        opps = reporter._generate_opportunities(stocks)
        assert any("골든크로스" in o for o in opps)

    def test_check_emergency_no_change(self, reporter):
        # 첫 호출 — 이전 가격 없으므로 None
        assert reporter.check_emergency(["005930"]) is None

    def test_check_emergency_detected(self, reporter, mock_store):
        # 이전 가격 설정
        reporter._last_prices = {"005930": 70000}

        # 5% 급등
        now = datetime.now().strftime("%Y-%m-%d %H:%M:00")
        mock_store.get_recent_ticks.return_value = [
            {"ticker": "005930", "timestamp": now, "current_price": 73500}
        ]

        result = reporter.check_emergency(["005930"])
        assert result is not None
        assert "급등" in result

    def test_sector_top_bottom(self, reporter):
        report = reporter.generate(holdings=["005930"])
        assert len(report.top_sectors) <= 3
        assert len(report.bottom_sectors) <= 3
        # 반도체가 1등 (change_pct=2.1)
        assert report.top_sectors[0]["name"] == "반도체"
        # bottom은 sorted_sectors[-3:] = [화학, 은행, 건설]
        bottom_names = [s["name"] for s in report.bottom_sectors]
        assert "건설" in bottom_names
        assert report.bottom_sectors[-1]["name"] == "건설"  # 꼴등

    def test_prompt_text_full_integration(self, reporter):
        """전체 통합: 생성 → to_prompt_text()"""
        report = reporter.generate(holdings=["005930"])
        text = report.to_prompt_text()
        assert "=== 상황보고서" in text
        assert "KOSPI" in text
        assert "005930" in text
        assert len(text) > 200  # 의미있는 길이

    def test_load_holdings_from_positions(self, reporter):
        holdings = reporter._load_holdings()
        assert "005930" in holdings

    def test_data_freshness(self, reporter):
        report = reporter.generate(holdings=["005930"])
        # 방금 생성된 틱이므로 몇 초 이내
        assert report.data_freshness_sec >= 0


# ═══════════════════════════════════════════════════
# 3. 빈 데이터 / 예외 처리 테스트
# ═══════════════════════════════════════════════════

class TestEdgeCases:
    @pytest.fixture
    def empty_store(self):
        store = MagicMock()
        store.get_recent_ticks.return_value = []
        store.get_today_candles.return_value = []
        store.get_today_investor_flow.return_value = []
        store.get_latest_market_context.return_value = None
        store.get_today_sector_prices.return_value = []
        return store

    @pytest.fixture
    def reporter_empty(self, empty_store, tmp_path):
        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]", encoding="utf-8")
        parquet_dir = tmp_path / "parquet"
        parquet_dir.mkdir()

        return SituationReporter(
            config={"intraday_monitor": {}},
            store_port=empty_store,
            positions_file=pos_file,
            parquet_dir=parquet_dir,
        )

    def test_generate_with_no_data(self, reporter_empty):
        report = reporter_empty.generate(holdings=["005930"])
        assert report.holdings_count == 1
        assert report.kospi == 0.0
        assert len(report.stocks) == 1
        assert report.stocks[0]["current_price"] == 0

    def test_generate_with_no_holdings(self, reporter_empty):
        report = reporter_empty.generate(holdings=[])
        assert report.holdings_count == 0
        assert report.stocks == []

    def test_trend_with_insufficient_candles(self, reporter_empty):
        assert reporter_empty._analyze_intraday_trend([]) == "데이터부족"
        assert reporter_empty._analyze_intraday_trend([{"close": 100}]) == "데이터부족"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
