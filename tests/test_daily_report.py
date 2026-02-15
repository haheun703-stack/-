"""일일 리포트 시스템 테스트 — 엔티티, 포트, 템플릿, 스케줄러 통합"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── Entity 테스트 ──────────────────────────────────


class TestGameModels:
    """6D 게임 엔티티 모델 테스트"""

    def test_game_design_analysis_creation(self):
        from src.entities.game_models import GameDesignAnalysis

        gda = GameDesignAnalysis(
            ticker="005930",
            name="삼성전자",
            designer="반도체 사이클 + AI 수혜",
            our_role="구조적 성장 동반자",
            edge="AI 메모리 수요 폭증, HBM3E 독점",
            trap_description="중국 경쟁 심화",
            trap_risk_pct=25,
            game_score=72.0,
            reasoning="게임이 유리한 구간",
        )
        assert gda.ticker == "005930"
        assert gda.game_score == 72.0
        assert gda.trap_risk_pct == 25

    def test_game_design_to_dict(self):
        from src.entities.game_models import GameDesignAnalysis

        gda = GameDesignAnalysis(
            ticker="T", name="N", designer="D", our_role="R",
            edge="E", trap_description="T", trap_risk_pct=30,
            game_score=60.0, reasoning="R",
        )
        d = gda.to_dict()
        assert d["ticker"] == "T"
        assert d["game_score"] == 60.0
        assert isinstance(d, dict)

    def test_dimension_scores_creation(self):
        from src.entities.game_models import DimensionScores

        ds = DimensionScores(ticker="005930", name="삼성전자")
        assert ds.d1_price_pct == 0.0
        assert ds.total_5d == 0.0

    def test_dimension_scores_calc_totals(self):
        from src.entities.game_models import DimensionScores

        ds = DimensionScores(
            ticker="005930", name="삼성전자",
            d1_price_pct=80.0,
            d2_target_gap_pct=20.0,
            d3_multifactor=60.0,
            d4_timing_score=70.0,
            d5_metagame_score=65.0,
            d6_game_score=72.0,
            d6_trap_pct=25.0,
        )
        ds.calc_totals()

        assert ds.total_5d > 0
        assert ds.total_6d > 0
        # 6D = 5D*0.8 + 6D*0.2
        expected_6d = ds.total_5d * 0.80 + ds.d6_game_score * 0.20
        assert abs(ds.total_6d - expected_6d) < 0.01

    def test_dimension_scores_to_dict(self):
        from src.entities.game_models import DimensionScores

        ds = DimensionScores(ticker="T", name="N")
        d = ds.to_dict()
        assert "d1_price_pct" in d
        assert "total_6d" in d

    def test_portfolio_report_data(self):
        from src.entities.game_models import PortfolioReportData

        report = PortfolioReportData(report_date="2026-02-16")
        assert report.stock_count == 0
        assert report.stocks == []
        assert report.action_plan == []

    def test_stock_report_data(self):
        from src.entities.game_models import StockReportData

        srd = StockReportData(
            rank=1, ticker="005930", name="삼성전자",
            current_price=72000, return_pct=5.2,
            shares=10, hold_days=30, investment=680000,
        )
        assert srd.rank == 1
        assert srd.action == ""
        assert srd.forecast == ""


# ─── Port 인터페이스 테스트 ──────────────────────────────────


class TestGameAnalystPort:
    """GameAnalystPort 인터페이스 테스트"""

    def test_port_is_abstract(self):
        from abc import ABC
        from src.use_cases.ports import GameAnalystPort

        assert issubclass(GameAnalystPort, ABC)

    def test_port_has_required_methods(self):
        from src.use_cases.ports import GameAnalystPort

        assert hasattr(GameAnalystPort, "analyze_game")
        assert hasattr(GameAnalystPort, "batch_analyze")

    def test_agent_implements_port(self):
        from src.agents.game_analyst import GameAnalystAgent
        from src.use_cases.ports import GameAnalystPort

        assert issubclass(GameAnalystAgent, GameAnalystPort)


# ─── Agent 테스트 ──────────────────────────────────


class TestGameAnalystAgent:
    """GameAnalyst 에이전트 단위 테스트"""

    def test_import(self):
        from src.agents.game_analyst import GameAnalystAgent
        assert GameAnalystAgent is not None

    def test_format_game_input(self):
        from src.agents.game_analyst import _format_game_input

        ctx = {
            "ticker": "005930",
            "name": "삼성전자",
            "current_price": 72000,
            "return_pct": 5.2,
            "hold_days": 30,
            "pct_52w_high": 85.0,
            "per": 12.5,
            "target_price": 90000,
            "multifactor_score": 60.0,
            "clock_hour": 7.5,
            "timing_score": 70.0,
            "crowd_pct": 40.0,
            "neglect_score": 60.0,
            "sector": "반도체",
        }
        result = _format_game_input(ctx)
        assert "005930" in result
        assert "삼성전자" in result
        assert "72,000" in result
        assert "PER" in result
        assert "나선시계" in result

    def test_format_game_input_minimal(self):
        from src.agents.game_analyst import _format_game_input

        result = _format_game_input({"ticker": "X"})
        assert "X" in result

    def test_agent_in_package(self):
        from src.agents import GameAnalystAgent
        assert GameAnalystAgent is not None


# ─── 템플릿 테스트 ──────────────────────────────────


class TestTemplates:
    """HTML 템플릿 존재 및 구조 테스트"""

    def test_portfolio_template_exists(self):
        assert Path("templates/portfolio_report.html").exists()

    def test_journal_template_exists(self):
        assert Path("templates/daily_journal.html").exists()

    def test_portfolio_template_jinja_syntax(self):
        """Jinja2 문법 오류 검증"""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader("templates"))
        # 파싱만 테스트 (렌더링은 데이터 필요)
        template = env.get_template("portfolio_report.html")
        assert template is not None

    def test_journal_template_jinja_syntax(self):
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("daily_journal.html")
        assert template is not None

    def test_portfolio_template_has_key_sections(self):
        content = Path("templates/portfolio_report.html").read_text(encoding="utf-8")
        assert "summary-bar" in content
        assert "stock-card" in content
        assert "d6-box" in content
        assert "section-title" in content
        assert "report.stocks" in content

    def test_journal_template_has_key_sections(self):
        content = Path("templates/daily_journal.html").read_text(encoding="utf-8")
        assert "journal.positions" in content
        assert "journal.trades" in content
        assert "journal.phases" in content


# ─── Portfolio Reporter 테스트 ──────────────────────────────────


class TestPortfolioReporter:
    """PortfolioReporter Use Case 테스트"""

    def test_import(self):
        from src.use_cases.portfolio_reporter import PortfolioReporter
        assert PortfolioReporter is not None

    def test_score_color(self):
        from src.use_cases.portfolio_reporter import _score_color
        assert _score_color(70) == "green"
        assert _score_color(55) == "yellow"
        assert _score_color(45) == "orange"
        assert _score_color(30) == "red"

    def test_forecast_emoji(self):
        from src.use_cases.portfolio_reporter import _forecast_emoji
        assert _forecast_emoji("up") == "📈"
        assert _forecast_emoji("down") == "📉"
        assert _forecast_emoji("flat") == "➡️"

    def test_action_emoji(self):
        from src.use_cases.portfolio_reporter import _action_emoji
        assert _action_emoji("green") == "✅"
        assert _action_emoji("red") == "🔴"

    def test_rank_emoji(self):
        from src.use_cases.portfolio_reporter import _rank_emoji
        assert "🥇" in _rank_emoji(1)
        assert "🥈" in _rank_emoji(2)

    def test_generate_no_positions(self):
        """보유 포지션 없을 때 None 반환"""
        from src.use_cases.portfolio_reporter import PortfolioReporter

        reporter = PortfolioReporter({})
        # positions.json이 없거나 비어있으면 None
        with patch("src.use_cases.portfolio_reporter.PositionTracker") as mock_tracker:
            mock_tracker.return_value.positions = []
            result = reporter.generate()
            assert result is None

    def test_determine_action_high_score(self):
        from src.entities.game_models import DimensionScores, StockReportData
        from src.use_cases.portfolio_reporter import PortfolioReporter

        reporter = PortfolioReporter({})
        stock = StockReportData(
            rank=1, ticker="T", name="N", current_price=1000,
            return_pct=5.0, shares=10, hold_days=30, investment=9500,
            dimensions=DimensionScores(
                ticker="T", name="N",
                d4_timing_score=75.0,
                d5_metagame_score=65.0,
                d5_neglect_pct=35.0,
                d6_game_score=70.0,
                d6_trap_pct=20.0,
                total_6d=60.0,
            ),
        )
        reporter._determine_action(stock)
        assert stock.action == "보유"
        assert stock.action_color == "green"

    def test_determine_action_high_trap(self):
        from src.entities.game_models import DimensionScores, StockReportData
        from src.use_cases.portfolio_reporter import PortfolioReporter

        reporter = PortfolioReporter({})
        stock = StockReportData(
            rank=1, ticker="T", name="N", current_price=1000,
            return_pct=5.0, shares=10, hold_days=30, investment=9500,
            dimensions=DimensionScores(
                ticker="T", name="N",
                d4_timing_score=30.0,
                d5_metagame_score=25.0,
                d5_neglect_pct=75.0,
                d6_game_score=35.0,
                d6_trap_pct=65.0,
                total_6d=35.0,
            ),
        )
        reporter._determine_action(stock)
        assert stock.action_color in ("red", "orange")


# ─── Daily Journal 테스트 ──────────────────────────────────


class TestDailyJournal:
    """DailyJournalWriter Use Case 테스트"""

    def test_import(self):
        from src.use_cases.daily_journal import DailyJournalWriter
        assert DailyJournalWriter is not None

    def test_journal_data_creation(self):
        from src.use_cases.daily_journal import JournalData

        j = JournalData(date="2026-02-16")
        assert j.date == "2026-02-16"
        assert j.phases == []
        assert j.trades == []

    def test_generate_lessons_no_issues(self):
        from src.use_cases.daily_journal import DailyJournalWriter, JournalData

        writer = DailyJournalWriter({})
        journal = JournalData(date="2026-02-16")
        writer._generate_lessons(journal)
        assert len(journal.lessons) >= 1
        assert "특이사항 없음" in journal.lessons[0]

    def test_generate_lessons_stop_loss(self):
        from src.use_cases.daily_journal import DailyJournalWriter, JournalData

        writer = DailyJournalWriter({})
        journal = JournalData(date="2026-02-16")
        journal.trades = [{"name": "테스트", "reason": "stop_loss"}]
        writer._generate_lessons(journal)
        assert any("손절" in l for l in journal.lessons)

    def test_generate_lessons_high_profit(self):
        from src.use_cases.daily_journal import DailyJournalWriter, JournalData

        writer = DailyJournalWriter({})
        journal = JournalData(date="2026-02-16")
        journal.positions = [{"name": "테스트", "pnl_pct": 12.5}]
        writer._generate_lessons(journal)
        assert any("이익 보전" in l for l in journal.lessons)

    def test_render_html(self):
        """HTML 렌더링이 에러 없이 동작하는지"""
        from src.use_cases.daily_journal import DailyJournalWriter, JournalData

        writer = DailyJournalWriter({})
        journal = JournalData(
            date="2026-02-16",
            day_type="거래일",
            kospi_str="2,500 (+0.5%)",
            total_eval=10000000,
            daily_pnl=50000,
            daily_pnl_pct=0.5,
            trade_count=0,
            generated_time="16:30",
        )
        journal.positions = [
            {
                "ticker": "005930", "name": "삼성전자", "shares": 10,
                "entry_price": 70000, "current_price": 72000,
                "pnl_pct": 2.86, "grade": "A", "hold_days": 15,
            }
        ]
        journal.phases = [
            {"time": "08:25", "name": "Phase 4.5: 장전 리포트", "status": "ok", "status_label": "완료", "note": "정상"},
        ]
        journal.lessons = ["특이사항 없음"]

        html = writer._render_html(journal)
        assert "삼성전자" in html
        assert "2026-02-16" in html
        assert "특이사항 없음" in html


# ─── 스케줄러 통합 테스트 ──────────────────────────────────


class TestSchedulerIntegration:
    """DailyScheduler Phase 4.5/9 등록 테스트"""

    def test_scheduler_has_morning_report(self):
        from scripts.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler.__new__(DailyScheduler)
        assert hasattr(scheduler, "phase_morning_report")

    def test_scheduler_has_eod_journal(self):
        from scripts.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler.__new__(DailyScheduler)
        assert hasattr(scheduler, "phase_eod_journal")

    def test_scheduler_phases_mapping(self):
        """--run-now 9까지 지원 확인"""
        import scripts.daily_scheduler as mod
        # argparse에서 choices=range(1,10) 확인
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "range(1, 10)" in source

    def test_print_schedule_includes_new_phases(self):
        """dry-run에 Phase 4.5, 9가 포함되는지"""
        source = Path("scripts/daily_scheduler.py").read_text(encoding="utf-8")
        assert "장전 5D/6D" in source
        assert "업무일지" in source
        assert "08:25" in source
        assert "16:30" in source
