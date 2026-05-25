"""test_portfolio_reporter_no_archive.py — P0-슬러지 archive import 제거 회귀.

배경 (퐝가님 5/24 안전 수정):
  src/use_cases/portfolio_reporter.py에 다음 import가 있었음:
    from scripts.archive.deprecated_agents.game_analyst import GameAnalystAgent

  CLAUDE.md 위반: "scripts/archive/ — 절대 참조·실행·import 금지"

  처리: _run_6d_analysis 메서드의 archive import 줄 제거 +
        deprecated 처리 (dimensions에 중립 기본값 50.0 적용).

검증:
  1. 모듈 import 성공 (archive 참조 흔적 없음)
  2. PortfolioReporter 클래스 임포트 가능
  3. _run_6d_analysis가 archive를 호출하지 않고 기본값만 설정
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPortfolioReporterNoArchive(unittest.TestCase):

    def test_01_module_imports_no_archive(self):
        """archive import 제거 후 모듈 import 성공."""
        # 캐시 제거
        if "src.use_cases.portfolio_reporter" in sys.modules:
            del sys.modules["src.use_cases.portfolio_reporter"]

        import src.use_cases.portfolio_reporter  # 성공해야 함
        # archive 흔적 없음
        src_text = open(
            src.use_cases.portfolio_reporter.__file__, "r", encoding="utf-8"
        ).read()
        self.assertNotIn(
            "scripts.archive", src_text,
            "scripts.archive import가 남아있음 — CLAUDE.md 위반",
        )
        self.assertNotIn(
            "from scripts.archive", src_text,
            "from scripts.archive import 흔적이 있음",
        )

    def test_02_portfolio_reporter_class_importable(self):
        """PortfolioReporter 클래스 import 가능."""
        from src.use_cases.portfolio_reporter import PortfolioReporter
        self.assertTrue(callable(PortfolioReporter))

    def test_03_run_6d_analysis_uses_defaults(self):
        """_run_6d_analysis가 archive 호출 없이 dimensions 기본값(50.0)을 설정."""
        from src.use_cases.portfolio_reporter import PortfolioReporter
        from src.entities.game_models import DimensionScores, StockReportData

        reporter = PortfolioReporter(config={})

        dims = DimensionScores(ticker="005930", name="삼성전자")
        stock = StockReportData(
            rank=0,
            ticker="005930",
            name="삼성전자",
            current_price=70_000,
            return_pct=1.5,
            shares=10,
            hold_days=5,
            investment=700_000,
            dimensions=dims,
        )

        # archive 모듈이 sys.modules에 들어오면 안 됨 — 호출 전후 검증
        reporter._run_6d_analysis([stock])

        self.assertEqual(stock.dimensions.d6_game_score, 50.0)
        self.assertEqual(stock.dimensions.d6_trap_pct, 50.0)

        # archive 모듈은 import되지 않아야 함
        loaded_modules = [m for m in sys.modules if "scripts.archive" in m]
        self.assertEqual(
            loaded_modules, [],
            f"archive 모듈이 로드됨: {loaded_modules}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
