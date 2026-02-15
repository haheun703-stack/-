"""수급 이면 데이터 레이어 단위 테스트"""

import pytest

from src.entities.supply_demand_models import (
    InvestorFlowData,
    ShortSellingData,
    SupplyDemandScore,
)
from src.supply_demand_analyzer import SupplyDemandAnalyzer


# ═══════════════════════════════════════════════════
# Entity 모델 테스트
# ═══════════════════════════════════════════════════
class TestShortSellingData:
    def test_default_values(self):
        s = ShortSellingData(ticker="005930", date="20260216")
        assert s.short_ratio == 0.0
        assert s.short_spike_ratio == 1.0
        assert s.is_overheated is False

    def test_to_dict(self):
        s = ShortSellingData(
            ticker="005930", date="20260216",
            short_ratio=5.2, short_balance_ratio=2.1,
        )
        d = s.to_dict()
        assert d["ticker"] == "005930"
        assert d["short_ratio"] == 5.2


class TestInvestorFlowData:
    def test_default_values(self):
        f = InvestorFlowData(ticker="005930", date="20260216")
        assert f.foreign_consecutive_days == 0
        assert f.institution_cumulative_20d == 0

    def test_to_dict(self):
        f = InvestorFlowData(
            ticker="005930", date="20260216",
            foreign_consecutive_days=7,
            institution_cumulative_20d=5_000_000_000,
        )
        d = f.to_dict()
        assert d["foreign_consecutive_days"] == 7


class TestSupplyDemandScore:
    def test_trap_v2_neutral(self):
        """모든 값 중립 시 함정률 ~50"""
        score = SupplyDemandScore(ticker="005930", date="20260216")
        trap = score.calc_trap_v2(crowd_heat=50)
        assert 40 <= trap <= 60

    def test_trap_v2_high_risk(self):
        """위험 신호 시 함정률 상승"""
        score = SupplyDemandScore(
            ticker="005930", date="20260216",
            short_risk=90, spoofing_risk=80,
            execution_score=20, program_pressure=20,
        )
        trap = score.calc_trap_v2(crowd_heat=80)
        assert trap > 70

    def test_trap_v2_safe(self):
        """안전 신호 시 함정률 하락"""
        score = SupplyDemandScore(
            ticker="005930", date="20260216",
            short_risk=20, spoofing_risk=10,
            execution_score=90, program_pressure=90,
        )
        trap = score.calc_trap_v2(crowd_heat=20)
        assert trap < 30


# ═══════════════════════════════════════════════════
# Analyzer 테스트
# ═══════════════════════════════════════════════════
class TestAnalyzerShortSelling:
    def setup_method(self):
        self.analyzer = SupplyDemandAnalyzer()

    def test_normal_short(self):
        """정상 공매도 → 중립 점수"""
        s = ShortSellingData(
            ticker="005930", date="20260216",
            short_spike_ratio=1.0, short_balance_ratio=1.0,
        )
        score = self.analyzer.analyze("005930", "20260216", short=s)
        assert 40 <= score.short_risk <= 60

    def test_spike_short(self):
        """공매도 스파이크 → 위험 상승"""
        s = ShortSellingData(
            ticker="005930", date="20260216",
            short_spike_ratio=2.5,
            short_balance_ratio=4.0,
            lending_change_5d=25,
        )
        score = self.analyzer.analyze("005930", "20260216", short=s)
        assert score.short_risk >= 85
        assert score.trap_adjustment > 0

    def test_overheated_short(self):
        """과열종목 지정 → 높은 위험"""
        s = ShortSellingData(
            ticker="005930", date="20260216",
            is_overheated=True,
        )
        score = self.analyzer.analyze("005930", "20260216", short=s)
        assert score.short_risk >= 60
        assert score.trap_adjustment >= 10


class TestAnalyzerInvestorFlow:
    def setup_method(self):
        self.analyzer = SupplyDemandAnalyzer()

    def test_strong_foreign_buying(self):
        """외국인 5일+ 연속 순매수 → 높은 수급 점수"""
        f = InvestorFlowData(
            ticker="005930", date="20260216",
            foreign_consecutive_days=7,
            institution_cumulative_20d=10_000_000_000,
            pension_net=1_000_000_000,
        )
        score = self.analyzer.analyze("005930", "20260216", flow=f)
        assert score.institutional >= 80
        assert score.smart_money_boost >= 0.3
        assert score.trap_adjustment < 0  # 함정률 하락

    def test_neutral_flow(self):
        """중립 수급"""
        f = InvestorFlowData(ticker="005930", date="20260216")
        score = self.analyzer.analyze("005930", "20260216", flow=f)
        assert 40 <= score.institutional <= 60
        assert score.smart_money_boost == 0.0

    def test_energy_boost_on_dual_buy(self):
        """외국인+기관 동시 순매수 → 에너지 부스트"""
        f = InvestorFlowData(
            ticker="005930", date="20260216",
            foreign_net=5_000_000_000,
            institution_net=3_000_000_000,
        )
        score = self.analyzer.analyze("005930", "20260216", flow=f)
        assert score.energy_adjustment > 0

    def test_energy_drag_on_dual_sell(self):
        """외국인+기관 동시 순매도 → 에너지 감소"""
        f = InvestorFlowData(
            ticker="005930", date="20260216",
            foreign_net=-5_000_000_000,
            institution_net=-3_000_000_000,
        )
        score = self.analyzer.analyze("005930", "20260216", flow=f)
        assert score.energy_adjustment < 0


class TestAnalyzerBatch:
    def test_batch_analyze(self):
        """일괄 분석"""
        analyzer = SupplyDemandAnalyzer()
        collected = {
            "005930": {
                "short": ShortSellingData(
                    ticker="005930", date="20260216", short_spike_ratio=1.2
                ),
                "flow": InvestorFlowData(
                    ticker="005930", date="20260216",
                    foreign_consecutive_days=3,
                ),
            },
            "035420": {
                "short": ShortSellingData(
                    ticker="035420", date="20260216", is_overheated=True
                ),
                "flow": None,
            },
        }
        results = analyzer.analyze_batch(collected, "20260216")
        assert "005930" in results
        assert "035420" in results
        assert results["035420"].short_risk >= 60

    def test_batch_with_none(self):
        """None 데이터 처리"""
        analyzer = SupplyDemandAnalyzer()
        collected = {"005930": {"short": None, "flow": None}}
        results = analyzer.analyze_batch(collected, "20260216")
        assert results["005930"].short_risk == 50  # 기본 중립
