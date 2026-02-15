"""
C-Suite 에이전트 테스트 — CFO, Risk Sentinel, Macro Analyst

테스트 구성:
  - TestCFOEntities: CFO 엔티티 모델 인스턴스화 및 직렬화 (5개)
  - TestRiskEntities: Risk Sentinel 엔티티 모델 (5개)
  - TestMacroEntities: Macro Analyst 엔티티 모델 (5개)
  - TestPortInterfaces: 포트 인터페이스 준수 검증 (3개)
  - TestCFOPromptFormatting: CFO 프롬프트 포매팅 (3개)
  - TestRiskSentinelFormatting: Risk Sentinel 포매팅 (3개)
  - TestMacroAnalystFormatting: Macro Analyst 포매팅 (3개)
  - TestAgentImports: 에이전트 패키지 import 검증 (3개)
"""


# ═══════════════════════════════════════════════
# 1. CFO 엔티티 모델 테스트
# ═══════════════════════════════════════════════

from src.entities.cfo_models import (
    CapitalAllocation,
    PortfolioHealthCheck,
    PortfolioRiskBudget,
    RiskLevel,
)


class TestCFOEntities:
    """CFO 엔티티 모델 테스트"""

    def test_risk_level_enum(self):
        """RiskLevel 열거형 값 확인"""
        assert RiskLevel.CONSERVATIVE.value == "conservative"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.AGGRESSIVE.value == "aggressive"
        assert RiskLevel.EMERGENCY.value == "emergency"

    def test_portfolio_risk_budget_defaults(self):
        """PortfolioRiskBudget 기본값 확인"""
        budget = PortfolioRiskBudget()
        assert budget.max_single_position_pct == 0.20
        assert budget.max_portfolio_risk_pct == 0.06
        assert budget.max_daily_loss_pct == 0.03

    def test_portfolio_risk_budget_remaining(self):
        """남은 리스크 예산 계산 (property)"""
        budget = PortfolioRiskBudget(
            max_portfolio_risk_pct=0.06,
            current_drawdown_pct=-0.03,
        )
        assert abs(budget.remaining_risk_budget - 0.03) < 1e-9

        # 낙폭이 최대 리스크를 초과하면 0
        budget2 = PortfolioRiskBudget(
            max_portfolio_risk_pct=0.06,
            current_drawdown_pct=-0.08,
        )
        assert budget2.remaining_risk_budget == 0

    def test_capital_allocation(self):
        """CapitalAllocation 인스턴스화"""
        alloc = CapitalAllocation(
            ticker="005930",
            recommended_size_pct=0.15,
            kelly_fraction=0.12,
            reasoning="삼성전자 A등급 진입",
        )
        assert alloc.ticker == "005930"
        assert alloc.recommended_size_pct == 0.15

    def test_health_check_to_dict(self):
        """PortfolioHealthCheck.to_dict() 검증"""
        health = PortfolioHealthCheck(
            overall_score=75,
            risk_level="moderate",
            positions_count=5,
            warnings=["집중도 높음"],
        )
        d = health.to_dict()
        assert d["overall_score"] == 75
        assert d["risk_level"] == "moderate"
        assert "집중도 높음" in d["warnings"]


# ═══════════════════════════════════════════════
# 2. Risk Sentinel 엔티티 모델 테스트
# ═══════════════════════════════════════════════

from src.entities.risk_models import (
    RiskDashboard,
    StressTestResult,
    TailRiskAlert,
    ThreatLevel,
)


class TestRiskEntities:
    """Risk Sentinel 엔티티 모델 테스트"""

    def test_threat_level_enum(self):
        """ThreatLevel 열거형 값 확인"""
        assert ThreatLevel.GREEN.value == "green"
        assert ThreatLevel.BLACK.value == "black"
        assert len(ThreatLevel) == 5

    def test_tail_risk_alert_defaults(self):
        """TailRiskAlert 기본값 확인"""
        alert = TailRiskAlert()
        assert alert.threat_level == "green"
        assert alert.recommended_action == "hold"
        assert alert.circuit_breaker_risk is False

    def test_tail_risk_alert_to_dict(self):
        """TailRiskAlert.to_dict() 검증"""
        alert = TailRiskAlert(
            threat_level="red",
            vkospi_level=42.5,
            alerts=["VKOSPI 40 돌파"],
            recommended_action="hedge",
        )
        d = alert.to_dict()
        assert d["threat_level"] == "red"
        assert d["vkospi_level"] == 42.5
        assert "VKOSPI 40 돌파" in d["alerts"]

    def test_stress_test_result(self):
        """StressTestResult 인스턴스화"""
        result = StressTestResult(
            scenario_name="KOSPI -5%",
            portfolio_impact_pct=-8.5,
            survival=True,
        )
        assert result.scenario_name == "KOSPI -5%"
        assert result.portfolio_impact_pct == -8.5
        assert result.survival is True

    def test_risk_dashboard_to_dict(self):
        """RiskDashboard.to_dict() 검증"""
        dashboard = RiskDashboard(
            timestamp="2026-02-16T10:00:00",
            overall_threat="yellow",
            portfolio_var_95=-0.025,
        )
        d = dashboard.to_dict()
        assert d["overall_threat"] == "yellow"
        assert d["portfolio_var_95"] == -0.025
        assert "tail_risk" in d
        assert "stress_tests" in d


# ═══════════════════════════════════════════════
# 3. Macro Analyst 엔티티 모델 테스트
# ═══════════════════════════════════════════════

from src.entities.macro_models import (
    MacroDashboard,
    MacroRegimeAnalysis,
    MarketRegime,
    SectorMomentum,
    SectorRotation,
)


class TestMacroEntities:
    """Macro Analyst 엔티티 모델 테스트"""

    def test_market_regime_enum(self):
        """MarketRegime 열거형 값 확인"""
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.CRISIS.value == "crisis"
        assert len(MarketRegime) == 6

    def test_sector_momentum_enum(self):
        """SectorMomentum 열거형 값 확인"""
        assert SectorMomentum.LEADING.value == "leading"
        assert SectorMomentum.LAGGING.value == "lagging"
        assert len(SectorMomentum) == 4

    def test_macro_regime_analysis_to_dict(self):
        """MacroRegimeAnalysis.to_dict() 검증"""
        analysis = MacroRegimeAnalysis(
            regime="bull",
            regime_confidence=0.85,
            key_factors=["외인 순매수", "KOSPI 신고가"],
        )
        d = analysis.to_dict()
        assert d["regime"] == "bull"
        assert d["regime_confidence"] == 0.85
        assert len(d["key_factors"]) == 2

    def test_sector_rotation(self):
        """SectorRotation 인스턴스화"""
        rotation = SectorRotation(
            rotation_signal="early_cycle",
            recommended_sectors=["금융", "반도체"],
            avoid_sectors=["건설"],
        )
        assert rotation.rotation_signal == "early_cycle"
        assert "반도체" in rotation.recommended_sectors

    def test_macro_dashboard_to_dict(self):
        """MacroDashboard.to_dict() 검증"""
        dashboard = MacroDashboard(
            timestamp="2026-02-16",
            kospi_trend="bullish",
            overall_stance="moderate",
        )
        d = dashboard.to_dict()
        assert d["kospi_trend"] == "bullish"
        assert d["overall_stance"] == "moderate"
        assert "regime" in d
        assert "sector_rotation" in d
        assert "market_breadth" in d


# ═══════════════════════════════════════════════
# 4. 포트 인터페이스 준수 테스트
# ═══════════════════════════════════════════════

from src.use_cases.ports import CFOPort, MacroAnalystPort, RiskSentinelPort


class TestPortInterfaces:
    """에이전트가 포트 인터페이스를 올바르게 구현하는지 검증"""

    def test_cfo_port_compliance(self):
        """CFOAgent가 CFOPort의 모든 추상 메서드를 구현하는지 확인"""
        from src.agents.cfo import CFOAgent

        # ABC의 추상 메서드 목록 확인
        required = {"allocate_capital", "health_check", "drawdown_analysis"}
        agent_methods = set(dir(CFOAgent))
        assert required.issubset(agent_methods)
        # 인스턴스화 가능해야 함 (추상 메서드 미구현 시 TypeError 발생)
        assert issubclass(CFOAgent, CFOPort)

    def test_risk_sentinel_port_compliance(self):
        """RiskSentinelAgent가 RiskSentinelPort의 모든 추상 메서드를 구현"""
        from src.agents.risk_sentinel import RiskSentinelAgent

        required = {"scan_tail_risk", "analyze_correlation", "stress_test"}
        agent_methods = set(dir(RiskSentinelAgent))
        assert required.issubset(agent_methods)
        assert issubclass(RiskSentinelAgent, RiskSentinelPort)

    def test_macro_analyst_port_compliance(self):
        """MacroAnalystAgent가 MacroAnalystPort의 모든 추상 메서드를 구현"""
        from src.agents.macro_analyst import MacroAnalystAgent

        required = {"analyze_regime", "analyze_sector_rotation", "analyze_breadth"}
        agent_methods = set(dir(MacroAnalystAgent))
        assert required.issubset(agent_methods)
        assert issubclass(MacroAnalystAgent, MacroAnalystPort)


# ═══════════════════════════════════════════════
# 5. CFO 프롬프트 포매팅 테스트
# ═══════════════════════════════════════════════

from src.agents.cfo import (
    _format_allocation_input,
    _format_drawdown_input,
    _format_health_input,
)


class TestCFOPromptFormatting:
    """CFO 프롬프트 포매팅 함수 테스트"""

    def test_format_allocation_input(self):
        """자본 배분 프롬프트 포매팅"""
        signal = {
            "ticker": "005930",
            "grade": "A",
            "zone_score": 0.82,
            "trigger_type": "T1_trix",
            "risk_reward_ratio": 2.5,
            "atr_value": 1500,
            "entry_price": 75000,
            "stop_loss": 72000,
        }
        portfolio = {
            "total_capital": 100_000_000,
            "cash": 50_000_000,
            "invested": 50_000_000,
            "positions": [
                {"ticker": "000660", "weight_pct": 15.0, "pnl_pct": 3.5},
            ],
        }
        risk_budget = {
            "portfolio_risk_pct": 0.04,
            "current_drawdown": -0.02,
            "max_drawdown": -0.15,
        }

        text = _format_allocation_input(signal, portfolio, risk_budget)
        assert "005930" in text
        assert "A" in text
        assert "000660" in text
        assert "100,000,000" in text

    def test_format_health_input(self):
        """포트폴리오 건강 진단 프롬프트 포매팅"""
        portfolio = {
            "total_capital": 100_000_000,
            "cash_pct": 0.4,
            "positions": [
                {"ticker": "005930", "name": "삼성전자", "sector": "반도체",
                 "weight_pct": 15.0, "pnl_pct": 5.0, "hold_days": 10},
            ],
        }
        market = {"kospi": 2700, "kospi_change_pct": 0.5, "regime": "sideways"}

        text = _format_health_input(portfolio, market)
        assert "삼성전자" in text
        assert "반도체" in text
        assert "sideways" in text

    def test_format_drawdown_input(self):
        """낙폭 분석 프롬프트 포매팅"""
        equity_curve = [
            {"date": "2026-02-01", "equity": 100_000_000},
            {"date": "2026-02-15", "equity": 92_000_000},
        ]
        positions = [{"ticker": "005930", "pnl_pct": -5.0, "weight_pct": 20.0}]

        text = _format_drawdown_input(equity_curve, positions)
        assert "92,000,000" in text
        assert "005930" in text


# ═══════════════════════════════════════════════
# 6. Risk Sentinel 프롬프트 포매팅 테스트
# ═══════════════════════════════════════════════

from src.agents.risk_sentinel import (
    DEFAULT_SCENARIOS,
    _format_returns_matrix,
)
from src.agents.risk_sentinel import (
    _format_market_data as risk_format_market_data,
)


class TestRiskSentinelFormatting:
    """Risk Sentinel 프롬프트 포매팅 테스트"""

    def test_format_market_data(self):
        """시장 데이터 포매팅"""
        market_data = {
            "kospi": 2700,
            "kosdaq": 900,
            "vkospi": 22.5,
            "vkospi_change_pct": 5.0,
            "market_breadth": 0.45,
            "foreign_flow": {
                "today": -1500,
                "5d_cumulative": -5000,
                "consecutive_sell_days": 3,
            },
        }
        portfolio = {
            "total_value": 100_000_000,
            "cash_ratio": 30,
            "holdings": [
                {"name": "삼성전자", "ticker": "005930", "weight_pct": 15,
                 "return_pct": 3.5, "sector": "반도체"},
            ],
        }

        text = risk_format_market_data(market_data, portfolio)
        assert "VKOSPI" in text
        assert "삼성전자" in text
        assert "연속 매도 일수" in text

    def test_format_returns_matrix(self):
        """수익률 매트릭스 포매팅"""
        matrix = {
            "005930": [0.5, -0.3, 1.2, -0.1, 0.8],
            "000660": [0.8, -0.5, 1.5, 0.2, -0.4],
        }
        text = _format_returns_matrix(matrix)
        assert "종목 수: 2" in text
        assert "005930" in text

    def test_default_scenarios_count(self):
        """기본 스트레스 시나리오 4개 확인"""
        assert len(DEFAULT_SCENARIOS) == 4
        names = [s["name"] for s in DEFAULT_SCENARIOS]
        assert "KOSPI -5% 급락" in names
        assert "반도체 섹터 -15%" in names


# ═══════════════════════════════════════════════
# 7. Macro Analyst 프롬프트 포매팅 테스트
# ═══════════════════════════════════════════════

from src.agents.macro_analyst import (
    SECTORS,
    _format_breadth_data,
    _format_sector_data,
)


class TestMacroAnalystFormatting:
    """Macro Analyst 프롬프트 포매팅 테스트"""

    def test_sectors_list(self):
        """14개 섹터 목록 확인"""
        assert len(SECTORS) == 14
        assert "반도체" in SECTORS
        assert "에너지" in SECTORS

    def test_format_sector_data(self):
        """섹터 데이터 포매팅"""
        data = {
            "sectors": [
                {"name": "반도체", "change_1d_pct": 1.5, "change_5d_pct": 3.2,
                 "change_20d_pct": 8.0, "volume_ratio": 1.3, "foreign_net": 500},
                {"name": "건설", "change_1d_pct": -0.8, "change_5d_pct": -2.1,
                 "change_20d_pct": -5.0, "volume_ratio": 0.7, "foreign_net": -200},
            ],
            "top_rs": ["반도체", "자동차"],
            "bottom_rs": ["건설"],
        }
        text = _format_sector_data(data)
        assert "반도체" in text
        assert "건설" in text
        assert "상대강도 상위" in text

    def test_format_breadth_data(self):
        """시장 폭 데이터 포매팅"""
        data = {
            "advancing": 500,
            "declining": 400,
            "unchanged": 100,
            "new_highs": 45,
            "new_lows": 12,
            "pct_above_ma20": 0.62,
            "pct_above_ma60": 0.48,
            "pct_above_ma200": 0.55,
        }
        text = _format_breadth_data(data)
        assert "500" in text
        assert "신고가" in text
        assert "20일선 위" in text


# ═══════════════════════════════════════════════
# 8. 에이전트 패키지 import 테스트
# ═══════════════════════════════════════════════


class TestAgentImports:
    """에이전트 패키지에서 올바르게 import 되는지 검증"""

    def test_import_cfo(self):
        """CFOAgent import 가능"""
        from src.agents import CFOAgent
        assert CFOAgent is not None

    def test_import_risk_sentinel(self):
        """RiskSentinelAgent import 가능"""
        from src.agents import RiskSentinelAgent
        assert RiskSentinelAgent is not None

    def test_import_macro_analyst(self):
        """MacroAnalystAgent import 가능"""
        from src.agents import MacroAnalystAgent
        assert MacroAnalystAgent is not None
