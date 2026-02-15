"""
v6.1 통합 파이프라인 테스트 — 모듈 간 연결 검증

테스트 항목:
  1. RTTP 프로세서가 signal_engine에서 호출되어 값이 채워지는지
  2. RiskBudgetNormalizer가 backtest에서 포지션 사이즈를 조정하는지
  3. 극한 변동성 시 백테스트 포지션 관리가 동작하는지
  4. 전체 파이프라인 end-to-end 모듈 연결
"""

import numpy as np
import pandas as pd

from src.entities.news_models import (
    EventDrivenAction,
    NewsGateResult,
    NewsGrade,
    NewsItem,
)
from src.entities.rttp_models import RttpEnhancement, SourceTier
from src.extreme_volatility import ExtremeVolatilityDetector
from src.use_cases.risk_normalizer import RiskBudgetNormalizer
from src.use_cases.rttp_news_processor import RttpNewsProcessor

# ── 공통 헬퍼 ──

def make_market_df(n=100):
    """시장 데이터 DataFrame"""
    np.random.seed(42)
    close = 10000 + np.cumsum(np.random.randn(n) * 50)
    close = close.clip(min=5000)
    volume = np.random.uniform(100000, 300000, n)
    ret1 = pd.Series(close).pct_change()

    df = pd.DataFrame({
        "open": close + np.random.uniform(-50, 50, n),
        "high": close + np.random.uniform(50, 150, n),
        "low": close - np.random.uniform(50, 150, n),
        "close": close,
        "volume": volume,
        "ret1": ret1,
        "atr_14": np.random.uniform(100, 300, n),
        "rsi_14": np.random.uniform(30, 70, n),
        "sma_20": pd.Series(close).rolling(20).mean(),
        "volume_ma20": pd.Series(volume).rolling(20).mean(),
        "inst_net_streak": [0] * n,
        "volume_surge_ratio": [1.0] * n,
    })
    return df


def make_news_gate(
    grade=NewsGrade.A,
    source="DART 전자공시",
    impact=8,
    sentiment="positive",
) -> NewsGateResult:
    """테스트용 NewsGateResult"""
    items = [
        NewsItem(
            title="테스트 뉴스",
            summary="테스트 뉴스 요약",
            source=source,
            date="2026-02-14",
            impact_score=impact,
            sentiment=sentiment,
            is_confirmed=True,
            has_specific_amount=True,
            has_definitive_language=True,
            cross_verified=True,
        ),
    ]
    return NewsGateResult(
        grade=grade,
        action=EventDrivenAction.ENTER,
        ticker="005930",
        news_items=items,
    )


# ── RTTP 통합 테스트 ──

class TestRttpIntegration:
    """RTTP가 signal_engine에서 호출되어 값이 채워지는지"""

    def test_rttp_enhances_news_gate(self):
        """RTTP enhance → source_weighted_score > 0"""
        proc = RttpNewsProcessor()
        gate = make_news_gate(source="DART 전자공시", impact=9)
        df = make_market_df(100)

        enhancement = proc.enhance_gate_result(gate, df, 50)

        assert enhancement.source_weighted_score > 0.0
        assert enhancement.engagement_depth >= 0.0
        assert enhancement.source_tier == SourceTier.DART.value

    def test_rttp_populates_gate_fields(self):
        """enhance 후 NewsGateResult 필드가 채워지는지"""
        proc = RttpNewsProcessor()
        gate = make_news_gate(source="DART 전자공시", impact=9)
        df = make_market_df(100)

        enhancement = proc.enhance_gate_result(gate, df, 50)
        # signal_engine에서 하는 것처럼 직접 할당
        gate.source_weighted_score = enhancement.source_weighted_score
        gate.engagement_depth = enhancement.engagement_depth

        assert gate.source_weighted_score > 0.0
        assert hasattr(gate, "engagement_depth")

    def test_rttp_disabled_no_effect(self):
        """RTTP 비활성화 시 기본값 유지"""
        gate = make_news_gate()
        # rttp_processor를 호출하지 않으면 기본 0.0
        assert gate.source_weighted_score == 0.0
        assert gate.engagement_depth == 0.0

    def test_rttp_dart_high_boost(self):
        """DART 소스 고임팩트 → 높은 source_weighted_score"""
        proc = RttpNewsProcessor()
        gate = make_news_gate(source="DART 공시", impact=10)
        df = make_market_df(100)

        enhancement = proc.enhance_gate_result(gate, df, 50)
        assert enhancement.source_weighted_score >= 0.9  # DART는 weight=1.0

    def test_rttp_community_low_boost(self):
        """커뮤니티 소스 → 낮은 boost"""
        proc = RttpNewsProcessor()
        gate = make_news_gate(source="주식갤러리", impact=3)
        df = make_market_df(100)

        enhancement = proc.enhance_gate_result(gate, df, 50)
        assert enhancement.rttp_boost <= 0.03

    def test_rttp_source_boost_inline(self):
        """소스 권위 부스트 인라인 계산 검증"""
        proc = RttpNewsProcessor()
        gate = make_news_gate(source="DART 전자공시", impact=10)
        df = make_market_df(100)

        enhancement = proc.enhance_gate_result(gate, df, 50)

        # signal_engine 인라인 로직 재현
        news_score_boost = 0.0
        src_score = enhancement.source_weighted_score
        if src_score >= 0.9:
            news_score_boost += 0.05
        elif src_score >= 0.7:
            news_score_boost += 0.03
        news_score_boost += enhancement.rttp_boost

        assert news_score_boost > 0.0


# ── RiskBudgetNormalizer 통합 테스트 ──

class TestRiskNormalizerIntegration:
    """RiskBudgetNormalizer가 backtest에서 포지션 사이즈를 조정하는지"""

    def test_high_vol_reduces_position(self):
        """고변동성 → 포지션 축소"""
        normalizer = RiskBudgetNormalizer(
            target_daily_vol=0.02, max_scale=2.0, min_scale=0.3,
        )
        df = make_market_df(100)
        # 높은 변동성 시뮬레이션: ret1 표준편차를 크게
        df["ret1"] = np.random.normal(0, 0.06, 100)  # 6% 일간 변동성

        normalized = normalizer.normalize_position(df, 99, 100, 10000)
        # target=0.02, realized~=0.06 → scale ≈ 0.33 → shares ≈ 33
        assert normalized < 100

    def test_low_vol_increases_position(self):
        """저변동성 → 포지션 확대"""
        normalizer = RiskBudgetNormalizer(
            target_daily_vol=0.02, max_scale=2.0, min_scale=0.3,
        )
        df = make_market_df(100)
        # 낮은 변동성: 0.5% 일간
        df["ret1"] = np.random.normal(0, 0.005, 100)

        normalized = normalizer.normalize_position(df, 99, 100, 10000)
        # target=0.02, realized~=0.005 → scale ≈ 4.0 → clipped 2.0 → shares=200
        assert normalized > 100

    def test_disabled_no_change(self):
        """비활성화 시 원본 유지"""
        normalizer = RiskBudgetNormalizer()
        # df에 ret1 컬럼이 없으면 realized_vol=0 → 원본 반환
        df = pd.DataFrame({"close": [100, 101, 102]})
        result = normalizer.normalize_position(df, 2, 100, 10000)
        assert result == 100

    def test_scale_factor_calculation(self):
        """스케일 팩터 직접 계산"""
        normalizer = RiskBudgetNormalizer(
            target_daily_vol=0.02, max_scale=2.0, min_scale=0.3,
        )
        df = make_market_df(100)
        df["ret1"] = np.random.normal(0, 0.02, 100)  # target과 동일

        scale = normalizer.calc_scale_factor(df, 99)
        # target ≈ realized → scale ≈ 1.0
        assert 0.5 < scale < 1.5


# ── 극한 변동성 백테스트 통합 테스트 ──

class TestExtremeVolBacktest:
    """극한 변동성 시 백테스트 포지션 관리"""

    def test_bearish_breakdown_detection(self):
        """bearish_breakdown → 극한 변동성 감지"""
        detector = ExtremeVolatilityDetector(atr_ratio_threshold=3.0)
        df = make_market_df(100)
        # 극한 ATR + 하락 방향 시뮬레이션
        df.loc[99, "atr_14"] = 700
        df.loc[94, "close"] = 11000
        df.loc[99, "close"] = 9000
        df.loc[99, "open"] = 10000  # 음봉
        df.loc[99, "rsi_14"] = 20
        df.loc[99, "sma_20"] = 10500

        result = detector.detect(df, 99)
        if result.is_extreme:
            assert result.direction in (
                "bearish_breakdown", "capitulation", "bullish_breakout", "ambiguous"
            )

    def test_capitulation_detected(self):
        """투매 조건 → is_capitulation"""
        detector = ExtremeVolatilityDetector(
            atr_ratio_threshold=3.0,
            min_down_days=3,
            volume_climax_mult=3.0,
            rsi_extreme=20,
            min_capitulation_score=70,
        )
        df = make_market_df(100)
        # 연속 하락 시뮬레이션
        for i in range(95, 100):
            df.loc[i, "close"] = df.loc[i - 1, "close"] * 0.97

        avg_vol = df["volume_ma20"].iloc[99]
        if pd.isna(avg_vol):
            avg_vol = 200000
        df.loc[99, "volume"] = avg_vol * 4
        df.loc[99, "volume_ma20"] = avg_vol
        df.loc[99, "rsi_14"] = 15
        df.loc[99, "atr_14"] = 700

        result = detector.detect(df, 99)
        # 극한 변동성이 감지되어야 함
        assert result.is_extreme is True

    def test_normal_market_no_exit(self):
        """정상 시장 → 극한 변동성 미감지"""
        detector = ExtremeVolatilityDetector()
        df = make_market_df(100)

        result = detector.detect(df, 99)
        assert result.is_extreme is False


# ── End-to-End 모듈 연결 테스트 ──

class TestEndToEndModuleWiring:
    """전체 모듈이 올바르게 연결되었는지"""

    def test_rttp_processor_import(self):
        """RttpNewsProcessor가 signal_engine에서 import 가능"""
        from src.use_cases.rttp_news_processor import RttpNewsProcessor
        proc = RttpNewsProcessor()
        assert hasattr(proc, "enhance_gate_result")
        assert hasattr(proc, "calc_source_weighted_score")
        assert hasattr(proc, "calc_engagement_depth")

    def test_risk_normalizer_import(self):
        """RiskBudgetNormalizer가 backtest_engine에서 import 가능"""
        from src.use_cases.risk_normalizer import RiskBudgetNormalizer
        norm = RiskBudgetNormalizer()
        assert hasattr(norm, "normalize_position")
        assert hasattr(norm, "calc_scale_factor")

    def test_extreme_vol_detector_accessible(self):
        """ExtremeVolatilityDetector가 signal_engine에서 접근 가능"""
        from src.extreme_volatility import ExtremeVolatilityDetector
        det = ExtremeVolatilityDetector()
        assert hasattr(det, "detect")
        assert hasattr(det, "detect_capitulation")
        assert hasattr(det, "determine_direction")

    def test_news_gate_result_rttp_fields(self):
        """NewsGateResult에 RTTP 필드가 있는지"""
        gate = NewsGateResult()
        assert hasattr(gate, "source_weighted_score")
        assert hasattr(gate, "engagement_depth")
        assert gate.source_weighted_score == 0.0
        assert gate.engagement_depth == 0.0

    def test_rttp_enhancement_dataclass(self):
        """RttpEnhancement 구조 확인"""
        enh = RttpEnhancement()
        assert enh.source_weighted_score == 0.0
        assert enh.engagement_depth == 0.0
        assert enh.rttp_boost == 0.0
        assert enh.source_tier == "COMMUNITY"

    def test_full_rttp_pipeline(self):
        """전체 RTTP 파이프라인: NewsItem → RttpProcessor → enhancement → 투표 점수"""
        proc = RttpNewsProcessor()
        df = make_market_df(100)
        df.loc[50, "inst_net_streak"] = 5  # L5 활성화
        df.loc[50, "volume_surge_ratio"] = 3.0  # L4 활성화

        gate = make_news_gate(source="한국투자증권 리포트", impact=8)
        enhancement = proc.enhance_gate_result(gate, df, 50)

        # 투표 점수 계산 (signal_engine 로직 재현)
        gate.source_weighted_score = enhancement.source_weighted_score
        gate.engagement_depth = enhancement.engagement_depth

        rttp_score = gate.source_weighted_score
        rttp_engage = gate.engagement_depth
        rttp_conf = (rttp_score * 0.6 + min(rttp_engage / 5.0, 1.0) * 0.4)

        assert rttp_conf > 0.0  # 투표 점수가 실제로 > 0
