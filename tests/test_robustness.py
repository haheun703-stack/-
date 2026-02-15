"""
v6.2 견고성 테스트 — edge case + 방어 코드 검증

테스트 항목:
  1. Martin Momentum: idx 범위 초과, 빈 DataFrame, 컬럼 누락, NaN
  2. Extreme Volatility: idx 초과, 컬럼 누락, 전부 NaN, 짧은 df
  3. RTTP: source=None, 빈 뉴스, 정상 호출
  4. Consensus: tau=0, 빈 투표, 단일 투표
  5. ConfigValidator: 정상/비정상/교차검증/비활성화/섹션누락
  6. Backtest 견고성: risk_norm 예외, trailing_mult 하한
"""

import numpy as np
import pandas as pd
import pytest

from src.martin_momentum import MartinMomentumEngine
from src.extreme_volatility import ExtremeVolatilityDetector
from src.use_cases.rttp_news_processor import RttpNewsProcessor
from src.use_cases.consensus_engine import ConsensusVerifier
from src.config_validator import ConfigValidator
from src.entities.consensus_models import ConsensusResult, LayerVote
from src.entities.news_models import (
    NewsGateResult, NewsGrade, EventDrivenAction, NewsItem,
)
from src.entities.rttp_models import RttpEnhancement


# ── 공통 헬퍼 ──

def make_market_df(n=100):
    """시장 데이터 DataFrame"""
    np.random.seed(42)
    close = 10000 + np.cumsum(np.random.randn(n) * 50)
    close = close.clip(min=5000)
    volume = np.random.uniform(100000, 300000, n)

    df = pd.DataFrame({
        "open": close + np.random.uniform(-50, 50, n),
        "high": close + np.random.uniform(50, 150, n),
        "low": close - np.random.uniform(50, 150, n),
        "close": close,
        "volume": volume,
        "ret1": pd.Series(close).pct_change(),
        "atr_14": np.random.uniform(100, 300, n),
        "rsi_14": np.random.uniform(30, 70, n),
        "sma_20": pd.Series(close).rolling(20).mean(),
        "volume_ma20": pd.Series(volume).rolling(20).mean(),
        "ema_8": pd.Series(close).ewm(span=8).mean(),
        "ema_24": pd.Series(close).ewm(span=24).mean(),
        "daily_sigma": np.random.uniform(0.01, 0.04, n),
    })
    return df


def make_news_gate(source="DART 전자공시", impact=8) -> NewsGateResult:
    """테스트용 NewsGateResult"""
    items = [
        NewsItem(
            title="테스트 뉴스",
            summary="테스트 뉴스 요약",
            source=source,
            date="2026-02-14",
            impact_score=impact,
            sentiment="positive",
            is_confirmed=True,
            has_specific_amount=True,
            has_definitive_language=True,
            cross_verified=True,
        ),
    ]
    return NewsGateResult(
        grade=NewsGrade.A,
        action=EventDrivenAction.ENTER,
        ticker="005930",
        news_items=items,
    )


# ── Martin Momentum Edge Cases ──

class TestMartinEdgeCases:
    """Martin Momentum 엔진 견고성"""

    def test_idx_out_of_range(self):
        """idx=999 with 100행 → 기본값"""
        engine = MartinMomentumEngine()
        df = make_market_df(100)
        result = engine.evaluate(df, 999)
        assert result.confidence == 0.0
        assert result.signal_type == "neutral"

    def test_negative_idx(self):
        """음수 idx → 기본값"""
        engine = MartinMomentumEngine()
        df = make_market_df(100)
        result = engine.evaluate(df, -1)
        assert result.confidence == 0.0

    def test_empty_df(self):
        """빈 DataFrame → 기본값"""
        engine = MartinMomentumEngine()
        df = pd.DataFrame()
        result = engine.evaluate(df, 0)
        assert result.confidence == 0.0

    def test_missing_ema_columns(self):
        """ema_8 컬럼 없음 → calc_ema2 = 0.0"""
        engine = MartinMomentumEngine()
        df = pd.DataFrame({"close": [100, 101, 102], "daily_sigma": [0.02, 0.02, 0.02]})
        ema2 = engine.calc_ema2(df, 1)
        assert ema2 == 0.0

    def test_missing_close_column(self):
        """close 컬럼 없음 → 기본값"""
        engine = MartinMomentumEngine()
        df = pd.DataFrame({"open": [100, 101, 102]})
        result = engine.evaluate(df, 1)
        assert result.confidence == 0.0

    def test_nan_close(self):
        """close=NaN → 기본값"""
        engine = MartinMomentumEngine()
        df = make_market_df(100)
        df.loc[50, "close"] = np.nan
        result = engine.evaluate(df, 50)
        assert result.confidence == 0.0

    def test_missing_daily_sigma(self):
        """daily_sigma 컬럼 없음 → vol_weight=1.0"""
        engine = MartinMomentumEngine()
        df = pd.DataFrame({"close": [100, 101, 102]})
        weight = engine.calc_vol_weight(df, 1)
        assert weight == 1.0

    def test_calc_ema2_idx_out_of_range(self):
        """calc_ema2 idx 범위 초과 → 0.0"""
        engine = MartinMomentumEngine()
        df = make_market_df(10)
        assert engine.calc_ema2(df, 999) == 0.0
        assert engine.calc_ema2(df, -5) == 0.0

    def test_calc_vol_weight_idx_out_of_range(self):
        """calc_vol_weight idx 범위 초과 → 1.0"""
        engine = MartinMomentumEngine()
        df = make_market_df(10)
        assert engine.calc_vol_weight(df, 999) == 1.0


# ── Extreme Volatility Edge Cases ──

class TestExtremeVolEdgeCases:
    """극한 변동성 탐지기 견고성"""

    def test_idx_exceeds_length(self):
        """idx >= len(df) → 기본값"""
        detector = ExtremeVolatilityDetector()
        df = make_market_df(100)
        result = detector.detect(df, 200)
        assert result.is_extreme is False

    def test_negative_idx(self):
        """음수 idx → 기본값"""
        detector = ExtremeVolatilityDetector()
        df = make_market_df(100)
        result = detector.detect(df, -1)
        assert result.is_extreme is False

    def test_capitulation_idx_exceeds(self):
        """capitulation: idx >= len(df) → 기본값"""
        detector = ExtremeVolatilityDetector()
        df = make_market_df(100)
        cap = detector.detect_capitulation(df, 200)
        assert cap.score == 0.0

    def test_direction_idx_exceeds(self):
        """direction: idx >= len(df) → ambiguous"""
        detector = ExtremeVolatilityDetector()
        df = make_market_df(100)
        direction = detector.determine_direction(df, 200)
        assert direction == "ambiguous"

    def test_capitulation_short_df(self):
        """10행만 → 정상 동작 (idx < min_down_days + 1)"""
        detector = ExtremeVolatilityDetector(min_down_days=3)
        df = make_market_df(10)
        cap = detector.detect_capitulation(df, 2)
        assert cap.score == 0.0  # idx < 4

    def test_normal_market_no_extreme(self):
        """정상 시장 → 극한 변동성 미감지"""
        detector = ExtremeVolatilityDetector()
        df = make_market_df(100)
        result = detector.detect(df, 99)
        assert result.is_extreme is False


# ── RTTP Edge Cases ──

class TestRttpEdgeCases:
    """RTTP 뉴스 프로세서 견고성"""

    def test_news_item_no_source(self):
        """source=None → COMMUNITY"""
        proc = RttpNewsProcessor()
        tier = proc._classify_source(None)
        assert tier == "COMMUNITY"

    def test_news_item_empty_source(self):
        """source="" → COMMUNITY"""
        proc = RttpNewsProcessor()
        tier = proc._classify_source("")
        assert tier == "COMMUNITY"

    def test_empty_gate_result(self):
        """빈 뉴스 리스트 → score=0"""
        proc = RttpNewsProcessor()
        gate = NewsGateResult()
        df = make_market_df(100)
        enhancement = proc.enhance_gate_result(gate, df, 50)
        assert enhancement.source_weighted_score == 0.0
        assert enhancement.engagement_depth == 0.0

    def test_enhancement_returns_valid(self):
        """정상 호출 → RttpEnhancement"""
        proc = RttpNewsProcessor()
        gate = make_news_gate(source="DART 전자공시", impact=9)
        df = make_market_df(100)
        enhancement = proc.enhance_gate_result(gate, df, 50)
        assert isinstance(enhancement, RttpEnhancement)
        assert enhancement.source_weighted_score > 0.0

    def test_idx_out_of_range(self):
        """idx 범위 초과 → engagement=0.0"""
        proc = RttpNewsProcessor()
        gate = make_news_gate()
        df = make_market_df(100)
        enhancement = proc.enhance_gate_result(gate, df, 999)
        assert enhancement.engagement_depth == 0.0


# ── Consensus Edge Cases ──

class TestConsensusEdgeCases:
    """Sci-CoE 합의 판정기 견고성"""

    def test_tau_zero(self):
        """tau=0 → consistency=0.0 (ZeroDivision 방지)"""
        verifier = ConsensusVerifier(tau=0.0)
        votes = [LayerVote("L0_grade", True, 0.8)]
        consistency = verifier.calc_consistency(votes)
        assert consistency == 0.0

    def test_tau_negative(self):
        """tau<0 → consistency=0.0"""
        verifier = ConsensusVerifier(tau=-1.0)
        votes = [LayerVote("L0_grade", True, 0.8)]
        consistency = verifier.calc_consistency(votes)
        assert consistency == 0.0

    def test_empty_votes(self):
        """빈 리스트 → 기본 ConsensusResult"""
        verifier = ConsensusVerifier()
        result = verifier.verify([])
        assert result.geometric_reward == 0.0
        assert result.consensus_grade == "reject"

    def test_single_vote(self):
        """1개 투표 → 정상 동작"""
        verifier = ConsensusVerifier(tau=0.8)
        votes = [LayerVote("L0_grade", True, 0.9)]
        result = verifier.verify(votes)
        assert result.consistency > 0.0
        assert result.passed_voters == 1

    def test_all_failed_votes(self):
        """모든 투표 실패 → geometric_reward=0"""
        verifier = ConsensusVerifier()
        votes = [
            LayerVote("L0_grade", False, 0.1),
            LayerVote("L1_regime", False, 0.2),
        ]
        result = verifier.verify(votes)
        assert result.reliability == 0.0


# ── ConfigValidator Edge Cases ──

class TestConfigValidator:
    """Config 범위 검증기 견고성"""

    def test_valid_config(self):
        """정상 config → 경고 0건"""
        config = {
            "martin_momentum": {
                "enabled": True,
                "n_fast": 8,
                "n_slow": 24,
                "epsilon": 0.6,
                "sigmoid_k": 5.0,
                "min_confidence": 0.3,
            },
            "extreme_volatility": {
                "enabled": True,
                "atr_ratio_threshold": 3.0,
                "vol_ratio_threshold": 5.0,
                "daily_range_threshold": 10.0,
            },
        }
        warnings = ConfigValidator.validate(config)
        assert len(warnings) == 0

    def test_negative_epsilon(self):
        """epsilon=-1 → 경고 1건"""
        config = {
            "martin_momentum": {
                "enabled": True,
                "epsilon": -1.0,
            },
        }
        warnings = ConfigValidator.validate(config)
        assert len(warnings) == 1
        assert "epsilon" in warnings[0]

    def test_max_less_than_min(self):
        """max_scale < min_scale → 경고 1건"""
        config = {
            "wavelsformer": {
                "risk_normalization": {
                    "enabled": True,
                    "max_scale": 0.5,
                    "min_scale": 0.8,
                },
            },
        }
        warnings = ConfigValidator.validate(config)
        # max_scale=0.5 < min_scale=0.8 → 교차검증 경고
        cross_warnings = [w for w in warnings if "max_scale <= min_scale" in w]
        assert len(cross_warnings) == 1

    def test_disabled_section_skip(self):
        """enabled=false → 검증 스킵"""
        config = {
            "martin_momentum": {
                "enabled": False,
                "epsilon": -999,  # 비정상이지만 비활성화이므로 무시
            },
        }
        warnings = ConfigValidator.validate(config)
        assert len(warnings) == 0

    def test_missing_section(self):
        """섹션 자체 없음 → 경고 0건"""
        config = {}
        warnings = ConfigValidator.validate(config)
        assert len(warnings) == 0

    def test_type_error(self):
        """파라미터 타입 오류 → 경고"""
        config = {
            "martin_momentum": {
                "enabled": True,
                "epsilon": "not_a_number",
            },
        }
        warnings = ConfigValidator.validate(config)
        assert len(warnings) == 1
        assert "타입 오류" in warnings[0]

    def test_out_of_range_high(self):
        """범위 상한 초과 → 경고"""
        config = {
            "extreme_volatility": {
                "enabled": True,
                "atr_ratio_threshold": 999.0,  # max=10.0
            },
        }
        warnings = ConfigValidator.validate(config)
        assert len(warnings) == 1
        assert "범위" in warnings[0]
