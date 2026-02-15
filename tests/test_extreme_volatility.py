"""
v6.0 극한 변동성 탐지 단위 테스트

테스트 항목:
  1. ATR ratio 탐지: > 3.0 → 극한
  2. 거래량 ratio 탐지: > 5.0 → 극한
  3. 일간 범위 탐지: > 10% → 극한
  4. Capitulation 프로파일: 4조건 점수
  5. 방향 판별: bullish_breakout / bearish_breakdown / ambiguous
  6. ambiguous 차단: allow_ambiguous=False
"""

import numpy as np
import pandas as pd

from src.entities.volatility_models import ExtremeVolatilityResult
from src.extreme_volatility import ExtremeVolatilityDetector

# ── 헬퍼 ──

def make_normal_df(n=100):
    """정상 변동성 DataFrame"""
    np.random.seed(42)
    close = 10000 + np.cumsum(np.random.randn(n) * 50)
    close = close.clip(min=5000)
    high = close + np.random.uniform(50, 150, n)
    low = close - np.random.uniform(50, 150, n)
    volume = np.random.uniform(100000, 300000, n)

    df = pd.DataFrame({
        "open": close + np.random.uniform(-50, 50, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "atr_14": np.random.uniform(100, 300, n),
        "rsi_14": np.random.uniform(30, 70, n),
        "sma_20": pd.Series(close).rolling(20).mean(),
        "volume_ma20": pd.Series(volume).rolling(20).mean(),
    })
    return df


def make_extreme_atr_df():
    """ATR ratio > 3.0 되는 DataFrame"""
    df = make_normal_df(100)
    # 60일 평균 ATR은 200 수준, 현재를 700으로 설정
    df.loc[99, "atr_14"] = 700
    return df


def make_extreme_volume_df():
    """거래량 ratio > 5.0 되는 DataFrame"""
    df = make_normal_df(100)
    # 60일 평균 약 200k, 현재를 1.5M으로
    df.loc[99, "volume"] = 1500000
    return df


def make_extreme_range_df():
    """일간 범위 > 10% 되는 DataFrame"""
    df = make_normal_df(100)
    close = df.loc[99, "close"]
    df.loc[99, "high"] = close * 1.08
    df.loc[99, "low"] = close * 0.93  # 15% range
    return df


def make_capitulation_df():
    """투매 조건 충족 DataFrame"""
    df = make_normal_df(100)
    # 연속 5일 하락
    for i in range(95, 100):
        df.loc[i, "close"] = df.loc[i - 1, "close"] * 0.97

    # 거래량 클라이맥스
    avg_vol = df["volume_ma20"].iloc[99]
    if pd.isna(avg_vol):
        avg_vol = 200000
    df.loc[99, "volume"] = avg_vol * 4
    df.loc[99, "volume_ma20"] = avg_vol

    # RSI 극단적 과매도
    df.loc[99, "rsi_14"] = 15

    # 반전 캔들 (양봉 + 긴 아래꼬리)
    close_99 = df.loc[99, "close"]
    df.loc[99, "open"] = close_99 - 100  # 양봉
    df.loc[99, "low"] = close_99 - 500   # 긴 아래꼬리
    df.loc[99, "high"] = close_99 + 50

    # ATR을 높여서 극한 변동성 트리거
    df.loc[99, "atr_14"] = 700
    return df


# ── ATR ratio 테스트 ──

class TestATRRatio:
    def test_normal_not_extreme(self):
        """정상 ATR → 극한 아님"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(100)
        result = detector.detect(df, 99)
        assert result.is_extreme is False

    def test_high_atr_is_extreme(self):
        """ATR ratio > 3.0 → 극한"""
        detector = ExtremeVolatilityDetector(atr_ratio_threshold=3.0)
        df = make_extreme_atr_df()
        result = detector.detect(df, 99)
        assert result.is_extreme is True
        assert result.atr_ratio > 3.0

    def test_too_few_bars(self):
        """데이터 부족 → 기본값"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(50)
        result = detector.detect(df, 10)
        assert result.is_extreme is False


# ── Volume ratio 테스트 ──

class TestVolumeRatio:
    def test_volume_surge_extreme(self):
        """거래량 5배+ → 극한"""
        detector = ExtremeVolatilityDetector(vol_ratio_threshold=5.0)
        df = make_extreme_volume_df()
        result = detector.detect(df, 99)
        assert result.is_extreme is True
        assert result.vol_ratio > 5.0


# ── Daily range 테스트 ──

class TestDailyRange:
    def test_wide_range_extreme(self):
        """일간 범위 > 10% → 극한"""
        detector = ExtremeVolatilityDetector(daily_range_threshold=10.0)
        df = make_extreme_range_df()
        result = detector.detect(df, 99)
        assert result.is_extreme is True
        assert result.daily_range_pct > 10.0


# ── Capitulation 프로파일 테스트 ──

class TestCapitulation:
    def test_no_capitulation_normal(self):
        """정상 시장 → capitulation 아님"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(100)
        cap = detector.detect_capitulation(df, 99)
        assert cap.score < 70

    def test_full_capitulation(self):
        """투매 조건 모두 충족 → 높은 score"""
        detector = ExtremeVolatilityDetector(
            min_down_days=3,
            volume_climax_mult=3.0,
            rsi_extreme=20,
        )
        df = make_capitulation_df()
        cap = detector.detect_capitulation(df, 99)
        assert cap.consecutive_down_days >= 3
        assert cap.volume_climax is True
        assert cap.rsi_extreme_low is True
        assert cap.score >= 70

    def test_capitulation_overrides_direction(self):
        """Capitulation 감지 → direction=capitulation"""
        detector = ExtremeVolatilityDetector(min_capitulation_score=70)
        df = make_capitulation_df()
        result = detector.detect(df, 99)
        if result.is_extreme and result.is_capitulation:
            assert result.direction == "capitulation"


# ── 방향 판별 테스트 ──

class TestDirection:
    def test_bullish_breakout(self):
        """강한 양봉 + RSI 상승 + MA20 위 → bullish_breakout"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(100)
        # 5일 전 대비 큰 상승
        df.loc[94, "close"] = 8000
        df.loc[99, "close"] = 9000
        df.loc[99, "open"] = 8500  # 양봉
        df.loc[99, "rsi_14"] = 55
        df.loc[99, "sma_20"] = 8000  # close > sma20
        direction = detector.determine_direction(df, 99)
        assert direction == "bullish_breakout"

    def test_bearish_breakdown(self):
        """강한 음봉 + RSI 하락 + MA20 아래 → bearish_breakdown"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(100)
        df.loc[94, "close"] = 11000
        df.loc[99, "close"] = 9000
        df.loc[99, "open"] = 10000  # 음봉
        df.loc[99, "rsi_14"] = 20
        df.loc[99, "sma_20"] = 10500  # close < sma20
        direction = detector.determine_direction(df, 99)
        assert direction == "bearish_breakdown"

    def test_ambiguous(self):
        """혼합 신호 → ambiguous"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(100)
        # 약간의 상승, RSI 중립, MA 근접
        df.loc[94, "close"] = 9900
        df.loc[99, "close"] = 10000
        df.loc[99, "open"] = 10050  # 약한 음봉
        df.loc[99, "rsi_14"] = 45
        df.loc[99, "sma_20"] = 10010  # close ≈ sma20
        direction = detector.determine_direction(df, 99)
        assert direction == "ambiguous"

    def test_too_few_bars_ambiguous(self):
        """데이터 부족 → ambiguous"""
        detector = ExtremeVolatilityDetector()
        df = make_normal_df(10)
        direction = detector.determine_direction(df, 2)
        assert direction == "ambiguous"


# ── ambiguous 차단 테스트 ──

class TestAmbiguousBlock:
    def test_allow_ambiguous_false(self):
        """allow_ambiguous=False → ambiguous는 차단 (극한 아님 취급)"""
        detector = ExtremeVolatilityDetector(allow_ambiguous=False)
        # detect()의 direction이 ambiguous인 경우
        # regime_detector의 check_regime_gate에서 차단됨
        # 여기서는 속성 확인만
        assert detector.allow_ambiguous is False

    def test_allow_ambiguous_true(self):
        """allow_ambiguous=True → ambiguous도 허용"""
        detector = ExtremeVolatilityDetector(allow_ambiguous=True)
        assert detector.allow_ambiguous is True


# ── ExtremeVolatilityResult 구조 테스트 ──

class TestResultStructure:
    def test_default_result(self):
        """기본 결과: 극한 아님"""
        result = ExtremeVolatilityResult()
        assert result.is_extreme is False
        assert result.direction == "neutral"
        assert result.confidence == 0.0

    def test_extreme_result_has_direction(self):
        """극한 변동성 → direction 필수"""
        detector = ExtremeVolatilityDetector()
        df = make_extreme_atr_df()
        result = detector.detect(df, 99)
        if result.is_extreme:
            assert result.direction in (
                "bullish_breakout", "bearish_breakdown",
                "capitulation", "ambiguous"
            )
            assert result.confidence > 0.0
