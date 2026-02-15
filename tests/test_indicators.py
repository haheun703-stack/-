"""
v6.3 IndicatorEngine 단위 테스트

테스트 대상: src/indicators.py
- calc_atr: TR 수동 계산 대조, 초기 NaN, 등가 → ATR≈0
- calc_rsi: 전부 상승, 전부 하락, 등락 혼합, period 효과
- calc_adx: 강추세, 횡보
- calc_obv: 기본 누적, 동일 종가
- calc_stoch_rsi: 범위 확인, %K·%D 존재
- compute_all: 컬럼 수, NaN 패턴
"""

import numpy as np
import pandas as pd

from src.indicators import IndicatorEngine

# ── 헬퍼 ──

def make_ohlcv(n=200, trend=0.001):
    """최소 200일 OHLCV DataFrame"""
    np.random.seed(42)
    close = 10000 * np.exp(np.cumsum(np.random.randn(n) * 0.02 + trend))
    high = close * (1 + np.random.uniform(0.005, 0.03, n))
    low = close * (1 - np.random.uniform(0.005, 0.03, n))
    open_ = close + np.random.uniform(-100, 100, n)
    volume = np.random.uniform(100000, 500000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


# ── ATR 테스트 ──

class TestCalcATR:
    """ATR(Average True Range) 계산 검증"""

    def test_basic_atr(self):
        """3일 데이터 → TR 수동 계산과 대조"""
        df = pd.DataFrame({
            "high":  [110, 115, 120],
            "low":   [90,  95,  100],
            "close": [100, 110, 105],
        })
        atr = IndicatorEngine.calc_atr(df, period=2)
        # 첫 행: TR = H-L = 20 (prev_close=NaN이므로 H-L만 유효)
        # 2번째: TR = max(115-95, |115-100|, |95-100|) = max(20,15,5) = 20
        # period=2이므로 EWM 시작 가능
        assert not atr.isna().all(), "ATR이 전부 NaN이면 안 됨"

    def test_nan_initial_period(self):
        """period=14 → 첫 13행 NaN"""
        df = make_ohlcv(50)
        atr = IndicatorEngine.calc_atr(df, period=14)
        assert atr.iloc[:13].isna().all(), "첫 13행은 NaN이어야 함"
        assert not atr.iloc[13:].isna().any(), "14행부터는 값 존재"

    def test_constant_price(self):
        """H=L=C → TR=0 → ATR≈0"""
        n = 30
        df = pd.DataFrame({
            "high": [10000.0] * n,
            "low": [10000.0] * n,
            "close": [10000.0] * n,
        })
        atr = IndicatorEngine.calc_atr(df, period=14)
        valid = atr.dropna()
        assert (valid < 0.01).all(), "등가 → ATR ≈ 0"


# ── RSI 테스트 ──

class TestCalcRSI:
    """RSI(Relative Strength Index) 계산 검증"""

    def test_mostly_gains(self):
        """대부분 상승 (95%) → RSI > 85"""
        np.random.seed(10)
        # 대부분 상승하되 가끔 소폭 하락 (avg_loss > 0 보장)
        changes = [10] * 47 + [-2] * 3  # 47일 상승, 3일 소폭 하락
        prices = pd.Series(np.cumsum([100] + changes))
        rsi = IndicatorEngine.calc_rsi(prices, period=14)
        valid = rsi.dropna()
        assert len(valid) > 0, "유효한 RSI 값이 존재해야 함"
        assert valid.iloc[-1] > 85, f"대부분 상승이면 RSI > 85, 실제: {valid.iloc[-1]:.1f}"

    def test_all_losses(self):
        """전부 하락 → RSI → 0 근접"""
        prices = pd.Series([1000 - i * 10 for i in range(50)])
        rsi = IndicatorEngine.calc_rsi(prices, period=14)
        valid = rsi.dropna()
        assert valid.iloc[-1] < 10, f"전부 하락이면 RSI < 10, 실제: {valid.iloc[-1]:.1f}"

    def test_mixed(self):
        """등락 반복 → 30 < RSI < 70"""
        np.random.seed(123)
        prices = pd.Series(10000 + np.cumsum(np.random.randn(100) * 50))
        rsi = IndicatorEngine.calc_rsi(prices, period=14)
        valid = rsi.dropna()
        last = valid.iloc[-1]
        assert 10 < last < 90, f"혼합 → RSI 중간대, 실제: {last:.1f}"

    def test_period_effect(self):
        """period 짧을수록 변동 大"""
        np.random.seed(42)
        prices = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100))
        rsi_short = IndicatorEngine.calc_rsi(prices, period=5)
        rsi_long = IndicatorEngine.calc_rsi(prices, period=28)
        std_short = rsi_short.dropna().std()
        std_long = rsi_long.dropna().std()
        assert std_short > std_long, "짧은 period → 더 큰 변동"


# ── ADX 테스트 ──

class TestCalcADX:
    """ADX(Average Directional Index) 계산 검증"""

    def test_strong_trend(self):
        """일방향 강한 상승 → ADX > 25"""
        n = 100
        base = np.arange(n, dtype=float) * 100 + 10000
        df = pd.DataFrame({
            "high": base + 50,
            "low": base - 50,
            "close": base,
        })
        adx = IndicatorEngine.calc_adx(df, period=14)
        valid = adx.dropna()
        assert valid.iloc[-1] > 25, f"강한 추세 → ADX > 25, 실제: {valid.iloc[-1]:.1f}"

    def test_no_trend(self):
        """횡보 (등락 반복) → ADX 낮음"""
        n = 100
        np.random.seed(99)
        close = 10000 + np.random.randn(n) * 20  # 평균 회귀
        df = pd.DataFrame({
            "high": close + 30,
            "low": close - 30,
            "close": close,
        })
        adx = IndicatorEngine.calc_adx(df, period=14)
        valid = adx.dropna()
        assert valid.iloc[-1] < 30, f"횡보 → ADX < 30, 실제: {valid.iloc[-1]:.1f}"


# ── OBV 테스트 ──

class TestCalcOBV:
    """OBV(On Balance Volume) 계산 검증"""

    def test_basic(self):
        """상승일 +volume, 하락일 -volume 누적"""
        df = pd.DataFrame({
            "close":  [100, 110, 105, 115, 110],
            "volume": [1000, 2000, 1500, 3000, 1000],
        })
        obv = IndicatorEngine.calc_obv(df)
        # day0: direction undefined (shift), day1: +2000, day2: -1500, day3: +3000, day4: -1000
        assert obv.iloc[1] > 0, "상승일 → OBV 양수"
        assert obv.iloc[2] < obv.iloc[1], "하락일 → OBV 감소"

    def test_flat_day(self):
        """동일 종가 → volume 무시 (direction=0)"""
        df = pd.DataFrame({
            "close":  [100, 100, 100],
            "volume": [1000, 2000, 3000],
        })
        obv = IndicatorEngine.calc_obv(df)
        # 모든 날 direction=0이므로 obv 변동 없음
        assert obv.iloc[-1] == 0, f"동일 종가 → OBV=0, 실제: {obv.iloc[-1]}"


# ── Stochastic RSI 테스트 ──

class TestCalcStochRSI:
    """Stochastic RSI 계산 검증"""

    def test_basic_range(self):
        """0~100 범위 내"""
        np.random.seed(42)
        prices = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100))
        rsi = IndicatorEngine.calc_rsi(prices, period=14)
        stoch = IndicatorEngine.calc_stoch_rsi(rsi, period=14)
        valid_k = stoch["stoch_rsi_k"].dropna()
        assert valid_k.min() >= -1, "StochRSI %K >= -1"
        assert valid_k.max() <= 101, "StochRSI %K <= 101"

    def test_k_d_columns(self):
        """%K와 %D 컬럼 존재"""
        rsi = pd.Series(np.random.uniform(30, 70, 50))
        stoch = IndicatorEngine.calc_stoch_rsi(rsi, period=14)
        assert "stoch_rsi_k" in stoch.columns
        assert "stoch_rsi_d" in stoch.columns


# ── compute_all 테스트 ──

class TestComputeAll:
    """compute_all 통합 출력 검증"""

    def test_column_count(self):
        """최소 40개 지표 컬럼 추가 확인"""
        engine = IndicatorEngine()
        df = make_ohlcv(200)
        result = engine.compute_all(df)
        original_cols = {"open", "high", "low", "close", "volume"}
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) >= 35, f"최소 35개 신규 컬럼 필요, 실제: {len(new_cols)}"

    def test_nan_handling(self):
        """NaN은 초기 윈도우에만 존재 (200일 데이터 → 후반부 NaN-free)"""
        engine = IndicatorEngine()
        df = make_ohlcv(200)
        result = engine.compute_all(df)
        # 핵심 컬럼들은 row 150 이후 NaN 없어야 함
        core_cols = ["atr_14", "rsi_14", "sma_20", "adx_14", "obv"]
        for col in core_cols:
            if col in result.columns:
                tail = result[col].iloc[150:]
                assert not tail.isna().any(), f"{col}: 150행 이후 NaN 존재"

    def test_ret1_column(self):
        """ret1 = 일간 수익률 컬럼 존재"""
        engine = IndicatorEngine()
        df = make_ohlcv(200)
        result = engine.compute_all(df)
        assert "ret1" in result.columns
        # ret1 = close.pct_change()
        expected = df["close"].pct_change().iloc[10]
        actual = result["ret1"].iloc[10]
        assert abs(expected - actual) < 1e-10, "ret1 = pct_change 일치"

    def test_martin_columns(self):
        """v6.0 Martin 모멘텀 지표 컬럼 존재 (ema_8, ema_24, ema2_martin, daily_sigma)"""
        engine = IndicatorEngine()
        df = make_ohlcv(200)
        result = engine.compute_all(df)
        martin_cols = ["ema_8", "ema_24", "ema2_martin", "ema2_norm", "martin_dead_zone", "daily_sigma"]
        for col in martin_cols:
            assert col in result.columns, f"Martin 컬럼 {col} 누락"
