"""순수 Python 기술적 분석 어댑터 - OHLCV에서 기술지표를 계산"""

from __future__ import annotations

from src.entities.models import ChartData, OHLCV, TechnicalIndicators
from src.use_cases.ports import TechnicalAnalysisPort


class TechnicalAnalysisAdapter(TechnicalAnalysisPort):
    """순수 Python으로 기술적 지표를 계산하는 어댑터"""

    def analyze(self, chart_data: ChartData) -> TechnicalIndicators:
        candles = chart_data.candles
        if not candles:
            return TechnicalIndicators()

        closes = [c.close for c in candles]

        ma5 = _sma(closes, 5)
        ma20 = _sma(closes, 20)
        ma60 = _sma(closes, 60)
        ma120 = _sma(closes, 120)

        rsi = _rsi(closes, 14)
        macd_val, macd_sig, macd_hist = _macd(closes)

        bollinger_upper, bollinger_lower = None, None
        if ma20 and len(closes) >= 20:
            std = (sum((c - ma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
            bollinger_upper = ma20 + 2 * std
            bollinger_lower = ma20 - 2 * std

        stoch_k, stoch_d = _stochastic(candles, 14, 3)

        return TechnicalIndicators(
            rsi=rsi,
            macd=macd_val,
            macd_signal=macd_sig,
            macd_histogram=macd_hist,
            bollinger_upper=bollinger_upper,
            bollinger_middle=ma20,
            bollinger_lower=bollinger_lower,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            ma5=ma5,
            ma20=ma20,
            ma60=ma60,
            ma120=ma120,
        )


# ─── 지표 계산 함수 ────────────────────────────────────────────

def _sma(data: list[float], period: int) -> float | None:
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


def _ema(data: list[float], period: int) -> float | None:
    if len(data) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(data[:period]) / period
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def _rsi(data: list[float], period: int = 14) -> float | None:
    if len(data) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = data[i] - data[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(data: list[float]) -> tuple[float | None, float | None, float | None]:
    ema12 = _ema(data, 12)
    ema26 = _ema(data, 26)
    if ema12 is None or ema26 is None:
        return None, None, None

    macd_line = ema12 - ema26

    # MACD Signal: 최근 MACD 값들의 9일 EMA
    if len(data) >= 35:
        macd_values = []
        for i in range(26, len(data)):
            e12 = _ema(data[: i + 1], 12)
            e26 = _ema(data[: i + 1], 26)
            if e12 is not None and e26 is not None:
                macd_values.append(e12 - e26)
        signal = _ema(macd_values, 9) if len(macd_values) >= 9 else None
        histogram = macd_line - signal if signal is not None else None
        return macd_line, signal, histogram

    return macd_line, None, None


def _stochastic(
    candles: list[OHLCV], k_period: int = 14, d_period: int = 3,
) -> tuple[float | None, float | None]:
    if len(candles) < k_period:
        return None, None

    # %K 계산
    k_values = []
    for i in range(k_period - 1, len(candles)):
        period_candles = candles[i - k_period + 1 : i + 1]
        highest = max(c.high for c in period_candles)
        lowest = min(c.low for c in period_candles)
        if highest == lowest:
            k_values.append(50.0)
        else:
            k_values.append((candles[i].close - lowest) / (highest - lowest) * 100)

    stoch_k = k_values[-1] if k_values else None

    # %D 계산 (K의 d_period 이동평균)
    stoch_d = None
    if len(k_values) >= d_period:
        stoch_d = sum(k_values[-d_period:]) / d_period

    return stoch_k, stoch_d
