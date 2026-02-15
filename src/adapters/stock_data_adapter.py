"""주식 데이터 어댑터 - 외부 API에서 데이터를 가져와 Entity로 변환"""

from __future__ import annotations

from datetime import date, timedelta

from src.entities.models import OHLCV, ChartData, Market, Stock, TechnicalIndicators
from src.use_cases.ports import StockDataPort


class MockStockDataAdapter(StockDataPort):
    """목 데이터 어댑터 (테스트/개발용) - 실제 API 연동 전 사용"""

    async def fetch(self, ticker: str, period_days: int = 120) -> tuple[Stock, ChartData]:
        stock = Stock(
            ticker=ticker,
            name=self._ticker_to_name(ticker),
            market=Market.KOSPI,
            sector="반도체",
        )

        candles = self._generate_mock_candles(period_days)
        indicators = self._calculate_indicators(candles)

        chart_data = ChartData(candles=candles, indicators=indicators)
        return stock, chart_data

    @staticmethod
    def _ticker_to_name(ticker: str) -> str:
        names = {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "035420": "NAVER",
            "035720": "카카오",
            "051910": "LG화학",
        }
        return names.get(ticker, f"종목_{ticker}")

    @staticmethod
    def _generate_mock_candles(period_days: int) -> list[OHLCV]:
        """목 캔들 데이터 생성"""
        import random
        random.seed(42)

        candles = []
        base_price = 70000.0
        today = date.today()

        for i in range(period_days):
            d = today - timedelta(days=period_days - i)
            change = random.uniform(-0.03, 0.03)
            open_price = base_price * (1 + change * 0.5)
            close_price = base_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.015))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.015))
            volume = int(random.uniform(5_000_000, 20_000_000))

            candles.append(OHLCV(
                date=d,
                open=round(open_price, 0),
                high=round(high_price, 0),
                low=round(low_price, 0),
                close=round(close_price, 0),
                volume=volume,
            ))
            base_price = close_price

        return candles

    @staticmethod
    def _calculate_indicators(candles: list[OHLCV]) -> TechnicalIndicators:
        """기본 기술적 지표 계산"""
        closes = [c.close for c in candles]

        def _sma(data: list[float], period: int) -> float | None:
            if len(data) < period:
                return None
            return sum(data[-period:]) / period

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

        ma5 = _sma(closes, 5)
        ma20 = _sma(closes, 20)
        ma60 = _sma(closes, 60)
        ma120 = _sma(closes, 120)

        # 볼린저 밴드 (20일)
        bollinger_middle = ma20
        bollinger_upper = None
        bollinger_lower = None
        if ma20 and len(closes) >= 20:
            std = (sum((c - ma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
            bollinger_upper = ma20 + 2 * std
            bollinger_lower = ma20 - 2 * std

        return TechnicalIndicators(
            rsi=_rsi(closes),
            ma5=ma5,
            ma20=ma20,
            ma60=ma60,
            ma120=ma120,
            bollinger_upper=bollinger_upper,
            bollinger_middle=bollinger_middle,
            bollinger_lower=bollinger_lower,
        )
