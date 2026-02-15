"""한국투자증권(KIS) API 어댑터 - mojito2를 통해 실시간 주식 데이터를 Entity로 변환"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import mojito

from src.entities.models import OHLCV, ChartData, InvestorFlow, Market, Stock, TechnicalIndicators
from src.use_cases.ports import StockDataPort


class KisStockDataAdapter(StockDataPort):
    """한국투자증권 API에서 실시간 주식 데이터를 가져오는 어댑터"""

    def __init__(self):
        is_mock = os.getenv("MODEL") != "REAL"
        self.broker = mojito.KoreaInvestment(
            api_key=os.getenv("KIS_APP_KEY"),
            api_secret=os.getenv("KIS_APP_SECRET"),
            acc_no=os.getenv("KIS_ACC_NO"),
            mock=is_mock,
        )

    async def fetch(self, ticker: str, period_days: int = 120) -> tuple[Stock, ChartData]:
        # 현재가 조회 (종목 기본 정보)
        price_data = self.broker.fetch_price(ticker)
        output = price_data.get("output", {})

        # 추가 정보 캐싱 (발행주식수, 외국인 보유)
        self._last_output = output

        stock = Stock(
            ticker=ticker,
            name=output.get("rprs_mrkt_kor_name", ""),
            market=Market.KOSPI if output.get("rprs_mrkt_kor_name") == "KOSPI200" else Market.KOSDAQ,
            sector=output.get("bstp_kor_isnm", ""),
        )

        # 일봉 OHLCV 조회
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=period_days * 2)).strftime("%Y%m%d")

        ohlcv_data = self.broker.fetch_ohlcv(
            ticker, timeframe="D", start_day=start_day, end_day=end_day,
        )
        rows = ohlcv_data.get("output2", [])

        # 날짜 역순 → 정순 정렬 후 최근 period_days개
        candles = []
        for r in reversed(rows):
            dt_str = r.get("stck_bsop_date", "")
            if not dt_str or not r.get("stck_clpr"):
                continue
            candles.append(OHLCV(
                date=datetime.strptime(dt_str, "%Y%m%d").date(),
                open=float(r["stck_oprc"]),
                high=float(r["stck_hgpr"]),
                low=float(r["stck_lwpr"]),
                close=float(r["stck_clpr"]),
                volume=int(r.get("acml_vol", 0)),
            ))

        candles = candles[-period_days:]

        # 기술적 지표 계산
        indicators = self._calculate_indicators(candles)

        return stock, ChartData(candles=candles, indicators=indicators)

    def fetch_stock_info(self, ticker: str) -> dict:
        """fetch_price() 응답에서 발행주식수/외국인보유 정보 추출

        Returns:
            {"total_shares": int, "frgn_hldn_qty": int, "frgn_hldn_rto": float}
        """
        try:
            output = getattr(self, "_last_output", None)
            if output is None:
                price_data = self.broker.fetch_price(ticker)
                output = price_data.get("output", {})

            total_shares = int(output.get("lstn_stcn", 0) or 0)
            frgn_qty = int(output.get("frgn_hldn_qty", 0) or 0)
            frgn_rto = float(output.get("frgn_hldn_rto", 0) or 0)

            return {
                "total_shares": total_shares,
                "frgn_hldn_qty": frgn_qty,
                "frgn_hldn_rto": frgn_rto,
            }
        except Exception:
            return {"total_shares": 0, "frgn_hldn_qty": 0, "frgn_hldn_rto": 0.0}

    @staticmethod
    def _calculate_indicators(candles: list[OHLCV]) -> TechnicalIndicators:
        """캔들 데이터에서 기술적 지표를 계산"""
        if not candles:
            return TechnicalIndicators()

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

        def _ema(data: list[float], period: int) -> float | None:
            if len(data) < period:
                return None
            multiplier = 2 / (period + 1)
            ema = sum(data[:period]) / period
            for price in data[period:]:
                ema = (price - ema) * multiplier + ema
            return ema

        def _macd(data: list[float]) -> tuple[float | None, float | None]:
            ema12 = _ema(data, 12)
            ema26 = _ema(data, 26)
            if ema12 is None or ema26 is None:
                return None, None
            macd_line = ema12 - ema26
            # MACD Signal (9-period EMA of MACD) - 간이 계산
            return macd_line, None

        ma5 = _sma(closes, 5)
        ma20 = _sma(closes, 20)
        ma60 = _sma(closes, 60)
        ma120 = _sma(closes, 120)

        macd_val, macd_sig = _macd(closes)

        # 볼린저 밴드 (20일)
        bollinger_upper = None
        bollinger_lower = None
        if ma20 and len(closes) >= 20:
            std = (sum((c - ma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
            bollinger_upper = ma20 + 2 * std
            bollinger_lower = ma20 - 2 * std

        # 스토캐스틱 (14일)
        stoch_k = None
        stoch_d = None
        if len(candles) >= 14:
            period_highs = [c.high for c in candles[-14:]]
            period_lows = [c.low for c in candles[-14:]]
            highest = max(period_highs)
            lowest = min(period_lows)
            if highest != lowest:
                stoch_k = (closes[-1] - lowest) / (highest - lowest) * 100

        return TechnicalIndicators(
            rsi=_rsi(closes),
            macd=macd_val,
            macd_signal=macd_sig,
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

    async def fetch_investor_flow(self, ticker: str) -> InvestorFlow:
        """한투 API로 투자자별 매매 동향 조회 (inquire-investor, tr_id: FHKST01010900)
        주의: 당일 데이터는 장 종료 후에만 제공됨
        """
        try:
            import requests

            base_url = "https://openapi.koreainvestment.com:9443"
            headers = {
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self.broker.access_token}",
                "appkey": os.getenv("KIS_APP_KEY"),
                "appsecret": os.getenv("KIS_APP_SECRET"),
                "tr_id": "FHKST01010900",
                "custtype": "P",
            }
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
            }

            resp = requests.get(
                f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-investor",
                headers=headers,
                params=params,
                timeout=10,
            )
            data = resp.json()
            items = data.get("output", [])

            if not items:
                return InvestorFlow(ticker=ticker, date=date.today())

            # 최신 데이터 (첫 번째 항목)
            item = items[0]

            foreign_net = int(item.get("frgn_ntby_qty", 0))
            inst_net = int(item.get("orgn_ntby_qty", 0))
            individual_net = int(item.get("prsn_ntby_qty", 0))

            return InvestorFlow(
                ticker=ticker,
                date=date.today(),
                foreign_net=foreign_net,
                inst_net=inst_net,
                individual_net=individual_net,
            )

        except Exception:
            return InvestorFlow(ticker=ticker, date=date.today())
