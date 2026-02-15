"""CSV 주식 데이터 어댑터 - stock_data_daily/ 폴더의 실제 CSV 데이터를 Entity로 변환"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from src.entities.models import OHLCV, ChartData, Market, Stock, TechnicalIndicators
from src.use_cases.ports import StockDataPort

# CSV 컬럼 매핑
# Date,Open,High,Low,Close,Volume,MA5,MA20,MA60,MA120,RSI,MACD,MACD_Signal,
# Upper_Band,Lower_Band,ATR,Stoch_K,Stoch_D,OBV,Next_Close,Target,MarketCap,
# EMA1,EMA2,EMA3,TRIX,TRIX_Signal,Plus_DM,Minus_DM,Plus_DM_14,Minus_DM_14,
# Plus_DI,Minus_DI,DX,ADX,Foreign_Net,Inst_Net

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "stock_data_daily"


def _safe_float(value: str) -> float | None:
    """빈 문자열이나 파싱 불가한 값은 None으로 반환"""
    if not value or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: str) -> int:
    f = _safe_float(value)
    return int(f) if f is not None else 0


class CsvStockDataAdapter(StockDataPort):
    """stock_data_daily/ 폴더의 CSV 파일에서 실제 주식 데이터를 읽는 어댑터"""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    async def fetch(self, ticker: str, period_days: int = 120) -> tuple[Stock, ChartData]:
        csv_path = self._find_csv(ticker)
        if not csv_path:
            raise FileNotFoundError(f"종목코드 {ticker}에 해당하는 CSV 파일을 찾을 수 없습니다.")

        # 파일명에서 종목명 추출 (예: "삼성전자_005930.csv" → "삼성전자")
        stock_name = csv_path.stem.rsplit("_", 1)[0]

        rows = self._read_csv(csv_path)

        # 최근 period_days 일만 사용
        recent_rows = rows[-period_days:] if len(rows) > period_days else rows

        candles = [
            OHLCV(
                date=date.fromisoformat(r["Date"]),
                open=float(r["Open"]),
                high=float(r["High"]),
                low=float(r["Low"]),
                close=float(r["Close"]),
                volume=_safe_int(r["Volume"]),
            )
            for r in recent_rows
            if r["Open"] and r["Close"]
        ]

        # 최근 행의 기술적 지표 사용
        last = recent_rows[-1] if recent_rows else {}
        indicators = TechnicalIndicators(
            rsi=_safe_float(last.get("RSI", "")),
            macd=_safe_float(last.get("MACD", "")),
            macd_signal=_safe_float(last.get("MACD_Signal", "")),
            bollinger_upper=_safe_float(last.get("Upper_Band", "")),
            bollinger_middle=_safe_float(last.get("MA20", "")),
            bollinger_lower=_safe_float(last.get("Lower_Band", "")),
            stochastic_k=_safe_float(last.get("Stoch_K", "")),
            stochastic_d=_safe_float(last.get("Stoch_D", "")),
            ma5=_safe_float(last.get("MA5", "")),
            ma20=_safe_float(last.get("MA20", "")),
            ma60=_safe_float(last.get("MA60", "")),
            ma120=_safe_float(last.get("MA120", "")),
        )

        stock = Stock(
            ticker=ticker,
            name=stock_name,
            market=Market.KOSPI,  # CSV에 시장 정보가 없으므로 기본값
            sector="",            # CSV에 업종 정보가 없으므로 비워둠
        )

        return stock, ChartData(candles=candles, indicators=indicators)

    def _find_csv(self, ticker: str) -> Path | None:
        """종목코드로 CSV 파일을 찾는다 (예: *_005930.csv)"""
        matches = list(self.data_dir.glob(f"*_{ticker}.csv"))
        return matches[0] if matches else None

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        """CSV 파일을 읽어 딕셔너리 리스트로 반환"""
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def list_available_tickers(self) -> list[tuple[str, str]]:
        """사용 가능한 종목 목록을 반환 [(종목명, 종목코드), ...]"""
        results = []
        for csv_file in sorted(self.data_dir.glob("*.csv")):
            parts = csv_file.stem.rsplit("_", 1)
            if len(parts) == 2:
                name, ticker = parts
                if ticker.isdigit():
                    results.append((name, ticker))
        return results
