"""
Step 2: indicators.py — 기술적 지표 계산 엔진

원본 OHLCV에서 전략에 필요한 모든 기술적 지표를 계산한다.
- ATR(14), RSI(14), Stochastic RSI, ADX(14)
- SMA(20, 60, 120, 200), OBV, 거래량 MA
- 60일 최고가 (Rolling High)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """모든 기술적 지표를 계산하는 엔진"""

    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # 개별 지표 계산 함수들
    # ──────────────────────────────────────────────

    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ATR(Average True Range) 계산
        True Range = MAX(|H-L|, |H-PC|, |L-PC|)
        ATR = EMA(True Range, period)
        """
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, min_periods=period).mean()
        return atr

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSI(Relative Strength Index) 계산"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calc_stoch_rsi(rsi: pd.Series, period: int = 14,
                       smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """
        Stochastic RSI 계산
        StochRSI = (RSI - RSI_Low) / (RSI_High - RSI_Low)
        %K = SMA(StochRSI, smooth_k)
        %D = SMA(%K, smooth_d)
        """
        rsi_low = rsi.rolling(period).min()
        rsi_high = rsi.rolling(period).max()

        stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
        stoch_rsi_k = stoch_rsi.rolling(smooth_k).mean() * 100
        stoch_rsi_d = stoch_rsi_k.rolling(smooth_d).mean()

        return pd.DataFrame({
            "stoch_rsi_k": stoch_rsi_k,
            "stoch_rsi_d": stoch_rsi_d,
        })

    @staticmethod
    def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ADX(Average Directional Index) 계산
        추세의 강도를 측정 (방향 무관). 0~100.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM, -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                            index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                             index=df.index)

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # Smoothed averages
        atr = tr.ewm(span=period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr

        # DX → ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, min_periods=period).mean()

        return adx

    @staticmethod
    def calc_obv(df: pd.DataFrame) -> pd.Series:
        """OBV(On Balance Volume) 계산"""
        close = df["close"]
        volume = df["volume"]

        direction = np.where(close > close.shift(1), 1,
                             np.where(close < close.shift(1), -1, 0))
        obv = (volume * direction).cumsum()
        return pd.Series(obv, index=df.index, name="obv")

    # ──────────────────────────────────────────────
    # 전체 지표 계산
    # ──────────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """단일 종목 DataFrame에 모든 지표를 추가"""
        result = df.copy()

        # 1. ATR(14)
        result["atr_14"] = self.calc_atr(df, 14)

        # 2. RSI(14)
        result["rsi_14"] = self.calc_rsi(df["close"], 14)

        # 3. Stochastic RSI
        stoch = self.calc_stoch_rsi(result["rsi_14"], 14, 3, 3)
        result["stoch_rsi_k"] = stoch["stoch_rsi_k"]
        result["stoch_rsi_d"] = stoch["stoch_rsi_d"]

        # 4. ADX(14)
        result["adx_14"] = self.calc_adx(df, 14)

        # 5. 이동평균선
        for window in [20, 60, 120, 200]:
            result[f"sma_{window}"] = df["close"].rolling(window).mean()

        # 6. OBV
        result["obv"] = self.calc_obv(df)

        # 7. 거래량 이동평균
        result["volume_ma5"] = df["volume"].rolling(5).mean()
        result["volume_ma20"] = df["volume"].rolling(20).mean()

        # 8. 60일 Rolling High (최근 고점)
        result["high_60"] = df["high"].rolling(60).max()
        result["high_120"] = df["high"].rolling(120).max()

        # 9. ATR 기반 조정폭 (ATR 배수)
        result["pullback_atr"] = (result["high_60"] - df["close"]) / result["atr_14"].replace(0, np.nan)

        # 10. 거래대금 이동평균 (Pre-screening용)
        if "trading_value" in df.columns:
            result["trading_value_ma60"] = df["trading_value"].rolling(60).mean()

        # ──────────────────────────────────────────────
        # v2.5 듀얼 트리거 전용 지표
        # ──────────────────────────────────────────────

        # 11. 5MA (시동 트리거용 단기 이평)
        result["sma_5"] = df["close"].rolling(5).mean()

        # 12. 전일 고가 (Trigger-1: 전일 고가 돌파 체크)
        result["prev_high"] = df["high"].shift(1)

        # 13. 거래량 서지 비율 (현재 거래량 / 20MA)
        result["volume_surge_ratio"] = df["volume"] / result["volume_ma20"].replace(0, np.nan)

        # 14. 거래량 수축 비율 (조정 끝 신호)
        result["volume_contraction_ratio"] = result["volume_ma5"] / result["volume_ma20"].replace(0, np.nan)

        # 15. Higher Low 감지 (N일 내 저점 갱신 안 함)
        result["rolling_low_3"] = df["low"].rolling(3).min()
        result["rolling_low_5"] = df["low"].rolling(5).min()
        result["rolling_low_10"] = df["low"].rolling(10).min()
        # 오늘 저가 > 3일 최저가 → Higher Low 형성 중
        result["higher_low_3d"] = (df["low"] > result["rolling_low_3"].shift(1)).astype(int)
        result["higher_low_5d"] = (df["low"] > result["rolling_low_5"].shift(1)).astype(int)

        # 16. 아래꼬리 비율 (캔들 분석)
        body_top = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
        body_bottom = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        result["lower_tail_ratio"] = (body_bottom - df["low"]) / candle_range

        # 17. 양봉 여부 (종가 > 시가)
        result["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # 18. N일 최고가 (돌파 트리거용)
        result["high_10"] = df["high"].rolling(10).max()
        result["high_20"] = df["high"].rolling(20).max()

        # 19. 20MA 위 연속 일수 (Trigger-2 확인용)
        above_sma20 = (df["close"] > result["sma_20"]).astype(int)
        # 연속 일수 계산: 0이 나오면 리셋
        streaks = []
        count = 0
        for v in above_sma20:
            if v == 1:
                count += 1
            else:
                count = 0
            streaks.append(count)
        result["days_above_sma20"] = streaks

        return result

    # ──────────────────────────────────────────────
    # 전종목 일괄 처리
    # ──────────────────────────────────────────────

    def process_all(self) -> int:
        """raw 디렉토리의 모든 parquet을 처리하여 processed에 저장"""
        raw_files = sorted(self.raw_dir.glob("*.parquet"))
        if not raw_files:
            logger.error("data/raw에 parquet 파일이 없습니다")
            return 0

        processed_count = 0
        for fpath in tqdm(raw_files, desc="📈 지표 계산"):
            ticker = fpath.stem
            try:
                df = pd.read_parquet(fpath)
                if len(df) < 200:  # 200일 미만 데이터는 지표 계산 불가
                    logger.debug(f"{ticker}: 데이터 부족 ({len(df)}일), 건너뜀")
                    continue

                result = self.compute_all(df)
                save_path = self.processed_dir / f"{ticker}.parquet"
                result.to_parquet(save_path)
                processed_count += 1

            except Exception as e:
                logger.error(f"{ticker} 지표 계산 실패: {e}")

        logger.info(f"✅ 지표 계산 완료: {processed_count}종목")
        return processed_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = IndicatorEngine()
    engine.process_all()
