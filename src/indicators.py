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

from .ou_estimator import OUEstimator
from .smart_money import calc_smart_money_z, calc_institutional_streak

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

        # ──────────────────────────────────────────────
        # v3.0 퀀트 레이어 지표 (10개 추가)
        # ──────────────────────────────────────────────

        # 20. 일간 수익률 (ret1) — 레짐 감지 HMM 입력
        result["ret1"] = df["close"].pct_change()

        # 21. ATR 비율 (ATR_pct) — 변동성 정규화
        result["ATR_pct"] = result["atr_14"] / df["close"] * 100

        # 22. 거래량 Z-score (vol_z) — 60일 기준 거래량 이상치 탐지
        vol_ma60 = df["volume"].rolling(60).mean()
        vol_std60 = df["volume"].rolling(60).std()
        result["vol_z"] = (df["volume"] - vol_ma60) / vol_std60.replace(0, np.nan)
        result["vol_z"] = result["vol_z"].fillna(0)

        # 23. MA60 기울기 (slope_ma60) — 추세 방향성 (L3 모멘텀)
        sma60_series = result["sma_60"]
        result["slope_ma60"] = sma60_series.pct_change(10) * 100  # 10일 변화율(%)

        # 24-29. OU 프로세스 파라미터 (kappa, mu, sigma, half_life, ou_z, snr)
        try:
            ou = OUEstimator(window=60)
            ou_params = ou.estimate_rolling(df["close"])
            for col in ["kappa", "mu", "sigma", "half_life", "ou_z", "snr"]:
                result[col] = ou_params[col]
        except Exception as e:
            logger.debug(f"OU 추정 실패: {e}")
            for col in ["kappa", "mu", "sigma", "half_life", "ou_z", "snr"]:
                result[col] = np.nan

        # 30. Smart Money Z-score
        result["smart_z"] = calc_smart_money_z(result)

        # ──────────────────────────────────────────────
        # v3.1 추가 지표 (TRIX / 볼린저 / MACD / streak / gap)
        # ──────────────────────────────────────────────

        # 31. TRIX(12,9) — Triple EMA 모멘텀
        ema1 = df["close"].ewm(span=12, min_periods=12).mean()
        ema2 = ema1.ewm(span=12, min_periods=12).mean()
        ema3 = ema2.ewm(span=12, min_periods=12).mean()
        result["trix"] = ema3.pct_change() * 100
        result["trix_signal"] = result["trix"].ewm(span=9, min_periods=9).mean()
        result["trix_golden_cross"] = (
            (result["trix"] > result["trix_signal"]) &
            (result["trix"].shift(1) <= result["trix_signal"].shift(1))
        ).astype(int)

        # 32. 볼린저 밴드 (20일, 2σ)
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        result["bb_upper"] = bb_mid + bb_std * 2
        result["bb_lower"] = bb_mid - bb_std * 2
        bb_width = result["bb_upper"] - result["bb_lower"]
        result["bb_width"] = bb_width / bb_mid.replace(0, np.nan)
        result["bb_position"] = (df["close"] - result["bb_lower"]) / bb_width.replace(0, np.nan)

        # 33. MACD(12,26,9)
        ema_fast = df["close"].ewm(span=12, min_periods=12).mean()
        ema_slow = df["close"].ewm(span=26, min_periods=26).mean()
        result["macd"] = ema_fast - ema_slow
        result["macd_signal"] = result["macd"].ewm(span=9, min_periods=9).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        # 34. 기관/외국인 연속 순매수 일수
        for col_name in ["inst_net", "foreign_net"]:
            if col_name in result.columns:
                result[f"{col_name}_streak"] = calc_institutional_streak(
                    result[col_name].fillna(0)
                )

        # 35. 갭업 비율 (전일 종가 대비 시가)
        result["gap_up_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan) * 100

        # ──────────────────────────────────────────────
        # v4.5 Dynamic RSI (변동성 적응형 과매도 기준)
        # ──────────────────────────────────────────────

        # 36. ATR/Price 비율 (변동성 정규화 기준)
        atr_p = result["atr_14"] / df["close"].replace(0, np.nan)
        atr_p_ma = atr_p.rolling(60, min_periods=20).mean()
        atr_p_norm = atr_p / atr_p_ma.replace(0, np.nan)

        # 37. Dynamic RSI Oversold Threshold
        #     T = clip(base - k * (norm - 1), min, max)
        #     변동성↑ → norm>1 → T↓ (엄격), 변동성↓ → norm<1 → T↑ (관대)
        result["dynamic_rsi_oversold"] = np.clip(
            30 - 10 * (atr_p_norm.fillna(1.0) - 1.0), 20, 40
        )

        # 38. RSI EMA(9) — 반전 확인용
        result["rsi_ema9"] = result["rsi_14"].ewm(span=9, min_periods=9).mean()

        # 39. RSI 상승 전환 (오늘 RSI > 어제 RSI)
        result["rsi_rising"] = (result["rsi_14"] > result["rsi_14"].shift(1)).astype(int)

        # 40. Dynamic RSI 과매도 진입 신호
        #     RSI <= Dynamic Threshold AND RSI 상승 전환 AND RSI > EMA(RSI,9)
        result["dynamic_rsi_signal"] = (
            (result["rsi_14"] <= result["dynamic_rsi_oversold"]) &
            (result["rsi_rising"] == 1) &
            (result["rsi_14"] > result["rsi_ema9"])
        ).astype(int)

        # ──────────────────────────────────────────────
        # v6.0 Martin Momentum 지표 (41~46)
        # Martin(2023) 논문: EMA2 필터 + Dead Zone + 변동성 정규화
        # ──────────────────────────────────────────────

        # 41. EMA(8) — Martin fast EMA
        result["ema_8"] = df["close"].ewm(span=8, min_periods=8).mean()

        # 42. EMA(24) — Martin slow EMA
        result["ema_24"] = df["close"].ewm(span=24, min_periods=24).mean()

        # 43. EMA2 = fast - slow (Martin 모멘텀 핵심 신호)
        result["ema2_martin"] = result["ema_8"] - result["ema_24"]

        # 44. EMA2 정규화 (% 단위, Dead Zone 비교용)
        result["ema2_norm"] = result["ema2_martin"] / df["close"].replace(0, np.nan) * 100

        # 45. Dead Zone 플래그 (|ema2_norm| < 0.6 → 신호 무시)
        result["martin_dead_zone"] = (result["ema2_norm"].abs() < 0.05).astype(int)

        # 46. 일간 실현 변동성 (20일, 변동성 정규화 포지션용)
        result["daily_sigma"] = result["ret1"].rolling(20, min_periods=10).std()

        # ──────────────────────────────────────────────
        # v6.4 Gate 강화 지표 (47~50)
        # files.zip BES v2.2 → Gate 4/5 + Z-Score 표준화
        # ──────────────────────────────────────────────

        # 47. 52주(252거래일) 최고가
        result["high_252"] = df["high"].rolling(252, min_periods=60).max()

        # 48. 현재가 대비 52주 최고가 비율 (1.0 = 신고가)
        result["pct_of_52w_high"] = df["close"] / result["high_252"].replace(0, np.nan)

        # 49. BES 구성 요소 Z-Score (유니버스 비교용은 아니지만 시계열 정규화)
        #     pullback_atr의 60일 Z-Score
        pa_ma = result["pullback_atr"].rolling(60, min_periods=20).mean()
        pa_std = result["pullback_atr"].rolling(60, min_periods=20).std()
        result["pullback_atr_zscore"] = (
            (result["pullback_atr"] - pa_ma) / pa_std.replace(0, np.nan)
        )

        # 50. RSI Z-Score (14일 RSI의 60일 정규화)
        rsi_ma = result["rsi_14"].rolling(60, min_periods=20).mean()
        rsi_std = result["rsi_14"].rolling(60, min_periods=20).std()
        result["rsi_zscore"] = (
            (result["rsi_14"] - rsi_ma) / rsi_std.replace(0, np.nan)
        )

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
