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
from .smart_money import calc_institutional_streak, calc_smart_money_z

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
    def calc_linreg_slope(series: pd.Series, window: int) -> pd.Series:
        """Rolling 선형회귀 기울기 (가격 대비 정규화)"""
        def _slope(arr):
            if len(arr) < window or np.isnan(arr).any():
                return np.nan
            x = np.arange(len(arr))
            slope = np.polyfit(x, arr, 1)[0]
            return slope / arr[-1] if arr[-1] != 0 else 0.0
        return series.rolling(window, min_periods=window).apply(_slope, raw=True)

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
    def calc_stochastic_slow(df: pd.DataFrame, k_period: int = 14,
                             d_period: int = 3, smooth: int = 3) -> pd.DataFrame:
        """
        클래식 Stochastics Slow 계산 (George Lane)
        Fast %K = (Close - Lowest Low) / (Highest High - Lowest Low) × 100
        Slow %K = SMA(Fast %K, smooth)
        Slow %D = SMA(Slow %K, d_period)
        """
        lowest = df["low"].rolling(k_period, min_periods=k_period).min()
        highest = df["high"].rolling(k_period, min_periods=k_period).max()

        fast_k = (df["close"] - lowest) / (highest - lowest).replace(0, np.nan) * 100
        slow_k = fast_k.rolling(smooth, min_periods=1).mean()
        slow_d = slow_k.rolling(d_period, min_periods=1).mean()

        return pd.DataFrame({
            "stoch_slow_k": slow_k,
            "stoch_slow_d": slow_d,
        })

    @staticmethod
    def calc_parabolic_sar(df: pd.DataFrame, af_init: float = 0.02,
                           af_step: float = 0.02, af_max: float = 0.20) -> pd.DataFrame:
        """Parabolic SAR 계산 (Wilder 원전 알고리즘).

        Returns:
            DataFrame with columns: sar, sar_trend(1=상승/-1=하강), sar_af, sar_ep
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        sar = np.full(n, np.nan)
        trend = np.zeros(n, dtype=int)
        af = np.full(n, af_init)
        ep = np.full(n, np.nan)

        if n < 5:
            return pd.DataFrame(
                {"sar": sar, "sar_trend": trend, "sar_af": af, "sar_ep": ep},
                index=df.index,
            )

        # 초기 추세 판별 (첫 5봉)
        if close[4] > close[0]:
            trend[4], sar[4], ep[4] = 1, np.min(low[:5]), np.max(high[:5])
        else:
            trend[4], sar[4], ep[4] = -1, np.max(high[:5]), np.min(low[:5])
        af[4] = af_init

        for i in range(5, n):
            prev_sar, prev_af, prev_ep, prev_trend = sar[i-1], af[i-1], ep[i-1], trend[i-1]

            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            # SAR 보정 (이전 2봉 범위 내로 제한)
            if prev_trend == 1:
                new_sar = min(new_sar, low[i-1], low[i-2])
            else:
                new_sar = max(new_sar, high[i-1], high[i-2])

            # 추세 반전 체크
            if prev_trend == 1 and low[i] <= new_sar:
                trend[i], new_sar, ep[i], af[i] = -1, prev_ep, low[i], af_init
            elif prev_trend == -1 and high[i] >= new_sar:
                trend[i], new_sar, ep[i], af[i] = 1, prev_ep, high[i], af_init
            else:
                trend[i] = prev_trend
                if prev_trend == 1:
                    ep[i] = max(prev_ep, high[i])
                    af[i] = min(prev_af + af_step, af_max) if high[i] > prev_ep else prev_af
                else:
                    ep[i] = min(prev_ep, low[i])
                    af[i] = min(prev_af + af_step, af_max) if low[i] < prev_ep else prev_af

            sar[i] = new_sar

        return pd.DataFrame(
            {"sar": sar, "sar_trend": trend, "sar_af": af, "sar_ep": ep},
            index=df.index,
        )

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

        # ──────────────────────────────────────────────
        # v8.0 Gate+Score Hybrid 지표 (51~58)
        # ──────────────────────────────────────────────

        # 51. 선형회귀 기울기 20일 (S4 모멘텀 감속용)
        result["linreg_slope_20"] = self.calc_linreg_slope(df["close"], 20)

        # 52. 선형회귀 기울기 5일 (S4 단기 감속 비교용)
        result["linreg_slope_5"] = self.calc_linreg_slope(df["close"], 5)

        # 53. EMA 곡률 (EMA20의 2차 미분 — S4 변곡점 탐지 핵심)
        ema20 = df["close"].ewm(span=20, min_periods=20).mean()
        ema20_diff1 = ema20.diff()
        ema20_diff2 = ema20_diff1.diff()
        result["ema_curvature"] = ema20_diff2 / df["close"].replace(0, np.nan)
        result["ema_curvature_prev"] = result["ema_curvature"].shift(1)

        # 54. 가격 5일 추세 (S5 OBV 다이버전스 비교용)
        result["price_trend_5d"] = df["close"].pct_change(5)

        # 55. OBV 5일 추세 (S5 매집 감지)
        result["obv_trend_5d"] = result["obv"].pct_change(5)

        # 56. MACD 히스토그램 전일값 (S4 감속 감지)
        result["macd_histogram_prev"] = result["macd_histogram"].shift(1)

        # 57. TRIX 전일값 / TRIX Signal 전일값 (T1 골든크로스 감지)
        result["trix_prev"] = result["trix"].shift(1)
        result["trix_signal_prev"] = result["trix_signal"].shift(1)

        # 58. RSI 전일값 (T2 RSI 상향돌파 감지)
        result["rsi_prev"] = result["rsi_14"].shift(1)

        # ──────────────────────────────────────────────
        # v8.3 수급 6-Layer 지표 (59~62)
        # 외국인합계 → 파생 지표 (S5 SmartMoney 강화)
        # ──────────────────────────────────────────────

        # 외국인 수급 컬럼 탐색
        foreign_col = None
        for fc in ["외국인합계", "foreign_net"]:
            if fc in result.columns:
                foreign_col = fc
                break

        if foreign_col is not None:
            fnet = result[foreign_col].fillna(0)

            # 59. 외국인 5일 누적 순매수 (S5 기관매집 감지)
            result["foreign_net_5d"] = fnet.rolling(5, min_periods=1).sum()

            # 60. 외국인 20일 누적 순매수 (장기 수급 추세)
            result["foreign_net_20d"] = fnet.rolling(20, min_periods=1).sum()

            # 61. 외국인 연속 순매수 일수 (매집 강도)
            consec = []
            count = 0
            for val in fnet:
                if val > 0:
                    count += 1
                else:
                    count = 0
                consec.append(count)
            result["foreign_consecutive_buy"] = consec

            # 62. 외국인+거래량 복합 신호 (수급 확인)
            #     외국인 순매수 + 거래량 평균 이상 → 강한 매집
            vol_above_avg = (df["volume"] > result["volume_ma20"]).astype(int)
            foreign_buying = (fnet > 0).astype(int)
            result["foreign_vol_confirm"] = vol_above_avg * foreign_buying
        else:
            result["foreign_net_5d"] = 0
            result["foreign_net_20d"] = 0
            result["foreign_consecutive_buy"] = 0
            result["foreign_vol_confirm"] = 0

        # ──────────────────────────────────────────────
        # v8.4 L2 공매도 레이어 지표 (63~67)
        # 숏커버링 = 포물선 연료
        # ──────────────────────────────────────────────

        if "short_ratio" in result.columns:
            sr = result["short_ratio"].fillna(0)

            # 63. 공매도 비중 40일 이동평균
            result["short_ratio_ma40"] = sr.rolling(40, min_periods=10).mean()

            # 64. 공매도 스파이크 (현재 / 40일 평균, 1.0=정상, 2.0=스파이크)
            sr_ma40 = result["short_ratio_ma40"].replace(0, np.nan)
            result["short_spike"] = sr / sr_ma40
            result["short_spike"] = result["short_spike"].fillna(1.0)
        else:
            result["short_ratio_ma40"] = 0.0
            result["short_spike"] = 1.0

        if "short_balance" in result.columns:
            sb = result["short_balance"].fillna(0)

            # 65. 공매도 잔고 5일 변화율 (%)
            sb_5d = sb.shift(5).replace(0, np.nan)
            result["short_balance_chg_5d"] = ((sb - sb_5d) / sb_5d * 100).fillna(0)
        else:
            result["short_balance_chg_5d"] = 0.0

        if "lending_balance" in result.columns:
            lb = result["lending_balance"].fillna(0)

            # 66. 대차잔고 5일 변화율 (%)
            lb_5d = lb.shift(5).replace(0, np.nan)
            result["lending_balance_chg_5d"] = ((lb - lb_5d) / lb_5d * 100).fillna(0)
        else:
            result["lending_balance_chg_5d"] = 0.0

        # 67. 숏커버링 신호 (공매도 잔고 5일 -20% 이상 감소)
        result["short_cover_signal"] = (
            result["short_balance_chg_5d"] < -20
        ).astype(int)

        # ──────────────────────────────────────────────
        # v8.4 L4 글로벌 매크로 지표 (68~71)
        # VIX/환율/SOXX → 시장 체제 보조 신호
        # ──────────────────────────────────────────────

        if "vix_close" in result.columns:
            vix = result["vix_close"].ffill()

            # 68. VIX Z-score (60일 기준)
            vix_ma60 = vix.rolling(60, min_periods=20).mean()
            vix_std60 = vix.rolling(60, min_periods=20).std()
            result["vix_zscore"] = (
                (vix - vix_ma60) / vix_std60.replace(0, np.nan)
            ).fillna(0)
        else:
            result["vix_zscore"] = 0.0

        if "usdkrw_close" in result.columns:
            usdkrw = result["usdkrw_close"].ffill()

            # 69. 원/달러 20일 변화율 (원화 강세 = 음수)
            usdkrw_20d = usdkrw.shift(20).replace(0, np.nan)
            result["usdkrw_trend_20d"] = (
                (usdkrw - usdkrw_20d) / usdkrw_20d * 100
            ).fillna(0)
        else:
            result["usdkrw_trend_20d"] = 0.0

        if "soxx_close" in result.columns:
            soxx = result["soxx_close"].ffill()

            # 70. SOXX 20일 수익률 (%)
            soxx_20d = soxx.shift(20).replace(0, np.nan)
            result["soxx_trend_20d"] = (
                (soxx - soxx_20d) / soxx_20d * 100
            ).fillna(0)
        else:
            result["soxx_trend_20d"] = 0.0

        # 71. 매크로 우호 신호 복합 (VIX 낮음 + 원화 강세 + 반도체 상승)
        result["macro_favorable"] = (
            (result["vix_zscore"] < -0.5) &
            (result["usdkrw_trend_20d"] < 0) &
            (result["soxx_trend_20d"] > 0)
        ).astype(int)

        # ──────────────────────────────────────────────
        # v8.4 L5 센티먼트 지표 (72~73)
        # 비관 극단 → 역발상 매수 신호
        # ──────────────────────────────────────────────

        if "sentiment_pessimism" in result.columns:
            sp = result["sentiment_pessimism"].fillna(0.5)
            # 72. 센티먼트 비관도 (0~1)
            result["sentiment_pessimism"] = sp

            # 73. 비관 극단 신호 (비관도 > 0.4 = 40%+ 비관 게시글)
            result["sentiment_extreme"] = (sp > 0.4).astype(int)
        else:
            result["sentiment_pessimism"] = 0.5
            result["sentiment_extreme"] = 0

        # ──────────────────────────────────────────────
        # v8.4 L6 연기금 지표 (74~75)
        # 연기금 순매수 = 장기 스마트머니 신호
        # ──────────────────────────────────────────────

        if "pension_net" in result.columns:
            pn = result["pension_net"].fillna(0)

            # 74. 연기금 5일 누적 순매수
            result["pension_net_5d"] = pn.rolling(5, min_periods=1).sum()
        else:
            result["pension_net_5d"] = 0

        # 75. pension_top_buyer는 backfill에서 직접 추가됨 (0/1 플래그)
        if "pension_top_buyer" not in result.columns:
            result["pension_top_buyer"] = 0

        # ──────────────────────────────────────────────
        # v10.4 Stochastics Slow (76~78)
        # 가격 기반 클래식 Stochastic — StochRSI와 병행
        # ──────────────────────────────────────────────

        # 76-77. Stochastics Slow %K, %D (14,3,3)
        stoch_slow = self.calc_stochastic_slow(df, k_period=14, d_period=3, smooth=3)
        result["stoch_slow_k"] = stoch_slow["stoch_slow_k"]
        result["stoch_slow_d"] = stoch_slow["stoch_slow_d"]

        # 78. Stoch Slow 골든크로스 (K가 D를 상향 돌파)
        result["stoch_slow_golden"] = (
            (result["stoch_slow_k"] > result["stoch_slow_d"]) &
            (result["stoch_slow_k"].shift(1) <= result["stoch_slow_d"].shift(1))
        ).astype(int)

        # ──────────────────────────────────────────────
        # v10.5 Parabolic SAR (79~83)
        # 트레일링 스톱 + 추세 반전 탐지
        # ──────────────────────────────────────────────

        # 79-82. Parabolic SAR 4개 컬럼
        sar_df = self.calc_parabolic_sar(df)
        result["sar"] = sar_df["sar"]
        result["sar_trend"] = sar_df["sar_trend"]        # 1=상승, -1=하강
        result["sar_af"] = sar_df["sar_af"]              # 가속계수 (0.02~0.20)
        result["sar_ep"] = sar_df["sar_ep"]              # 극단점 (추세 내 최고/최저)

        # 83. SAR 반전 신호 (전일 하강 → 오늘 상승 = 매수 반전)
        result["sar_reversal_up"] = (
            (result["sar_trend"] == 1) & (result["sar_trend"].shift(1) == -1)
        ).astype(int)

        return result

    # ──────────────────────────────────────────────
    # 전종목 일괄 처리
    # ──────────────────────────────────────────────

    def _load_macro_data(self) -> pd.DataFrame | None:
        """글로벌 매크로 데이터 로드 (없으면 None)"""
        macro_path = Path("data/macro/global_indices.parquet")
        if macro_path.exists():
            try:
                df = pd.read_parquet(macro_path)
                df.index = pd.to_datetime(df.index)
                logger.info(f"매크로 데이터 로드: {len(df)}일, {list(df.columns)}")
                return df
            except Exception as e:
                logger.warning(f"매크로 데이터 로드 실패: {e}")
        return None

    def process_all(self) -> int:
        """raw 디렉토리의 모든 parquet을 처리하여 processed에 저장"""
        raw_files = sorted(self.raw_dir.glob("*.parquet"))
        if not raw_files:
            logger.error("data/raw에 parquet 파일이 없습니다")
            return 0

        # L4 매크로 데이터 사전 로드
        macro_df = self._load_macro_data()

        processed_count = 0
        for fpath in tqdm(raw_files, desc="📈 지표 계산"):
            ticker = fpath.stem
            try:
                df = pd.read_parquet(fpath)
                if len(df) < 200:  # 200일 미만 데이터는 지표 계산 불가
                    logger.debug(f"{ticker}: 데이터 부족 ({len(df)}일), 건너뜀")
                    continue

                # L4 매크로 데이터 merge (날짜 기준)
                if macro_df is not None:
                    df.index = pd.to_datetime(df.index)
                    for col in macro_df.columns:
                        if col not in df.columns:
                            df = df.join(macro_df[[col]], how="left")
                    df = df.ffill()

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
