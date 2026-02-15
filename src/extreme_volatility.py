"""
v6.0 극한 변동성 탐지기

Alparslan & Kim(2021) 논문 핵심 구현:
1. 극한 변동성 탐지: 3조건 OR (ATR ratio, vol ratio, daily range)
2. Capitulation 프로파일: 연속하락 + 거래량클라이맥스 + RSI극단 + 반전캔들
3. 방향 판별: ambiguous면 시그널 억제 (방향 판별 없는 돌파는 위험)

교훈: 모든 돌파를 매수로 처리하면 안 됨. 방향 판별 필수.
"""

import logging

import numpy as np
import pandas as pd

from .entities.volatility_models import (
    CapitulationProfile,
    ExtremeVolatilityResult,
)

logger = logging.getLogger(__name__)


class ExtremeVolatilityDetector:
    """극한 변동성 탐지 + Capitulation 프로파일"""

    def __init__(
        self,
        atr_ratio_threshold: float = 3.0,
        vol_ratio_threshold: float = 5.0,
        daily_range_threshold: float = 10.0,
        min_down_days: int = 3,
        volume_climax_mult: float = 3.0,
        rsi_extreme: float = 20.0,
        min_capitulation_score: float = 70.0,
        allow_ambiguous: bool = False,
    ):
        self.atr_ratio_threshold = atr_ratio_threshold
        self.vol_ratio_threshold = vol_ratio_threshold
        self.daily_range_threshold = daily_range_threshold
        self.min_down_days = min_down_days
        self.volume_climax_mult = volume_climax_mult
        self.rsi_extreme = rsi_extreme
        self.min_capitulation_score = min_capitulation_score
        self.allow_ambiguous = allow_ambiguous

    def detect(self, df: pd.DataFrame, idx: int) -> ExtremeVolatilityResult:
        """
        극한 변동성 탐지.

        3조건 OR:
        - ATR ratio > 3.0 (현재 ATR / 60일 평균 ATR)
        - Volume ratio > 5.0 (현재 거래량 / 60일 평균 거래량)
        - Daily range > 10% ((고-저)/종가)
        """
        if idx < 60 or idx >= len(df):
            return ExtremeVolatilityResult()

        row = df.iloc[idx]
        close = row["close"]
        high = row["high"]
        low = row["low"]

        if pd.isna(close) or close <= 0:
            return ExtremeVolatilityResult()

        # ATR ratio
        atr_val = row.get("atr_14", np.nan)
        atr_60_avg = df["atr_14"].iloc[max(0, idx - 60):idx].mean() if "atr_14" in df.columns else np.nan
        atr_ratio = float(atr_val / atr_60_avg) if not pd.isna(atr_val) and not pd.isna(atr_60_avg) and atr_60_avg > 0 else 0.0

        # Volume ratio
        vol = row.get("volume", 0)
        vol_60_avg = df["volume"].iloc[max(0, idx - 60):idx].mean() if "volume" in df.columns else np.nan
        vol_ratio = float(vol / vol_60_avg) if not pd.isna(vol_60_avg) and vol_60_avg > 0 else 0.0

        # Daily range %
        daily_range = (high - low) / close * 100 if close > 0 else 0.0

        # 극한 변동성 판정 (3조건 OR)
        is_extreme = (
            atr_ratio > self.atr_ratio_threshold
            or vol_ratio > self.vol_ratio_threshold
            or daily_range > self.daily_range_threshold
        )

        if not is_extreme:
            return ExtremeVolatilityResult(
                atr_ratio=round(atr_ratio, 2),
                vol_ratio=round(vol_ratio, 2),
                daily_range_pct=round(daily_range, 2),
            )

        # 방향 판별
        direction = self.determine_direction(df, idx)

        # Capitulation 체크
        cap = self.detect_capitulation(df, idx)
        is_cap = cap.score >= self.min_capitulation_score
        if is_cap:
            direction = "capitulation"

        # 신뢰도 계산
        extreme_count = sum([
            atr_ratio > self.atr_ratio_threshold,
            vol_ratio > self.vol_ratio_threshold,
            daily_range > self.daily_range_threshold,
        ])
        confidence = 0.3 + extreme_count * 0.2
        if is_cap:
            confidence += 0.2
        if direction != "ambiguous":
            confidence += 0.1
        confidence = min(confidence, 1.0)

        return ExtremeVolatilityResult(
            is_extreme=True,
            atr_ratio=round(atr_ratio, 2),
            vol_ratio=round(vol_ratio, 2),
            daily_range_pct=round(daily_range, 2),
            direction=direction,
            is_capitulation=is_cap,
            confidence=round(confidence, 4),
        )

    def detect_capitulation(self, df: pd.DataFrame, idx: int) -> CapitulationProfile:
        """
        투매(Capitulation) 프로파일 탐지.

        조건:
        1. 연속 하락 N일 (min_down_days)
        2. 거래량 클라이맥스 (volume_climax_mult 배 이상)
        3. RSI < rsi_extreme (극단적 과매도)
        4. 반전 캔들 (양봉 + 아래꼬리)
        """
        if idx < self.min_down_days + 1 or idx >= len(df):
            return CapitulationProfile()

        row = df.iloc[idx]
        score = 0.0

        # 1. 연속 하락 일수
        down_days = 0
        for i in range(1, min(idx, 10) + 1):
            prev_idx = idx - i
            if prev_idx < 0:
                break
            if df["close"].iloc[prev_idx] > df["close"].iloc[prev_idx + 1]:
                down_days += 1
            else:
                break
        if down_days >= self.min_down_days:
            score += 25.0 + min((down_days - self.min_down_days) * 5, 15)

        # 2. 거래량 클라이맥스
        vol = row.get("volume", 0)
        vol_ma20 = row.get("volume_ma20", np.nan)
        vol_climax = False
        if not pd.isna(vol_ma20) and vol_ma20 > 0 and vol >= vol_ma20 * self.volume_climax_mult:
            vol_climax = True
            score += 25.0

        # 3. RSI 극단적 과매도
        rsi = row.get("rsi_14", np.nan)
        rsi_extreme = False
        if not pd.isna(rsi) and rsi < self.rsi_extreme:
            rsi_extreme = True
            score += 25.0

        # 4. 반전 캔들
        close = row["close"]
        open_price = row["open"]
        low = row["low"]
        is_bullish = close > open_price
        body = abs(close - open_price)
        lower_wick = min(close, open_price) - low
        reversal = False
        if is_bullish and body > 0 and lower_wick > body * 1.5:
            reversal = True
            score += 25.0

        return CapitulationProfile(
            consecutive_down_days=down_days,
            volume_climax=vol_climax,
            rsi_extreme_low=rsi_extreme,
            reversal_candle=reversal,
            score=round(score, 1),
        )

    def determine_direction(self, df: pd.DataFrame, idx: int) -> str:
        """
        방향 판별.

        방향 판별 없는 극한 변동성은 위험 → ambiguous로 분류.
        ambiguous는 기본적으로 시그널 억제.
        """
        if idx < 5 or idx >= len(df):
            return "ambiguous"

        row = df.iloc[idx]
        close = row["close"]
        open_price = row["open"]

        # 5일 수익률
        close_5d_ago = df["close"].iloc[idx - 5]
        ret_5d = (close - close_5d_ago) / close_5d_ago if close_5d_ago > 0 else 0

        # RSI
        rsi = row.get("rsi_14", np.nan)

        # 양봉 여부
        is_bullish = close > open_price

        # MA 관계
        sma20 = row.get("sma_20", np.nan)
        above_ma20 = not pd.isna(sma20) and close > sma20

        # 판별 로직
        bullish_signals = sum([
            ret_5d > 0.05,
            is_bullish,
            above_ma20,
            not pd.isna(rsi) and rsi > 40,
        ])
        bearish_signals = sum([
            ret_5d < -0.05,
            not is_bullish,
            not above_ma20,
            not pd.isna(rsi) and rsi < 30,
        ])

        if bullish_signals >= 3:
            return "bullish_breakout"
        elif bearish_signals >= 3:
            return "bearish_breakdown"
        else:
            return "ambiguous"
