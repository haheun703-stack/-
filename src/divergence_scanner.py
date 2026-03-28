"""
v3.2 OBV/TRIX 다이버전스 스캐너

OBV 다이버전스: 주가↓ + OBV↑ → 매집 흔적 (선형 회귀 기반)
TRIX 다이버전스: 주가 저점↓↓ + TRIX 저점↑↑ → 모멘텀 반전 (극값 비교 기반)

엔티티(news_models.DivergenceSignal)에만 의존.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.entities.news_models import DivergenceSignal

_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_sm_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else _CFG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("smart_money_v2", {})


class DivergenceScanner:
    """OBV/주가 다이버전스 스캐너"""

    def __init__(self, config_path: Path | str | None = None):
        cfg = _load_sm_config(config_path)
        self.lookback = cfg.get("obv_divergence_lookback", 30)
        self.slope_threshold = cfg.get("obv_slope_threshold", 0.0)

    def scan(self, df: pd.DataFrame, lookback: int | None = None) -> DivergenceSignal:
        """
        다이버전스 감지.

        Args:
            df: 'Close'/'close' + 'OBV'/'obv' 컬럼이 있는 DataFrame
            lookback: 검색 기간 (기본 settings 값)
        """
        lb = lookback or self.lookback

        # 컬럼명 정규화
        close_col = "Close" if "Close" in df.columns else "close"
        obv_col = "OBV" if "OBV" in df.columns else "obv"

        if close_col not in df.columns or obv_col not in df.columns:
            return DivergenceSignal(reason="Close/OBV 컬럼 없음")

        if len(df) < lb:
            return DivergenceSignal(
                lookback_days=lb,
                reason=f"데이터 부족 ({len(df)} < {lb})",
            )

        recent = df.tail(lb)
        close = recent[close_col].astype(float).values
        obv = recent[obv_col].astype(float).values

        # NaN 체크
        if np.isnan(close).any() or np.isnan(obv).any():
            valid = ~(np.isnan(close) | np.isnan(obv))
            close = close[valid]
            obv = obv[valid]
            if len(close) < 10:
                return DivergenceSignal(lookback_days=lb, reason="유효 데이터 부족")

        # 선형 회귀 기울기
        x = np.arange(len(close))
        price_slope = np.polyfit(x, close, 1)[0]
        price_slope_pct = price_slope / (close.mean() + 1e-9)

        obv_slope = np.polyfit(x, obv, 1)[0]
        obv_slope_pct = obv_slope / (abs(obv.mean()) + 1)

        # 추세 분류
        price_trend = self._classify_trend(price_slope_pct)
        obv_trend = self._classify_trend(obv_slope_pct)

        # ── 불리시 다이버전스 ──
        if price_trend in ("falling", "flat") and obv_trend == "rising":
            strength, confidence = self._calc_strength(price_slope_pct, obv_slope_pct)
            confidence = self._confirm_with_lows(close, obv, confidence)
            return DivergenceSignal(
                type="bullish",
                strength=strength,
                confidence=min(100, confidence),
                price_trend=price_trend,
                obv_trend=obv_trend,
                lookback_days=lb,
                reason=f"불리시 다이버전스 ({strength}): 주가 {price_trend} + OBV {obv_trend}",
            )

        # ── 베어리시 다이버전스 ──
        if price_trend == "rising" and obv_trend in ("falling", "flat"):
            strength, confidence = self._calc_strength(price_slope_pct, obv_slope_pct)
            return DivergenceSignal(
                type="bearish",
                strength=strength,
                confidence=min(100, confidence),
                price_trend=price_trend,
                obv_trend=obv_trend,
                lookback_days=lb,
                reason=f"베어리시 다이버전스 ({strength}): 주가 {price_trend} + OBV {obv_trend}",
            )

        # ── 다이버전스 없음 ──
        return DivergenceSignal(
            price_trend=price_trend,
            obv_trend=obv_trend,
            lookback_days=lb,
            reason=f"다이버전스 미감지 (주가 {price_trend}, OBV {obv_trend})",
        )

    # ──────────────────────────────────────────
    # v3.2 TRIX 극값 기반 다이버전스
    # ──────────────────────────────────────────

    def scan_trix(
        self, df: pd.DataFrame, lookback: int = 40, order: int = 3,
    ) -> tuple[str, float]:
        """TRIX 극값 비교로 다이버전스 감지.

        Args:
            df: 'close'/'Close' + 'trix'/'TRIX' 컬럼이 있는 DataFrame
            lookback: 극값 탐색 기간 (일)
            order: argrelextrema order (극값 판정 반경)

        Returns:
            (type, strength) — type: "bullish"/"bearish"/"none", strength: 0.0~1.0
        """
        from scipy.signal import argrelextrema

        close_col = "Close" if "Close" in df.columns else "close"
        trix_col = "TRIX" if "TRIX" in df.columns else "trix"

        if close_col not in df.columns or trix_col not in df.columns:
            return "none", 0.0

        if len(df) < lookback:
            return "none", 0.0

        recent = df.tail(lookback).copy()
        close_arr = recent[close_col].astype(float).values
        trix_arr = recent[trix_col].astype(float).values

        if np.isnan(close_arr).any() or np.isnan(trix_arr).any():
            valid = ~(np.isnan(close_arr) | np.isnan(trix_arr))
            close_arr = close_arr[valid]
            trix_arr = trix_arr[valid]
            if len(close_arr) < 10:
                return "none", 0.0

        # 극값 추출 (로컬 최저/최고점)
        price_lows = argrelextrema(close_arr, np.less, order=order)[0]
        price_highs = argrelextrema(close_arr, np.greater, order=order)[0]
        trix_lows = argrelextrema(trix_arr, np.less, order=order)[0]
        trix_highs = argrelextrema(trix_arr, np.greater, order=order)[0]

        # ── 불리시 다이버전스: 가격 Lower Low + TRIX Higher Low ──
        bull_type, bull_str = self._check_extrema_divergence(
            close_arr, trix_arr, price_lows, trix_lows, "bullish",
        )
        if bull_type == "bullish":
            return bull_type, bull_str

        # ── 베어리시 다이버전스: 가격 Higher High + TRIX Lower High ──
        bear_type, bear_str = self._check_extrema_divergence(
            close_arr, trix_arr, price_highs, trix_highs, "bearish",
        )
        if bear_type == "bearish":
            return bear_type, bear_str

        return "none", 0.0

    def scan_all(self, df: pd.DataFrame, lookback: int | None = None) -> DivergenceSignal:
        """OBV + TRIX 다이버전스 통합 스캔.

        기존 OBV scan() 결과에 TRIX 극값 다이버전스를 추가한다.
        두 신호가 동시 발생하면 confidence를 상향 조정한다.
        """
        signal = self.scan(df, lookback)

        # TRIX 다이버전스 추가
        trix_type, trix_strength = self.scan_trix(df)
        signal.trix_type = trix_type
        signal.trix_strength = trix_strength

        # OBV + TRIX 동시 발생 → 신뢰도 보너스
        if signal.type != "none" and trix_type == signal.type:
            signal.confidence = min(100, signal.confidence + 15.0)
            signal.reason += f" + TRIX {trix_type}({trix_strength:.2f})"

        return signal

    @staticmethod
    def _check_extrema_divergence(
        price_arr: np.ndarray,
        trix_arr: np.ndarray,
        price_extrema: np.ndarray,
        trix_extrema: np.ndarray,
        div_type: str,
    ) -> tuple[str, float]:
        """극값 페어 비교로 다이버전스 판정.

        bullish: 가격 저점↓ + TRIX 저점↑ (Lower Low vs Higher Low)
        bearish: 가격 고점↑ + TRIX 고점↓ (Higher High vs Lower High)
        """
        if len(price_extrema) < 2 or len(trix_extrema) < 2:
            return "none", 0.0

        # 가장 최근 2개 극값
        p1_idx, p2_idx = price_extrema[-2], price_extrema[-1]
        # TRIX 극값은 가격 극값 +-7일 이내 매칭
        t1_idx = _find_nearest(trix_extrema, p1_idx, max_gap=7)
        t2_idx = _find_nearest(trix_extrema, p2_idx, max_gap=7)

        if t1_idx is None or t2_idx is None:
            return "none", 0.0

        p1, p2 = price_arr[p1_idx], price_arr[p2_idx]
        t1, t2 = trix_arr[t1_idx], trix_arr[t2_idx]

        if div_type == "bullish":
            # 가격 Lower Low + TRIX Higher Low
            if p2 < p1 and t2 > t1:
                price_drop = (p1 - p2) / (p1 + 1e-9)
                trix_rise = t2 - t1
                strength = min(1.0, (price_drop * 10 + abs(trix_rise)) / 2)
                return "bullish", strength
        else:
            # 가격 Higher High + TRIX Lower High
            if p2 > p1 and t2 < t1:
                price_rise = (p2 - p1) / (p1 + 1e-9)
                trix_drop = t1 - t2
                strength = min(1.0, (price_rise * 10 + abs(trix_drop)) / 2)
                return "bearish", strength

        return "none", 0.0

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _classify_trend(slope_pct: float) -> str:
        if slope_pct < -0.001:
            return "falling"
        elif slope_pct > 0.001:
            return "rising"
        return "flat"

    @staticmethod
    def _calc_strength(price_slope_pct: float, obv_slope_pct: float) -> tuple[str, float]:
        div_strength = abs(obv_slope_pct) + abs(price_slope_pct)
        if div_strength > 0.01:
            return "strong", 85.0
        elif div_strength > 0.005:
            return "moderate", 65.0
        return "weak", 45.0

    @staticmethod
    def _confirm_with_lows(close: np.ndarray, obv: np.ndarray, base_confidence: float) -> float:
        """저점 비교로 불리시 다이버전스 추가 확인"""
        mid = len(close) // 2
        if mid < 2:
            return base_confidence

        first_half_low = close[:mid].min()
        second_half_low = close[mid:].min()
        first_half_obv_low = obv[:mid].min()
        second_half_obv_low = obv[mid:].min()

        # 주가 저점 하락 + OBV 저점 상승 = 교과서적 다이버전스
        if second_half_low < first_half_low and second_half_obv_low > first_half_obv_low:
            return base_confidence + 15.0

        return base_confidence


def _find_nearest(arr: np.ndarray, target: int, max_gap: int = 7) -> int | None:
    """배열에서 target에 가장 가까운 값 찾기 (max_gap 이내)."""
    if len(arr) == 0:
        return None
    diffs = np.abs(arr - target)
    min_idx = diffs.argmin()
    if diffs[min_idx] <= max_gap:
        return int(arr[min_idx])
    return None
