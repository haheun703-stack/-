"""
v3.1 OBV/주가 다이버전스 스캐너

불리시 다이버전스: 주가↓ + OBV↑ → 매집 흔적
베어리시 다이버전스: 주가↑ + OBV↓ → 스마트머니 이탈

선형 회귀 기울기 기반 추세 판정.
엔티티(news_models.DivergenceSignal)에만 의존.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

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
