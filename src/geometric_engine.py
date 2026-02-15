"""
기하학적 퀀트 매매 엔진 (Geometric Quant Trading Engine)

v1.1: Bug fixes + L7 보조 레이어 통합
  - #1 dict 순서 의존 제거
  - #2 엘리어트 슬라이딩 윈도우 탐색
  - #3 신뢰도 계산 정규화
  - #4 5파 매도 시그널 추가
  - #5 zigzag_pct 동적 조정 (ATR 기반)
  - #6 전역 warnings 억제 제거

v2.0: GeoQuant 통합 (10지표 + 7프로파일)
  - Approach A: 구조적 패턴 (harmonic, elliott, slope)
  - Approach B: 에너지 수렴 (squeeze, curvature, slope_mom, confluence)
  - Approach C: 극단 회귀 (mean_rev, vol_climax, band_breach)
  - 7 weight profiles: default/reversal/breakout/capitulation/bull/bear/sideways

Compatible with: QUANTUM STRATEGY v3.1 / v4.0 / v4.5 / v4.7
Dependencies: numpy, pandas, scipy
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# =============================================================================
# 0. 헬퍼 함수 (NumPy 기반 계산 유틸리티)
# =============================================================================

def _ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    """지수이동평균 (pandas 미사용)"""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = float(arr[0])
    for i in range(1, len(arr)):
        out[i] = alpha * float(arr[i]) + (1 - alpha) * out[i - 1]
    return out


def _sma_np(arr: np.ndarray, n: int) -> np.ndarray:
    """단순이동평균 (cumsum 방식)"""
    arr_f = arr.astype(float)
    out = np.full(len(arr_f), np.nan)
    cs = np.cumsum(arr_f)
    out[n - 1:] = (cs[n - 1:] - np.concatenate([[0], cs[:-n]])) / n
    return out


def _linreg_slope_np(arr: np.ndarray, n: int) -> np.ndarray:
    """Rolling 선형회귀 기울기"""
    out = np.full(len(arr), np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1: i + 1].astype(float)
        y_mean = y.mean()
        out[i] = np.sum((x - x_mean) * (y - y_mean)) / x_var
    return out


# =============================================================================
# 0-1. IndicatorResult (신규 지표 통일 반환형)
# =============================================================================

@dataclass
class IndicatorResult:
    """개별 지표 분석 결과"""
    name: str       # 지표 ID
    signal: str     # BUY / SELL / HOLD
    score: float    # 0~100
    detail: str     # 한줄 설명
    meta: dict      # 상세 메타데이터


# =============================================================================
# 0-2. 가중치 프로파일 (7종)
# =============================================================================

WEIGHT_PROFILES = {
    "default": {
        "harmonic": 0.15, "elliott": 0.10, "slope": 0.08,
        "squeeze": 0.15, "curvature": 0.12, "slope_mom": 0.08, "confluence": 0.08,
        "mean_rev": 0.10, "vol_climax": 0.07, "band_breach": 0.07,
    },
    "reversal": {
        "harmonic": 0.25, "elliott": 0.15, "slope": 0.08,
        "squeeze": 0.05, "curvature": 0.10, "slope_mom": 0.05, "confluence": 0.12,
        "mean_rev": 0.08, "vol_climax": 0.05, "band_breach": 0.07,
    },
    "breakout": {
        "harmonic": 0.08, "elliott": 0.08, "slope": 0.05,
        "squeeze": 0.25, "curvature": 0.18, "slope_mom": 0.12, "confluence": 0.08,
        "mean_rev": 0.05, "vol_climax": 0.06, "band_breach": 0.05,
    },
    "capitulation": {
        "harmonic": 0.05, "elliott": 0.03, "slope": 0.05,
        "squeeze": 0.02, "curvature": 0.15, "slope_mom": 0.05, "confluence": 0.10,
        "mean_rev": 0.25, "vol_climax": 0.20, "band_breach": 0.10,
    },
    "bull": {
        "harmonic": 0.12, "elliott": 0.08, "slope": 0.08,
        "squeeze": 0.20, "curvature": 0.12, "slope_mom": 0.12, "confluence": 0.08,
        "mean_rev": 0.08, "vol_climax": 0.05, "band_breach": 0.07,
    },
    "bear": {
        "harmonic": 0.18, "elliott": 0.15, "slope": 0.10,
        "squeeze": 0.10, "curvature": 0.08, "slope_mom": 0.07, "confluence": 0.05,
        "mean_rev": 0.12, "vol_climax": 0.08, "band_breach": 0.07,
    },
    "sideways": {
        "harmonic": 0.15, "elliott": 0.08, "slope": 0.05,
        "squeeze": 0.20, "curvature": 0.08, "slope_mom": 0.08, "confluence": 0.16,
        "mean_rev": 0.08, "vol_climax": 0.05, "band_breach": 0.07,
    },
}


# =============================================================================
# 1. 피벗 포인트(고점/저점) 추출
# =============================================================================

class PivotDetector:
    """
    ZigZag 기반 피벗 포인트 추출기
    - find_peaks보다 안정적인 ZigZag 알고리즘 사용
    - 실시간 매매를 위한 '확정 피벗' 판단 로직 포함
    """

    def __init__(self, zigzag_pct: float = 5.0, min_bars: int = 5):
        self.zigzag_pct = zigzag_pct / 100.0
        self.min_bars = min_bars

    @classmethod
    def from_atr(cls, df: pd.DataFrame, atr_mult: float = 1.5,
                 min_bars: int = 5) -> "PivotDetector":
        """ATR 기반 동적 zigzag_pct 생성 (Bug #5 수정)"""
        if "atr_14" in df.columns:
            atr = df["atr_14"].iloc[-1]
            price = df["close"].iloc[-1]
        else:
            # ATR 직접 계산
            high = df["high"]
            low = df["low"]
            prev_close = df["close"].shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            price = df["close"].iloc[-1]

        if pd.isna(atr) or price == 0:
            return cls(zigzag_pct=5.0, min_bars=min_bars)

        # ATR/Price 비율 * 배수 → zigzag %
        zigzag_pct = max(2.0, min(10.0, (atr / price) * 100 * atr_mult))
        return cls(zigzag_pct=zigzag_pct, min_bars=min_bars)

    def find_pivots_zigzag(self, prices: np.ndarray) -> list[dict]:
        """ZigZag 알고리즘으로 고점/저점 추출"""
        if len(prices) < 3:
            return []

        pivots = []
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        last_pivot_type = None

        # 초기 방향 결정
        for i in range(1, len(prices)):
            change = (prices[i] - prices[0]) / prices[0]
            if abs(change) >= self.zigzag_pct:
                if change > 0:
                    pivots.append({"index": 0, "price": prices[0], "type": "low"})
                    last_pivot_type = "low"
                else:
                    pivots.append({"index": 0, "price": prices[0], "type": "high"})
                    last_pivot_type = "high"
                last_pivot_idx = 0
                last_pivot_price = prices[0]
                break

        if not pivots:
            return []

        # ZigZag 순회
        current_extreme_idx = last_pivot_idx
        current_extreme_price = last_pivot_price

        for i in range(1, len(prices)):
            if last_pivot_type == "low":
                if prices[i] > current_extreme_price:
                    current_extreme_idx = i
                    current_extreme_price = prices[i]
                elif current_extreme_price > 0 and \
                     (current_extreme_price - prices[i]) / current_extreme_price >= self.zigzag_pct:
                    if current_extreme_idx - last_pivot_idx >= self.min_bars:
                        pivots.append({
                            "index": current_extreme_idx,
                            "price": current_extreme_price,
                            "type": "high"
                        })
                        last_pivot_idx = current_extreme_idx
                        last_pivot_price = current_extreme_price
                        last_pivot_type = "high"
                    current_extreme_idx = i
                    current_extreme_price = prices[i]
            else:
                if prices[i] < current_extreme_price:
                    current_extreme_idx = i
                    current_extreme_price = prices[i]
                elif current_extreme_price > 0 and \
                     (prices[i] - current_extreme_price) / current_extreme_price >= self.zigzag_pct:
                    if current_extreme_idx - last_pivot_idx >= self.min_bars:
                        pivots.append({
                            "index": current_extreme_idx,
                            "price": current_extreme_price,
                            "type": "low"
                        })
                        last_pivot_idx = current_extreme_idx
                        last_pivot_price = current_extreme_price
                        last_pivot_type = "low"
                    current_extreme_idx = i
                    current_extreme_price = prices[i]

        return pivots

    def find_pivots_scipy(
        self, prices: np.ndarray, prominence: float = None, distance: int = 10
    ) -> list[dict]:
        """scipy.signal.find_peaks 기반 피벗 추출 (보조용)"""
        if prominence is None:
            prominence = np.std(prices) * 0.5

        high_idx, _ = find_peaks(prices, prominence=prominence, distance=distance)
        low_idx, _ = find_peaks(-prices, prominence=prominence, distance=distance)

        pivots = []
        for idx in high_idx:
            pivots.append({"index": int(idx), "price": prices[idx], "type": "high"})
        for idx in low_idx:
            pivots.append({"index": int(idx), "price": prices[idx], "type": "low"})

        pivots.sort(key=lambda x: x["index"])
        return pivots

    def is_pivot_confirmed(self, prices: np.ndarray, pivot_idx: int,
                           pivot_type: str, confirm_bars: int = 3) -> bool:
        """피벗이 확정되었는지 판단 (실시간 매매용)"""
        if pivot_idx + confirm_bars >= len(prices):
            return False

        pivot_price = prices[pivot_idx]
        after_prices = prices[pivot_idx + 1: pivot_idx + 1 + confirm_bars]

        if pivot_type == "high":
            return all(p < pivot_price for p in after_prices)
        else:
            return all(p > pivot_price for p in after_prices)


# =============================================================================
# 2. 하모닉 패턴 감지
# =============================================================================

class HarmonicPatternType(Enum):
    GARTLEY = "Gartley"
    BAT = "Bat"
    BUTTERFLY = "Butterfly"
    CRAB = "Crab"
    SHARK = "Shark"


@dataclass
class HarmonicPattern:
    """감지된 하모닉 패턴 정보"""
    pattern_type: HarmonicPatternType
    direction: str
    points: dict
    ratios: dict
    score: float
    prz_low: float
    prz_high: float
    stop_loss: float
    target_1: float
    target_2: float


class HarmonicDetector:
    """
    하모닉 패턴 감지 엔진
    - 5가지 주요 패턴 (Gartley, Bat, Butterfly, Crab, Shark)
    - 피보나치 비율 허용 오차(tolerance) 기반 매칭
    """

    PATTERNS = {
        HarmonicPatternType.GARTLEY: {
            "AB_XA": (0.588, 0.618, 0.648),
            "BC_AB": (0.382, 0.618, 0.886),
            "CD_BC": (1.272, 1.618, 1.618),
            "AD_XA": (0.766, 0.786, 0.806),
        },
        HarmonicPatternType.BAT: {
            "AB_XA": (0.382, 0.500, 0.500),
            "BC_AB": (0.382, 0.618, 0.886),
            "CD_BC": (1.618, 2.000, 2.618),
            "AD_XA": (0.876, 0.886, 0.896),
        },
        HarmonicPatternType.BUTTERFLY: {
            "AB_XA": (0.766, 0.786, 0.806),
            "BC_AB": (0.382, 0.618, 0.886),
            "CD_BC": (1.618, 2.000, 2.618),
            "AD_XA": (1.250, 1.272, 1.414),
        },
        HarmonicPatternType.CRAB: {
            "AB_XA": (0.382, 0.500, 0.618),
            "BC_AB": (0.382, 0.618, 0.886),
            "CD_BC": (2.240, 3.140, 3.618),
            "AD_XA": (1.586, 1.618, 1.648),
        },
        HarmonicPatternType.SHARK: {
            "AB_XA": (1.130, 1.272, 1.414),
            "BC_AB": (1.618, 2.000, 2.240),
            "CD_BC": (0.886, 1.130, 1.130),
            "AD_XA": (0.876, 0.886, 1.130),
        },
    }

    def __init__(self, tolerance: float = 0.03,
                 zigzag_pct: float = 3.0, min_bars: int = 5):
        self.tolerance = tolerance
        self.pivot_detector = PivotDetector(zigzag_pct=zigzag_pct, min_bars=min_bars)

    def _check_ratio(self, actual: float, min_r: float, max_r: float) -> tuple[bool, float]:
        expanded_min = min_r * (1 - self.tolerance)
        expanded_max = max_r * (1 + self.tolerance)
        ideal = (min_r + max_r) / 2

        if expanded_min <= actual <= expanded_max:
            deviation = abs(actual - ideal) / ideal
            return True, deviation
        return False, 999.0

    def _calculate_ratios(self, X: float, A: float, B: float,
                          C: float, D: float) -> dict[str, float]:
        xa = abs(A - X)
        ab = abs(B - A)
        bc = abs(C - B)
        cd = abs(D - C)
        ad = abs(D - A)

        if xa == 0 or ab == 0 or bc == 0:
            return {}

        return {
            "AB_XA": ab / xa,
            "BC_AB": bc / ab,
            "CD_BC": cd / bc,
            "AD_XA": ad / xa,
        }

    def _calculate_score(self, ratios: dict[str, float],
                         pattern_type: HarmonicPatternType) -> float:
        pattern_def = self.PATTERNS[pattern_type]
        total_deviation = 0
        count = 0

        for key, (min_r, ideal_r, max_r) in pattern_def.items():
            if key in ratios:
                deviation = abs(ratios[key] - ideal_r) / ideal_r
                total_deviation += deviation
                count += 1

        if count == 0:
            return 0

        avg_deviation = total_deviation / count
        score = max(0, 100 * (1 - avg_deviation * 5))
        return round(score, 1)

    def detect_patterns(self, df: pd.DataFrame,
                        lookback: int = 100) -> list[HarmonicPattern]:
        """하모닉 패턴 감지 메인 함수"""
        prices = df["close"].values[-lookback:]
        pivots = self.pivot_detector.find_pivots_zigzag(prices)

        if len(pivots) < 5:
            return []

        detected = []

        for i in range(len(pivots) - 4):
            pts = pivots[i:i + 5]

            X_price = pts[0]["price"]
            A_price = pts[1]["price"]
            B_price = pts[2]["price"]
            C_price = pts[3]["price"]
            D_price = pts[4]["price"]

            if pts[0]["type"] == "low":
                direction = "bullish"
            else:
                direction = "bearish"

            ratios = self._calculate_ratios(X_price, A_price, B_price, C_price, D_price)
            if not ratios:
                continue

            for p_type, p_def in self.PATTERNS.items():
                all_match = True
                for key, (min_r, ideal_r, max_r) in p_def.items():
                    if key not in ratios:
                        all_match = False
                        break
                    matched, _ = self._check_ratio(ratios[key], min_r, max_r)
                    if not matched:
                        all_match = False
                        break

                if all_match:
                    score = self._calculate_score(ratios, p_type)
                    if score < 50:
                        continue

                    xa_range = abs(A_price - X_price)

                    if direction == "bullish":
                        prz_low = D_price * 0.995
                        prz_high = D_price * 1.005
                        stop_loss = D_price - xa_range * 0.1
                        target_1 = D_price + xa_range * 0.382
                        target_2 = D_price + xa_range * 0.618
                    else:
                        prz_low = D_price * 0.995
                        prz_high = D_price * 1.005
                        stop_loss = D_price + xa_range * 0.1
                        target_1 = D_price - xa_range * 0.382
                        target_2 = D_price - xa_range * 0.618

                    pattern = HarmonicPattern(
                        pattern_type=p_type,
                        direction=direction,
                        points={
                            "X": (pts[0]["index"], X_price),
                            "A": (pts[1]["index"], A_price),
                            "B": (pts[2]["index"], B_price),
                            "C": (pts[3]["index"], C_price),
                            "D": (pts[4]["index"], D_price),
                        },
                        ratios=ratios,
                        score=score,
                        prz_low=round(prz_low, 2),
                        prz_high=round(prz_high, 2),
                        stop_loss=round(stop_loss, 2),
                        target_1=round(target_1, 2),
                        target_2=round(target_2, 2),
                    )
                    detected.append(pattern)

        detected.sort(key=lambda x: x.score, reverse=True)
        return detected


# =============================================================================
# 3. 엘리어트 파동 분석
# =============================================================================

class WaveType(Enum):
    IMPULSE = "Impulse"
    CORRECTIVE = "Corrective"


@dataclass
class ElliottWave:
    """감지된 엘리어트 파동 정보"""
    wave_type: WaveType
    direction: str
    waves: dict
    current_wave: str
    confidence: float
    rules_passed: list[str]
    rules_failed: list[str]
    fib_targets: dict


class ElliottWaveAnalyzer:
    """
    엘리어트 파동 분석 엔진
    - 충격파 5파 규칙 검증
    - 슬라이딩 윈도우 방식 다중 피벗 조합 탐색 (Bug #2 수정)
    - 5파 완료 시 매도 시그널 (Bug #4 수정)
    """

    def __init__(self, zigzag_pct: float = 3.0, min_bars: int = 3):
        self.pivot_detector = PivotDetector(zigzag_pct=zigzag_pct, min_bars=min_bars)

    def _validate_impulse_rules(self, waves: dict) -> tuple[list[str], list[str]]:
        """충격파 3대 절대 규칙 검증"""
        passed = []
        failed = []

        if not all(k in waves for k in ["1", "2", "3", "4", "5"]):
            return passed, ["불완전한 파동 데이터"]

        w1_start = waves["1"][2]
        w1_end = waves["1"][3]
        w2_end = waves["2"][3]
        w3_end = waves["3"][3]
        w4_end = waves["4"][3]
        w5_end = waves["5"][3]

        w1_len = abs(w1_end - w1_start)
        w3_start = waves["3"][2]
        w3_len = abs(w3_end - w3_start)
        w5_start = waves["5"][2]
        w5_len = abs(w5_end - w5_start)

        is_uptrend = w1_end > w1_start

        # 규칙 1: 2파 되돌림 제한
        if is_uptrend:
            if w2_end > w1_start:
                passed.append("규칙1: 2파가 1파 시작점 위에서 종료")
            else:
                failed.append("규칙1: 2파가 1파 시작점 아래로 이탈")
        else:
            if w2_end < w1_start:
                passed.append("규칙1: 2파가 1파 시작점 아래에서 종료")
            else:
                failed.append("규칙1: 2파가 1파 시작점 위로 이탈")

        # 규칙 2: 3파 길이
        if w3_len >= w1_len and w3_len >= w5_len:
            passed.append(f"규칙2: 3파({w3_len:.0f})가 1파({w1_len:.0f}), 5파({w5_len:.0f})보다 김")
        elif w3_len < w1_len and w3_len < w5_len:
            failed.append(f"규칙2: 3파({w3_len:.0f})가 가장 짧음")
        else:
            passed.append(f"규칙2: 3파({w3_len:.0f})가 가장 짧지 않음")

        # 규칙 3: 4파-1파 중첩 금지
        if is_uptrend:
            if w4_end > w1_end:
                passed.append("규칙3: 4파 저점이 1파 고점 위")
            else:
                failed.append("규칙3: 4파가 1파 가격영역 침범")
        else:
            if w4_end < w1_end:
                passed.append("규칙3: 4파 고점이 1파 저점 아래")
            else:
                failed.append("규칙3: 4파가 1파 가격영역 침범")

        # 가이드라인
        w2_retrace = abs(w2_end - w1_end) / w1_len if w1_len > 0 else 0
        if 0.382 <= w2_retrace <= 0.786:
            passed.append(f"가이드: 2파 되돌림 {w2_retrace:.1%} (38.2~78.6%)")

        w3_extension = w3_len / w1_len if w1_len > 0 else 0
        if w3_extension >= 1.618:
            passed.append(f"가이드: 3파 확장 {w3_extension:.2f}x (>=1.618)")

        return passed, failed

    def _calculate_fib_targets(self, waves: dict) -> dict:
        """파동 기반 피보나치 목표가 계산"""
        targets = {}

        if "1" in waves and "2" in waves:
            w1_len = abs(waves["1"][3] - waves["1"][2])
            w2_end = waves["2"][3]
            is_up = waves["1"][3] > waves["1"][2]

            if is_up:
                targets["3파_1.000"] = round(w2_end + w1_len * 1.000, 2)
                targets["3파_1.618"] = round(w2_end + w1_len * 1.618, 2)
                targets["3파_2.618"] = round(w2_end + w1_len * 2.618, 2)
            else:
                targets["3파_1.000"] = round(w2_end - w1_len * 1.000, 2)
                targets["3파_1.618"] = round(w2_end - w1_len * 1.618, 2)
                targets["3파_2.618"] = round(w2_end - w1_len * 2.618, 2)

        if "1" in waves and "3" in waves and "4" in waves:
            w1_len = abs(waves["1"][3] - waves["1"][2])
            w4_end = waves["4"][3]
            is_up = waves["1"][3] > waves["1"][2]

            if is_up:
                targets["5파_0.618"] = round(w4_end + w1_len * 0.618, 2)
                targets["5파_1.000"] = round(w4_end + w1_len * 1.000, 2)
            else:
                targets["5파_0.618"] = round(w4_end - w1_len * 0.618, 2)
                targets["5파_1.000"] = round(w4_end - w1_len * 1.000, 2)

        return targets

    def _try_impulse(self, pivots_6: list[dict], prices: np.ndarray,
                     expected_start_type: str) -> ElliottWave | None:
        """6개 피벗으로 충격파 5파 구성 시도"""
        # 교대 패턴 검증
        for i, p in enumerate(pivots_6):
            if expected_start_type == "low":
                expected = "high" if i % 2 == 1 else "low"
            else:
                expected = "low" if i % 2 == 1 else "high"
            if p["type"] != expected:
                return None

        direction = "up" if expected_start_type == "low" else "down"

        waves = {}
        for w_num in range(5):
            waves[str(w_num + 1)] = (
                pivots_6[w_num]["index"], pivots_6[w_num + 1]["index"],
                pivots_6[w_num]["price"], pivots_6[w_num + 1]["price"],
            )

        passed, failed = self._validate_impulse_rules(waves)
        total_rules = len(passed) + len(failed)
        confidence = (len(passed) / total_rules * 100) if total_rules > 0 else 0

        # 절대 규칙 위반 1개 이하만 허용
        absolute_failures = sum(1 for f in failed if f.startswith("규칙"))
        if absolute_failures > 1:
            return None

        fib_targets = self._calculate_fib_targets(waves)

        # 현재 파동 위치 추정 (Bug #4: 5파 완료 감지 추가)
        last_price = prices[-1]
        last_pivot_idx = pivots_6[-1]["index"]

        if direction == "up":
            if last_pivot_idx >= len(prices) - 5:
                # 마지막 피벗이 최근이면 5파 완료 가능성
                if last_price < pivots_6[-1]["price"]:
                    current = "5_completed"
                else:
                    current = "5"
            elif last_price > pivots_6[4]["price"]:
                current = "5"
            elif last_price < pivots_6[3]["price"]:
                current = "4"
            else:
                current = "4~5 전환"
        else:
            if last_pivot_idx >= len(prices) - 5:
                if last_price > pivots_6[-1]["price"]:
                    current = "5_completed"
                else:
                    current = "5"
            else:
                current = "분석중"

        return ElliottWave(
            wave_type=WaveType.IMPULSE,
            direction=direction,
            waves=waves,
            current_wave=current,
            confidence=round(confidence, 1),
            rules_passed=passed,
            rules_failed=failed,
            fib_targets=fib_targets,
        )

    def analyze(self, df: pd.DataFrame, lookback: int = 200) -> ElliottWave | None:
        """
        엘리어트 파동 분석 (Bug #2: 슬라이딩 윈도우 탐색)
        여러 피벗 조합을 시도하여 최적의 파동 구조를 찾음
        """
        prices = df["close"].values[-lookback:]
        pivots = self.pivot_detector.find_pivots_zigzag(prices)

        if len(pivots) < 6:
            return None

        best_wave = None
        best_confidence = 0

        # 슬라이딩 윈도우: 뒤에서부터 6개씩 조합 시도
        for end_idx in range(len(pivots), 5, -1):
            start_idx = end_idx - 6
            if start_idx < 0:
                break

            candidate = pivots[start_idx:end_idx]

            # 상승 충격파 시도 (저-고-저-고-저-고)
            wave = self._try_impulse(candidate, prices, "low")
            if wave and wave.confidence > best_confidence:
                best_wave = wave
                best_confidence = wave.confidence

            # 하락 충격파 시도 (고-저-고-저-고-저)
            wave = self._try_impulse(candidate, prices, "high")
            if wave and wave.confidence > best_confidence:
                best_wave = wave
                best_confidence = wave.confidence

            # 신뢰도 80% 이상이면 탐색 중단
            if best_confidence >= 80:
                break

        return best_wave


# =============================================================================
# 4. 추세 각도 분석
# =============================================================================

@dataclass
class SlopeAnalysis:
    """각도 분석 결과"""
    raw_slope: float
    normalized_angle: float
    trend_strength: str
    signal: str
    description: str


class SlopeAnalyzer:
    """정규화된 추세 각도 분석기"""

    THRESHOLDS = {
        "weak": (0, 20),
        "moderate": (20, 45),
        "strong": (45, 70),
        "overshoot": (70, 90),
    }

    def __init__(self, lookback_periods: list[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 60]

    def calculate_normalized_angle(self, prices: np.ndarray,
                                    period: int = 20) -> float:
        if len(prices) < period:
            return 0.0

        segment = prices[-period:]
        price_start = segment[0]
        price_end = segment[-1]

        if price_start == 0:
            return 0.0

        norm_price_change = (price_end - price_start) / price_start
        norm_time_change = 1.0

        angle = np.degrees(np.arctan2(norm_price_change, norm_time_change))
        return round(angle, 2)

    def classify_trend(self, angle: float) -> tuple[str, str]:
        abs_angle = abs(angle)

        for strength, (low, high) in self.THRESHOLDS.items():
            if low <= abs_angle < high:
                if strength == "overshoot":
                    signal = "sell" if angle > 0 else "buy"
                else:
                    signal = "neutral"
                return strength, signal

        return "weak", "neutral"

    def analyze(self, df: pd.DataFrame) -> list[SlopeAnalysis]:
        prices = df["close"].values
        results = []

        for period in self.lookback_periods:
            if len(prices) < period:
                continue

            angle = self.calculate_normalized_angle(prices, period)
            strength, signal = self.classify_trend(angle)

            desc_parts = [f"{period}봉 기준"]
            if angle > 0:
                desc_parts.append(f"상승 {abs(angle):.1f}도")
            elif angle < 0:
                desc_parts.append(f"하락 {abs(angle):.1f}도")
            else:
                desc_parts.append("횡보")
            desc_parts.append(f"({strength})")

            if strength == "overshoot":
                desc_parts.append("[오버슈팅 경고]")

            segment = prices[-period:]
            raw_slope = (segment[-1] - segment[0]) / period

            results.append(SlopeAnalysis(
                raw_slope=round(raw_slope, 4),
                normalized_angle=angle,
                trend_strength=strength,
                signal=signal,
                description=" ".join(desc_parts),
            ))

        return results

    def detect_slope_divergence(self, prices: np.ndarray,
                                 short_period: int = 5,
                                 long_period: int = 20) -> str | None:
        if len(prices) < long_period:
            return None

        short_angle = self.calculate_normalized_angle(prices, short_period)
        long_angle = self.calculate_normalized_angle(prices, long_period)

        if long_angle > 10 and short_angle < -5:
            return "bearish_divergence"
        elif long_angle < -10 and short_angle > 5:
            return "bullish_divergence"
        return None


# =============================================================================
# 5. Approach B — 에너지 수렴 지표
# =============================================================================

class SqueezeDetector:
    """볼린저밴드 ⊂ 켈트너채널 스퀴즈 감지"""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_mult: float = 1.5):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        n = len(close)

        if n < max(self.bb_period, self.kc_period) + 5:
            return IndicatorResult("squeeze", "HOLD", 0, "데이터 부족", {})

        bb_mid = _sma_np(close, self.bb_period)
        bb_std_val = np.full(n, np.nan)
        for i in range(self.bb_period - 1, n):
            bb_std_val[i] = np.std(close[i - self.bb_period + 1: i + 1], ddof=1)
        bb_upper = bb_mid + self.bb_std * bb_std_val
        bb_lower = bb_mid - self.bb_std * bb_std_val

        kc_mid = _ema_np(close, self.kc_period)
        tr = np.abs(np.diff(close, prepend=close[0]))
        atr = _ema_np(tr, self.kc_period)
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr

        valid = ~(np.isnan(bb_upper) | np.isnan(kc_upper))
        squeeze = np.zeros(n, dtype=bool)
        squeeze[valid] = (bb_upper[valid] < kc_upper[valid]) & (bb_lower[valid] > kc_lower[valid])

        squeeze_bars = 0
        for i in range(n - 1, -1, -1):
            if squeeze[i]:
                squeeze_bars += 1
            else:
                break

        is_squeeze = bool(squeeze[-1]) if not np.isnan(bb_upper[-1]) else False

        if not np.isnan(bb_mid[-1]) and bb_mid[-1] > 0:
            bandwidth = (bb_upper[-1] - bb_lower[-1]) / bb_mid[-1] * 100
        else:
            bandwidth = 0

        lookback = min(100, n)
        recent_bw = []
        for i in range(n - lookback, n):
            if not np.isnan(bb_mid[i]) and bb_mid[i] > 0:
                recent_bw.append((bb_upper[i] - bb_lower[i]) / bb_mid[i] * 100)
        if recent_bw:
            squeeze_pct = round(
                (1 - np.searchsorted(sorted(recent_bw), bandwidth) / len(recent_bw)) * 100, 1)
        else:
            squeeze_pct = 0

        release_dir = None
        if not is_squeeze and squeeze_bars == 0:
            for i in range(n - 2, max(n - 6, 0), -1):
                if squeeze[i]:
                    release_dir = "up" if close[-1] > bb_mid[-1] else "down"
                    break

        vol_surge = False
        if volume is not None and len(volume) >= 20:
            vol_avg = np.mean(volume[-20:])
            if vol_avg > 0 and volume[-1] > vol_avg * 1.5:
                vol_surge = True

        sig = "HOLD"
        score = squeeze_pct

        if is_squeeze and squeeze_bars >= 3:
            score = squeeze_pct * 0.7
            detail = f"스퀴즈 진행중 ({squeeze_bars}봉) | 수축도 {squeeze_pct:.0f}%"
        elif release_dir and vol_surge:
            sig = "BUY" if release_dir == "up" else "SELL"
            score = min(squeeze_pct + 30, 100)
            detail = f"스퀴즈 해소 {release_dir} + 거래량 급증 | 수축도 {squeeze_pct:.0f}%"
        elif release_dir:
            sig = "BUY" if release_dir == "up" else "SELL"
            score = squeeze_pct * 0.6
            detail = f"스퀴즈 해소 {release_dir} (거래량 미확인) | 수축도 {squeeze_pct:.0f}%"
        else:
            detail = f"밴드폭 {bandwidth:.2f}% | 수축도 {squeeze_pct:.0f}%"

        return IndicatorResult("squeeze", sig, round(score, 1), detail, {
            "is_squeeze": is_squeeze, "squeeze_bars": squeeze_bars,
            "release_dir": release_dir, "bandwidth": round(bandwidth, 4),
            "squeeze_pct": round(squeeze_pct, 1), "vol_surge": vol_surge,
        })


class CurvatureAnalyzer:
    """EMA 곡률(2차도함수) 기반 변곡점 감지"""

    def __init__(self, ema_span: int = 20, window: int = 10):
        self.ema_span = ema_span
        self.window = window

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        prices = df["close"].values.astype(float)

        if len(prices) < self.ema_span + self.window + 5:
            return IndicatorResult("curvature", "HOLD", 0, "데이터 부족", {})

        smoothed = _ema_np(prices, self.ema_span)
        f_prime = np.gradient(smoothed, 1.0)
        f_double = np.gradient(f_prime, 1.0)

        denom = (1 + f_prime ** 2) ** 1.5
        kappa = np.where(denom != 0, f_double / denom, 0)

        recent_kappa = kappa[-self.window:]
        inflection_type = None
        inflection_bars_ago = None

        for i in range(len(recent_kappa) - 1, 0, -1):
            if recent_kappa[i - 1] < 0 and recent_kappa[i] >= 0:
                inflection_type = "bullish"
                inflection_bars_ago = len(recent_kappa) - 1 - i
                break
            elif recent_kappa[i - 1] > 0 and recent_kappa[i] <= 0:
                inflection_type = "bearish"
                inflection_bars_ago = len(recent_kappa) - 1 - i
                break

        cur_kappa = float(kappa[-1])
        cur_slope = float(f_prime[-1])

        kappa_range = np.max(np.abs(kappa[-50:])) if len(kappa) >= 50 else np.max(np.abs(kappa))
        kappa_strength = min(abs(cur_kappa) / kappa_range * 100, 100) if kappa_range > 0 else 0

        sig = "HOLD"
        score = 0.0

        if inflection_type == "bullish" and inflection_bars_ago is not None and inflection_bars_ago <= 3:
            sig = "BUY"
            score = min(kappa_strength + 20, 100)
            detail = f"곡률 상승 변곡점 ({inflection_bars_ago}봉 전) | k={cur_kappa:.6f} | 기울기={cur_slope:.2f}"
        elif inflection_type == "bearish" and inflection_bars_ago is not None and inflection_bars_ago <= 3:
            sig = "SELL"
            score = min(kappa_strength + 20, 100)
            detail = f"곡률 하락 변곡점 ({inflection_bars_ago}봉 전) | k={cur_kappa:.6f} | 기울기={cur_slope:.2f}"
        else:
            curve_dir = "상승곡선" if cur_kappa > 0 else "하락곡선" if cur_kappa < 0 else "직선"
            detail = f"곡률 {curve_dir} | k={cur_kappa:.6f} | 변곡점 미감지"
            score = kappa_strength * 0.3

        return IndicatorResult("curvature", sig, round(score, 1), detail, {
            "curvature": cur_kappa, "slope": round(cur_slope, 4),
            "inflection_type": inflection_type,
            "inflection_bars_ago": inflection_bars_ago,
            "kappa_strength": round(kappa_strength, 1),
        })


class SlopeMomentumAnalyzer:
    """기울기 가속도(모멘텀) 측정"""

    def __init__(self, slope_period: int = 20, momentum_period: int = 5):
        self.slope_period = slope_period
        self.momentum_period = momentum_period

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        prices = df["close"].values.astype(float)

        if len(prices) < self.slope_period + self.momentum_period + 5:
            return IndicatorResult("slope_mom", "HOLD", 0, "데이터 부족", {})

        slopes = _linreg_slope_np(prices, self.slope_period)

        valid_mask = ~np.isnan(slopes)
        angles = np.full_like(slopes, np.nan)
        for i in range(len(slopes)):
            if valid_mask[i] and prices[i] != 0:
                norm_slope = slopes[i] * self.slope_period / prices[i]
                angles[i] = np.degrees(np.arctan(norm_slope))

        angle_mom = np.full_like(angles, np.nan)
        for i in range(self.momentum_period, len(angles)):
            if not np.isnan(angles[i]) and not np.isnan(angles[i - self.momentum_period]):
                angle_mom[i] = angles[i] - angles[i - self.momentum_period]

        cur_angle = float(angles[-1]) if not np.isnan(angles[-1]) else 0
        cur_mom = float(angle_mom[-1]) if not np.isnan(angle_mom[-1]) else 0
        prev_angle = float(angles[-2]) if len(angles) > 1 and not np.isnan(angles[-2]) else 0

        zero_cross_up = prev_angle < 0 and cur_angle >= 0
        zero_cross_down = prev_angle > 0 and cur_angle <= 0
        accelerating = cur_mom > 5

        sig = "HOLD"
        score = 0.0

        if zero_cross_up and (cur_angle >= 30 or accelerating):
            sig = "BUY"
            score = min(abs(cur_mom) * 3 + 40, 100)
            detail = f"기울기 양전환 + 가속 | 각도 {cur_angle:.1f} | 모멘텀 +{cur_mom:.1f}"
        elif zero_cross_up:
            sig = "BUY"
            score = min(abs(cur_mom) * 2 + 20, 100)
            detail = f"기울기 양전환 | 각도 {cur_angle:.1f} | 모멘텀 +{cur_mom:.1f}"
        elif zero_cross_down and (cur_angle <= -30 or cur_mom < -5):
            sig = "SELL"
            score = min(abs(cur_mom) * 3 + 40, 100)
            detail = f"기울기 음전환 + 가속 | 각도 {cur_angle:.1f} | 모멘텀 {cur_mom:.1f}"
        elif zero_cross_down:
            sig = "SELL"
            score = min(abs(cur_mom) * 2 + 20, 100)
            detail = f"기울기 음전환 | 각도 {cur_angle:.1f} | 모멘텀 {cur_mom:.1f}"
        else:
            direction = "상승" if cur_angle > 0 else "하락" if cur_angle < 0 else "횡보"
            detail = f"기울기 {direction} {abs(cur_angle):.1f} | 모멘텀 {cur_mom:+.1f}"
            score = abs(cur_mom) * 1.5

        return IndicatorResult("slope_mom", sig, round(min(score, 100), 1), detail, {
            "angle": round(cur_angle, 2), "momentum": round(cur_mom, 2),
            "zero_cross": "up" if zero_cross_up else "down" if zero_cross_down else None,
            "accelerating": accelerating,
        })


class ConfluenceAnalyzer:
    """피보나치 되돌림 합치구간 감지"""

    FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618]

    def __init__(self, zigzag_pct: float = 5.0, bin_pct: float = 0.5,
                 min_count: int = 3):
        self.pivot_detector = PivotDetector(zigzag_pct=zigzag_pct, min_bars=5)
        self.bin_pct = bin_pct
        self.min_count = min_count

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        prices = df["close"].values.astype(float)

        if len(prices) < 50:
            return IndicatorResult("confluence", "HOLD", 0, "데이터 부족", {})

        pivots = self.pivot_detector.find_pivots_zigzag(prices)
        if len(pivots) < 3:
            return IndicatorResult("confluence", "HOLD", 0, "피벗 부족", {})

        fib_prices = []
        for i in range(len(pivots) - 1):
            p1 = pivots[i]["price"]
            p2 = pivots[i + 1]["price"]
            swing = abs(p2 - p1)
            for fib in self.FIB_LEVELS:
                if pivots[i + 1]["type"] == "low":
                    fib_prices.append(p2 + swing * fib)
                else:
                    fib_prices.append(p2 - swing * fib)

        if not fib_prices:
            return IndicatorResult("confluence", "HOLD", 0, "피보나치 계산 불가", {})

        fib_arr = np.array(fib_prices)
        fib_arr = fib_arr[fib_arr > 0]

        cur_price = prices[-1]
        bin_size = cur_price * self.bin_pct / 100
        if bin_size == 0:
            return IndicatorResult("confluence", "HOLD", 0, "계산 오류", {})

        price_min = np.min(fib_arr) * 0.95
        price_max = np.max(fib_arr) * 1.05
        bins = np.arange(price_min, price_max, bin_size)
        hist, _ = np.histogram(fib_arr, bins=bins)

        zones = []
        for i in range(len(hist)):
            if hist[i] >= self.min_count:
                zone_price = (bins[i] + bins[i + 1]) / 2
                zones.append({
                    "price": round(zone_price, 0),
                    "count": int(hist[i]),
                    "distance_pct": round((zone_price - cur_price) / cur_price * 100, 2),
                    "type": "지지" if zone_price < cur_price else "저항",
                })
        zones.sort(key=lambda z: abs(z["distance_pct"]))
        nearby = [z for z in zones if abs(z["distance_pct"]) <= 2.0]

        sig = "HOLD"
        score = 0.0

        if nearby:
            nearest = nearby[0]
            confluence_count = sum(z["count"] for z in nearby)
            score = min(confluence_count * 15, 100)
            if nearest["type"] == "지지" and nearest["distance_pct"] > -1.0:
                sig = "BUY"
                detail = f"피보나치 합치 지지구간 | {nearest['price']:,.0f} ({nearest['count']}중첩) | 거리 {nearest['distance_pct']:+.1f}%"
            elif nearest["type"] == "저항" and nearest["distance_pct"] < 1.0:
                sig = "SELL"
                detail = f"피보나치 합치 저항구간 | {nearest['price']:,.0f} ({nearest['count']}중첩) | 거리 {nearest['distance_pct']:+.1f}%"
            else:
                detail = f"합치 구간 감지 | 최근접 {nearest['price']:,.0f} ({nearest['count']}중첩)"
        else:
            detail = f"현재가 근처 합치 구간 없음 (전체 {len(zones)}개 구간)"

        return IndicatorResult("confluence", sig, round(score, 1), detail, {
            "zones": zones[:10], "nearby": nearby, "total_zones": len(zones),
        })


# =============================================================================
# 6. Approach C — 극단 회귀 지표
# =============================================================================

class MeanReversionAnalyzer:
    """다중기간 이격도 + 극단 회귀 시그널"""

    def __init__(self, periods: tuple = (20, 60, 120)):
        self.periods = periods

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        prices = df["close"].values.astype(float)

        if len(prices) < max(self.periods) + 5:
            return IndicatorResult("mean_rev", "HOLD", 0, "데이터 부족", {})

        cur_price = prices[-1]
        deviations = {}
        for p in self.periods:
            if len(prices) < p:
                continue
            ma = _ema_np(prices, p)
            if ma[-1] == 0:
                continue
            deviations[f"ema_{p}"] = round((cur_price - ma[-1]) / ma[-1] * 100, 2)

        if not deviations:
            return IndicatorResult("mean_rev", "HOLD", 0, "이격도 계산 불가", {})

        primary_key = f"ema_{max(self.periods)}"
        if primary_key not in deviations:
            primary_key = list(deviations.keys())[-1]
        primary_dev = deviations[primary_key]

        longest_p = max(p for p in self.periods if len(prices) >= p)
        ma_long = _ema_np(prices, longest_p)
        hist_devs = []
        for i in range(longest_p, len(prices)):
            if ma_long[i] > 0:
                hist_devs.append((prices[i] - ma_long[i]) / ma_long[i] * 100)

        percentile = round(
            np.searchsorted(sorted(hist_devs), primary_dev) / len(hist_devs) * 100, 1
        ) if hist_devs else 50

        extreme_type = None
        if primary_dev <= -25:
            extreme_type = "extreme_oversold"
        elif primary_dev <= -15:
            extreme_type = "oversold"
        elif primary_dev >= 25:
            extreme_type = "extreme_overbought"
        elif primary_dev >= 15:
            extreme_type = "overbought"

        recovery_signal = False
        if len(prices) >= 5:
            recent_ma = _ema_np(prices, min(20, longest_p))
            dev_3ago = (prices[-3] - recent_ma[-3]) / recent_ma[-3] * 100 if recent_ma[-3] > 0 else 0
            dev_now = (prices[-1] - recent_ma[-1]) / recent_ma[-1] * 100 if recent_ma[-1] > 0 else 0
            if extreme_type in ("extreme_oversold", "oversold") and dev_now > dev_3ago or extreme_type in ("extreme_overbought", "overbought") and dev_now < dev_3ago:
                recovery_signal = True

        short_dev = deviations.get("ema_20", 0)
        sig = "HOLD"
        score = 0.0

        if extreme_type == "extreme_oversold" and recovery_signal:
            sig = "BUY"
            score = min(abs(primary_dev) * 2, 100)
            detail = f"극단 과매도 + 회복 시작 | 이격 {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
        elif extreme_type == "extreme_oversold":
            sig = "BUY"
            score = min(abs(primary_dev) * 1.5, 85)
            detail = f"극단 과매도 구간 | 이격 {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
        elif extreme_type == "oversold" and recovery_signal:
            sig = "BUY"
            score = min(abs(primary_dev) * 1.8, 80)
            detail = f"과매도 회복 시그널 | 이격 {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
        elif extreme_type == "extreme_overbought" and recovery_signal:
            sig = "SELL"
            score = min(abs(primary_dev) * 2, 100)
            detail = f"극단 과매수 + 반락 시작 | 이격 {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
        elif extreme_type == "extreme_overbought":
            sig = "SELL"
            score = min(abs(primary_dev) * 1.5, 85)
            detail = f"극단 과매수 구간 | 이격 {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
        elif extreme_type == "overbought" and recovery_signal:
            sig = "SELL"
            score = min(abs(primary_dev) * 1.8, 80)
            detail = f"과매수 반락 시그널 | 이격 {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
        else:
            detail = f"이격도 정상 범위 | {primary_dev:+.1f}% | 백분위 {percentile:.0f}%"
            score = min(abs(primary_dev), 30)

        return IndicatorResult("mean_rev", sig, round(score, 1), detail, {
            "deviations": deviations, "percentile": percentile,
            "extreme_type": extreme_type, "recovery_signal": recovery_signal,
            "primary_dev": round(primary_dev, 2), "short_dev": round(short_dev, 2),
        })


class VolumeClimaxDetector:
    """거래량 클라이맥스(투매/매집) 감지"""

    def __init__(self, spike_mult: float = 3.0, exhaust_bars: int = 5,
                 avg_period: int = 20):
        self.spike_mult = spike_mult
        self.exhaust_bars = exhaust_bars
        self.avg_period = avg_period

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        if "volume" not in df.columns:
            return IndicatorResult("vol_climax", "HOLD", 0, "거래량 데이터 없음", {})

        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)
        n = len(close)

        if n < self.avg_period + self.exhaust_bars + 5:
            return IndicatorResult("vol_climax", "HOLD", 0, "데이터 부족", {})

        vol_ma = _sma_np(volume, self.avg_period)

        pct_change = np.zeros(n)
        for i in range(1, n):
            if close[i - 1] > 0:
                pct_change[i] = (close[i] - close[i - 1]) / close[i - 1] * 100

        best_climax = None
        best_score = 0
        search_start = max(self.avg_period, n - 30)

        for i in range(search_start, n):
            if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
                continue
            spike_ratio = volume[i] / vol_ma[i]
            if spike_ratio < self.spike_mult:
                continue

            price_chg = pct_change[i]
            if price_chg < -3:
                climax_type = "selling"
            elif price_chg > 3:
                climax_type = "buying"
            else:
                continue

            bars_after = min(self.exhaust_bars, n - 1 - i)
            exhaust_count = 0
            if bars_after >= 2:
                for j in range(1, bars_after + 1):
                    if volume[i + j] < volume[i + j - 1]:
                        exhaust_count += 1

            price_stabilized = False
            if bars_after >= 2:
                if climax_type == "selling":
                    price_stabilized = close[min(i + bars_after, n - 1)] >= close[i] * 0.97
                else:
                    price_stabilized = close[min(i + bars_after, n - 1)] <= close[i] * 1.03

            exhaust_ratio = exhaust_count / max(bars_after, 1)
            c_score = min(spike_ratio / self.spike_mult * 30, 40)
            c_score += exhaust_ratio * 30
            c_score += 20 if price_stabilized else 0
            c_score += min(abs(price_chg) / 5 * 10, 15)

            bars_since = n - 1 - i
            if c_score > best_score:
                best_score = c_score
                best_climax = {
                    "climax_type": climax_type, "spike_ratio": round(spike_ratio, 2),
                    "price_change": round(price_chg, 2),
                    "exhaust_count": exhaust_count, "exhaust_bars": bars_after,
                    "exhaust_ratio": round(exhaust_ratio, 2),
                    "price_stabilized": price_stabilized,
                    "bars_since": bars_since, "score": round(min(c_score, 100), 1),
                }

        if best_climax and best_climax["score"] >= 40:
            c = best_climax
            score = c["score"]
            if c["climax_type"] == "selling":
                sig = "BUY"
                if c["bars_since"] <= 3 and c["exhaust_ratio"] >= 0.5:
                    detail = f"투매 클라이맥스 | 거래량 {c['spike_ratio']:.1f}x | {c['bars_since']}봉 전 | 소진 {c['exhaust_ratio']:.0%}"
                    score = min(score + 15, 100)
                elif c["bars_since"] <= 5:
                    detail = f"투매 감지 | 거래량 {c['spike_ratio']:.1f}x | {c['bars_since']}봉 전"
                else:
                    detail = f"투매 흔적 ({c['bars_since']}봉 전) | 거래량 {c['spike_ratio']:.1f}x"
                    score *= 0.6
            else:
                sig = "SELL"
                if c["bars_since"] <= 3:
                    detail = f"매수 클라이맥스 | 거래량 {c['spike_ratio']:.1f}x | {c['bars_since']}봉 전"
                else:
                    detail = f"매수 클라이맥스 흔적 ({c['bars_since']}봉 전) | 거래량 {c['spike_ratio']:.1f}x"
                    score *= 0.6

            return IndicatorResult("vol_climax", sig, round(min(score, 100), 1), detail, c)

        return IndicatorResult("vol_climax", "HOLD", 0, "거래량 클라이맥스 미감지", {})


class BandBreachDetector:
    """볼린저밴드 이탈->복귀 반전 시그널"""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 lookback: int = 25):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> IndicatorResult:
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        n = len(close)

        if n < self.bb_period + self.lookback + 5:
            return IndicatorResult("band_breach", "HOLD", 0, "데이터 부족", {})

        bb_mid = _sma_np(close, self.bb_period)
        bb_std_val = np.full(n, np.nan)
        for i in range(self.bb_period - 1, n):
            bb_std_val[i] = np.std(close[i - self.bb_period + 1: i + 1], ddof=1)
        bb_upper = bb_mid + self.bb_std * bb_std_val
        bb_lower = bb_mid - self.bb_std * bb_std_val

        if not np.isnan(bb_mid[-1]) and bb_mid[-1] > 0:
            bandwidth = (bb_upper[-1] - bb_lower[-1]) / bb_mid[-1] * 100
        else:
            bandwidth = 0

        hist_bw = []
        for i in range(self.bb_period - 1, n):
            if not np.isnan(bb_mid[i]) and bb_mid[i] > 0:
                hist_bw.append((bb_upper[i] - bb_lower[i]) / bb_mid[i] * 100)
        bw_percentile = round(
            np.searchsorted(sorted(hist_bw), bandwidth) / len(hist_bw) * 100, 1
        ) if hist_bw else 50

        breach_type = None
        breach_depth = 0.0
        recovery_bars = None
        breach_bar = None

        for i in range(n - self.lookback, n):
            if i < self.bb_period or np.isnan(bb_lower[i]):
                continue

            if close[i] < bb_lower[i]:
                depth = (bb_lower[i] - close[i]) / bb_lower[i] * 100 if bb_lower[i] > 0 else 0
                if depth > breach_depth:
                    breach_depth = depth
                    breach_bar = i
                    recovered = False
                    for j in range(i + 1, n):
                        if not np.isnan(bb_lower[j]) and close[j] > bb_lower[j]:
                            recovered = True
                            recovery_bars = j - i
                            break
                    breach_type = "lower_recovery" if recovered else "lower_breach"

            elif close[i] > bb_upper[i]:
                depth = (close[i] - bb_upper[i]) / bb_upper[i] * 100 if bb_upper[i] > 0 else 0
                if depth > breach_depth:
                    breach_depth = depth
                    breach_bar = i
                    recovered = False
                    for j in range(i + 1, n):
                        if not np.isnan(bb_upper[j]) and close[j] < bb_upper[j]:
                            recovered = True
                            recovery_bars = j - i
                            break
                    breach_type = "upper_recovery" if recovered else "upper_breach"

        vol_confirmed = False
        if volume is not None and breach_bar is not None and breach_bar >= 20:
            vol_avg = np.mean(volume[max(0, breach_bar - 20): breach_bar])
            if vol_avg > 0 and volume[breach_bar] > vol_avg * 2.0:
                vol_confirmed = True

        sig = "HOLD"
        score = 0.0

        if breach_type == "lower_recovery":
            sig = "BUY"
            score = min(breach_depth * 8 + (30 if vol_confirmed else 0) + bw_percentile * 0.3, 100)
            bars_info = f"{recovery_bars}봉만에 복귀" if recovery_bars else ""
            vol_info = " + 거래량 확인" if vol_confirmed else ""
            detail = f"밴드 하단 이탈->복귀 | 이탈깊이 {breach_depth:.1f}%{vol_info} | {bars_info} | 밴드폭 {bw_percentile:.0f}%ile"
        elif breach_type == "lower_breach":
            score = min(breach_depth * 5, 60)
            detail = f"밴드 하단 이탈 중 | 이탈깊이 {breach_depth:.1f}% | 복귀 대기"
        elif breach_type == "upper_recovery":
            sig = "SELL"
            score = min(breach_depth * 8 + (30 if vol_confirmed else 0) + bw_percentile * 0.3, 100)
            bars_info = f"{recovery_bars}봉만에 복귀" if recovery_bars else ""
            vol_info = " + 거래량 확인" if vol_confirmed else ""
            detail = f"밴드 상단 이탈->복귀 | 이탈깊이 {breach_depth:.1f}%{vol_info} | {bars_info}"
        elif breach_type == "upper_breach":
            score = min(breach_depth * 5, 60)
            detail = f"밴드 상단 이탈 중 | 이탈깊이 {breach_depth:.1f}%"
        else:
            detail = f"밴드 이탈 미감지 | 밴드폭 {bw_percentile:.0f}%ile"

        return IndicatorResult("band_breach", sig, round(min(score, 100), 1), detail, {
            "breach_type": breach_type, "breach_depth": round(breach_depth, 2),
            "recovery_bars": recovery_bars, "bandwidth": round(bandwidth, 4),
            "bw_percentile": bw_percentile, "vol_confirmed": vol_confirmed,
        })


# =============================================================================
# 7. 통합 시그널 엔진
# =============================================================================

@dataclass
class GeometricSignal:
    """기하학적 분석 통합 시그널"""
    timestamp: str
    ticker: str
    action: str
    confidence: float
    harmonic: dict | None
    elliott: dict | None
    slope: dict | None
    reasoning: list[str]


class GeometricQuantEngine:
    """
    기하학적 퀀트 매매 통합 엔진 v2.0

    10지표 × 7프로파일 체제:
      Approach A: harmonic, elliott, slope (구조적 패턴)
      Approach B: squeeze, curvature, slope_mom, confluence (에너지 수렴)
      Approach C: mean_rev, vol_climax, band_breach (극단 회귀)
    """

    APPROACH_A = ("harmonic", "elliott", "slope")
    APPROACH_B = ("squeeze", "curvature", "slope_mom", "confluence")
    APPROACH_C = ("mean_rev", "vol_climax", "band_breach")

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Approach A: 기존 지표 (버그 수정 코드 유지)
        self.harmonic = HarmonicDetector(
            tolerance=self.config.get("harmonic_tolerance", 0.03),
            zigzag_pct=self.config.get("zigzag_pct", 3.0),
            min_bars=self.config.get("min_bars", 5),
        )
        self.elliott = ElliottWaveAnalyzer(
            zigzag_pct=self.config.get("elliott_zigzag_pct",
                                       self.config.get("zigzag_pct", 3.0)),
            min_bars=self.config.get("elliott_min_bars", 3),
        )
        self.slope_analyzer = SlopeAnalyzer(
            lookback_periods=self.config.get("slope_periods", [5, 10, 20, 60])
        )

        # Approach B: 에너지 수렴 지표
        sq_cfg = self.config.get("squeeze", {})
        self.squeeze = SqueezeDetector(
            bb_period=sq_cfg.get("bb_period", 20),
            bb_std=sq_cfg.get("bb_std", 2.0),
            kc_period=sq_cfg.get("kc_period", 20),
            kc_mult=sq_cfg.get("kc_mult", 1.5),
        )
        cv_cfg = self.config.get("curvature", {})
        self.curvature = CurvatureAnalyzer(
            ema_span=cv_cfg.get("ema_span", 20),
            window=cv_cfg.get("window", 10),
        )
        self.slope_mom = SlopeMomentumAnalyzer(
            slope_period=self.config.get("slope_mom_period", 20),
            momentum_period=self.config.get("slope_mom_momentum", 5),
        )
        cf_cfg = self.config.get("confluence", {})
        self.confluence = ConfluenceAnalyzer(
            zigzag_pct=cf_cfg.get("zigzag_pct", 5.0),
            bin_pct=cf_cfg.get("bin_pct", 0.5),
            min_count=cf_cfg.get("min_count", 3),
        )

        # Approach C: 극단 회귀 지표
        mr_cfg = self.config.get("mean_reversion", {})
        self.mean_rev = MeanReversionAnalyzer(
            periods=tuple(mr_cfg.get("periods", [20, 60, 120])),
        )
        vc_cfg = self.config.get("volume_climax", {})
        self.vol_climax = VolumeClimaxDetector(
            spike_mult=vc_cfg.get("spike_mult", 3.0),
            exhaust_bars=vc_cfg.get("exhaust_bars", 5),
        )
        bb_cfg = self.config.get("band_breach", {})
        self.band_breach = BandBreachDetector(
            bb_period=bb_cfg.get("bb_period", 20),
            bb_std=bb_cfg.get("bb_std", 2.0),
            lookback=bb_cfg.get("lookback", 25),
        )

        # 프로파일 기반 가중치
        self.profile = self.config.get("profile", "default")
        self.weights = WEIGHT_PROFILES.get(
            self.profile, WEIGHT_PROFILES["default"]
        ).copy()

        # 하위호환: 기존 w_harmonic 등 개별 설정이 있으면 오버라이드
        for key in ("harmonic", "elliott", "slope"):
            legacy_key = f"w_{key}"
            if legacy_key in self.config:
                self.weights[key] = self.config[legacy_key]

    def _run_approach_a(self, df: pd.DataFrame, ticker: str
                        ) -> tuple[dict, dict, dict, list[str], dict[str, float]]:
        """Approach A: 구조적 패턴 (harmonic, elliott, slope)"""
        reasoning = []
        scores = {"buy": 0.0, "sell": 0.0}
        harmonic_result = None
        elliott_result = None
        slope_result = None
        lookback = self.config.get("lookback", 200)

        # 하모닉 패턴
        try:
            patterns = self.harmonic.detect_patterns(df, lookback=lookback)
        except Exception as e:
            logger.debug("하모닉 감지 실패 %s: %s", ticker, e)
            patterns = []

        if patterns:
            best = patterns[0]
            harmonic_result = {
                "pattern": best.pattern_type.value, "direction": best.direction,
                "score": best.score,
                "prz": f"{best.prz_low} ~ {best.prz_high}",
                "target_1": best.target_1, "target_2": best.target_2,
                "stop_loss": best.stop_loss,
            }
            w = self.weights.get("harmonic", 0)
            if best.direction == "bullish" and best.score >= 70:
                scores["buy"] += w * (best.score / 100)
                reasoning.append(f"하모닉 {best.pattern_type.value} Bullish ({best.score}%)")
            elif best.direction == "bearish" and best.score >= 70:
                scores["sell"] += w * (best.score / 100)
                reasoning.append(f"하모닉 {best.pattern_type.value} Bearish ({best.score}%)")
            else:
                reasoning.append(f"하모닉 {best.pattern_type.value} (신뢰도 부족)")
        else:
            reasoning.append("하모닉 패턴 미감지")

        # 엘리어트 파동
        try:
            wave = self.elliott.analyze(df, lookback=lookback)
        except Exception as e:
            logger.debug("엘리어트 분석 실패 %s: %s", ticker, e)
            wave = None

        if wave:
            elliott_result = {
                "type": wave.wave_type.value, "direction": wave.direction,
                "current_wave": wave.current_wave, "confidence": wave.confidence,
                "targets": wave.fib_targets,
                "rules_passed": wave.rules_passed, "rules_failed": wave.rules_failed,
            }
            w = self.weights.get("elliott", 0)
            if wave.direction == "up" and wave.current_wave in ["2", "4", "4~5 전환"]:
                scores["buy"] += w * (wave.confidence / 100)
                reasoning.append(f"충격파 {wave.current_wave}파 매수구간 ({wave.confidence}%)")
            elif wave.direction == "down" and wave.current_wave in ["2", "4"]:
                scores["sell"] += w * (wave.confidence / 100)
                reasoning.append(f"하락 {wave.current_wave}파 매도구간 ({wave.confidence}%)")
            elif wave.current_wave == "5_completed":
                scores["sell"] += w * (wave.confidence / 100) * 0.8
                reasoning.append(f"5파 완료 추세전환경고 ({wave.confidence}%)")
            elif wave.direction == "up" and wave.current_wave == "5":
                scores["sell"] += w * (wave.confidence / 100) * 0.4
                reasoning.append(f"상승 5파 진행 추세말기 ({wave.confidence}%)")
            else:
                reasoning.append(f"엘리어트 {wave.direction} {wave.current_wave}파")
        else:
            reasoning.append("엘리어트 구조 미감지")

        # 추세 각도
        try:
            slopes = self.slope_analyzer.analyze(df)
        except Exception as e:
            logger.debug("각도 분석 실패 %s: %s", ticker, e)
            slopes = []

        if slopes:
            primary = slopes[-1] if len(slopes) > 1 else slopes[0]
            slope_result = {
                "angle": primary.normalized_angle, "strength": primary.trend_strength,
                "all_periods": [
                    {"period": p, "angle": s.normalized_angle, "strength": s.trend_strength}
                    for p, s in zip(self.slope_analyzer.lookback_periods, slopes)
                ],
            }
            w = self.weights.get("slope", 0)
            if primary.trend_strength == "overshoot":
                if primary.normalized_angle > 0:
                    scores["sell"] += w
                else:
                    scores["buy"] += w
                reasoning.append(f"추세 {primary.trend_strength} {primary.normalized_angle:.1f}도")
            else:
                reasoning.append(f"추세 {primary.normalized_angle:.1f}도 ({primary.trend_strength})")

            try:
                div = self.slope_analyzer.detect_slope_divergence(df["close"].values)
            except Exception:
                div = None
            if div == "bullish_divergence":
                scores["buy"] += self.weights.get("slope", 0) * 0.3
                reasoning.append("기울기 상승 다이버전스")
            elif div == "bearish_divergence":
                scores["sell"] += self.weights.get("slope", 0) * 0.3
                reasoning.append("기울기 하락 다이버전스")

        return harmonic_result, elliott_result, slope_result, reasoning, scores

    def _run_approach_bc(self, df: pd.DataFrame, ticker: str
                         ) -> tuple[dict[str, IndicatorResult], list[str], dict[str, float]]:
        """Approach B+C: 에너지 수렴 + 극단 회귀 (7개 신규 지표)"""
        reasoning = []
        scores = {"buy": 0.0, "sell": 0.0}
        indicators = {}

        analyzers = {
            "squeeze": self.squeeze,
            "curvature": self.curvature,
            "slope_mom": self.slope_mom,
            "confluence": self.confluence,
            "mean_rev": self.mean_rev,
            "vol_climax": self.vol_climax,
            "band_breach": self.band_breach,
        }

        for name, analyzer in analyzers.items():
            try:
                result = analyzer.analyze(df)
            except Exception as e:
                logger.debug("%s 분석 실패 %s: %s", name, ticker, e)
                result = IndicatorResult(name, "HOLD", 0, f"{name} 오류", {})

            indicators[name] = result
            w = self.weights.get(name, 0)

            if result.signal == "BUY" and result.score > 0:
                scores["buy"] += w * (result.score / 100)
                reasoning.append(result.detail)
            elif result.signal == "SELL" and result.score > 0:
                scores["sell"] += w * (result.score / 100)
                reasoning.append(result.detail)
            elif result.score > 30:
                reasoning.append(result.detail)

        return indicators, reasoning, scores

    def generate_signal(self, df: pd.DataFrame,
                        ticker: str = "UNKNOWN") -> GeometricSignal:
        """10지표 통합 시그널 생성"""
        # Approach A
        harmonic_r, elliott_r, slope_r, reason_a, scores_a = \
            self._run_approach_a(df, ticker)

        # Approach B+C
        indicators_bc, reason_bc, scores_bc = \
            self._run_approach_bc(df, ticker)

        # 점수 합산
        buy_score = scores_a["buy"] + scores_bc["buy"]
        sell_score = scores_a["sell"] + scores_bc["sell"]
        reasoning = reason_a + reason_bc

        # 최종 판단 (Bug #1, #3 유지)
        max_possible = sum(self.weights.values())
        if buy_score == 0 and sell_score == 0:
            action = "HOLD"
            confidence = 0.0
        elif buy_score > sell_score:
            action = "BUY"
            confidence = round(buy_score / max_possible * 100, 1)
        elif sell_score > buy_score:
            action = "SELL"
            confidence = round(sell_score / max_possible * 100, 1)
        else:
            action = "HOLD"
            confidence = 0.0

        min_conf = self.config.get("min_confidence", 30)
        if confidence < min_conf:
            action = "HOLD"
            reasoning.append(f"종합 신뢰도 {confidence}% -- 관망 권장")
        else:
            reasoning.append(f"종합 판단: {action} (신뢰도 {confidence}%)")

        # 지표 결과를 GeometricSignal에 저장
        self._last_indicators = indicators_bc

        return GeometricSignal(
            timestamp=str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else str(len(df)),
            ticker=ticker,
            action=action,
            confidence=confidence,
            harmonic=harmonic_r,
            elliott=elliott_r,
            slope=slope_r,
            reasoning=reasoning,
        )

    def get_indicator_votes(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> dict:
        """
        v5.0: 10지표 개별 투표 결과 반환 (Diversity 계산용).

        Returns:
            dict: {indicator_name: {"signal": str, "score": float, ...}}
        """
        signal = self.generate_signal(df, ticker)
        votes = {}

        # Approach A 결과
        if signal.harmonic:
            direction = signal.harmonic.get("direction", "neutral")
            votes["harmonic"] = {
                "signal": "BUY" if direction == "bullish" else "SELL" if direction == "bearish" else "HOLD",
                "score": signal.harmonic.get("score", 0),
            }
        else:
            votes["harmonic"] = {"signal": "HOLD", "score": 0}

        if signal.elliott:
            wave = signal.elliott.get("current_wave", "")
            direction = signal.elliott.get("direction", "")
            if direction == "up" and wave in ("2", "4", "4~5 전환"):
                sig = "BUY"
            elif wave == "5_completed":
                sig = "SELL"
            else:
                sig = "HOLD"
            votes["elliott"] = {
                "signal": sig,
                "score": signal.elliott.get("confidence", 0),
            }
        else:
            votes["elliott"] = {"signal": "HOLD", "score": 0}

        if signal.slope:
            angle = signal.slope.get("angle", 0)
            strength = signal.slope.get("strength", "weak")
            if strength == "overshoot":
                sig = "SELL" if angle > 0 else "BUY"
            else:
                sig = "HOLD"
            votes["slope"] = {"signal": sig, "score": abs(angle)}
        else:
            votes["slope"] = {"signal": "HOLD", "score": 0}

        # Approach B+C 결과 (직접 접근)
        for name, result in getattr(self, "_last_indicators", {}).items():
            votes[name] = {
                "signal": result.signal,
                "score": result.score,
                "detail": result.detail,
                "meta": result.meta,
            }

        return votes

    def run_with_profile(
        self, df: pd.DataFrame, ticker: str, profile: str
    ) -> dict:
        """
        v5.0: 지정 프로파일로 분석 실행 (BoN 선택기용).

        원래 프로파일을 복원하여 부작용 없이 실행.

        Returns:
            generate_l7_result() 결과 dict
        """
        original_profile = self.profile
        original_weights = self.weights.copy()

        try:
            self.profile = profile
            self.weights = WEIGHT_PROFILES.get(profile, WEIGHT_PROFILES["default"]).copy()
            return self.generate_l7_result(df, ticker)
        finally:
            self.profile = original_profile
            self.weights = original_weights

    def generate_l7_result(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> dict:
        """
        L7 보조 레이어용 결과 반환 (v2.0: 10지표 + 프로파일).
        하위호환: 기존 geo_* 필드 모두 유지.
        """
        signal = self.generate_signal(df, ticker)

        # Approach B/C 지표 결과 직렬화
        ind_dict = {}
        for name, result in getattr(self, "_last_indicators", {}).items():
            ind_dict[name] = {
                "signal": result.signal, "score": result.score,
                "detail": result.detail, "meta": result.meta,
            }

        return {
            # 하위호환 필드
            "geo_action": signal.action,
            "geo_confidence": signal.confidence,
            "geo_harmonic": signal.harmonic,
            "geo_elliott": signal.elliott,
            "geo_slope": signal.slope,
            "geo_reasoning": signal.reasoning,
            "geo_confirms_buy": signal.action == "BUY" and signal.confidence >= 30,
            "geo_warns_sell": signal.action == "SELL" and signal.confidence >= 30,
            # v2.0 신규 필드
            "geo_profile": self.profile,
            "geo_indicators": ind_dict,
        }
