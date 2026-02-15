"""
TGCI (TRIX Golden Cross Indicator) 독립 스코어러
=================================================
indicators.py compute_all()에서 계산된 TRIX/RSI/볼린저/MACD/OBV 컬럼을
조합하여 독립 100점 스코어를 산출한다.

순수 함수 — DataFrame row(dict)만 받으며 외부 의존성 없음.
"""
import math
from typing import Any


class TGCIScorer:
    """TRIX Golden Cross Indicator — 독립 100점 스코어러

    배점 구조 (최대 103점, 100점 캡):
        (1) TRIX 골든크로스: 35점 + 기울기 보너스 최대 8점
        (2) RSI 매수 구간: 최대 25점
        (3) 거래량 폭발: 최대 25점
        (4) OBV 집적: 10점
        (5) 방어 감점: BB 과열 -10, MACD 약화 -8
    """

    @staticmethod
    def score(row: dict[str, Any], config: dict = None) -> dict[str, Any]:
        """TGCI 점수 계산.

        Args:
            row: indicators.py compute_all() 결과의 한 행 (dict).
                 필수 키: trix, trix_signal, trix_golden_cross,
                          rsi_14, volume, volume_ma20,
                          bb_position, macd_histogram
            config: TGCI 설정 (없으면 기본값 사용)

        Returns:
            {"score": 0-100, "grade": str, "details": dict}
        """
        cfg = config or {}
        pts = 0
        details = {}

        # ── (1) TRIX 골든크로스: 35점 + 기울기 보너스 최대 8점 ──
        golden = _safe_get(row, "trix_golden_cross", 0)
        trix_val = _safe_get(row, "trix", 0.0)
        trix_sig = _safe_get(row, "trix_signal", 0.0)
        cross_pts = cfg.get("golden_cross_pts", 35)
        slope_max = cfg.get("slope_bonus_max", 8)

        if golden:
            pts += cross_pts
            slope = trix_val - trix_sig
            if slope > 0:
                pts += min(slope_max, slope * 100)
            details["trix_cross"] = True
            details["trix_slope_bonus"] = round(min(slope_max, max(0, slope * 100)), 1)
        else:
            details["trix_cross"] = False
            details["trix_slope_bonus"] = 0

        # ── (2) RSI 매수 구간: 최대 25점 ──
        rsi = _safe_get(row, "rsi_14", 50.0)
        rsi_range = cfg.get("rsi_optimal_range", [40, 60])
        if rsi_range[0] <= rsi <= rsi_range[1]:
            rsi_pts = 25
        elif (rsi_range[0] - 10) <= rsi < rsi_range[0] or rsi_range[1] < rsi <= (rsi_range[1] + 10):
            rsi_pts = 15
        else:
            rsi_pts = 0
        pts += rsi_pts
        details["rsi"] = round(rsi, 1)
        details["rsi_pts"] = rsi_pts

        # ── (3) 거래량 폭발: 최대 25점 ──
        volume = _safe_get(row, "volume", 0)
        vol_ma20 = _safe_get(row, "volume_ma20", 1)
        vol_mult = cfg.get("volume_explosion_mult", 2.5)
        vol_ratio = volume / max(vol_ma20, 1)
        vol_pts = min(25, int(vol_ratio / vol_mult * 25))
        pts += vol_pts
        details["vol_ratio"] = round(vol_ratio, 2)
        details["vol_pts"] = vol_pts

        # ── (4) OBV 집적: 10점 ──
        obv = _safe_get(row, "obv", 0)
        obv_prev = _safe_get(row, "obv_prev", obv)
        if obv > obv_prev:
            pts += 10
            details["obv_rising"] = True
        else:
            details["obv_rising"] = False

        # ── (5) 방어 감점 ──
        bb_pos = _safe_get(row, "bb_position", 0.5)
        bb_threshold = cfg.get("bb_overheat_threshold", 0.95)
        if bb_pos > bb_threshold:
            pts -= 10
            details["bb_penalty"] = -10
        else:
            details["bb_penalty"] = 0

        macd_hist = _safe_get(row, "macd_histogram", 0)
        macd_hist_prev = _safe_get(row, "macd_histogram_prev", macd_hist)
        if macd_hist < 0 and macd_hist_prev > 0:
            pts -= 8
            details["macd_penalty"] = -8
        else:
            details["macd_penalty"] = 0

        # ── 최종 점수 ──
        score = max(0, min(100, int(pts)))
        grade = _score_to_grade(score)

        return {
            "score": score,
            "grade": grade,
            "details": details,
        }


def _safe_get(row: dict, key: str, default: Any = 0) -> Any:
    """NaN-safe dict get."""
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def _score_to_grade(score: int) -> str:
    """점수 → 등급 변환."""
    if score >= 85:
        return "S"
    if score >= 70:
        return "A"
    if score >= 55:
        return "B"
    if score >= 40:
        return "C"
    return "D"
