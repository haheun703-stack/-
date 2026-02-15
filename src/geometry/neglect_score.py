"""
5D-⑤ Crowd Neglect Score — 군중 무관심 점수 계산

네이버 스크래핑 없이 이미 존재하는 지표 데이터만으로 프록시 계산.
세 축으로 무관심 수준을 측정:
  1. 거래량 위축 (volume_neglect)
  2. 신용잔고 감소 (credit_neglect)
  3. 변동성 축소 (volatility_neglect)

핵심 가정: "관심이 식은 자리"에서 시작점이 만들어진다.
  → 거래량 바닥 + 신용 축소 + BB 수축 = 군중이 떠난 구간

의존성: numpy, pandas (indicators.py의 bb_upper, bb_lower, volume_ma5 등 사용)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NeglectScorer:
    """군중 무관심(Crowd Neglect) 점수 계산기"""

    # 무관심 수준 분류 기준
    LEVEL_THRESHOLDS = {
        "극도": 0.8,
        "높음": 0.6,
        "보통": 0.4,
    }

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("geometry", {}).get("neglect", {})
        self.volume_weight = cfg.get("volume_weight", 0.40)
        self.credit_weight = cfg.get("credit_weight", 0.30)
        self.volatility_weight = cfg.get("volatility_weight", 0.30)

    # ─── 거래량 무관심 ─────────────────────────────

    def volume_neglect(self, volumes: np.ndarray, lookback: int = 120) -> float:
        """
        거래량 위축 점수 계산.

        현재 5일 평균 거래량 / 과거 lookback일 평균 거래량의 비율로 측정.
        거래량이 과거 대비 적을수록 높은 점수 (최대 1.0).

        Parameters:
            volumes: 거래량 배열 (최신이 마지막)
            lookback: 과거 비교 기간 (기본 120일)

        Returns:
            0.0~1.0 (거래량이 적을수록 높은 점수)
        """
        volumes = np.asarray(volumes, dtype=float)

        if len(volumes) < 10:
            logger.debug("거래량 데이터 부족: %d일 < 10일", len(volumes))
            return 0.3  # 중립

        # 최근 5일 평균
        current_vol_ma5 = float(np.mean(volumes[-5:]))

        # 과거 lookback일 평균 (최근 5일 제외)
        hist_end = max(len(volumes) - 5, 0)
        hist_start = max(hist_end - lookback, 0)
        if hist_end - hist_start < 5:
            avg_vol = float(np.mean(volumes))
        else:
            avg_vol = float(np.mean(volumes[hist_start:hist_end]))

        if avg_vol <= 0:
            return 0.3

        ratio = current_vol_ma5 / avg_vol
        score = max(0.0, 1.0 - ratio)

        # 거래량이 과거 평균의 20% 이하면 만점
        if ratio <= 0.2:
            score = 1.0

        return min(1.0, score)

    # ─── 신용잔고 무관심 ───────────────────────────

    def credit_neglect(self, credit_balances: np.ndarray | None, lookback: int = 20) -> float:
        """
        신용잔고 변화율 기반 무관심 점수.

        신용잔고가 감소할수록 높은 점수 (개인 투자자 이탈 징후).
        데이터가 없으면 중립값(0.3) 반환.

        Parameters:
            credit_balances: 신용잔고 배열 (최신이 마지막), None이면 데이터 없음
            lookback: 변화율 계산 기간 (기본 20일)

        Returns:
            0.0~1.0 (신용잔고 감소율이 클수록 높은 점수)
        """
        if credit_balances is None:
            return 0.3

        credit_balances = np.asarray(credit_balances, dtype=float)
        if len(credit_balances) < 2:
            return 0.3

        # lookback 기간 전 값과 현재 값 비교
        start_idx = max(0, len(credit_balances) - lookback)
        start_val = float(credit_balances[start_idx])
        end_val = float(credit_balances[-1])

        if start_val <= 0:
            return 0.3

        change_rate = (end_val - start_val) / start_val  # 음수면 감소

        # 40% 감소면 만점
        score = max(0.0, -change_rate / 0.4)
        return min(1.0, score)

    # ─── 변동성 무관심 ─────────────────────────────

    def volatility_neglect(self, prices: np.ndarray, bb_width_current: float | None = None) -> float:
        """
        볼린저밴드 폭 축소 기반 무관심 점수.

        BB 폭이 과거 대비 축소될수록 높은 점수 (변동성 소멸 = 관심 소멸).

        Parameters:
            prices: 종가 배열 (최소 120일, 최신이 마지막)
            bb_width_current: 현재 BB 폭 (없으면 직접 계산)

        Returns:
            0.0~1.0 (BB 폭이 축소될수록 높은 점수)
        """
        prices = np.asarray(prices, dtype=float)

        if len(prices) < 20:
            logger.debug("가격 데이터 부족: %d일 < 20일", len(prices))
            return 0.3

        # BB 폭 시계열 계산 (20일 기준)
        window = 20
        bb_widths = []
        for i in range(window - 1, len(prices)):
            segment = prices[i - window + 1 : i + 1]
            mid = np.mean(segment)
            std = np.std(segment, ddof=1) if len(segment) > 1 else 0.0
            upper = mid + 2 * std
            lower = mid - 2 * std
            width = (upper - lower) / mid if mid > 0 else 0.0
            bb_widths.append(width)

        if not bb_widths:
            return 0.3

        bb_widths = np.array(bb_widths)

        # 현재 BB 폭
        if bb_width_current is None:
            bb_width_current = float(bb_widths[-1])

        # 과거 120일 평균 BB 폭 (최근 제외)
        lookback = min(120, len(bb_widths) - 1)
        if lookback < 5:
            avg_bb_width = float(np.mean(bb_widths))
        else:
            avg_bb_width = float(np.mean(bb_widths[-lookback - 1 : -1]))

        if avg_bb_width <= 0:
            return 0.3

        current_ratio = bb_width_current / avg_bb_width
        score = max(0.0, 1.0 - current_ratio)

        return min(1.0, score)

    # ─── 통합 점수 ─────────────────────────────────

    def score(self, row: dict, df: pd.DataFrame | None = None) -> dict:
        """
        통합 무관심 점수 계산.

        row에서 indicators.py가 계산한 컬럼들을 활용하고,
        df가 있으면 과거 데이터를 이용해 정밀 계산.

        Parameters:
            row: 현재 시점의 지표 dict (volume, volume_ma5, volume_ma20,
                 bb_upper, bb_lower, close 등)
            df: 과거 전체 DataFrame (있으면 정밀 계산)

        Returns:
            {
                "volume_score": float,
                "credit_score": float,
                "volatility_score": float,
                "total_score": float,
                "neglect_level": str,
                "interpretation": str,
            }
        """
        # ── 거래량 점수 ──
        if df is not None and "volume" in df.columns:
            vol_score = self.volume_neglect(df["volume"].values)
        elif "volume_ma5" in row and "volume_ma20" in row:
            # row만으로 근사 계산
            ma5 = float(row.get("volume_ma5", 0))
            ma20 = float(row.get("volume_ma20", 0))
            if ma20 > 0:
                ratio = ma5 / ma20
                vol_score = max(0.0, min(1.0, 1.0 - ratio))
            else:
                vol_score = 0.3
        else:
            vol_score = 0.3

        # ── 신용잔고 점수 ──
        if df is not None and "credit_balance" in df.columns:
            cred_score = self.credit_neglect(df["credit_balance"].values)
        else:
            credit_bal = row.get("credit_balance")
            if credit_bal is not None:
                cred_score = 0.5  # 단일값으로는 변화율 계산 불가, 중간값
            else:
                cred_score = 0.3  # 데이터 없음 → 중립

        # ── 변동성 점수 ──
        bb_width_current = None
        if "bb_upper" in row and "bb_lower" in row and "close" in row:
            close = float(row["close"])
            bb_u = float(row.get("bb_upper", 0))
            bb_l = float(row.get("bb_lower", 0))
            if close > 0:
                bb_width_current = (bb_u - bb_l) / close

        if df is not None and "close" in df.columns:
            vol_neg_score = self.volatility_neglect(
                df["close"].values, bb_width_current
            )
        elif bb_width_current is not None and "bb_width" in row:
            # bb_width 컬럼이 있으면 근사 (현재 vs 과거 비교 불가 → 절대 수준 사용)
            bb_w = float(row["bb_width"])
            # BB 폭이 4% 이하이면 상당히 축소된 상태
            vol_neg_score = max(0.0, min(1.0, 1.0 - bb_w / 0.10))
        else:
            vol_neg_score = 0.3

        # ── 가중평균 ──
        total = (
            self.volume_weight * vol_score
            + self.credit_weight * cred_score
            + self.volatility_weight * vol_neg_score
        )
        total = min(1.0, max(0.0, total))

        # ── 수준 분류 ──
        if total > self.LEVEL_THRESHOLDS["극도"]:
            level = "극도"
        elif total > self.LEVEL_THRESHOLDS["높음"]:
            level = "높음"
        elif total > self.LEVEL_THRESHOLDS["보통"]:
            level = "보통"
        else:
            level = "낮음"

        # ── 해석 ──
        interpretation = self._interpret(level, vol_score, cred_score, vol_neg_score)

        return {
            "volume_score": round(vol_score, 3),
            "credit_score": round(cred_score, 3),
            "volatility_score": round(vol_neg_score, 3),
            "total_score": round(total, 3),
            "neglect_level": level,
            "interpretation": interpretation,
        }

    # ─── 해석 생성 ─────────────────────────────────

    @staticmethod
    def _interpret(level: str, vol: float, cred: float, volat: float) -> str:
        """한국어 해석 문자열 생성"""
        parts = []

        if vol >= 0.7:
            parts.append("거래량 극도 위축")
        elif vol >= 0.5:
            parts.append("거래량 위축")
        elif vol >= 0.3:
            parts.append("거래량 보통")
        else:
            parts.append("거래량 활발")

        if cred >= 0.7:
            parts.append("신용잔고 대폭 감소")
        elif cred >= 0.4:
            parts.append("신용잔고 감소")
        else:
            parts.append("신용잔고 보통")

        if volat >= 0.7:
            parts.append("변동성 극도 축소")
        elif volat >= 0.5:
            parts.append("변동성 축소")
        elif volat >= 0.3:
            parts.append("변동성 보통")
        else:
            parts.append("변동성 활발")

        detail = ", ".join(parts)

        level_desc = {
            "극도": "군중이 완전히 이탈한 상태 — 시작점 형성 가능성 높음",
            "높음": "관심이 상당히 식은 상태 — 바닥 탐색 구간",
            "보통": "관심도 중립 — 추가 확인 필요",
            "낮음": "시장 관심 유지 중 — 무관심 전략 비적합",
        }

        return f"{level_desc[level]} ({detail})"

    # ─── 프롬프트 텍스트 ───────────────────────────

    @staticmethod
    def to_prompt_text(result: dict) -> str:
        """Claude API 입력용 텍스트"""
        lines = [
            "[군중 무관심 점수]",
            f"  거래량 위축: {result['volume_score']:.2f}",
            f"  신용잔고 감소: {result['credit_score']:.2f}",
            f"  변동성 축소: {result['volatility_score']:.2f}",
            f"  종합 점수: {result['total_score']:.2f} ({result['neglect_level']})",
            f"  해석: {result['interpretation']}",
        ]
        return "\n".join(lines)
