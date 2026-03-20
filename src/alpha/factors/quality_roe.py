"""Q1: ROE 안정성 서브팩터

높은 ROE + 낮은 변동 = 높은 품질.
raw = roe_mean / roe_std (안정성 지표)
+ roe_mean 양수 보너스, 음수 페널티.

유니버스 cross-sectional Z-Score 정규화 → 0.0~1.0 스코어.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class QualityROE:
    """Q1 ROE 안정성 팩터"""

    def __init__(self, quality_data: dict | None = None):
        if quality_data is None:
            quality_data = _load_quality()
        self._quality = quality_data

    def score_raw(self, ticker: str) -> float | None:
        """단일 종목 raw ROE stability 스코어."""
        q = self._quality.get(ticker, {})
        stability = q.get("roe_stability")
        roe_mean = q.get("roe_mean")

        if stability is None or roe_mean is None:
            return None

        # 음수 ROE → 구조적 결함
        if roe_mean < 0:
            return -1.0

        # raw = stability (mean/std) + ROE level 보정
        raw = stability
        if roe_mean >= 0.15:
            raw += 1.0  # 고수익 보너스
        elif roe_mean < 0.03:
            raw *= 0.7  # 저수익 감쇄

        return raw

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 Z-Score 정규화 (0.0~1.0)."""
        raw_scores: dict[str, float] = {}
        for ticker in self._quality:
            raw = self.score_raw(ticker)
            if raw is not None:
                raw_scores[ticker] = raw

        return _zscore_normalize(raw_scores)


class QualityDebt:
    """Q2 부채 건전성 팩터"""

    def __init__(self, quality_data: dict | None = None):
        if quality_data is None:
            quality_data = _load_quality()
        self._quality = quality_data

    def score_raw(self, ticker: str) -> float | None:
        """단일 종목 raw 부채 건전성 스코어."""
        q = self._quality.get(ticker, {})
        debt_ratio = q.get("debt_ratio")

        if debt_ratio is None:
            return None

        # 금융주 보정: 부채비율 80%+ → 은행/증권 특성, 일반 기준 적용 불가
        if debt_ratio > 0.85:
            return 0.3  # 금융주는 중립 하단

        # debt_health = 1 - debt_ratio (0~1, 높을수록 건전)
        return 1.0 - debt_ratio

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 Z-Score 정규화."""
        raw_scores: dict[str, float] = {}
        for ticker in self._quality:
            raw = self.score_raw(ticker)
            if raw is not None:
                raw_scores[ticker] = raw

        return _zscore_normalize(raw_scores)


class QualityAccruals:
    """Q3 이익 품질 팩터"""

    def __init__(self, quality_data: dict | None = None):
        if quality_data is None:
            quality_data = _load_quality()
        self._quality = quality_data

    def score_raw(self, ticker: str) -> float | None:
        """단일 종목 raw Accruals Quality 스코어."""
        q = self._quality.get(ticker, {})
        accruals = q.get("accruals_ratio")

        if accruals is None:
            return None

        # accruals_ratio = operating_cf / net_income
        # >1: 이익이 현금으로 뒷받침 (좋음)
        # <0: 순이익 양수인데 영업CF 음수 (위험)
        # 0~1: 이익 중 일부만 현금 (보통)
        if accruals < 0:
            return -0.5  # 현금흐름 경고
        elif accruals < 0.5:
            return accruals * 0.5  # 약한 현금 뒷받침
        elif accruals <= 3.0:
            return 0.5 + (accruals - 0.5) * 0.2  # 1.0 스케일
        else:
            # >3.0은 한계효용 감소 (감가상각이 큰 자본집약적 기업)
            return min(1.2, 1.0 + (accruals - 3.0) * 0.02)

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 Z-Score 정규화."""
        raw_scores: dict[str, float] = {}
        for ticker in self._quality:
            raw = self.score_raw(ticker)
            if raw is not None:
                raw_scores[ticker] = raw

        return _zscore_normalize(raw_scores)


class QualityDividend:
    """Q4 배당 지속성 팩터"""

    def __init__(self, quality_data: dict | None = None):
        if quality_data is None:
            quality_data = _load_quality()
        self._quality = quality_data

    def score_raw(self, ticker: str) -> float | None:
        """단일 종목 raw 배당 지속성 스코어."""
        q = self._quality.get(ticker, {})
        payout = q.get("dividend_payout")

        if payout is None:
            return 0.0  # 무배당 = 최저점 (성장주는 다른 팩터에서 보상)

        # 이상적 배당성향: 20%~60% → 높은 점수
        # <20%: 너무 인색
        # >80%: 지속 불가능 위험
        # >100%: 이익잉여금 지급 → 경고
        if payout > 1.0:
            return 0.1  # 초과 지급 경고
        elif payout > 0.8:
            return 0.3  # 높은 배당성향 리스크
        elif payout >= 0.2:
            # 20~80% 구간에서 bell curve (50% 정점)
            center = 0.5
            deviation = abs(payout - center) / 0.3
            return max(0.3, 1.0 - deviation * 0.3)
        elif payout > 0:
            return 0.2 + payout  # 소액 배당
        else:
            return 0.0

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 Z-Score 정규화."""
        raw_scores: dict[str, float] = {}
        for ticker in self._quality:
            raw = self.score_raw(ticker)
            if raw is not None:
                raw_scores[ticker] = raw

        return _zscore_normalize(raw_scores)


# ═══════════════════════════════════════════════
# 공통 유틸리티
# ═══════════════════════════════════════════════


def _load_quality() -> dict:
    """financial_quarterly.json에서 quality 섹션 로드."""
    path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "v2_migration" / "financial_quarterly.json"
    if not path.exists():
        logger.warning("financial_quarterly.json 없음 — 빈 데이터 사용")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("quality", {})


def _zscore_normalize(raw_scores: dict[str, float]) -> dict[str, float]:
    """cross-sectional Z-Score → 0.0~1.0 정규화.

    CDF 근사 방식: z → clip to [0, 1] using percentile rank.
    """
    if not raw_scores:
        return {}

    values = np.array(list(raw_scores.values()))
    mean = np.mean(values)
    std = np.std(values)

    if std < 1e-8:
        return {t: 0.5 for t in raw_scores}

    result: dict[str, float] = {}
    for ticker, raw in raw_scores.items():
        z = (raw - mean) / std
        # Percentile rank 방식 (z → 0~1)
        # z=-2 → ~0.02, z=0 → 0.50, z=+2 → ~0.98
        from scipy.stats import norm

        pctile = float(norm.cdf(z))
        result[ticker] = round(pctile, 4)

    return result
