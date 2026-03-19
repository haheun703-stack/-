"""V1: EBITDA/EV 서브팩터 + V2: FCF Yield 서브팩터

V1: EBITDA/EV = ebitda_approx / (market_cap + total_debt)
   높을수록 저평가 → 높은 점수

V2: FCF Yield = FCF / market_cap
   높을수록 주주가치 창출력 우수 → 높은 점수

유니버스 cross-sectional Z-Score 정규화 → 0.0~1.0 스코어.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class ValueEbitdaEV:
    """V1 EBITDA/EV 팩터"""

    def __init__(
        self,
        financial_data: dict | None = None,
        market_cap_data: dict | None = None,
    ):
        if financial_data is None:
            financial_data = _load_financial()
        if market_cap_data is None:
            market_cap_data = _load_market_cap()

        self._quality = financial_data.get("quality", {})
        self._bs_data = financial_data.get("bs_data", {})
        self._market_cap = market_cap_data

    def score_raw(self, ticker: str) -> float | None:
        """단일 종목 raw EBITDA/EV 스코어."""
        q = self._quality.get(ticker, {})
        ebitda = q.get("ebitda_approx")
        if ebitda is None or ebitda <= 0:
            return None

        # 시가총액 (억원 → 원)
        mc_entry = self._market_cap.get(ticker)
        if mc_entry is None:
            return None
        market_cap = mc_entry.get("hts_avls", 0) * 1e8  # 억원 → 원
        if market_cap <= 0:
            return None

        # 순부채: 최신 분기 BS에서 total_debt
        total_debt = self._get_latest_debt(ticker)

        # EV = market_cap + total_debt (현금 미차감, 보수적)
        ev = market_cap + (total_debt if total_debt else 0)
        if ev <= 0:
            return None

        return ebitda / ev

    def _get_latest_debt(self, ticker: str) -> float | None:
        """최신 분기 BS에서 total_debt 추출."""
        bs = self._bs_data.get(ticker, {})
        if not bs:
            return None
        latest_key = sorted(bs.keys())[-1]
        return bs[latest_key].get("total_debt")

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 Z-Score 정규화 (0.0~1.0)."""
        raw_scores: dict[str, float] = {}
        for ticker in self._quality:
            raw = self.score_raw(ticker)
            if raw is not None:
                raw_scores[ticker] = raw

        return _zscore_normalize(raw_scores)


class ValueFCFYield:
    """V2 FCF Yield 팩터"""

    def __init__(
        self,
        financial_data: dict | None = None,
        market_cap_data: dict | None = None,
    ):
        if financial_data is None:
            financial_data = _load_financial()
        if market_cap_data is None:
            market_cap_data = _load_market_cap()

        self._quality = financial_data.get("quality", {})
        self._market_cap = market_cap_data

    def score_raw(self, ticker: str) -> float | None:
        """단일 종목 raw FCF Yield 스코어."""
        q = self._quality.get(ticker, {})
        fcf = q.get("fcf")
        if fcf is None:
            return None

        # 시가총액
        mc_entry = self._market_cap.get(ticker)
        if mc_entry is None:
            return None
        market_cap = mc_entry.get("hts_avls", 0) * 1e8
        if market_cap <= 0:
            return None

        # FCF Yield = FCF / market_cap
        # 음수 FCF는 허용 (Z-Score에서 하위로 밀림)
        return fcf / market_cap

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 Z-Score 정규화 (0.0~1.0)."""
        raw_scores: dict[str, float] = {}
        for ticker in self._quality:
            raw = self.score_raw(ticker)
            if raw is not None:
                raw_scores[ticker] = raw

        return _zscore_normalize(raw_scores)


# ═══════════════════════════════════════════════
# 공통 유틸리티
# ═══════════════════════════════════════════════


def _load_financial() -> dict:
    """financial_quarterly.json 전체 로드."""
    path = PROJECT_ROOT / "data" / "v2_migration" / "financial_quarterly.json"
    if not path.exists():
        logger.warning("financial_quarterly.json 없음")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_market_cap() -> dict:
    """market_cap_cache.json 로드."""
    path = PROJECT_ROOT / "data" / "market_cap_cache.json"
    if not path.exists():
        logger.warning("market_cap_cache.json 없음")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _zscore_normalize(raw_scores: dict[str, float]) -> dict[str, float]:
    """cross-sectional Z-Score → 0.0~1.0 정규화."""
    if not raw_scores:
        return {}

    values = np.array(list(raw_scores.values()))
    mean = np.mean(values)
    std = np.std(values)

    if std < 1e-8:
        return {t: 0.5 for t in raw_scores}

    from scipy.stats import norm

    result: dict[str, float] = {}
    for ticker, raw in raw_scores.items():
        z = (raw - mean) / std
        pctile = float(norm.cdf(z))
        result[ticker] = round(pctile, 4)

    return result
