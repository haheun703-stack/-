"""US-KR Market History 공용 모듈 — 심볼/스코어/패턴매칭 핵심부.

origin: scripts/archive/backfill/backfill_us_kr_history.py (2026-05-17 이전)
이전 사유: CLAUDE.md LOCK 규칙(`scripts/archive/` 참조 금지) 위반 해소.

호출처:
    - scripts/update_us_kr_daily.py        (US_SYMBOLS, KR_SYMBOLS, _clamp, _calc_overnight_score)
    - scripts/us_overnight_signal.py        (PatternMatcher)

backfill 전체(데이터 다운로드/DB 생성/리포트 등)는 archive에 그대로 보존.
일상 운영(BAT-A/BAT-D 정기 수집)에 필요한 핵심부만 본 모듈로 추출.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
US_DIR = PROJECT_ROOT / "data" / "us_market"
DB_PATH = US_DIR / "us_kr_history.db"

logger = logging.getLogger(__name__)


# ── US 심볼 ──
US_SYMBOLS = {
    # Tier 1: 핵심 지표
    "us_sp500":   "SPY",
    "us_nasdaq":  "QQQ",
    "us_dow":     "DIA",
    "us_vix":     "^VIX",
    "us_soxx":    "SOXX",
    # Tier 2: 매크로
    "us_oil":     "USO",
    "us_gold":    "GLD",
    "us_dollar":  "UUP",
    "us_bond10y": "TLT",
    "us_china":   "FXI",
    # Tier 3: 테마
    "us_tsla":    "TSLA",
    "us_xbi":     "XBI",
    "us_xle":     "XLE",
    "us_xlf":     "XLF",
    # 한국 프록시 (미국 상장 Korea ETF — 가장 강력한 KOSPI 선행지표)
    "us_ewy":     "EWY",
}

# ── KR 섹터 ETF (KODEX 시리즈) ──
KR_SYMBOLS = {
    "kr_kospi":    "^KS11",
    "kr_kosdaq":   "^KQ11",
    "kr_semi":     "091160.KS",   # KODEX 반도체
    "kr_ev":       "305720.KS",   # KODEX 2차전지산업
    "kr_bio":      "244580.KS",   # KODEX 바이오
    "kr_bank":     "091170.KS",   # KODEX 은행
    "kr_steel":    "117680.KS",   # KODEX 철강
    "kr_it":       "315930.KS",   # KODEX IT플러스
    "kr_energy":   "117460.KS",   # KODEX 에너지화학
    "kr_domestic": "069500.KS",   # KODEX 200 (내수 대용)
}

# 실패 시 대체 심볼
KR_FALLBACK = {
    "kr_semi":     ["091230.KS"],
    "kr_ev":       ["364690.KS"],
    "kr_bio":      ["227540.KS"],
    "kr_it":       ["261060.KS"],
    "kr_energy":   ["139230.KS"],
    "kr_domestic": ["229200.KS"],
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _calc_overnight_score(record: dict) -> float:
    """Level 1 US Overnight Score (-100 ~ +100).

    가중치: EWY 25% > NASDAQ 20% > SP500 15% = VIX 15% = SOXX 15% > Dollar 10%
    EWY는 미국 상장 한국 ETF로 KOSPI와 직접 연동 → 가장 강력한 선행지표.
    """
    score = 0.0

    # EWY (25%) — 한국 프록시, 가장 높은 가중치
    v = record.get("us_ewy_chg")
    if v is not None:
        score += _clamp(v * 12.5, -25, 25)

    # NASDAQ (20%)
    v = record.get("us_nasdaq_chg")
    if v is not None:
        score += _clamp(v * 10, -20, 20)

    # S&P 500 (15%)
    v = record.get("us_sp500_chg")
    if v is not None:
        score += _clamp(v * 7.5, -15, 15)

    # VIX (15%, 역방향)
    v = record.get("us_vix_chg")
    if v is not None:
        score += _clamp(v * -3.75, -15, 15)

    # SOXX (15%)
    v = record.get("us_soxx_chg")
    if v is not None:
        score += _clamp(v * 10, -15, 15)

    # Dollar (10%, 역방향)
    v = record.get("us_dollar_chg")
    if v is not None:
        score += _clamp(v * -7, -10, 10)

    return round(_clamp(score, -100, 100), 1)


class PatternMatcher:
    """역사적 패턴 매칭으로 보정값 산출."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DB_PATH)
        self.min_samples = 15

    def find_similar_patterns(
        self,
        today_us: dict,
        top_pct: int = 20,
    ) -> pd.DataFrame | None:
        """오늘과 유사한 과거 패턴 검색.

        Args:
            today_us: {"us_nasdaq_chg": float, "us_sp500_chg": float, ...}
            top_pct: 상위 몇 % 유사도를 선택할지
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM us_kr_history ORDER BY date DESC", conn)
        conn.close()

        if len(df) < self.min_samples:
            logger.warning(f"데이터 부족: {len(df)}건 (최소 {self.min_samples}건)")
            return None

        features = [
            "us_nasdaq_chg", "us_sp500_chg", "us_vix_chg",
            "us_soxx_chg", "us_dollar_chg", "us_ewy_chg",
        ]

        today_vector = np.array([
            today_us.get("us_nasdaq_chg", 0) or 0,
            today_us.get("us_sp500_chg", 0) or 0,
            today_us.get("us_vix_chg", 0) or 0,
            today_us.get("us_soxx_chg", 0) or 0,
            today_us.get("us_dollar_chg", 0) or 0,
            today_us.get("us_ewy_chg", 0) or 0,
        ])

        feature_df = df[features].fillna(0)
        means = feature_df.mean()
        stds = feature_df.std().replace(0, 1)

        norm_today = (today_vector - means.values) / stds.values
        norm_hist = (feature_df - means) / stds

        distances = np.sqrt(((norm_hist - norm_today) ** 2).sum(axis=1))
        df["distance"] = distances

        threshold = np.percentile(distances.dropna(), top_pct)
        similar = df[df["distance"] <= threshold].copy()

        return similar.sort_values("distance")

    def analyze_patterns(self, similar_days: pd.DataFrame | None) -> dict:
        """유사 패턴 분석 -> 보정값 산출."""
        if similar_days is None or len(similar_days) < self.min_samples:
            return {"status": "insufficient_data", "pattern_adjustment": 0, "confidence": 0}

        result = {"sample_count": len(similar_days), "status": "ok"}

        # KOSPI 예측
        kospi = similar_days["kr_kospi_chg"].dropna()
        result["kospi"] = {
            "mean_chg": round(kospi.mean(), 3),
            "median_chg": round(kospi.median(), 3),
            "std": round(kospi.std(), 3),
            "positive_rate": round((kospi > 0).mean() * 100, 1),
        }

        # 시가 갭 예측
        gap = similar_days["kr_kospi_open_gap"].dropna()
        if len(gap) > 0:
            result["kospi_open_gap"] = {
                "mean_gap": round(gap.mean(), 3),
                "median_gap": round(gap.median(), 3),
            }

        # 섹터별 예측
        sector_cols = {
            "반도체":   "kr_semi_chg",
            "2차전지":  "kr_ev_chg",
            "바이오":   "kr_bio_chg",
            "은행":     "kr_bank_chg",
            "철강":     "kr_steel_chg",
            "IT":       "kr_it_chg",
            "에너지":   "kr_oil_chg",
            "내수":     "kr_domestic_chg",
        }

        result["sectors"] = {}
        for sector_name, col in sector_cols.items():
            if col in similar_days.columns:
                s = similar_days[col].dropna()
                if len(s) >= 10:
                    result["sectors"][sector_name] = {
                        "mean_chg": round(s.mean(), 3),
                        "positive_rate": round((s > 0).mean() * 100, 1),
                        "sample_count": len(s),
                    }

        # 패턴 보정값 (-15 ~ +15)
        confidence = min(len(kospi) / 50, 1.0)
        pattern_adj = kospi.mean() * 5 * confidence
        result["pattern_adjustment"] = round(_clamp(pattern_adj, -15, 15), 1)
        result["confidence"] = round(confidence, 2)

        return result
