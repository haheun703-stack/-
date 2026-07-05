"""역사 PER 밴드 (자기 5년 이력 대비 현재 밸류에이션 위치) — 미래가치 엔진 V축 정밀화.

설계: docs/02-design/future-value-engine_2026-07-04.md (v1 로드맵 2번)
배경: 7/4 백테스트에서 '횡단면 저PER'(유니버스 내 상대)이 유일하게 견고했다.
      본 모듈은 한 단계 더 — '이 종목이 자기 역사 대비 싼가?'(평균회귀 신호)를
      더한다. 지주사 NAV(6/25)에서 검증된 z-band 방법론과 동형.

데이터 (VPS 실측 7/5):
  - EPS: data/dart_cache/fundamentals_historical.csv (2015~2025 분기별·2980종·★단일분기)
  - 가격: data/processed/{ticker}.parquet (일봉 종가)
  TTM EPS = 최근 4분기 EPS 합. PER = 종가 / TTM_EPS.

★핵심 원칙:
  1. point-in-time: 각 분기 EPS는 '결산일 + DISCLOSURE_LAG_DAYS(60)' 이후에만
     알려진 것으로 취급 → 백테스트 lookahead 차단(DART 45일 규칙 + 여유).
  2. 흑자 게이트: TTM EPS ≤ 0 이면 PER 무의미 → 밴드 미산출(경기민감·적자주 트랩
     회피). earnings_positive_ratio 낮으면 신뢰도 하향.
  3. 밴드 = 트레일링 WINDOW(기본 5년) PER의 분위(p25/median/p75). 현재 PER의
     percent-rank(0=역대최저=최저평가 ~ 1=역대최고=최고평가).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FUND_CSV = DATA_DIR / "dart_cache" / "fundamentals_historical.csv"
PROCESSED_DIR = DATA_DIR / "processed"

DISCLOSURE_LAG_DAYS = 60      # 결산일 → 공시 가용일 (DART 분기 45일 규칙 + 여유)
BAND_WINDOW_YEARS = 5         # 밴드 기준 트레일링 기간
MIN_BAND_DAYS = 250           # 밴드 신뢰 최소 표본(≈1년)
LOW_PCT, HIGH_PCT = 0.25, 0.75  # 저평가/고평가 밴드 경계

_fund_cache: pd.DataFrame | None = None
_ttm_cache: dict[str, pd.Series | None] = {}


def _load_fundamentals() -> pd.DataFrame:
    global _fund_cache
    if _fund_cache is None:
        df = pd.read_csv(FUND_CSV)
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        # 결산일 → 공시 가용일 (point-in-time)
        df["settle"] = pd.to_datetime(df["settle_date"], errors="coerce")
        df["avail"] = df["settle"] + timedelta(days=DISCLOSURE_LAG_DAYS)
        df = df.dropna(subset=["avail", "eps"]).sort_values(["ticker", "settle"])
        _fund_cache = df
    return _fund_cache


def _ttm_eps_series(ticker: str) -> pd.Series | None:
    """공시 가용일 색인 TTM EPS 시계열 (최근 4분기 합, 4분기 미만은 제외)."""
    if ticker in _ttm_cache:
        return _ttm_cache[ticker]
    df = _load_fundamentals()
    s = df[df["ticker"] == ticker]
    if len(s) < 4:
        _ttm_cache[ticker] = None
        return None
    ttm = s["eps"].rolling(4).sum()
    out = pd.Series(ttm.values, index=s["avail"].values).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    _ttm_cache[ticker] = out if len(out) else None
    return _ttm_cache[ticker]


def _load_close(ticker: str) -> pd.Series | None:
    try:
        df = pd.read_parquet(PROCESSED_DIR / f"{ticker}.parquet")
    except Exception:
        return None
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    col = "Close" if "Close" in df.columns else "close"
    if col not in df.columns:
        return None
    return pd.Series(df[col].values, index=idx).sort_index()


def _per_series(ticker: str, as_of: pd.Timestamp | None = None) -> pd.Series | None:
    """일별 트레일링 PER 시계열 (TTM EPS > 0 구간만). as_of 이후는 절단(point-in-time)."""
    ttm = _ttm_eps_series(ticker)
    close = _load_close(ticker)
    if ttm is None or close is None or close.empty:
        return None
    if as_of is not None:
        close = close[close.index <= as_of]
        ttm = ttm[ttm.index <= as_of]
    if close.empty or ttm.empty:
        return None
    # 각 거래일에 '그날까지 공시된' 최신 TTM EPS를 매칭 (forward-fill, lookahead 없음)
    eps_aligned = ttm.reindex(close.index, method="ffill")
    per = close / eps_aligned
    per = per[(eps_aligned > 0) & np.isfinite(per) & (per > 0) & (per < 500)]  # 이상치 컷
    return per if len(per) else None


def compute_per_band(ticker: str, as_of: pd.Timestamp | str | None = None) -> dict | None:
    """종목의 역사 PER 밴드 + 현재 위치.

    Returns dict or None(데이터 부족):
      current_per, band_p25, band_median, band_p75, pct_rank(0~1),
      n_days, coverage_years, earnings_positive_ratio, signal, reliable
    """
    if isinstance(as_of, str):
        as_of = pd.Timestamp(as_of)
    per = _per_series(ticker, as_of)
    if per is None:
        return None
    window_start = per.index[-1] - pd.Timedelta(days=int(BAND_WINDOW_YEARS * 365.25))
    band = per[per.index >= window_start]
    if len(band) < MIN_BAND_DAYS:
        return None
    current = float(band.iloc[-1])
    p25, p50, p75 = (float(band.quantile(q)) for q in (0.25, 0.50, 0.75))
    pct_rank = float((band < current).mean())
    coverage_years = round((band.index[-1] - band.index[0]).days / 365.25, 1)

    # 흑자 안정성: 밴드 기간 내 TTM EPS 양수 비율 (경기민감·적자 트랩 방어)
    ttm = _ttm_eps_series(ticker)
    if as_of is not None:
        ttm = ttm[ttm.index <= as_of]
    pos_ratio = float((ttm > 0).mean()) if ttm is not None and len(ttm) else 0.0

    if pct_rank <= LOW_PCT:
        signal = "저평가(밴드하단)"
    elif pct_rank >= HIGH_PCT:
        signal = "고평가(밴드상단)"
    else:
        signal = "중립"
    # 신뢰: 5년+ 커버 & 흑자비율 80%+ (안정적 흑자 기업만 PER 밴드 유효)
    reliable = coverage_years >= 3.0 and pos_ratio >= 0.8

    return {
        "current_per": round(current, 1),
        "band_p25": round(p25, 1), "band_median": round(p50, 1), "band_p75": round(p75, 1),
        "pct_rank": round(pct_rank, 3),
        "n_days": len(band), "coverage_years": coverage_years,
        "earnings_positive_ratio": round(pos_ratio, 2),
        "signal": signal, "reliable": reliable,
    }
