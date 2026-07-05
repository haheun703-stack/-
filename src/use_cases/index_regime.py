"""시장 지수 레짐 계산 (KOSPI/KOSDAQ 공용) — FV 엔진 v1-3번.

src/etf/data_bridge.calc_kospi_regime()의 로직(MA20/MA60 + RV20 백분위)을
지수 무관하게 일반화. brain.py 계열(매매 크리티컬)을 건드리지 않기 위해 별도 모듈.
동일 규칙이므로 KOSPI 레짐도 이 함수로 재현 가능(정합성 검증됨).

레짐: BULL(MA20 위+저변동) / CAUTION(MA20 위+고변동 or MA20아래 MA60위 애매)
      / BEAR(MA20 아래 MA60 위) / CRISIS(MA60 아래).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

_DEFAULT = {"regime": "CAUTION", "close": 0.0, "ma20": 0.0, "ma60": 0.0,
            "rv_pct": 0.5, "ma20_above": False, "ma60_above": False}


def calc_index_regime(index_csv: str | Path) -> dict:
    """지수 CSV(Date,close,...) → 레짐 dict. calc_kospi_regime와 동일 규칙."""
    path = Path(index_csv)
    if not path.exists():
        return dict(_DEFAULT)
    try:
        df = pd.read_csv(path)
        date_col = next((c for c in df.columns if c.lower() == "date"), df.columns[0])
        close_col = next((c for c in df.columns if c.lower() == "close"), None)
        if close_col is None:
            return dict(_DEFAULT)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        if len(df) < 60:
            return dict(_DEFAULT)

        close_s = pd.to_numeric(df[close_col], errors="coerce")
        ma20 = float(close_s.rolling(20).mean().iloc[-1])
        ma60 = float(close_s.rolling(60).mean().iloc[-1])
        close = float(close_s.iloc[-1])

        log_ret = np.log(close_s / close_s.shift(1))
        rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        rv20_pct = rv20.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        rv_pct = float(rv20_pct.iloc[-1]) if not pd.isna(rv20_pct.iloc[-1]) else 0.5

        if ma20 <= 0 or ma60 <= 0 or np.isnan(ma20) or np.isnan(ma60):
            regime = "CAUTION"
        elif close > ma20:
            regime = "BULL" if rv_pct < 0.50 else "CAUTION"
        elif close > ma60:
            regime = "BEAR"
        else:
            regime = "CRISIS"

        return {"regime": regime, "close": round(close, 2),
                "ma20": round(ma20, 2), "ma60": round(ma60, 2),
                "rv_pct": round(rv_pct, 2),
                "ma20_above": close > ma20, "ma60_above": close > ma60}
    except Exception as e:  # noqa: BLE001
        logger.warning("[index_regime] %s 계산 실패: %s", path.name, e)
        return dict(_DEFAULT)


def kospi_regime() -> dict:
    return calc_index_regime(DATA_DIR / "kospi_index.csv")


def kosdaq_regime() -> dict:
    return calc_index_regime(DATA_DIR / "kosdaq_index.csv")
