"""KOSPI 레짐 실시간 계산 — 단일 공용 함수.

kospi_index.csv를 읽어서 MA20/MA60 + 실현변동성 기반으로
현재 KOSPI 레짐(BULL/CAUTION/BEAR/CRISIS)을 판정한다.

이 모듈이 프로젝트 전체의 유일한 레짐 계산 소스이다.
(기존 kospi_regime.json 유물을 대체)

Usage:
    from src.utils.kospi_regime_calc import get_kospi_regime
    regime = get_kospi_regime()
    # {"regime": "CAUTION", "slots": 3, "close": 6244.1, "ma20": 5519.9, ...}
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_KOSPI_CSV = PROJECT_ROOT / "data" / "kospi_index.csv"
DEFAULT_SETTINGS_YAML = PROJECT_ROOT / "config" / "settings.yaml"

# 레짐별 기본 슬롯
_REGIME_SLOTS = {"BULL": 5, "CAUTION": 3, "BEAR": 2, "CRISIS": 0}


def get_kospi_regime(
    kospi_csv: Path | None = None,
    settings_yaml: Path | None = None,
    stale_days: int = 3,
) -> dict:
    """KOSPI 레짐 판정 (MA20/MA60 + 실현변동성).

    Args:
        kospi_csv: kospi_index.csv 경로. None이면 기본 경로 사용.
        settings_yaml: settings.yaml 경로 (SW-3 역발상 슬롯 오버라이드용).
        stale_days: CSV 마지막 날짜가 이 영업일 이상 오래되면 UNKNOWN 반환.

    Returns:
        {"regime": str, "slots": int, "close": float,
         "ma20": float, "ma60": float, "rv_pct": float,
         "date": str, "stale": bool}
    """
    csv_path = kospi_csv or DEFAULT_KOSPI_CSV
    fallback = {
        "regime": "UNKNOWN", "slots": 0, "close": 0,
        "ma20": 0, "ma60": 0, "rv_pct": 0.5,
        "date": "", "stale": True,
    }

    if not csv_path.exists():
        logger.warning("kospi_index.csv 없음: %s", csv_path)
        return fallback

    try:
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True).sort_index()
    except Exception as e:
        logger.error("kospi_index.csv 읽기 실패: %s", e)
        return fallback

    if len(df) < 60:
        logger.warning("kospi_index.csv 데이터 부족: %d행", len(df))
        return {**fallback, "regime": "CAUTION", "slots": 3}

    # stale 체크
    last_date = df.index[-1].date()
    today = datetime.now().date()
    calendar_gap = (today - last_date).days
    stale = calendar_gap > stale_days + 2  # 영업일 3일 ≈ 달력 5일
    if stale:
        logger.warning(
            "kospi_index.csv stale: 마지막 %s (오늘 %s, %d일 차이)",
            last_date, today, calendar_gap,
        )

    # 기술 지표 계산
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["rv20_pct"] = df["rv20"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    row = df.iloc[-1]
    close = float(row["close"])
    ma20 = float(row["ma20"]) if not pd.isna(row["ma20"]) else 0
    ma60 = float(row["ma60"]) if not pd.isna(row["ma60"]) else 0
    rv_pct = float(row.get("rv20_pct", 0.5)) if not pd.isna(row.get("rv20_pct", 0.5)) else 0.5

    # 레짐 분류
    if ma20 == 0 or ma60 == 0:
        regime, slots = "CAUTION", 3
    elif close > ma20:
        regime, slots = ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
    elif close > ma60:
        regime, slots = "BEAR", 2
    else:
        regime, slots = "CRISIS", 0

    # SW-3: 역발상 매수 — BEAR/CRISIS 슬롯 오버라이드
    yaml_path = settings_yaml or DEFAULT_SETTINGS_YAML
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            sw_cfg = yaml.safe_load(f).get("swing_philosophy", {})
        if sw_cfg.get("enabled"):
            contrarian = sw_cfg.get("contrarian", {})
            if regime in contrarian:
                slots = contrarian[regime].get("slots", slots)
    except Exception:
        pass

    if stale:
        regime = "UNKNOWN"
        slots = 0
        logger.warning("stale 데이터 → 레짐 UNKNOWN 반환")

    return {
        "regime": regime,
        "slots": slots,
        "close": close,
        "ma20": ma20,
        "ma60": ma60,
        "rv_pct": rv_pct,
        "date": str(last_date),
        "stale": stale,
    }
