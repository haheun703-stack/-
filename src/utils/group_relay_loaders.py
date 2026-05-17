"""그룹 릴레이 CSV 로더 — stock_data_daily 일봉 데이터 조회.

origin: scripts/archive/analysis/group_relay_backtest.py (2026-05-17 이전)
이전 사유: CLAUDE.md LOCK 규칙(`scripts/archive/` 참조 금지) 위반 해소.
group_relay_detector.py가 importlib 동적 로딩으로 우회하던 패턴을
정규 import로 전환하기 위해 정식 위치(src/utils/)로 이전.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STOCK_DATA_DIR = PROJECT_ROOT / "stock_data_daily"

logger = logging.getLogger(__name__)


def find_csv_by_ticker(ticker: str) -> Path | None:
    """stock_data_daily/에서 티커에 해당하는 CSV 찾기.

    파일명 형식: '종목명_티커.csv'
    """
    pattern = f"*_{ticker}.csv"
    matches = list(STOCK_DATA_DIR.glob(pattern))
    if matches:
        return matches[0]
    return None


def load_daily_closes(ticker: str, min_rows: int = 60) -> pd.Series | None:
    """티커의 일봉 종가 Series 반환 (index=Date)."""
    csv_path = find_csv_by_ticker(ticker)
    if csv_path is None:
        logger.debug("CSV 없음: %s", ticker)
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if len(df) < min_rows:
            logger.debug("데이터 부족: %s (%d행)", ticker, len(df))
            return None
        df = df.sort_values("Date")
        df = df.set_index("Date")
        return df["Close"]
    except Exception as e:
        logger.debug("CSV 로드 실패: %s — %s", ticker, e)
        return None


