"""KOSPI 인덱스 업데이트 (레짐 판별용).

기존 data/kospi_index.csv를 보존하면서 최신 데이터만 추가한다.
파일이 없거나 비어 있으면 yfinance에서 최대한 수집(max 기간).

사용: python -u -X utf8 scripts/update_kospi_index.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "kospi_index.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_existing() -> pd.DataFrame | None:
    """기존 CSV 로드. 비어 있거나 파싱 실패 시 None."""
    if not OUT.exists():
        return None
    try:
        df = pd.read_csv(OUT, index_col="Date", parse_dates=True)
        if df.empty or len(df) < 2:
            return None
        return df.sort_index()
    except Exception as e:
        logger.warning("기존 CSV 읽기 실패: %s", e)
        return None


def _fetch_yfinance(start: str | None = None) -> pd.DataFrame:
    """yfinance에서 KOSPI(^KS11) 데이터 다운로드.

    start가 None이면 최대 기간(max) 수집.
    """
    import yfinance as yf

    if start:
        df = yf.download("^KS11", start=start, progress=False)
    else:
        df = yf.download("^KS11", period="max", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [c[0].lower() for c in df.columns]
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def main():
    existing = _load_existing()

    if existing is not None and len(existing) >= 60:
        # 기존 데이터 보존 — 마지막 날짜부터 최신만 추가
        last_date = existing.index[-1]
        start_str = (last_date - timedelta(days=3)).strftime("%Y-%m-%d")
        logger.info(
            "기존 데이터: %d행 (%s ~ %s), %s부터 보충",
            len(existing), existing.index[0].date(), last_date.date(), start_str,
        )

        new_df = _fetch_yfinance(start=start_str)
        if new_df.empty:
            logger.warning("yfinance 신규 데이터 없음 (주말/휴장일)")
            print(f"KOSPI 기존 유지: {len(existing)} rows")
            return

        # 병합 (yfinance 우선 — 최신 데이터가 더 정확)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        added = len(combined) - len(existing)
        logger.info("신규 %d행 추가 → 총 %d행", added, len(combined))
    else:
        # 파일 없거나 데이터 부족 — 전체 수집
        logger.info("기존 데이터 없음/부족 — yfinance max 수집")
        combined = _fetch_yfinance()
        if combined.empty:
            logger.error("yfinance 수집 실패")
            sys.exit(1)
        logger.info("수집: %d행 (%s ~ %s)",
                     len(combined), combined.index[0].date(), combined.index[-1].date())

    combined.to_csv(OUT)
    print(f"KOSPI {len(combined)} rows saved → {OUT.name}")


if __name__ == "__main__":
    main()
