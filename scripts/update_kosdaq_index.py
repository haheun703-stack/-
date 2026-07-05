"""KOSDAQ 인덱스 업데이트 (레짐 판별용) — update_kospi_index.py 동형.

기존 data/kosdaq_index.csv를 보존하며 최신만 추가. 없으면 yfinance max 수집.
FV 엔진 v1-3번(시장별 국면)의 입력. yfinance ^KQ11 (VPS 고정IP 화이트리스트).

사용: python -u -X utf8 scripts/update_kosdaq_index.py
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "data" / "kosdaq_index.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _load_existing() -> pd.DataFrame | None:
    if not OUT.exists():
        return None
    try:
        df = pd.read_csv(OUT, index_col="Date", parse_dates=True)
        return df.sort_index() if len(df) >= 2 else None
    except Exception as e:
        logger.warning("기존 CSV 읽기 실패: %s", e)
        return None


def _fetch_yfinance(start: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = (yf.download("^KQ11", start=start, progress=False) if start
          else yf.download("^KQ11", period="max", progress=False))
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [c[0].lower() for c in df.columns]
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def main() -> int:
    existing = _load_existing()
    if existing is not None and len(existing) >= 60:
        last_date = existing.index[-1]
        start_str = (last_date - timedelta(days=3)).strftime("%Y-%m-%d")
        logger.info("기존 %d행 (%s~%s), %s부터 보충", len(existing),
                    existing.index[0].date(), last_date.date(), start_str)
        new_df = _fetch_yfinance(start=start_str)
        if new_df.empty:
            logger.warning("yfinance 신규 없음(휴장)")
            print(f"KOSDAQ 기존 유지: {len(existing)} rows")
            return 0
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        logger.info("신규 %d행 → 총 %d행", len(combined) - len(existing), len(combined))
    else:
        logger.info("기존 없음/부족 — yfinance max 수집")
        combined = _fetch_yfinance()
        if combined.empty:
            logger.error("yfinance 수집 실패")
            return 1
        logger.info("수집 %d행 (%s~%s)", len(combined),
                    combined.index[0].date(), combined.index[-1].date())
    combined.to_csv(OUT)
    print(f"KOSDAQ {len(combined)} rows saved → {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
