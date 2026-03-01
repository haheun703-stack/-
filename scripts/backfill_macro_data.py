"""
Phase 2: L4 글로벌 매크로 데이터 backfill

VIX, USD/KRW, KOSPI, SOXX 4개 지수를 수집하여
data/macro/global_indices.parquet에 저장.

사용법:
  python scripts/backfill_macro_data.py
"""

import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.macro_adapter import MacroAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    adapter = MacroAdapter()

    start = "2019-01-01"
    end = "2026-02-14"

    logger.info(f"글로벌 매크로 데이터 수집: {start} ~ {end}")
    df = adapter.fetch_all(start, end)

    if df.empty:
        logger.error("매크로 데이터 수집 실패")
        sys.exit(1)

    logger.info(f"수집 완료: {len(df)}일, 컬럼: {list(df.columns)}")

    # 통계 출력
    for col in df.columns:
        non_null = df[col].notna().sum()
        logger.info(f"  {col}: {non_null}일 데이터")

    path = adapter.save(df)
    logger.info(f"저장 완료: {path}")


if __name__ == "__main__":
    main()
