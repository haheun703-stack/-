"""FRED API로 CPI/PCE/실업률 데이터 수집 — BAT-D 19.91단계"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# PYTHONPATH 안전장치
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from src.macro.cpi_tracker import CPITracker

    tracker = CPITracker()
    result = tracker.update()

    cpi = result.get("cpi_yoy", "N/A")
    pce = result.get("core_pce_yoy", "N/A")
    unemp = result.get("unemployment_rate", "N/A")
    stag = result.get("stagflation_signal", "N/A")

    logger.info(f"CPI YoY: {cpi}%, Core PCE: {pce}%, 실업률: {unemp}%")
    logger.info(f"스태그플레이션: {stag}")


if __name__ == "__main__":
    main()
