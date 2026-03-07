"""BRAIN 자본배분 실행 스크립트

BAT-D에서 매일 저녁 실행.
NIGHTWATCH + KOSPI 레짐 + VIX → ARM별 자본배분 결정.

Usage:
    python -u -X utf8 scripts/run_brain.py
    python -u -X utf8 scripts/run_brain.py --send     # 텔레그램 발송
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.brain import Brain, BRAIN_OUTPUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="BRAIN 자본배분 실행")
    parser.add_argument("--send", action="store_true", help="텔레그램 브리핑 발송")
    args = parser.parse_args()

    print("=" * 60)
    print("  BRAIN — 자본배분 중앙 두뇌 (Phase 1)")
    print("=" * 60)

    brain = Brain()
    decision = brain.compute()

    # 결과 출력
    print(f"\n{decision.briefing}")
    print(f"\n신뢰도: {decision.confidence:.0%}")
    print(f"저장: {BRAIN_OUTPUT_PATH}")

    # 텔레그램 발송
    if args.send:
        try:
            from src.telegram_sender import send_message
            send_message(decision.briefing)
            logger.info("텔레그램 브리핑 발송 완료")
        except Exception as e:
            logger.warning("텔레그램 발송 실패: %s", e)

    return decision


if __name__ == "__main__":
    main()
