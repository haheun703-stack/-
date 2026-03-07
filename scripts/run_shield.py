"""SHIELD 포트폴리오 방어 시스템 실행

BAT-D에서 BRAIN 전에 실행.
usage:
    python -u -X utf8 scripts/run_shield.py
    python -u -X utf8 scripts/run_shield.py --send    # 텔레그램 발송
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shield import Shield, SHIELD_OUTPUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SHIELD 포트폴리오 방어 시스템")
    parser.add_argument("--send", action="store_true", help="텔레그램 경보 발송")
    args = parser.parse_args()

    print("=" * 60)
    print("  SHIELD -- 포트폴리오 방어 시스템")
    print("=" * 60)

    shield = Shield()
    report = shield.check()

    # 결과 출력
    print(f"\n종합 등급: {report.overall_level}")
    print(f"MDD: {report.mdd_status.current_mdd_pct:+.2f}% "
          f"(peak {report.mdd_status.peak_equity:,.0f}원)")
    print(f"섹터 오버랩: {len(report.sector_overlaps)}건")
    print(f"종목 경고: {len(report.stock_alerts)}건")
    print(f"시스템적 리스크: {report.systemic_risk.severity}")

    if report.telegram_message:
        print(f"\n{report.telegram_message}")

    print(f"\n저장: {SHIELD_OUTPUT_PATH}")

    # 텔레그램 발송 (ORANGE/RED만)
    if args.send:
        alert_levels = shield.shield_cfg.get("alert_levels", ["ORANGE", "RED"])
        if report.overall_level in alert_levels:
            try:
                from src.telegram_sender import send_message
                send_message(report.telegram_message)
                logger.info("SHIELD 텔레그램 경보 발송 완료")
            except Exception as e:
                logger.warning("텔레그램 발송 실패: %s", e)
        else:
            logger.info("SHIELD %s -- 발송 불필요 (기준: %s)",
                        report.overall_level, alert_levels)

    return report


if __name__ == "__main__":
    main()
