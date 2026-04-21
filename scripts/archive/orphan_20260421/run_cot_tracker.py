"""COT Tracker 실행 스크립트.

Usage: python -u -X utf8 scripts/run_cot_tracker.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cot_tracker import CotTracker


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print("  COT Tracker — Slow Eye 주간 시그널")
    print("=" * 60)

    tracker = CotTracker()
    result = tracker.compute()

    if not result.get("contracts"):
        print(f"\n시그널 생성 실패: {result.get('reason', '알 수 없음')}")
        return

    print(f"\nCOT 복합 방향: {result['composite_direction']} "
          f"(score={result['composite_score']:+.3f})")
    print(f"데이터: {result['report_date']} ({result['stale_days']}일 전)")

    print("\n계약별 포지션:")
    for name, c in result["contracts"].items():
        z = c.get("z", 0)
        slow_z = c.get("slow_z", z)
        fast_z = c.get("fast_z", z)
        fast_used = c.get("fast_z_used", False)
        direction = c.get("direction", "N/A")
        net = c.get("net", 0)
        chg_1w = c.get("net_change_1w", 0)
        pct = c.get("percentile_52w", 50)
        z_tag = "FAST" if fast_used else "slow"
        print(f"  {c['label']:>16s}: net={net:>+12,d}  z={z:+.2f}[{z_tag}]  "
              f"(s={slow_z:+.2f}/f={fast_z:+.2f})  "
              f"{direction:>16s}  1W={chg_1w:>+10,d}  pct={pct:.0f}%")

    print("\n시그널:")
    for sig, val in result["signals"].items():
        status = "ON" if val else "off"
        print(f"  {sig:>16s}: {status}")


if __name__ == "__main__":
    main()
