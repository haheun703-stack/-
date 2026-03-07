"""유동성 사이클 시그널 실행 스크립트.

Usage: python -u -X utf8 scripts/run_liquidity_signal.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.liquidity_tracker import LiquidityTracker


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print("  유동성 사이클 — FRED 5대 지표 중기 시그널")
    print("=" * 60)

    tracker = LiquidityTracker()
    result = tracker.compute()

    if not result.get("indicators"):
        print(f"\n시그널 생성 실패: {result.get('reason', '알 수 없음')}")
        return

    print(f"\n데이터: {result['data_date']} ({result['stale_days']}일 전)")
    print(f"유동성 레짐: {result['regime']}")
    print(f"복합 방향: {result['composite_direction']} "
          f"(score={result['composite_score']:+.3f})")

    print("\n지표 상세:")
    ind = result["indicators"]

    nl = ind.get("net_liquidity", {})
    print(f"  {'Net Liquidity':>16s}: {nl.get('value', 0):,.1f}B "
          f"(z={nl.get('z', 0):+.2f}, 20d: {nl.get('change_20d', 0):+,.1f}B)")

    m2 = ind.get("m2_yoy_pct", {})
    print(f"  {'M2 YoY':>16s}: {m2.get('value', 0):+.2f}% (z={m2.get('z', 0):+.2f})")

    res = ind.get("reserves", {})
    print(f"  {'은행 지준':>16s}: {res.get('value', 0):,.1f}B "
          f"(z={res.get('z', 0):+.2f}, 20d: {res.get('change_20d', 0):+,.1f}B)")

    rrp = ind.get("rrp", {})
    tga = ind.get("tga", {})
    print(f"  {'RRP':>16s}: {rrp.get('value', 0):,.1f}B (20d: {rrp.get('change_20d', 0):+,.1f}B)")
    print(f"  {'TGA':>16s}: {tga.get('value', 0):,.1f}B (20d: {tga.get('change_20d', 0):+,.1f}B)")

    walcl = ind.get("walcl", {})
    print(f"  {'WALCL':>16s}: {walcl.get('value', 0):,.1f}B (20d: {walcl.get('change_20d', 0):+,.1f}B)")

    print("\n시그널:")
    for sig, val in result["signals"].items():
        status = "ON" if val else "off"
        print(f"  {sig:>24s}: {status}")


if __name__ == "__main__":
    main()
