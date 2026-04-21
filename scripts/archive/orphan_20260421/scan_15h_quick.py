#!/usr/bin/env python3
"""15시 장마감 직후 원클릭 수급 스캔

1) extend_parquet_data.py → 당일 최종 시세 갱신 (~2분)
2) collect_investor_bulk.py --date 오늘 --core-only → 당일 수급 수집 (~2분)
3) scan_type1_relay.py --top 10 → 수급 릴레이 시드 종목 출력

Usage:
    python scripts/scan_15h_quick.py              # 오늘 날짜 자동
    python scripts/scan_15h_quick.py --date 20260420  # 특정 날짜
    python scripts/scan_15h_quick.py --skip-price   # 시세 갱신 스킵 (수급+스캔만)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_step(step_name: str, cmd: list[str], timeout: int = 300) -> bool:
    """단계 실행 + 경과시간 출력."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            capture_output=False,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  OK ({elapsed:.1f}초)")
            return True
        else:
            print(f"\n  FAIL (returncode={result.returncode}, {elapsed:.1f}초)")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT ({timeout}초 초과)")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="15시 장마감 후 원클릭 수급 스캔")
    parser.add_argument("--date", type=str, default=None, help="날짜 (YYYYMMDD)")
    parser.add_argument("--skip-price", action="store_true", help="시세 갱신 스킵")
    parser.add_argument("--top", type=int, default=10, help="상위 N종목")
    args = parser.parse_args()

    today = args.date or datetime.now().strftime("%Y%m%d")

    print(f"\n{'#'*60}")
    print(f"  15시 수급 스캔 — {today}")
    print(f"  시작: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#'*60}")

    total_start = time.time()

    # Step 1: 시세 갱신
    if not args.skip_price:
        ok = run_step(
            "시세 갱신 (extend_parquet)",
            [PYTHON, "-u", "scripts/extend_parquet_data.py"],
            timeout=300,
        )
        if not ok:
            print("\n  WARNING: 시세 갱신 실패 — 기존 데이터로 계속 진행")

    # Step 2: 당일 수급 수집
    ok = run_step(
        f"수급 수집 ({today})",
        [PYTHON, "-u", "scripts/collect_investor_bulk.py", "--date", today, "--core-only"],
        timeout=300,
    )
    if not ok:
        print("\n  CRITICAL: 수급 수집 실패 — 스캔 불가")
        sys.exit(1)

    # Step 3: 수급 릴레이 스캔
    run_step(
        "수급 릴레이 스캔",
        [PYTHON, "-u", "-X", "utf8", "scripts/scan_type1_relay.py", "--top", str(args.top)],
        timeout=120,
    )

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"  완료: {datetime.now().strftime('%H:%M:%S')} (총 {total_elapsed:.1f}초)")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
