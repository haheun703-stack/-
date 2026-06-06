"""FLOWX Market OS v1 morning plan CLI (3단계).

사용:
    python -u -X utf8 scripts/morning_plan_07.py --no-remote
    python -u -X utf8 scripts/morning_plan_07.py --no-remote --no-write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.morning_plan_07 import run_morning_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 morning plan (read-only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="JSON/MD 저장 없이 콘솔 요약만")
    args = parser.parse_args()

    plan, json_path, md_path = run_morning_plan(
        days=args.days, prefer_remote=not args.no_remote, write=not args.no_write
    )

    print("=" * 72)
    print("FLOWX Market OS v1 — morning_plan_07")
    print("=" * 72)
    print(f"as_of_date: {plan['as_of_date']}")
    print(f"시장 종합 정책: {plan['market_regime']}")
    print(f"한 줄: {plan['version']} | {plan['market_regime_rule']}")
    tiers = plan["tiers"]
    print(f"CORE {len(tiers['CORE'])} / WATCH {len(tiers['WATCH'])} / CONTROL {len(tiers['CONTROL'])}")
    for reason in plan["blocked_or_shadow_reason"]:
        print(f"  차단/shadow: {reason}")
    for warn in plan["data_warnings"]:
        print(f"  [경고] {warn}")
    if json_path:
        print(f"저장: {json_path}")
        print(f"저장: {md_path}")
    else:
        print("저장 없음 (--no-write)")
    print("실주문 0 / 매도 BLOCKED / scheduler 변경 0 / SAJANG 변경 0 / PAPER_OPEN 금지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
