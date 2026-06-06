"""FLOWX Market OS v1 engine policy map CLI (2단계).

사용:
    python -u -X utf8 scripts/engine_policy_map.py --no-remote
    python -u -X utf8 scripts/engine_policy_map.py --no-write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.engine_policy_map import run_policy_map  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 engine policy map (read-only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="JSON 저장 없이 콘솔 요약만")
    args = parser.parse_args()

    policy, path = run_policy_map(
        days=args.days, prefer_remote=not args.no_remote, write=not args.no_write
    )

    print("=" * 72)
    print("FLOWX Market OS v1 — engine_policy_map")
    print("=" * 72)
    print(f"as_of_date: {policy['as_of_date']}")
    print(f"시장국면(보수적 통합): {policy['market_regime']}")
    print(policy["market_regime_rule"])
    print("-" * 72)
    print("엔진 정책:")
    for engine, mode in policy["engines"].items():
        print(f"  {engine:14s} -> {mode}")
    print("-" * 72)
    print("기초자산별:")
    for ticker, info in policy["per_ticker"].items():
        if not info.get("data_available"):
            print(f"  [{ticker}] DATA_UNAVAILABLE")
            continue
        hyst = " (히스테리시스 대기)" if info.get("in_hysteresis_window") else ""
        print(f"  [{ticker}] {info['hard_gate_regime']} (raw={info.get('c60_regime_raw')}){hyst}")
    if policy["shadow_advisories"]:
        regs = ", ".join(f"{a['ticker']}:{a['regime']}" for a in policy["shadow_advisories"])
        print(f"shadow advisories(권한 0): {regs}")
    if path:
        print(f"\n저장: {path}")
    else:
        print("\n저장 없음 (--no-write)")
    print(f"매도 자동화: {policy['sell_automation']} / PAPER_OPEN: {policy['paper_open_allowed']}")
    print("실주문 0 / scheduler 변경 0 / SAJANG 변경 0 / auto promotion 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
