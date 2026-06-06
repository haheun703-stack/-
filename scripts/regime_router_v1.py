"""FLOWX Market OS v1 regime router CLI.

사용:
    python -u -X utf8 scripts/regime_router_v1.py --no-remote
    python -u -X utf8 scripts/regime_router_v1.py --no-write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.regime_router_v1 import run_router  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 regime router (read-only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="JSON 저장 없이 콘솔 요약만")
    args = parser.parse_args()

    document, path = run_router(days=args.days, prefer_remote=not args.no_remote, write=not args.no_write)

    print("=" * 72)
    print("FLOWX Market OS v1 — regime_router_v1")
    print("=" * 72)
    print(f"as_of_date: {document['as_of_date']}")
    print(document["hard_gate_policy"])
    for ticker, route in document["routes"].items():
        if not route.get("data_available"):
            print(f"[{ticker}] DATA_UNAVAILABLE → 신규진입 차단 / SHADOW_ONLY")
            continue
        print(
            f"[{ticker}] {route.get('name')} | {route['hard_gate_regime']} "
            f"({route['c60_regime']}) | close/ma60={route['current_close']}/{route['current_ma60']} "
            f"| 신규진입={route['allow_new_entries']} | SmartEntry={route['smart_entry_observation']} "
            f"| 매도={route['sell_automation']}"
        )
        if route.get("shadow_labels"):
            labels = ", ".join(label["regime"] for label in route["shadow_labels"])
            print(f"  shadow_labels: {labels} (엔진 전환권 0)")
    if path:
        print(f"\n저장: {path}")
    else:
        print("\n저장 없음 (--no-write)")
    print("실주문 0 / scheduler 변경 0 / SAJANG 변경 0 / auto promotion 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
