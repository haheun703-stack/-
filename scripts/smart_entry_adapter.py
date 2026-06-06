"""FLOWX Market OS v1 smart_entry_adapter CLI (5단계).

사용:
    python -u -X utf8 scripts/smart_entry_adapter.py --no-remote
    python -u -X utf8 scripts/smart_entry_adapter.py --no-remote --no-write

--paper-open은 6/8 전 비활성(SHADOW_OPEN 유지). PAPER_OPEN 금지.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.smart_entry_adapter import run_smart_entry_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 smart_entry_adapter (shadow only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="JSON 저장 없이 콘솔 요약만")
    parser.add_argument("--paper-open", action="store_true",
                        help="6/8 전 비활성 — 무시되고 SHADOW_OPEN 유지(PAPER_OPEN 금지)")
    args = parser.parse_args()

    if args.paper_open:
        print("[차단] --paper-open 무시: 6/8 전 PAPER_OPEN 금지 → SHADOW_OPEN 유지")

    document, path = run_smart_entry_adapter(
        days=args.days, prefer_remote=not args.no_remote, write=not args.no_write
    )

    print("=" * 72)
    print("FLOWX Market OS v1 — smart_entry_adapter (5단계, shadow only)")
    print("=" * 72)
    print(f"as_of_date: {document['as_of_date']}")
    print(f"시장국면: {document['market_regime']} | smart_entry={document['smart_entry_mode']}")
    c = document["counts"]
    print(f"SHADOW_OPEN {c['shadow_open']} / CONTROL_ONLY {c['control_only']} / BLOCKED {c['blocked']}")
    for e in document["shadow_entries"]:
        print(f"  [SHADOW_OPEN] {e['ticker']} {e['name']} ({e['tier']}) 진입참조 {e['close']} 손절 {e['stop_loss']}")
    for b in document["blocked"]:
        print(f"  [{b['status']}] {b['ticker']} {b['name']} ({b['tier']})")
    if path:
        print(f"\n저장: {path}")
    else:
        print("\n저장 없음 (--no-write)")
    s = document["safety"]
    print(f"real_order={s['real_order']} / order_adapter={s['order_adapter']} / "
          f"dry_run={s['dry_run']} / sell_automation={s['sell_automation']}")
    print("실주문 0 / scheduler 변경 0 / SAJANG 변경 0 / PAPER_OPEN 금지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
