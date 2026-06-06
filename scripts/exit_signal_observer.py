"""FLOWX Market OS v1 exit_signal_observer CLI (6단계).

사용:
    python -u -X utf8 scripts/exit_signal_observer.py --no-remote
    python -u -X utf8 scripts/exit_signal_observer.py --no-remote --no-write

매도 자동화 없음. "팔았으면 어땠나"만 관찰·기록한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.exit_signal_observer import run_exit_observer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 exit_signal_observer (observe only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="JSON/MD 저장 없이 콘솔 요약만")
    args = parser.parse_args()

    document, json_path, md_path = run_exit_observer(
        days=args.days, prefer_remote=not args.no_remote, write=not args.no_write
    )

    print("=" * 72)
    print("FLOWX Market OS v1 — exit_signal_observer (6단계, observe only)")
    print("=" * 72)
    print(f"observation_date: {document['observation_date']}")
    print(f"시장국면: {document['market_regime']} | 관찰 {document['counts']['observed']}종목")
    for o in document["observations"]:
        kinds = ", ".join(
            s.get("kind") or s.get("horizon") or str(s.get("level_pct"))
            for s in o.get("exit_signals_triggered", [])
        ) or "-"
        print(f"  {o['ticker']} {o['name']} ({o['tier']}) | 진입 {o['virtual_entry_price']} "
              f"→ 현재 {o['current_close']} | MFE {o['mfe_pct']} / MAE {o['mae_pct']} "
              f"| {o['hold_status']} | exit: {kinds}")
    if json_path:
        print(f"\n저장: {json_path}")
        print(f"저장: {md_path}")
    else:
        print("\n저장 없음 (--no-write)")
    s = document["safety"]
    print(f"real_order={s['real_order']} / sell_automation={s['sell_automation']} / "
          f"order_intent_created={s['order_intent_created']} / position_modified={s['position_modified']}")
    print("실주문 0 / 매도 자동화 BLOCKED / scheduler 변경 0 / SAJANG 변경 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
