"""FLOWX Market OS v1 daily_review CLI (7단계).

사용:
    python -u -X utf8 scripts/daily_review.py --no-remote
    python -u -X utf8 scripts/daily_review.py --no-remote --no-write

후보선정 성능(as_of 종가)과 실행 성능(virtual_entry_price)을 분리 복기. 매매·주문 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.daily_review import run_daily_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 daily_review (review only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="JSON/MD 저장 없이 콘솔 요약만")
    args = parser.parse_args()

    doc, json_path, md_path = run_daily_review(
        days=args.days, prefer_remote=not args.no_remote, write=not args.no_write
    )

    print("=" * 72)
    print("FLOWX Market OS v1 — daily_review (7단계, review only)")
    print("=" * 72)
    print(f"observation_date: {doc['observation_date']} | 국면: {doc['market_regime']}")
    print(doc["one_line"])
    cr = doc["candidate_performance"]
    er = doc["execution_performance"]
    print(f"[A 후보선정] 기준={cr['basis']} | 후보 {cr['candidate_count']} | "
          f"missed_winner {len(cr['missed_winner'])} / false_positive {len(cr['false_positive'])}")
    print(f"[B 실행성능] 기준={er['basis']} | 관찰 {er['entry_count']}")
    print(f"[C exit] {doc['exit_observer_summary']['exit_type_trigger_counts']}")
    if doc["data_warnings"]:
        print(f"[경고] {len(doc['data_warnings'])}건 OHLCV 없음")
    if json_path:
        print(f"\n저장: {json_path}")
        print(f"저장: {md_path}")
    else:
        print("\n저장 없음 (--no-write)")
    s = doc["safety"]
    print(f"real_order={s['real_order']} / sell_automation={s['sell_automation']} / "
          f"policy_changed={s['policy_changed']} / position_modified={s['position_modified']}")
    print("실주문 0 / 매도 자동화 BLOCKED / scheduler 변경 0 / SAJANG 변경 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
