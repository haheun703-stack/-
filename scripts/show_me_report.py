"""FLOWX Market OS v1 show_me_report CLI (8단계 — 마지막).

사용:
    python -u -X utf8 scripts/show_me_report.py --no-remote
    python -u -X utf8 scripts/show_me_report.py --no-remote --no-write

FLOWX 1~7 산출을 그림·표로 보여준다. 관측 전용 — 정책 변경권 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.show_me_report import run_show_me  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FLOWX Market OS v1 show_me_report (observation only)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 로드 끄기")
    parser.add_argument("--no-write", action="store_true", help="저장 없이 콘솔 요약만")
    parser.add_argument("--no-png", action="store_true", help="PNG 차트 생략(MD/JSON만)")
    args = parser.parse_args()

    doc, json_path, md_path, png_paths = run_show_me(
        days=args.days, prefer_remote=not args.no_remote,
        write=not args.no_write, render_png=not args.no_png,
    )

    print("=" * 72)
    print("FLOWX Market OS v1 — show_me_report (8단계, observation only)")
    print("=" * 72)
    print(f"observation_date: {doc['observation_date']} | 국면: {doc['market_regime']}")
    print(doc["one_line"])
    cf = doc["candidate_flow"]
    print(f"후보 {cf['candidate_count']} · CORE {cf['tier_counts'].get('CORE')} / "
          f"WATCH {cf['tier_counts'].get('WATCH')} / CONTROL {cf['tier_counts'].get('CONTROL')}")
    print(f"후보선정 기준={doc['candidate_performance'].get('basis')} / "
          f"실행 기준={doc['execution_performance'].get('basis')}")
    if json_path:
        print(f"\n저장: {json_path}")
        print(f"저장: {md_path}")
        for p in png_paths:
            print(f"저장: {p}")
        if not png_paths:
            print("PNG: 생성 안 됨(matplotlib 없음 또는 --no-png)")
    else:
        print("\n저장 없음 (--no-write)")
    sp = doc["safety_panel"]
    print(f"real_order={sp['real_order']} / sell_automation={sp['sell_automation']} / "
          f"paper_open_allowed={sp['paper_open_allowed']} / policy_changed={sp['policy_changed']}")
    print("실주문 0 / 매도 자동화 BLOCKED / PAPER_OPEN 금지 / scheduler·SAJANG 변경 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
