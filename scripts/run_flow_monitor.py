"""FlowMonitor CLI 진입점 — 자동매수 흐름 추적 (2026-05-19 신규)

cron 추가 (VPS):
  # 5/20 D-Day 매 5분 14:00~15:30 (auto_buy_executor + owner_rule_monitor 종료 후)
  */5 14-15 * * 3 cd /home/ubuntu/quantum-master && /home/ubuntu/quantum-master/venv/bin/python scripts/run_flow_monitor.py >> /tmp/flow_monitor.log 2>&1

  # 5/21~5/23 이월 대비
  */5 14-15 * * 4,5,1 cd /home/ubuntu/quantum-master && /home/ubuntu/quantum-master/venv/bin/python scripts/run_flow_monitor.py >> /tmp/flow_monitor.log 2>&1

사용:
  python scripts/run_flow_monitor.py             # 추적 + 매수 0건 시 카톡 (matter_threshold 적용)
  python scripts/run_flow_monitor.py --no-tg     # 텔레그램 OFF (콘솔만)
  python scripts/run_flow_monitor.py --force-tg  # 항상 텔레그램 발송 (디버그)
  python scripts/run_flow_monitor.py --log /tmp/auto_buy_executor.log  # 로그 경로 명시
  python scripts/run_flow_monitor.py --json      # dict를 JSON으로 stdout 출력 (다른 스크립트 연계용)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.flow_monitor import FlowMonitor  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="자동매수 흐름 추적 (FlowMonitor)")
    parser.add_argument("--log", default=None, help="auto_buy_executor 로그 경로 (기본 /tmp/auto_buy_executor.log)")
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 OFF")
    parser.add_argument("--force-tg", action="store_true", help="matter_threshold 무시하고 무조건 발송")
    parser.add_argument("--json", action="store_true", help="dict를 JSON으로 stdout 출력")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    monitor = FlowMonitor(log_path=args.log)
    trace = monitor.trace_latest_run()

    if args.json:
        print(json.dumps(trace, ensure_ascii=False, indent=2))
    else:
        print(monitor.format_console(trace))

    if not args.no_tg:
        monitor.report_to_telegram(trace, force=args.force_tg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
