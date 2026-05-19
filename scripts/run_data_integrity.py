"""DataIntegrity 에이전트 CLI 진입점.

cron 호출 시점:
- 06:30 (BAT-A 후 확인)
- 16:50 (BAT-D 후 확인)
- 18:50 (BAT-HEALTH 후 통합 확인)

실행:
    python -u -X utf8 scripts/run_data_integrity.py
    python -u -X utf8 scripts/run_data_integrity.py --no-tg   # 텔레그램 발송 끔
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_integrity import DataIntegrity


def main() -> int:
    parser = argparse.ArgumentParser(description="DataIntegrity 에이전트 — 시각별 데이터 도착 추적")
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 전송 끔 (로컬 점검용)")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 결과 출력")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = DataIntegrity()
    report = agent.build_report()
    result = report.to_dict()

    # 콘솔 요약 (한국어)
    summary = agent._format_telegram(report)
    print(summary)

    # 상세 (각 그룹별)
    print("\n────────── 상세 ──────────")
    print("[BAT 결과]")
    for c in report.bat_results:
        print(f"  {('OK' if c.ok else 'NG'):3} {c.name:18} {c.status:8} | {c.detail}")
    print("[정보봇/JGIS]")
    for c in report.intel_bot:
        print(f"  {('OK' if c.ok else 'NG'):3} {c.name:28} {c.status:8} | {c.detail}")
    if report.kis_token:
        c = report.kis_token
        print(f"[KIS 토큰]\n  {('OK' if c.ok else 'NG'):3} {c.name:18} {c.status:8} | {c.detail}")
    print("[사전 조건]")
    for c in report.prerequisite_files:
        print(f"  {('OK' if c.ok else 'NG'):3} {c.name:24} {c.status:8} | {c.detail}")

    if args.json:
        import json
        print("\n────────── JSON ──────────")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    # 텔레그램 발송
    if not args.no_tg:
        try:
            from src.telegram_sender import send_message
            send_message(summary)
        except Exception as e:
            logging.warning(f"텔레그램 전송 실패(무시): {e}")

    # 종료 코드: missing/stale 있으면 1, 정상이면 0
    has_issue = bool(report.missing or report.stale)
    return 1 if has_issue else 0


if __name__ == "__main__":
    sys.exit(main())
