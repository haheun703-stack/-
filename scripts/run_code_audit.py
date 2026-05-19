"""CodeAuditor CLI — 5/20 가동 직전·매일 18:00·커밋 직후 자동 검수 (2026-05-19 신규).

사용:
  python scripts/run_code_audit.py              # 7파일 전체 검수 + 텔레그램
  python scripts/run_code_audit.py --no-tg      # 텔레그램 OFF (콘솔만)
  python scripts/run_code_audit.py --file scripts/auto_buy_executor.py  # 단일 파일

호출 시점:
  - post-commit git hook (커밋 직후)
  - cron: 0 18 * * 1-5 (평일 18:00 장마감 후)
  - 5/20 13:55 (가동 직전 최종 검수)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.agents.code_auditor import CodeAuditor


def main() -> int:
    parser = argparse.ArgumentParser(description="CodeAuditor — 5/20 가동 자동 검수")
    parser.add_argument("--file", help="단일 파일 검수 (PROJECT_ROOT 기준 상대 경로)")
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 발송 OFF")
    parser.add_argument("--json", action="store_true", help="JSON 출력 (cron 파이프 친화)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    auditor = CodeAuditor()

    if args.file:
        findings = auditor.audit_file(args.file)
        result = {
            "critical": [f for f in findings if f["severity"] == "CRITICAL"],
            "high": [f for f in findings if f["severity"] == "HIGH"],
            "medium": [f for f in findings if f["severity"] == "MEDIUM"],
        }
        result["counts"] = {
            "critical": len(result["critical"]),
            "high": len(result["high"]),
            "medium": len(result["medium"]),
        }
        result["passed"] = not result["critical"] and not result["high"]
        # S1 표준 필드도 단일 파일 모드에 부여
        result["agent"] = "code_auditor"
        if result["counts"]["critical"] > 0:
            result["status"] = "FAIL"
        elif result["counts"]["high"] > 0 or result["counts"]["medium"] > 0:
            result["status"] = "WARN"
        else:
            result["status"] = "OK"
        result["summary"] = (
            f"CRITICAL {result['counts']['critical']} | "
            f"HIGH {result['counts']['high']} | "
            f"MEDIUM {result['counts']['medium']}"
        )
    else:
        result = auditor.audit_all()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        counts = result.get("counts", {"critical": 0, "high": 0, "medium": 0})
        print("=" * 70)
        print(f"  CodeAuditor 결과 — CRITICAL {counts['critical']} | HIGH {counts['high']} | MEDIUM {counts['medium']}")
        print("=" * 70)
        for sev in ("critical", "high", "medium"):
            items = result[sev]
            if not items:
                continue
            print(f"\n[{sev.upper()}] {len(items)}건")
            for f in items:
                print(f"  - {f['file']}:{f['line']}  {f['rule']}")
                print(f"      → {f['msg']}")
        if result["passed"]:
            print("\n✅ 7파일 검수 통과 (CRITICAL/HIGH 0건)")
        else:
            print("\n🚨 위험 발견 — 5/20 가동 검토 필요")

    if not args.no_tg:
        try:
            auditor.report_to_telegram(result)
        except Exception as e:
            logging.warning("텔레그램 발송 실패 (무시): %s", e)

    # 종료코드: CRITICAL/HIGH 있으면 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
