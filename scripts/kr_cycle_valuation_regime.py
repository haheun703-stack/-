"""한국 장기 사이클/밸류 레짐 수동 실행 스크립트.

사용:
  python -u -X utf8 scripts/kr_cycle_valuation_regime.py
  python -u -X utf8 scripts/kr_cycle_valuation_regime.py --no-write

출력:
  data_store/regime/kr_cycle_valuation_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.use_cases.kr_cycle_valuation_regime import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="파일 저장 없이 콘솔 출력만 수행")
    args = parser.parse_args()

    report, path = run(write=not args.no_write)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if path:
        print(f"\n[SAVED] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
