"""run_env_check.py — EnvChecker CLI 진입점 (2026-05-19 신규)

배경:
  5/19 단타봇 D-Day 사고 (VERIFICATION_MODE=false 토글 OFF로 매매 0건) 방지.
  5/20 퀀트봇 자율 자동매매 첫 가동 전 .env / 사전조건 / cron 점검.

사용:
  python scripts/run_env_check.py             # 텔레그램 + 출력
  python scripts/run_env_check.py --no-tg     # 출력만 (개발/CI)
  python scripts/run_env_check.py -v          # 상세 로그

종료 코드:
  0  — 전 항목 OK
  1  — 1건 이상 FAIL

cron 등록 예시 (VPS):
  # 매일 06:00 (BAT-A 직전)
  0 6 * * * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_env_check.py
  # 5/20 13:55 (가동 5분 전)
  55 13 20 5 * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_env_check.py
  # 5/20 14:00-14:55 매 5분 (auto_buy_executor 동기)
  */5 14 20 5 * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_env_check.py --no-tg
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path 안전장치 (BAT/cron 환경 대응)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.env_checker import main

if __name__ == "__main__":
    sys.exit(main())
