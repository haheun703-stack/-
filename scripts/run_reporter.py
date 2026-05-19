"""run_reporter.py — Reporter CLI 진입점 (2026-05-19 신규)

배경:
  5/20 퀀트봇 자율 자동매매 첫 가동의 검수팀(EnvChecker / CodeAuditor /
  FlowMonitor / DataIntegrity 4명) 결과를 종합해 사장님 카톡으로 1일 4회 보고.

  핵심 안전망: 정해진 시각에 카톡이 안 오면 = 검수팀 OFF.
  → 사장님이 즉시 알아챔 (단타봇 VERIFICATION_MODE=false 재발 방지).

사용:
  python scripts/run_reporter.py --slot morning_06        # 06:00 점호
  python scripts/run_reporter.py --slot pre_trade_1355    # 13:55 가동 5분 전
  python scripts/run_reporter.py --slot post_trade_1600   # 16:00 종료 직후
  python scripts/run_reporter.py --slot daily_close_1900  # 19:00 일일 마감
  python scripts/run_reporter.py --slot morning_06 --no-tg  # dry-run

종료 코드:
  0 — 정상
  2 — 무응답 워커 1명 이상 (cron 모니터링 신호)

cron 등록 예시 (VPS):
  # 매일 06:00 — 5명 점호
  0  6 * * * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_reporter.py --slot morning_06
  # 매일 13:55 — 가동 5분 전 ALL CHECK (5/20 D-Day에 가장 중요)
  55 13 * * * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_reporter.py --slot pre_trade_1355
  # 매일 16:00 — 종료 후 결과
  0  16 * * * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_reporter.py --slot post_trade_1600
  # 매일 19:00 — 일일 마감
  0  19 * * * cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_reporter.py --slot daily_close_1900
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path 안전장치 (BAT/cron 환경 대응)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.reporter import main

if __name__ == "__main__":
    sys.exit(main())
