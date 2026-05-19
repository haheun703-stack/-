"""run_market_regime_gate.py — MarketRegimeGate CLI 진입점 (2026-05-19 신규)

배경:
  5/19 11:14 KST 실측: KODEX 200 -5.05% / KODEX 인버스 +5.05% /
  KODEX 200선물인버스2X +9.48%. 만약 5/20이었다면 매수 → 즉시 -3% 손절.
  5명 검수팀이 시장 약세는 검출 못 함 → 6번째 워커 신규.

사용:
  python scripts/run_market_regime_gate.py             # 텔레그램 + 출력
  python scripts/run_market_regime_gate.py --no-tg     # 출력만 (개발/CI)
  python scripts/run_market_regime_gate.py -v          # 상세 로그

종료 코드:
  0  — NORMAL (시장 정상)
  1  — BEARISH / STRONG_BEARISH (KILL_SWITCH 자동 활성화됨)

cron 등록 권장 (VPS, 5/20 가동 직전):
  55 13 20 5 * cd ~/quantum-master && ./venv/bin/python3.11 \
      scripts/run_market_regime_gate.py >> /tmp/market_regime.log 2>&1
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path 안전장치 (BAT/cron 환경 대응)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env 로드 (KIS API 키)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from src.agents.market_regime_gate import main

if __name__ == "__main__":
    sys.exit(main())
