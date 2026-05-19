"""run_market_scan.py — MarketScanner CLI 진입점 (2026-05-19 신규, v2 5분 단축).

배경 (5/19 사장님 지시 v2 13:40):
  사장님이 1년 동안 짚어주신 패턴 "AI가 코드만 만지고 시장 안 봄"을
  5분 자동 cron + 메인 AI 응답 시작 시 latest.json 자동 읽기로 해소.

  v2 강화:
  - "15분마다는 부족" → 5분 cron으로 단축
  - "분봉도 봐야지" → 직전 5분 스캔 대비 변동률 자동 계산
  - "삼성/SK하이닉스 대장이니까" → 대장주 2종 추가 (총 8종)

차이 (vs MarketRegimeGate):
  - MarketRegimeGate: 13:55 + 14:00~14:55 매 5분 (5/20만), KILL_SWITCH 활성화.
  - MarketScanner: 매일 09:00~15:45 매 5분, 정보만 (KILL_SWITCH 안 함).

사용:
  python scripts/run_market_scan.py

cron 등록 권장 (VPS, 평일 09:00~15:45 매 5분):
  */5 9-15 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 \
      scripts/run_market_scan.py >> /tmp/market_scan.log 2>&1
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

from src.agents.market_scanner import main

if __name__ == "__main__":
    sys.exit(main())
