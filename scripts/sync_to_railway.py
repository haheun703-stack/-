"""
로컬 데이터 → Railway 서버 동기화

BAT-D 완료 후 호출:
  python scripts/sync_to_railway.py

환경변수:
  RAILWAY_URL    — Railway 배포 URL (예: https://xxx.railway.app)
  JARVIS_SECRET  — 동기화 토큰 (.env에서 로드)
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sync")

RAILWAY_URL = os.getenv("RAILWAY_URL", "")
SYNC_TOKEN = os.getenv("JARVIS_SECRET", "")
DATA_DIR = PROJECT_ROOT / "data"

# 동기화 대상 파일 목록
SYNC_FILES = [
    "tomorrow_picks.json",
    "etf_master.json",
    "picks_history.json",
    "overnight_signal.json",
    "sector_rotation/sector_momentum.json",
    "sector_rotation/sector_daily_report.json",
    "sector_rotation/etf_signal.json",
    "sector_rotation/sector_zscore.json",
    "sector_rotation/sector_investor_flow.json",
    "whale_detect.json",
    "dual_buying.json",
    "pullback_candidates.json",
    "group_relay_signals.json",
]


def sync_file(filename: str) -> bool:
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("[SKIP] %s 없음", filename)
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("[FAIL] %s 파싱 오류: %s", filename, e)
        return False

    try:
        resp = requests.post(
            f"{RAILWAY_URL}/api/sync",
            json={"file": filename, "data": data},
            headers={"X-Sync-Token": SYNC_TOKEN},
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("[OK] %s", filename)
            return True
        else:
            logger.error("[FAIL] %s → %d: %s", filename, resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.error("[FAIL] %s → %s", filename, e)
        return False


def main():
    if not RAILWAY_URL:
        logger.error("RAILWAY_URL 환경변수가 설정되지 않았습니다.")
        logger.info("  .env에 추가: RAILWAY_URL=https://xxx.railway.app")
        sys.exit(1)

    if not SYNC_TOKEN:
        logger.error("JARVIS_SECRET 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    logger.info("동기화 시작: %s → %s", DATA_DIR, RAILWAY_URL)

    ok = fail = skip = 0
    for f in SYNC_FILES:
        if (DATA_DIR / f).exists():
            if sync_file(f):
                ok += 1
            else:
                fail += 1
        else:
            skip += 1

    logger.info("동기화 완료: 성공 %d / 실패 %d / 스킵 %d", ok, fail, skip)


if __name__ == "__main__":
    main()
