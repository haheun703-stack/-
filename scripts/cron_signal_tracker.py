"""FLOWX STEP 2 크론 — 퀀트봇 시그널 트래킹 오케스트레이터.

2가지 모드:
  --mode log   : 시그널 기록 (08:20, BAT-D 이후)
  --mode track : 성과추적 → 종료판정 → 성적표 (16:10, 장마감 후)

BAT 스케줄:
  scripts/schedule_N_signal_log.bat    → 08:20 KST (시그널 기록)
  scripts/schedule_O_signal_track.bat  → 16:10 KST (성과+종료+성적표)

Usage:
    python scripts/cron_signal_tracker.py --mode log
    python scripts/cron_signal_tracker.py --mode track
    python scripts/cron_signal_tracker.py --mode log --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def send_error_alert(step: str, error: str):
    """실패 시 텔레그램 에러 알림."""
    try:
        from src.telegram_sender import send_message
        msg = f"🚨 FLOWX 시그널 트래커 실패\n\n단계: {step}\n에러: {error[:500]}"
        send_message(msg)
    except Exception:
        pass


def run_log_mode(dry_run: bool = False):
    """시그널 기록 모드 (08:20)."""
    print("\n[1/1] 시그널 기록...")
    try:
        from scripts.signal_logger import log_signals
        count = log_signals(dry_run=dry_run)
        print(f"  완료: {count}건 기록")
    except Exception as e:
        error_msg = f"시그널 기록 실패: {e}\n{traceback.format_exc()}"
        print(f"  [FAIL] {error_msg}")
        send_error_alert("시그널 기록", str(e))


def run_track_mode(dry_run: bool = False):
    """성과추적 모드 (16:10)."""

    # STEP 1~3: 성과추적/종료판정/성적표 — 모듈 아카이브됨 (4/20)
    # performance_tracker, signal_closer, scoreboard_aggregator →
    # scripts/archive/deprecated/ 이동 완료. Supabase signals 테이블로 대체됨.
    print("\n[1/3~3/3] 성과추적 기능 deprecated — Supabase signals 직접 관리로 전환됨")


def main():
    parser = argparse.ArgumentParser(description="FLOWX 시그널 트래커 크론")
    parser.add_argument("--mode", choices=["log", "track"], required=True,
                        help="log=시그널기록(08:20), track=성과추적(16:10)")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 실행")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    now = datetime.now()
    print(f"\n{'='*50}")
    print(f"  FLOWX 시그널 트래커 [{args.mode.upper()}] | {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    if args.mode == "log":
        run_log_mode(dry_run=args.dry_run)
    else:
        run_track_mode(dry_run=args.dry_run)

    print(f"\n{'='*50}")
    print(f"  완료 | {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
