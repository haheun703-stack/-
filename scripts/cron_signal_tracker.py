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

    # STEP 1: 성과 업데이트
    print("\n[1/3] 성과 업데이트...")
    try:
        from scripts.performance_tracker import update_performance
        result = update_performance(dry_run=dry_run)
        print(f"  완료: {result['updated']}/{result['total']}건 업데이트")
    except Exception as e:
        print(f"  [FAIL] 성과 업데이트 실패: {e}")
        send_error_alert("성과 업데이트", str(e))

    # STEP 2: 종료 판정
    print("\n[2/3] 종료 판정...")
    try:
        from scripts.signal_closer import close_signals
        result = close_signals(dry_run=dry_run)
        print(f"  완료: CLOSED {result['closed']} / STOPPED {result['stopped']} / OPEN {result['open']}")
    except Exception as e:
        print(f"  [FAIL] 종료 판정 실패: {e}")
        send_error_alert("종료 판정", str(e))

    # STEP 3: 성적표 집계
    print("\n[3/3] 성적표 집계...")
    try:
        from scripts.scoreboard_aggregator import aggregate_scoreboard
        rows = aggregate_scoreboard(dry_run=dry_run)
        print(f"  완료: {len(rows)}건 집계")
    except Exception as e:
        print(f"  [FAIL] 성적표 집계 실패: {e}")
        send_error_alert("성적표 집계", str(e))


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
