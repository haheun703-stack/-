"""[DEPRECATED 2026-05-17] FLOWX 모닝 브리핑 크론.

폐지 사유 (커밋 c095e9a, 옵션 C):
  - 의존하던 morning_briefing_generator.py / theme_scan_runner.py가
    scripts/archive/deprecated/로 폐기 (CLAUDE.md LOCK 규칙 적용 대상)
  - VPS crontab BAT-M_morning 라인 제거 완료 — 호출 경로 0건
  - 본 파일은 보존만 하고 실행 차단 (수동 실행 시 즉시 종료)

대체:
  - 통합 아침 브리핑은 BAT-B (07:00) scripts/run_morning_briefing.py 가 담당
  - 통합 함수는 src/use_cases/morning_briefing.build_unified_morning()

Usage (실행 차단됨 — 참조용):
    python scripts/cron_morning_briefing.py            # → [DEPRECATED] 출력 후 종료
    python scripts/cron_morning_briefing.py --dry-run  # → 동일
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
        msg = f"🚨 FLOWX 모닝 브리핑 실패\n\n단계: {step}\n에러: {error[:500]}"
        send_message(msg)
    except Exception:
        pass  # 에러 알림 자체가 실패하면 무시


def main():
    # [DEPRECATED 2026-05-17] CLAUDE.md LOCK 규칙(archive 참조 금지) 적용 + cron 폐지.
    # 의존 모듈(morning_briefing_generator, theme_scan_runner)이 archive로 폐기됨.
    # 대체: BAT-B → scripts/run_morning_briefing.py + src/use_cases/morning_briefing
    print("[DEPRECATED] cron_morning_briefing.py는 폐지됨 (커밋 c095e9a, 옵션 C).")
    print("            대체: BAT-B(07:00) → scripts/run_morning_briefing.py")
    return

    # ── 아래 코드는 보존만 (실행 차단) ──
    parser = argparse.ArgumentParser(description="FLOWX 모닝 브리핑 크론")
    parser.add_argument("--dry-run", action="store_true", help="업로드/발송 없이 생성만")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"  FLOWX 모닝 브리핑 | {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # ── STEP 1: 기존 데이터 수집 (테마 + 뉴스) ──
    print("\n[1/4] 기존 데이터 수집...")
    try:
        # 테마 스캔
        try:
            from scripts.theme_scan_runner import run_theme_scan
            run_theme_scan(use_grok=True, send_telegram=False)
            print("  [OK] 테마 스캔 완료")
        except Exception as e:
            print(f"  [WARN] 테마 스캔 스킵: {e}")

        # 뉴스 스캔
        try:
            from main import step_news_scan
            step_news_scan(send_telegram=False)
            print("  [OK] 뉴스 스캔 완료")
        except Exception as e:
            print(f"  [WARN] 뉴스 스캔 스킵: {e}")
    except Exception as e:
        print(f"  [WARN] 데이터 수집 일부 실패 (계속 진행): {e}")

    # ── STEP 2: 브리핑 생성 ──
    print("\n[2/4] 브리핑 생성...")
    try:
        from scripts.morning_briefing_generator import generate_morning_briefing
        briefing = generate_morning_briefing(date_str)
        print(f"  시장상태: {briefing['market_status']}")
        print(f"  추천: {len(briefing['news_picks'])}종목")
        print(f"  전체: {len(briefing['briefing_full'])}자 / 요약: {len(briefing['briefing_summary'])}자")
    except Exception as e:
        error_msg = f"브리핑 생성 실패: {e}\n{traceback.format_exc()}"
        print(f"  [FAIL] {error_msg}")
        send_error_alert("브리핑 생성", str(e))
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY-RUN] 업로드/발송 스킵")
        print("\n[전체 브리핑]")
        print(briefing["briefing_full"])
        return

    # ── STEP 3: Supabase 업로드 ──
    print("\n[3/4] Supabase 업로드...")
    try:
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        if uploader.is_active:
            ok = uploader.upload_morning_briefing(briefing)
            print(f"  Supabase: {'OK' if ok else 'FAIL'}")
            if not ok:
                send_error_alert("Supabase 업로드", "upload_morning_briefing 반환값 False")
        else:
            print("  [WARN] Supabase 미연결 — 스킵")
    except Exception as e:
        print(f"  [FAIL] Supabase 업로드 실패: {e}")
        send_error_alert("Supabase 업로드", str(e))

    # ── STEP 4: 텔레그램 발송 ──
    print("\n[4/4] 텔레그램 발송...")
    try:
        from src.telegram_sender import send_message

        # 개인채널: 전체 브리핑
        ok_full = send_message(briefing["briefing_full"])
        print(f"  개인채널 (전체): {'OK' if ok_full else 'FAIL'}")

        # FLOWX채널: 요약본 (별도 채널ID 필요 시 추가)
        # TODO: FLOWX 공개채널 ID 설정 후 활성화
        # ok_summary = send_message(briefing["briefing_summary"], chat_id=FLOWX_CHANNEL_ID)
        # print(f"  FLOWX채널 (요약): {'OK' if ok_summary else 'FAIL'}")
        print("  FLOWX채널: 채널ID 설정 후 활성화 예정")

    except Exception as e:
        print(f"  [FAIL] 텔레그램 발송 실패: {e}")
        send_error_alert("텔레그램 발송", str(e))

    print(f"\n{'='*50}")
    print(f"  완료 | {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
