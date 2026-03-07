#!/usr/bin/env python3
"""
섹터 릴레이 엔진 실행 스크립트
================================
Usage:
  python scripts/run_relay_engine.py --update      # US 데이터 업데이트
  python scripts/run_relay_engine.py --signal       # 릴레이 경보 판정
  python scripts/run_relay_engine.py --telegram     # 텔레그램 전송
  python scripts/run_relay_engine.py --all          # 전부 실행
  python scripts/run_relay_engine.py --show         # 결과 표시만

BAT 연동:
  BAT-A (06:10): --update --signal --telegram
  BAT-B (08:00): --signal --telegram (재판정)
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# PYTHONPATH 안전장치
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.relay.config import load_relay_config, DATA_DIR
from src.relay.us_tracker import update_and_save, load_us_leaders
from src.relay.relay_engine import run_relay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("relay_runner")


def cmd_update():
    """US 대장주 데이터 업데이트."""
    print("\n" + "=" * 60)
    print("  US 대장주 데이터 업데이트")
    print("=" * 60)

    data = update_and_save()
    if not data:
        print("  US 데이터 업데이트 실패!")
        return False

    for sec_key, sec_data in data.items():
        name = sec_data.get("name", sec_key)
        leaders = sec_data.get("leaders", {})
        strong = sec_data.get("strong_count", 0)
        total = len(leaders)
        print(f"\n  [{name}] 강세 {strong}/{total}")
        for t, info in leaders.items():
            arrow = "+" if info.get("is_strong") else ("-" if info.get("is_weak") else "=")
            print(f"    {t:6s} ${info['close']:>8.2f} ({info['ret_1d']:+.1f}%) [{arrow}]"
                  f"  강화>${info['strength_level']:.2f}  약화<${info['weakness_level']:.2f}")

    print(f"\n  저장: {DATA_DIR / 'us_leaders.json'}")
    return True


def cmd_signal():
    """릴레이 경보 판정."""
    print("\n" + "=" * 60)
    print("  섹터 릴레이 경보 판정")
    print("=" * 60)

    result = run_relay()

    # 결과 출력
    print(f"\n  생성: {result['generated_at']}")
    print(f"  활성 경보: {result['active_alerts']}")
    print(f"  실행 준비: {result['execution_ready']}")
    print(f"  총 경보 점수: {result['total_alert_score']}")

    for sec_key, sec in result.get("sectors", {}).items():
        phase = sec["phase"]
        name = sec["name"]
        alert = sec["alert_level"]
        stars = "*" * alert

        if phase == 0:
            print(f"\n  [{name}] 비활성")
            continue

        print(f"\n  [{name}] Phase {phase} ({sec['phase_name']}) {stars}")
        print(f"    US 강세: {sec.get('us_strong_count', 0)}/{sec.get('us_min_strong', 2)}")
        print(f"    뉴스 점수: {sec.get('news_score', 0)}")
        if sec.get("news_keywords_matched"):
            print(f"    매칭 키워드: {', '.join(sec['news_keywords_matched'])}")
        print(f"    KR 대장주 강세: {sec.get('kr_leaders_strong', False)}")

        # KR 액션
        for name_kr, info in sec.get("kr_leaders_action", {}).items():
            print(f"    {name_kr}: {info['action']} — {info.get('reason', '')}")
        for name_kr, info in sec.get("kr_secondaries_action", {}).items():
            print(f"    {name_kr}: {info['action']} — {info.get('reason', '')}")

    print(f"\n  추천: {result['recommendation']}")
    print(f"\n  저장: {DATA_DIR / 'relay_signal.json'}")
    return result


def cmd_telegram(result: dict = None):
    """텔레그램 전송."""
    if result is None:
        signal_path = DATA_DIR / "relay_signal.json"
        if not signal_path.exists():
            print("  relay_signal.json 없음 — --signal 먼저 실행하세요")
            return
        result = json.loads(signal_path.read_text(encoding="utf-8"))

    msg = result.get("telegram_summary", "")
    if not msg:
        print("  텔레그램 메시지 없음")
        return

    print("\n  텔레그램 메시지:")
    print("-" * 40)
    print(msg)
    print("-" * 40)

    # 텔레그램 전송
    try:
        from src.telegram_sender import send_message
        send_message(msg)
        print("  텔레그램 전송 완료!")
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)
        print(f"  텔레그램 전송 실패: {e}")


def cmd_show():
    """저장된 결과 표시."""
    signal_path = DATA_DIR / "relay_signal.json"
    if not signal_path.exists():
        print("  relay_signal.json 없음")
        return

    result = json.loads(signal_path.read_text(encoding="utf-8"))
    print(f"\n  생성: {result.get('generated_at', '?')}")
    print(f"  활성: {result.get('active_alerts', [])}")
    print(f"  실행준비: {result.get('execution_ready', [])}")
    print(f"\n  추천: {result.get('recommendation', '')}")
    print(f"\n  텔레그램:")
    print(result.get("telegram_summary", ""))


def main():
    parser = argparse.ArgumentParser(description="섹터 릴레이 엔진")
    parser.add_argument("--update", action="store_true", help="US 대장주 데이터 업데이트")
    parser.add_argument("--signal", action="store_true", help="릴레이 경보 판정")
    parser.add_argument("--telegram", action="store_true", help="텔레그램 전송")
    parser.add_argument("--all", action="store_true", help="전부 실행")
    parser.add_argument("--show", action="store_true", help="결과 표시만")
    args = parser.parse_args()

    if not any([args.update, args.signal, args.telegram, args.all, args.show]):
        parser.print_help()
        return

    result = None

    if args.all or args.update:
        cmd_update()

    if args.all or args.signal:
        result = cmd_signal()

    if args.all or args.telegram:
        cmd_telegram(result)

    if args.show:
        cmd_show()


if __name__ == "__main__":
    main()
