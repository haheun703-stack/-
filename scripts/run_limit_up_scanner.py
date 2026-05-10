#!/usr/bin/env python3
"""
상한가 풀림 실시간 감지 — CLI 엔트리포인트

사용법:
  # 후보 생성 (BAT-D에서 매일 실행)
  python scripts/run_limit_up_scanner.py --generate

  # 장중 실시간 스캔 (dry-run)
  python scripts/run_limit_up_scanner.py --scan --dry-run

  # 장중 실시간 스캔 (live 주문)
  python scripts/run_limit_up_scanner.py --scan --live

  # 1회 스캔 (테스트)
  python scripts/run_limit_up_scanner.py --once

  # 후보 목록 확인
  python scripts/run_limit_up_scanner.py --status
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import yaml

from src.use_cases.limit_up_scanner import LimitUpScanner, CANDIDATES_PATH, STATE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """settings.yaml 로드"""
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def create_scanner(config: dict, live: bool = False) -> LimitUpScanner:
    """스캐너 인스턴스 생성 (어댑터 주입)"""
    # config override
    lu_cfg = config.setdefault("limit_up_scanner", {})
    if live:
        lu_cfg["dry_run"] = False
    else:
        lu_cfg["dry_run"] = True

    # KIS 어댑터 초기화
    intraday_adapter = None
    order_adapter = None

    try:
        from src.adapters.kis_intraday_adapter import KisIntradayAdapter
        intraday_adapter = KisIntradayAdapter()
        logger.info("✓ KIS Intraday 어댑터 초기화")
    except Exception as e:
        logger.error("✗ KIS Intraday 어댑터 실패: %s", e)

    if live:
        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            order_adapter = KisOrderAdapter()
            logger.info("✓ KIS Order 어댑터 초기화")
        except Exception as e:
            logger.error("✗ KIS Order 어댑터 실패: %s", e)

    return LimitUpScanner(
        intraday_adapter=intraday_adapter,
        order_adapter=order_adapter,
        config=config,
    )


def cmd_generate(config: dict):
    """후보 생성 (BAT-D용)"""
    print("=" * 60)
    print(" 상한가 풀림 감시 후보 생성")
    print("=" * 60)

    scanner = LimitUpScanner(config=config)
    candidates = scanner.generate_candidates()

    print(f"\n총 {len(candidates)}종목 후보 생성")
    print(f"저장: {CANDIDATES_PATH}")

    if candidates:
        print(f"\n{'순위':<4} {'종목':<8} {'상한가횟수':<10} {'마지막상한가':<12} {'경과일':<6} {'상한가(원)'}")
        print("-" * 60)
        for i, c in enumerate(candidates[:20], 1):
            print(
                f"{i:<4} {c['ticker']:<8} "
                f"{c['limit_up_count']:<10} "
                f"{c['last_limit_up']:<12} "
                f"{c['days_since_last']:<6} "
                f"{c['limit_price']:>10,}"
            )
        if len(candidates) > 20:
            print(f"  ... +{len(candidates) - 20}종목")


def cmd_scan(config: dict, live: bool):
    """장중 실시간 스캔"""
    mode = "🔥 LIVE" if live else "🧪 DRY-RUN"
    print("=" * 60)
    print(f" 상한가 풀림 실시간 감지 [{mode}]")
    print("=" * 60)

    scanner = create_scanner(config, live=live)
    scanner.load_candidates()

    if not scanner.candidates:
        print("❌ 후보 종목 없음. 먼저 --generate 실행 필요.")
        return

    print(f"\n후보: {len(scanner.candidates)}종목")
    print(f"설정: 폴링 {scanner.scan_interval}초 / 집중 {scanner.focus_interval}초")
    print(f"풀림 기준: 상한가 × {scanner.unlock_threshold}")
    print(f"포지션: {scanner.position_pct*100:.0f}% × 최대 {scanner.max_positions}종목")
    print(f"\n스캔 시작... (Ctrl+C로 중단)\n")

    scanner.run()

    # 종료 후 결과
    print(f"\n{'='*60}")
    print(f" 금일 결과")
    print(f"{'='*60}")
    print(f" 상한가 도달: {len(scanner.focus_list)}종목")
    print(f" 풀림 매수: {len(scanner.filled_today)}건")
    for entry in scanner.filled_today:
        print(
            f"   {entry['ticker']} @ {entry['entry_price']:,}원 "
            f"× {entry['quantity']}주 = {entry['amount']:,}원 "
            f"[{entry['status']}]"
        )


def cmd_once(config: dict):
    """1회성 스캔 (테스트)"""
    print("=" * 60)
    print(" 상한가 풀림 1회 스캔 (테스트)")
    print("=" * 60)

    scanner = create_scanner(config, live=False)
    scanner.load_candidates()

    if not scanner.candidates:
        print("❌ 후보 종목 없음.")
        return

    results = scanner.run_once()
    print(f"\n스캔 완료: {results['scanned']}종목 조회")
    print(f"상한가 도달: {len(results['limit_up_detected'])}종목")

    for det in results["limit_up_detected"]:
        status = "🟢 풀림" if det["is_unlocked"] else "🔴 잠김"
        print(
            f"  {status} {det['ticker']} — "
            f"현재가 {det['current_price']:,} / "
            f"고가 {det['high_price']:,} / "
            f"상한가 {det['limit_price']:,}"
        )

    if results["unlocked"]:
        print(f"\n✅ 풀린 종목: {', '.join(results['unlocked'])}")


def cmd_status():
    """후보/상태 확인"""
    print("=" * 60)
    print(" 상한가 풀림 감지 — 현재 상태")
    print("=" * 60)

    # 후보 목록
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"\n📋 후보: {len(candidates)}종목 ({CANDIDATES_PATH.name})")
        for c in candidates[:10]:
            print(
                f"  {c['ticker']} — 상한가 {c['limit_up_count']}회, "
                f"마지막 {c['last_limit_up']}, {c['days_since_last']}일 전"
            )
        if len(candidates) > 10:
            print(f"  ... +{len(candidates) - 10}종목")
    else:
        print("\n❌ 후보 파일 없음. --generate 실행 필요.")

    # 상태
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"\n📊 마지막 실행: {state.get('date', 'N/A')}")
        filled = state.get("filled_today", [])
        print(f"  체결: {len(filled)}건")
        for entry in filled:
            print(f"    {entry['ticker']} @ {entry['entry_price']:,}원 [{entry['status']}]")
    else:
        print("\n(실행 기록 없음)")


def main():
    parser = argparse.ArgumentParser(description="상한가 풀림 실시간 감지")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="후보 생성 (BAT-D용)")
    group.add_argument("--scan", action="store_true", help="장중 실시간 스캔")
    group.add_argument("--once", action="store_true", help="1회 스캔 (테스트)")
    group.add_argument("--status", action="store_true", help="현재 상태 확인")

    parser.add_argument("--live", action="store_true", help="실제 주문 (기본: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="주문 미실행 (기본)")

    args = parser.parse_args()
    config = load_config()

    if args.generate:
        cmd_generate(config)
    elif args.scan:
        cmd_scan(config, live=args.live)
    elif args.once:
        cmd_once(config)
    elif args.status:
        cmd_status()


if __name__ == "__main__":
    main()
