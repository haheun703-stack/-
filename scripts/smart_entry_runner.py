"""
AI 스마트 진입 실행기 — BAT-E에서 호출

사용법:
  python scripts/smart_entry_runner.py                 # 전체 세션 (08:55~10:30)
  python scripts/smart_entry_runner.py --analysis      # 분석만 (주문 없음, 장외 테스트)
  python scripts/smart_entry_runner.py --live           # 실전 매수 (1단계: 1종목 50만원)

기본값: dry_run=True (실제 주문 안 나감)

안전장치:
  - --live 시 시작 전 Y/N 확인 (--force로 스킵 가능)
  - max_stocks / max_amount는 settings.yaml smart_entry.live 섹션에서 관리
  - data/KILL_SWITCH 파일 생성 시 즉시 중단
  - 당일 중복 실행 방지 (order_audit.db)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smart_entry")


def main():
    parser = argparse.ArgumentParser(description="AI 스마트 진입 실행기")
    parser.add_argument("--analysis", action="store_true",
                        help="분석만 수행 (주문 없음, 장외 테스트)")
    parser.add_argument("--live", action="store_true",
                        help="실전 매수 모드 (dry_run=False)")
    parser.add_argument("--force", action="store_true",
                        help="--live 시 확인 프롬프트 스킵")
    parser.add_argument("--picks", type=str, default=None,
                        help="추천 종목 JSON 경로 (기본: data/tomorrow_picks.json)")
    args = parser.parse_args()

    # KIS 어댑터 초기화
    from src.adapters.kis_intraday_adapter import KisIntradayAdapter
    intraday = KisIntradayAdapter()

    order_adapter = None
    dry_run = not args.live  # --live 플래그 없으면 dry_run=True

    if args.live:
        # ─── 라이브 모드 확인 ───
        from src.adapters.kis_order_adapter import KisOrderAdapter
        order_adapter = KisOrderAdapter()

        if not args.force:
            print("=" * 50)
            print("  ⚠️  실전 매수 모드 (LIVE)")
            print("  실제 주문이 한국투자증권으로 전송됩니다.")
            print("  킬스위치: data/KILL_SWITCH 파일 생성 시 즉시 중단")
            print("=" * 50)
            confirm = input("  계속하시겠습니까? (Y/n): ").strip()
            if confirm.lower() not in ("y", "yes", ""):
                print("  → 취소됨")
                return

        logger.warning("=" * 50)
        logger.warning("  실전 매수 모드 (LIVE) 활성화!")
        logger.warning("  실제 주문이 한국투자증권으로 전송됩니다.")
        logger.warning("=" * 50)

    # 설정 로드
    import yaml
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    config = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # SmartEntryEngine 생성
    from src.use_cases.smart_entry import SmartEntryEngine
    engine = SmartEntryEngine(
        intraday_adapter=intraday,
        order_adapter=order_adapter,
        dry_run=dry_run,
        config=config,
    )

    # 추천 종목 경로
    picks_path = args.picks or str(
        Path(__file__).resolve().parent.parent / "data" / "tomorrow_picks.json"
    )

    # 실행 모드 분기
    if args.analysis:
        logger.info("[모드] 분석 전용 (Analysis Only)")
        report = engine.run_analysis_only()
    else:
        logger.info("[모드] %s", "LIVE 세션" if args.live else "DRY-RUN 세션")
        report = engine.run_full_session()

    # 결과 출력
    mode_str = "LIVE" if not dry_run else "DRY-RUN"
    print(f"\n{'='*50}")
    print(f"[결과] 대상 {report['total_candidates']}종목 ({mode_str})")
    print(f"  체결: {report['filled']}  |  스킵: {report['skipped']}  |  미체결: {report['unfilled']}")

    if not dry_run:
        live_cfg = config.get("smart_entry", {}).get("live", {})
        print(f"  안전: max_stocks={live_cfg.get('max_stocks', 1)}, "
              f"max_amount={live_cfg.get('max_amount_per_stock', 500000):,}원")

    for d in report.get("details", []):
        emoji = {"buy": "O", "skip": "X", "wait": "-", "holding": "="}.get(d["decision"], "?")
        print(f"  [{emoji}] {d['name']}({d['ticker']}) "
              f"갭{d['gap_pct']:+.1f}% [{d['gap_type']}] "
              f"→ {d['decision']} (지정가 {d['order_price']:,})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
