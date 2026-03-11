#!/usr/bin/env python3
"""KRX 국적별 외국인 수급 스캔 — 일별 수집 + 추이 분석 + 텔레그램 발송

Usage:
    python scripts/scan_nationality.py                  # 전일 수집 + 분석
    python scripts/scan_nationality.py --backfill 10    # 최근 10거래일 백필
    python scripts/scan_nationality.py --date 20260310  # 특정 날짜 수집
    python scripts/scan_nationality.py --analyze        # 분석만 (수집 스킵)
    python scripts/scan_nationality.py --send           # 텔레그램 발송
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# PYTHONPATH 안전장치
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.krx_nationality_collector import collect_and_store, backfill, DB_PATH
from src.use_cases.nationality_signal import run_analysis, format_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    """텔레그램 발송."""
    try:
        from src.telegram_sender import send_message
        send_message(message)
        logger.info("텔레그램 발송 완료")
        return True
    except Exception as e:
        logger.error(f"텔레그램 발송 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="KRX 국적별 외국인 수급 스캔")
    parser.add_argument("--backfill", type=int, default=0,
                        help="최근 N거래일 백필")
    parser.add_argument("--date", type=str, default=None,
                        help="특정 날짜 수집 (YYYYMMDD)")
    parser.add_argument("--analyze", action="store_true",
                        help="분석만 실행 (수집 스킵)")
    parser.add_argument("--send", action="store_true",
                        help="텔레그램 발송")
    args = parser.parse_args()

    # ─── 백필 모드 ───
    if args.backfill > 0:
        logger.info(f"=== 백필 모드: 최근 {args.backfill}거래일 ===")
        results = backfill(days=args.backfill)
        for r in results:
            status = "✅" if r["status"] == "OK" else "⏭️" if r["status"] == "SKIP" else "❌"
            logger.info(f"  {r['date']}: {status} {r['stocks']}종목 {r['rows']}행")
        # 백필 후 분석도 실행
        signals = run_analysis()
        logger.info(f"분석 완료: {len(signals)}종목")
        return

    # ─── 수집 ───
    if not args.analyze:
        logger.info("=== 국적별 외국인 수급 수집 ===")
        result = collect_and_store(date=args.date)
        logger.info(
            f"수집: {result['date']} / {result['stocks']}종목 / "
            f"{result['rows']}행 / {result['status']}"
        )
        if result["status"] == "LOGIN_FAIL":
            logger.error("KRX 로그인 실패 — 중단")
            sys.exit(1)

    # ─── 분석 ───
    logger.info("=== 국적별 수급 분석 ===")
    signals = run_analysis()

    if not signals:
        logger.warning("분석 결과 없음 (데이터 부족, 최소 2거래일 필요)")
        return

    # 결과 출력
    print()
    print("=" * 65)
    print("  🌍 국적별 외국인 수급 시그널")
    print("=" * 65)

    for sig in signals:
        if sig.signal == "NEUTRAL":
            continue
        emoji = {"STRONG_BUY": "🔴", "BUY": "🟠", "CAUTION": "🟡", "SELL": "🔵"}.get(
            sig.signal, "⚪"
        )
        label = {"STRONG_BUY": "강력매수", "BUY": "매수", "CAUTION": "주의", "SELL": "매도"}.get(
            sig.signal, sig.signal
        )
        score_str = f"+{sig.score}" if sig.score > 0 else str(sig.score)
        print(f"\n  {emoji} {label}  {sig.name}  {score_str}점")
        for reason in sig.reasons:
            print(f"      · {reason}")

    # 요약
    summary = {
        "STRONG_BUY": sum(1 for s in signals if s.signal == "STRONG_BUY"),
        "BUY": sum(1 for s in signals if s.signal == "BUY"),
        "NEUTRAL": sum(1 for s in signals if s.signal == "NEUTRAL"),
        "CAUTION": sum(1 for s in signals if s.signal == "CAUTION"),
        "SELL": sum(1 for s in signals if s.signal == "SELL"),
    }
    print(f"\n  요약: 강매수 {summary['STRONG_BUY']} / 매수 {summary['BUY']} / "
          f"중립 {summary['NEUTRAL']} / 주의 {summary['CAUTION']} / 매도 {summary['SELL']}")
    print("=" * 65)

    # ─── 텔레그램 ───
    if args.send:
        msg = format_telegram(signals)
        send_telegram(msg)


if __name__ == "__main__":
    main()
