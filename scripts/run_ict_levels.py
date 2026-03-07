"""
ICT 프리미엄 레벨 + OR/IR + Equal Level 계산 CLI 래퍼

사용법:
  python scripts/run_ict_levels.py                    # 오늘 프리미엄 레벨 + OR/IR + Equal Level 계산
  python scripts/run_ict_levels.py --backfill 14      # 과거 14일 OR/IR 백필
  python scripts/run_ict_levels.py --accuracy          # daily_bias 정확도 측정
  python scripts/run_ict_levels.py --date 2026-03-06   # 특정 날짜
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.ict.premium_levels import compute_premium_levels, format_premium_briefing
from src.ict.opening_range import compute_or_ir, backfill_or_ir, measure_bias_accuracy, format_or_ir_briefing
from src.ict.equal_level_detector import compute_equal_levels, format_equal_levels_briefing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ict_levels")


def main():
    parser = argparse.ArgumentParser(description="ICT 프리미엄 레벨 + OR/IR 계산")
    parser.add_argument("--date", type=str, help="기준 날짜 (YYYY-MM-DD)")
    parser.add_argument("--backfill", type=int, help="과거 N일 OR/IR 백필")
    parser.add_argument("--accuracy", action="store_true", help="daily_bias 정확도 측정")
    parser.add_argument("--symbols", type=str, help="특정 종목만 (쉼표 구분)")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    if args.backfill:
        logger.info("OR/IR %d일 백필 시작", args.backfill)
        result = backfill_or_ir(args.backfill)
        print(f"백필 완료: {result['dates_processed']}일, {result['total_records']}건")
        return

    if args.accuracy:
        acc = measure_bias_accuracy(days=20)
        print(f"\n=== Daily Bias 정확도 ({acc['total']}건) ===")
        print(f"전체: {acc['correct']}/{acc['total']} = {acc['accuracy_pct']}%")
        for bias, stats in acc["by_bias"].items():
            if stats["total"] > 0:
                pct = round(stats["correct"] / stats["total"] * 100, 1)
                print(f"  {bias}: {stats['correct']}/{stats['total']} = {pct}%")
        return

    # 1. 프리미엄 레벨 계산
    logger.info("프리미엄 레벨 계산: %s", date_str)
    levels = compute_premium_levels(date_str, symbols)
    logger.info("프리미엄 레벨: %d종목", len(levels))

    # 2. OR/IR 계산
    logger.info("OR/IR 계산: %s", date_str)
    or_ir = compute_or_ir(date_str)
    logger.info("OR/IR: %d종목", len(or_ir))

    # 3. bias 분포
    if or_ir:
        bias_dist = {}
        for r in or_ir:
            b = r["daily_bias"]
            bias_dist[b] = bias_dist.get(b, 0) + 1
        print(f"\nDaily Bias 분포: {bias_dist}")

    # 3.5. Equal Level 탐지
    logger.info("Equal Level 탐지: %s", date_str)
    eq_levels = compute_equal_levels(date_str, symbols)
    eq_count = sum(
        1 for lv in eq_levels
        if lv.get("equal_lows") or lv.get("equal_highs")
    )
    logger.info("Equal Levels: %d종목 (유효 %d)", len(eq_levels), eq_count)

    # 4. 요약 브리핑 (보유 종목)
    positions_path = Path("data/positions.json")
    held_symbols = []
    if positions_path.exists():
        with open(positions_path, encoding="utf-8") as f:
            positions = json.load(f)
        if isinstance(positions, list):
            pos_list = positions
        elif isinstance(positions, dict):
            pos_list = positions.get("positions", [])
        else:
            pos_list = []
        held_symbols = [
            p.get("ticker", p.get("symbol", ""))
            for p in pos_list
            if p.get("shares", p.get("quantity", 0)) > 0
        ]

    if held_symbols and levels:
        print("\n" + format_premium_briefing(levels, held_symbols))
    if held_symbols and or_ir:
        print("\n" + format_or_ir_briefing(or_ir, held_symbols + ["069500"]))
    if held_symbols and eq_levels:
        eq_brief = format_equal_levels_briefing(eq_levels, held_symbols)
        if eq_brief:
            print("\n" + eq_brief)


if __name__ == "__main__":
    main()
