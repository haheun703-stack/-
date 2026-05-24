"""register_adaptive_queue.py — 적응형 매매법 큐 수동 등록 스크립트.

배경 (퐝가님 5/24 지시):
  "익절 후 다음 종목 들어갈 게 있어야 되지 않나?"

  자동 후보 풀(soubujang_pool)은 18:00 갱신이라 즉시 다음 종목 진입 불가.
  → 정보봇/단타봇 신호 보고 퐝가님이 즉시 큐 등록 가능하게.

사용 (단일 등록):
  python scripts/register_adaptive_queue.py \\
      --ticker 110990 --peak 23400 --cash 3000000 --name 디아이티

사용 (5/26 초기 5종목 일괄 등록 — 5/23 실측 기준):
  python scripts/register_adaptive_queue.py --batch initial_5_26

사용 (현재 큐 조회):
  python scripts/register_adaptive_queue.py --list

사용 (특정 종목 청산):
  python scripts/register_adaptive_queue.py --clear 110990

흐름:
  등록 직후 → 다음 cron(/30분)에서 MVP-2가 가격 도달 단계 즉시 매수
  매수 체결 → MVP-2.5 trailing 자동 시작
  꺾임 → 매도 → QUICK_SOLD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.use_cases.adaptive_buy_queue import (
    register_buy_queue,
    load_queues,
    clear_queue,
    get_queue_status,
)


# 5/26 초기 일괄 등록 (5/23 실측 step5_soubujang_pool 통과 종목)
BATCH_INITIAL_5_26 = [
    # (ticker, peak_price, cash, name)
    # 디아이티: 천장 -1.8% 영역 (가장 진입 임박)
    ("110990", 23_400, 1_500_000, "디아이티"),
    # 원익IPS: -19.7% 영역 (L2 -20% 가까움 → 즉시 매수 트리거 예상)
    ("240810", 25_000, 1_500_000, "원익IPS"),
    # ISC: -28.2% 영역 (L3 -30% 가까움)
    ("095340", 62_000, 1_500_000, "ISC"),
    # 일진전기: -26.8% 영역 (L3 -30% 가까움)
    ("103590", 25_000, 1_500_000, "일진전기"),
    # 산일전기: -17.2% 영역 (L2 -20% 가까움)
    ("062040", 105_000, 1_500_000, "산일전기"),
    # 현대무벡스: -22.6% 영역
    ("319400", 50_000, 1_500_000, "현대무벡스"),
]


def cmd_register_single(ticker: str, peak: int, cash: int, name: str = "") -> int:
    """단일 종목 큐 등록."""
    print(f"▶ 큐 등록 시도: {name or ticker} (천장 {peak:,}, 가용 {cash:,})")
    result = register_buy_queue(ticker, peak, cash, name)

    if result["success"]:
        print(f"✅ 등록 완료 ({'천장 갱신' if result.get('is_update') else '신규'})")
        for stage in result["stages"]:
            status = stage["status"]
            print(
                f"   L{stage['level']}: 지정가 {stage['target_price']:,}원 × {stage['qty']}주 "
                f"= {stage['alloc_amount']:,}원 [{status}]"
            )
        return 0
    else:
        print(f"❌ 등록 실패: {result.get('error', '')}")
        return 1


def cmd_batch(name: str) -> int:
    """일괄 등록."""
    batches = {"initial_5_26": BATCH_INITIAL_5_26}
    if name not in batches:
        print(f"❌ 알 수 없는 배치: {name} (가능: {list(batches.keys())})")
        return 1

    items = batches[name]
    print(f"▶ 일괄 등록 [{name}]: {len(items)} 종목")
    print("=" * 60)

    success = 0
    for ticker, peak, cash, n in items:
        ret = cmd_register_single(ticker, peak, cash, n)
        if ret == 0:
            success += 1
        print()

    print("=" * 60)
    print(f"✅ {success}/{len(items)} 등록 완료")
    return 0 if success == len(items) else 1


def cmd_list() -> int:
    """현재 활성 큐 조회."""
    queues = load_queues()
    if not queues:
        print("📋 활성 큐 없음")
        return 0

    print(f"📋 활성 큐 {len(queues)} 종목")
    print("=" * 80)
    for ticker, entry in queues.items():
        print(f"\n[{entry.get('name', ticker)} ({ticker})]")
        print(f"  천장: {entry.get('peak_price', 0):,}원 / 등록: {entry.get('registered_at', '')[:16]}")
        for stage in entry.get("stages", []):
            level = stage.get("level")
            status = stage.get("status")
            target = stage.get("target_price", 0)
            qty = stage.get("qty", 0)

            extra = ""
            if status == "FILLED":
                qp = stage.get("quick_profit_target", 0)
                extra = f"  매수 {stage.get('actual_price', 0):,} → 익절 대기 {qp:,}"
            elif status == "QUICK_ARMED":
                tp = stage.get("trailing_peak", 0)
                extra = f"  Trailing 추적 중 (고점 {tp:,})"
            elif status == "QUICK_SOLD":
                sold = stage.get("quick_profit_sold_price", 0)
                extra = f"  매도 완료 @ {sold:,}"

            print(f"  L{level}: 지정가 {target:,} × {qty}주 [{status}]{extra}")
    print("=" * 80)
    return 0


def cmd_clear(ticker: str) -> int:
    """특정 종목 큐 청산."""
    entry = get_queue_status(ticker)
    if not entry:
        print(f"❌ 종목 미등록: {ticker}")
        return 1
    name = entry.get("name", ticker)
    if clear_queue(ticker):
        print(f"✅ 청산 완료: {name} ({ticker})")
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # 단일 등록 옵션 (기본 모드)
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--peak", type=int, help="천장 가격 (원)")
    parser.add_argument("--cash", type=int, default=1_500_000, help="가용 현금 (원)")
    parser.add_argument("--name", type=str, default="")

    # 일괄/조회/청산 모드
    parser.add_argument("--batch", type=str, help="배치 이름 (예: initial_5_26)")
    parser.add_argument("--list", action="store_true", help="현재 큐 조회")
    parser.add_argument("--clear", type=str, help="특정 종목 청산")

    args = parser.parse_args()

    if args.list:
        return cmd_list()
    if args.clear:
        return cmd_clear(args.clear)
    if args.batch:
        return cmd_batch(args.batch)
    if args.ticker and args.peak:
        return cmd_register_single(args.ticker, args.peak, args.cash, args.name)

    parser.print_help()
    print("\n예시:")
    print("  python scripts/register_adaptive_queue.py --batch initial_5_26")
    print("  python scripts/register_adaptive_queue.py --ticker 110990 --peak 23400 --cash 1500000 --name 디아이티")
    print("  python scripts/register_adaptive_queue.py --list")
    print("  python scripts/register_adaptive_queue.py --clear 110990")
    return 1


if __name__ == "__main__":
    sys.exit(main())
