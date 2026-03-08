"""수동 주문 CLI — Bash에서 직접 매수/매도/현재가 조회.

사용법:
  # 금액 기준 매수 (자동 수량 계산)
  python tools/manual_order.py buy 삼성전자 --amount 500000

  # 수량 기준 매수
  python tools/manual_order.py buy 005930 --qty 10

  # 지정가 매수
  python tools/manual_order.py buy 삼성전자 --amount 500000 --limit 82000

  # 전량 매도
  python tools/manual_order.py sell 삼성전자 --all

  # 수량 매도
  python tools/manual_order.py sell 005930 --qty 5

  # 현재가 조회
  python tools/manual_order.py price 삼성전자

  # 잔고 조회
  python tools/manual_order.py balance

  # AI 분석 후 매수 (TradeAdvisor 연동)
  python tools/manual_order.py buy 삼성전자 --amount 500000 --ai

  # 드라이런 (주문 없이 시뮬레이션)
  python tools/manual_order.py buy 삼성전자 --amount 500000 --dry
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 호가 단위 맞춤 (KRX 규칙) ──

def _tick_round(price: int, reference: int) -> int:
    """KRX 호가 단위 반올림."""
    if reference < 2000:
        tick = 1
    elif reference < 5000:
        tick = 5
    elif reference < 20000:
        tick = 10
    elif reference < 50000:
        tick = 50
    elif reference < 200000:
        tick = 100
    elif reference < 500000:
        tick = 500
    else:
        tick = 1000
    return (price // tick) * tick


LIMIT_BUY_DISCOUNT = 0.005   # 현재가 -0.5%
LIMIT_SELL_PREMIUM = 0.005   # 현재가 +0.5%


# ── 종목 검색 ──

def resolve_stock(query: str) -> tuple[str, str]:
    """종목명 또는 코드 → (ticker, name). 실패 시 sys.exit."""
    from src.stock_name_resolver import resolve_name, ticker_to_name

    # 6자리 숫자 → 종목코드로 직접 사용
    if query.isdigit() and len(query) == 6:
        name = ticker_to_name(query)
        return query, name

    # 종목명 검색
    matches = resolve_name(query)
    if not matches:
        print(f"[오류] '{query}' 종목을 찾을 수 없습니다.")
        sys.exit(1)

    if len(matches) == 1:
        name, ticker = matches[0]
        return ticker, name

    # 여러 결과
    print(f"[검색결과] '{query}'에 대해 {len(matches)}건:")
    for i, (name, ticker) in enumerate(matches[:10], 1):
        print(f"  {i}. {name} ({ticker})")
    try:
        choice = int(input("번호 선택: ")) - 1
        if 0 <= choice < len(matches):
            name, ticker = matches[choice]
            return ticker, name
    except (ValueError, EOFError):
        pass
    print("[오류] 올바른 번호를 입력하세요.")
    sys.exit(1)


# ── 현재가 조회 ──

def cmd_price(args):
    """현재가 조회."""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    ticker, name = resolve_stock(args.stock)
    adapter = KisOrderAdapter()
    info = adapter.fetch_current_price(ticker)

    current = info.get("current_price", 0)
    change = info.get("change_pct", 0)
    volume = info.get("volume", 0)
    high = info.get("high", 0)
    low = info.get("low", 0)

    sign = "+" if change >= 0 else ""
    print()
    print(f"  {name} ({ticker})")
    print(f"  ━━━━━━━━━━━━━━━━━━")
    print(f"  현재가: {current:>10,}원 ({sign}{change:.2f}%)")
    print(f"  고가:   {high:>10,}원")
    print(f"  저가:   {low:>10,}원")
    print(f"  거래량: {volume:>10,}")
    print()


# ── 잔고 조회 ──

def cmd_balance(args):
    """전체 잔고 조회."""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    adapter = KisOrderAdapter()
    balance = adapter.fetch_balance()

    holdings = balance.get("holdings", [])
    cash = balance.get("available_cash", 0)
    total_eval = balance.get("total_eval", 0)
    total_pnl = balance.get("total_pnl", 0)

    print()
    print("  ━━━━ 보유 종목 ━━━━")
    if not holdings:
        print("  (보유 종목 없음)")
    else:
        for h in holdings:
            pnl = h.get("pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(f"  {h['name']:>8s} ({h['ticker']}) "
                  f"{h['quantity']:>5}주 "
                  f"평단 {h['avg_price']:>8,.0f}원 "
                  f"현재 {h['current_price']:>8,}원 "
                  f"({sign}{pnl:.1f}%)")

    print(f"\n  예수금:   {cash:>12,}원")
    print(f"  평가총액: {total_eval:>12,}원")
    pnl_sign = "+" if total_pnl >= 0 else ""
    print(f"  평가손익: {pnl_sign}{total_pnl:>11,}원")
    print()


# ── 매수 ──

def cmd_buy(args):
    """매수 주문."""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    ticker, name = resolve_stock(args.stock)
    adapter = KisOrderAdapter()

    # 현재가 조회
    info = adapter.fetch_current_price(ticker)
    current = info.get("current_price", 0)
    if current <= 0:
        print(f"[오류] 현재가 조회 실패: {name}({ticker})")
        sys.exit(1)

    # 수량 결정
    if args.amount:
        qty = args.amount // current
        if qty <= 0:
            print(f"[오류] 금액({args.amount:,}원)으로 {name} 1주도 못 삼 "
                  f"(현재가 {current:,}원)")
            sys.exit(1)
        actual_amount = qty * current
        print(f"  금액 {args.amount:,}원 → {qty}주 매수 "
              f"(실투자 약 {actual_amount:,}원)")
    elif args.qty:
        qty = args.qty
    else:
        print("[오류] --amount 또는 --qty를 지정하세요.")
        sys.exit(1)

    # 주문 가격 결정
    if args.limit:
        order_price = _tick_round(args.limit, current)
        order_type = "지정가"
    else:
        order_price = _tick_round(int(current * (1 - LIMIT_BUY_DISCOUNT)), current)
        order_type = f"지정가(현재가 -0.5%)"

    invest_total = order_price * qty

    # 예수금 확인
    cash = adapter.get_available_cash()
    if cash > 0 and invest_total > cash:
        print(f"\n  [예수금 부족]")
        print(f"  주문금액: {invest_total:,}원")
        print(f"  예수금:   {cash:,}원")
        print(f"  부족금:   {invest_total - cash:,}원")
        sys.exit(1)

    # AI 분석 (옵션)
    if args.ai:
        print(f"\n  AI 분석 중...")
        try:
            from src.agents.trade_advisor import TradeAdvisor
            advisor = TradeAdvisor()
            result = asyncio.run(advisor.analyze_buy(ticker, qty, current))
            verdict = result.get("verdict", "ERROR")
            confidence = result.get("confidence", 0)
            print(f"\n  AI 판단: {verdict} (신뢰도 {confidence}%)")
            print(f"  기술: {result.get('technical_summary', 'N/A')}")
            print(f"  촉매: {result.get('catalyst', 'N/A')}")
            if result.get("risk_warning"):
                print(f"  주의: {result['risk_warning']}")
            print(f"  제안: {result.get('suggestion', 'N/A')}")

            if verdict == "SKIP" and not args.force:
                print(f"\n  AI가 매수 비추합니다. 강제 실행하려면 --force 추가.")
                sys.exit(0)
        except Exception as e:
            print(f"  AI 분석 실패: {e} (주문은 계속)")

    # 주문 확인
    print(f"\n  ━━━━ 매수 주문 확인 ━━━━")
    print(f"  종목: {name} ({ticker})")
    print(f"  수량: {qty}주")
    print(f"  가격: {order_price:,}원 ({order_type})")
    print(f"  총액: {invest_total:,}원")
    if cash > 0:
        print(f"  예수금: {cash:,}원 (잔여 {cash - invest_total:,}원)")

    if args.dry:
        print(f"\n  [드라이런] 실제 주문 없이 종료.")
        return

    # 최종 확인
    try:
        confirm = input("\n  실행하시겠습니까? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "n"

    if confirm != "y":
        print("  주문 취소.")
        return

    # 주문 실행
    if args.limit:
        order = adapter.buy_limit(ticker, order_price, qty)
    else:
        order = adapter.buy_limit(ticker, order_price, qty)

    status = getattr(order, "status", "UNKNOWN")
    if str(status) in ("PENDING", "OrderStatus.PENDING"):
        print(f"\n  [매수 접수] {name} {qty}주 @ {order_price:,}원")
        print(f"  주문번호: {order.order_id}")

        # 텔레그램 알림
        try:
            from src.telegram_sender import send_message
            send_message(
                f"[수동 매수] {name}({ticker}) {qty}주\n"
                f"지정가: {order_price:,}원 (총 {invest_total:,}원)\n"
                f"주문번호: {order.order_id}"
            )
        except Exception:
            pass
    else:
        print(f"\n  [매수 실패] {getattr(order, 'message', status)}")


# ── 매도 ──

def cmd_sell(args):
    """매도 주문."""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    ticker, name = resolve_stock(args.stock)
    adapter = KisOrderAdapter()

    # 보유 수량 확인
    holdings = adapter.fetch_holdings()
    holding = None
    for h in holdings:
        if h["ticker"] == ticker:
            holding = h
            break

    if not holding:
        print(f"[오류] {name}({ticker}) 보유 중이지 않습니다.")
        sys.exit(1)

    held_qty = holding["quantity"]
    avg_price = holding.get("avg_price", 0)
    current_price = holding.get("current_price", 0)
    pnl_pct = holding.get("pnl_pct", 0)

    # 수량 결정
    if args.all:
        qty = held_qty
        print(f"  전량 매도: {qty}주")
    elif args.qty:
        qty = args.qty
        if qty > held_qty:
            print(f"[오류] 보유 {held_qty}주 < 요청 {qty}주")
            sys.exit(1)
    else:
        print(f"[오류] --qty 또는 --all을 지정하세요.")
        sys.exit(1)

    # 현재가 조회 (최신)
    info = adapter.fetch_current_price(ticker)
    current = info.get("current_price", 0) or current_price
    if current <= 0:
        print(f"[오류] 현재가 조회 실패: {name}({ticker})")
        sys.exit(1)

    # 주문 가격 결정
    if args.limit:
        order_price = _tick_round(args.limit, current)
        order_type = "지정가"
    else:
        order_price = _tick_round(int(current * (1 + LIMIT_SELL_PREMIUM)), current)
        order_type = f"지정가(현재가 +0.5%)"

    sell_total = order_price * qty
    pnl_sign = "+" if pnl_pct >= 0 else ""

    # AI 분석 (옵션)
    if args.ai:
        print(f"\n  AI 분석 중...")
        try:
            from src.agents.trade_advisor import TradeAdvisor
            advisor = TradeAdvisor()
            holding_info = {
                "entry_price": avg_price,
                "current_price": current,
                "pnl_pct": pnl_pct,
            }
            result = asyncio.run(advisor.analyze_sell(ticker, qty, holding_info))
            verdict = result.get("verdict", "ERROR")
            confidence = result.get("confidence", 0)
            print(f"\n  AI 판단: {verdict} (신뢰도 {confidence}%)")
            print(f"  촉매: {result.get('catalyst_status', 'N/A')}")
            print(f"  기술: {result.get('technical_signal', 'N/A')}")
            print(f"  제안: {result.get('suggestion', 'N/A')}")

            if verdict == "HOLD" and not args.force:
                print(f"\n  AI가 홀딩 추천합니다. 강제 매도하려면 --force 추가.")
                sys.exit(0)
        except Exception as e:
            print(f"  AI 분석 실패: {e} (주문은 계속)")

    # 주문 확인
    print(f"\n  ━━━━ 매도 주문 확인 ━━━━")
    print(f"  종목: {name} ({ticker})")
    print(f"  보유: {held_qty}주 (평단 {avg_price:,.0f}원, {pnl_sign}{pnl_pct:.1f}%)")
    print(f"  매도: {qty}주")
    print(f"  가격: {order_price:,}원 ({order_type})")
    print(f"  예상: {sell_total:,}원")

    if args.dry:
        print(f"\n  [드라이런] 실제 주문 없이 종료.")
        return

    # 최종 확인
    try:
        confirm = input("\n  실행하시겠습니까? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "n"

    if confirm != "y":
        print("  주문 취소.")
        return

    # 주문 실행
    order = adapter.sell_limit(ticker, order_price, qty)

    status = getattr(order, "status", "UNKNOWN")
    if str(status) in ("PENDING", "OrderStatus.PENDING"):
        print(f"\n  [매도 접수] {name} {qty}주 @ {order_price:,}원")
        print(f"  주문번호: {order.order_id}")

        # 텔레그램 알림
        try:
            from src.telegram_sender import send_message
            send_message(
                f"[수동 매도] {name}({ticker}) {qty}주\n"
                f"지정가: {order_price:,}원 (총 {sell_total:,}원)\n"
                f"수익률: {pnl_sign}{pnl_pct:.1f}%\n"
                f"주문번호: {order.order_id}"
            )
        except Exception:
            pass
    else:
        print(f"\n  [매도 실패] {getattr(order, 'message', status)}")


# ── CLI 파서 ──

def main():
    parser = argparse.ArgumentParser(
        description="수동 주문 CLI — 매수/매도/현재가/잔고",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python tools/manual_order.py buy 삼성전자 --amount 500000     # 50만원어치 매수
  python tools/manual_order.py buy 005930 --qty 10              # 10주 매수
  python tools/manual_order.py buy 풍산 --amount 300000 --ai    # AI 분석 후 매수
  python tools/manual_order.py sell 현대차 --all                 # 전량 매도
  python tools/manual_order.py sell 010140 --qty 20 --ai        # AI 분석 후 20주 매도
  python tools/manual_order.py price 삼성전자                    # 현재가 조회
  python tools/manual_order.py balance                           # 잔고 조회
  python tools/manual_order.py buy 삼성전자 --amount 1000000 --dry  # 드라이런
""",
    )

    sub = parser.add_subparsers(dest="command", help="명령")

    # buy
    p_buy = sub.add_parser("buy", help="매수")
    p_buy.add_argument("stock", help="종목명 또는 종목코드")
    buy_group = p_buy.add_mutually_exclusive_group(required=True)
    buy_group.add_argument("--amount", type=int, help="투자 금액 (원)")
    buy_group.add_argument("--qty", type=int, help="매수 수량 (주)")
    p_buy.add_argument("--limit", type=int, help="지정가 (원). 미지정 시 현재가 -0.5%%")
    p_buy.add_argument("--ai", action="store_true", help="AI 분석 후 매수")
    p_buy.add_argument("--force", action="store_true", help="AI SKIP 판정 무시 강제 매수")
    p_buy.add_argument("--dry", action="store_true", help="드라이런 (주문 없이 시뮬레이션)")

    # sell
    p_sell = sub.add_parser("sell", help="매도")
    p_sell.add_argument("stock", help="종목명 또는 종목코드")
    sell_group = p_sell.add_mutually_exclusive_group(required=True)
    sell_group.add_argument("--qty", type=int, help="매도 수량 (주)")
    sell_group.add_argument("--all", action="store_true", help="전량 매도")
    p_sell.add_argument("--limit", type=int, help="지정가 (원). 미지정 시 현재가 +0.5%%")
    p_sell.add_argument("--ai", action="store_true", help="AI 분석 후 매도")
    p_sell.add_argument("--force", action="store_true", help="AI HOLD 판정 무시 강제 매도")
    p_sell.add_argument("--dry", action="store_true", help="드라이런 (주문 없이 시뮬레이션)")

    # price
    p_price = sub.add_parser("price", help="현재가 조회")
    p_price.add_argument("stock", help="종목명 또는 종목코드")

    # balance
    sub.add_parser("balance", help="잔고 조회")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "buy":
        cmd_buy(args)
    elif args.command == "sell":
        cmd_sell(args)
    elif args.command == "price":
        cmd_price(args)
    elif args.command == "balance":
        cmd_balance(args)


if __name__ == "__main__":
    main()
