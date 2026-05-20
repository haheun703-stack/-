"""차트영웅 진입 필터 — 5/21 1주차 워밍업 (Phase 1).

퐝가님 5/20 지시:
  "무조건 시장에가서 사는게 아니라 VWAP, 체결강도, 호가, 수급 등등을 보면서 들어가야지"

Phase 1 (오늘 5/20 안에): 2개 시그널 즉시 구현
  ✅ VWAP 근사 (현재가 < typical price = (H+L+C)/3)
  ✅ 외+기 5일 누적 수급 (foreign + institution >= 0)

Phase 2 (5/22+ 점진): 2개 시그널 추가
  ⏳ 체결강도 (KIS inquire-ccnl)
  ⏳ 호가 (매수벽 vs 매도벽, KIS inquire-orderbook)

5/21 1주차 진입 조건:
  - D+1 양봉 확인 (필수, close_cycle 기존 로직)
  - + 본 필터 4개 중 2개 이상 통과 (Phase 1은 2/2 통과 = should_enter=True)

진입 거부 시 SKIP_FILTER로 close_cycle에서 분기.
"""

import logging

from src.adapters.kis_nxt_kit import get_nx_price, get_supply_5day

logger = logging.getLogger(__name__)


# 통과 임계값 (Phase 1 = 2/2, Phase 2 진입 시 4/4 중 2 이상)
PASS_THRESHOLD = 2


def evaluate_entry_signals(ticker: str) -> dict:
    """진입 시그널 평가.

    Args:
        ticker: 종목 코드 (6자리)

    Returns:
        {
          "ticker": "005930",
          "vwap_below": True,        # 현재가 < VWAP 근사
          "supply_positive": True,    # 외+기 5일 합 ≥ 0
          "strength_high": None,      # Phase 2 (체결강도)
          "orderbook_buy_wall": None, # Phase 2 (호가)
          "pass_count": 2,
          "pass_threshold": 2,
          "should_enter": True,
          "details": {
            "current_price": 50400,
            "vwap_proxy": 49983,
            "foreign_5d_백만원": 12345,
            "institution_5d_백만원": 6789,
          },
        }
    """
    result = {
        "ticker": ticker,
        "vwap_below": False,
        "supply_positive": False,
        "strength_high": None,
        "orderbook_buy_wall": None,
        "pass_count": 0,
        "pass_threshold": PASS_THRESHOLD,
        "should_enter": False,
        "api_failures": [],  # W1: silent SKIP vs 실제 거부 구분 (5/20 보강)
        "details": {},
    }

    # ─────────────────────────────────────────────
    # Signal 1: VWAP 근사 (typical price)
    # ─────────────────────────────────────────────
    price_info = get_nx_price(ticker)
    if not price_info:
        result["api_failures"].append("get_nx_price")
    if price_info:
        current = price_info["price"]
        high = price_info["high"]
        low = price_info["low"]
        if current > 0 and high > 0 and low > 0:
            vwap_proxy = (high + low + current) / 3
            result["vwap_below"] = current < vwap_proxy
            result["details"]["current_price"] = current
            result["details"]["vwap_proxy"] = round(vwap_proxy, 0)
            logger.info(
                "[Entry Filter] %s VWAP: current=%d < proxy=%d? %s",
                ticker, current, vwap_proxy, result["vwap_below"]
            )

    # ─────────────────────────────────────────────
    # Signal 2: 외+기 5일 누적 수급 (>= 0)
    # ─────────────────────────────────────────────
    supply = get_supply_5day(ticker)
    if not supply:
        result["api_failures"].append("get_supply_5day")
    if supply:
        f = supply.get("foreign_5d_백만원", 0)
        i = supply.get("institution_5d_백만원", 0)
        total = f + i
        result["supply_positive"] = total >= 0
        result["details"]["foreign_5d_백만원"] = f
        result["details"]["institution_5d_백만원"] = i
        result["details"]["total_5d_백만원"] = total
        logger.info(
            "[Entry Filter] %s Supply: 외인=%d + 기관=%d = %d >= 0? %s",
            ticker, f, i, total, result["supply_positive"]
        )

    # ─────────────────────────────────────────────
    # Phase 2 placeholders (5/22+ 구현 예정)
    # ─────────────────────────────────────────────
    # result["strength_high"] = check_chegyeol_strength(ticker)
    # result["orderbook_buy_wall"] = check_orderbook(ticker)

    # ─────────────────────────────────────────────
    # 통과 카운트 + 진입 판정
    # ─────────────────────────────────────────────
    pass_count = 0
    if result["vwap_below"]:
        pass_count += 1
    if result["supply_positive"]:
        pass_count += 1
    # Phase 2 추가 시 여기에 추가
    result["pass_count"] = pass_count
    result["should_enter"] = pass_count >= PASS_THRESHOLD

    return result


def filter_picks_for_entry(picks_with_confirm: list[dict]) -> list[dict]:
    """D+1 양봉 확인된 picks에 진입 필터 추가 평가.

    Args:
        picks_with_confirm: d1_confirm.monitor_picks() 출력
                            (will_enter=True 종목들)

    Returns:
        진입 필터 결과를 각 pick에 병합한 리스트.
        - filter_result: dict (evaluate_entry_signals 출력)
        - will_enter_final: bool (양봉 + 필터 모두 통과)
    """
    out = []
    for p in picks_with_confirm:
        ticker = p.get("ticker", "")
        if not ticker:
            continue
        filter_result = evaluate_entry_signals(ticker)
        will_enter_d1 = p.get("will_enter", False)
        p["filter_result"] = filter_result
        p["will_enter_final"] = will_enter_d1 and filter_result["should_enter"]
        out.append(p)
    return out


if __name__ == "__main__":
    # 단독 실행 시 샘플 검증 (실제 호출은 KIS API 필요)
    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "005930"  # 삼성전자 기본
    print(f"=== Entry Filter Test: {ticker} ===")
    r = evaluate_entry_signals(ticker)
    print(f"  VWAP below:    {r['vwap_below']}")
    print(f"  Supply +:      {r['supply_positive']}")
    print(f"  Pass count:    {r['pass_count']}/{r['pass_threshold']}")
    print(f"  Should enter:  {r['should_enter']}")
    print(f"  Details:       {r['details']}")
