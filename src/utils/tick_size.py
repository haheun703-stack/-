"""KRX 호가 단위 보정 — 중복 제거 (5/27 슬러지 검수).

배경: _adjust_to_tick이 3곳 중복 (kis_order_adapter / ai_chain_queue / momentum_chase).
공통 util로 이동.

KRX 호가 단위:
  <1,000원       : 1원
  1,000~5,000   : 5원
  5,000~20,000  : 10원
  20,000~50,000 : 50원
  50,000~200,000: 100원
  200,000~500,000: 500원
  >=500,000     : 1,000원
"""


def adjust_to_tick(price: int) -> int:
    """가격을 KRX 호가 단위로 보정 (floor)."""
    if price <= 0:
        return price
    if price < 1_000:
        tick = 1
    elif price < 5_000:
        tick = 5
    elif price < 20_000:
        tick = 10
    elif price < 50_000:
        tick = 50
    elif price < 200_000:
        tick = 100
    elif price < 500_000:
        tick = 500
    else:
        tick = 1_000
    return (price // tick) * tick


def adjust_to_tick_ceil(price: int) -> int:
    """가격을 KRX 호가 단위로 올림 보정 (시장가/매도 시 유용)."""
    if price <= 0:
        return price
    floored = adjust_to_tick(price)
    if floored == price:
        return price
    # tick size를 다시 계산
    if price < 1_000:
        tick = 1
    elif price < 5_000:
        tick = 5
    elif price < 20_000:
        tick = 10
    elif price < 50_000:
        tick = 50
    elif price < 200_000:
        tick = 100
    elif price < 500_000:
        tick = 500
    else:
        tick = 1_000
    return floored + tick
