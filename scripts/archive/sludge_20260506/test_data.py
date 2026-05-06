"""1단계 데이터 확인 스크립트 - 삼성전자 데이터가 정상적으로 나오는지 검증"""

import asyncio
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
from src.adapters.naver_finance import NaverFinanceAdapter


def to_eok(value: int | float, price: float = 1.0) -> str:
    """주수 × 가격 → 억원 문자열 변환 (가격 없으면 주수 그대로)"""
    if price > 1:
        amount = value * price / 1_0000_0000
        return f"{amount:+,.0f}억"
    return f"{value:+,}"


async def main(ticker: str = "005930") -> None:
    print(f"\n{'='*50}")
    print(f"  1단계 데이터 확인 - {ticker}")
    print(f"  {datetime.now().strftime('%Y.%m.%d %H:%M')}")
    print(f"{'='*50}")

    # ── 한투 API: 현재가 + 일봉 ──
    print("\n[한투 API] 데이터 조회 중...")
    kis = KisStockDataAdapter()
    stock, chart_data = await kis.fetch(ticker, period_days=20)

    latest = chart_data.latest
    if latest:
        prev = chart_data.candles[-2] if len(chart_data.candles) >= 2 else None
        change_pct = ((latest.close - prev.close) / prev.close * 100) if prev else 0

        print(f"\n{'━'*40}")
        print(f"  {stock.name} ({stock.ticker})")
        print(f"{'-'*40}")
        print(f"  현재가: {latest.close:,.0f}원 ({change_pct:+.1f}%)")
        print(f"  시장: {stock.market.value} | 업종: {stock.sector}")
        print(f"  거래량: {latest.volume:,}")

        # 최근 5일 일봉
        print(f"\n{'━'*40}")
        print(f"  최근 5일 일봉")
        print(f"{'-'*40}")
        for c in chart_data.candles[-5:]:
            print(f"  {c.date} | 시 {c.open:>8,.0f} 고 {c.high:>8,.0f} "
                  f"저 {c.low:>8,.0f} 종 {c.close:>8,.0f} | 거래량 {c.volume:>12,}")
    else:
        print("  [경고] 일봉 데이터가 비어있습니다.")

    # ── 기술 지표 ──
    ind = chart_data.indicators
    print(f"\n{'━'*40}")
    print(f"  기술 지표")
    print(f"{'-'*40}")
    print(f"  RSI(14): {f'{ind.rsi:.1f}' if ind.rsi else 'N/A'}")
    print(f"  MACD: {f'{ind.macd:.2f}' if ind.macd else 'N/A'}")
    print(f"  MA5: {f'{ind.ma5:,.0f}' if ind.ma5 else 'N/A'} | "
          f"MA20: {f'{ind.ma20:,.0f}' if ind.ma20 else 'N/A'} | "
          f"MA60: {f'{ind.ma60:,.0f}' if ind.ma60 else 'N/A'}")

    # ── 네이버 금융: 투자자별 매매 ──
    print(f"\n{'━'*40}")
    print(f"  투자자별 매매 (네이버 금융)")
    print(f"{'-'*40}")
    naver = NaverFinanceAdapter()
    flow = await naver.fetch(ticker)
    current_price = latest.close if latest else 0
    print(f"  외국인: {to_eok(flow.foreign_net, current_price)} ({flow.foreign_net:+,}주)")
    print(f"  기  관: {to_eok(flow.inst_net, current_price)} ({flow.inst_net:+,}주)")
    print(f"  개  인: {to_eok(flow.individual_net, current_price)} ({flow.individual_net:+,}주)")

    print(f"\n{'='*50}")
    print(f"  1단계 데이터 확인 완료!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "005930"
    asyncio.run(main(ticker))
