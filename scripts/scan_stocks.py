"""다중 종목 수급 스캐너 - 여러 종목을 한눈에 스캔"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
from src.adapters.naver_finance import NaverFinanceAdapter

# 기본 관심 종목
DEFAULT_STOCKS = "005930,000660,035420,035720,006400"
STOCK_NAMES = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "006400": "삼성SDI",
    "005380": "현대차",
    "051910": "LG화학",
    "068270": "셀트리온",
    "105560": "KB금융",
    "055550": "신한지주",
}


def to_eok(shares: int, price: float) -> str:
    """주수 × 가격 → 억원 문자열"""
    amount = shares * price / 1_0000_0000
    return f"{amount:+,.0f}억"


async def scan(codes: list[str]) -> None:
    kis = KisStockDataAdapter()
    naver = NaverFinanceAdapter()

    now = datetime.now().strftime("%Y.%m.%d %H:%M")
    print(f"\n{'━'*60}")
    print(f"  수급 스캔 ({now})")
    print(f"{'━'*60}")
    print(f"  {'종목':<10} {'현재가':>10} {'등락률':>7}  {'외국인':>8}  {'기관':>8}  {'거래량':>7}")
    print(f"  {'─'*10} {'─'*10} {'─'*7}  {'─'*8}  {'─'*8}  {'─'*7}")

    for code in codes:
        try:
            # KIS API: 현재가 + 일봉
            stock, chart = await kis.fetch(code, period_days=5)
            latest = chart.latest
            if not latest:
                print(f"  {code:<10} 데이터 없음")
                continue

            # 등락률 계산
            prev = chart.candles[-2] if len(chart.candles) >= 2 else None
            change_pct = ((latest.close - prev.close) / prev.close * 100) if prev else 0

            # 거래량 비율 (전일 대비)
            vol_ratio = 0
            if prev and prev.volume > 0:
                vol_ratio = latest.volume / prev.volume * 100

            # 네이버: 투자자별 수급
            flow = await naver.fetch(code)

            name = stock.name or STOCK_NAMES.get(code, code)
            foreign_str = to_eok(flow.foreign_net, latest.close)
            inst_str = to_eok(flow.inst_net, latest.close)

            # 특수 표시
            flags = ""
            if flow.foreign_net > 0 and flow.inst_net > 0:
                flags = " [동시매수]"
            if vol_ratio >= 150:
                flags += " [거래량!]"

            print(
                f"  {name:<10} {latest.close:>10,.0f} {change_pct:>+6.1f}%  "
                f"{foreign_str:>8}  {inst_str:>8}  {vol_ratio:>5.0f}%{flags}"
            )

        except Exception as e:
            print(f"  {code:<10} [오류] {e}")

        # API rate limit: 종목 간 0.1초 대기
        time.sleep(0.1)

    print(f"{'━'*60}")
    print(f"  [동시매수] = 외국인+기관 동시 순매수  [거래량!] = 전일대비 150%+")
    print()


def main():
    parser = argparse.ArgumentParser(description="다중 종목 수급 스캐너")
    parser.add_argument(
        "--stocks", type=str, default=DEFAULT_STOCKS,
        help=f"종목코드 (쉼표 구분, 기본: {DEFAULT_STOCKS})",
    )
    args = parser.parse_args()

    codes = [c.strip() for c in args.stocks.split(",") if c.strip()]
    asyncio.run(scan(codes))


if __name__ == "__main__":
    main()
