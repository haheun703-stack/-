"""실시간 수급 모니터링 실행 스크립트"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.adapters.console_output import ConsoleOutputAdapter
from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
from src.adapters.naver_finance import NaverFinanceAdapter
from src.use_cases.realtime_monitor import RealtimeMonitorInteractor


async def run(tickers: list[str], interval: int, threshold_eok: int) -> None:
    """실시간 수급 모니터링 실행"""
    monitor = RealtimeMonitorInteractor(
        flow_port=NaverFinanceAdapter(),
        stock_port=KisStockDataAdapter(),
        output=ConsoleOutputAdapter(),
    )
    await monitor.monitor(
        codes=tickers,
        interval=interval,
        threshold_eok=threshold_eok,
    )


def main():
    parser = argparse.ArgumentParser(description="실시간 수급 모니터링")
    parser.add_argument(
        "--stocks", type=str, default="005930,000660,035420",
        help="종목코드 (쉼표 구분, 기본: 삼성전자,SK하이닉스,NAVER)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="조회 간격 (초, 기본: 300 = 5분)",
    )
    parser.add_argument(
        "--threshold", type=int, default=50,
        help="알림 기준 (억원, 기본: 50)",
    )
    args = parser.parse_args()

    tickers = [c.strip() for c in args.stocks.split(",") if c.strip()]

    print(f"\n  [모니터링] {len(tickers)}종목, {args.interval}초 간격, 알림기준: {args.threshold}억")
    print(f"  종목: {', '.join(tickers)}")

    try:
        asyncio.run(run(tickers, args.interval, args.threshold))
    except KeyboardInterrupt:
        print("\n  [종료] Ctrl+C")


if __name__ == "__main__":
    main()
