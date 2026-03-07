"""
장마감 후 유니버스 전체 분봉(5분/15분) 수집 → parquet 아카이브

사용법:
  python scripts/collect_intraday_candles.py              # 전체 유니버스 수집
  python scripts/collect_intraday_candles.py --dry-run    # 005930 1종목 테스트
  python scripts/collect_intraday_candles.py --tickers 005930,000660  # 특정 종목
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.adapters.kis_intraday_adapter import KisIntradayAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_candles")

DATA_ROOT = Path("data/intraday")
UNIVERSE_CSV = Path("data/universe.csv")


def load_universe() -> list[str]:
    """universe.csv에서 티커 목록 로드"""
    df = pd.read_csv(UNIVERSE_CSV)
    return df["ticker"].astype(str).str.zfill(6).tolist()


def save_candles_parquet(
    candles: list[dict], ticker: str, date_str: str, period: int,
) -> Path | None:
    """분봉 데이터를 parquet으로 저장"""
    if not candles:
        return None

    folder = DATA_ROOT / f"{period}min" / date_str
    folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ticker 컬럼 제거 (파일명에 이미 있음)
    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    path = folder / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    return path


def collect_one_ticker(
    adapter: KisIntradayAdapter, ticker: str, date_str: str,
) -> dict:
    """단일 종목 수집 → 5분봉 + 15분봉 parquet 저장"""
    # 전체 1분봉 수집
    one_min = adapter.fetch_full_day_1m_candles(ticker, date_str)
    if not one_min:
        return {"ticker": ticker, "status": "no_data", "1min": 0, "5min": 0, "15min": 0}

    # 5분봉 집계 + 저장
    candles_5 = adapter._aggregate_candles(ticker, one_min, 5)
    save_candles_parquet(candles_5, ticker, date_str, 5)

    # 15분봉 집계 + 저장
    candles_15 = adapter._aggregate_candles(ticker, one_min, 15)
    save_candles_parquet(candles_15, ticker, date_str, 15)

    return {
        "ticker": ticker,
        "status": "ok",
        "1min": len(one_min),
        "5min": len(candles_5),
        "15min": len(candles_15),
    }


def main():
    parser = argparse.ArgumentParser(description="유니버스 분봉 수집 → parquet 아카이브")
    parser.add_argument("--dry-run", action="store_true", help="005930 1종목만 테스트")
    parser.add_argument("--tickers", type=str, help="수집 종목 (쉼표 구분)")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 종목 결정
    if args.dry_run:
        tickers = ["005930"]
        logger.info("[dry-run] 삼성전자(005930) 1종목만 수집")
    elif args.tickers:
        tickers = [t.strip().zfill(6) for t in args.tickers.split(",")]
    else:
        tickers = load_universe()
    logger.info("수집 대상: %d종목, 날짜: %s", len(tickers), date_str)

    # KIS 어댑터 초기화
    adapter = KisIntradayAdapter()

    # 수집 루프
    stats = {"ok": 0, "no_data": 0, "error": 0}
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        try:
            result = collect_one_ticker(adapter, ticker, date_str)
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1

            if status == "ok":
                logger.info(
                    "[%d/%d] %s — 1분봉 %d건 → 5분봉 %d + 15분봉 %d",
                    i, len(tickers), ticker,
                    result["1min"], result["5min"], result["15min"],
                )
            else:
                logger.warning("[%d/%d] %s — 데이터 없음", i, len(tickers), ticker)

        except Exception as e:
            stats["error"] += 1
            logger.error("[%d/%d] %s — 오류: %s", i, len(tickers), ticker, e)

    elapsed = time.time() - t0

    # 결과 출력
    print(f"\n{'='*50}")
    print(f"분봉 수집 완료 ({elapsed:.1f}초)")
    print(f"  날짜: {date_str}")
    print(f"  종목: {len(tickers)}개")
    print(f"  성공: {stats['ok']}  |  데이터없음: {stats['no_data']}  |  오류: {stats['error']}")
    print(f"  저장 경로: {DATA_ROOT}/{{5min,15min}}/{date_str}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
