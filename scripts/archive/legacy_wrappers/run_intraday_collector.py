"""
Phase 1: 장중 데이터 수집기 실행 스크립트

사용법:
  # 스케줄러 모드 (장 운영시간 동안 자동 수집)
  python scripts/run_intraday_collector.py

  # 1회성 수집 (테스트용)
  python scripts/run_intraday_collector.py --once

  # 특정 종목 지정
  python scripts/run_intraday_collector.py --tickers 005930,000660

  # DB 상태 확인
  python scripts/run_intraday_collector.py --status
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.adapters.kis_intraday_adapter import KisIntradayAdapter
from src.adapters.sqlite_intraday_store import SqliteIntradayStore
from src.use_cases.intraday_collector import IntradayCollector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("intraday")


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_holdings_from_positions() -> list[str]:
    """data/positions.json에서 현재 보유종목 목록 로드"""
    import json
    pos_file = Path("data/positions.json")
    if pos_file.exists():
        try:
            positions = json.loads(pos_file.read_text(encoding="utf-8"))
            tickers = [p["ticker"] for p in positions if p.get("shares", 0) > 0]
            if tickers:
                logger.info("보유종목 %d개 로드: %s", len(tickers), tickers)
                return tickers
        except Exception as e:
            logger.warning("positions.json 로드 실패: %s", e)
    return []


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 장중 데이터 수집기")
    parser.add_argument("--once", action="store_true", help="1회성 수집 후 종료")
    parser.add_argument("--status", action="store_true", help="DB 상태만 확인")
    parser.add_argument("--tickers", type=str, help="수집 종목 (쉼표 구분)")
    parser.add_argument("--db-path", type=str, help="SQLite DB 경로")
    args = parser.parse_args()

    config = load_config()
    intraday_cfg = config.get("intraday_monitor", {})

    # DB 초기화
    db_path = args.db_path or intraday_cfg.get("db_path", "data/intraday.db")
    store = SqliteIntradayStore(db_path)

    # --status: DB 상태만 보여주고 종료
    if args.status:
        stats = store.get_db_stats()
        print("\n=== 장중 DB 상태 ===")
        for table, count in stats.items():
            print(f"  {table}: {count:,}건")
        print()
        return

    # KIS 어댑터 초기화
    kis = KisIntradayAdapter()

    # 보유종목 결정
    if args.tickers:
        holdings = [t.strip() for t in args.tickers.split(",")]
    else:
        holdings = get_holdings_from_positions()

    if not holdings:
        # 기본 테스트 종목
        holdings = ["005930", "000660"]
        logger.info("보유종목 미설정 — 테스트 종목 사용: %s", holdings)

    # 수집기 생성
    collector = IntradayCollector(
        config=config,
        data_port=kis,
        store_port=store,
        holdings=holdings,
    )

    # --once: 1회 수집
    if args.once:
        print(f"\n1회성 수집 시작 (종목: {holdings})")
        result = collector.collect_once()
        print(f"\n=== 수집 결과 ===")
        print(f"  시각: {result['timestamp']}")
        print(f"  틱: {result['ticks']}건")
        print(f"  5분봉: {result['candles']}건")
        print(f"  투자자수급: {result['flows']}건")
        print(f"  시장지수: {'수집' if result['market'] else '실패'}")
        print(f"  업종시세: {result['sectors']}건")
        print()

        # DB 상태
        stats = store.get_db_stats()
        print("=== DB 현황 ===")
        for table, count in stats.items():
            print(f"  {table}: {count:,}건")
        return

    # 스케줄러 모드
    print(f"\n장중 데이터 수집기 시작")
    print(f"  보유종목: {holdings}")
    print(f"  수집 주기: 틱={intraday_cfg.get('tick_interval_sec', 60)}초, "
          f"캔들={intraday_cfg.get('candle_interval_min', 5)}분, "
          f"수급={intraday_cfg.get('flow_interval_min', 10)}분")
    print(f"  DB: {db_path}")
    print(f"\nCtrl+C로 종료\n")

    # Graceful shutdown
    def shutdown(signum, frame):
        print("\n수집기 종료 중...")
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    collector.start()

    # 메인 스레드 유지
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        collector.stop()


if __name__ == "__main__":
    main()
