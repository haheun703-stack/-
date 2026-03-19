"""Alpha Engine V2 — STEP 3-1 재무 데이터 수집 러너

유니버스 종목의 8분기 재무 데이터(BS+CF)를 수집하고
퀄리티 메트릭(ROE, 부채비율, Accruals, 배당성향, FCF)을 계산하여 저장.

사용법:
  # 테스트 (5종목만)
  python -u -X utf8 scripts/v2_collect_financial_data.py --test

  # BS만 수집 (multi-company API, 빠름)
  python -u -X utf8 scripts/v2_collect_financial_data.py --bs-only

  # 전체 수집 (BS + CF)
  python -u -X utf8 scripts/v2_collect_financial_data.py

출력:
  data/v2_migration/financial_quarterly.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = "data/v2_migration/financial_quarterly.json"


def get_universe_tickers() -> list[str]:
    """data/processed/*.parquet에서 유니버스 종목코드 추출."""
    parquets = glob.glob(str(PROJECT_ROOT / "data/processed/*.parquet"))
    tickers = sorted(
        os.path.basename(f).replace(".parquet", "") for f in parquets
    )
    # 우선주(끝자리 5,K 등) 제외 — DART 연결재무제표가 없는 경우가 많음
    tickers = [t for t in tickers if t[-1].isdigit() and t[-1] != "5"]
    return tickers


def main():
    parser = argparse.ArgumentParser(description="V2 재무 데이터 수집")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (5종목)")
    parser.add_argument("--bs-only", action="store_true", help="BS만 수집 (CF 스킵)")
    parser.add_argument("--tickers", type=str, help="콤마 구분 종목코드 (수동 지정)")
    args = parser.parse_args()

    from src.adapters.dart_financial_adapter import DartFinancialAdapter

    adapter = DartFinancialAdapter()

    if not adapter.is_available:
        logger.error("DART_API_KEY 미설정 — 수집 불가")
        sys.exit(1)

    # 종목 결정
    if args.tickers:
        tickers = [t.strip().zfill(6) for t in args.tickers.split(",")]
    elif args.test:
        # 테스트: 대표 5종목
        tickers = ["005930", "000660", "035420", "068270", "105560"]
        logger.info("테스트 모드: %s", tickers)
    else:
        tickers = get_universe_tickers()

    logger.info("대상 종목: %d개", len(tickers))

    # 수집
    results = adapter.collect_universe(
        tickers,
        collect_cf=not args.bs_only,
    )

    # 저장
    adapter.save(results, str(PROJECT_ROOT / OUTPUT_PATH))

    # 요약
    meta = results["meta"]
    quality = results["quality"]
    logger.info("=" * 60)
    logger.info("수집 완료!")
    logger.info("  종목: %d / BS: %d / CF: %d / Quality: %d",
                meta["ticker_count"], meta["bs_count"],
                meta["cf_count"], meta["quality_count"])
    logger.info("  API 호출: %d건", meta["api_calls"])

    # Quality 요약 통계
    roe_values = [q["roe_mean"] for q in quality.values() if q.get("roe_mean") is not None]
    debt_values = [q["debt_ratio"] for q in quality.values() if q.get("debt_ratio") is not None]
    accruals_values = [q["accruals_ratio"] for q in quality.values() if q.get("accruals_ratio") is not None]

    if roe_values:
        import numpy as np
        logger.info("  ROE 평균: %.1f%% (중위수 %.1f%%)",
                    np.mean(roe_values) * 100, np.median(roe_values) * 100)
    if debt_values:
        logger.info("  부채비율 평균: %.1f%% (중위수 %.1f%%)",
                    np.mean(debt_values) * 100, np.median(debt_values) * 100)
    if accruals_values:
        logger.info("  Accruals 평균: %.2f (중위수 %.2f)",
                    np.mean(accruals_values), np.median(accruals_values))

    # 샘플 출력
    logger.info("\n=== 샘플 (처음 5종목) ===")
    for ticker in list(quality.keys())[:5]:
        q = quality[ticker]
        roe_str = f"{q['roe_mean']*100:.1f}%" if q.get("roe_mean") is not None else "N/A"
        debt_str = f"{q['debt_ratio']*100:.1f}%" if q.get("debt_ratio") is not None else "N/A"
        accruals_str = f"{q['accruals_ratio']:.2f}" if q.get("accruals_ratio") is not None else "N/A"
        div_str = f"{q['dividend_payout']*100:.1f}%" if q.get("dividend_payout") is not None else "N/A"
        logger.info(
            "  %s: ROE=%s 부채=%s Accruals=%s 배당=%s (데이터 %d분기)",
            ticker, roe_str, debt_str, accruals_str, div_str,
            q.get("data_points", 0),
        )


if __name__ == "__main__":
    main()
