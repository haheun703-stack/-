"""섹터 순환매 엔진 — Phase 1-1: ETF 기반 섹터 매핑 자동 구축.

pykrx API로 TIGER 섹터 ETF 구성종목을 조회하여
섹터→종목 매핑 + ETF 일별 시세를 수집한다.

사용법:
  python scripts/sector_etf_builder.py --init     # 최초 구축 (구성종목 + 120일 시세)
  python scripts/sector_etf_builder.py --daily     # 일별 시세 업데이트
  python scripts/sector_etf_builder.py --refresh   # 구성종목 재조회 (월 1회)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx
except ImportError:
    logger.error("pykrx 미설치: pip install pykrx")
    sys.exit(1)

# ─────────────────────────────────────────────
# 디렉토리 설정
# ─────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
COMP_DIR = DATA_DIR / "etf_compositions"
DAILY_DIR = DATA_DIR / "etf_daily"

for d in [DATA_DIR, COMP_DIR, DAILY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# TIGER 국내 섹터 ETF 정의 (pykrx 검증 완료)
# ─────────────────────────────────────────────

SECTOR_ETFS = {
    # 섹터명: (ETF코드, ETF명, 카테고리)
    "증권": ("157500", "TIGER 증권", "sector"),
    "은행": ("091220", "TIGER 은행", "sector"),
    "보험": ("140710", "TIGER 보험", "sector"),
    "반도체": ("091230", "TIGER 반도체", "sector"),
    "2차전지": ("305540", "TIGER 2차전지테마", "theme"),
    "바이오": ("364970", "TIGER 바이오TOP10", "theme"),
    "헬스케어": ("143860", "TIGER 헬스케어", "sector"),
    "IT": ("139260", "TIGER 200 IT", "sector"),
    "건설": ("139220", "TIGER 200 건설", "sector"),
    "금융": ("139270", "TIGER 200 금융", "sector"),
    "에너지화학": ("139250", "TIGER 200 에너지화학", "sector"),
    "철강소재": ("139240", "TIGER 200 철강소재", "sector"),
    "소프트웨어": ("157490", "TIGER 소프트웨어", "sector"),
    "미디어": ("228810", "TIGER 미디어컨텐츠", "sector"),
    "게임": ("300610", "TIGER K게임", "theme"),
    "인터넷": ("365000", "TIGER 인터넷TOP10", "theme"),
    "방산": ("463250", "TIGER K방산&우주", "theme"),
    "조선": ("494670", "TIGER 조선TOP10", "theme"),
    # 그룹형
    "삼성그룹": ("138520", "TIGER 삼성그룹", "group"),
    "현대차그룹": ("138540", "TIGER 현대차그룹플러스", "group"),
    # 시장 대표
    "KRX300": ("292160", "TIGER KRX300", "market"),
    "코리아TOP10": ("292150", "TIGER 코리아TOP10", "market"),
}


# ─────────────────────────────────────────────
# 1. ETF 구성종목 조회
# ─────────────────────────────────────────────

def fetch_compositions(ref_date: str = "") -> dict:
    """모든 섹터 ETF의 구성종목을 조회하여 저장."""
    if not ref_date:
        ref_date = krx.get_nearest_business_day_in_a_week(
            datetime.now().strftime("%Y%m%d"), prev=True
        )

    logger.info("ETF 구성종목 조회 기준일: %s", ref_date)

    sector_map = {}
    stock_to_sector = {}
    etf_universe = {}

    for sector_name, (etf_code, etf_name, category) in SECTOR_ETFS.items():
        try:
            # pykrx: ticker first, date second
            pdf = krx.get_etf_portfolio_deposit_file(etf_code, ref_date)
            time.sleep(0.5)

            if pdf is None or pdf.empty:
                logger.warning("%s (%s): 구성종목 없음", sector_name, etf_code)
                continue

            # 구성종목 정리 — 현금/선물 등 비주식 항목 제외
            stocks = []
            # 금액 컬럼이 Series인지 확인 (중복 인덱스 대비)
            if "금액" in pdf.columns:
                amounts = pdf["금액"]
                total_value = float(amounts.sum())
            else:
                total_value = 1.0

            for ticker in pdf.index.unique():
                # 6자리 숫자 티커만 (주식)
                if not ticker.isdigit() or len(ticker) != 6:
                    continue

                try:
                    name = krx.get_market_ticker_name(ticker)
                    time.sleep(0.03)
                    if name is None:
                        name = ticker
                except Exception:
                    name = ticker

                weight = 0.0
                if total_value > 0 and "금액" in pdf.columns:
                    val = amounts.loc[ticker]
                    # 중복 인덱스 시 합산
                    val = float(val.sum()) if hasattr(val, "sum") and not isinstance(val, (int, float)) else float(val)
                    weight = round(val / total_value * 100, 2)

                stocks.append({
                    "code": ticker,
                    "name": str(name),
                    "weight": float(weight),
                })

                # 역매핑
                if ticker not in stock_to_sector:
                    stock_to_sector[ticker] = []
                if sector_name not in stock_to_sector[ticker]:
                    stock_to_sector[ticker].append(sector_name)

            # 비중 기준 정렬
            stocks.sort(key=lambda x: x["weight"], reverse=True)

            sector_map[sector_name] = {
                "etf_code": etf_code,
                "stocks": stocks,
            }

            etf_universe[sector_name] = {
                "etf_code": etf_code,
                "etf_name": etf_name,
                "category": category,
                "stock_count": len(stocks),
            }

            # 개별 구성종목 파일 저장
            comp_path = COMP_DIR / f"{etf_code}_{sector_name}.json"
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump({"sector": sector_name, "etf_code": etf_code,
                           "date": ref_date, "stocks": stocks},
                          f, ensure_ascii=False, indent=2)

            logger.info("  %s (%s): %d종목", sector_name, etf_code, len(stocks))

        except Exception as e:
            logger.error("%s (%s): %s", sector_name, etf_code, e)

    # 통합 파일 저장
    with open(DATA_DIR / "etf_universe.json", "w", encoding="utf-8") as f:
        json.dump(etf_universe, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "sector_map.json", "w", encoding="utf-8") as f:
        json.dump(sector_map, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "stock_to_sector.json", "w", encoding="utf-8") as f:
        json.dump(stock_to_sector, f, ensure_ascii=False, indent=2)

    logger.info(
        "구성종목 수집 완료: %d섹터, %d종목 매핑",
        len(sector_map), len(stock_to_sector),
    )
    return sector_map


# ─────────────────────────────────────────────
# 2. ETF 일별 시세 수집
# ─────────────────────────────────────────────

def fetch_etf_daily(days: int = 120) -> int:
    """모든 섹터 ETF의 일별 OHLCV를 수집."""
    end_date = krx.get_nearest_business_day_in_a_week(
        datetime.now().strftime("%Y%m%d"), prev=True
    )
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")

    logger.info("ETF 시세 수집: %s ~ %s (%d일)", start_date, end_date, days)

    count = 0
    for sector_name, (etf_code, etf_name, _) in SECTOR_ETFS.items():
        try:
            ohlcv = krx.get_etf_ohlcv_by_date(start_date, end_date, etf_code)
            time.sleep(0.3)

            if ohlcv is None or ohlcv.empty:
                logger.warning("%s (%s): 시세 없음", sector_name, etf_code)
                continue

            ohlcv.index.name = "date"
            col_map = {
                "NAV": "nav", "시가": "open", "고가": "high",
                "저가": "low", "종가": "close",
                "거래량": "volume", "거래대금": "trading_value",
                "기초지수": "base_index",
            }
            ohlcv = ohlcv.rename(columns=col_map)

            out_path = DAILY_DIR / f"{etf_code}.parquet"
            ohlcv.to_parquet(out_path)
            count += 1

            logger.info("  %s (%s): %d거래일", sector_name, etf_code, len(ohlcv))

        except Exception as e:
            logger.error("%s (%s): %s", sector_name, etf_code, e)

    logger.info("ETF 시세 수집 완료: %d개", count)
    return count


def update_etf_daily() -> int:
    """기존 시세에 최신 데이터를 증분 추가."""
    end_date = krx.get_nearest_business_day_in_a_week(
        datetime.now().strftime("%Y%m%d"), prev=True
    )

    count = 0
    for sector_name, (etf_code, _, _) in SECTOR_ETFS.items():
        out_path = DAILY_DIR / f"{etf_code}.parquet"

        try:
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                last_date = existing.index[-1].strftime("%Y%m%d")
                if last_date >= end_date:
                    continue  # 이미 최신
                start = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
                existing = None

            ohlcv = krx.get_etf_ohlcv_by_date(start, end_date, etf_code)
            time.sleep(0.3)

            if ohlcv is None or ohlcv.empty:
                continue

            ohlcv.index.name = "date"
            col_map = {
                "NAV": "nav", "시가": "open", "고가": "high",
                "저가": "low", "종가": "close",
                "거래량": "volume", "거래대금": "trading_value",
                "기초지수": "base_index",
            }
            ohlcv = ohlcv.rename(columns=col_map)

            if existing is not None:
                combined = pd.concat([existing, ohlcv])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
                combined.to_parquet(out_path)
            else:
                ohlcv.to_parquet(out_path)

            count += 1

        except Exception as e:
            logger.error("%s (%s): %s", sector_name, etf_code, e)

    logger.info("ETF 시세 업데이트: %d개 갱신", count)
    return count


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="섹터 ETF 유니버스 구축")
    parser.add_argument("--init", action="store_true",
                        help="최초 구축 (구성종목 + 120일 시세)")
    parser.add_argument("--daily", action="store_true",
                        help="일별 시세 업데이트")
    parser.add_argument("--refresh", action="store_true",
                        help="구성종목 재조회")
    parser.add_argument("--days", type=int, default=200,
                        help="시세 수집 기간 (일, 기본 200)")
    args = parser.parse_args()

    if args.init:
        print("=" * 50)
        print("  섹터 ETF 유니버스 최초 구축")
        print(f"  대상: {len(SECTOR_ETFS)}개 섹터 ETF")
        print("=" * 50)
        fetch_compositions()
        fetch_etf_daily(days=args.days)
        print("\n최초 구축 완료!")

    elif args.daily:
        update_etf_daily()

    elif args.refresh:
        fetch_compositions()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
