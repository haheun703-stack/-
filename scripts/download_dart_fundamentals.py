"""
DART OpenAPI 전종목 재무 데이터 일괄 다운로드 스크립트

기능:
  1. stock_data_daily/ 폴더의 전종목 티커 추출
  2. 다중회사 API(100개씩)로 재무제표 일괄 조회
  3. data/dart_cache/fundamentals_all.csv 생성
  4. 이후 스캐너/fundamental.py에서 CSV 로딩하여 사용

사용법:
    python scripts/download_dart_fundamentals.py                # 전종목 다운로드
    python scripts/download_dart_fundamentals.py --year 2025    # 특정 연도
    python scripts/download_dart_fundamentals.py --check        # 현황만 확인
    python scripts/download_dart_fundamentals.py --ticker 005930  # 단일종목 테스트

API 한도: 일 10,000건
  - 고유번호 다운로드: 1건
  - 다중회사(100개): ~29건 (2,860종목 / 100)
  - 단일회사 fallback: 종목당 1~4건
  → 전종목 1회 실행: ~50건 (여유 충분)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# 프로젝트 루트 path 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dart_downloader")

DATA_DIR = Path(__file__).resolve().parent.parent / "stock_data_daily"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "dart_cache"
OUTPUT_FILE = CACHE_DIR / "fundamentals_all.csv"


def get_all_tickers_from_csv() -> list[tuple[str, str]]:
    """stock_data_daily/ 폴더에서 (ticker, name) 추출"""
    tickers = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        name = f.stem  # 예: "삼성전자_005930"
        if "_" in name:
            parts = name.rsplit("_", 1)
            stock_name = parts[0]
            ticker = parts[1]
            if ticker.isdigit() and len(ticker) == 6:
                tickers.append((ticker, stock_name))
    return tickers


def download_all(year: int, reprt_codes: list[str] | None = None):
    """전종목 재무 데이터 일괄 다운로드"""
    from src.adapters.dart_adapter import DartAdapter

    dart = DartAdapter()
    if not dart.is_available:
        logger.error("DART_API_KEY 미설정. .env 파일에 키를 입력해주세요.")
        return

    tickers = get_all_tickers_from_csv()
    logger.info(f"대상 종목: {len(tickers)}개")

    if not reprt_codes:
        reprt_codes = ["11014", "11012", "11013", "11011"]  # Q3→반기→Q1→연간

    # ──────────────────────────────────────────────
    # Phase 1: 다중회사 API로 100개씩 일괄 조회
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Phase 1: 다중회사 API 일괄 조회 (year={year})")
    logger.info("=" * 60)

    all_ticker_codes = [t[0] for t in tickers]
    multi_results = []

    for reprt_code in reprt_codes:
        logger.info(f"  보고서 코드: {reprt_code}")
        batches = [
            all_ticker_codes[i:i + 100]
            for i in range(0, len(all_ticker_codes), 100)
        ]

        batch_total = 0
        for batch_idx, batch in enumerate(batches):
            df = dart.fetch_multi_financials(batch, year, reprt_code)
            if df is not None and len(df) > 0:
                df["_reprt_code"] = reprt_code
                df["_year"] = year
                multi_results.append(df)
                batch_total += len(df)

            logger.info(
                f"    배치 {batch_idx + 1}/{len(batches)}: "
                f"{len(df) if df is not None else 0}건"
            )
            time.sleep(0.2)

        logger.info(f"  → {reprt_code} 합계: {batch_total}건")

        if batch_total > 0:
            break  # 가장 최신 보고서에서 데이터 확보되면 중단

    if not multi_results:
        logger.warning("다중회사 API에서 데이터 없음. 단일회사 조회로 전환.")

    # ──────────────────────────────────────────────
    # Phase 2: 결과 파싱 → fundamentals_all.csv
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2: 결과 파싱 및 CSV 생성")
    logger.info("=" * 60)

    combined = pd.concat(multi_results, ignore_index=True) if multi_results else pd.DataFrame()

    if len(combined) == 0:
        logger.warning("데이터 없음. 단일회사 fallback 시작...")
        combined = _fallback_single_company(dart, tickers, year, reprt_codes)

    if len(combined) == 0:
        logger.error("재무 데이터 수집 실패.")
        return

    # 핵심 계정만 추출하여 요약 테이블 생성
    summary = _build_summary(combined, tickers)
    summary["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 저장
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info(f"저장 완료: {OUTPUT_FILE}")
    logger.info(f"  종목 수: {len(summary)}")
    logger.info(f"  매출 있는 종목: {summary['revenue_억'].notna().sum()}")
    logger.info(f"  영업이익 있는 종목: {summary['op_income_억'].notna().sum()}")
    logger.info(f"  API 호출 수: {dart.get_api_calls_count()}")
    logger.info("=" * 60)


def _fallback_single_company(
    dart, tickers: list[tuple[str, str]], year: int, reprt_codes: list[str]
) -> pd.DataFrame:
    """다중회사 API 실패 시 단일회사 API로 개별 조회"""
    results = []
    total = len(tickers)

    for i, (ticker, name) in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info(f"  진행: {i + 1}/{total} ({(i + 1) / total * 100:.0f}%)")

        for reprt_code in reprt_codes:
            df = dart.fetch_financial_statement(ticker, year, reprt_code)
            if df is not None and len(df) > 0:
                df["_reprt_code"] = reprt_code
                df["_year"] = year
                if "stock_code" not in df.columns:
                    df["stock_code"] = ticker
                results.append(df)
                break  # 최신 보고서 확보 시 다음 종목

        time.sleep(0.05)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _build_summary(combined: pd.DataFrame, tickers: list[tuple[str, str]]) -> pd.DataFrame:
    """raw 재무제표 → 종목별 요약 테이블"""
    # ticker → name 매핑
    name_map = {t[0]: t[1] for t in tickers}

    # stock_code 컬럼 정규화
    if "stock_code" in combined.columns:
        combined["_ticker"] = combined["stock_code"].astype(str).str.zfill(6)
    elif "corp_code" in combined.columns:
        # corp_code → ticker 역매핑이 필요하면 건너뜀
        logger.warning("stock_code 없음, corp_code만 존재")
        combined["_ticker"] = ""
    else:
        combined["_ticker"] = ""

    # 연결(CFS) 우선 필터
    if "fs_div" in combined.columns:
        cfs = combined[combined["fs_div"] == "CFS"]
        if len(cfs) == 0:
            cfs = combined  # 개별만 있으면 그대로
    else:
        cfs = combined

    rows = []
    processed_tickers = set()

    for ticker, name in tickers:
        if ticker in processed_tickers:
            continue

        subset = cfs[cfs["_ticker"] == ticker]
        if len(subset) == 0:
            continue

        revenue = _extract_amount(subset, "매출액")
        op_income = _extract_amount(subset, "영업이익")
        net_income = _extract_amount(subset, "당기순이익")

        revenue_억 = round(revenue / 1e8, 1) if revenue else None
        op_income_억 = round(op_income / 1e8, 1) if op_income else None
        net_income_억 = round(net_income / 1e8, 1) if net_income else None
        op_margin = round(op_income / revenue * 100, 2) if (revenue and op_income and revenue > 0) else None

        reprt_code = subset["_reprt_code"].iloc[0] if "_reprt_code" in subset.columns else ""
        year = subset["_year"].iloc[0] if "_year" in subset.columns else ""

        rows.append({
            "ticker": ticker,
            "name": name,
            "revenue_억": revenue_억,
            "op_income_억": op_income_억,
            "net_income_억": net_income_억,
            "op_margin_pct": op_margin,
            "profitable": op_income > 0 if op_income is not None else None,
            "year": year,
            "reprt_code": reprt_code,
        })
        processed_tickers.add(ticker)

    return pd.DataFrame(rows)


def _extract_amount(df: pd.DataFrame, account_name: str) -> float | None:
    """특정 계정과목 금액 추출"""
    if "account_nm" not in df.columns:
        return None

    row = df[df["account_nm"] == account_name]
    if len(row) == 0:
        return None

    amt_str = row.iloc[0].get("thstrm_amount", "")
    if pd.isna(amt_str) or amt_str == "" or amt_str is None:
        return None
    try:
        return float(str(amt_str).replace(",", ""))
    except (ValueError, TypeError):
        return None


def download_single(ticker: str, year: int):
    """단일 종목 테스트"""
    from src.adapters.dart_adapter import DartAdapter

    dart = DartAdapter()
    if not dart.is_available:
        logger.error("DART_API_KEY 미설정.")
        return

    logger.info(f"단일 종목 조회: {ticker} (year={year})")

    result = dart.get_key_financials(ticker, year)
    if result.get("revenue") is not None:
        logger.info(f"  매출:       {result['revenue']:,.0f}억원")
        logger.info(f"  영업이익:   {result['operating_income']:,.0f}억원")
        logger.info(f"  순이익:     {result['net_income']:,.0f}억원")
        logger.info(f"  영업이익률: {result['operating_margin']:.2f}%")
        logger.info(f"  흑자여부:   {result['profitable']}")
    else:
        logger.warning(f"  데이터 없음 (올해 미공시일 수 있음)")

    logger.info(f"  API 호출: {dart.get_api_calls_count()}건")


def check_status():
    """현재 캐시 현황 확인"""
    tickers = get_all_tickers_from_csv()
    logger.info(f"stock_data_daily/ 종목 수: {len(tickers)}")

    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        logger.info(f"fundamentals_all.csv: {len(df)}종목")
        logger.info(f"  매출 데이터: {df['revenue_억'].notna().sum()}종목")
        logger.info(f"  영업이익 데이터: {df['op_income_억'].notna().sum()}종목")
        logger.info(f"  흑자 종목: {df['profitable'].sum() if 'profitable' in df.columns else 'N/A'}")
        logger.info(f"  최종 업데이트: {df['updated_at'].iloc[0] if 'updated_at' in df.columns else 'N/A'}")
    else:
        logger.info("fundamentals_all.csv: 미생성 (다운로드 필요)")

    # DART 캐시 현황
    cache_files = list(CACHE_DIR.glob("finstate_*.csv")) if CACHE_DIR.exists() else []
    corp_codes = CACHE_DIR / "corp_codes.csv"
    logger.info(f"DART 캐시: corp_codes={'있음' if corp_codes.exists() else '없음'}, 개별 캐시={len(cache_files)}개")

    # API 키 확인
    api_key = os.getenv("DART_API_KEY", "")
    logger.info(f"DART_API_KEY: {'설정됨 ({0}...{1})'.format(api_key[:4], api_key[-4:]) if len(api_key) > 8 else '미설정'}")


def main():
    parser = argparse.ArgumentParser(description="DART 전종목 재무 데이터 다운로드")
    parser.add_argument("--year", type=int, default=datetime.now().year,
                        help="사업연도 (기본: 올해)")
    parser.add_argument("--ticker", type=str, default=None,
                        help="단일 종목 테스트 (예: 005930)")
    parser.add_argument("--check", action="store_true",
                        help="현황만 확인")
    parser.add_argument("--prev-year", action="store_true",
                        help="전년도도 함께 다운로드")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    if args.ticker:
        download_single(args.ticker, args.year)
        return

    download_all(args.year)

    if args.prev_year:
        logger.info(f"\n전년도({args.year - 1}) 추가 다운로드...")
        download_all(args.year - 1)


if __name__ == "__main__":
    main()
