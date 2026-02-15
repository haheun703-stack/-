"""기존 parquet 데이터를 최신 날짜까지 증분 업데이트

기존 data/raw/*.parquet 파일의 마지막 날짜 이후 데이터를 pykrx에서 가져와 추가.
전체 재수집(3~4시간) 대신 증분 업데이트(3~5분)로 빠르게 처리.

사용법:
  python scripts/extend_parquet_data.py                    # 오늘까지
  python scripts/extend_parquet_data.py --end 20250214     # 특정 날짜까지
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    logger.error("pykrx 미설치: pip install pykrx")


def extend_single(parquet_path: Path, end_date: str) -> dict:
    """단일 parquet 파일 증분 업데이트"""
    ticker = parquet_path.stem
    result = {"ticker": ticker, "status": "skip", "new_rows": 0}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    last_date = df.index.max()
    last_date_str = last_date.strftime("%Y%m%d")

    # 이미 최신이면 skip
    if last_date_str >= end_date:
        return result

    # 다음 거래일부터 조회
    fetch_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    try:
        # OHLCV (pykrx 반환: 시가/고가/저가/종가/거래량/등락률 — 6컬럼)
        new_ohlcv = krx.get_market_ohlcv_by_date(fetch_start, end_date, ticker, adjusted=True)
        if new_ohlcv is None or new_ohlcv.empty:
            return result

        new_ohlcv.index.name = "date"
        # pykrx 한글 컬럼 → 영문 매핑
        col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
                    "거래량": "volume", "등락률": "price_change", "거래대금": "trading_value"}
        new_ohlcv = new_ohlcv.rename(columns=col_map)

        # trading_value 컬럼이 없으면 0으로 추가
        if "trading_value" not in new_ohlcv.columns:
            new_ohlcv["trading_value"] = 0

        # 중복 제거
        existing_dates = set(df.index)
        new_ohlcv = new_ohlcv[~new_ohlcv.index.isin(existing_dates)]

        if new_ohlcv.empty:
            return result

        # 새 데이터를 기존 parquet 컬럼 구조에 맞춤
        new_rows = pd.DataFrame(index=new_ohlcv.index)
        new_rows.index.name = "date"

        for col in df.columns:
            if col in new_ohlcv.columns:
                new_rows[col] = new_ohlcv[col]
            else:
                new_rows[col] = 0.0  # 없는 컬럼은 0으로 채움

        # 투자자 매매동향 업데이트 시도
        try:
            inv_df = krx.get_market_trading_value_by_date(fetch_start, end_date, ticker)
            if inv_df is not None and not inv_df.empty:
                inv_df.index.name = "date"
                inv_col_map = {"기관합계": "기관합계", "외국인합계": "외국인합계", "개인": "개인"}
                for kr_col, en_col in inv_col_map.items():
                    if kr_col in inv_df.columns and en_col in new_rows.columns:
                        common_idx = inv_df.index.intersection(new_rows.index)
                        if len(common_idx) > 0:
                            new_rows.loc[common_idx, en_col] = inv_df.loc[common_idx, kr_col]
            time.sleep(0.3)
        except Exception:
            pass

        # 펀더멘탈 업데이트 시도
        try:
            fund_df = krx.get_market_fundamental_by_date(fetch_start, end_date, ticker)
            if fund_df is not None and not fund_df.empty:
                fund_df.index.name = "date"
                fund_col_map = {"BPS": "fund_BPS", "PER": "fund_PER", "PBR": "fund_PBR",
                                "EPS": "fund_EPS", "DIV": "fund_DIV", "DPS": "fund_DPS"}
                for kr_col, en_col in fund_col_map.items():
                    if kr_col in fund_df.columns and en_col in new_rows.columns:
                        common_idx = fund_df.index.intersection(new_rows.index)
                        if len(common_idx) > 0:
                            new_rows.loc[common_idx, en_col] = fund_df.loc[common_idx, kr_col]
            time.sleep(0.3)
        except Exception:
            pass

        # 기존 + 신규 합치기
        combined = pd.concat([df, new_rows])
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

        # 저장
        combined.to_parquet(parquet_path)

        result["status"] = "ok"
        result["new_rows"] = len(new_rows)
        result["new_end"] = combined.index.max().strftime("%Y-%m-%d")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="parquet 데이터 증분 업데이트")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYYMMDD)")
    args = parser.parse_args()

    if not PYKRX_AVAILABLE:
        logger.error("pykrx 미설치. 종료.")
        return

    end_date = args.end or datetime.today().strftime("%Y%m%d")
    raw_dir = project_root / "data" / "raw"
    parquets = sorted(raw_dir.glob("*.parquet"))

    logger.info(f"증분 업데이트 시작: {len(parquets)}종목 → {end_date}까지")

    updated = 0
    skipped = 0
    errors = 0
    error_list = []

    for i, p in enumerate(parquets):
        result = extend_single(p, end_date)

        if result["status"] == "ok":
            updated += 1
            logger.info(f"  [{i+1}/{len(parquets)}] {result['ticker']}: +{result['new_rows']}행 → {result.get('new_end', '')}")
        elif result["status"] == "error":
            errors += 1
            error_list.append(f"{result['ticker']}: {result.get('error', '')}")
            logger.warning(f"  [{i+1}/{len(parquets)}] {result['ticker']}: 오류 - {result.get('error', '')}")
        else:
            skipped += 1

        # pykrx rate limit
        if result["status"] == "ok":
            time.sleep(0.5)

        if (i + 1) % 20 == 0:
            logger.info(f"  --- 진행: {i+1}/{len(parquets)} | 업데이트: {updated} | 스킵: {skipped}")

    logger.info(f"\n{'='*50}")
    logger.info(f"증분 업데이트 완료 (→ {end_date})")
    logger.info(f"  업데이트: {updated}종목")
    logger.info(f"  스킵(최신): {skipped}종목")
    logger.info(f"  오류: {errors}종목")
    if error_list:
        for e in error_list[:10]:
            logger.info(f"    - {e}")
    logger.info(f"{'='*50}")

    # 검증
    if updated > 0:
        sample = parquets[0]
        df = pd.read_parquet(sample)
        logger.info(f"\n검증: {sample.stem} → {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}rows)")


if __name__ == "__main__":
    main()
