"""유니버스 재구성 스크립트.

시총 기준으로 KOSPI+KOSDAQ 종목을 필터링하고,
신규 종목의 raw parquet(3년 OHLCV)을 다운로드한 뒤,
전종목 processed parquet(지표 계산)을 생성한다.

사용법:
  python scripts/rebuild_universe.py                    # 기본 (1.5조)
  python scripts/rebuild_universe.py --min-cap 2.0      # 시총 2.0조 이상
  python scripts/rebuild_universe.py --download-only     # 다운로드만
  python scripts/rebuild_universe.py --process-only      # 지표 계산만
"""

from __future__ import annotations

import argparse
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

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UNIVERSE_PATH = PROJECT_ROOT / "data" / "universe.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

try:
    from pykrx import stock as krx
except ImportError:
    logger.error("pykrx 미설치: pip install pykrx")
    sys.exit(1)


# ─────────────────────────────────────────────
# 1단계: 시총 기준 종목 선정
# ─────────────────────────────────────────────

def select_universe(min_cap_trillion: float = 1.5, ref_date: str = "") -> pd.DataFrame:
    """시총 기준 유니버스 선정. Returns DataFrame(ticker, name, market_cap)."""
    if not ref_date:
        # 최근 거래일
        ref_date = krx.get_nearest_business_day_in_a_week(
            datetime.now().strftime("%Y%m%d"), prev=True
        )

    logger.info("시총 조회 기준일: %s", ref_date)
    min_cap = min_cap_trillion * 1e12

    kospi = krx.get_market_cap(ref_date, market="KOSPI")
    time.sleep(0.5)
    kosdaq = krx.get_market_cap(ref_date, market="KOSDAQ")

    all_cap = pd.concat([kospi, kosdaq])
    qualified = all_cap[all_cap["시가총액"] >= min_cap].copy()
    qualified = qualified.sort_values("시가총액", ascending=False)

    # 스팩/리츠/우선주/ETF 제거
    excluded = set()
    for ticker in qualified.index:
        try:
            name = krx.get_market_ticker_name(ticker)
            time.sleep(0.05)
        except Exception:
            name = ""
        if _is_excluded(name, ticker):
            excluded.add(ticker)

    qualified = qualified.drop(index=list(excluded), errors="ignore")

    # 종목명 추가
    names = {}
    for ticker in qualified.index:
        try:
            names[ticker] = krx.get_market_ticker_name(ticker)
            time.sleep(0.05)
        except Exception:
            names[ticker] = ticker

    result = pd.DataFrame({
        "ticker": qualified.index,
        "name": [names.get(t, t) for t in qualified.index],
        "market_cap": qualified["시가총액"].values,
    })

    logger.info(
        "유니버스 선정: %d종목 (시총 %.1f조 이상, %d종목 제외)",
        len(result), min_cap_trillion, len(excluded),
    )
    return result


def _is_excluded(name: str, ticker: str) -> bool:
    """스팩/리츠/우선주/ETF/ETN 제거."""
    if not name:
        return False
    excludes = ["스팩", "SPAC", "리츠", "REIT", "ETN"]
    for kw in excludes:
        if kw in name:
            return True
    # 우선주 (코드 끝 5~9)
    if ticker[-1] in "56789" and ticker[-2] == "0":
        return True
    return False


# ─────────────────────────────────────────────
# 2단계: raw parquet 다운로드
# ─────────────────────────────────────────────

def download_raw(tickers: list[str], years: int = 3) -> dict:
    """신규 종목 OHLCV raw parquet 다운로드."""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")

    stats = {"success": 0, "skip": 0, "fail": 0}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        raw_path = RAW_DIR / f"{ticker}.parquet"
        if raw_path.exists():
            stats["skip"] += 1
            continue

        try:
            ohlcv = krx.get_market_ohlcv_by_date(start_date, end_date, ticker, adjusted=True)
            if ohlcv is None or ohlcv.empty:
                logger.warning("[%d/%d] %s: 데이터 없음", i + 1, total, ticker)
                stats["fail"] += 1
                continue

            ohlcv.index.name = "date"
            col_map = {
                "시가": "open", "고가": "high", "저가": "low", "종가": "close",
                "거래량": "volume", "등락률": "price_change", "거래대금": "trading_value",
            }
            ohlcv = ohlcv.rename(columns=col_map)
            if "trading_value" not in ohlcv.columns:
                ohlcv["trading_value"] = 0

            ohlcv.to_parquet(raw_path)
            stats["success"] += 1

            if (i + 1) % 20 == 0:
                logger.info("[%d/%d] 다운로드 진행 중... (성공 %d)", i + 1, total, stats["success"])

            time.sleep(0.3)  # pykrx rate limit
        except Exception as e:
            logger.error("[%d/%d] %s: %s", i + 1, total, ticker, e)
            stats["fail"] += 1
            time.sleep(1)

    logger.info(
        "다운로드 완료: 성공 %d / 스킵 %d / 실패 %d",
        stats["success"], stats["skip"], stats["fail"],
    )
    return stats


# ─────────────────────────────────────────────
# 3단계: processed parquet (지표 계산)
# ─────────────────────────────────────────────

def process_all(tickers: list[str]) -> dict:
    """raw → processed (IndicatorEngine으로 기술적 지표 계산)."""
    from src.indicators import IndicatorEngine

    engine = IndicatorEngine()

    # raw 디렉토리에 있는 것만 처리 (유니버스 한정)
    valid_raw = set(p.stem for p in RAW_DIR.glob("*.parquet"))
    target = [t for t in tickers if t in valid_raw]

    logger.info("지표 계산 대상: %d종목 (raw 존재)", len(target))

    # IndicatorEngine.process_all()은 raw 전체를 처리하므로
    # 유니버스 한정 처리를 위해 직접 compute_all 호출
    stats = {"success": 0, "skip": 0, "fail": 0}
    total = len(target)

    for i, ticker in enumerate(target):
        raw_path = RAW_DIR / f"{ticker}.parquet"
        out_path = PROCESSED_DIR / f"{ticker}.parquet"

        try:
            df = pd.read_parquet(raw_path)
            if len(df) < 200:
                logger.debug("[%d/%d] %s: 데이터 부족 (%d행)", i + 1, total, ticker, len(df))
                stats["skip"] += 1
                continue

            result = engine.compute_all(df)
            result.to_parquet(out_path)
            stats["success"] += 1

            if (i + 1) % 50 == 0:
                logger.info("[%d/%d] 지표 계산 진행 중...", i + 1, total)
        except Exception as e:
            logger.error("[%d/%d] %s: %s", i + 1, total, ticker, e)
            stats["fail"] += 1

    logger.info(
        "지표 계산 완료: 성공 %d / 스킵 %d / 실패 %d",
        stats["success"], stats["skip"], stats["fail"],
    )
    return stats


# ─────────────────────────────────────────────
# 4단계: 시총 미달 종목 정리
# ─────────────────────────────────────────────

def cleanup_dropped(valid_tickers: set[str]) -> int:
    """유니버스에서 탈락한 종목의 processed parquet 제거."""
    removed = 0
    for pf in PROCESSED_DIR.glob("*.parquet"):
        if pf.stem not in valid_tickers:
            pf.unlink()
            removed += 1
            logger.info("탈락 제거: %s", pf.stem)
    return removed


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="유니버스 재구성")
    parser.add_argument("--min-cap", type=float, default=1.5,
                        help="최소 시총 (조 단위, 기본 1.5)")
    parser.add_argument("--download-only", action="store_true",
                        help="다운로드만 실행")
    parser.add_argument("--process-only", action="store_true",
                        help="지표 계산만 실행")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="탈락 종목 삭제 안 함")
    args = parser.parse_args()

    print("=" * 50)
    print(f"  유니버스 재구성 (시총 {args.min_cap}조 이상)")
    print("=" * 50)

    # 1) 종목 선정
    if not args.process_only:
        universe = select_universe(args.min_cap)
        universe.to_csv(UNIVERSE_PATH, index=False)
        print(f"\n유니버스 저장: {UNIVERSE_PATH} ({len(universe)}종목)")

        tickers = universe["ticker"].tolist()

        # 2) 다운로드
        existing_raw = set(p.stem for p in RAW_DIR.glob("*.parquet"))
        new_tickers = [t for t in tickers if t not in existing_raw]
        print(f"신규 다운로드 대상: {len(new_tickers)}종목")

        if new_tickers:
            download_raw(new_tickers)

        if args.download_only:
            print("다운로드 완료 (--download-only)")
            return
    else:
        # process-only: 기존 universe.csv 사용
        if UNIVERSE_PATH.exists():
            universe = pd.read_csv(UNIVERSE_PATH)
            tickers = universe["ticker"].astype(str).str.zfill(6).tolist()
        else:
            tickers = [p.stem for p in RAW_DIR.glob("*.parquet")]

    # 3) 지표 계산
    print(f"\n지표 계산 대상: {len(tickers)}종목")
    process_all(tickers)

    # 4) 탈락 종목 정리
    if not args.no_cleanup:
        valid = set(str(t).zfill(6) for t in tickers)
        removed = cleanup_dropped(valid)
        if removed:
            print(f"탈락 종목 {removed}개 processed에서 제거")

    # 결과
    final_count = len(list(PROCESSED_DIR.glob("*.parquet")))
    print(f"\n최종 유니버스: {final_count}종목")
    print("완료!")


if __name__ == "__main__":
    main()
