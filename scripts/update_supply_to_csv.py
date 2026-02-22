"""stock_data_daily/ CSV의 Foreign_Net, Inst_Net 컬럼 갱신

pykrx를 통해 투자자별 순매수 수량을 조회하고,
CSV 파일의 Foreign_Net(외국인합계), Inst_Net(기관합계) 컬럼을 업데이트한다.

사용법:
  python scripts/update_supply_to_csv.py                  # 전체 CSV 갱신
  python scripts/update_supply_to_csv.py --check          # 현황만 확인
  python scripts/update_supply_to_csv.py --tickers 005930 036560  # 특정 종목만

주의:
  - pykrx rate limit: 종목당 ~1초 대기
  - 전체 2,800+ 종목 실행 시 약 50분 소요
  - 주요 종목만 갱신하려면 --tickers 옵션 사용
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx
except ImportError:
    logger.error("pykrx 미설치: pip install pykrx")
    sys.exit(1)

DATA_DIR = Path(__file__).resolve().parent.parent / "stock_data_daily"


def find_supply_gap(csv_path: Path) -> dict:
    """CSV에서 수급 데이터 갭을 찾는다."""
    fname = csv_path.stem
    parts = fname.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return {"file": fname, "ticker": "", "gap_start": "", "gap_end": "", "gap_days": 0}

    ticker = parts[1]
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns or len(df) == 0:
        return {"file": fname, "ticker": ticker, "gap_start": "", "gap_end": "", "gap_days": 0}

    if "Foreign_Net" not in df.columns:
        return {"file": fname, "ticker": ticker, "gap_start": df["Date"].iloc[0],
                "gap_end": df["Date"].iloc[-1], "gap_days": len(df)}

    last_date = str(df["Date"].iloc[-1])[:10]

    # 수급 데이터가 0인 마지막 연속 구간 찾기
    has_data = df[(df["Foreign_Net"] != 0) | (df["Inst_Net"] != 0)]
    if len(has_data) == 0:
        # 전체가 0 — 마지막 60일만 갱신 (너무 오래되면 pykrx 한계)
        gap_start_idx = max(0, len(df) - 60)
        gap_start = str(df["Date"].iloc[gap_start_idx])[:10]
        return {"file": fname, "ticker": ticker, "gap_start": gap_start,
                "gap_end": last_date, "gap_days": len(df) - gap_start_idx}

    last_supply_date = str(has_data["Date"].iloc[-1])[:10]
    if last_supply_date >= last_date:
        return {"file": fname, "ticker": ticker, "gap_start": "", "gap_end": "", "gap_days": 0}

    # 갭: 수급 마지막 날짜 다음 ~ CSV 마지막 날짜
    gap_rows = df[df["Date"] > last_supply_date]
    gap_start = str(gap_rows["Date"].iloc[0])[:10] if len(gap_rows) > 0 else ""

    return {"file": fname, "ticker": ticker, "gap_start": gap_start,
            "gap_end": last_date, "gap_days": len(gap_rows)}


def update_single_csv(csv_path: Path, force_start: str = None) -> dict:
    """단일 CSV 파일의 Foreign_Net/Inst_Net 갱신."""
    fname = csv_path.stem
    parts = fname.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return {"file": fname, "status": "skip", "updated": 0, "error": "종목코드 추출 실패"}

    ticker = parts[1]
    result = {"file": fname, "ticker": ticker, "status": "skip", "updated": 0, "error": ""}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    if "Date" not in df.columns or len(df) == 0:
        result["status"] = "error"
        result["error"] = "Date 컬럼 없음"
        return result

    if "Foreign_Net" not in df.columns:
        df["Foreign_Net"] = 0.0
    if "Inst_Net" not in df.columns:
        df["Inst_Net"] = 0.0

    last_date = str(df["Date"].iloc[-1])[:10]

    # 갭 날짜 계산
    if force_start:
        gap_start = force_start
    else:
        has_data = df[(df["Foreign_Net"] != 0) | (df["Inst_Net"] != 0)]
        if len(has_data) > 0:
            last_supply = str(has_data["Date"].iloc[-1])[:10]
            if last_supply >= last_date:
                result["status"] = "skip"
                return result
            gap_start = last_supply.replace("-", "")
        else:
            # 전체 0: 최근 60일만
            gap_start_idx = max(0, len(df) - 60)
            gap_start = str(df["Date"].iloc[gap_start_idx])[:10].replace("-", "")

    gap_end = last_date.replace("-", "")

    # pykrx 조회
    try:
        inv_df = krx.get_market_trading_volume_by_date(gap_start, gap_end, ticker)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"pykrx 조회 실패: {e}"
        return result

    if inv_df is None or inv_df.empty:
        result["status"] = "no_data"
        return result

    # 날짜 매핑
    inv_df.index = inv_df.index.strftime("%Y-%m-%d")

    updated = 0
    for idx in range(len(df)):
        date_str = str(df.at[idx, "Date"])[:10]
        if date_str in inv_df.index:
            row = inv_df.loc[date_str]
            foreign = row.get("외국인합계", 0)
            inst = row.get("기관합계", 0)

            if foreign != 0 or inst != 0:
                old_f = df.at[idx, "Foreign_Net"]
                old_i = df.at[idx, "Inst_Net"]
                if old_f == 0 and old_i == 0:
                    df.at[idx, "Foreign_Net"] = float(foreign)
                    df.at[idx, "Inst_Net"] = float(inst)
                    updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        result["status"] = "ok"
        result["updated"] = updated
    else:
        result["status"] = "no_change"

    return result


def check_status():
    """전체 CSV 수급 갭 현황 확인."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]

    total = len(csv_files)
    needs_update = 0
    all_zero = 0
    up_to_date = 0

    logger.info(f"전체 CSV 파일: {total}개")

    for f in csv_files:
        gap = find_supply_gap(f)
        if gap["gap_days"] > 0:
            needs_update += 1
        elif gap["gap_start"] == "" and gap["gap_end"] == "":
            up_to_date += 1

    logger.info(f"  최신: {up_to_date}개")
    logger.info(f"  갱신 필요: {needs_update}개")
    logger.info(f"  기타: {total - up_to_date - needs_update}개")


def main():
    parser = argparse.ArgumentParser(description="CSV 수급(외인/기관) 데이터 갱신")
    parser.add_argument("--tickers", nargs="*", help="특정 종목코드만 갱신")
    parser.add_argument("--check", action="store_true", help="현황만 확인")
    parser.add_argument("--universe", action="store_true",
                        help="data/processed/*.parquet 유니버스만 갱신 (84종목)")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    # 대상 파일 결정
    if args.tickers:
        # 특정 종목
        csv_files = []
        for t in args.tickers:
            matches = list(DATA_DIR.glob(f"*_{t}.csv"))
            csv_files.extend(matches)
        if not csv_files:
            logger.error(f"지정한 종목의 CSV 파일을 찾을 수 없습니다: {args.tickers}")
            return
    elif args.universe:
        # 유니버스(84종목) + 관심종목
        parquet_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        tickers = set()
        for pf in parquet_dir.glob("*.parquet"):
            tickers.add(pf.stem)

        # 관심종목 추가 (현재 보유 + 분석 대상)
        extra_tickers = [
            "036560",  # KZ정밀
            "185750",  # 종근당
            "267270",  # HD현대건설기계
            "039130",  # 하나투어
            "021240",  # 코웨이
            "006360",  # GS건설
            "011200",  # HMM
            "006805",  # 미래에셋증권우
        ]
        tickers.update(extra_tickers)

        csv_files = []
        for t in tickers:
            matches = list(DATA_DIR.glob(f"*_{t}.csv"))
            csv_files.extend(matches)
    else:
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.startswith("_")]

    total = len(csv_files)
    logger.info(f"수급 갱신 대상: {total}개 CSV")

    ok_count = 0
    skip_count = 0
    error_count = 0
    total_updated = 0
    start_time = time.time()

    for i, csv_path in enumerate(csv_files):
        result = update_single_csv(csv_path)

        if result["status"] == "ok":
            ok_count += 1
            total_updated += result["updated"]
            logger.info(f"  [{i+1}/{total}] {result['file']}: +{result['updated']}행 수급 갱신")
        elif result["status"] == "error":
            error_count += 1
            logger.warning(f"  [{i+1}/{total}] {result['file']}: 오류 - {result['error']}")
        else:
            skip_count += 1

        # pykrx rate limit
        if result["status"] in ("ok", "no_data", "no_change"):
            time.sleep(1.0)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  --- 진행: {i+1}/{total} ({elapsed:.0f}초) | 갱신: {ok_count} | 스킵: {skip_count}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"수급 갱신 완료")
    logger.info(f"{'='*50}")
    logger.info(f"  대상: {total}개")
    logger.info(f"  갱신: {ok_count}개 (총 {total_updated}행)")
    logger.info(f"  스킵: {skip_count}개")
    logger.info(f"  오류: {error_count}개")
    logger.info(f"  소요: {elapsed:.0f}초 ({elapsed/60:.1f}분)")


if __name__ == "__main__":
    main()
