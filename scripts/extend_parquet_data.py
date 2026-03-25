"""기존 parquet 데이터를 최신 날짜까지 증분 업데이트

기존 data/raw/*.parquet 파일의 마지막 날짜 이후 데이터를 pykrx에서 가져와 추가.
ThreadPoolExecutor 병렬 처리로 1,071종목을 5분 이내에 완료.

사용법:
  python scripts/extend_parquet_data.py                    # 오늘까지
  python scripts/extend_parquet_data.py --end 20250214     # 특정 날짜까지
  python scripts/extend_parquet_data.py --sample 50        # 50종목 샘플 테스트
  python scripts/extend_parquet_data.py --workers 3        # 워커 수 조정
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# KIS API fallback (pykrx 수급 API 장애 대비)
try:
    from src.adapters.kis_investor_adapter import fetch_investor_by_ticker as kis_fetch_investor
    KIS_INVESTOR_AVAILABLE = True
except ImportError:
    KIS_INVESTOR_AVAILABLE = False

# FinanceDataReader fallback (OHLCV)
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False


def _fill_supply_only(df: pd.DataFrame, parquet_path: Path,
                      end_date: str, ticker: str) -> dict:
    """OHLCV 이미 존재하는 날짜에 수급(기관/외인/개인)만 채워넣기.

    수급이 0인 최근 N일을 찾아서 pykrx로 수급만 재수집.
    """
    result = {"ticker": ticker, "status": "skip", "new_rows": 0}

    supply_cols = ["기관합계", "외국인합계", "개인"]
    for col in supply_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 수급이 0인 최근 행 찾기 (최대 최근 5일)
    recent = df.tail(5)
    zero_mask = (recent[supply_cols].abs().sum(axis=1) == 0)
    zero_dates = recent[zero_mask].index

    if len(zero_dates) == 0:
        return result  # 수급 이미 있음

    fetch_start = zero_dates.min().strftime("%Y%m%d")
    fetch_end = zero_dates.max().strftime("%Y%m%d")

    inv_df = None
    try:
        inv_df = krx.get_market_trading_value_by_date(fetch_start, fetch_end, ticker)
        if inv_df is not None and not inv_df.empty:
            inv_df.index.name = "date"
        else:
            inv_df = None
        time.sleep(0.3)
    except Exception:
        inv_df = None

    # pykrx 실패 → KIS fallback
    if inv_df is None and KIS_INVESTOR_AVAILABLE:
        try:
            kis_df = kis_fetch_investor(ticker)
            if kis_df is not None and not kis_df.empty:
                kis_df.index.name = "date"
                inv_df = kis_df
        except Exception:
            pass

    if inv_df is None:
        return result

    filled = 0
    inv_col_map = {"기관합계": "기관합계", "외국인합계": "외국인합계", "개인": "개인"}
    for kr_col, en_col in inv_col_map.items():
        if kr_col in inv_df.columns:
            common_idx = inv_df.index.intersection(zero_dates)
            if len(common_idx) > 0:
                df.loc[common_idx, en_col] = inv_df.loc[common_idx, kr_col]
                filled = len(common_idx)

    if filled > 0:
        df.to_parquet(parquet_path)
        result["status"] = "ok"
        result["new_rows"] = filled

    return result


def extend_single(parquet_path: Path, end_date: str, *,
                   skip_supply: bool = False, supply_only: bool = False) -> dict:
    """단일 parquet 파일 증분 업데이트

    Args:
        skip_supply: True → 수급(투자자 매매동향) 스킵, 종가/거래량만
        supply_only: True → OHLCV 이미 있는 날짜에 수급만 채워넣기
    """
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

    # ── supply_only 모드: 기존 OHLCV 행에 수급만 채워넣기 ──
    if supply_only:
        return _fill_supply_only(df, parquet_path, end_date, ticker)

    # 이미 최신이면 skip
    if last_date_str >= end_date:
        return result

    # 다음 거래일부터 조회
    fetch_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    try:
        # OHLCV 조회: pykrx → FDR fallback 체인
        new_ohlcv = None
        ohlcv_source = "pykrx"

        # 1차: pykrx
        try:
            new_ohlcv = krx.get_market_ohlcv_by_date(fetch_start, end_date, ticker, adjusted=True)
            if new_ohlcv is not None and not new_ohlcv.empty:
                new_ohlcv.index.name = "date"
                col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
                            "거래량": "volume", "등락률": "price_change", "거래대금": "trading_value"}
                new_ohlcv = new_ohlcv.rename(columns=col_map)
            else:
                new_ohlcv = None
        except Exception as e:
            logger.debug("[%s] pykrx OHLCV 실패: %s", ticker, e)
            new_ohlcv = None

        # 2차: FDR fallback
        if new_ohlcv is None and FDR_AVAILABLE:
            try:
                fdr_start = f"{fetch_start[:4]}-{fetch_start[4:6]}-{fetch_start[6:]}"
                fdr_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
                fdr_df = fdr.DataReader(ticker, fdr_start, fdr_end)
                if fdr_df is not None and not fdr_df.empty:
                    fdr_df.index = pd.to_datetime(fdr_df.index)
                    fdr_df.index.name = "date"
                    col_map = {"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume", "Change": "price_change"}
                    fdr_df = fdr_df.rename(columns=col_map)
                    new_ohlcv = fdr_df
                    ohlcv_source = "fdr"
                    logger.debug("[%s] FDR fallback OHLCV 성공", ticker)
            except Exception as e:
                logger.debug("[%s] FDR OHLCV 실패: %s", ticker, e)

        if new_ohlcv is None or new_ohlcv.empty:
            return result

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

        # 수급 컬럼이 기존 parquet에 없으면 미리 추가 (전체 유니버스 수급 수집)
        for supply_col in ["기관합계", "외국인합계", "개인"]:
            if supply_col not in df.columns:
                df[supply_col] = 0.0
            if supply_col not in new_rows.columns:
                new_rows[supply_col] = 0.0

        # 투자자 매매동향 업데이트 시도 (pykrx → KIS fallback)
        if not skip_supply:
            inv_df = None
            try:
                inv_df = krx.get_market_trading_value_by_date(fetch_start, end_date, ticker)
                if inv_df is not None and not inv_df.empty:
                    inv_df.index.name = "date"
                else:
                    inv_df = None
                time.sleep(0.3)
            except Exception:
                inv_df = None

            # pykrx 실패 → KIS API fallback
            if inv_df is None and KIS_INVESTOR_AVAILABLE:
                try:
                    kis_df = kis_fetch_investor(ticker)
                    if kis_df is not None and not kis_df.empty:
                        kis_df.index.name = "date"
                        inv_df = kis_df
                        logger.debug("[%s] KIS fallback 수급 조회 성공", ticker)
                except Exception as e:
                    logger.debug("[%s] KIS fallback 실패: %s", ticker, e)

            if inv_df is not None:
                inv_col_map = {"기관합계": "기관합계", "외국인합계": "외국인합계", "개인": "개인"}
                for kr_col, en_col in inv_col_map.items():
                    if kr_col in inv_df.columns and en_col in new_rows.columns:
                        common_idx = inv_df.index.intersection(new_rows.index)
                        if len(common_idx) > 0:
                            new_rows.loc[common_idx, en_col] = inv_df.loc[common_idx, kr_col]

        # 펀더멘탈 업데이트 시도
        if not skip_supply:
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
        if ohlcv_source != "pykrx":
            result["ohlcv_source"] = ohlcv_source

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def _worker_fn(parquet_path: Path, end_date: str, skip_supply: bool, supply_only: bool) -> dict:
    """워커 스레드에서 실행되는 단일 종목 처리 함수."""
    try:
        result = extend_single(parquet_path, end_date,
                               skip_supply=skip_supply,
                               supply_only=supply_only)
        # per-thread rate limit (pykrx 보호)
        if result["status"] == "ok":
            time.sleep(0.3)
        return result
    except Exception as e:
        return {"ticker": parquet_path.stem, "status": "error", "error": str(e), "new_rows": 0}


def _save_pipeline_errors(error_list: list, script_name: str = "extend_parquet_data"):
    """에러 로그를 data/pipeline_errors.json에 저장."""
    out_path = project_root / "data" / "pipeline_errors.json"
    entry = {
        "script": script_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "errors": error_list,
    }
    # 기존 파일이 있으면 append (최근 7일치만 유지)
    history = []
    if out_path.exists():
        try:
            history = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    history.append(entry)
    # 최근 50건만 유지
    history = history[-50:]
    out_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="parquet 데이터 증분 업데이트")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYYMMDD)")
    parser.add_argument("--skip-supply", action="store_true",
                        help="수급(투자자 매매동향) 수집 스킵 — 종가/거래량만 수집")
    parser.add_argument("--supply-only", action="store_true",
                        help="수급만 재수집 (OHLCV는 이미 있는 종목만)")
    parser.add_argument("--sample", type=int, default=0,
                        help="테스트용: N종목만 샘플 실행")
    parser.add_argument("--workers", type=int, default=5,
                        help="병렬 워커 수 (기본 5, 최대 5)")
    args = parser.parse_args()

    if not PYKRX_AVAILABLE:
        logger.error("pykrx 미설치. 종료.")
        return

    max_workers = min(args.workers, 5)  # 5 이상 올리지 않음

    end_date = args.end or datetime.today().strftime("%Y%m%d")
    raw_dir = project_root / "data" / "raw"
    parquets = sorted(raw_dir.glob("*.parquet"))

    if args.sample > 0:
        parquets = parquets[:args.sample]
        logger.info(f"샘플 모드: {args.sample}종목만 실행")

    mode = "supply-only" if args.supply_only else ("ohlcv-only" if args.skip_supply else "full")
    logger.info(f"증분 업데이트 시작: {len(parquets)}종목 → {end_date}까지 (모드: {mode}, workers: {max_workers})")

    t_start = time.time()
    updated = 0
    skipped = 0
    errors = 0
    error_list = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_fn, p, end_date, args.skip_supply, args.supply_only): p
            for p in parquets
        }

        for future in as_completed(futures):
            done_count += 1
            result = future.result()

            if result["status"] == "ok":
                updated += 1
                logger.info(f"  [{done_count}/{len(parquets)}] {result['ticker']}: +{result['new_rows']}행 → {result.get('new_end', '')}")
            elif result["status"] == "error":
                errors += 1
                error_list.append({"code": result["ticker"], "error": result.get("error", "")})
                logger.warning(f"  [{done_count}/{len(parquets)}] {result['ticker']}: 오류 - {result.get('error', '')}")
            else:
                skipped += 1

            if done_count % 50 == 0:
                elapsed = time.time() - t_start
                logger.info(f"  --- 진행: {done_count}/{len(parquets)} | 업데이트: {updated} | 스킵: {skipped} | 에러: {errors} | {elapsed:.0f}초")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*50}")
    logger.info(f"증분 업데이트 완료 (→ {end_date})")
    logger.info(f"  업데이트: {updated}종목")
    logger.info(f"  스킵(최신): {skipped}종목")
    logger.info(f"  오류: {errors}종목")
    logger.info(f"  소요시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    if error_list:
        for e in error_list[:10]:
            logger.info(f"    - {e['code']}: {e['error']}")
    logger.info(f"{'='*50}")

    # 에러 기록 저장
    _save_pipeline_errors(error_list)

    # 에러율 5% 이상이면 텔레그램 알림
    total = len(parquets)
    if total > 0 and len(error_list) > 0:
        error_rate = len(error_list) / total * 100
        if error_rate >= 5:
            try:
                from src.telegram_sender import send_message
                top_errors = ", ".join(e["code"] for e in error_list[:5])
                msg = (
                    f"⚠️ 데이터 파이프라인 경고\n"
                    f"스크립트: extend_parquet_data\n"
                    f"전체: {total}종목 | 실패: {len(error_list)}건 ({error_rate:.1f}%)\n"
                    f"주요 실패: {top_errors}\n"
                    f"전체 로그: data/pipeline_errors.json"
                )
                send_message(msg)
                logger.info("텔레그램 에러 알림 발송 완료")
            except Exception as e:
                logger.warning(f"텔레그램 알림 실패: {e}")

    # 검증
    if updated > 0:
        sample_p = parquets[0]
        df = pd.read_parquet(sample_p)
        logger.info(f"\n검증: {sample_p.stem} → {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}rows)")


if __name__ == "__main__":
    main()
