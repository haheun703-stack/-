"""기존 parquet 데이터를 최신 날짜까지 증분 업데이트

KRX 전종목 일괄 API (OHLCV/PER) + KIS 수급 API 기반.
Phase 1 최적화: 종목별 개별 API 호출 → 전종목 일괄 조회 전환.
기존 13분 → 2분 이내 완료 (1071종목 기준).

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
import requests

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

# KIS API (수급 primary)
try:
    from src.adapters.kis_investor_adapter import fetch_investor_by_ticker as kis_fetch_investor
    KIS_INVESTOR_AVAILABLE = True
except ImportError:
    KIS_INVESTOR_AVAILABLE = False

# FinanceDataReader (OHLCV fallback)
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False


# ════════════════════════════════════════════════════
# KRX 전종목 일괄 조회
# ════════════════════════════════════════════════════

class KRXBulkFetcher:
    """KRX 전종목 OHLCV/PER/PBR 일괄 조회 (공개 STAT API).

    1회 호출로 KOSPI 또는 KOSDAQ 전종목 데이터 취득.
    기존 pykrx 종목별 개별 호출(1071회) -> 일괄 2~4회로 대체.
    """

    BASE_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ko-KR,ko;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd",
        })
        # 세션 쿠키 획득
        try:
            self.session.get(
                "https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd",
                timeout=10,
            )
        except Exception:
            pass
        self._logged_keys = False

    @staticmethod
    def _num(val) -> float:
        """KRX 숫자 문자열 -> float (쉼표 제거)."""
        if not val or val == "-" or val == "":
            return 0.0
        return float(str(val).replace(",", "").strip())

    def fetch_all_ohlcv(self, date: str) -> dict[str, dict]:
        """전종목 OHLCV 일괄 조회 (KOSPI + KOSDAQ, 총 2회 호출).

        Returns:
            {ticker: {"open": .., "high": .., "low": .., "close": ..,
                      "volume": .., "trading_value": .., "price_change": ..}}
        """
        result = {}
        for mkt in ["STK", "KSQ"]:
            try:
                r = self.session.post(
                    f"{self.BASE_URL}?bld=dbms/MDC/STAT/standard/MDCSTAT01501",
                    data={
                        "locale": "ko_KR",
                        "mktId": mkt,
                        "trdDd": date,
                        "share": "1",
                        "money": "1",
                        "csvxls_isNo": "false",
                    },
                    timeout=30,
                )
                data = r.json()
                rows = data.get("OutBlock_1", data.get("output", []))
                if not rows:
                    logger.warning(
                        "KRX OHLCV %s %s: 빈 응답 (keys=%s)",
                        mkt, date, list(data.keys())[:5],
                    )
                    continue

                # 첫 호출 시 응답 키 로그 (디버깅)
                if not self._logged_keys and rows:
                    logger.info("KRX OHLCV 응답 키: %s", list(rows[0].keys()))
                    self._logged_keys = True

                for row in rows:
                    ticker = row.get("ISU_SRT_CD", "")
                    if not (len(ticker) == 6 and ticker.isdigit()):
                        continue
                    result[ticker] = {
                        "open": self._num(row.get("TDD_OPNPRC")),
                        "high": self._num(row.get("TDD_HGPRC")),
                        "low": self._num(row.get("TDD_LWPRC")),
                        "close": self._num(row.get("TDD_CLSPRC")),
                        "volume": self._num(row.get("ACC_TRDVOL")),
                        "trading_value": self._num(row.get("ACC_TRDVAL")),
                        "price_change": self._num(row.get("FLUC_RT")),
                    }
                time.sleep(0.5)
            except Exception as e:
                logger.warning("KRX OHLCV %s %s 실패: %s", mkt, date, e)
        if result:
            logger.info("KRX OHLCV %s: %d종목 일괄 조회", date, len(result))
        return result

    def fetch_all_fundamental(self, date: str) -> dict[str, dict]:
        """전종목 PER/PBR/배당수익률 일괄 조회.

        Returns:
            {ticker: {"fund_BPS": .., "fund_PER": .., "fund_PBR": ..,
                      "fund_EPS": .., "fund_DIV": .., "fund_DPS": ..}}
        """
        result = {}
        for mkt in ["STK", "KSQ"]:
            try:
                r = self.session.post(
                    f"{self.BASE_URL}?bld=dbms/MDC/STAT/standard/MDCSTAT03501",
                    data={
                        "locale": "ko_KR",
                        "mktId": mkt,
                        "trdDd": date,
                        "share": "1",
                        "money": "1",
                        "csvxls_isNo": "false",
                    },
                    timeout=30,
                )
                data = r.json()
                rows = data.get("output", data.get("OutBlock_1", []))
                if not rows:
                    logger.warning(
                        "KRX 기본지표 %s %s: 빈 응답 (keys=%s)",
                        mkt, date, list(data.keys())[:5],
                    )
                    continue

                for row in rows:
                    ticker = row.get("ISU_SRT_CD", "")
                    if not (len(ticker) == 6 and ticker.isdigit()):
                        continue
                    result[ticker] = {
                        "fund_BPS": self._num(row.get("BPS")),
                        "fund_PER": self._num(row.get("PER")),
                        "fund_PBR": self._num(row.get("PBR")),
                        "fund_EPS": self._num(row.get("EPS")),
                        "fund_DIV": self._num(row.get("DIV")),
                        "fund_DPS": self._num(row.get("DPS")),
                    }
                time.sleep(0.5)
            except Exception as e:
                logger.warning("KRX 기본지표 %s %s 실패: %s", mkt, date, e)
        if result:
            logger.info("KRX 기본지표 %s: %d종목 일괄 조회", date, len(result))
        return result


# ════════════════════════════════════════════════════
# 단일 종목 처리
# ════════════════════════════════════════════════════

def _fill_supply_only(df: pd.DataFrame, parquet_path: Path,
                      end_date: str, ticker: str) -> dict:
    """OHLCV 이미 존재하는 날짜에 수급(기관/외인/개인)만 채워넣기."""
    result = {"ticker": ticker, "status": "skip", "new_rows": 0}

    supply_cols = ["기관합계", "외국인합계", "개인", "기타법인"]
    for col in supply_cols:
        if col not in df.columns:
            df[col] = 0.0

    recent = df.tail(5)
    zero_mask = (recent[supply_cols].abs().sum(axis=1) == 0)
    zero_dates = recent[zero_mask].index

    if len(zero_dates) == 0:
        return result

    inv_df = None

    # 1차: KIS API (안정적, 30일치 반환)
    if KIS_INVESTOR_AVAILABLE:
        try:
            kis_df = kis_fetch_investor(ticker)
            if kis_df is not None and not kis_df.empty:
                kis_df.index.name = "date"
                inv_df = kis_df
        except Exception:
            pass

    # 2차: pykrx fallback
    if inv_df is None and PYKRX_AVAILABLE:
        fetch_start = zero_dates.min().strftime("%Y%m%d")
        fetch_end = zero_dates.max().strftime("%Y%m%d")
        try:
            inv_df = krx.get_market_trading_value_by_date(fetch_start, fetch_end, ticker)
            if inv_df is not None and not inv_df.empty:
                inv_df.index.name = "date"
            else:
                inv_df = None
            time.sleep(0.3)
        except Exception:
            inv_df = None

    if inv_df is None:
        return result

    filled = 0
    for col in supply_cols:
        if col in inv_df.columns:
            if col not in df.columns:
                df[col] = 0.0
            common_idx = inv_df.index.intersection(zero_dates)
            if len(common_idx) > 0:
                df.loc[common_idx, col] = inv_df.loc[common_idx, col]
                filled = len(common_idx)

    if filled > 0:
        df.to_parquet(parquet_path)
        result["status"] = "ok"
        result["new_rows"] = filled

    return result


def extend_single(parquet_path: Path, end_date: str, *,
                   skip_supply: bool = False, supply_only: bool = False,
                   ohlcv_cache: dict | None = None,
                   fund_cache: dict | None = None) -> dict:
    """단일 parquet 파일 증분 업데이트.

    Args:
        ohlcv_cache: {date_str: {ticker: {open,high,low,...}}} KRX 일괄 캐시
        fund_cache: {date_str: {ticker: {fund_BPS,...}}} KRX 일괄 캐시
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

    if supply_only:
        return _fill_supply_only(df, parquet_path, end_date, ticker)

    if last_date_str >= end_date:
        return result

    fetch_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    try:
        # ── OHLCV: KRX bulk cache -> pykrx -> FDR ──
        new_ohlcv = None
        ohlcv_source = "krx_bulk"

        # 1차: KRX 일괄 캐시
        if ohlcv_cache:
            cached_rows = []
            for ds, tickers_data in ohlcv_cache.items():
                if fetch_start <= ds <= end_date and ticker in tickers_data:
                    cached_rows.append({"date": pd.Timestamp(ds), **tickers_data[ticker]})
            if cached_rows:
                new_ohlcv = pd.DataFrame(cached_rows).set_index("date")
                new_ohlcv.index.name = "date"

        # 2차: pykrx fallback
        if new_ohlcv is None and PYKRX_AVAILABLE:
            ohlcv_source = "pykrx"
            try:
                new_ohlcv = krx.get_market_ohlcv_by_date(
                    fetch_start, end_date, ticker, adjusted=True,
                )
                if new_ohlcv is not None and not new_ohlcv.empty:
                    new_ohlcv.index.name = "date"
                    col_map = {
                        "시가": "open", "고가": "high", "저가": "low",
                        "종가": "close", "거래량": "volume",
                        "등락률": "price_change", "거래대금": "trading_value",
                    }
                    new_ohlcv = new_ohlcv.rename(columns=col_map)
                else:
                    new_ohlcv = None
            except Exception as e:
                logger.debug("[%s] pykrx OHLCV 실패: %s", ticker, e)
                new_ohlcv = None

        # 3차: FDR fallback
        if new_ohlcv is None and FDR_AVAILABLE:
            ohlcv_source = "fdr"
            try:
                fdr_start = f"{fetch_start[:4]}-{fetch_start[4:6]}-{fetch_start[6:]}"
                fdr_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
                fdr_df = fdr.DataReader(ticker, fdr_start, fdr_end)
                if fdr_df is not None and not fdr_df.empty:
                    fdr_df.index = pd.to_datetime(fdr_df.index)
                    fdr_df.index.name = "date"
                    col_map = {
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                        "Change": "price_change",
                    }
                    fdr_df = fdr_df.rename(columns=col_map)
                    new_ohlcv = fdr_df
            except Exception as e:
                logger.debug("[%s] FDR OHLCV 실패: %s", ticker, e)

        if new_ohlcv is None or new_ohlcv.empty:
            return result

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
                new_rows[col] = 0.0

        for supply_col in ["기관합계", "외국인합계", "개인", "기타법인"]:
            if supply_col not in df.columns:
                df[supply_col] = 0.0
            if supply_col not in new_rows.columns:
                new_rows[supply_col] = 0.0

        # ── 수급: KIS primary -> pykrx fallback ──
        if not skip_supply:
            inv_df = None

            # 1차: KIS API (안정적, rate limit 자체 관리)
            if KIS_INVESTOR_AVAILABLE:
                try:
                    kis_df = kis_fetch_investor(ticker)
                    if kis_df is not None and not kis_df.empty:
                        kis_df.index.name = "date"
                        inv_df = kis_df
                except Exception as e:
                    logger.debug("[%s] KIS 수급 실패: %s", ticker, e)

            # 2차: pykrx fallback
            if inv_df is None and PYKRX_AVAILABLE:
                try:
                    inv_df = krx.get_market_trading_value_by_date(
                        fetch_start, end_date, ticker,
                    )
                    if inv_df is not None and not inv_df.empty:
                        inv_df.index.name = "date"
                    else:
                        inv_df = None
                    time.sleep(0.3)
                except Exception:
                    inv_df = None

            if inv_df is not None:
                for col in ["기관합계", "외국인합계", "개인", "기타법인"]:
                    if col in inv_df.columns:
                        if col not in new_rows.columns:
                            new_rows[col] = 0.0
                        common_idx = inv_df.index.intersection(new_rows.index)
                        if len(common_idx) > 0:
                            new_rows.loc[common_idx, col] = inv_df.loc[common_idx, col]

        # ── PER/PBR: KRX bulk cache -> pykrx fallback ──
        if not skip_supply:
            fund_applied = False

            # 1차: KRX 일괄 캐시
            if fund_cache:
                for ds, tickers_data in fund_cache.items():
                    if fetch_start <= ds <= end_date and ticker in tickers_data:
                        fund_data = tickers_data[ticker]
                        ts = pd.Timestamp(ds)
                        if ts in new_rows.index:
                            for col, val in fund_data.items():
                                if col in new_rows.columns:
                                    new_rows.loc[ts, col] = val
                            fund_applied = True

            # 2차: pykrx fallback
            if not fund_applied and PYKRX_AVAILABLE:
                try:
                    fund_df = krx.get_market_fundamental_by_date(
                        fetch_start, end_date, ticker,
                    )
                    if fund_df is not None and not fund_df.empty:
                        fund_df.index.name = "date"
                        fund_col_map = {
                            "BPS": "fund_BPS", "PER": "fund_PER",
                            "PBR": "fund_PBR", "EPS": "fund_EPS",
                            "DIV": "fund_DIV", "DPS": "fund_DPS",
                        }
                        for kr_col, en_col in fund_col_map.items():
                            if kr_col in fund_df.columns and en_col in new_rows.columns:
                                common_idx = fund_df.index.intersection(new_rows.index)
                                if len(common_idx) > 0:
                                    new_rows.loc[common_idx, en_col] = fund_df.loc[common_idx, kr_col]
                    time.sleep(0.3)
                except Exception:
                    pass

        # 합치기
        combined = pd.concat([df, new_rows])
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(parquet_path)

        result["status"] = "ok"
        result["new_rows"] = len(new_rows)
        result["new_end"] = combined.index.max().strftime("%Y-%m-%d")
        if ohlcv_source != "krx_bulk":
            result["ohlcv_source"] = ohlcv_source

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def _worker_fn(parquet_path: Path, end_date: str, skip_supply: bool,
               supply_only: bool, ohlcv_cache: dict, fund_cache: dict) -> dict:
    """워커 스레드 함수."""
    try:
        return extend_single(
            parquet_path, end_date,
            skip_supply=skip_supply, supply_only=supply_only,
            ohlcv_cache=ohlcv_cache, fund_cache=fund_cache,
        )
    except Exception as e:
        return {"ticker": parquet_path.stem, "status": "error",
                "error": str(e), "new_rows": 0}


def _save_pipeline_errors(error_list: list, script_name: str = "extend_parquet_data"):
    """에러 로그를 data/pipeline_errors.json에 저장."""
    out_path = project_root / "data" / "pipeline_errors.json"
    entry = {
        "script": script_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "errors": error_list,
    }
    history = []
    if out_path.exists():
        try:
            history = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    history.append(entry)
    history = history[-50:]
    out_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="parquet 데이터 증분 업데이트")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYYMMDD)")
    parser.add_argument("--skip-supply", action="store_true",
                        help="수급(투자자 매매동향) 수집 스킵 -- 종가/거래량만 수집")
    parser.add_argument("--supply-only", action="store_true",
                        help="수급만 재수집 (OHLCV는 이미 있는 종목만)")
    parser.add_argument("--sample", type=int, default=0,
                        help="테스트용: N종목만 샘플 실행")
    parser.add_argument("--workers", type=int, default=5,
                        help="병렬 워커 수 (기본 5, 최대 5)")
    args = parser.parse_args()

    max_workers = min(args.workers, 5)
    end_date = args.end or datetime.today().strftime("%Y%m%d")
    raw_dir = project_root / "data" / "raw"
    parquets = sorted(raw_dir.glob("*.parquet"))

    if args.sample > 0:
        parquets = parquets[:args.sample]
        logger.info("샘플 모드: %d종목만 실행", args.sample)

    mode = "supply-only" if args.supply_only else ("ohlcv-only" if args.skip_supply else "full")
    logger.info(
        "증분 업데이트 시작: %d종목 -> %s까지 (모드: %s, workers: %d)",
        len(parquets), end_date, mode, max_workers,
    )

    t_start = time.time()

    # ── KRX 전종목 일괄 조회 (OHLCV + PER/PBR) ──
    ohlcv_cache: dict = {}
    fund_cache: dict = {}

    if not args.supply_only:
        try:
            fetcher = KRXBulkFetcher()
            # 최근 3영업일 일괄 조회 (누락 대비, 주말 자동 스킵)
            today = datetime.strptime(end_date, "%Y%m%d")
            fetch_dates = []
            d = today
            while len(fetch_dates) < 3:
                if d.weekday() < 5:
                    fetch_dates.append(d.strftime("%Y%m%d"))
                d -= timedelta(days=1)

            logger.info("KRX 일괄 조회 시작: %s", sorted(fetch_dates))
            for ds in sorted(fetch_dates):
                ohlcv = fetcher.fetch_all_ohlcv(ds)
                if ohlcv:
                    ohlcv_cache[ds] = ohlcv
                if not args.skip_supply:
                    fund = fetcher.fetch_all_fundamental(ds)
                    if fund:
                        fund_cache[ds] = fund
                time.sleep(0.5)

            bulk_elapsed = time.time() - t_start
            logger.info(
                "KRX 일괄 조회 완료: OHLCV %d일 / 기본지표 %d일 (%.1f초)",
                len(ohlcv_cache), len(fund_cache), bulk_elapsed,
            )
        except Exception as e:
            logger.warning("KRX 일괄 조회 실패 (pykrx/FDR fallback 사용): %s", e)

    # ── 종목별 증분 업데이트 ──
    updated = 0
    skipped = 0
    errors = 0
    error_list = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _worker_fn, p, end_date, args.skip_supply, args.supply_only,
                ohlcv_cache, fund_cache,
            ): p
            for p in parquets
        }

        for future in as_completed(futures):
            done_count += 1
            result = future.result()

            if result["status"] == "ok":
                updated += 1
                src_tag = ""
                if result.get("ohlcv_source"):
                    src_tag = f" [{result['ohlcv_source']}]"
                logger.info(
                    "  [%d/%d] %s: +%d행 -> %s%s",
                    done_count, len(parquets), result["ticker"],
                    result["new_rows"], result.get("new_end", ""), src_tag,
                )
            elif result["status"] == "error":
                errors += 1
                error_list.append({
                    "code": result["ticker"],
                    "error": result.get("error", ""),
                })
                logger.warning(
                    "  [%d/%d] %s: 오류 - %s",
                    done_count, len(parquets), result["ticker"],
                    result.get("error", ""),
                )
            else:
                skipped += 1

            if done_count % 100 == 0:
                elapsed = time.time() - t_start
                logger.info(
                    "  --- 진행: %d/%d | 업데이트: %d | 스킵: %d | 에러: %d | %.0f초",
                    done_count, len(parquets), updated, skipped, errors, elapsed,
                )

    elapsed = time.time() - t_start
    logger.info("\n%s", "=" * 50)
    logger.info("증분 업데이트 완료 (-> %s)", end_date)
    logger.info("  업데이트: %d종목", updated)
    logger.info("  스킵(최신): %d종목", skipped)
    logger.info("  오류: %d종목", errors)
    logger.info("  소요시간: %.1f초 (%.1f분)", elapsed, elapsed / 60)
    if error_list:
        for e in error_list[:10]:
            logger.info("    - %s: %s", e["code"], e["error"])
    logger.info("=" * 50)

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
                logger.warning("텔레그램 알림 실패: %s", e)

    # 검증
    if updated > 0:
        sample_p = parquets[0]
        df = pd.read_parquet(sample_p)
        logger.info(
            "\n검증: %s -> %s ~ %s (%drows)",
            sample_p.stem, df.index.min().date(), df.index.max().date(), len(df),
        )


if __name__ == "__main__":
    main()
