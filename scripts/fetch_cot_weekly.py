"""CFTC COT (Commitments of Traders) — 주간 데이터 수집 + 백필

스마트 머니 포지셔닝 추적:
  - S&P 500 E-mini: Asset Manager 순포지션 (TFF 보고서)
  - Gold COMEX: Managed Money 순포지션 (Disaggregated 보고서)
  - 10Y T-Note: Asset Manager 순포지션 (TFF 보고서)
  - WTI Crude Oil: Managed Money 순포지션 (Disaggregated 보고서)

데이터 소스: CFTC 공개 보고서 (API 키 불필요)
  - TFF: https://www.cftc.gov/files/dea/history/fut_fin_txt_{YYYY}.zip
  - Disagg: https://www.cftc.gov/files/dea/history/fut_disagg_txt_{YYYY}.zip

저장: data/cot/cot_weekly.parquet

사용법:
    python scripts/fetch_cot_weekly.py              # 최신 주간만 업데이트
    python scripts/fetch_cot_weekly.py --backfill 3  # 3년 백필
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── 경로 설정 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COT_DIR = PROJECT_ROOT / "data" / "cot"
PARQUET_PATH = COT_DIR / "cot_weekly.parquet"

logger = logging.getLogger(__name__)

# ── CFTC URL 패턴 ──
CFTC_HIST_BASE = "https://www.cftc.gov/files/dea/history"
CFTC_CURRENT_BASE = "https://www.cftc.gov/dea/newcot"

REPORT_URLS = {
    "fin": {
        "history": f"{CFTC_HIST_BASE}/fut_fin_txt_{{year}}.zip",
        "current": f"{CFTC_CURRENT_BASE}/FinFutWk.txt",
    },
    "disagg": {
        "history": f"{CFTC_HIST_BASE}/fut_disagg_txt_{{year}}.zip",
        "current": f"{CFTC_CURRENT_BASE}/f_disagg.txt",
    },
}

# ── 4개 계약 정의 ──
CONTRACTS = {
    "sp500": {
        "report": "fin",
        "code": "13874A",
        "name_contains": "E-MINI S&P 500",
        "long_col": "Asset_Mgr_Positions_Long_All",
        "short_col": "Asset_Mgr_Positions_Short_All",
        "oi_col": "Open_Interest_All",
        "label": "S&P 500 E-mini",
    },
    "treasury10y": {
        "report": "fin",
        "code": "043602",
        "name_contains": "10-YEAR",
        "long_col": "Asset_Mgr_Positions_Long_All",
        "short_col": "Asset_Mgr_Positions_Short_All",
        "oi_col": "Open_Interest_All",
        "label": "10Y T-Note",
    },
    "gold": {
        "report": "disagg",
        "code": "088691",
        "name_contains": "GOLD",
        "long_col": "M_Money_Positions_Long_All",
        "short_col": "M_Money_Positions_Short_All",
        "oi_col": "Open_Interest_All",
        "label": "Gold COMEX",
    },
    "crude_oil": {
        "report": "disagg",
        "code": "067651",
        "name_contains": "CRUDE OIL, LIGHT SWEET",
        "long_col": "M_Money_Positions_Long_All",
        "short_col": "M_Money_Positions_Short_All",
        "oi_col": "Open_Interest_All",
        "label": "WTI Crude Oil",
    },
}


# ================================================================
# 다운로드 유틸리티
# ================================================================

def _download_and_extract(url: str, max_retries: int = 3) -> pd.DataFrame:
    """URL에서 ZIP 다운로드 → CSV 파싱. TXT URL이면 직접 파싱."""
    for attempt in range(max_retries):
        try:
            logger.info("다운로드: %s (시도 %d/%d)", url, attempt + 1, max_retries)
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    # ZIP 안의 첫 번째 파일
                    names = zf.namelist()
                    if not names:
                        logger.warning("ZIP 파일 비어있음: %s", url)
                        return pd.DataFrame()
                    with zf.open(names[0]) as f:
                        df = pd.read_csv(f, low_memory=False)
            else:
                # TXT 직접 파싱
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

            # 컬럼명 공백 제거
            df.columns = [c.strip() for c in df.columns]
            logger.info("  → %d행 로드", len(df))
            return df

        except zipfile.BadZipFile:
            logger.warning("ZIP 파일 손상: %s", url)
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logger.warning("다운로드 실패 (시도 %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.info("  %d초 후 재시도...", wait)
                time.sleep(wait)

    logger.error("다운로드 최종 실패: %s", url)
    return pd.DataFrame()


def _filter_contract(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """CFTC_Contract_Market_Code 기준 필터. fallback으로 name_contains."""
    if df.empty:
        return df

    # 1차: 코드 기준
    code_col = "CFTC_Contract_Market_Code"
    if code_col in df.columns:
        filtered = df[df[code_col].astype(str).str.strip() == cfg["code"]]
        if not filtered.empty:
            return filtered

    # 2차: 이름 기준
    name_col = "Market_and_Exchange_Names"
    if name_col in df.columns:
        pattern = cfg["name_contains"]
        filtered = df[df[name_col].str.contains(pattern, case=False, na=False)]
        if not filtered.empty:
            logger.info("  코드 미발견, 이름('%s')으로 필터: %d행", pattern, len(filtered))
            return filtered

    logger.warning("  계약 미발견: code=%s, name=%s", cfg["code"], cfg["name_contains"])
    return pd.DataFrame()


def _extract_positions(df: pd.DataFrame, cfg: dict, contract_name: str) -> pd.DataFrame:
    """필터된 DataFrame에서 날짜 + 포지션 추출."""
    if df.empty:
        return pd.DataFrame()

    # 날짜 컬럼 찾기
    date_col = None
    for candidate in ["Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYMMDD"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        logger.warning("날짜 컬럼 미발견: %s", contract_name)
        return pd.DataFrame()

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df[date_col].astype(str).str.strip())

    long_col = cfg["long_col"]
    short_col = cfg["short_col"]
    oi_col = cfg.get("oi_col", "Open_Interest_All")

    result[f"{contract_name}_long"] = pd.to_numeric(df[long_col], errors="coerce")
    result[f"{contract_name}_short"] = pd.to_numeric(df[short_col], errors="coerce")
    result[f"{contract_name}_net"] = result[f"{contract_name}_long"] - result[f"{contract_name}_short"]

    if oi_col in df.columns:
        result[f"{contract_name}_oi"] = pd.to_numeric(df[oi_col], errors="coerce")
    else:
        result[f"{contract_name}_oi"] = 0

    result = result.set_index("date").sort_index()
    return result


# ================================================================
# 백필 / 업데이트
# ================================================================

def backfill(years: int = 3) -> pd.DataFrame:
    """연도별 CFTC ZIP 다운로드 → 4개 계약 추출 → parquet 저장."""
    current_year = datetime.now().year
    year_range = range(current_year - years, current_year + 1)

    # 보고서 타입별 연도별 DataFrame 수집
    report_dfs: dict[str, list[pd.DataFrame]] = {"fin": [], "disagg": []}

    for year in year_range:
        for rtype in ("fin", "disagg"):
            url = REPORT_URLS[rtype]["history"].format(year=year)
            df = _download_and_extract(url)
            if not df.empty:
                report_dfs[rtype].append(df)
            time.sleep(1)  # CFTC rate limit 방지

    # 보고서 타입별 병합
    merged_reports: dict[str, pd.DataFrame] = {}
    for rtype, dfs in report_dfs.items():
        if dfs:
            merged_reports[rtype] = pd.concat(dfs, ignore_index=True)
            logger.info("%s 보고서: 총 %d행", rtype, len(merged_reports[rtype]))
        else:
            merged_reports[rtype] = pd.DataFrame()

    # 4개 계약별 포지션 추출
    all_positions: list[pd.DataFrame] = []
    for name, cfg in CONTRACTS.items():
        rtype = cfg["report"]
        report_df = merged_reports.get(rtype, pd.DataFrame())
        filtered = _filter_contract(report_df, cfg)
        positions = _extract_positions(filtered, cfg, name)
        if not positions.empty:
            all_positions.append(positions)
            logger.info("  %s: %d주", cfg["label"], len(positions))
        else:
            logger.warning("  %s: 데이터 없음!", cfg["label"])

    if not all_positions:
        logger.error("모든 계약 데이터 수집 실패!")
        return pd.DataFrame()

    # 날짜 기준 병합 (outer join → NaN은 forward fill)
    result = all_positions[0]
    for df in all_positions[1:]:
        result = result.join(df, how="outer")

    result = result.sort_index()
    # 중복 날짜 제거 (최신 우선)
    result = result[~result.index.duplicated(keep="last")]

    # 저장
    COT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PARQUET_PATH)
    logger.info("저장: %s (%d행 × %d열)", PARQUET_PATH, len(result), len(result.columns))

    return result


def update() -> pd.DataFrame:
    """최신 주간 데이터로 증분 업데이트."""
    # 기존 parquet 로드
    existing = pd.DataFrame()
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        logger.info("기존 데이터: %d행 (최신: %s)", len(existing), existing.index.max())

    # 현재 연도 전체 다운로드 (current URL은 현재 주만이라 history URL 사용)
    current_year = datetime.now().year
    new_positions: list[pd.DataFrame] = []

    for rtype in ("fin", "disagg"):
        url = REPORT_URLS[rtype]["history"].format(year=current_year)
        df = _download_and_extract(url)
        if df.empty:
            # fallback: current URL
            url = REPORT_URLS[rtype]["current"]
            df = _download_and_extract(url)

        if not df.empty:
            for name, cfg in CONTRACTS.items():
                if cfg["report"] != rtype:
                    continue
                filtered = _filter_contract(df, cfg)
                positions = _extract_positions(filtered, cfg, name)
                if not positions.empty:
                    new_positions.append(positions)

        time.sleep(1)

    if not new_positions:
        logger.warning("새 데이터 없음. 기존 유지.")
        return existing

    # 새 데이터 병합
    new_data = new_positions[0]
    for df in new_positions[1:]:
        new_data = new_data.join(df, how="outer")

    if existing.empty:
        result = new_data
    else:
        result = pd.concat([existing, new_data])

    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]

    # 저장
    COT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PARQUET_PATH)
    logger.info("업데이트 완료: %d행 (최신: %s)", len(result), result.index.max())

    return result


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CFTC COT 데이터 수집")
    parser.add_argument("--backfill", type=int, default=0,
                        help="백필 년수 (0=증분만)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print("  CFTC COT — 주간 스마트머니 데이터 수집")
    print("=" * 60)

    if args.backfill > 0:
        print(f"\n{args.backfill}년 백필 시작...")
        df = backfill(args.backfill)
    else:
        print("\n증분 업데이트...")
        df = update()

    if not df.empty:
        print(f"\n결과: {len(df)}주 × {len(df.columns)}열")
        print(f"기간: {df.index.min().date()} ~ {df.index.max().date()}")
        # 최신 주 요약
        latest = df.iloc[-1]
        print("\n최신 주 포지션:")
        for name, cfg in CONTRACTS.items():
            net_col = f"{name}_net"
            if net_col in df.columns:
                net = latest.get(net_col, 0)
                print(f"  {cfg['label']:>16s}: net = {net:>+12,.0f}")
    else:
        print("\n데이터 수집 실패!")


if __name__ == "__main__":
    main()
