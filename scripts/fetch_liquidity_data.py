"""FRED API 유동성 사이클 데이터 수집.

5대 지표:
  RRPONTSYD  — Reverse Repo (일간)
  WTREGEN    — Treasury General Account (주간)
  WM2NS      — M2 통화공급 (주간)
  WALCL      — Fed 밸런스시트 (주간)
  TOTRESNS   — 은행 총지준 (격주)

Usage:
    python -u -X utf8 scripts/fetch_liquidity_data.py              # 업데이트
    python -u -X utf8 scripts/fetch_liquidity_data.py --backfill 3 # 3년 백필
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "liquidity_cycle"
PARQUET_PATH = DATA_DIR / "liquidity_daily.parquet"

# FRED 지표 정의
INDICATORS = {
    "rrp": {
        "code": "RRPONTSYD",
        "label": "Reverse Repo",
        "freq": "daily",
        "unit_div": 1.0,  # 이미 십억 단위
    },
    "tga": {
        "code": "WTREGEN",
        "label": "Treasury General Account",
        "freq": "weekly",
        "unit_div": 1000.0,  # 백만 → 십억
    },
    "m2": {
        "code": "WM2NS",
        "label": "M2 Money Supply",
        "freq": "weekly",
        "unit_div": 1.0,  # 이미 십억 단위
    },
    "walcl": {
        "code": "WALCL",
        "label": "Fed Balance Sheet",
        "freq": "weekly",
        "unit_div": 1000.0,  # 백만 → 십억
    },
    "totresns": {
        "code": "TOTRESNS",
        "label": "Total Reserves",
        "freq": "biweekly",
        "unit_div": 1000.0,  # 백만 → 십억
    },
}


def _get_fred():
    """fredapi.Fred 인스턴스 생성."""
    from fredapi import Fred

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY가 .env에 없습니다")
    return Fred(api_key=api_key)


def fetch_indicator(
    fred, code: str, start: str, end: str, retries: int = 3
) -> pd.Series:
    """FRED에서 단일 지표 가져오기 (재시도 포함)."""
    for attempt in range(retries):
        try:
            series = fred.get_series(code, observation_start=start, observation_end=end)
            logger.info("  %s: %d개 관측치 (%s ~ %s)", code, len(series), start, end)
            return series
        except Exception as e:
            logger.warning("  %s 실패 (시도 %d/%d): %s", code, attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    logger.error("  %s 최종 실패", code)
    return pd.Series(dtype=float)


def build_dataframe(fred, start: str, end: str) -> pd.DataFrame:
    """5개 지표를 하나의 DataFrame으로 병합."""
    dfs = {}
    for name, cfg in INDICATORS.items():
        series = fetch_indicator(fred, cfg["code"], start, end)
        if series.empty:
            continue
        # 단위 변환
        if cfg["unit_div"] != 1.0:
            series = series / cfg["unit_div"]
        dfs[name] = series
        time.sleep(0.5)  # rate limit 방지

    if not dfs:
        logger.error("모든 지표 수집 실패")
        return pd.DataFrame()

    # 병합 (outer join → ffill로 주간/격주 데이터 보간)
    df = pd.DataFrame(dfs)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    # 주간/격주 데이터를 일간으로 ffill
    df = df.ffill()

    # 영업일만 유지
    bdays = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bdays).ffill()
    df.index.name = "date"

    # 파생 컬럼
    if all(c in df.columns for c in ["walcl", "tga", "rrp"]):
        df["net_liquidity"] = df["walcl"] - df["tga"] - df["rrp"]

    # M2 YoY (52주 = 약 260 영업일 전 대비)
    if "m2" in df.columns and len(df) > 260:
        df["m2_yoy_pct"] = (df["m2"] / df["m2"].shift(260) - 1) * 100
    elif "m2" in df.columns:
        df["m2_yoy_pct"] = 0.0

    # NaN 행 제거 (초기 데이터 부족)
    df = df.dropna(subset=["walcl", "rrp"], how="any")

    return df


def backfill(years: int = 3) -> pd.DataFrame:
    """N년치 데이터 백필."""
    fred = _get_fred()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")

    print(f"FRED 유동성 데이터 백필: {start} ~ {end}")
    df = build_dataframe(fred, start, end)

    if df.empty:
        print("데이터 수집 실패")
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH)
    print(f"저장: {PARQUET_PATH} ({len(df)}행, {list(df.columns)})")
    return df


def update() -> pd.DataFrame:
    """기존 parquet에 최신 데이터 추가."""
    fred = _get_fred()

    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        last_date = existing.index.max()
        start = (last_date - timedelta(days=7)).strftime("%Y-%m-%d")
        print(f"증분 업데이트: {start} ~")
    else:
        print("기존 데이터 없음 → 1년 백필로 전환")
        return backfill(years=1)

    end = datetime.now().strftime("%Y-%m-%d")
    new_df = build_dataframe(fred, start, end)

    if new_df.empty:
        print("새 데이터 없음, 기존 유지")
        return existing

    # 병합 (새 데이터 우선)
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PARQUET_PATH)
    added = len(combined) - len(existing)
    print(f"업데이트 완료: {len(combined)}행 (신규 {added}행)")
    return combined


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="FRED 유동성 사이클 데이터 수집")
    parser.add_argument(
        "--backfill", type=int, default=0, help="N년치 백필 (0=업데이트만)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FRED 유동성 사이클 데이터 수집")
    print("=" * 60)

    if args.backfill > 0:
        df = backfill(years=args.backfill)
    else:
        df = update()

    if not df.empty:
        print(f"\n최신 데이터 ({df.index[-1].date()}):")
        for col in ["rrp", "tga", "walcl", "net_liquidity", "m2", "totresns", "m2_yoy_pct"]:
            if col in df.columns:
                val = df[col].iloc[-1]
                if col == "m2_yoy_pct":
                    print(f"  {col:>16s}: {val:+.2f}%")
                else:
                    print(f"  {col:>16s}: {val:,.1f}B")


if __name__ == "__main__":
    main()
