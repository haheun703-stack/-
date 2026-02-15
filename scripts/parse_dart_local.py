#!/usr/bin/env python3
"""
DART 로컬 ZIP 파일 → fundamentals_all.csv 변환 스크립트
======================================================
DART 공시정보활용마당에서 다운로드한 ZIP 파일들을 파싱하여
FundamentalEngine이 사용하는 fundamentals_all.csv 형태로 변환한다.

사용법:
    python scripts/parse_dart_local.py

입력: DART 재무정보_보고서,재무상태표,손익계산서,현금흐름표,자본변동표/
출력: data/dart_cache/fundamentals_all.csv
"""
import os
import sys
import zipfile
import io
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ── 핵심 항목 코드 매핑 ──
# 손익계산서(PL)에서 추출할 항목
REVENUE_CODES = {"ifrs-full_Revenue", "dart_OperatingRevenue"}
REVENUE_NAMES = {
    "매출액", "수익(매출액)", "매출", "영업수익", "영업수익(매출)",
    "Ⅰ.수익(매출액)", "I. 영업수익", "I.영업수익", "Ⅰ.영업수익",
}

OP_INCOME_CODES = {"dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"}
OP_INCOME_NAMES = {
    "영업이익", "영업이익(손실)", "영업손익", "Ⅳ.영업이익(손실)",
    "III. 영업이익", "III. 영업이익(손실)", "III.영업이익(손실)", "Ⅲ.영업이익",
    "Ⅵ.영업이익(손실)",
}

NET_INCOME_CODES = {"ifrs-full_ProfitLoss"}
NET_INCOME_NAMES = {
    "당기순이익", "당기순이익(손실)", "당기순손익", "분기순이익(손실)",
    "Ⅵ.당기순이익", "VIII. 당기순이익", "VIII. 당기순이익(손실)",
    "VIII.당기순이익", "VIII.당기순이익(손실)", "XI.당기순이익(손실)",
    "분기순이익", "VII. 분기순이익", "VIII. 분기순이익", "VIII. 분기순이익(손실)",
}

EPS_CODES = {"ifrs-full_BasicEarningsLossPerShare"}
EPS_NAMES = {
    "기본주당순이익(손실)", "기본주당순이익", "보통주기본주당순이익(손실)", "기본주당순손익",
    "기본보통주당순이익", "기본보통주당순이익(손실)", "보통주기본주당이익(손실)",
    "기본주당이익(손실)", "기본주당이익", "기본주당손익", "보통주기본주당이익",
    "보통주 기본주당이익",
}


def decode_zip_filename(raw_filename: str) -> str:
    """ZIP 파일 내 cp949 인코딩 파일명 디코딩."""
    try:
        return raw_filename.encode('cp437').decode('cp949')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw_filename


def parse_amount(val: str) -> float | None:
    """금액 문자열 → float 변환 (쉼표 제거, 빈 값 처리)."""
    if not val or not val.strip():
        return None
    val = val.strip().replace(",", "").replace(" ", "")
    if not val or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def extract_ticker(code_str: str) -> str | None:
    """'[005930]' → '005930'"""
    m = re.search(r'\[(\d{6})\]', code_str)
    return m.group(1) if m else None


def read_all_tsvs_from_zip(zip_path: str, name_filter: str,
                           exclude_filters: list = None) -> list[pd.DataFrame]:
    """ZIP 내 매칭되는 TSV 파일들을 개별 DataFrame 리스트로 반환.

    컬럼 구조가 다른 파일(일반 vs 금융)이 있으므로 concat하지 않고 개별 반환.

    Args:
        zip_path: ZIP 파일 경로
        name_filter: 파일명에 포함되어야 하는 문자열 (e.g., '손익계산서')
        exclude_filters: 파일명에 포함되면 제외 (e.g., ['연결'])
    """
    if exclude_filters is None:
        exclude_filters = ["연결"]

    dfs = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                decoded = decode_zip_filename(info.filename)
                if name_filter not in decoded:
                    continue
                if any(ex in decoded for ex in exclude_filters):
                    continue

                with zf.open(info.filename) as f:
                    content = f.read()
                    text = content.decode('cp949', errors='replace')
                    df = pd.read_csv(
                        io.StringIO(text), sep='\t',
                        dtype=str, on_bad_lines='skip'
                    )
                    dfs.append(df)
    except Exception as e:
        print(f"  [WARN] {zip_path}: {e}")

    return dfs


def _find_current_period_col(df: pd.DataFrame) -> int:
    """'당기' 금액이 있는 컬럼 인덱스를 찾는다.

    컬럼명에서 '당기'로 시작하는 첫 번째 컬럼을 반환.
    일반: '당기 1분기 3개월', '당기 1분기 누적', '당기'
    금융: '당기 1분기말', '당기'
    """
    for i, col_name in enumerate(df.columns):
        col_name = col_name.strip()
        if col_name.startswith("당기"):
            return i
    # fallback: 12번 컬럼
    return 12


def _process_df(df: pd.DataFrame, result: dict):
    """단일 DataFrame에서 종목별 핵심 항목을 result에 누적."""
    if df is None or df.empty:
        return

    # 컬럼 정리 (빈 컬럼명 제거)
    cols = [c for c in df.columns if c.strip()]
    df = df[cols] if len(cols) < len(df.columns) else df

    # '당기' 금액 컬럼 위치 찾기 (파일마다 다를 수 있음)
    amt_col = _find_current_period_col(df)

    for _, row in df.iterrows():
        ticker_raw = str(row.iloc[1]).strip() if len(row) > 1 else ""
        ticker = extract_ticker(ticker_raw)
        if not ticker:
            continue

        if ticker not in result:
            result[ticker] = {
                "company": str(row.iloc[2]).strip() if len(row) > 2 else "",
                "market": str(row.iloc[3]).strip() if len(row) > 3 else "",
                "sector_name": str(row.iloc[5]).strip() if len(row) > 5 else "",
                "settle_date": str(row.iloc[7]).strip() if len(row) > 7 else "",
                "revenue": None,
                "op_income": None,
                "net_income": None,
                "eps": None,
            }

        item_code = str(row.iloc[10]).strip() if len(row) > 10 else ""
        item_name = str(row.iloc[11]).strip() if len(row) > 11 else ""

        # 당기 금액 (해당 파일의 '당기' 컬럼 사용)
        amount = None
        if len(row) > amt_col:
            amount = parse_amount(str(row.iloc[amt_col]))

        if amount is None:
            continue

        # 매출액 / 영업수익
        if item_code in REVENUE_CODES or item_name in REVENUE_NAMES:
            if result[ticker]["revenue"] is None:
                result[ticker]["revenue"] = amount

        # 영업이익
        if item_code in OP_INCOME_CODES or item_name in OP_INCOME_NAMES:
            if result[ticker]["op_income"] is None:
                result[ticker]["op_income"] = amount

        # 당기순이익
        if item_code in NET_INCOME_CODES or item_name in NET_INCOME_NAMES:
            if result[ticker]["net_income"] is None:
                result[ticker]["net_income"] = amount

        # EPS
        if item_code in EPS_CODES or item_name in EPS_NAMES:
            if result[ticker]["eps"] is None:
                result[ticker]["eps"] = amount


def extract_pl_data(zip_path: str) -> dict:
    """손익계산서 ZIP에서 종목별 핵심 항목 추출.

    Returns:
        {ticker: {revenue, op_income, net_income, eps}}
    """
    result = {}

    # 1. 포괄손익계산서 (일반+증권+은행+보험+금융기타, 연결만 제외)
    ci_dfs = read_all_tsvs_from_zip(zip_path, "포괄손익계산서", exclude_filters=["연결"])

    # 2. 손익계산서 (일반만 — 금융사는 포괄손익에서 처리)
    pl_dfs = read_all_tsvs_from_zip(zip_path, "손익계산서",
                                     exclude_filters=["연결", "금융기타", "보험", "은행", "증권", "포괄"])

    # 각 DataFrame을 개별 처리 (컬럼 구조가 파일마다 다름)
    for df in pl_dfs:
        _process_df(df, result)
    for df in ci_dfs:
        _process_df(df, result)

    return result


def find_latest_quarter_zips(base_dir: str) -> dict:
    """연도별로 가장 최근 분기의 PL ZIP 파일 찾기.

    Returns:
        {year: zip_path}  — 해당 연도의 최신 분기 PL ZIP
    """
    result = {}
    for year_dir in sorted(Path(base_dir).iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = year_dir.name

        # PL 파일 찾기 (4Q → 3Q → 2Q → 1Q 우선)
        pl_files = []
        for f in year_dir.iterdir():
            if f.suffix == '.zip' and '_PL_' in f.name:
                pl_files.append(f)

        if not pl_files:
            continue

        # 분기 내림차순 정렬 (4Q > 3Q > 2Q > 1Q)
        def quarter_key(p):
            m = re.search(r'_(\d)Q_', p.name)
            return int(m.group(1)) if m else 0

        pl_files.sort(key=quarter_key, reverse=True)
        result[year] = str(pl_files[0])

    return result


def build_fundamentals_csv(base_dir: str, output_path: str):
    """전체 DART ZIP → fundamentals_all.csv 빌드."""
    print("=" * 60)
    print("DART 로컬 파일 → fundamentals_all.csv 변환")
    print("=" * 60)

    # 각 연도의 최신 분기 PL ZIP 찾기
    latest_zips = find_latest_quarter_zips(base_dir)
    print(f"\n발견된 연도: {sorted(latest_zips.keys())}")

    # 가장 최근 연도부터 처리 (최신 데이터 우선)
    all_data = {}
    for year in sorted(latest_zips.keys(), reverse=True):
        zip_path = latest_zips[year]
        quarter = re.search(r'_(\d)Q_', zip_path).group(1)
        print(f"\n  [{year} {quarter}Q] {Path(zip_path).name}")

        data = extract_pl_data(zip_path)
        print(f"    추출 종목: {len(data)}")

        for ticker, info in data.items():
            if ticker not in all_data:
                all_data[ticker] = {
                    "year": year,
                    "quarter": quarter,
                    **info,
                }

    print(f"\n총 종목: {len(all_data)}")

    # DataFrame 구성
    rows = []
    for ticker, info in sorted(all_data.items()):
        revenue_억 = info["revenue"] / 1e8 if info["revenue"] else None
        op_income_억 = info["op_income"] / 1e8 if info["op_income"] else None
        net_income_억 = info["net_income"] / 1e8 if info["net_income"] else None

        op_margin = None
        if revenue_억 and revenue_억 > 0 and op_income_억 is not None:
            op_margin = round(op_income_억 / revenue_억 * 100, 2)

        profitable = op_income_억 > 0 if op_income_억 is not None else None

        rows.append({
            "ticker": ticker,
            "company": info.get("company", ""),
            "market": info.get("market", ""),
            "sector_name": info.get("sector_name", ""),
            "year": info["year"],
            "quarter": info["quarter"],
            "revenue_억": round(revenue_억, 1) if revenue_억 else None,
            "op_income_억": round(op_income_억, 1) if op_income_억 else None,
            "net_income_억": round(net_income_억, 1) if net_income_억 else None,
            "op_margin_pct": op_margin,
            "profitable": profitable,
            "eps": info.get("eps"),
        })

    df = pd.DataFrame(rows)

    # 통계
    has_revenue = df["revenue_억"].notna().sum()
    has_op = df["op_income_억"].notna().sum()
    has_net = df["net_income_억"].notna().sum()
    print(f"\n데이터 커버리지:")
    print(f"  매출액:    {has_revenue}/{len(df)} ({has_revenue/len(df)*100:.1f}%)")
    print(f"  영업이익:  {has_op}/{len(df)} ({has_op/len(df)*100:.1f}%)")
    print(f"  순이익:    {has_net}/{len(df)} ({has_net/len(df)*100:.1f}%)")

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path} ({len(df)} 종목)")

    return df


def build_historical_fundamentals(base_dir: str, output_path: str):
    """모든 연도/분기의 히스토리컬 재무데이터 빌드.

    백테스트에서 look-ahead bias 방지를 위해 사용.
    """
    print("\n" + "=" * 60)
    print("히스토리컬 재무데이터 빌드 (백테스트용)")
    print("=" * 60)

    all_rows = []
    for year_dir in sorted(Path(base_dir).iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = year_dir.name

        for f in sorted(year_dir.iterdir()):
            if not f.suffix == '.zip' or '_PL_' not in f.name:
                continue

            m = re.search(r'_(\d)Q_', f.name)
            quarter = m.group(1) if m else "?"
            print(f"  [{year} {quarter}Q] {f.name}")

            data = extract_pl_data(str(f))
            for ticker, info in data.items():
                revenue_억 = info["revenue"] / 1e8 if info["revenue"] else None
                op_income_억 = info["op_income"] / 1e8 if info["op_income"] else None
                net_income_억 = info["net_income"] / 1e8 if info["net_income"] else None

                op_margin = None
                if revenue_억 and revenue_억 > 0 and op_income_억 is not None:
                    op_margin = round(op_income_억 / revenue_억 * 100, 2)

                all_rows.append({
                    "ticker": ticker,
                    "company": info.get("company", ""),
                    "year": year,
                    "quarter": quarter,
                    "settle_date": info.get("settle_date", ""),
                    "revenue_억": round(revenue_억, 1) if revenue_억 else None,
                    "op_income_억": round(op_income_억, 1) if op_income_억 else None,
                    "net_income_억": round(net_income_억, 1) if net_income_억 else None,
                    "op_margin_pct": op_margin,
                    "profitable": op_income_억 > 0 if op_income_억 is not None else None,
                    "eps": info.get("eps"),
                })

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n히스토리컬 저장: {output_path} ({len(df)} 행, {df['ticker'].nunique()} 종목)")

    return df


if __name__ == "__main__":
    base = "DART 재무정보_보고서,재무상태표,손익계산서,현금흐름표,자본변동표"

    if not Path(base).exists():
        print(f"[ERROR] '{base}' 폴더가 없습니다.")
        sys.exit(1)

    # 1. 최신 분기 기준 fundamentals_all.csv (FundamentalEngine 직접 사용)
    df_latest = build_fundamentals_csv(base, "data/dart_cache/fundamentals_all.csv")

    # 2. 히스토리컬 전체 (백테스트용)
    df_hist = build_historical_fundamentals(base, "data/dart_cache/fundamentals_historical.csv")

    print("\n" + "=" * 60)
    print("변환 완료!")
    print(f"  최신 재무: data/dart_cache/fundamentals_all.csv ({len(df_latest)} 종목)")
    print(f"  히스토리컬: data/dart_cache/fundamentals_historical.csv ({len(df_hist)} 행)")
    print("=" * 60)
