"""
stock_data_daily/ CSV 일괄 업데이트 스크립트

기능:
  1. 각 CSV 파일의 마지막 날짜를 확인
  2. 빠진 날짜(미수집일)를 FinanceDataReader로 보완
  3. 기술적 지표(37개 컬럼) 재계산 후 append
  4. --schedule 옵션: 매일 17시 자동 실행

사용법:
    python scripts/update_daily_data.py                  # 전체 업데이트 (오늘까지)
    python scripts/update_daily_data.py --date 2026-02-12  # 특정 날짜까지 업데이트
    python scripts/update_daily_data.py --schedule       # 매일 17시 자동 실행
    python scripts/update_daily_data.py --check          # 미수집 현황만 확인
"""

import argparse
import glob
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("daily_updater")

DATA_DIR = Path(__file__).resolve().parent.parent / "stock_data_daily"
LOG_FILE = DATA_DIR / "_update_log.txt"

# ── FinanceDataReader import ──
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False
    logger.error("FinanceDataReader 미설치. pip install finance-datareader")


# ============================================================
# 기술적 지표 계산 (기존 CSV 컬럼과 동일하게)
# ============================================================

def calc_ma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line


def calc_bollinger(close: pd.Series, period=20, std_mult=2):
    ma = calc_ma(close, period)
    std = close.rolling(window=period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, lower


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period=14, d_period=3):
    lowest = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()
    denom = highest - lowest
    k = ((close - lowest) / denom.replace(0, np.nan)) * 100
    d = k.rolling(window=d_period).mean()
    return k, d


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def calc_trix(close: pd.Series, span1=12, span2=26, span3=9, signal_span=9):
    ema1 = calc_ema(close, span1)
    ema2 = calc_ema(close, span2)
    ema3 = calc_ema(close, span3)
    trix = ema3.pct_change() * 100
    trix_signal = calc_ema(trix, signal_span)
    return ema1, ema2, ema3, trix, trix_signal


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm_raw = plus_dm.copy()
    minus_dm_raw = minus_dm.copy()

    # 조건에 맞지 않으면 0
    plus_dm_raw[(plus_dm <= minus_dm) | (plus_dm <= 0)] = 0
    minus_dm_raw[(minus_dm <= plus_dm) | (minus_dm <= 0)] = 0

    atr = calc_atr(high, low, close, period)

    plus_dm_smooth = plus_dm_raw.rolling(window=period).mean()
    minus_dm_smooth = minus_dm_raw.rolling(window=period).mean()

    plus_di = (plus_dm_smooth / atr.replace(0, np.nan)) * 100
    minus_di = (minus_dm_smooth / atr.replace(0, np.nan)) * 100

    di_sum = plus_di + minus_di
    dx = ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan)) * 100
    adx = dx.rolling(window=period).mean()

    return plus_dm, minus_dm, plus_dm_smooth, minus_dm_smooth, plus_di, minus_di, dx, adx


def recalc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """전체 DataFrame에 기술적 지표 37개 컬럼 재계산"""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # 이동평균
    df["MA5"] = calc_ma(close, 5)
    df["MA20"] = calc_ma(close, 20)
    df["MA60"] = calc_ma(close, 60)
    df["MA120"] = calc_ma(close, 120)

    # RSI
    df["RSI"] = calc_rsi(close, 14)

    # MACD
    macd, macd_sig = calc_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = macd_sig

    # 볼린저밴드
    upper, lower = calc_bollinger(close)
    df["Upper_Band"] = upper
    df["Lower_Band"] = lower

    # ATR
    df["ATR"] = calc_atr(high, low, close)

    # 스토캐스틱
    stoch_k, stoch_d = calc_stochastic(high, low, close)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d

    # OBV
    df["OBV"] = calc_obv(close, volume)

    # Next_Close, Target
    # shift(-1): 마지막 행은 구조적으로 NaN (다음 거래일 데이터 미존재)
    # → 다음 업데이트 시 recalc으로 자동 채워짐
    df["Next_Close"] = close.shift(-1)
    df["Target"] = (df["Next_Close"] > close).astype(int)

    # 마지막 행 NaN 처리: 분석 시 오류 방지를 위해 0 대신 NaN 유지
    # (0으로 채우면 "다음날 종가=0" 으로 오인 → 오히려 위험)

    # MarketCap (기존 값 유지, 신규는 0)
    if "MarketCap" not in df.columns:
        df["MarketCap"] = 0

    # EMA / TRIX
    ema1, ema2, ema3, trix, trix_sig = calc_trix(close)
    df["EMA1"] = ema1
    df["EMA2"] = ema2
    df["EMA3"] = ema3
    df["TRIX"] = trix
    df["TRIX_Signal"] = trix_sig

    # ADX
    plus_dm, minus_dm, plus_dm_14, minus_dm_14, plus_di, minus_di, dx, adx = calc_adx(high, low, close)
    df["Plus_DM"] = plus_dm
    df["Minus_DM"] = minus_dm
    df["Plus_DM_14"] = plus_dm_14
    df["Minus_DM_14"] = minus_dm_14
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    df["DX"] = dx
    df["ADX"] = adx

    # Foreign_Net, Inst_Net (FDR에서 제공 안 함 → 기존 값 유지, 신규는 0)
    if "Foreign_Net" not in df.columns:
        df["Foreign_Net"] = 0.0
    if "Inst_Net" not in df.columns:
        df["Inst_Net"] = 0.0

    return df


# ============================================================
# 데이터 무결성 검증 (재발방지)
# ============================================================

def verify_data_integrity(df: pd.DataFrame) -> dict:
    """
    업데이트 후 데이터 무결성 검증.

    검증 항목:
      1. Next_Close: 마지막 행 제외, 중간 행에 NaN 없는지
      2. Date: 날짜 순서 정렬, 중복 없는지
      3. Close: 0 또는 NaN 행 없는지
      4. 컬럼 누락 없는지
    """
    errors = []
    warnings = []

    # 1) Next_Close 중간 누락 검증 (마지막 행은 NaN 허용)
    if "Next_Close" in df.columns and len(df) > 1:
        middle_rows = df.iloc[:-1]  # 마지막 행 제외
        nc_nulls = middle_rows["Next_Close"].isna().sum()
        if nc_nulls > 0:
            null_dates = middle_rows[middle_rows["Next_Close"].isna()]["Date"].tolist()
            errors.append(
                f"Next_Close 중간 누락 {nc_nulls}건: {null_dates[:5]}"
            )

    # 2) 날짜 정렬 및 중복 검증
    if "Date" in df.columns:
        dates = df["Date"].tolist()
        if dates != sorted(dates):
            errors.append("날짜 정렬 오류 — Date 컬럼이 오름차순이 아님")
        dup_dates = df[df["Date"].duplicated()]["Date"].tolist()
        if dup_dates:
            errors.append(f"날짜 중복 {len(dup_dates)}건: {dup_dates[:5]}")

    # 3) Close 값 검증
    if "Close" in df.columns:
        zero_close = (df["Close"] == 0).sum()
        nan_close = df["Close"].isna().sum()
        if zero_close > 0:
            errors.append(f"Close=0 행 {zero_close}건")
        if nan_close > 0:
            errors.append(f"Close=NaN 행 {nan_close}건")

    # 4) 필수 컬럼 존재 검증
    required = ["Date", "Open", "High", "Low", "Close", "Volume", "Next_Close", "Target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"필수 컬럼 누락: {missing}")

    # 5) 마지막 행 Next_Close NaN 경고 (정상이지만 기록)
    if "Next_Close" in df.columns and len(df) > 0:
        last_nc = df["Next_Close"].iloc[-1]
        if pd.isna(last_nc):
            warnings.append(
                f"마지막 행({df['Date'].iloc[-1]}) Next_Close=NaN — "
                f"다음 거래일 업데이트 시 자동 채워짐"
            )

    return {
        "errors": errors,
        "warnings": warnings,
        "total_rows": len(df),
        "last_date": str(df["Date"].iloc[-1]) if len(df) > 0 else "",
    }


# ============================================================
# 핵심: 단일 CSV 파일 업데이트
# ============================================================

def update_single_csv(csv_path: Path, target_date: str) -> dict:
    """
    단일 CSV 파일을 target_date까지 업데이트.

    Returns:
        {"file": str, "ticker": str, "last_date": str,
         "new_rows": int, "status": "ok"|"skip"|"error", "error": str}
    """
    fname = csv_path.stem
    result = {"file": fname, "ticker": "", "last_date": "", "new_rows": 0,
              "status": "skip", "error": ""}

    # 파일명에서 종목코드 추출 (예: 삼성전자_005930 → 005930)
    parts = fname.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        result["status"] = "skip"
        result["error"] = "종목코드 추출 실패"
        return result

    ticker = parts[1]
    result["ticker"] = ticker

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"CSV 읽기 실패: {e}"
        return result

    if "Date" not in df.columns or len(df) == 0:
        result["status"] = "error"
        result["error"] = "Date 컬럼 없음 또는 빈 파일"
        return result

    last_date = str(df["Date"].iloc[-1])[:10]
    result["last_date"] = last_date

    # 이미 최신이면 skip
    if last_date >= target_date:
        result["status"] = "skip"
        return result

    # 빠진 날짜 범위 조회 (마지막 날짜 다음날 ~ target_date)
    fetch_start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        new_data = fdr.DataReader(ticker, fetch_start, target_date)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"FDR 조회 실패: {e}"
        return result

    if new_data.empty:
        result["status"] = "skip"
        return result

    # 새 데이터를 기존 형식에 맞게 변환
    new_rows = pd.DataFrame({
        "Date": new_data.index.strftime("%Y-%m-%d"),
        "Open": new_data["Open"].values,
        "High": new_data["High"].values,
        "Low": new_data["Low"].values,
        "Close": new_data["Close"].values,
        "Volume": new_data["Volume"].values,
    })

    # 기존 Foreign_Net, Inst_Net, MarketCap은 0으로 초기화
    for col in ["Foreign_Net", "Inst_Net", "MarketCap"]:
        new_rows[col] = 0.0

    # 기존 데이터와 합치기
    # 기존 df에서 기술지표 제외한 OHLCV + 보존 컬럼만 남긴 뒤 합침
    base_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Foreign_Net", "Inst_Net", "MarketCap"]

    # 기존 df에서 base 컬럼 추출
    existing_base = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Foreign_Net", "Inst_Net", "MarketCap"]:
        if col in df.columns:
            existing_base[col] = df[col].values
        else:
            existing_base[col] = 0.0

    # 중복 날짜 제거
    existing_dates = set(existing_base["Date"].values)
    new_rows_filtered = new_rows[~new_rows["Date"].isin(existing_dates)]

    if new_rows_filtered.empty:
        result["status"] = "skip"
        return result

    # 합치기
    combined = pd.concat([existing_base, new_rows_filtered], ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    # 기술적 지표 전체 재계산
    combined = recalc_all_indicators(combined)

    # 컬럼 순서 보장 (원본 CSV와 동일)
    expected_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_Signal",
        "Upper_Band", "Lower_Band", "ATR", "Stoch_K", "Stoch_D", "OBV",
        "Next_Close", "Target", "MarketCap",
        "EMA1", "EMA2", "EMA3", "TRIX", "TRIX_Signal",
        "Plus_DM", "Minus_DM", "Plus_DM_14", "Minus_DM_14",
        "Plus_DI", "Minus_DI", "DX", "ADX",
        "Foreign_Net", "Inst_Net",
    ]
    for col in expected_cols:
        if col not in combined.columns:
            combined[col] = 0.0
    combined = combined[expected_cols]

    # 저장 전 무결성 검증
    integrity = verify_data_integrity(combined)
    if integrity["errors"]:
        for err in integrity["errors"]:
            logger.warning(f"  [{fname}] 무결성 경고: {err}")

    # 저장
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result["new_rows"] = len(new_rows_filtered)
    result["integrity"] = integrity
    result["status"] = "ok"
    return result


# ============================================================
# 일괄 업데이트
# ============================================================

def update_all(target_date: str = None, check_only: bool = False):
    """stock_data_daily/ 전체 CSV 파일 업데이트"""
    if not FDR_AVAILABLE:
        logger.error("FinanceDataReader 미설치. 종료.")
        return

    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    # _update_log.txt 제외
    csv_files = [f for f in csv_files if not f.name.startswith("_")]

    total = len(csv_files)
    logger.info(f"대상 파일: {total}개 | 목표 날짜: {target_date}")

    if check_only:
        _check_status(csv_files, target_date)
        return

    updated = 0
    skipped = 0
    errors = 0
    error_list = []
    integrity_issues = []

    start_time = time.time()

    for i, csv_path in enumerate(csv_files):
        result = update_single_csv(csv_path, target_date)

        if result["status"] == "ok":
            updated += 1
            logger.info(f"  [{i+1}/{total}] {result['file']}: +{result['new_rows']}행 추가")
            # 무결성 이슈 수집
            integrity = result.get("integrity", {})
            if integrity.get("errors"):
                integrity_issues.append((result["file"], integrity["errors"]))
        elif result["status"] == "error":
            errors += 1
            error_list.append(f"{result['file']}: {result['error']}")
            logger.warning(f"  [{i+1}/{total}] {result['file']}: 오류 - {result['error']}")
        else:
            skipped += 1

        # FDR rate limit (초당 5회 정도)
        if result["status"] == "ok":
            time.sleep(0.3)

        # 진행률 (100개마다)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  --- 진행: {i+1}/{total} ({elapsed:.0f}초) | 업데이트: {updated} | 스킵: {skipped} | 오류: {errors}")

    elapsed = time.time() - start_time

    # 무결성 검증 결과 요약
    integrity_msg = ""
    if integrity_issues:
        integrity_msg = f"\n  ⚠ 무결성 경고: {len(integrity_issues)}개 파일"
        for fname, errs in integrity_issues[:5]:
            for e in errs:
                integrity_msg += f"\n    - {fname}: {e}"
    else:
        integrity_msg = "\n  ✓ 무결성 검증 통과 (Next_Close 중간 누락 없음)"

    # 결과 요약
    summary = (
        f"\n{'='*50}\n"
        f"업데이트 완료 ({target_date})\n"
        f"{'='*50}\n"
        f"  총 파일: {total}개\n"
        f"  업데이트: {updated}개\n"
        f"  스킵(최신): {skipped}개\n"
        f"  오류: {errors}개\n"
        f"  소요시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)"
        f"{integrity_msg}\n"
        f"\n  ※ 마지막 행({target_date}) Next_Close=NaN은 정상\n"
        f"    → 다음 거래일 업데이트 시 자동 채워짐\n"
        f"{'='*50}"
    )
    logger.info(summary)

    # 로그 파일 기록
    _write_log(target_date, updated, skipped, errors, error_list)

    return {"updated": updated, "skipped": skipped, "errors": errors}


def _check_status(csv_files: list, target_date: str):
    """미수집 현황만 확인 (업데이트 안 함)"""
    from collections import Counter

    dates = []
    no_date = 0

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, usecols=["Date"])
            if len(df) > 0:
                dates.append(str(df["Date"].iloc[-1])[:10])
            else:
                no_date += 1
        except Exception:
            no_date += 1

    counter = Counter(dates)
    logger.info(f"\n{'='*50}")
    logger.info(f"미수집 현황 (목표: {target_date})")
    logger.info(f"{'='*50}")
    logger.info(f"  총 파일: {len(csv_files)}개")

    needs_update = 0
    for date, cnt in sorted(counter.items()):
        marker = " <-- 최신" if date >= target_date else " ** 업데이트 필요"
        if date < target_date:
            needs_update += cnt
        logger.info(f"  {date}: {cnt}개{marker}")

    if no_date:
        logger.info(f"  (날짜 없음): {no_date}개")

    logger.info(f"\n  업데이트 필요: {needs_update}개 / {len(csv_files)}개")
    logger.info(f"{'='*50}")


def _write_log(target_date: str, updated: int, skipped: int, errors: int, error_list: list):
    """업데이트 로그 파일 기록"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{now}] 업데이트 target={target_date} | "
                f"updated={updated} skipped={skipped} errors={errors}\n")
        if error_list:
            for err in error_list[:10]:
                f.write(f"  ERROR: {err}\n")


# ============================================================
# 스케줄러 (매일 17시)
# ============================================================

def run_scheduler():
    """매일 17:00에 자동 업데이트 실행"""
    logger.info("스케줄러 시작: 매일 17:00 자동 업데이트")
    logger.info("  종료: Ctrl+C")

    while True:
        now = datetime.now()
        target_time = now.replace(hour=17, minute=0, second=0, microsecond=0)

        # 이미 17시가 지났으면 내일 17시
        if now >= target_time:
            target_time += timedelta(days=1)

        wait_seconds = (target_time - now).total_seconds()
        logger.info(f"  다음 실행: {target_time.strftime('%Y-%m-%d %H:%M')} "
                     f"(대기: {wait_seconds/3600:.1f}시간)")

        try:
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            logger.info("스케줄러 종료")
            break

        # 17시 도달 → 업데이트 실행
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"\n{'='*50}")
        logger.info(f"[스케줄] {today} 17:00 자동 업데이트 시작")
        logger.info(f"{'='*50}")

        try:
            result = update_all(today)

            # 업데이트 완료 후 텔레그램 알림 (선택)
            try:
                from src.telegram_sender import send_message
                msg = (
                    f"[퀀텀전략 v3.0] 일일 데이터 업데이트 완료\n"
                    f"날짜: {today}\n"
                    f"업데이트: {result['updated']}개 | "
                    f"스킵: {result['skipped']}개 | "
                    f"오류: {result['errors']}개"
                )
                send_message(msg)
            except Exception:
                pass  # 텔레그램 실패해도 무시

        except Exception as e:
            logger.error(f"[스케줄] 업데이트 실패: {e}")


# ============================================================
# 미수집일 보완 (과거 누락분 채우기)
# ============================================================

def backfill_missing(csv_path: Path, target_date: str) -> dict:
    """
    단일 CSV에서 중간 누락된 거래일도 채우기.
    (연속적이지 않은 날짜 gap 탐지 → FDR로 보완)
    """
    fname = csv_path.stem
    parts = fname.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return {"file": fname, "status": "skip", "filled": 0}

    ticker = parts[1]

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {"file": fname, "status": "error", "filled": 0}

    if "Date" not in df.columns or len(df) == 0:
        return {"file": fname, "status": "error", "filled": 0}

    # 기존 날짜 집합
    existing_dates = set(df["Date"].values)
    first_date = df["Date"].iloc[0]
    last_date = df["Date"].iloc[-1]

    # FDR로 전체 기간 거래일 조회
    try:
        full_data = fdr.DataReader(ticker, first_date, target_date)
    except Exception as e:
        return {"file": fname, "status": "error", "filled": 0, "error": str(e)}

    if full_data.empty:
        return {"file": fname, "status": "skip", "filled": 0}

    # 누락된 날짜 찾기
    all_trading_dates = set(full_data.index.strftime("%Y-%m-%d"))
    missing_dates = all_trading_dates - existing_dates

    if not missing_dates:
        return {"file": fname, "status": "skip", "filled": 0}

    # 누락분 데이터 추출
    missing_data = full_data[full_data.index.strftime("%Y-%m-%d").isin(missing_dates)]

    new_rows = pd.DataFrame({
        "Date": missing_data.index.strftime("%Y-%m-%d"),
        "Open": missing_data["Open"].values,
        "High": missing_data["High"].values,
        "Low": missing_data["Low"].values,
        "Close": missing_data["Close"].values,
        "Volume": missing_data["Volume"].values,
        "Foreign_Net": 0.0,
        "Inst_Net": 0.0,
        "MarketCap": 0.0,
    })

    # 기존 base 추출
    base_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    existing_base = df[base_cols].copy()
    for col in ["Foreign_Net", "Inst_Net", "MarketCap"]:
        existing_base[col] = df[col].values if col in df.columns else 0.0

    combined = pd.concat([existing_base, new_rows], ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    combined = recalc_all_indicators(combined)

    expected_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_Signal",
        "Upper_Band", "Lower_Band", "ATR", "Stoch_K", "Stoch_D", "OBV",
        "Next_Close", "Target", "MarketCap",
        "EMA1", "EMA2", "EMA3", "TRIX", "TRIX_Signal",
        "Plus_DM", "Minus_DM", "Plus_DM_14", "Minus_DM_14",
        "Plus_DI", "Minus_DI", "DX", "ADX",
        "Foreign_Net", "Inst_Net",
    ]
    for col in expected_cols:
        if col not in combined.columns:
            combined[col] = 0.0
    combined = combined[expected_cols]

    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {"file": fname, "status": "ok", "filled": len(missing_dates)}


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = argparse.ArgumentParser(
        description="stock_data_daily CSV 일괄 업데이트"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="목표 날짜 (기본: 오늘, 예: 2026-02-12)",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="매일 17시 자동 실행 모드",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="미수집 현황만 확인 (업데이트 안 함)",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="중간 누락일도 보완 (느림, 전체 재조회)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="전체 CSV 무결성 검증 (Next_Close 중간 누락, 날짜 중복 등)",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduler()
    elif args.verify:
        # 전체 무결성 검증 모드
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.startswith("_")]
        logger.info(f"무결성 검증: {len(csv_files)}개 파일")
        issue_count = 0
        for i, csv_path in enumerate(csv_files):
            try:
                df = pd.read_csv(csv_path)
                result = verify_data_integrity(df)
                if result["errors"]:
                    issue_count += 1
                    logger.warning(f"  [{csv_path.stem}] 오류 {len(result['errors'])}건:")
                    for e in result["errors"]:
                        logger.warning(f"    - {e}")
            except Exception as e:
                issue_count += 1
                logger.error(f"  [{csv_path.stem}] 읽기 실패: {e}")
            if (i + 1) % 500 == 0:
                logger.info(f"  --- 진행: {i+1}/{len(csv_files)}")
        if issue_count == 0:
            logger.info(f"\n✓ 전체 {len(csv_files)}개 파일 무결성 검증 통과")
        else:
            logger.warning(f"\n⚠ {issue_count}/{len(csv_files)}개 파일에서 문제 발견")
    elif args.backfill:
        target = args.date or datetime.now().strftime("%Y-%m-%d")
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.startswith("_")]
        logger.info(f"중간 누락 보완 모드: {len(csv_files)}개 파일")
        filled_total = 0
        for i, f in enumerate(csv_files):
            r = backfill_missing(f, target)
            if r["status"] == "ok":
                filled_total += r["filled"]
                logger.info(f"  [{i+1}/{len(csv_files)}] {r['file']}: +{r['filled']}행 보완")
            if (i + 1) % 100 == 0:
                logger.info(f"  --- 진행: {i+1}/{len(csv_files)}")
            time.sleep(0.3)
        logger.info(f"보완 완료: 총 {filled_total}행 추가")
    else:
        target = args.date or datetime.now().strftime("%Y-%m-%d")
        update_all(target, check_only=args.check)
