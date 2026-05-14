"""효성_004800.csv 일회성 복구 스크립트

문제:
- VPS + 로컬 stock_data_daily/효성_004800.csv 모두 0 byte
- BAT-D update_daily_data에서 5/11부터 매일 "No columns to parse" 에러
- 자동 백필 로직 없음 (라인 322-324)

해결:
- FDR로 효성(004800) OHLCV fetch (2015-12-24 ~ 어제)
- update_daily_data.recalc_all_indicators()로 37컬럼 일괄 계산
- 정상 CSV 포맷(utf-8-sig BOM)으로 저장
- 로컬 → VPS scp는 별도

실행:
    python -X utf8 scripts/one_off/fix_hyosung_004800.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import FinanceDataReader as fdr

from scripts.update_daily_data import recalc_all_indicators


TICKER = "004800"
NAME = "효성"
CSV_PATH = PROJECT_ROOT / "stock_data_daily" / f"{NAME}_{TICKER}.csv"
START_DATE = "2015-12-24"
END_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def main():
    print(f"[1/5] FDR fetch: {TICKER} {START_DATE} ~ {END_DATE}")
    df = fdr.DataReader(TICKER, START_DATE, END_DATE)
    if df.empty:
        print(f"[ERROR] FDR 데이터 비어있음")
        sys.exit(1)
    print(f"  → {len(df)}행 수집")

    # 컬럼 표준화: Date, Open, High, Low, Close, Volume + 보조 컬럼
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # 기본 컬럼 순서 + 보조 컬럼 추가 (recalc_all_indicators가 필요로 함)
    base = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    base["Foreign_Net"] = 0.0
    base["Inst_Net"] = 0.0
    base["MarketCap"] = 0.0

    print(f"[2/5] recalc_all_indicators 호출 (37컬럼 일괄 계산)")
    result = recalc_all_indicators(base)
    print(f"  → 컬럼 {len(result.columns)}개")

    # 정상 CSV와 동일한 컬럼 순서로 정렬
    expected_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "MA5", "MA20", "MA60", "MA120",
        "RSI", "MACD", "MACD_Signal", "Upper_Band", "Lower_Band",
        "ATR", "Stoch_K", "Stoch_D", "OBV",
        "Next_Close", "Target", "MarketCap",
        "EMA1", "EMA2", "EMA3", "TRIX", "TRIX_Signal",
        "Plus_DM", "Minus_DM", "Plus_DM_14", "Minus_DM_14",
        "Plus_DI", "Minus_DI", "DX", "ADX",
        "Foreign_Net", "Inst_Net",
    ]
    missing = [c for c in expected_cols if c not in result.columns]
    if missing:
        print(f"[WARN] 누락 컬럼: {missing} → 0으로 채움")
        for c in missing:
            result[c] = 0.0

    result = result[expected_cols]
    print(f"[3/5] 컬럼 순서 정렬 완료 ({len(result.columns)}컬럼)")

    print(f"[4/5] CSV 저장: {CSV_PATH}")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # 검증
    import os
    size = os.path.getsize(CSV_PATH)
    print(f"  → 파일 크기: {size:,} byte")

    re_read = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"  → 재읽기 검증: {len(re_read)}행 × {len(re_read.columns)}컬럼")
    print(f"  → 마지막 날짜: {re_read['Date'].iloc[-1]}")
    print(f"  → 마지막 종가: {re_read['Close'].iloc[-1]:,.0f}")

    print(f"[5/5] 완료")
    print(f"  → VPS scp: scp -i ... {CSV_PATH} ubuntu@13.209.153.221:~/quantum-master/stock_data_daily/")


if __name__ == "__main__":
    main()
