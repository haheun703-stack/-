#!/usr/bin/env python3
"""누락 종목 OHLCV 데이터 수집 → parquet 저장"""
from pykrx import stock as pykrx_stock
import pandas as pd
import os, time

RAW_DIR = "data/raw"

MISSING = {
    "320000": "한울반도체",
    "323350": "다원넥스뷰",
    "429270": "시지트로닉스",
    "067170": "오텍",
    "048770": "TPC로보틱스",
    "396270": "넥스트칩",
    "076610": "해성옵틱스",
    "321260": "프로이천",
    "452200": "민테크",
    "013000": "세우글로벌",
    "192410": "오늘이엔엠",
    "222040": "코스맥스엔비티",
    "005870": "휴니드",
    "065450": "빅텍",
    "095270": "웨이브일렉트로",
    "448710": "코츠테크놀로지",
    "065170": "비엘팜텍",
    "215090": "솔디펜스",
    "221840": "하이즈항공",
    "020760": "일진디스플",
    "024740": "한일단조",
    "014970": "삼륭물산",
}

existing = set(f.replace(".parquet", "") for f in os.listdir(RAW_DIR) if f.endswith(".parquet"))
print(f"기존 parquet: {len(existing)}개")

START = "20190101"
END = "20260508"

success = []
failed = []
skipped = []

for code, name in MISSING.items():
    if code in existing:
        print(f"  SKIP {code} {name} (이미 존재)")
        skipped.append((code, name))
        continue

    try:
        df = pykrx_stock.get_market_ohlcv_by_date(START, END, code)
        time.sleep(0.5)

        if df is None or len(df) == 0:
            print(f"  FAIL {code} {name} (데이터 없음)")
            failed.append((code, name, "no data"))
            continue

        col_map = {}
        for c in df.columns:
            if "시가" in c: col_map[c] = "open"
            elif "고가" in c: col_map[c] = "high"
            elif "저가" in c: col_map[c] = "low"
            elif "종가" in c: col_map[c] = "close"
            elif "거래량" in c: col_map[c] = "volume"
            elif "거래대금" in c: col_map[c] = "trading_value"
            elif "등락률" in c: col_map[c] = "price_change"

        if col_map:
            df = df.rename(columns=col_map)

        required = ["open", "high", "low", "close", "volume"]
        has_all = all(c in df.columns for c in required)

        if not has_all:
            print(f"  FAIL {code} {name} (컬럼 부족: {list(df.columns)})")
            failed.append((code, name, "missing cols"))
            continue

        df = df[df["close"] > 0]

        if len(df) < 10:
            print(f"  FAIL {code} {name} (데이터 {len(df)}행)")
            failed.append((code, name, f"{len(df)} rows"))
            continue

        path = f"{RAW_DIR}/{code}.parquet"
        df.to_parquet(path)
        d0 = df.index[0].strftime("%Y-%m-%d")
        d1 = df.index[-1].strftime("%Y-%m-%d")
        print(f"  OK   {code} {name} ({len(df)}행, {d0}~{d1})")
        success.append((code, name, len(df)))

    except Exception as e:
        msg = str(e)[:80]
        print(f"  ERR  {code} {name}: {msg}")
        failed.append((code, name, msg))

    time.sleep(0.3)

print(f"\n=== 결과 ===")
print(f"  성공: {len(success)}개")
print(f"  실패: {len(failed)}개")
print(f"  스킵(이미존재): {len(skipped)}개")

total = len([f for f in os.listdir(RAW_DIR) if f.endswith(".parquet")])
print(f"  총 parquet: {total}개")

if failed:
    print("\n실패 목록:")
    for code, name, reason in failed:
        print(f"  {code} {name}: {reason}")
