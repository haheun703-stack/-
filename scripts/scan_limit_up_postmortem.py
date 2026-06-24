#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
상한가 사후학습 트랙 — limit-up postmortem

매일 전종목 상한가/급등 종목을 수집하고, 각 종목에 대해
  - 우리 유니버스 포함 여부 (data/processed/{code}.parquet 존재)
  - D-1(상한가 직전일) 선행지표 (RSI/ADX/거래량응축·급증/외인·기관 연속매수)
  - 수급 분류 (외인선행 / 기관선행 / 쌍끌이선행 / 무수급(테마·개인))
  - 테마 매칭 (theme_pickup_log.csv)
를 태깅하여 학습 DB(SQLite) + 일자별 JSON으로 적재한다.

목적: "수급이 선행했는데 우리가 추천에서 놓친 상한가" = 개선의 금맥을 매일 축적.

운영(BAT-D 후반, parquet 적재 [2]단계 이후 실행):
    set PYTHONPATH=D:\\sub-agent-project_퀀트봇
    python -u -X utf8 scripts/scan_limit_up_postmortem.py
검증(서버 /tmp ad-hoc):
    QM_ROOT=~/quantum-master ./venv/bin/python3.11 -u /tmp/scan_limit_up_postmortem.py

TODO(후속): 실적(흑자전환·영업익증감) 태깅, DART 공시/뉴스 촉매 매칭.
"""
import sys
import os

# ── 경로 안전장치 (PYTHONPATH/QM_ROOT 우선, 없으면 파일 기준) ──────────────
ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

import json
import sqlite3
import argparse
import datetime as dt
from collections import Counter

import pandas as pd

PROCESSED = os.path.join(ROOT, "data", "processed")
OUTDIR = os.path.join(ROOT, "data", "limit_up_postmortem")
DBPATH = os.path.join(OUTDIR, "limit_up.db")
THEME_CSV = os.path.join(ROOT, "data", "theme_pickup_log.csv")

# 급등 하한(상한가급) ~ 데이터오염(액면분할/병합) 상한 제외
SURGE_LO = 25.0
SURGE_HI = 31.5


def _f(v, nd=0):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return round(float(v), nd)
    except Exception:
        return None


def _i(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0
        return int(v)
    except Exception:
        return 0


def fetch_surges():
    """FDR StockListing(KRX) 당일 snapshot에서 상한가/급등 종목 추출."""
    import FinanceDataReader as fdr
    lst = fdr.StockListing("KRX")
    if "Code" not in lst.columns or "ChagesRatio" not in lst.columns:
        raise RuntimeError(f"StockListing 컬럼 이상: {list(lst.columns)}")
    lst["Code"] = lst["Code"].astype(str).str.zfill(6)
    m = (lst["ChagesRatio"] >= SURGE_LO) & (lst["ChagesRatio"] <= SURGE_HI)
    cols = [c for c in ["Code", "Name", "Market", "Close", "ChagesRatio", "Volume", "Amount", "Marcap"] if c in lst.columns]
    return lst.loc[m, cols].sort_values("ChagesRatio", ascending=False).reset_index(drop=True)


def d1_signal(code, target_date=None):
    """유니버스 내 종목의 상한가 직전일(D-1) 선행지표. 유니버스 밖이면 None.

    target_date(상한가 발생일) 지정 시 parquet에서 그 행을 찾아 직전 거래일을 D-1로.
    parquet에 발생일 미반영(stale)이면 {"stale":1}만 반환 → 수급분류 '미확정'.
    """
    p = os.path.join(PROCESSED, f"{code}.parquet")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
        if len(df) < 2:
            return None
        if target_date:
            try:
                loc = int(df.index.get_indexer([pd.Timestamp(target_date)])[0])
            except Exception:
                loc = -1
            if loc <= 0:  # 발생일 미반영(stale) 또는 첫행 → D-1 확정 불가
                return {"stale": 1, "parquet_last": str(df.index[-1])[:10]}
            d1 = df.iloc[loc - 1]
        else:
            d1 = df.iloc[-2]  # 폴백(--date 없을 때): 마지막행=상한가일 가정
        return {
            "rsi": _f(d1.get("rsi_14")),
            "adx": _f(d1.get("adx_14")),
            "volsurge": _f(d1.get("volume_surge_ratio"), 2),
            "volcontract": _f(d1.get("volume_contraction_ratio"), 2),
            "foreign_streak": _i(d1.get("foreign_consecutive_buy")),
            "inst_streak": _i(d1.get("inst_consecutive_buy")),
            "parquet_last": str(df.index[-1])[:10],
            "stale": 0,
        }
    except Exception:
        return None


def classify(sig):
    if sig is None:
        return "유니버스밖(미추적)"
    if sig.get("stale"):
        return "유니버스내(parquet미반영)"
    f = sig.get("foreign_streak", 0) or 0
    i = sig.get("inst_streak", 0) or 0
    if f > 0 and i > 0:
        return "쌍끌이선행"
    if f > 0:
        return "외인선행"
    if i > 0:
        return "기관선행"
    return "무수급(테마/개인)"


def load_theme_codes():
    if not os.path.exists(THEME_CSV):
        return set()
    try:
        t = pd.read_csv(THEME_CSV, dtype=str)
        for c in t.columns:
            if "code" in c.lower() or "코드" in c or "ticker" in c.lower():
                return set(t[c].dropna().astype(str).str.zfill(6))
    except Exception:
        pass
    return set()


def ensure_db():
    os.makedirs(OUTDIR, exist_ok=True)
    con = sqlite3.connect(DBPATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS limit_up(
            date TEXT, code TEXT, name TEXT, market TEXT, change_pct REAL,
            close REAL, marcap REAL, in_universe INTEGER, supply_class TEXT,
            d1_rsi REAL, d1_adx REAL, d1_volsurge REAL, d1_volcontract REAL,
            d1_foreign_streak INTEGER, d1_inst_streak INTEGER, theme_flag INTEGER,
            PRIMARY KEY(date, code))"""
    )
    con.commit()
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="상한가 발생일 (기본: 오늘). D-1 정합성 기준")
    args = ap.parse_args()

    # 상한가 snapshot은 '오늘'이 기본 — 날짜 라벨/PK 결정적(종목순서·parquet신선도 비종속)
    run_date = args.date or dt.date.today().isoformat()

    try:
        surges = fetch_surges()
    except Exception as e:
        print(f"[fetch_surges] FDR 수집 실패 — 그날 skip: {e}")
        return
    theme_codes = load_theme_codes()

    recs, rows = [], []
    for _, r in surges.iterrows():
        code = r["Code"]
        sig = d1_signal(code, run_date)
        in_uni = 1 if sig is not None else 0
        sc = classify(sig)
        theme = 1 if code in theme_codes else 0
        date = run_date
        recs.append({
            "date": date, "code": code, "name": r.get("Name"), "market": r.get("Market"),
            "change_pct": _f(r.get("ChagesRatio"), 2), "close": _f(r.get("Close")),
            "marcap": _f(r.get("Marcap")), "in_universe": in_uni,
            "supply_class": sc, "theme_flag": theme, "d1": sig,
        })
        rows.append((
            date, code, r.get("Name"), r.get("Market"), _f(r.get("ChagesRatio"), 2),
            _f(r.get("Close")), _f(r.get("Marcap")), in_uni, sc,
            (sig or {}).get("rsi"), (sig or {}).get("adx"),
            (sig or {}).get("volsurge"), (sig or {}).get("volcontract"),
            (sig or {}).get("foreign_streak"), (sig or {}).get("inst_streak"), theme,
        ))

    con = ensure_db()
    con.executemany(
        "INSERT OR REPLACE INTO limit_up VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()

    rundate = run_date
    os.makedirs(OUTDIR, exist_ok=True)
    jpath = os.path.join(OUTDIR, f"{rundate}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    n = len(recs)
    uni = sum(r["in_universe"] for r in recs)
    catchable = sum(1 for r in recs if r["supply_class"] in ("외인선행", "기관선행", "쌍끌이선행"))
    print(f"=== 상한가 사후학습 [{rundate}] ===")
    print(f"상한가/급등(+{SURGE_LO}~{SURGE_HI}%): {n}종목 | 유니버스내 {uni} | 유니버스밖 {n - uni}")
    if n:
        print(f"D-1 수급선행(=우리가 잡을 수 있던 유형): {catchable}종목 ({catchable * 100 // n}%)")
    for k, v in Counter(r["supply_class"] for r in recs).most_common():
        print(f"   {k}: {v}")
    print(f"적재: {jpath}")
    print(f"      {DBPATH}")
    print("--- 상위 15 ---")
    for r in recs[:15]:
        d1 = r["d1"]
        sig = (f"RSI{d1['rsi']} ADX{d1['adx']} 외인{d1['foreign_streak']} 기관{d1['inst_streak']}"
               if d1 else "유니버스밖")
        print(f"  {r['code']} {str(r['name'])[:11]:<11} +{r['change_pct']}% [{r['supply_class']}] {sig}")


if __name__ == "__main__":
    main()
