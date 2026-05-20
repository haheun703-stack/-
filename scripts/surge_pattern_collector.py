"""
급등주 패턴 자동 수집 + 다차원 분석
=========================================
목적
----
퐝가님(5/20 밤) 지시 — "봇이 직접 매일 +10%↑ 종목을 자동 수집하고
시그널/수급/가격/타이밍/체결강도 패턴을 학습해야 한다"

운영
----
- 매일 18:30 cron (theme_pickup 18:00 직후)
- 입력: VPS /home/ubuntu/jgis/stock_data_daily/*.csv (39컬럼)
- 출력: data/surge_pattern.db (SQLite 3테이블)
- 봇 매수 로직 무변경, 학습/관찰 전용

테이블 구조
-----------
1. surge_stocks       — 당일 +10%↑ 종목 리스트
2. surge_features     — 각 종목의 다차원 특성 (시그널+수급+가격+추세)
3. surge_followup     — D+1, D+3, D+5, D+10 사후 추적 (점진 갱신)

5/27 1주차 회고 시
- 누적 데이터로 패턴 클러스터링
- 어떤 조합이 상한가/+10%를 만들었는지 통계 추출
- 단타/스윙 보유 기간 룰 도출

CLI
---
  python3.11 scripts/surge_pattern_collector.py             # 오늘 데이터
  python3.11 scripts/surge_pattern_collector.py --date 20260520
  python3.11 scripts/surge_pattern_collector.py --dry-run   # 파일 미저장
  python3.11 scripts/surge_pattern_collector.py --followup  # 사후 추적만 갱신
"""

import argparse
import datetime
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
JGIS_BASE = Path(os.environ.get("JGIS_BASE", "/home/ubuntu/jgis/stock_data_daily"))
QUANTUM_DATA = Path(os.environ.get("QUANTUM_DATA", "/home/ubuntu/quantum-master/data"))
DB_FILE = QUANTUM_DATA / "surge_pattern.db"
INVESTOR_DB = QUANTUM_DATA / "investor_flow" / "investor_daily.db"

# ─────────────────────────────────────────────
# 임계값
# ─────────────────────────────────────────────
SURGE_PCT = 10.0   # D-1 등락률 ≥ +10%
LIMIT_PCT = 29.5   # 상한가 (한국 시장 ±30%)
STRONG_PCT = 20.0  # 강세 ≥ +20%
CHG_MAX = 35.0     # 상한선 (한국 ±30% + 여유) — 초과 시 액면분할/감자/권리락 이상치로 판단
MIN_VOL = 1000     # 최소 거래량 (관리종목 노이즈 제거)


# ─────────────────────────────────────────────
# DB 스키마
# ─────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS surge_stocks (
    date         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    name         TEXT,
    close        REAL,
    prev_close   REAL,
    chg_pct      REAL,
    volume       INTEGER,
    vol_ratio    REAL,
    surge_type   TEXT,           -- LIMIT_UP / STRONG / NORMAL
    inserted_at  TEXT,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS surge_features (
    date         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    -- 시그널 (jgis 39컬럼 활용)
    ma5          REAL, ma20 REAL, ma60 REAL, ma120 REAL,
    ma5_dist     REAL,            -- (close - MA5) / MA5 * 100
    ma20_dist    REAL,
    ma60_dist    REAL,
    rsi          REAL,
    adx          REAL,
    plus_di      REAL,
    minus_di     REAL,
    atr          REAL,
    obv          REAL,
    obv_5d_chg   REAL,
    vol_ratio    REAL,
    regime_tag   TEXT,
    macd         REAL,
    macd_signal  REAL,
    bb_pos       REAL,            -- (close - Lower) / (Upper - Lower)
    -- 수급
    foreign_1d   INTEGER,
    foreign_5d   INTEGER,
    foreign_20d  INTEGER,
    inst_1d      INTEGER,
    inst_5d      INTEGER,
    inst_20d     INTEGER,
    pension_5d   REAL,            -- 연기금 5일 합 (억원)
    finance_5d   REAL,            -- 금융투자 5일 합 (억원)
    -- 가격 추세
    ret_1d       REAL,
    ret_5d       REAL,
    ret_20d      REAL,
    ret_60d      REAL,
    high_60d     REAL,
    low_60d      REAL,
    dist_high60  REAL,            -- 60일 신고가 거리 %
    -- 메타
    market_cap   REAL,
    inserted_at  TEXT,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS surge_followup (
    surge_date     TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    d1_close       REAL, d1_chg_pct  REAL,
    d3_close       REAL, d3_chg_pct  REAL,
    d5_close       REAL, d5_chg_pct  REAL,
    d10_close      REAL, d10_chg_pct REAL,
    max_close_10d  REAL, max_chg_10d REAL,
    min_close_10d  REAL, min_chg_10d REAL,
    updated_at     TEXT,
    PRIMARY KEY (surge_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_features_date ON surge_features(date);
CREATE INDEX IF NOT EXISTS idx_followup_date ON surge_followup(surge_date);
"""


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def parse_filename_ticker(filename: str) -> tuple[str, str] | None:
    """녹십자_006280.csv → ('녹십자', '006280')"""
    m = re.match(r"^(.+)_(\d{6})\.csv$", filename)
    return (m.group(1), m.group(2)) if m else None


def load_ohlcv(ticker_csv: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(ticker_csv)
        if "Date" not in df.columns or len(df) < 25:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return None


def date_to_str(d) -> str:
    """datetime/Timestamp/str → '20260520'"""
    if isinstance(d, str):
        return d.replace("-", "")[:8]
    return d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)[:10].replace("-", "")


# ─────────────────────────────────────────────
# 1) 급등주 추출
# ─────────────────────────────────────────────
def extract_surge_stocks(target_date: str) -> list[dict]:
    """target_date(YYYYMMDD)의 +10%↑ 종목 추출"""
    surges = []
    target_dt = pd.Timestamp(f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}")

    files = list(JGIS_BASE.glob("*.csv"))
    print(f"[surge] 전수 스캔: {len(files)} 종목")

    for fp in files:
        parsed = parse_filename_ticker(fp.name)
        if not parsed:
            continue
        name, ticker = parsed
        df = load_ohlcv(fp)
        if df is None:
            continue
        match = df[df["Date"] == target_dt]
        if len(match) == 0:
            continue
        idx = match.index[0]
        if idx == 0:
            continue
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev_close = float(prev["Close"])
        close = float(row["Close"])
        if prev_close <= 0:
            continue
        chg = (close / prev_close - 1) * 100
        vol = int(row.get("Volume", 0) or 0)
        if vol < MIN_VOL:
            continue
        if chg < SURGE_PCT:
            continue
        if chg > CHG_MAX:
            # 액면분할/감자/권리락 이상치 — 5/20 dry-run 국일제지 +791%, 자연과환경 +711% 등
            continue

        if chg >= LIMIT_PCT:
            surge_type = "LIMIT_UP"
        elif chg >= STRONG_PCT:
            surge_type = "STRONG"
        else:
            surge_type = "NORMAL"

        surges.append({
            "date": target_date,
            "ticker": ticker,
            "name": name,
            "close": close,
            "prev_close": prev_close,
            "chg_pct": round(chg, 2),
            "volume": vol,
            "vol_ratio": round(float(row.get("Vol_Ratio", 0) or 0), 2),
            "surge_type": surge_type,
        })

    surges.sort(key=lambda x: -x["chg_pct"])
    return surges


# ─────────────────────────────────────────────
# 2) 다차원 특성 추출
# ─────────────────────────────────────────────
def safe(v, default=None):
    try:
        if pd.isna(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def extract_features(target_date: str, surges: list[dict]) -> list[dict]:
    features = []
    target_dt = pd.Timestamp(f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}")

    # 수급 보강 (investor_daily.db: 연기금/금투)
    investor_lookup = {}
    if INVESTOR_DB.exists():
        try:
            conn = sqlite3.connect(str(INVESTOR_DB))
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT date FROM investor_daily "
                "WHERE date <= ? ORDER BY date DESC LIMIT 5",
                (target_date,),
            )
            recent_dates = [r[0] for r in cur.fetchall()]
            if recent_dates:
                placeholders = ",".join("?" * len(recent_dates))
                cur.execute(
                    f"SELECT ticker, investor, SUM(net_val) FROM investor_daily "
                    f"WHERE date IN ({placeholders}) GROUP BY ticker, investor",
                    recent_dates,
                )
                for tkr, inv, total in cur.fetchall():
                    investor_lookup.setdefault(tkr, {})[inv] = total / 1e8  # 원 → 억원
            conn.close()
        except Exception as e:
            print(f"[warn] investor_daily.db 로드 실패: {e}")

    for s in surges:
        ticker = s["ticker"]
        fp_list = list(JGIS_BASE.glob(f"*_{ticker}.csv"))
        if not fp_list:
            continue
        df = load_ohlcv(fp_list[0])
        if df is None:
            continue
        match = df[df["Date"] == target_dt]
        if len(match) == 0:
            continue
        idx = match.index[0]
        row = df.iloc[idx]
        close = float(row["Close"])

        def get(col, idx_offset=0):
            i = idx + idx_offset
            if i < 0 or i >= len(df):
                return None
            return safe(df.iloc[i].get(col))

        # MA 이격
        ma5 = get("MA5"); ma20 = get("MA20"); ma60 = get("MA60"); ma120 = get("MA120")
        ma5_dist = ((close / ma5 - 1) * 100) if ma5 and ma5 > 0 else None
        ma20_dist = ((close / ma20 - 1) * 100) if ma20 and ma20 > 0 else None
        ma60_dist = ((close / ma60 - 1) * 100) if ma60 and ma60 > 0 else None

        # OBV 5일 변화
        obv_now = get("OBV"); obv_5d_ago = get("OBV", -5)
        obv_5d_chg = (obv_now - obv_5d_ago) if (obv_now is not None and obv_5d_ago is not None) else None

        # 볼린저 위치
        upper = get("Upper_Band"); lower = get("Lower_Band")
        bb_pos = ((close - lower) / (upper - lower)) if (upper and lower and upper > lower) else None

        # 수급 (jgis Foreign_Net, Inst_Net 일별)
        def sum_n(col, n):
            start = max(0, idx - n + 1)
            try:
                return int(df.iloc[start:idx + 1][col].sum())
            except Exception:
                return None

        foreign_1d = int(safe(row.get("Foreign_Net"), 0) or 0)
        foreign_5d = sum_n("Foreign_Net", 5)
        foreign_20d = sum_n("Foreign_Net", 20)
        inst_1d = int(safe(row.get("Inst_Net"), 0) or 0)
        inst_5d = sum_n("Inst_Net", 5)
        inst_20d = sum_n("Inst_Net", 20)

        inv = investor_lookup.get(ticker, {})
        pension_5d = round(inv.get("연기금", 0), 2)
        finance_5d = round(inv.get("금융투자", 0), 2)

        # 가격 추세
        def ret_n(n):
            i = idx - n
            if i < 0:
                return None
            past = float(df.iloc[i]["Close"])
            return round((close / past - 1) * 100, 2) if past > 0 else None

        # 60일 신고가/저가
        win60 = df.iloc[max(0, idx - 59):idx + 1]
        high_60d = float(win60["High"].max()) if len(win60) > 0 else None
        low_60d = float(win60["Low"].min()) if len(win60) > 0 else None
        dist_high60 = ((close / high_60d - 1) * 100) if high_60d and high_60d > 0 else None

        features.append({
            "date": target_date,
            "ticker": ticker,
            "ma5": ma5, "ma20": ma20, "ma60": ma60, "ma120": ma120,
            "ma5_dist": round(ma5_dist, 2) if ma5_dist is not None else None,
            "ma20_dist": round(ma20_dist, 2) if ma20_dist is not None else None,
            "ma60_dist": round(ma60_dist, 2) if ma60_dist is not None else None,
            "rsi": round(safe(row.get("RSI"), 0) or 0, 2),
            "adx": round(safe(row.get("ADX"), 0) or 0, 2),
            "plus_di": round(safe(row.get("Plus_DI"), 0) or 0, 2),
            "minus_di": round(safe(row.get("Minus_DI"), 0) or 0, 2),
            "atr": round(safe(row.get("ATR"), 0) or 0, 2),
            "obv": safe(row.get("OBV")),
            "obv_5d_chg": obv_5d_chg,
            "vol_ratio": round(safe(row.get("Vol_Ratio"), 0) or 0, 2),
            "regime_tag": str(safe(row.get("Regime_Tag"), "")),
            "macd": round(safe(row.get("MACD"), 0) or 0, 4),
            "macd_signal": round(safe(row.get("MACD_Signal"), 0) or 0, 4),
            "bb_pos": round(bb_pos, 3) if bb_pos is not None else None,
            "foreign_1d": foreign_1d, "foreign_5d": foreign_5d, "foreign_20d": foreign_20d,
            "inst_1d": inst_1d, "inst_5d": inst_5d, "inst_20d": inst_20d,
            "pension_5d": pension_5d, "finance_5d": finance_5d,
            "ret_1d": ret_n(1), "ret_5d": ret_n(5),
            "ret_20d": ret_n(20), "ret_60d": ret_n(60),
            "high_60d": high_60d, "low_60d": low_60d,
            "dist_high60": round(dist_high60, 2) if dist_high60 is not None else None,
            "market_cap": safe(row.get("MarketCap")),
        })

    return features


# ─────────────────────────────────────────────
# 3) 사후 추적 갱신
# ─────────────────────────────────────────────
def update_followup(target_date: str):
    """과거 급등 종목들의 D+1, D+3, D+5, D+10 사후 추적"""
    conn = sqlite3.connect(str(DB_FILE))
    cur = conn.cursor()
    cur.execute("SELECT date, ticker FROM surge_stocks WHERE date <= ? ORDER BY date DESC LIMIT 200", (target_date,))
    pairs = cur.fetchall()
    conn.close()

    target_dt = pd.Timestamp(f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}")
    updated = 0

    for surge_date, ticker in pairs:
        fp_list = list(JGIS_BASE.glob(f"*_{ticker}.csv"))
        if not fp_list:
            continue
        df = load_ohlcv(fp_list[0])
        if df is None:
            continue
        surge_dt = pd.Timestamp(f"{surge_date[:4]}-{surge_date[4:6]}-{surge_date[6:8]}")
        match = df[df["Date"] == surge_dt]
        if len(match) == 0:
            continue
        idx = match.index[0]
        surge_close = float(df.iloc[idx]["Close"])

        result = {"surge_date": surge_date, "ticker": ticker}
        for n_label, n in [("d1", 1), ("d3", 3), ("d5", 5), ("d10", 10)]:
            future_idx = idx + n
            if future_idx < len(df) and df.iloc[future_idx]["Date"] <= target_dt:
                fc = float(df.iloc[future_idx]["Close"])
                result[f"{n_label}_close"] = fc
                result[f"{n_label}_chg_pct"] = round((fc / surge_close - 1) * 100, 2)
            else:
                result[f"{n_label}_close"] = None
                result[f"{n_label}_chg_pct"] = None

        # max/min in next 10 days
        future_win = df.iloc[idx + 1:idx + 11]
        future_win = future_win[future_win["Date"] <= target_dt]
        if len(future_win) > 0:
            mx = float(future_win["Close"].max()); mn = float(future_win["Close"].min())
            result["max_close_10d"] = mx
            result["max_chg_10d"] = round((mx / surge_close - 1) * 100, 2)
            result["min_close_10d"] = mn
            result["min_chg_10d"] = round((mn / surge_close - 1) * 100, 2)
        else:
            result["max_close_10d"] = result["max_chg_10d"] = None
            result["min_close_10d"] = result["min_chg_10d"] = None

        result["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

        conn2 = sqlite3.connect(str(DB_FILE))
        conn2.execute("""
            INSERT INTO surge_followup
            (surge_date, ticker, d1_close, d1_chg_pct, d3_close, d3_chg_pct,
             d5_close, d5_chg_pct, d10_close, d10_chg_pct,
             max_close_10d, max_chg_10d, min_close_10d, min_chg_10d, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(surge_date, ticker) DO UPDATE SET
              d1_close=excluded.d1_close, d1_chg_pct=excluded.d1_chg_pct,
              d3_close=excluded.d3_close, d3_chg_pct=excluded.d3_chg_pct,
              d5_close=excluded.d5_close, d5_chg_pct=excluded.d5_chg_pct,
              d10_close=excluded.d10_close, d10_chg_pct=excluded.d10_chg_pct,
              max_close_10d=excluded.max_close_10d, max_chg_10d=excluded.max_chg_10d,
              min_close_10d=excluded.min_close_10d, min_chg_10d=excluded.min_chg_10d,
              updated_at=excluded.updated_at
        """, (result["surge_date"], result["ticker"],
              result["d1_close"], result["d1_chg_pct"],
              result["d3_close"], result["d3_chg_pct"],
              result["d5_close"], result["d5_chg_pct"],
              result["d10_close"], result["d10_chg_pct"],
              result["max_close_10d"], result["max_chg_10d"],
              result["min_close_10d"], result["min_chg_10d"],
              result["updated_at"]))
        conn2.commit()
        conn2.close()
        updated += 1

    return updated


# ─────────────────────────────────────────────
# DB 저장
# ─────────────────────────────────────────────
def init_db():
    QUANTUM_DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_surges(surges: list[dict]):
    if not surges:
        return
    conn = sqlite3.connect(str(DB_FILE))
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for s in surges:
        conn.execute("""
            INSERT INTO surge_stocks (date, ticker, name, close, prev_close, chg_pct,
                                       volume, vol_ratio, surge_type, inserted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(date, ticker) DO UPDATE SET
              close=excluded.close, prev_close=excluded.prev_close,
              chg_pct=excluded.chg_pct, volume=excluded.volume,
              vol_ratio=excluded.vol_ratio, surge_type=excluded.surge_type,
              inserted_at=excluded.inserted_at
        """, (s["date"], s["ticker"], s["name"], s["close"], s["prev_close"],
              s["chg_pct"], s["volume"], s["vol_ratio"], s["surge_type"], now))
    conn.commit()
    conn.close()


def save_features(features: list[dict]):
    if not features:
        return
    conn = sqlite3.connect(str(DB_FILE))
    now = datetime.datetime.now().isoformat(timespec="seconds")
    cols = [c for c in features[0].keys()]
    placeholders = ",".join("?" * (len(cols) + 1))
    col_list = ",".join(cols) + ",inserted_at"
    update_clause = ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("date", "ticker"))

    for f in features:
        values = [f[c] for c in cols] + [now]
        conn.execute(f"""
            INSERT INTO surge_features ({col_list}) VALUES ({placeholders})
            ON CONFLICT(date, ticker) DO UPDATE SET {update_clause}, inserted_at=excluded.inserted_at
        """, values)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--followup", action="store_true", help="사후 추적만 갱신")
    args = parser.parse_args()

    target_date = args.date or datetime.date.today().strftime("%Y%m%d")
    print(f"[surge] date={target_date} jgis={JGIS_BASE} db={DB_FILE}")
    init_db()

    if args.followup:
        n = update_followup(target_date)
        print(f"[followup] {n}건 갱신")
        return

    surges = extract_surge_stocks(target_date)
    print(f"[surge] +{SURGE_PCT}%↑ 종목: {len(surges)}건")

    # 요약
    if surges:
        limit_up = sum(1 for s in surges if s["surge_type"] == "LIMIT_UP")
        strong = sum(1 for s in surges if s["surge_type"] == "STRONG")
        normal = sum(1 for s in surges if s["surge_type"] == "NORMAL")
        print(f"  상한가(≥{LIMIT_PCT}%): {limit_up} / 강세({STRONG_PCT}~{LIMIT_PCT}%): {strong} / 보통({SURGE_PCT}~{STRONG_PCT}%): {normal}")
        print(f"\n  TOP 10 등락률:")
        for s in surges[:10]:
            print(f"    {s['name']:<14} {s['ticker']}  {s['chg_pct']:>+6.2f}%  거래량 {s['volume']:>12,}  Vol배 {s['vol_ratio']:>5.2f}  [{s['surge_type']}]")

    features = extract_features(target_date, surges)
    print(f"\n[features] {len(features)}건 특성 추출")

    if args.dry_run:
        print("[dry-run] 파일 미저장")
        return

    save_surges(surges)
    save_features(features)
    print(f"[저장] {DB_FILE}")

    # 사후 추적도 함께 갱신
    n = update_followup(target_date)
    print(f"[followup] {n}건 갱신")


if __name__ == "__main__":
    main()
