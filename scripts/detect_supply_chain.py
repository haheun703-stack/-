#!/usr/bin/env python3
"""수급 바톤터치 감지 (P1) — 주도 주체 교체 패턴 탐지

한국장 핵심: 외인 매도 → 기관 인수 = 바톤터치 (가격 받침 확인)

바톤터치 유형:
- 3단_릴레이: 외인→기관 교체 + 연기금 매수 (35점)
- 외인기관_교체: 외인 매도 전환 + 기관 매수 인수 (25점)
- 기관외인_교체: 기관 매도 전환 + 외인 매수 인수 (25점)
- 연기금_인수: 전일 주도 매도 → 연기금 >=20억 매수 (20점)
- 스마트머니_합류: 금투 >=30억 or 기타법인 >=20억 은밀 매집 (15점)

기술적 보조:
- 가격 받침: 전일 종가 대비 -3% 이내 (+5)
- 거래량 폭발: 5일 평균 x2 (+5)
- RSI 적정: 30~70 (+3)
- 수급 점증: 3일+ 같은 방향 (+5~10)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
OUTPUT_DIR = PROJECT_ROOT / "data"

# 주도 주체 매핑 (DB 컬럼명 → 표시명)
ACTOR_LABELS = {
    "fgn": "외인",
    "inst": "기관",
    "pension": "연기금",
    "finance": "금투",
    "corp": "기타법인",
}

# 주도 주체 판별 대상 (개인 제외)
LEADER_COLS = ["fgn", "inst", "pension", "finance", "corp"]


# ─────────────────────────────────────────────
# 1. 데이터 로더
# ─────────────────────────────────────────────

def load_supply_from_db(lookback_days: int = 10) -> pd.DataFrame:
    """DB에서 최근 N거래일 6유형+개인 수급 피벗 로드 (억원 단위)."""
    if not DB_PATH.exists():
        logger.warning("수급 DB 없음: %s", DB_PATH)
        return pd.DataFrame()

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    dates_df = pd.read_sql_query(
        f"SELECT DISTINCT date FROM investor_daily ORDER BY date DESC LIMIT {lookback_days}",
        conn,
    )
    if dates_df.empty:
        conn.close()
        return pd.DataFrame()

    min_date = dates_df["date"].min()

    df = pd.read_sql_query(
        """
        SELECT date, ticker, name,
               SUM(CASE WHEN investor = '외국인'   THEN net_val ELSE 0 END) AS fgn,
               SUM(CASE WHEN investor = '기관합계' THEN net_val ELSE 0 END) AS inst,
               SUM(CASE WHEN investor = '기타법인' THEN net_val ELSE 0 END) AS corp,
               SUM(CASE WHEN investor = '연기금'   THEN net_val ELSE 0 END) AS pension,
               SUM(CASE WHEN investor = '금융투자' THEN net_val ELSE 0 END) AS finance,
               SUM(CASE WHEN investor = '개인'     THEN net_val ELSE 0 END) AS retail
        FROM investor_daily
        WHERE date >= ?
        GROUP BY date, ticker
        """,
        conn,
        params=[min_date],
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # 원 → 억원
    for col in ["fgn", "inst", "corp", "pension", "finance", "retail"]:
        df[col] = (df[col] / 1e8).round(1)

    logger.info(
        "수급 DB: %d행 / %d종목 / %d거래일 (%s~)",
        len(df), df["ticker"].nunique(), df["date"].nunique(), min_date,
    )
    return df


def load_price_data(ticker: str) -> pd.DataFrame | None:
    """CSV에서 종가 + 기술적 지표 계산."""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return None
    try:
        df = pd.read_csv(csvs[0], header=0)
        if len(df.columns) < 6 or len(df) < 30:
            return None
        df = df.iloc[:, :6]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = df["date"].astype(str).str.replace("-", "")
        df = df.sort_values("date").reset_index(drop=True)

        # 기술적 지표
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma20_dev"] = (df["close"] / df["ma20"] - 1) * 100
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)

        # RSI 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        return df.set_index("date")
    except Exception:
        return None


# ─────────────────────────────────────────────
# 2. 바톤터치 감지
# ─────────────────────────────────────────────

def find_leader(row: pd.Series) -> tuple[str, float]:
    """5유형 중 순매수 최대 주체 반환. (컬럼명, 금액)."""
    vals = {col: float(row[col]) for col in LEADER_COLS}
    leader = max(vals, key=vals.get)
    return leader, vals[leader]


def classify_baton_touch(
    prev_row: pd.Series, curr_row: pd.Series
) -> tuple[str, str, str, int]:
    """전일/당일 수급 비교 → (유형명, from_leader, to_leader, 기본점수).

    바톤터치 조건:
    1. 전일 주도 주체가 유의미한 매수 (>=20억)
    2. 당일 해당 주체가 매도 전환 (<0)
    3. 당일 새로운 주체가 유의미한 매수 (>=20억)
    """
    prev_leader, prev_amount = find_leader(prev_row)
    if prev_amount < 20:
        return "", "", "", 0

    # 당일: 전일 주도가 매도 전환했는가?
    curr_prev_leader_val = float(curr_row[prev_leader])
    if curr_prev_leader_val >= 0:
        return "", "", "", 0

    # 당일: 새 주도 주체
    curr_leader, curr_amount = find_leader(curr_row)
    if curr_amount < 20 or curr_leader == prev_leader:
        return "", "", "", 0

    # 바톤터치 확인 — 유형 분류
    fgn_t = float(curr_row["fgn"])
    inst_t = float(curr_row["inst"])
    pension_t = float(curr_row["pension"])
    finance_t = float(curr_row["finance"])
    corp_t = float(curr_row["corp"])

    from_label = ACTOR_LABELS.get(prev_leader, prev_leader)
    to_label = ACTOR_LABELS.get(curr_leader, curr_leader)

    # 3단 릴레이: 외인→기관 + 연기금 매수
    if prev_leader == "fgn" and curr_leader == "inst" and pension_t > 0:
        return "3단_릴레이", from_label, to_label, 35

    # 외인→기관 교체
    if prev_leader == "fgn" and curr_leader in ("inst", "pension", "finance"):
        return "외인기관_교체", from_label, to_label, 25

    # 기관→외인 교체
    if prev_leader in ("inst", "pension", "finance") and curr_leader == "fgn":
        return "기관외인_교체", from_label, to_label, 25

    # 연기금 인수
    if curr_leader == "pension" and pension_t >= 20:
        return "연기금_인수", from_label, to_label, 20

    # 스마트머니 합류
    if curr_leader in ("finance", "corp"):
        if finance_t >= 30 or corp_t >= 20:
            return "스마트머니_합류", from_label, to_label, 15

    # 일반 교체
    return "일반_교체", from_label, to_label, 10


def calc_streak_bonus(ticker_df: pd.DataFrame) -> int:
    """최근 수급 점증(3일+) → 가산점."""
    if len(ticker_df) < 3:
        return 0

    recent = ticker_df.tail(5)
    combined = (recent["fgn"] + recent["inst"]).values

    streak = 0
    for i in range(len(combined) - 1, 0, -1):
        if combined[i] > 0 and combined[i] >= combined[i - 1]:
            streak += 1
        else:
            break

    if streak >= 3:
        return 10
    if streak >= 2:
        return 5
    return 0


# ─────────────────────────────────────────────
# 3. 메인 스캔
# ─────────────────────────────────────────────

def detect_supply_chain(
    lookback: int = 10,
    top_n: int = 30,
) -> list[dict]:
    """바톤터치 감지. 종목 리스트 반환."""
    supply_df = load_supply_from_db(lookback_days=lookback)
    if supply_df.empty:
        logger.warning("수급 데이터 없음")
        return []

    dates = sorted(supply_df["date"].unique())
    if len(dates) < 2:
        logger.warning("최소 2거래일 필요 (현재 %d일)", len(dates))
        return []

    latest_date = dates[-1]
    prev_date = dates[-2]
    logger.info("바톤터치 비교: %s(전일) → %s(당일)", prev_date, latest_date)

    today_supply = supply_df[supply_df["date"] == latest_date].set_index("ticker")
    prev_supply = supply_df[supply_df["date"] == prev_date].set_index("ticker")

    # 두 날짜 모두 데이터 있는 종목
    common_tickers = today_supply.index.intersection(prev_supply.index)
    logger.info("공통 종목: %d개", len(common_tickers))

    results: list[dict] = []

    for ticker in common_tickers:
        prev_row = prev_supply.loc[ticker]
        curr_row = today_supply.loc[ticker]

        # 바톤터치 분류
        chain_type, from_leader, to_leader, base_score = classify_baton_touch(
            prev_row, curr_row
        )
        if not chain_type:
            continue

        name = str(curr_row.get("name", ticker))[:10]

        # 가격 데이터
        pdf = load_price_data(ticker)
        if pdf is None:
            continue

        if latest_date not in pdf.index:
            continue

        prow_today = pdf.loc[latest_date]
        close = prow_today["close"]
        if close <= 0:
            continue

        # 가격 받침 확인: 전일 종가 대비 -3% 이내
        tech_score = 0
        tech_flags = []

        if prev_date in pdf.index:
            prev_close = pdf.loc[prev_date]["close"]
            if prev_close > 0:
                price_cushion = (close / prev_close - 1) * 100
            else:
                price_cushion = 0.0
        else:
            price_cushion = 0.0

        if price_cushion >= -3:
            tech_score += 5
            tech_flags.append("받침")

        # 거래량 폭발
        vol_ratio = prow_today.get("vol_ratio", np.nan)
        if not np.isnan(vol_ratio) and vol_ratio >= 2:
            tech_score += 5
            tech_flags.append(f"V{vol_ratio:.1f}x")

        # RSI 적정
        rsi = prow_today.get("rsi", np.nan)
        if not np.isnan(rsi) and 30 <= rsi <= 70:
            tech_score += 3
            tech_flags.append(f"RSI{int(rsi)}")

        # 20MA 눌림목
        ma20_dev = prow_today.get("ma20_dev", np.nan)
        if not np.isnan(ma20_dev) and -15 <= ma20_dev <= 5:
            tech_score += 3
            tech_flags.append("눌림")

        # 수급 점증 보너스
        ticker_hist = supply_df[supply_df["ticker"] == ticker].sort_values("date")
        streak_bonus = calc_streak_bonus(ticker_hist)
        if streak_bonus > 0:
            tech_flags.append(f"점증+{streak_bonus}")

        final_score = base_score + tech_score + streak_bonus

        # 당일 등락률
        ret0 = (close / pdf.loc[prev_date]["close"] - 1) * 100 if prev_date in pdf.index else 0.0

        results.append({
            "ticker": ticker,
            "name": name,
            "close": int(close),
            "ret0": round(float(ret0), 1),
            "chain_type": chain_type,
            "from_leader": from_leader,
            "to_leader": to_leader,
            "base_score": base_score,
            "tech_score": tech_score,
            "streak_bonus": streak_bonus,
            "final_score": final_score,
            "fgn": round(float(curr_row["fgn"]), 1),
            "inst": round(float(curr_row["inst"]), 1),
            "pension": round(float(curr_row["pension"]), 1),
            "finance": round(float(curr_row["finance"]), 1),
            "corp": round(float(curr_row["corp"]), 1),
            "retail": round(float(curr_row["retail"]), 1),
            "prev_fgn": round(float(prev_row["fgn"]), 1),
            "prev_inst": round(float(prev_row["inst"]), 1),
            "prev_pension": round(float(prev_row.get("pension", 0)), 1),
            "price_cushion": round(float(price_cushion), 1),
            "ma20_dev": round(float(ma20_dev), 1) if not np.isnan(ma20_dev) else 0.0,
            "rsi": round(float(rsi), 1) if not np.isnan(rsi) else 50.0,
            "vol_ratio": round(float(vol_ratio), 1) if not np.isnan(vol_ratio) else 1.0,
            "tech_flags": "+".join(tech_flags) if tech_flags else "-",
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    logger.info("바톤터치 감지: %d종목", len(results))
    return results[:top_n]


# ─────────────────────────────────────────────
# 4. 리포트 + 저장
# ─────────────────────────────────────────────

def print_report(results: list[dict]):
    """콘솔 리포트."""
    today = datetime.now().strftime("%Y-%m-%d")
    SEP = "=" * 115

    print(f"\n{SEP}")
    print(f"  수급 바톤터치 감지 — {today}")
    print(f"  철학: 주도 주체 교체 = 매집 릴레이 (받침 확인 후 진입)")
    print(SEP)

    print(f"\n  [바톤터치 감지] {len(results)}종목")
    if results:
        print(
            f"  {'종목':>10} {'종가':>9} {'등락':>6} {'유형':>14} "
            f"{'교체':>12} {'외인':>7} {'기관':>7} {'연기금':>7} "
            f"{'받침':>6} {'기술':>12} {'점수':>5}"
        )
        print(f"  {'─' * 113}")

        for c in results:
            baton = f"{c['from_leader']}→{c['to_leader']}"
            print(
                f"  {c['name']:>10} {c['close']:>9,} {c['ret0']:>+5.1f}% "
                f"{c['chain_type']:>14} {baton:>12} "
                f"{c['fgn']:>+6.0f} {c['inst']:>+6.0f} {c['pension']:>+6.0f} "
                f"{c['price_cushion']:>+5.1f}% "
                f"{c['tech_flags']:>12} {c['final_score']:>5}"
            )
    else:
        print("  바톤터치 감지 없음")

    # ── 유형 설명 ──
    print(f"\n  [유형 설명]")
    print(f"    3단_릴레이    : 외인→기관 교체 + 연기금 매수 (최강 바톤, 35점)")
    print(f"    외인기관_교체  : 외인 매도 전환 → 기관 매수 인수 (25점)")
    print(f"    기관외인_교체  : 기관 매도 전환 → 외인 매수 인수 (25점)")
    print(f"    연기금_인수    : 전일 주도 매도 → 연기금 >=20억 매수 (20점)")
    print(f"    스마트머니_합류 : 금투/기타법인 은밀 매집 (15점)")
    print(f"    일반_교체      : 기타 주체 교체 (10점)")
    print(f"\n  [진입 전략]")
    print(f"    바톤터치 = 매집 릴레이 시그널 → 가격 받침(-3% 이내) 확인 후 D+1 양봉 진입")
    print(f"    3단 릴레이(외인→기관+연기금) = 최강 시그널, 적극 관심")
    print(SEP)


def save_output(results: list[dict]):
    """JSON + CSV 저장."""
    today = datetime.now().strftime("%Y%m%d")

    output = {
        "date": today,
        "type": "supply_chain",
        "count": len(results),
        "baton_touches": results,
    }

    json_path = OUTPUT_DIR / f"supply_chain_{today}.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("JSON 저장: %s", json_path)

    if results:
        df = pd.DataFrame(results)
        csv_path = OUTPUT_DIR / f"supply_chain_{today}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV 저장: %s", csv_path)


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="수급 바톤터치 감지 (P1)")
    parser.add_argument("--lookback", type=int, default=10, help="수급 조회 거래일 수")
    parser.add_argument("--top", type=int, default=30, help="상위 N종목")
    args = parser.parse_args()

    results = detect_supply_chain(lookback=args.lookback, top_n=args.top)
    print_report(results)
    save_output(results)


if __name__ == "__main__":
    main()
