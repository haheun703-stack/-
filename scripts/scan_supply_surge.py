#!/usr/bin/env python3
"""수급 급변 스캔 (P0) — 6유형 수급 급변 감지 + 기술적 필터 보조

한국장 핵심 철학: 수급이 원인, 차트가 결과
- A_쌍끌이: 외인 >=50억 AND 기관 >=50억 (30점)
- B_기관연기금: 기관 >=50억 AND 연기금 >=20억 (25점)
- C_3주체합류: 외인+기관+연기금 모두 양수, 합산 >=50억 (25점)
- D_외인폭발: 외인 >=100억 단독 (20점)
- E_연기금매집: 연기금 >=20억 (15점)
- F_금투기타: 금투 >=30억 OR 기타법인 >=20억 (10점)
- X_개인추격: 개인 주도 + 기관/외인 이탈 (매도 시그널)
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


# ─────────────────────────────────────────────
# 1. 데이터 로더
# ─────────────────────────────────────────────

def load_supply_from_db(lookback_days: int = 10) -> pd.DataFrame:
    """DB에서 최근 N거래일 7유형(6core+개인) 수급 피벗 로드."""
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
        df["ret0"] = df["close"].pct_change() * 100

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
# 2. 수급 유형 분류
# ─────────────────────────────────────────────

def classify_supply_type(row: pd.Series) -> tuple[str, int]:
    """당일 수급 → (유형명, 기본점수). 빈 문자열이면 해당 없음."""
    fgn = row["fgn"]
    inst = row["inst"]
    pension = row["pension"]
    finance = row["finance"]
    corp = row["corp"]
    retail = row["retail"]
    total_abs = abs(fgn) + abs(inst) + abs(pension) + abs(finance) + abs(corp) + abs(retail)

    # X: 개인추격 (매도 시그널)
    if total_abs > 0 and retail > 0 and fgn < 0 and inst < 0:
        if retail / total_abs > 0.6:
            return "X_개인추격", -10

    # A: 쌍끌이
    if fgn >= 50 and inst >= 50:
        return "A_쌍끌이", 30

    # B: 기관+연기금
    if inst >= 50 and pension >= 20:
        return "B_기관연기금", 25

    # C: 3주체 합류
    if fgn > 0 and inst > 0 and pension > 0 and (fgn + inst + pension) >= 50:
        return "C_3주체합류", 25

    # D: 외인 폭발
    if fgn >= 100:
        return "D_외인폭발", 20

    # E: 연기금 매집
    if pension >= 20:
        return "E_연기금매집", 15

    # F: 금투/기타법인
    if finance >= 30 or corp >= 20:
        return "F_금투기타", 10

    return "", 0


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

def scan_supply_surge(
    lookback: int = 10,
    top_n: int = 50,
) -> tuple[list[dict], list[dict]]:
    """수급 급변 스캔. (매수후보, 매도시그널) 반환."""
    supply_df = load_supply_from_db(lookback_days=lookback)
    if supply_df.empty:
        logger.warning("수급 데이터 없음")
        return [], []

    latest_date = supply_df["date"].max()
    logger.info("최신 거래일: %s", latest_date)

    today_supply = supply_df[supply_df["date"] == latest_date].copy()
    logger.info("당일 수급: %d종목", len(today_supply))

    candidates: list[dict] = []
    sell_signals: list[dict] = []

    for _, row in today_supply.iterrows():
        ticker = row["ticker"]
        name = str(row.get("name", ticker))

        # 수급 유형 분류
        stype, base_score = classify_supply_type(row)
        if not stype:
            continue

        # 가격 데이터
        pdf = load_price_data(ticker)
        if pdf is None or latest_date not in pdf.index:
            continue

        prow = pdf.loc[latest_date]
        close = prow["close"]
        if close <= 0:
            continue

        # 기술적 지표
        ma20_dev = prow.get("ma20_dev", np.nan)
        rsi = prow.get("rsi", np.nan)
        vol_ratio = prow.get("vol_ratio", np.nan)
        ret0 = prow.get("ret0", np.nan)

        tech_score = 0
        tech_flags = []

        # 20MA 눌림목 (-15% ~ +5%)
        if not np.isnan(ma20_dev) and -15 <= ma20_dev <= 5:
            tech_score += 5
            tech_flags.append("눌림")

        # RSI 적정 (30~70)
        if not np.isnan(rsi) and 30 <= rsi <= 70:
            tech_score += 3
            tech_flags.append(f"RSI{int(rsi)}")

        # 거래량 폭발 (2배+)
        if not np.isnan(vol_ratio) and vol_ratio >= 2:
            tech_score += 5
            tech_flags.append(f"V{vol_ratio:.1f}x")

        # 수급 점증 보너스
        ticker_hist = supply_df[supply_df["ticker"] == ticker].sort_values("date")
        streak_bonus = calc_streak_bonus(ticker_hist)
        if streak_bonus > 0:
            tech_flags.append(f"점증+{streak_bonus}")

        # 5일 누적 수급
        cum_fgn = ticker_hist["fgn"].sum()
        cum_inst = ticker_hist["inst"].sum()
        cum_pension = ticker_hist["pension"].sum()

        final_score = base_score + tech_score + streak_bonus

        entry = {
            "ticker": ticker,
            "name": name[:10],
            "close": int(close),
            "ret0": round(float(ret0), 1) if not np.isnan(ret0) else 0.0,
            "type": stype,
            "base_score": base_score,
            "tech_score": tech_score,
            "streak_bonus": streak_bonus,
            "final_score": final_score,
            "fgn": round(float(row["fgn"]), 1),
            "inst": round(float(row["inst"]), 1),
            "pension": round(float(row["pension"]), 1),
            "finance": round(float(row["finance"]), 1),
            "corp": round(float(row["corp"]), 1),
            "retail": round(float(row["retail"]), 1),
            "cum_fgn_5d": round(float(cum_fgn), 1),
            "cum_inst_5d": round(float(cum_inst), 1),
            "cum_pension_5d": round(float(cum_pension), 1),
            "ma20_dev": round(float(ma20_dev), 1) if not np.isnan(ma20_dev) else 0.0,
            "rsi": round(float(rsi), 1) if not np.isnan(rsi) else 50.0,
            "vol_ratio": round(float(vol_ratio), 1) if not np.isnan(vol_ratio) else 1.0,
            "tech_flags": "+".join(tech_flags) if tech_flags else "-",
        }

        if stype == "X_개인추격":
            sell_signals.append(entry)
        else:
            candidates.append(entry)

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    sell_signals.sort(key=lambda x: x["retail"], reverse=True)

    logger.info("매수 후보: %d종목 / 매도 시그널: %d종목", len(candidates), len(sell_signals))
    return candidates[:top_n], sell_signals[:20]


# ─────────────────────────────────────────────
# 4. 리포트 + 저장
# ─────────────────────────────────────────────

def print_report(candidates: list[dict], sell_signals: list[dict]):
    """콘솔 리포트."""
    today = datetime.now().strftime("%Y-%m-%d")
    SEP = "=" * 105

    print(f"\n{SEP}")
    print(f"  수급 급변 스캔 — {today}")
    print(f"  철학: 수급이 원인, 차트가 결과 (한국장)")
    print(SEP)

    # ── 매수 후보 ──
    print(f"\n  [매수 후보] {len(candidates)}종목")
    if candidates:
        print(
            f"  {'종목':>10} {'종가':>9} {'등락':>6} {'유형':>14} "
            f"{'외인':>7} {'기관':>7} {'연기금':>7} "
            f"{'5d외':>7} {'5d기':>7} {'기술':>12} {'점수':>5}"
        )
        print(f"  {'─' * 103}")

        for c in candidates:
            print(
                f"  {c['name']:>10} {c['close']:>9,} {c['ret0']:>+5.1f}% "
                f"{c['type']:>14} {c['fgn']:>+6.0f} {c['inst']:>+6.0f} "
                f"{c['pension']:>+6.0f} {c['cum_fgn_5d']:>+6.0f} {c['cum_inst_5d']:>+6.0f} "
                f"{c['tech_flags']:>12} {c['final_score']:>5}"
            )
    else:
        print("  후보 없음")

    # ── 매도 시그널 ──
    print(f"\n  [매도 시그널 — 개인추격] {len(sell_signals)}종목")
    if sell_signals:
        print(f"  {'종목':>10} {'종가':>9} {'등락':>6} {'개인':>7} {'외인':>7} {'기관':>7} {'거래량':>6}")
        print(f"  {'─' * 65}")
        for s in sell_signals:
            print(
                f"  {s['name']:>10} {s['close']:>9,} {s['ret0']:>+5.1f}% "
                f"{s['retail']:>+6.0f} {s['fgn']:>+6.0f} {s['inst']:>+6.0f} "
                f"{s['vol_ratio']:>5.1f}x"
            )

    # ── 유형 설명 ──
    print(f"\n  [유형 설명]")
    print(f"    A_쌍끌이     : 외인 >=50억 AND 기관 >=50억 (최강 시그널)")
    print(f"    B_기관연기금  : 기관 >=50억 AND 연기금 >=20억 (스마트머니)")
    print(f"    C_3주체합류   : 외인+기관+연기금 모두 양수, 합 >=50억")
    print(f"    D_외인폭발    : 외인 >=100억 단독 (PF 1.55)")
    print(f"    E_연기금매집  : 연기금 >=20억")
    print(f"    F_금투기타    : 금투 >=30억 OR 기타법인 >=20억")
    print(f"    X_개인추격    : 개인주도+기관/외인이탈 (D+3 -4% 매도시그널)")
    print(f"\n  [진입 전략]")
    print(f"    수급 스캔은 후보 선별용 → D+1 양봉 확인 후 진입 (PF 1.82)")
    print(f"    개인추격 종목 보유 중이면 → 즉시 매도 검토 (D+3 회피율 76%)")
    print(SEP)


def save_output(candidates: list[dict], sell_signals: list[dict]):
    """JSON + CSV 저장."""
    today = datetime.now().strftime("%Y%m%d")

    output = {
        "date": today,
        "type": "supply_surge",
        "buy_count": len(candidates),
        "sell_count": len(sell_signals),
        "buy_candidates": candidates,
        "sell_signals": sell_signals,
    }

    json_path = OUTPUT_DIR / f"supply_surge_{today}.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("JSON 저장: %s", json_path)

    if candidates:
        df = pd.DataFrame(candidates)
        csv_path = OUTPUT_DIR / f"supply_surge_{today}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV 저장: %s", csv_path)


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="수급 급변 스캔 (P0)")
    parser.add_argument("--lookback", type=int, default=10, help="수급 조회 거래일 수")
    parser.add_argument("--top", type=int, default=50, help="상위 N종목")
    args = parser.parse_args()

    candidates, sell_signals = scan_supply_surge(
        lookback=args.lookback, top_n=args.top
    )
    print_report(candidates, sell_signals)
    save_output(candidates, sell_signals)


if __name__ == "__main__":
    main()
