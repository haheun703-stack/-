#!/usr/bin/env python3
"""타입 1: 수급 릴레이 스캐너 — NXT 진입 후보

수급이 조용히 쌓이고 있는데 아직 안 터진 종목을 찾는다.
- 최근 3~5일 외인/기관 연속 순매수 (축적)
- 오늘 시드 조건: +1~5% 빨간봉 + 거래량 증가
- MA20 근처 (아직 크게 안 오름)

BAT-D G4 이후 실행 → "오늘 NXT 진입 후보" 리스트 출력

Usage:
    python scripts/scan_type1_relay.py
    python scripts/scan_type1_relay.py --min-days 3    # 최소 축적 일수
    python scripts/scan_type1_relay.py --top 20         # 상위 N종목
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
OUTPUT_DIR = PROJECT_ROOT / "data"
NXT_MASTER_PATH = PROJECT_ROOT / "data" / "nxt" / "nxt_master.json"


def load_nxt_tickers() -> set[str]:
    """NXT 거래 가능 종목 티커 로드."""
    if not NXT_MASTER_PATH.exists():
        logger.warning("NXT 마스터 없음: %s — NXT 필터 비활성", NXT_MASTER_PATH)
        return set()
    try:
        with open(NXT_MASTER_PATH, encoding="utf-8") as f:
            data = json.load(f)
        tickers = set(data.get("ticker_set", {}).keys())
        logger.info("NXT 마스터: %d종목 로드", len(tickers))
        return tickers
    except Exception as e:
        logger.warning("NXT 마스터 로드 실패: %s", e)
        return set()

# ─── 시드 조건 ───
SEED_RET_MIN = 1.0       # D+0 수익률 하한 (%)
SEED_RET_MAX = 5.0       # D+0 수익률 상한 (%)
SEED_VOL_RATIO_MIN = 1.3 # 거래량/MA20 하한
SEED_MA20_DEV_MIN = -5.0 # 20MA 이격 하한 (%)
SEED_MA20_DEV_MAX = 5.0  # 20MA 이격 상한 (%)
SEED_RET60_MAX = 30.0    # 60일 수익률 상한 (%)
SEED_TV_MIN = 30.0       # 거래대금 하한 (억원)
SEED_TV_MAX = 10000.0    # 거래대금 상한 (억원)

# ─── 수급 축적 조건 ───
ACCUM_MIN_DAYS = 3        # 최소 연속 순매수 일수
ACCUM_NET_MIN = 5.0       # 순매수 일일 최소 (억원) — 너무 작으면 노이즈


def load_csv_data(csv_path: Path, lookback: int = 70) -> pd.DataFrame | None:
    """CSV에서 최근 데이터 로드."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return None

    if "Date" not in df.columns or len(df) < 30:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").tail(lookback).reset_index(drop=True)
    return df


def detect_accumulation(df: pd.DataFrame, min_days: int = 3, net_min: float = 5.0) -> dict:
    """수급 축적 감지 — 외인/기관 연속 순매수 일수 + 누적 금액.

    Returns:
        {foreign_streak, inst_streak, corp_streak, foreign_cum, inst_cum, corp_cum,
         dual_streak, best_streak, accum_score}
    """
    recent = df.tail(10).copy()  # 최근 10일

    fn = recent.get("Foreign_Net", pd.Series([0]*len(recent)))
    ins = recent.get("Inst_Net", pd.Series([0]*len(recent)))
    crp = recent.get("Corp_Net", pd.Series([0]*len(recent)))

    def streak_and_cum(series, threshold=0):
        """연속 양수 일수 + 누적."""
        vals = series.values
        streak = 0
        for v in reversed(vals):
            if pd.isna(v) or v <= threshold:
                break
            streak += 1
        cum = float(series.tail(min(streak, 10)).sum()) if streak > 0 else 0.0
        return streak, round(cum, 1)

    f_streak, f_cum = streak_and_cum(fn)
    i_streak, i_cum = streak_and_cum(ins)
    c_streak, c_cum = streak_and_cum(crp)

    # 쌍끌이 연속일수 (외인+기관 동시 양수)
    dual_streak = 0
    for fv, iv in zip(reversed(fn.values), reversed(ins.values)):
        if pd.isna(fv) or pd.isna(iv):
            break
        if fv > 0 and iv > 0:
            dual_streak += 1
        else:
            break

    best_streak = max(f_streak, i_streak, c_streak)

    # 축적 점수 (100점 만점)
    score = 0
    # 외인 축적
    if f_streak >= min_days and f_cum >= net_min * min_days:
        score += min(30, f_streak * 6 + f_cum / 10)
    # 기관 축적
    if i_streak >= min_days and i_cum >= net_min * min_days:
        score += min(30, i_streak * 6 + i_cum / 10)
    # 기타법인 축적
    if c_streak >= min_days and c_cum >= net_min * min_days:
        score += min(15, c_streak * 3 + c_cum / 20)
    # 쌍끌이 보너스
    if dual_streak >= 2:
        score += min(25, dual_streak * 8)

    return {
        "foreign_streak": f_streak,
        "inst_streak": i_streak,
        "corp_streak": c_streak,
        "foreign_cum": f_cum,
        "inst_cum": i_cum,
        "corp_cum": c_cum,
        "dual_streak": dual_streak,
        "best_streak": best_streak,
        "accum_score": round(min(score, 100), 1),
    }


def check_seed_condition(df: pd.DataFrame) -> dict | None:
    """오늘(마지막 행) 시드 조건 충족 확인."""
    if len(df) < 21:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    close = row.get("Close", 0)
    prev_close = prev.get("Close", 0)
    if close <= 0 or prev_close <= 0:
        return None

    ret = (close / prev_close - 1) * 100
    volume = row.get("Volume", 0)
    ma20 = row.get("MA20", 0)

    # 거래량 비율 (MA20 거래량)
    vol_20 = df["Volume"].tail(20).mean()
    vol_ratio = volume / vol_20 if vol_20 > 0 else 0

    # MA20 이격
    ma20_dev = (close / ma20 - 1) * 100 if ma20 > 0 else 0

    # 60일 수익률
    if len(df) >= 61:
        close_60 = df.iloc[-61]["Close"]
        ret60 = (close / close_60 - 1) * 100 if close_60 > 0 else 0
    else:
        ret60 = 0

    # 거래대금 (억원)
    tv = close * volume / 1e8

    # 시드 조건 체크
    if not (SEED_RET_MIN <= ret <= SEED_RET_MAX):
        return None
    if vol_ratio < SEED_VOL_RATIO_MIN:
        return None
    if not (SEED_MA20_DEV_MIN <= ma20_dev <= SEED_MA20_DEV_MAX):
        return None
    if ret60 > SEED_RET60_MAX:
        return None
    if not (SEED_TV_MIN <= tv <= SEED_TV_MAX):
        return None

    return {
        "close": int(close),
        "ret_d0": round(ret, 2),
        "vol_ratio": round(vol_ratio, 2),
        "ma20_dev": round(ma20_dev, 2),
        "ret60": round(ret60, 1),
        "tv": round(tv, 1),
        "rsi": round(float(row.get("RSI", 0)), 1),
    }


def scan_type1(min_accum_days: int = 3, top_n: int = 30) -> list[dict]:
    """타입 1 수급 릴레이 스캔. NXT 거래 가능 종목만 포함."""
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    nxt_tickers = load_nxt_tickers()
    logger.info("CSV 파일: %d개 / 축적 최소 %d일 / NXT %d종목",
                len(csv_files), min_accum_days, len(nxt_tickers))

    candidates = []
    nxt_filtered = 0
    processed = 0

    for path in csv_files:
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, ticker = parts

        # NXT 거래 불가 종목 제거 (NXT 마스터가 있을 때만)
        if nxt_tickers and ticker not in nxt_tickers:
            nxt_filtered += 1
            continue

        df = load_csv_data(path)
        if df is None:
            continue

        # 1. 시드 조건 확인 (오늘 빨간봉)
        seed = check_seed_condition(df)
        if seed is None:
            continue

        # 2. 수급 축적 확인
        accum = detect_accumulation(df, min_days=min_accum_days)
        if accum["best_streak"] < min_accum_days:
            continue
        if accum["accum_score"] < 15:
            continue

        # 3. 오늘 수급 확인 (적어도 하나의 주체가 양수)
        today_fn = float(df.iloc[-1].get("Foreign_Net", 0) or 0)
        today_in = float(df.iloc[-1].get("Inst_Net", 0) or 0)
        today_cp = float(df.iloc[-1].get("Corp_Net", 0) or 0)

        # 최종 점수 = 축적점수 + 시드 품질 보너스
        final_score = accum["accum_score"]
        # 눌림 보너스 (MA20 -3~0%)
        if -3.0 <= seed["ma20_dev"] <= 0:
            final_score += 15
        # 저변동 보너스
        vol5d = df["Close"].tail(5).pct_change().std() * 100
        if vol5d < 3.0:
            final_score += 10
        # 쌍끌이 당일 보너스
        if today_fn > 10 and today_in > 10:
            final_score += 10

        candidates.append({
            "ticker": ticker,
            "name": name,
            "close": seed["close"],
            "ret_d0": seed["ret_d0"],
            "vol_ratio": seed["vol_ratio"],
            "ma20_dev": seed["ma20_dev"],
            "ret60": seed["ret60"],
            "tv": seed["tv"],
            "rsi": seed["rsi"],
            "foreign_net": round(today_fn, 1),
            "inst_net": round(today_in, 1),
            "corp_net": round(today_cp, 1),
            "foreign_streak": accum["foreign_streak"],
            "inst_streak": accum["inst_streak"],
            "foreign_cum": accum["foreign_cum"],
            "inst_cum": accum["inst_cum"],
            "dual_streak": accum["dual_streak"],
            "accum_score": accum["accum_score"],
            "final_score": round(min(final_score, 100), 1),
        })

        processed += 1
        if processed % 500 == 0:
            logger.info("  %d종목 처리... (%d후보)", processed, len(candidates))

    # 점수순 정렬
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    candidates = candidates[:top_n]

    if nxt_filtered:
        logger.info("NXT 필터: %d종목 제외 (NXT 거래 불가)", nxt_filtered)
    logger.info("스캔 완료: %d후보 (상위 %d)", len(candidates), top_n)
    return candidates


def print_report(candidates: list[dict]):
    """리포트 출력."""
    today = datetime.now().strftime("%Y-%m-%d")

    print()
    print("=" * 78)
    print(f"  타입 1: 수급 릴레이 시드 — {today} | NXT 진입 후보")
    print("=" * 78)
    print(f"  후보: {len(candidates)}종목")
    print()

    if not candidates:
        print("  오늘 시드 없음")
        print("=" * 78)
        return

    print(f"  {'종목':>12} {'종가':>8} {'수익률':>6} {'거래량':>5} {'MA20':>6} "
          f"{'외인':>6} {'기관':>6} {'축적':>4} {'쌍끌이':>4} {'점수':>5}")
    print(f"  {'─' * 74}")

    for c in candidates:
        # 축적 표시
        accum_str = ""
        if c["foreign_streak"] >= 3:
            accum_str += f"F{c['foreign_streak']}"
        if c["inst_streak"] >= 3:
            accum_str += f"I{c['inst_streak']}"

        dual_str = f"D{c['dual_streak']}" if c["dual_streak"] >= 2 else ""

        print(f"  {c['name'][:10]:>12} {c['close']:>8,} "
              f"{c['ret_d0']:>+5.1f}% {c['vol_ratio']:>4.1f}x "
              f"{c['ma20_dev']:>+5.1f}% "
              f"{c['foreign_net']:>+5.0f} {c['inst_net']:>+5.0f} "
              f"{accum_str:>4} {dual_str:>4} "
              f"{c['final_score']:>5.0f}")

    print()
    print("  [해석]")
    print("    축적: F3=외인3일연속, I5=기관5일연속, D2=쌍끌이2일")
    print("    진입: NXT(18~03시) 종가 근처 지정가 매수")
    print("    보유: D+1 양봉 확인 → 보유 / 음봉 → 즉시 손절")
    print("=" * 78)


def save_output(candidates: list[dict]):
    """결과 저장."""
    today = datetime.now().strftime("%Y%m%d")

    # JSON
    output = {
        "date": today,
        "type": "relay_seed",
        "count": len(candidates),
        "candidates": candidates,
    }
    json_path = OUTPUT_DIR / f"type1_relay_{today}.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON 저장: %s", json_path)

    # CSV
    if candidates:
        df = pd.DataFrame(candidates)
        csv_path = OUTPUT_DIR / f"type1_relay_{today}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV 저장: %s", csv_path)


def main():
    parser = argparse.ArgumentParser(description="타입 1: 수급 릴레이 스캐너")
    parser.add_argument("--min-days", type=int, default=3, help="최소 축적 일수")
    parser.add_argument("--top", type=int, default=30, help="상위 N종목")
    args = parser.parse_args()

    candidates = scan_type1(min_accum_days=args.min_days, top_n=args.top)
    print_report(candidates)
    save_output(candidates)


if __name__ == "__main__":
    main()
