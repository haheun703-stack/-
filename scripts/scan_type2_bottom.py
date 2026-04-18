#!/usr/bin/env python3
"""타입 2: 바닥 반등 스캐너 — KRX 15시 확인 후 진입

바닥에서 고개를 든 종목을 찾는다:
- 52주 고점 대비 30%+ 하락 (피보나치 바닥권)
- 오늘 +3% 이상 빨간봉 (첫 반등 신호)
- 수급 양전환 (순매도→순매수 전환)
- 진입 후 놔두고 천천히 먹는 스타일

BAT-D G4 이후 실행 → "내일 KRX 15시 확인 후보" 리스트 출력

Usage:
    python scripts/scan_type2_bottom.py
    python scripts/scan_type2_bottom.py --min-drop 25   # 최소 하락률(%)
    python scripts/scan_type2_bottom.py --top 20         # 상위 N종목
"""

from __future__ import annotations

import argparse
import json
import logging
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
OUTPUT_DIR = PROJECT_ROOT / "data"

# ─── 바닥 조건 ───
MIN_DROP_PCT = 30.0       # 52주 고점 대비 최소 하락률 (%)
MIN_RED_CANDLE = 3.0      # 오늘 최소 양봉 ��기 (%)
MIN_TV = 30.0             # 최소 거래대금 (억원)
MAX_TV = 10000.0          # 최대 거래대금 (억원)
MIN_LOOKBACK = 60         # 최소 데이터 일수 (52주 대신 60일 이상)

# ─── 피보나치 레벨 ───
FIB_LEVELS = {
    "fib_236": 0.236,
    "fib_382": 0.382,
    "fib_500": 0.500,
    "fib_618": 0.618,
    "fib_786": 0.786,
}


def load_csv_data(csv_path: Path) -> pd.DataFrame | None:
    """CSV 전체 로드."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return None

    if "Date" not in df.columns or len(df) < MIN_LOOKBACK:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def calc_fib_data(df: pd.DataFrame) -> dict | None:
    """52주 피보나치 데이터 계산."""
    # 최근 250거래일 (약 1년)
    recent = df.tail(250)
    if len(recent) < MIN_LOOKBACK:
        return None

    high_52w = recent["High"].max()
    low_52w = recent["Low"].min()
    close = float(df.iloc[-1]["Close"])

    if high_52w <= low_52w or close <= 0:
        return None

    drop_pct = (close / high_52w - 1) * 100  # 음수
    rng = high_52w - low_52w
    position_pct = (close - low_52w) / rng * 100  # 0=최저, 100=최고

    fibs = {}
    for name, ratio in FIB_LEVELS.items():
        level = low_52w + rng * ratio
        fibs[name] = round(level, 0)

    # 현재가가 어느 피보나치 구간인지
    if close <= fibs["fib_236"]:
        fib_zone = "DEEP"       # 23.6% 이하 = 극바닥
    elif close <= fibs["fib_382"]:
        fib_zone = "BOTTOM"     # 23.6~38.2% = 바닥권
    elif close <= fibs["fib_500"]:
        fib_zone = "LOW"        # 38.2~50% = 저가권
    elif close <= fibs["fib_618"]:
        fib_zone = "MID"        # 50~61.8% = 중간
    else:
        fib_zone = "HIGH"       # 61.8%+ = 고가권

    return {
        "high_52w": round(high_52w),
        "low_52w": round(low_52w),
        "drop_pct": round(drop_pct, 1),
        "position_pct": round(position_pct, 1),
        "fib_zone": fib_zone,
        **fibs,
    }


def check_bottom_reversal(df: pd.DataFrame) -> dict | None:
    """바닥 반등 시그널 확인."""
    if len(df) < 10:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(row.get("Close", 0))
    prev_close = float(prev.get("Close", 0))
    if close <= 0 or prev_close <= 0:
        return None

    ret = (close / prev_close - 1) * 100

    # +3% 이상 빨간봉
    if ret < MIN_RED_CANDLE:
        return None

    volume = float(row.get("Volume", 0))

    # 거래대금 (억원)
    tv = close * volume / 1e8
    if not (MIN_TV <= tv <= MAX_TV):
        return None

    # 거래량 비율
    vol_20 = df["Volume"].tail(20).mean()
    vol_ratio = volume / vol_20 if vol_20 > 0 else 0

    # RSI (있으면 사용)
    rsi = float(row.get("RSI", 50))

    # MA20, MA60
    ma20 = float(row.get("MA20", 0))
    ma60 = float(row.get("MA60", 0))

    # 캔들 분석: 긴 양봉인지 (실체 크기)
    open_p = float(row.get("Open", close))
    high = float(row.get("High", close))
    low = float(row.get("Low", close))
    body = abs(close - open_p)
    total = high - low if high > low else 1
    body_ratio = body / total * 100  # 실체 비율 (%)

    return {
        "close": int(close),
        "ret_d0": round(ret, 2),
        "vol_ratio": round(vol_ratio, 2),
        "tv": round(tv, 1),
        "rsi": round(rsi, 1),
        "ma20": round(ma20),
        "ma60": round(ma60),
        "body_ratio": round(body_ratio, 1),
    }


def detect_supply_turn(df: pd.DataFrame) -> dict:
    """수급 양전환 감지 — 이전 순매도에서 최근 순매수로 전환."""
    recent = df.tail(10)

    fn = recent.get("Foreign_Net", pd.Series([0]*len(recent))).fillna(0)
    ins = recent.get("Inst_Net", pd.Series([0]*len(recent))).fillna(0)
    crp = recent.get("Corp_Net", pd.Series([0]*len(recent))).fillna(0)

    # 최근 3일 vs 이전 5일 비교
    fn_recent3 = fn.tail(3).sum()
    fn_prev5 = fn.iloc[:5].sum() if len(fn) >= 8 else fn.iloc[:3].sum()
    fn_turn = fn_recent3 > 0 and fn_prev5 <= 0  # 음→양 전환

    ins_recent3 = ins.tail(3).sum()
    ins_prev5 = ins.iloc[:5].sum() if len(ins) >= 8 else ins.iloc[:3].sum()
    ins_turn = ins_recent3 > 0 and ins_prev5 <= 0

    # 오늘 수급
    today_fn = float(fn.iloc[-1])
    today_in = float(ins.iloc[-1])
    today_cp = float(crp.iloc[-1])

    # 수급 점수 (100점)
    score = 0

    # 양전환 보너스
    if fn_turn:
        score += 20
    if ins_turn:
        score += 20

    # 오늘 양수
    if today_fn > 10:
        score += 15
    elif today_fn > 0:
        score += 8
    if today_in > 10:
        score += 15
    elif today_in > 0:
        score += 8

    # 쌍끌이
    if today_fn > 0 and today_in > 0:
        score += 10

    # ��근 3일 누적
    if fn_recent3 > 30:
        score += 10
    if ins_recent3 > 30:
        score += 10

    return {
        "foreign_net": round(today_fn, 1),
        "inst_net": round(today_in, 1),
        "corp_net": round(today_cp, 1),
        "foreign_3d": round(float(fn_recent3), 1),
        "inst_3d": round(float(ins_recent3), 1),
        "foreign_turn": fn_turn,
        "inst_turn": ins_turn,
        "supply_score": round(min(score, 100), 1),
    }


def scan_type2(min_drop: float = 30.0, top_n: int = 30) -> list[dict]:
    """타입 2 바닥 반등 스캔."""
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    logger.info("CSV 파일: %d개 / 최소 하락률 %d%%", len(csv_files), min_drop)

    candidates = []
    processed = 0

    for path in csv_files:
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, ticker = parts

        df = load_csv_data(path)
        if df is None:
            continue

        # 1. 피보나치 바닥 확인
        fib = calc_fib_data(df)
        if fib is None:
            continue
        if abs(fib["drop_pct"]) < min_drop:
            continue
        # HIGH 구간은 제외 (이미 많이 올라온 종목)
        if fib["fib_zone"] == "HIGH":
            continue

        # 2. 오늘 빨간봉 (+3%+)
        reversal = check_bottom_reversal(df)
        if reversal is None:
            continue

        # 3. 수급 양전환
        supply = detect_supply_turn(df)

        # ��종 점수 = 피보나치 위치 + 반등 품질 + 수급
        score = 0

        # 피보나치 위치 점수 (바닥일수록 높음)
        if fib["fib_zone"] == "DEEP":
            score += 30
        elif fib["fib_zone"] == "BOTTOM":
            score += 25
        elif fib["fib_zone"] == "LOW":
            score += 15
        elif fib["fib_zone"] == "MID":
            score += 10

        # 반등 품질 (빨간봉 크기 + 거래량)
        score += min(20, reversal["ret_d0"] * 3)  # +3% → 9점, +5% → 15점
        if reversal["vol_ratio"] >= 2.0:
            score += 15
        elif reversal["vol_ratio"] >= 1.5:
            score += 10

        # 실체 비율 (긴 양봉이면 강력)
        if reversal["body_ratio"] >= 70:
            score += 10
        elif reversal["body_ratio"] >= 50:
            score += 5

        # RSI 과매도 반등 (30~45 최적)
        if 25 <= reversal["rsi"] <= 45:
            score += 10
        elif 45 < reversal["rsi"] <= 55:
            score += 5

        # 수급 점수 (최대 30)
        score += min(30, supply["supply_score"] * 0.3)

        candidates.append({
            "ticker": ticker,
            "name": name,
            "close": reversal["close"],
            "ret_d0": reversal["ret_d0"],
            "vol_ratio": reversal["vol_ratio"],
            "tv": reversal["tv"],
            "rsi": reversal["rsi"],
            "body_ratio": reversal["body_ratio"],
            "drop_pct": fib["drop_pct"],
            "fib_zone": fib["fib_zone"],
            "position_pct": fib["position_pct"],
            "high_52w": fib["high_52w"],
            "foreign_net": supply["foreign_net"],
            "inst_net": supply["inst_net"],
            "corp_net": supply["corp_net"],
            "foreign_3d": supply["foreign_3d"],
            "inst_3d": supply["inst_3d"],
            "foreign_turn": supply["foreign_turn"],
            "inst_turn": supply["inst_turn"],
            "supply_score": supply["supply_score"],
            "final_score": round(min(score, 100), 1),
        })

        processed += 1
        if processed % 500 == 0:
            logger.info("  %d종목 처리... (%d후보)", processed, len(candidates))

    # 점수순 정렬
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    candidates = candidates[:top_n]

    logger.info("스캔 완료: %d후보 (상위 %d)", len(candidates), top_n)
    return candidates


def print_report(candidates: list[dict]):
    """리포트 출력."""
    today = datetime.now().strftime("%Y-%m-%d")

    print()
    print("=" * 78)
    print(f"  타입 2: 바닥 반등 시드 — {today} | KRX 15시 확인 후보")
    print("=" * 78)
    print(f"  후보: {len(candidates)}종목")
    print()

    if not candidates:
        print("  오늘 바닥 반등 없음")
        print("=" * 78)
        return

    print(f"  {'종목':>12} {'종가':>8} {'수익률':>6} {'하락':>6} {'구간':>6} "
          f"{'거래량':>5} {'외인':>6} {'기관':>6} {'전환':>4} {'점수':>5}")
    print(f"  {'─' * 74}")

    for c in candidates:
        # 양전환 표시
        turn = ""
        if c["foreign_turn"]:
            turn += "F"
        if c["inst_turn"]:
            turn += "I"
        if not turn:
            turn = "-"

        print(f"  {c['name'][:10]:>12} {c['close']:>8,} "
              f"{c['ret_d0']:>+5.1f}% {c['drop_pct']:>+5.0f}% "
              f"{c['fib_zone']:>6} "
              f"{c['vol_ratio']:>4.1f}x "
              f"{c['foreign_net']:>+5.0f} {c['inst_net']:>+5.0f} "
              f"{turn:>4} "
              f"{c['final_score']:>5.0f}")

    print()
    print("  [해석]")
    print("    하락: 52주 고점 대비 하락률 / 구간: DEEP(극바닥)~MID(중간)")
    print("    전환: F=외인양전환, I=기관양전환 (이전 매도→최근 매수)")
    print("    진입: 내일 KRX 장중 양봉 확인 → 15시 동시호가 진입")
    print("    보유: 3~5일 (피보나치 38.2% 회복 목표)")
    print("=" * 78)


def save_output(candidates: list[dict]):
    """결과 저장."""
    today = datetime.now().strftime("%Y%m%d")

    output = {
        "date": today,
        "type": "bottom_reversal",
        "count": len(candidates),
        "candidates": candidates,
    }
    json_path = OUTPUT_DIR / f"type2_bottom_{today}.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("JSON 저장: %s", json_path)

    if candidates:
        # bool → str 변환 for CSV
        for c in candidates:
            c["foreign_turn"] = str(c["foreign_turn"])
            c["inst_turn"] = str(c["inst_turn"])
        df = pd.DataFrame(candidates)
        csv_path = OUTPUT_DIR / f"type2_bottom_{today}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV 저장: %s", csv_path)


def main():
    parser = argparse.ArgumentParser(description="타입 2: 바닥 반등 스캐너")
    parser.add_argument("--min-drop", type=float, default=30.0, help="최소 하락�� (%%)")
    parser.add_argument("--top", type=int, default=30, help="상위 N종목")
    args = parser.parse_args()

    candidates = scan_type2(min_drop=args.min_drop, top_n=args.top)
    print_report(candidates)
    save_output(candidates)


if __name__ == "__main__":
    main()
