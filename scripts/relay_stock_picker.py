"""[A] 릴레이 후행 섹터 내 종목 선정.

발화 감지 후 후행 섹터 내에서 매수 종목 우선순위를 계산한다.

4가지 기준으로 점수화:
  1. 전일 대기 상태: 오늘 등락률 -2% ~ +3% (아직 안 움직인 것)
  2. 거래량 이상: 최근 5일 평균 대비 120% 이상
  3. 120일선 근접: 현재가가 120일선 ±5% 이내
  4. 시총 중간값: 섹터 내 시총 중간 20~70% (1등 제외)

서보성 원칙: "1등 말고 2,3등을 사라" — 시총 1위는 후순위.

사용법:
  python scripts/relay_stock_picker.py --sector 생명보험
  python scripts/relay_stock_picker.py --sector 손해보험 --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"

# 점수 배점
SCORE_WAITING = 30       # 대기 상태 (아직 안 움직임)
SCORE_VOLUME = 25        # 거래량 이상
SCORE_MA120 = 25         # 120일선 근접
SCORE_MIDCAP = 20        # 시총 중간값


def load_sector_stocks(sector: str) -> pd.DataFrame:
    """네이버 섹터맵에서 특정 섹터 종목 로드."""
    path = DATA_DIR / "naver_sector_map.csv"
    df = pd.read_csv(path, dtype={"ticker": str})
    sector_df = df[df["sector"] == sector].copy()
    sector_df["market_cap"] = pd.to_numeric(sector_df["market_cap"], errors="coerce")
    return sector_df.sort_values("market_cap", ascending=False).reset_index(drop=True)


def load_stock_latest(ticker: str, name: str) -> dict | None:
    """종목의 최신 데이터 로드 (stock_data_daily CSV)."""
    # 파일명: 종목명_티커.csv
    csv_path = DAILY_DIR / f"{name}_{ticker}.csv"
    if not csv_path.exists():
        # 이름이 다를 수 있으므로 glob으로 탐색
        matches = list(DAILY_DIR.glob(f"*_{ticker}.csv"))
        if not matches:
            return None
        csv_path = matches[0]

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.dropna(subset=["Close"]).sort_values("Date")
        if len(df) < 10:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 최근 5일 평균 거래량
        vol_5d = df["Volume"].tail(6).head(5).mean()

        return {
            "ticker": ticker,
            "name": csv_path.stem.rsplit("_", 1)[0],
            "date": str(latest["Date"].date()),
            "close": float(latest["Close"]),
            "prev_close": float(prev["Close"]),
            "change_pct": round(
                (float(latest["Close"]) - float(prev["Close"]))
                / float(prev["Close"]) * 100, 2
            ) if float(prev["Close"]) > 0 else 0,
            "volume": float(latest["Volume"]),
            "vol_5d_avg": float(vol_5d) if vol_5d > 0 else 1,
            "vol_ratio": round(float(latest["Volume"]) / vol_5d, 2) if vol_5d > 0 else 1,
            "ma120": float(latest.get("MA120", 0)) if pd.notna(latest.get("MA120")) else 0,
            "rsi": float(latest.get("RSI", 50)) if pd.notna(latest.get("RSI")) else 50,
            "market_cap": float(latest.get("MarketCap", 0)) if pd.notna(latest.get("MarketCap")) else 0,
        }
    except Exception:
        return None


def score_stocks(
    sector_stocks: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """섹터 내 종목 점수화 + 상위 N개 반환."""

    candidates = []
    for _, row in sector_stocks.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        data = load_stock_latest(ticker, name)
        if not data or data["close"] <= 0:
            continue
        candidates.append(data)

    if not candidates:
        return []

    # 시총 백분위 계산
    caps = sorted([c["market_cap"] for c in candidates if c["market_cap"] > 0])
    if caps:
        p20 = np.percentile(caps, 20)
        p70 = np.percentile(caps, 70)
    else:
        p20, p70 = 0, float("inf")

    # 시총 1위 식별 (서보성 원칙: 1등 후순위)
    max_cap = max(c["market_cap"] for c in candidates)

    results = []
    for c in candidates:
        score = 0
        reasons = []

        # 1. 대기 상태 (-2% ~ +3%)
        chg = c["change_pct"]
        if -2 <= chg <= 3:
            score += SCORE_WAITING
            reasons.append(f"대기중({chg:+.1f}%)")
        elif -5 <= chg < -2:
            score += SCORE_WAITING * 0.5
            reasons.append(f"소폭하락({chg:+.1f}%)")

        # 2. 거래량 이상 (5일 평균 대비 120%+)
        vr = c["vol_ratio"]
        if vr >= 2.0:
            score += SCORE_VOLUME
            reasons.append(f"거래량{vr:.1f}x")
        elif vr >= 1.5:
            score += SCORE_VOLUME * 0.8
            reasons.append(f"거래량{vr:.1f}x")
        elif vr >= 1.2:
            score += SCORE_VOLUME * 0.5
            reasons.append(f"거래량{vr:.1f}x")

        # 3. 120일선 근접 (±5% 이내)
        if c["ma120"] > 0:
            gap = abs(c["close"] - c["ma120"]) / c["ma120"]
            if gap <= 0.02:
                score += SCORE_MA120
                reasons.append(f"120일선밀착({gap*100:.1f}%)")
            elif gap <= 0.05:
                score += SCORE_MA120 * 0.7
                reasons.append(f"120일선근접({gap*100:.1f}%)")
            elif gap <= 0.10:
                score += SCORE_MA120 * 0.3
                reasons.append(f"120일선({gap*100:.1f}%)")

        # 4. 시총 중간값 (20~70 퍼센타일)
        cap = c["market_cap"]
        if cap > 0:
            if p20 <= cap <= p70:
                score += SCORE_MIDCAP
                reasons.append("시총중간")
            elif cap > p70 and cap < max_cap:
                score += SCORE_MIDCAP * 0.5
                reasons.append("시총상위")

            # 서보성 원칙: 시총 1위 감점
            if cap == max_cap:
                score -= 10
                reasons.append("시총1위감점")

        results.append({
            **c,
            "score": round(score),
            "reasons": reasons,
        })

    # 점수 내림차순 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def pick_relay_stocks(
    follow_sector: str,
    top_n: int = 3,
) -> list[dict]:
    """후행 섹터 종목 선정 메인 함수."""
    sector_stocks = load_sector_stocks(follow_sector)
    if sector_stocks.empty:
        print(f"  '{follow_sector}' 섹터에 종목이 없습니다.")
        return []

    print(f"  [{follow_sector}] {len(sector_stocks)}종목 중 상위 {top_n}개 선정...")
    return score_stocks(sector_stocks, top_n=top_n)


def print_picks(picks: list[dict], follow_sector: str):
    """선정 결과 출력."""
    if not picks:
        print(f"  {follow_sector}: 선정 종목 없음")
        return

    print(f"\n  [{follow_sector}] 매수 후보 {len(picks)}종목")
    print(f"  {'순위':>4} {'종목':>12} {'코드':>8} {'점수':>4} "
          f"{'등락률':>7} {'거래량':>6} {'RSI':>5}  사유")
    print(f"  {'─' * 70}")

    for i, p in enumerate(picks, 1):
        reason_str = " | ".join(p["reasons"])
        print(f"  {i:>3}위 {p['name']:>12} ({p['ticker']}) "
              f"{p['score']:>3}점 "
              f"{p['change_pct']:>+6.1f}% "
              f"{p['vol_ratio']:>5.1f}x "
              f"{p['rsi']:>4.0f}  "
              f"{reason_str}")


def main():
    parser = argparse.ArgumentParser(description="릴레이 후행 섹터 종목 선정")
    parser.add_argument("--sector", required=True, help="후행 섹터명 (예: 생명보험)")
    parser.add_argument("--top", type=int, default=3, help="상위 N개 (기본 3)")
    args = parser.parse_args()

    picks = pick_relay_stocks(args.sector, top_n=args.top)
    print_picks(picks, args.sector)


if __name__ == "__main__":
    main()
