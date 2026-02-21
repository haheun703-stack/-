"""[A] 릴레이 후행 섹터 내 종목 선정.

발화 감지 후 후행 섹터 내에서 매수 종목 우선순위를 계산한다.

5가지 기준으로 점수화:
  1. 전일 대기 상태: 오늘 등락률 -2% ~ +3% (아직 안 움직인 것)
  2. 거래량 이상: 최근 5일 평균 대비 120% 이상
  3. 120일선 근접: 현재가가 120일선 ±5% 이내
  4. 시총 중간값: 섹터 내 시총 중간 20~70%
  5. 수급 흐름: 외국인/기관 연속 순매수 일수

사용법:
  python scripts/relay_stock_picker.py --sector 생명보험
  python scripts/relay_stock_picker.py --sector 손해보험 --top 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"
PARQUET_DIR = PROJECT_ROOT / "data" / "processed"

# 점수 배점 (총 100점)
SCORE_WAITING = 25       # 대기 상태 (아직 안 움직임)
SCORE_VOLUME = 20        # 거래량 이상
SCORE_MA120 = 20         # 120일선 근접
SCORE_MIDCAP = 15        # 시총 중간값
SCORE_FLOW = 20          # 외국인/기관 수급 흐름


def _load_parquet_flow(ticker: str) -> dict | None:
    """parquet에서 외국인 연속매수 일수를 로드."""
    pq_path = PARQUET_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path, columns=["foreign_consecutive_buy"])
        if df.empty:
            return None
        last = df.iloc[-1]
        foreign = int(last.get("foreign_consecutive_buy", 0))
        return {"foreign_streak": foreign, "inst_streak": 0}
    except Exception:
        return None


def _fetch_flow_pykrx(ticker: str, days: int = 30) -> dict | None:
    """pykrx에서 최근 외국인/기관 순매수 데이터를 실시간 조회."""
    try:
        from pykrx import stock as krx
        from datetime import datetime, timedelta

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        df = krx.get_market_trading_value_by_date(start, end, ticker)
        if df.empty:
            return None

        # 외국인합계, 기관합계 컬럼
        foreign_col = "외국인합계" if "외국인합계" in df.columns else None
        inst_col = "기관합계" if "기관합계" in df.columns else None

        foreign_streak = 0
        inst_streak = 0

        if foreign_col:
            vals = df[foreign_col].values
            for i in range(len(vals) - 1, -1, -1):
                if vals[i] > 0:
                    foreign_streak += 1
                else:
                    break

        if inst_col:
            vals = df[inst_col].values
            for i in range(len(vals) - 1, -1, -1):
                if vals[i] > 0:
                    inst_streak += 1
                else:
                    break

        return {"foreign_streak": foreign_streak, "inst_streak": inst_streak}
    except Exception:
        return None


def _calc_consecutive_buy(df: pd.DataFrame, col: str) -> int:
    """최근부터 거슬러 올라가며 연속 순매수(양수) 일수를 계산."""
    if col not in df.columns:
        return 0
    vals = df[col].fillna(0).values
    streak = 0
    for i in range(len(vals) - 1, -1, -1):
        if vals[i] > 0:
            streak += 1
        else:
            break
    return streak


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

        # 외국인/기관 연속 순매수 일수 계산
        # 1차: CSV에서 계산
        foreign_streak = _calc_consecutive_buy(df, "Foreign_Net")
        inst_streak = _calc_consecutive_buy(df, "Inst_Net")
        # 2차: parquet에 더 최신 데이터가 있으면 사용
        pq_flow = _load_parquet_flow(ticker)
        if pq_flow:
            if pq_flow["foreign_streak"] > foreign_streak:
                foreign_streak = pq_flow["foreign_streak"]
            if pq_flow["inst_streak"] > inst_streak:
                inst_streak = pq_flow["inst_streak"]

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
            "foreign_streak": foreign_streak,
            "inst_streak": inst_streak,
        }
    except Exception:
        return None


def score_stocks(
    sector_stocks: pd.DataFrame,
    top_n: int = 3,
    refresh_flow: bool = True,
) -> list[dict]:
    """섹터 내 종목 점수화 + 상위 N개 반환.

    2단계 점수화:
    1차: 5기준(대기/거래량/MA120/시총/수급) 전종목 점수화.
        수급은 CSV→parquet 순으로 로드 (최근 갱신 안 됐으면 0).
    2차: 상위 2N개 후보 중 수급 미확보 종목만 pykrx 실시간 조회 → 보강.
    """

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
            elif cap > p70:
                score += SCORE_MIDCAP * 0.5
                reasons.append("시총상위")

        # 5. 외국인/기관 수급 흐름
        fs = c.get("foreign_streak", 0)
        is_ = c.get("inst_streak", 0)
        if fs >= 3 and is_ >= 3:
            score += SCORE_FLOW
            reasons.append(f"외인{fs}일+기관{is_}일연속매수")
        elif fs >= 3 or is_ >= 3:
            score += SCORE_FLOW * 0.7
            tag = f"외인{fs}일" if fs >= 3 else f"기관{is_}일"
            reasons.append(f"{tag}연속매수")
        elif fs >= 1 or is_ >= 1:
            score += SCORE_FLOW * 0.3
            parts = []
            if fs >= 1:
                parts.append(f"외인{fs}일")
            if is_ >= 1:
                parts.append(f"기관{is_}일")
            reasons.append(f"{'+'.join(parts)}매수")

        results.append({
            **c,
            "score": round(score),
            "reasons": reasons,
        })

    # 점수 내림차순 정렬
    results.sort(key=lambda x: x["score"], reverse=True)

    # 2단계: 상위 후보 수급 보강 (pykrx 실시간 조회)
    if refresh_flow:
        shortlist = results[: top_n * 2]  # 최종 N개의 2배 후보
        refreshed = 0
        for c in shortlist:
            # CSV/parquet에서 수급이 이미 있으면 건너뜀
            if c.get("foreign_streak", 0) > 0 or c.get("inst_streak", 0) > 0:
                continue
            flow = _fetch_flow_pykrx(c["ticker"])
            time.sleep(0.3)  # pykrx rate limit 방어
            if flow:
                c["foreign_streak"] = flow["foreign_streak"]
                c["inst_streak"] = flow["inst_streak"]
                # 수급 점수 재계산
                fs = flow["foreign_streak"]
                is_ = flow["inst_streak"]
                if fs >= 3 and is_ >= 3:
                    c["score"] += round(SCORE_FLOW)
                    c["reasons"].append(f"외인{fs}일+기관{is_}일연속매수")
                elif fs >= 3 or is_ >= 3:
                    c["score"] += round(SCORE_FLOW * 0.7)
                    tag = f"외인{fs}일" if fs >= 3 else f"기관{is_}일"
                    c["reasons"].append(f"{tag}연속매수")
                elif fs >= 1 or is_ >= 1:
                    c["score"] += round(SCORE_FLOW * 0.3)
                    parts = []
                    if fs >= 1:
                        parts.append(f"외인{fs}일")
                    if is_ >= 1:
                        parts.append(f"기관{is_}일")
                    c["reasons"].append(f"{'+'.join(parts)}매수")
                refreshed += 1
        if refreshed > 0:
            # 수급 보강 후 재정렬
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
          f"{'등락률':>7} {'거래량':>6} {'RSI':>5} {'외인':>4} {'기관':>4}  사유")
    print(f"  {'─' * 80}")

    for i, p in enumerate(picks, 1):
        reason_str = " | ".join(p["reasons"])
        fs = p.get("foreign_streak", 0)
        is_ = p.get("inst_streak", 0)
        fs_str = f"{fs}일" if fs > 0 else "-"
        is_str = f"{is_}일" if is_ > 0 else "-"
        print(f"  {i:>3}위 {p['name']:>12} ({p['ticker']}) "
              f"{p['score']:>3}점 "
              f"{p['change_pct']:>+6.1f}% "
              f"{p['vol_ratio']:>5.1f}x "
              f"{p['rsi']:>4.0f} "
              f"{fs_str:>4} {is_str:>4}  "
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
