"""[C] 릴레이 포지션 청산 조건.

3가지 청산 트리거를 매일 체크:

  1. 목표 수익 도달
     승률 80%+: 목표 +8%
     승률 70%+: 목표 +6%
     승률 60%+: 목표 +5%

  2. 릴레이 완료 신호
     후행 섹터가 발화 조건 충족 시 (avg_return +3.5% & breadth 60%+)
     → "릴레이 소진, 다음 섹터로 이동" 청산

  3. 래그 초과 타임아웃
     설정 래그 +1일 지나도 후행 섹터 무반응 시 손절
     예: 래그1일 패턴인데 2일째 섹터 등락률 0% 이하 → 청산

사용법:
  # 코드에서 호출
  from relay_exit import check_exit_conditions
  result = check_exit_conditions(position_info, current_data)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"

# 목표 수익률 (승률 기반)
PROFIT_TARGETS = [
    (80, 8.0),    # 승률 80%+ → +8%
    (70, 6.0),    # 승률 70%+ → +6%
    (60, 5.0),    # 승률 60%+ → +5%
    (50, 4.0),    # 승률 50%+ → +4%
]

# 손절 기준
STOP_LOSS = -5.0          # 최대 손절 -5%
FIRE_THRESHOLD = 3.5      # 릴레이 완료 감지 기준
BREADTH_THRESHOLD = 0.6   # 릴레이 완료 breadth 기준


def get_profit_target(win_rate: float) -> float:
    """승률 기반 목표 수익률 반환."""
    for wr_min, target in PROFIT_TARGETS:
        if win_rate >= wr_min:
            return target
    return 3.0


def check_exit_conditions(
    entry_price: float,
    current_price: float,
    entry_date: str,
    current_date: str,
    win_rate: float,
    best_lag: int,
    follow_sector_return: float = 0.0,
    follow_sector_breadth: float = 0.0,
    trading_days_held: int = 0,
) -> dict:
    """청산 조건 체크.

    Args:
        entry_price: 진입 가격
        current_price: 현재 가격
        entry_date: 진입일 (YYYY-MM-DD)
        current_date: 현재일 (YYYY-MM-DD)
        win_rate: 패턴 승률 (%)
        best_lag: 패턴 최적 래그 (일)
        follow_sector_return: 후행 섹터 오늘 평균 등락률 (%)
        follow_sector_breadth: 후행 섹터 오늘 상승비율 (0~1)
        trading_days_held: 보유 거래일 수

    Returns:
        {exit: bool, reason: str, action: str, pnl_pct: float, details: dict}
    """
    if entry_price <= 0:
        return {"exit": False, "reason": "진입가 오류", "action": "HOLD",
                "pnl_pct": 0, "details": {}}

    pnl_pct = (current_price - entry_price) / entry_price * 100
    profit_target = get_profit_target(win_rate)

    details = {
        "pnl_pct": round(pnl_pct, 2),
        "profit_target": profit_target,
        "stop_loss": STOP_LOSS,
        "days_held": trading_days_held,
        "lag_timeout": best_lag + 1,
        "follow_return": follow_sector_return,
        "follow_breadth": follow_sector_breadth,
    }

    # ── 트리거 1: 목표 수익 도달 ──
    if pnl_pct >= profit_target:
        return {
            "exit": True,
            "reason": f"목표 수익 도달 ({pnl_pct:+.1f}% >= +{profit_target}%)",
            "action": "SELL_TARGET",
            "pnl_pct": round(pnl_pct, 2),
            "details": details,
        }

    # ── 트리거 2: 손절 ──
    if pnl_pct <= STOP_LOSS:
        return {
            "exit": True,
            "reason": f"손절 ({pnl_pct:+.1f}% <= {STOP_LOSS}%)",
            "action": "SELL_STOPLOSS",
            "pnl_pct": round(pnl_pct, 2),
            "details": details,
        }

    # ── 트리거 3: 릴레이 완료 (후행 섹터 발화) ──
    if (follow_sector_return >= FIRE_THRESHOLD
            and follow_sector_breadth >= BREADTH_THRESHOLD):
        return {
            "exit": True,
            "reason": (f"릴레이 완료 — 후행 섹터 발화 "
                       f"({follow_sector_return:+.1f}%, "
                       f"breadth {follow_sector_breadth*100:.0f}%)"),
            "action": "SELL_RELAY_DONE",
            "pnl_pct": round(pnl_pct, 2),
            "details": details,
        }

    # ── 트리거 4: 래그 초과 타임아웃 ──
    timeout_days = best_lag + 1
    if trading_days_held >= timeout_days and follow_sector_return <= 0:
        return {
            "exit": True,
            "reason": (f"래그 초과 타임아웃 — "
                       f"{trading_days_held}일 보유, "
                       f"후행 섹터 {follow_sector_return:+.1f}% 무반응"),
            "action": "SELL_TIMEOUT",
            "pnl_pct": round(pnl_pct, 2),
            "details": details,
        }

    # ── 보유 유지 ──
    return {
        "exit": False,
        "reason": "보유 유지",
        "action": "HOLD",
        "pnl_pct": round(pnl_pct, 2),
        "details": details,
    }


def get_sector_today_return(sector: str) -> tuple[float, float]:
    """섹터의 오늘 평균 등락률과 breadth 계산.

    Returns:
        (avg_return, breadth)
    """
    naver_map = pd.read_csv(
        DATA_DIR / "naver_sector_map.csv", dtype={"ticker": str}
    )
    sector_tickers = naver_map[naver_map["sector"] == sector]["ticker"].tolist()

    returns = []
    for ticker in sector_tickers:
        matches = list(DAILY_DIR.glob(f"*_{ticker}.csv"))
        if not matches:
            continue
        try:
            df = pd.read_csv(matches[0], usecols=["Date", "Close"])
            df = df.dropna().sort_values("Date")
            if len(df) < 2:
                continue
            ret = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
            returns.append(ret)
        except Exception:
            continue

    if not returns:
        return 0.0, 0.0

    avg_ret = float(np.mean(returns))
    breadth = sum(1 for r in returns if r > 0) / len(returns)
    return round(avg_ret, 2), round(breadth, 2)


def main():
    """예시 실행."""
    print(f"\n{'=' * 50}")
    print(f"  릴레이 청산 조건 체크 (예시)")
    print(f"{'=' * 50}")

    # 예시: 증권→생명보험 래그1일, 승률 70.6%
    # 진입가 88,000 → 현재 93,000
    result = check_exit_conditions(
        entry_price=88000,
        current_price=93000,
        entry_date="2026-02-19",
        current_date="2026-02-20",
        win_rate=70.6,
        best_lag=1,
        follow_sector_return=17.4,
        follow_sector_breadth=0.8,
        trading_days_held=1,
    )

    print(f"\n  진입가: 88,000 → 현재가: 93,000")
    print(f"  수익률: {result['pnl_pct']:+.1f}%")
    print(f"  판정: {result['action']}")
    print(f"  사유: {result['reason']}")
    print(f"")

    d = result["details"]
    print(f"  목표: +{d['profit_target']}%")
    print(f"  손절: {d['stop_loss']}%")
    print(f"  보유일: {d['days_held']}일 (타임아웃: {d['lag_timeout']}일)")
    print(f"  후행 섹터: {d['follow_return']:+.1f}% / breadth {d['follow_breadth']*100:.0f}%")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
