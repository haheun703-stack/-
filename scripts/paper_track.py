"""PAPER 모의 원장 추적 (사장님 5/31 — 삼성SDI paper 진입 검증).

data/paper_ledger.json의 PAPER_OPEN 포지션을 최신 종가로 손익·손절·목표 점검.
실주문 0 / KIS 미접촉 / 기록·추적만. 매일(거래일) 실행하면 paper 성과 누적.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

LEDGER = PROJECT_ROOT / "data" / "paper_ledger.json"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def latest_close(ticker: str):
    f = PROCESSED / f"{ticker}.parquet"
    if not f.exists():
        return None, None
    df = pd.read_parquet(f).sort_index()
    df = df[df["close"] > 0]
    if len(df) == 0:
        return None, None
    return float(df["close"].iloc[-1]), str(df.index[-1].date())


def main() -> int:
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    print("=== PAPER 모의 원장 추적 (실주문 0) ===\n")
    for t in data["paper_trades"]:
        if t["status"] != "PAPER_OPEN":
            continue
        cur, asof = latest_close(t["ticker"])
        if cur is None:
            print(f"{t['name']}({t['ticker']}): 가격 없음")
            continue
        entry = t["entry_price"]
        pnl_pct = (cur / entry - 1) * 100
        pnl_won = int((cur - entry) * t["qty"])
        hit_stop = cur <= t["stop_loss_price"]
        status = "🔴 손절도달" if hit_stop else "🟢 보유중"
        print(f"{t['name']}({t['ticker']}) [{t['sector']}]")
        print(f"  진입 {entry:,}원 × {t['qty']}주 → 현재 {cur:,}원 ({asof})")
        print(f"  손익 {pnl_pct:+.1f}% ({pnl_won:+,}원) | 손절선 {t['stop_loss_price']:,} | {status}")
        print(f"  목표 D+{t['hold_target_days']} | 근거: {t['thesis'][:60]}...")
        print()
    print("★ 실주문 0 / KIS 미접촉 / AUTO_TRADING_ENABLED=0 불변. 매 거래일 실행하면 성과 누적.")
    print("  다음 거래일(6/1) 시가 확정 후 entry_price 재조정 가능.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
