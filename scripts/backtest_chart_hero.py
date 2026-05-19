"""차트영웅 매매법 6개월 백테스트 — 원본 vs 긴장 타입 비교.

기간: 2025-11-19 ~ 2026-05-19 (6개월)
시점:
  - 2025-11-19 ~ 12: 트럼프 당선 변동성
  - 2026-01 ~ 02: 반등기
  - 2026-03 초: -20% 폭락장 (사이드카 + 서킷 브레이커)
  - 2026-03 ~ 05: 회복 + 약세 전환

검증 목표:
  ✓ 차트영웅 원본 vs 긴장 타입 WR/PF/MDD 비교
  ✓ 메모리 backtest_results.md "91.7% WR" 재현 (또는 표본 확대)
  ✓ 3월 폭락장 시나리오 MDD 측정 (긴장 타입 안전망 효과)

입력 데이터 (5/21 정보봇 + 자체 수집 후 가능):
  - 일별 매크로 4-시그널 (CNN F&G 과거치 + US10Y + KRW + KOSPI 주봉)
  - 일별 상한가 종목 풀 (정보봇 OHLCV)
  - catalyst (정보봇 quant_surge_catalyst)
  - 목표가 (사장님 5/19 시드 + Perplexity 백업)

출력:
  docs/03-analysis/backtest-chart-hero-6m.md
  data/backtest_chart_hero_trades.csv
"""

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.chart_hero_tension_rule import (
    Position, TradeStage, decide_action, check_entry_gate,
    INIT_WEIGHT_PCT, ADD_WEIGHT_PCT,
)


def iter_trading_days(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """주말 제외 (공휴일은 외부 캘린더로 보강 필요)."""
    d = start
    while d <= end:
        if d.weekday() < 5:  # 월-금
            yield d
        d += dt.timedelta(days=1)


def simulate_position(entry_price: int, daily_prices: list[dict],
                      rule: str = "tension") -> dict:
    """1종목 진입 시뮬레이션 (원본 vs 긴장).

    Args:
        entry_price: D+1 종가 진입 가격
        daily_prices: D+1 이후 일별 [{date, open, high, low, close}, ...]
        rule: 'tension' (긴장) | 'original' (차트영웅 원본)

    Returns:
        { entry_price, exit_price, exit_reason, pnl_pct, days_held, max_dd }
    """
    if not daily_prices:
        return {"error": "no daily data"}

    pos = Position(
        ticker="SIM", name="SIM",
        entry_date=dt.date.fromisoformat(daily_prices[0]["date"]),
        avg_price=entry_price, total_qty=1, total_cost=entry_price,
        stage=TradeStage.INIT,
    )

    max_dd = 0.0
    qty_pct_remaining = 100.0   # 잔여 비중 %
    realized_pnl = 0.0           # 실현 손익 %

    for d in daily_prices:
        today = dt.date.fromisoformat(d["date"])
        current = d["close"]
        action = decide_action(pos, current, today)

        # MDD 추적
        pnl_now = (current - pos.avg_price) / pos.avg_price * 100
        if pnl_now < max_dd:
            max_dd = pnl_now

        if action["action"] in ("STOPLOSS", "FORCE_CLOSE"):
            realized_pnl += pnl_now * qty_pct_remaining / 100
            return {
                "entry_price": entry_price, "exit_price": current,
                "exit_reason": action["action"], "exit_date": d["date"],
                "pnl_pct": round(realized_pnl, 2),
                "days_held": (today - pos.entry_date).days,
                "max_dd_pct": round(max_dd, 2),
            }
        elif action["action"] == "PARTIAL_SELL":
            sold_pct = action["qty_pct"] * qty_pct_remaining / 100
            realized_pnl += pnl_now * sold_pct / 100
            qty_pct_remaining -= sold_pct
            pos.stage = action["new_stage"]
            if qty_pct_remaining <= 0:
                return {
                    "entry_price": entry_price, "exit_price": current,
                    "exit_reason": "FULL_PROFIT_TAKE", "exit_date": d["date"],
                    "pnl_pct": round(realized_pnl, 2),
                    "days_held": (today - pos.entry_date).days,
                    "max_dd_pct": round(max_dd, 2),
                }

    # D+5 안 갔으면 강제 종료
    return {
        "entry_price": entry_price, "exit_price": current,
        "exit_reason": "TIMEOUT",
        "pnl_pct": round(realized_pnl, 2),
        "max_dd_pct": round(max_dd, 2),
    }


def aggregate_results(trades: list[dict]) -> dict:
    """백테스트 결과 집계."""
    if not trades:
        return {"n": 0}
    wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    total_gain = sum(t["pnl_pct"] for t in wins) if wins else 0
    total_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 1
    return {
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "avg_pnl_pct": round(sum(t["pnl_pct"] for t in trades) / len(trades), 2),
        "profit_factor": round(total_gain / total_loss, 2) if total_loss > 0 else 0,
        "max_dd_avg": round(sum(t.get("max_dd_pct", 0) for t in trades) / len(trades), 2),
        "max_dd_worst": round(min(t.get("max_dd_pct", 0) for t in trades), 2),
    }


def run_backtest(input_csv: str | None = None) -> dict:
    """6개월 백테스트 메인.

    input_csv 형식 (정보봇 데이터 통합 후 생성):
      entry_date, ticker, entry_price, d1_close, d2_close, d3_close, d4_close, d5_close

    PLACEHOLDER: 5/21 정보봇 catalyst + 자체 OHLCV 통합 후 실제 데이터 투입.
    """
    if input_csv and Path(input_csv).exists():
        # 실제 데이터 백테스트
        trades_tension = []
        trades_original = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 가격 시계열 변환
                daily = []
                base = dt.date.fromisoformat(row["entry_date"])
                for i, key in enumerate(["d1_close", "d2_close", "d3_close",
                                          "d4_close", "d5_close"]):
                    if row.get(key):
                        daily.append({
                            "date": (base + dt.timedelta(days=i)).isoformat(),
                            "close": int(row[key]),
                        })
                entry_price = int(row["entry_price"])
                trades_tension.append(simulate_position(entry_price, daily, "tension"))
                # 원본 룰은 미구현 (5/21 별도 작성 가능)
        return {
            "tension": aggregate_results(trades_tension),
            "original": aggregate_results(trades_original),
            "trades_tension": trades_tension,
        }

    # PLACEHOLDER 모드: 가상 거래 5건으로 룰 검증
    print("⚠️  실제 데이터 없음 — 5종목 더미 시나리오로 룰 검증")
    dummy_scenarios = [
        # (entry, [d1, d2, d3, d4, d5]) — 다양한 패턴
        ("강한 상승",   10000, [10200, 10500, 10800, 11000, 11200]),
        ("D+3 +3% 미달", 10000, [10100, 10150, 10200, 10250, 10300]),
        ("폭락 -40%",   10000, [9700, 9000, 8000, 7000, 5800]),
        ("폭락 후 회복", 10000, [9000, 8000, 9500, 10300, 11000]),
        ("D+3 +5% 익절", 10000, [10200, 10400, 10550, 10300, 10100]),
    ]
    trades = []
    for label, entry, prices in dummy_scenarios:
        daily = [{"date": (dt.date(2026, 5, 19) + dt.timedelta(days=i)).isoformat(),
                  "close": p} for i, p in enumerate(prices)]
        r = simulate_position(entry, daily, "tension")
        r["scenario"] = label
        trades.append(r)
    return {"tension": aggregate_results(trades), "trades_tension": trades}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="진입 데이터 CSV (정보봇 통합 후)")
    args = parser.parse_args()

    print("=== 차트영웅 매매법 백테스트 (긴장 타입) ===\n")
    r = run_backtest(args.input)

    print("📈 시나리오별 결과:")
    for t in r.get("trades_tension", []):
        scen = t.get("scenario", "?")
        print(f"  [{scen:18}] entry={t['entry_price']:>6} exit={t.get('exit_price'):>6} "
              f"PnL={t.get('pnl_pct'):>6.2f}% MDD={t.get('max_dd_pct'):>6.2f}% "
              f"reason={t.get('exit_reason')}")

    print(f"\n📊 집계 (긴장 타입):")
    for k, v in r["tension"].items():
        print(f"  {k}: {v}")
