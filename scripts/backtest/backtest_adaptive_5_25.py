"""
적응형 매매법 (MVP-1~4) 3개월 백테스트 — 2026-05-25
====================================================

백테스트 룰:
  R0 baseline: 손절 없음
  R1: -3% 즉시 매도
  R2: -5% 즉시 매도 ★ (퐝가님 지시)
  R3: -7% 즉시 매도
  M0: 매크로 가드 미적용
  M1: KOSPI(프록시=삼성전자 005930) -1.5%↓ 시 큐 매수 차단
  T0: Open가 매수
  T1: 09:10 이후 매수 → (Open + Close) / 2 가정

전략 (현재 코드 그대로 시뮬레이션):
  - MVP-2 분할매수 큐: 천장(30일 high)의 -10%/-20%/-30% 1주씩
  - MVP-2.5 Trailing Quick Profit: +7% ARMED → trailing_peak × 0.98 깨지면 매도
  - MVP-1 천장 -3%: 천장 신선도 5일 이내 + High ≥ peak × 0.97 → 매도
  - 미청산 포지션: 마지막 날 Close로 강제 청산

데이터: VPS /home/ubuntu/jgis/stock_data_daily/{한글명}_{ticker}.csv (39컬럼)
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================
# 설정
# ============================================================
JGIS_DIR = "/home/ubuntu/jgis/stock_data_daily"
KOSPI_PROXY_TICKER = "005930"  # 삼성전자

TARGET_TICKERS = [
    # 5/25 큐 등록 3
    "110990", "062040", "240810",
    # 큐 미등록 3
    "095340", "103590", "319400",
    # KOSPI 시총 상위 + 소부장
    "005930", "000660", "035420", "035720", "005380", "051910",
    "006400", "207940", "068270", "005490", "012330", "028260",
    "105560", "055550", "033780", "066570", "003550", "017670",
    "015760", "086790", "009150", "018260", "032830", "316140",
    "024110", "086280", "010130", "011200", "009540", "010950",
    "267250", "034730", "003670", "000270", "096770", "030200",
    "161390", "071050", "088980", "028050", "003490", "047810",
    "042660", "010140", "012450", "272210", "079550", "064350",
    "099320", "108320", "137310", "079430", "093370", "105840",
    "112610", "122870", "042700", "058470",
]

# MVP-2 분할 비율
SPLIT_LEVELS = [0.90, 0.80, 0.70]  # -10%, -20%, -30%
PEAK_WINDOW = 30                    # 천장 룩백 일수
PEAK_FRESHNESS = 5                  # MVP-1 천장 신선도
QUICK_PROFIT_ARM = 1.07             # +7% ARMED
QUICK_PROFIT_TRAIL = 0.98           # trailing_peak × 0.98 깨지면 매도
MVP1_SELL = 0.97                    # 천장 × 0.97 도달 시 매도

# 룰 그리드 (6개 핵심 조합 + T0/T1 별도)
RULE_GRID = [
    # (stop_loss_pct, macro_guard, time_avoid, label)
    (None, False, False, "R0_M0_T0"),
    (-3.0, False, False, "R1_M0_T0"),
    (-5.0, False, False, "R2_M0_T0"),
    (-7.0, False, False, "R3_M0_T0"),
    (-5.0, True,  False, "R2_M1_T0"),
    (-5.0, False, True,  "R2_M0_T1"),
    (None, True,  False, "R0_M1_T0"),
    (-3.0, True,  False, "R1_M1_T0"),
]


# ============================================================
# 데이터 로딩
# ============================================================
def list_csvs(jgis_dir: str) -> dict[str, str]:
    """ticker → 파일 경로 매핑"""
    paths = {}
    for p in glob.glob(os.path.join(jgis_dir, "*.csv")):
        name = os.path.basename(p)
        # {한글}_{ticker}.csv
        if "_" in name:
            t = name.rsplit("_", 1)[1].replace(".csv", "")
            paths[t] = p
    return paths


def load_csv(path: str) -> list[dict]:
    """CSV → list[dict] (날짜 오름차순)"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "Date": r["Date"],
                    "Open": float(r["Open"]) if r["Open"] else None,
                    "High": float(r["High"]) if r["High"] else None,
                    "Low": float(r["Low"]) if r["Low"] else None,
                    "Close": float(r["Close"]) if r["Close"] else None,
                })
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda x: x["Date"])
    # None 행 제거
    rows = [r for r in rows if r["Open"] and r["Close"] and r["Low"] and r["High"]]
    return rows


def build_kospi_daily_change(kospi_rows: list[dict]) -> dict[str, float]:
    """KOSPI 프록시: 전일 대비 변동률 (%)"""
    out = {}
    prev_close = None
    for r in kospi_rows:
        if prev_close and prev_close > 0:
            out[r["Date"]] = (r["Close"] - prev_close) / prev_close * 100.0
        prev_close = r["Close"]
    return out


# ============================================================
# 백테스트 엔진
# ============================================================
@dataclass
class Trade:
    ticker: str
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float
    sell_reason: str  # STOP_LOSS / QUICK_PROFIT / PEAK_3PCT / FORCED_CLOSE
    holding_days: int
    pnl_pct: float
    d1_pct: Optional[float] = None
    d3_pct: Optional[float] = None
    d5_pct: Optional[float] = None
    stage: int = 0  # L1/L2/L3


@dataclass
class Position:
    ticker: str
    buy_date: str
    buy_price: float
    buy_idx: int           # rows 내 인덱스
    stage: int
    trailing_armed: bool = False
    trailing_peak: float = 0.0


def get_peak_at(rows: list[dict], idx: int, window: int = PEAK_WINDOW) -> tuple[float, int]:
    """rows[idx] 시점 기준 직전 window일의 최고가 + 그 날 인덱스"""
    start = max(0, idx - window)
    sub = rows[start:idx]
    if not sub:
        return rows[idx]["High"], idx
    peak = max(sub, key=lambda r: r["High"])
    peak_idx = start + sub.index(peak)
    return peak["High"], peak_idx


def simulate_ticker(
    ticker: str,
    rows: list[dict],
    kospi_change: dict[str, float],
    stop_loss_pct: Optional[float],
    macro_guard: bool,
    time_avoid: bool,
) -> list[Trade]:
    """단일 종목 시뮬레이션"""
    trades: list[Trade] = []
    positions: list[Position] = []
    # 큐 상태: stage 별로 PENDING(미체결) / FILLED(체결완료, 청산 후 재개 안 함)
    stage_filled = {1: False, 2: False, 3: False}

    n = len(rows)
    # 천장 룩백을 위해 PEAK_WINDOW 이후부터 시작
    start_idx = PEAK_WINDOW
    if n <= start_idx + 1:
        return trades

    for i in range(start_idx, n):
        today = rows[i]
        date = today["Date"]

        # 매크로 가드 (KOSPI -1.5%↓)
        kospi_blocked = False
        if macro_guard:
            chg = kospi_change.get(date)
            if chg is not None and chg <= -1.5:
                kospi_blocked = True

        # ----- 천장 재계산 (전일까지의 30일 최고) -----
        peak, peak_idx = get_peak_at(rows, i, PEAK_WINDOW)

        # ----- 1. 큐 매수 확인 -----
        if not kospi_blocked:
            for stage_no, ratio in enumerate(SPLIT_LEVELS, start=1):
                if stage_filled[stage_no]:
                    continue
                target_price = peak * ratio
                if today["Low"] <= target_price:
                    # 매수 가격 결정
                    if time_avoid:
                        # T1: 09:10 이후 진입. 시초가 갭하락 회피 → "Open과 target_price 중 비싼 쪽"
                        # 즉 갭하락(Open이 target보다 낮음)이면 10분 후 반등 가정(target_price 사용)
                        # 갭상승(Open이 target보다 높음)이면 그래도 Low에 닿았으니 target_price 사용
                        # 다만 시초가 회피 슬리피지로 target_price × 1.005 (+0.5%) 적용
                        entry_price = target_price * 1.005
                        # 단, today High를 초과하면 매수 무효 처리(체결 불가)
                        if entry_price > today["High"]:
                            continue
                    else:
                        # T0: Open이 target_price보다 낮으면 Open, 아니면 target_price
                        entry_price = min(target_price, today["Open"]) if today["Open"] <= target_price else target_price
                    positions.append(Position(
                        ticker=ticker,
                        buy_date=date,
                        buy_price=entry_price,
                        buy_idx=i,
                        stage=stage_no,
                    ))
                    stage_filled[stage_no] = True

        # ----- 2. 보유 종목 매도 검사 -----
        new_positions = []
        for pos in positions:
            sold = False
            sell_reason = ""
            sell_price = 0.0

            # (a) 자동 손절 (룰 1) — 우선순위 최고
            if stop_loss_pct is not None:
                stop_price = pos.buy_price * (1 + stop_loss_pct / 100.0)
                if today["Low"] <= stop_price:
                    sell_price = stop_price
                    sell_reason = "STOP_LOSS"
                    sold = True

            # (b) Trailing Quick Profit
            if not sold:
                arm_price = pos.buy_price * QUICK_PROFIT_ARM
                if not pos.trailing_armed:
                    if today["High"] >= arm_price:
                        pos.trailing_armed = True
                        pos.trailing_peak = today["High"]
                else:
                    # 이미 ARMED면 today High로 trailing_peak 갱신
                    if today["High"] > pos.trailing_peak:
                        pos.trailing_peak = today["High"]
                    # trailing_peak × 0.98 깨졌나
                    trail_stop = pos.trailing_peak * QUICK_PROFIT_TRAIL
                    if today["Low"] <= trail_stop:
                        sell_price = trail_stop
                        sell_reason = "QUICK_PROFIT"
                        sold = True

            # (c) MVP-1 천장 -3% 매도
            if not sold:
                # 보유 시점 천장 신선도 검사
                fresh = (i - peak_idx) <= PEAK_FRESHNESS
                if fresh and today["High"] >= peak * MVP1_SELL:
                    sell_price = peak * MVP1_SELL
                    sell_reason = "PEAK_3PCT"
                    sold = True

            if sold:
                holding_days = i - pos.buy_idx
                pnl_pct = (sell_price - pos.buy_price) / pos.buy_price * 100.0

                # D+1/D+3/D+5 종가 기준
                def fwd_pct(offset: int) -> Optional[float]:
                    j = pos.buy_idx + offset
                    if j >= n:
                        return None
                    return (rows[j]["Close"] - pos.buy_price) / pos.buy_price * 100.0

                trades.append(Trade(
                    ticker=ticker,
                    buy_date=pos.buy_date,
                    buy_price=pos.buy_price,
                    sell_date=date,
                    sell_price=sell_price,
                    sell_reason=sell_reason,
                    holding_days=holding_days,
                    pnl_pct=pnl_pct,
                    d1_pct=fwd_pct(1),
                    d3_pct=fwd_pct(3),
                    d5_pct=fwd_pct(5),
                    stage=pos.stage,
                ))
                # 청산하면 stage는 다시 사용 가능하지 않음 (1회 큐)
            else:
                new_positions.append(pos)
        positions = new_positions

    # ----- 미청산 강제 청산 -----
    if positions:
        last = rows[-1]
        last_idx = n - 1
        for pos in positions:
            sell_price = last["Close"]
            pnl_pct = (sell_price - pos.buy_price) / pos.buy_price * 100.0

            def fwd_pct(offset: int) -> Optional[float]:
                j = pos.buy_idx + offset
                if j >= n:
                    return None
                return (rows[j]["Close"] - pos.buy_price) / pos.buy_price * 100.0

            trades.append(Trade(
                ticker=ticker,
                buy_date=pos.buy_date,
                buy_price=pos.buy_price,
                sell_date=last["Date"],
                sell_price=sell_price,
                sell_reason="FORCED_CLOSE",
                holding_days=last_idx - pos.buy_idx,
                pnl_pct=pnl_pct,
                d1_pct=fwd_pct(1),
                d3_pct=fwd_pct(3),
                d5_pct=fwd_pct(5),
                stage=pos.stage,
            ))

    return trades


# ============================================================
# 집계
# ============================================================
def aggregate(trades: list[Trade]) -> dict:
    if not trades:
        return {
            "n_trades": 0,
            "avg_holding_days": 0.0,
            "win_rate": 0.0,
            "avg_d1": 0.0,
            "avg_d3": 0.0,
            "avg_d5": 0.0,
            "rate_5pct": 0.0,
            "rate_10pct": 0.0,
            "mdd": 0.0,
            "pf": 0.0,
            "cum_pnl_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "n_stop_loss": 0,
            "n_quick_profit": 0,
            "n_peak_3pct": 0,
            "n_forced_close": 0,
        }

    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    losses_sum = sum(t.pnl_pct for t in trades if t.pnl_pct < 0)
    gains_sum = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    pf = gains_sum / abs(losses_sum) if losses_sum < 0 else float("inf") if gains_sum > 0 else 0.0

    d1_vals = [t.d1_pct for t in trades if t.d1_pct is not None]
    d3_vals = [t.d3_pct for t in trades if t.d3_pct is not None]
    d5_vals = [t.d5_pct for t in trades if t.d5_pct is not None]

    # +5%↑, +10%↑: 매도 시점 기준 (실현 수익률)
    rate_5pct = sum(1 for t in trades if t.pnl_pct >= 5.0) / n * 100.0
    rate_10pct = sum(1 for t in trades if t.pnl_pct >= 10.0) / n * 100.0

    mdd = min(t.pnl_pct for t in trades)
    cum = sum(t.pnl_pct for t in trades)

    return {
        "n_trades": n,
        "avg_holding_days": round(sum(t.holding_days for t in trades) / n, 2),
        "win_rate": round(wins / n * 100.0, 2),
        "avg_d1": round(sum(d1_vals) / len(d1_vals), 2) if d1_vals else 0.0,
        "avg_d3": round(sum(d3_vals) / len(d3_vals), 2) if d3_vals else 0.0,
        "avg_d5": round(sum(d5_vals) / len(d5_vals), 2) if d5_vals else 0.0,
        "rate_5pct": round(rate_5pct, 2),
        "rate_10pct": round(rate_10pct, 2),
        "mdd": round(mdd, 2),
        "pf": round(pf, 2) if pf != float("inf") else 99.99,
        "cum_pnl_pct": round(cum, 2),
        "avg_pnl_pct": round(cum / n, 2),
        "n_stop_loss": sum(1 for t in trades if t.sell_reason == "STOP_LOSS"),
        "n_quick_profit": sum(1 for t in trades if t.sell_reason == "QUICK_PROFIT"),
        "n_peak_3pct": sum(1 for t in trades if t.sell_reason == "PEAK_3PCT"),
        "n_forced_close": sum(1 for t in trades if t.sell_reason == "FORCED_CLOSE"),
    }


# ============================================================
# 메인
# ============================================================
def main(jgis_dir: str = JGIS_DIR, out_dir: str = "."):
    print(f"[1/4] CSV 인덱싱: {jgis_dir}")
    paths = list_csvs(jgis_dir)
    print(f"  전체 CSV: {len(paths)}개")

    # 타겟 종목 필터
    available = {t: paths[t] for t in TARGET_TICKERS if t in paths}
    missing = [t for t in TARGET_TICKERS if t not in paths]
    print(f"  타겟 매칭: {len(available)}/{len(TARGET_TICKERS)}개 (누락 {len(missing)})")
    if missing:
        print(f"  누락 ticker: {missing}")

    # KOSPI 프록시 로드
    if KOSPI_PROXY_TICKER not in paths:
        print(f"  ! KOSPI 프록시 {KOSPI_PROXY_TICKER} 없음. M1 비활성")
        kospi_change = {}
    else:
        print(f"[2/4] KOSPI 프록시 ({KOSPI_PROXY_TICKER}) 로딩")
        kospi_rows = load_csv(paths[KOSPI_PROXY_TICKER])
        kospi_change = build_kospi_daily_change(kospi_rows)
        print(f"  KOSPI 일별 변동률 {len(kospi_change)}일")

    # 종목별 데이터 로드
    print(f"[3/4] 종목 데이터 로딩")
    data: dict[str, list[dict]] = {}
    for t, p in available.items():
        rows = load_csv(p)
        if len(rows) >= PEAK_WINDOW + 5:
            data[t] = rows
    print(f"  유효 종목 (>={PEAK_WINDOW+5}행): {len(data)}개")

    # 데이터 범위
    all_dates = []
    for rows in data.values():
        all_dates.append(rows[0]["Date"])
        all_dates.append(rows[-1]["Date"])
    if all_dates:
        date_min = min(all_dates)
        date_max = max(all_dates)
    else:
        date_min = date_max = "N/A"
    print(f"  데이터 범위: {date_min} ~ {date_max}")

    # 룰별 시뮬레이션
    print(f"[4/4] 룰 그리드 시뮬레이션 ({len(RULE_GRID)}개 조합)")
    results = {}
    detailed = {}
    for stop_loss_pct, macro_guard, time_avoid, label in RULE_GRID:
        all_trades: list[Trade] = []
        for t, rows in data.items():
            trades = simulate_ticker(
                ticker=t,
                rows=rows,
                kospi_change=kospi_change,
                stop_loss_pct=stop_loss_pct,
                macro_guard=macro_guard,
                time_avoid=time_avoid,
            )
            all_trades.extend(trades)
        agg = aggregate(all_trades)
        results[label] = agg
        detailed[label] = [asdict(t) for t in all_trades]
        print(f"  {label}: trades={agg['n_trades']:>4}, win={agg['win_rate']:>5.2f}%, "
              f"D1={agg['avg_d1']:>+6.2f}%, MDD={agg['mdd']:>+6.2f}%, "
              f"PF={agg['pf']:>5.2f}, cum={agg['cum_pnl_pct']:>+8.2f}%")

    # JSON 출력
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "backtest_adaptive_5_25_results.json")
    payload = {
        "meta": {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_range": [date_min, date_max],
            "n_tickers_target": len(TARGET_TICKERS),
            "n_tickers_used": len(data),
            "missing_tickers": missing,
            "rule_grid_count": len(RULE_GRID),
            "kospi_proxy": KOSPI_PROXY_TICKER,
            "split_levels": SPLIT_LEVELS,
            "peak_window": PEAK_WINDOW,
            "peak_freshness": PEAK_FRESHNESS,
            "quick_profit_arm": QUICK_PROFIT_ARM,
            "quick_profit_trail": QUICK_PROFIT_TRAIL,
            "mvp1_sell": MVP1_SELL,
        },
        "results": results,
        "trades_detail": detailed,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 JSON 저장: {out_json}")
    return payload


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(out_dir=out_dir)
