"""
v11.0 백테스트 v2 — 현실적 자본 시뮬레이션

핵심 차이:
  - 초기 자본 1억, 종목당 배분
  - 최대 동시 보유 5종목
  - 일일 최대 신규 진입 2종목
  - 슬리피지 0.3%
  - 포지션 사이징 (Grade × Freshness)
  - US Overnight 레짐 연동 (가용 슬롯 제한)
  - 익절: +10% 반매도 + 트레일링 -5%
  - 손절: max(ATR×2, -7%), 캡 -10%
  - 시간 손절: 10일 내 +3% 미도달 시 청산

비교:
  A) v10.1 시그널 + 리스크 없음 (기존 방식)
  B) v10.1 시그널 + 포지션 사이징
  C) v10.1 시그널 + 포지션 사이징 + 레짐 캡
  D) v10.1 시그널 + 포지션 사이징 + 레짐 캡 + 고급 익절/손절
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

# ── 설정 ──
START_DATE = "2025-03-01"
END_DATE = "2026-02-13"
INITIAL_CAPITAL = 100_000_000  # 1억
MAX_POSITIONS = 5
MAX_DAILY_ENTRY = 2
SLIPPAGE = 0.005  # 0.5% (한국 중형주 현실적)
COMMISSION = 0.00015  # 편도
TAX = 0.0018  # 매도세


@dataclass
class Position:
    ticker: str
    name: str
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    allocated: float  # 투입금
    stop_loss: float
    target: float
    grade: str
    freshness: float
    counter: int
    # 트레일링
    peak_price: float = 0.0
    half_sold: bool = False
    half_sold_pnl: float = 0.0
    days_held: int = 0


@dataclass
class DayResult:
    date: pd.Timestamp
    equity: float
    cash: float
    positions: int
    daily_pnl: float = 0.0


def load_parquets():
    pq_dir = Path("data/processed")
    data = {}
    for f in sorted(pq_dir.glob("*.parquet")):
        ticker = f.stem
        if len(ticker) == 6 and ticker.isdigit():
            df = pd.read_parquet(f)
            if len(df) > 252:
                data[ticker] = df
    return data


def load_name_map():
    name_map = {}
    csv_dir = Path("stock_data_daily")
    if csv_dir.exists():
        import re
        for f in csv_dir.glob("*.csv"):
            m = re.search(r"_(\d{6})$", f.stem)
            if m:
                name_map[m.group(1)] = f.stem[:f.stem.rfind("_")]
    return name_map


def calc_trix_counter(df, idx):
    trix = df.get("trix")
    sig = df.get("trix_signal")
    if trix is None or sig is None:
        return 0
    t_now, s_now = trix.iloc[idx], sig.iloc[idx]
    if pd.isna(t_now) or pd.isna(s_now):
        return 0
    above = t_now > s_now
    counter = 0
    for i in range(idx, max(idx - 60, -1), -1):
        t, s = trix.iloc[i], sig.iloc[i]
        if pd.isna(t) or pd.isna(s):
            break
        if above and t > s:
            counter += 1
        elif not above and t <= s:
            counter -= 1
        else:
            break
    return counter


def calc_freshness(counter, rsi, vol_c):
    if counter <= 0 or counter >= 16 or vol_c > 2.0:
        return 0.0
    if counter >= 8 and rsi > 65:
        return 0.0
    c = 1.15 if counter <= 3 else (1.00 if counter <= 7 else 0.70)
    r = 1.00 if rsi <= 55 else (0.90 if rsi <= 65 else 0.75)
    v = 1.05 if vol_c < 0.7 else (1.00 if vol_c <= 1.3 else 0.85)
    return round(c * r * v, 3)


def calc_di(df, idx, period=14):
    if len(df) < period + 2 or idx < period + 1:
        return 0.0, 0.0
    h = df["high"].values
    lo = df["low"].values
    p = sum(max(0, h[i] - h[i - 1]) for i in range(idx - period + 1, idx + 1))
    m = sum(max(0, lo[i - 1] - lo[i]) for i in range(idx - period + 1, idx + 1))
    return p, m


def get_us_regime(df_ref, idx):
    """US Overnight 레짐 (간이: 전일 수익률 기반)"""
    if idx < 5:
        return "NEUTRAL"
    closes = df_ref["close"].values
    ret_5d = (closes[idx] / closes[idx - 5] - 1) * 100
    if ret_5d > 3:
        return "STRONG_BULL"
    elif ret_5d > 1:
        return "BULL"
    elif ret_5d > -1:
        return "NEUTRAL"
    elif ret_5d > -3:
        return "BEAR"
    else:
        return "STRONG_BEAR"


REGIME_SLOTS = {
    "STRONG_BULL": 5,
    "BULL": 4,
    "NEUTRAL": 3,
    "BEAR": 2,
    "STRONG_BEAR": 0,
}


def scan_signals(data_dict, day_idx_map):
    """v10.1 시그널 스캔 (TRIX GC + Freshness)"""
    signals = []
    for ticker, df in data_dict.items():
        if ticker not in day_idx_map:
            continue
        idx = day_idx_map[ticker]
        if idx < 120:
            continue

        row = df.iloc[idx]
        close = float(row.get("close", 0) or 0)
        if close < 5000:
            continue

        adx = float(row.get("adx_14", 0) or 0)
        rsi = float(row.get("rsi_14", 50) or 50)
        trix_val = float(row.get("trix", 0) or 0)
        trix_sig = float(row.get("trix_signal", 0) or 0)
        macd = float(row.get("macd", 0) or 0)
        macd_sig = float(row.get("macd_signal", 0) or 0)
        ma120 = float(row.get("sma_120", 0) or 0)

        if adx < 15:
            continue
        plus_di, minus_di = calc_di(df, idx)
        if plus_di < minus_di and adx > 20:
            continue

        # Trigger
        trix_above = trix_val > trix_sig
        macd_above = macd > macd_sig
        if not (trix_above and macd_above) and not trix_above:
            continue

        trigger = "confirm" if trix_above and macd_above else "impulse"

        # Counter + Freshness
        counter = calc_trix_counter(df, idx)
        vol = df["volume"].values
        vol_5 = np.mean(vol[max(0, idx - 4):idx + 1])
        vol_20 = np.mean(vol[max(0, idx - 19):idx + 1])
        vol_c = vol_5 / vol_20 if vol_20 > 0 else 1.0

        freshness = calc_freshness(counter, rsi, vol_c)
        if freshness == 0.0:
            continue

        # K11: 폭락
        h252 = df["high"].iloc[max(0, idx - 252):idx + 1].max()
        if h252 > 0 and (close / h252 - 1) < -0.20:
            continue
        # K12: MA120
        if ma120 > 0 and close < ma120:
            continue

        # ATR
        atr_arr = df["high"].values - df["low"].values
        atr = np.mean(atr_arr[max(0, idx - 13):idx + 1])
        if atr <= 0:
            atr = close * 0.02

        low_10 = df["low"].iloc[max(0, idx - 9):idx + 1].min()
        stop = max(low_10 - 0.5 * atr, close - 3 * atr)
        high_60 = df["high"].iloc[max(0, idx - 59):idx + 1].max()
        target = max(high_60, close + 1.5 * atr)

        risk = close - stop
        rr = (target - close) / risk if risk > 0 else 0
        if rr < 1.0:
            continue

        # Zone
        ma20 = np.mean(df["close"].values[max(0, idx - 19):idx + 1])
        ma60 = float(row.get("sma_60", 0) or 0)
        zone = 0.0
        if close > ma20: zone += 0.2
        if ma60 > 0 and close > ma60: zone += 0.2
        if ma120 > 0 and close > ma120: zone += 0.2
        if trix_above: zone += 0.2
        if 35 <= rsi <= 55: zone += 0.2

        rank = rr * zone * freshness

        # Grade (간이)
        grade = "S" if counter <= 3 else ("A" if counter <= 7 else "B")

        signals.append({
            "ticker": ticker,
            "close": close,
            "stop": stop,
            "target": target,
            "rr": rr,
            "zone": zone,
            "rank": rank,
            "grade": grade,
            "counter": counter,
            "freshness": freshness,
            "rsi": rsi,
            "atr": atr,
        })

    signals.sort(key=lambda s: s["rank"], reverse=True)
    return signals


def calc_position_size(capital, grade, freshness, mode):
    """포지션 사이징"""
    if mode == "A":
        return capital  # 무한 자본 모드

    base = capital / MAX_POSITIONS

    # Grade 가중
    grade_mult = {"S": 1.2, "A": 1.0, "B": 0.8}.get(grade, 1.0)

    # Freshness 가중
    if freshness >= 1.10:
        fresh_mult = 1.0
    elif freshness >= 0.90:
        fresh_mult = 0.8
    else:
        fresh_mult = 0.5

    if mode in ("B", "C", "D", "D_old"):
        return base * grade_mult * fresh_mult
    return base


def run_backtest(data_dict, name_map, mode="D"):
    """
    mode:
      A = 리스크 없음 (기존 방식, 자본 무한)
      B = 포지션 사이징만
      C = B + 레짐 캡
      D = C + 고급 익절/손절
    """
    ref_ticker = "005930"  # 삼성전자 (날짜 기준)
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= START_DATE) & (ref_df.index <= END_DATE)]

    capital = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    trades = []
    daily_results = []

    for date in dates:
        # 날짜 인덱스 매핑
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        day_pnl = 0.0

        # ── 1. 기존 보유 종목 관리 ──
        closed = []
        for pos in positions:
            if pos.ticker not in day_idx_map:
                continue
            idx = day_idx_map[pos.ticker]
            df = data_dict[pos.ticker]
            if idx >= len(df):
                continue

            row = df.iloc[idx]
            high = float(row["high"])
            low = float(row["low"])
            close_d = float(row["close"])
            pos.days_held += 1
            pos.peak_price = max(pos.peak_price, high)

            exit_reason = None
            sell_price = 0.0

            if mode in ("D", "D_old"):
                # D: 튜닝 (trail -8%, time 15d)
                # D_old: 원본 (trail -5%, time 10d)
                trail_pct = 0.92 if mode == "D" else 0.95
                time_limit = 15 if mode == "D" else 10

                atr_stop = pos.buy_price - pos.buy_price * 0.07
                stop_price = max(pos.stop_loss, atr_stop)
                stop_price = max(stop_price, pos.buy_price * 0.90)  # 캡 -10%

                # 1차 익절: +10% → 반매도
                if not pos.half_sold and high >= pos.buy_price * 1.10:
                    sell_price = pos.buy_price * 1.10
                    half_shares = pos.shares // 2
                    if half_shares > 0:
                        proceeds = half_shares * sell_price * (1 - COMMISSION - TAX)
                        cash += proceeds
                        pos.half_sold = True
                        pos.half_sold_pnl = (sell_price / pos.buy_price - 1)
                        pos.shares -= half_shares
                        day_pnl += proceeds - half_shares * pos.buy_price

                # 트레일링 스톱 (반매도 후)
                if pos.half_sold and low <= pos.peak_price * trail_pct:
                    exit_reason = "trailing"
                    sell_price = pos.peak_price * trail_pct

                # 기본 손절
                elif low <= stop_price:
                    exit_reason = "stop"
                    sell_price = stop_price

                # 목표가
                elif high >= pos.target:
                    exit_reason = "target"
                    sell_price = pos.target

                # 시간 손절
                elif pos.days_held >= time_limit and not pos.half_sold:
                    if close_d < pos.buy_price * 1.03:
                        exit_reason = "time_stop"
                        sell_price = close_d

                # 최대 보유 20일
                elif pos.days_held >= 20:
                    exit_reason = "timeout"
                    sell_price = close_d

            else:
                # 단순 모드 (A/B/C)
                if low <= pos.stop_loss:
                    exit_reason = "stop"
                    sell_price = pos.stop_loss
                elif high >= pos.target:
                    exit_reason = "target"
                    sell_price = pos.target
                elif pos.days_held >= 20:
                    exit_reason = "timeout"
                    sell_price = close_d

            if exit_reason:
                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                cost = pos.shares * pos.buy_price
                pnl = proceeds - cost + (pos.half_sold_pnl * pos.allocated * 0.5 if pos.half_sold else 0)
                pnl_pct = (sell_price / pos.buy_price - 1) if not pos.half_sold else (
                    (pos.half_sold_pnl + (sell_price / pos.buy_price - 1)) / 2
                )

                if mode == "A":
                    pnl_pct_net = pnl_pct - COMMISSION * 2 - TAX
                else:
                    cash += proceeds
                    pnl_pct_net = pnl_pct - COMMISSION * 2 - TAX

                day_pnl += pnl if mode != "A" else 0

                trades.append({
                    "ticker": pos.ticker,
                    "name": pos.name,
                    "buy_date": pos.buy_date,
                    "sell_date": date,
                    "buy_price": pos.buy_price,
                    "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net,
                    "pnl_won": pnl if mode != "A" else pnl_pct_net * INITIAL_CAPITAL / 100,
                    "exit": exit_reason,
                    "days": pos.days_held,
                    "grade": pos.grade,
                    "counter": pos.counter,
                    "freshness": pos.freshness,
                    "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입 ──
        held_tickers = {p.ticker for p in positions}

        # 레짐 캡
        if mode in ("C", "D", "D_old"):
            ref_idx = day_idx_map.get(ref_ticker, 0)
            regime = get_us_regime(ref_df, ref_idx)
            max_slots = REGIME_SLOTS.get(regime, 3)
        else:
            regime = "NEUTRAL"
            max_slots = MAX_POSITIONS

        available_slots = max_slots - len(positions)
        daily_entries = 0

        if available_slots > 0:
            signals = scan_signals(data_dict, day_idx_map)

            for sig in signals:
                if daily_entries >= MAX_DAILY_ENTRY and mode != "A":
                    break
                if available_slots <= 0:
                    break
                if sig["ticker"] in held_tickers:
                    continue

                ticker = sig["ticker"]
                df = data_dict[ticker]
                idx = day_idx_map[ticker]

                # 다음날 시가 매수
                if idx + 1 >= len(df):
                    continue
                next_open = float(df.iloc[idx + 1]["open"])
                if next_open <= 0:
                    continue

                buy_price = next_open * (1 + SLIPPAGE)

                # 포지션 사이징
                alloc = calc_position_size(cash if mode != "A" else INITIAL_CAPITAL,
                                           sig["grade"], sig["freshness"], mode)
                alloc = min(alloc, cash) if mode != "A" else alloc

                if alloc < 100000:  # 최소 10만원
                    continue

                shares = int(alloc / buy_price)
                if shares <= 0:
                    continue

                actual_cost = shares * buy_price * (1 + COMMISSION)

                if mode != "A":
                    if actual_cost > cash:
                        continue
                    cash -= actual_cost

                # 손절/목표 조정
                atr = sig["atr"]
                stop = max(buy_price - atr * 2, buy_price * 0.93)  # ATR×2 or -7%
                stop = max(stop, buy_price * 0.90)  # 캡 -10%
                target = sig["target"]

                pos = Position(
                    ticker=ticker,
                    name=name_map.get(ticker, ticker),
                    buy_date=date,
                    buy_price=buy_price,
                    shares=shares,
                    allocated=actual_cost,
                    stop_loss=stop,
                    target=target,
                    grade=sig["grade"],
                    freshness=sig["freshness"],
                    counter=sig["counter"],
                    peak_price=buy_price,
                )
                positions.append(pos)
                held_tickers.add(ticker)
                daily_entries += 1
                available_slots -= 1

        # ── 3. 일일 마감 ──
        # 보유 종목 시가 평가
        position_value = 0
        for pos in positions:
            if pos.ticker in day_idx_map:
                idx = day_idx_map[pos.ticker]
                close_d = float(data_dict[pos.ticker].iloc[idx]["close"])
                position_value += pos.shares * close_d

        if mode == "A":
            equity = INITIAL_CAPITAL + sum(t["pnl_pct"] for t in trades) * INITIAL_CAPITAL
        else:
            equity = cash + position_value

        daily_results.append(DayResult(
            date=date, equity=equity, cash=cash,
            positions=len(positions), daily_pnl=day_pnl
        ))

    # 미청산 포지션 강제 청산
    for pos in positions:
        last_close = float(data_dict[pos.ticker].iloc[-1]["close"])
        pnl_pct = (last_close / pos.buy_price - 1) - COMMISSION * 2 - TAX
        trades.append({
            "ticker": pos.ticker, "name": pos.name,
            "buy_date": pos.buy_date, "sell_date": dates[-1],
            "buy_price": pos.buy_price, "sell_price": last_close,
            "pnl_pct": pnl_pct, "pnl_won": pos.shares * last_close - pos.allocated,
            "exit": "force_close", "days": pos.days_held,
            "grade": pos.grade, "counter": pos.counter,
            "freshness": pos.freshness, "allocated": pos.allocated,
        })

    return trades, daily_results


def report(trades, daily_results, label, mode):
    if not trades:
        print(f"\n=== {label}: 거래 없음 ===")
        return {}

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # 자본 곡선 기반 MDD
    equities = [d.equity for d in daily_results]
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100
    mdd = np.min(dd_pct)

    final_equity = equities[-1] if equities else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    # 월별 수익
    monthly = defaultdict(float)
    for t in trades:
        key = t["sell_date"].strftime("%Y-%m")
        monthly[key] += t["pnl_pct"] * 100

    exits = defaultdict(int)
    for t in trades:
        exits[t["exit"]] += 1

    print(f"\n{'=' * 58}")
    print(f"  {label}")
    print(f"{'=' * 58}")
    print(f"  초기 자본: {INITIAL_CAPITAL / 1e8:.1f}억 → 최종: {final_equity / 1e8:.2f}억")
    print(f"  총 수익률: {total_return:+.1f}%")
    print(f"  총 거래: {len(trades)}건")
    print(f"  승률: {win_rate:.1f}% ({len(wins)}승 / {len(losses)}패)")
    print(f"  평균 수익: +{avg_win:.2f}% | 평균 손실: {avg_loss:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  청산: 목표{exits.get('target',0)} 손절{exits.get('stop',0)} "
          f"트레일{exits.get('trailing',0)} 시간{exits.get('time_stop',0)} "
          f"타임아웃{exits.get('timeout',0)}")
    print(f"  평균 보유: {np.mean([t['days'] for t in trades]):.1f}일")

    if mode != "A":
        print(f"\n  월별 수익률:")
        for m in sorted(monthly.keys()):
            bar = "+" * int(abs(monthly[m]) / 2) if monthly[m] > 0 else "-" * int(abs(monthly[m]) / 2)
            print(f"    {m}: {monthly[m]:+6.1f}% {bar}")

    return {
        "label": label,
        "trades": len(trades),
        "win_rate": win_rate,
        "pf": pf,
        "total_return": total_return,
        "mdd": mdd,
        "final_equity": final_equity,
    }


def main():
    print("데이터 로딩...")
    data_dict = load_parquets()
    name_map = load_name_map()
    print(f"  {len(data_dict)}종목 로드")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  초기 자본: {INITIAL_CAPITAL / 1e8:.1f}억")

    results = []

    for mode, label in [
        ("B", "B) 포지션 사이징 (5종목, 2일진입)"),
        ("D_old", "D_old) B+레짐+익절손절 (trail-5%, time10d)"),
        ("D", "D) B+레짐+익절손절 튜닝 (trail-8%, time15d)"),
    ]:
        print(f"\n[{mode}] {label} 실행 중...")
        trades, daily = run_backtest(data_dict, name_map, mode)
        r = report(trades, daily, label, mode)
        if r:
            results.append(r)

    # 비교 테이블
    if len(results) >= 2:
        print(f"\n{'=' * 70}")
        print(f"  종합 비교")
        print(f"{'=' * 70}")
        print(f"  {'모드':<42} {'거래':>5} {'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>8}")
        print(f"  {'-' * 65}")
        for r in results:
            print(f"  {r['label']:<42} {r['trades']:>5} "
                  f"{r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['total_return']:>+7.1f}% {r['mdd']:>7.1f}%")

        if len(results) >= 3:
            b, d_old, d = results[0], results[1], results[2]
            print(f"\n  슬리피지: 0.5% (현실적 한국 중형주 기준)")
            print(f"\n  B vs D_old vs D 비교:")
            print(f"    B:     수익률 {b['total_return']:+.1f}%, MDD {b['mdd']:.1f}%, PF {b['pf']:.2f}")
            print(f"    D_old: 수익률 {d_old['total_return']:+.1f}%, MDD {d_old['mdd']:.1f}%, PF {d_old['pf']:.2f}")
            print(f"    D:     수익률 {d['total_return']:+.1f}%, MDD {d['mdd']:.1f}%, PF {d['pf']:.2f}")
            print(f"\n  D_old→D 튜닝 효과:")
            print(f"    PF:     {d_old['pf']:.2f} → {d['pf']:.2f} ({d['pf']-d_old['pf']:+.2f})")
            print(f"    MDD:    {d_old['mdd']:.1f}% → {d['mdd']:.1f}% ({d['mdd']-d_old['mdd']:+.1f}%p)")
            print(f"    수익률: {d_old['total_return']:.1f}% → {d['total_return']:.1f}%")


if __name__ == "__main__":
    main()
