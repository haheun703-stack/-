"""
Phase 1 확신 사이징 백테스트 — v10.3 기존 vs 확신 비대칭 비교
================================================================
핵심 변경 (파라미터 2개만 추가):
  확신 판정 = Grade + 수급(vol_ratio) 기반
    HIGH: Grade S/A + vol_ratio < 0.8 → 슬롯 1.5배
    MID:  그 외 → 슬롯 1.0배 (기존과 동일)
    LOW:  Grade B + vol_ratio > 1.2 → 슬롯 0.5배

  손절: -7% 고정 유지 (변경 없음)
  게이트/트리거: 변경 없음
  레짐 캡: C_new 동일 (BULL 5, CAUTION 3, BEAR 2, CRISIS 0)

사용법:
  python -u -X utf8 scripts/backtest_conviction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# ── 설정 (backtest_v2.py와 동일) ──
START_DATE = "2025-03-01"
END_DATE = "2026-02-13"
INITIAL_CAPITAL = 100_000_000
MAX_POSITIONS = 5
MAX_DAILY_ENTRY = 2
SLIPPAGE = 0.005
COMMISSION = 0.00015
TAX = 0.0018

# ── 확신 사이징 파라미터 (추가분: 2개만) ──
CONVICTION_HIGH_MULT = 1.5   # HIGH: 기본 슬롯의 1.5배
CONVICTION_LOW_MULT = 0.5    # LOW: 기본 슬롯의 0.5배
# MID는 1.0 (기존과 동일)


@dataclass
class Position:
    ticker: str
    name: str
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    allocated: float
    stop_loss: float
    target: float
    grade: str
    freshness: float
    counter: int
    conviction: str  # HIGH / MID / LOW
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


# ── 데이터 로딩 (backtest_v2.py 동일) ──

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


def load_kospi_index():
    kospi_path = Path("data/kospi_index.csv")
    if not kospi_path.exists():
        return None
    df = pd.read_csv(kospi_path, index_col="Date", parse_dates=True).sort_index()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["rv20_pct"] = df["rv20"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    return df


# ── 유틸 함수 (backtest_v2.py 동일) ──

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


def get_kospi_regime(kospi_df, date):
    if kospi_df is None:
        return "CAUTION", 3
    prev = kospi_df[kospi_df.index < date]
    if len(prev) < 60:
        return "CAUTION", 3
    row = prev.iloc[-1]
    close = row["close"]
    ma20, ma60 = row["ma20"], row["ma60"]
    rv_pct = row.get("rv20_pct", 0.5)
    if pd.isna(ma20) or pd.isna(ma60):
        return "CAUTION", 3
    if close > ma20:
        return ("BULL", 5) if (not pd.isna(rv_pct) and rv_pct < 0.50) else ("CAUTION", 3)
    elif close > ma60:
        return "BEAR", 2
    else:
        return "CRISIS", 0


# ── 시그널 스캔 (vol_ratio 추가 반환) ──

def scan_signals(data_dict, day_idx_map):
    """v10.1 시그널 스캔 — vol_ratio 추가 반환 (확신 판정용)."""
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

        trix_above = trix_val > trix_sig
        macd_above = macd > macd_sig
        if not (trix_above and macd_above) and not trix_above:
            continue

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

        ma20 = np.mean(df["close"].values[max(0, idx - 19):idx + 1])
        ma60 = float(row.get("sma_60", 0) or 0)
        zone = 0.0
        if close > ma20: zone += 0.2
        if ma60 > 0 and close > ma60: zone += 0.2
        if ma120 > 0 and close > ma120: zone += 0.2
        if trix_above: zone += 0.2
        if 35 <= rsi <= 55: zone += 0.2

        rank = rr * zone * freshness
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
            "vol_ratio": vol_c,  # 확신 판정용 추가
        })

    signals.sort(key=lambda s: s["rank"], reverse=True)
    return signals


# ── 확신 판정 ──

def judge_conviction(grade: str, vol_ratio: float) -> str:
    """확신도 판정.

    HIGH: Grade S/A + 거래량 수축 (vol_ratio < 0.8) = 수급 확인됨
    LOW:  Grade B + 거래량 팽창 (vol_ratio > 1.2) = 수급 미확인
    MID:  그 외
    """
    if grade in ("S", "A") and vol_ratio < 0.8:
        return "HIGH"
    elif grade == "B" and vol_ratio > 1.2:
        return "LOW"
    return "MID"


# ── 포지션 사이징 ──

def calc_position_size(capital, grade, freshness, conviction, mode):
    """포지션 사이징.

    mode="C_new": 기존 v10.3 (확신 무시)
    mode="P1":    Phase 1 확신 사이징 (conviction 반영)
    """
    base = capital / MAX_POSITIONS

    # Grade 가중 (기존 동일)
    grade_mult = {"S": 1.2, "A": 1.0, "B": 0.8}.get(grade, 1.0)

    # Freshness 가중 (기존 동일)
    if freshness >= 1.10:
        fresh_mult = 1.0
    elif freshness >= 0.90:
        fresh_mult = 0.8
    else:
        fresh_mult = 0.5

    size = base * grade_mult * fresh_mult

    # Phase 1: 확신 배수 적용
    if mode == "P1":
        if conviction == "HIGH":
            size *= CONVICTION_HIGH_MULT
        elif conviction == "LOW":
            size *= CONVICTION_LOW_MULT
        # MID = 1.0 (변경 없음)

    return size


# ── 백테스트 루프 ──

def run_backtest(data_dict, name_map, mode, kospi_df):
    """
    mode:
      C_new = 기존 v10.3 (KOSPI 레짐 캡 + 균등 사이징)
      P1    = Phase 1 확신 사이징 (C_new + 확신 비대칭)
    """
    ref_ticker = "005930"
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= START_DATE) & (ref_df.index <= END_DATE)]

    cash = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades = []
    daily_results = []
    conviction_stats = {"HIGH": 0, "MID": 0, "LOW": 0}

    for date in dates:
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        day_pnl = 0.0

        # ── 1. 기존 보유 관리 (C_new와 동일 — 손절 -7% 고정) ──
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

            # 손절 가격 (-7% 고정, ATR 기반 하한, 캡 -10%)
            atr_stop = pos.buy_price - pos.buy_price * 0.07
            stop_price = max(pos.stop_loss, atr_stop)
            stop_price = max(stop_price, pos.buy_price * 0.90)

            # 1차 익절: +10% → 반매도
            if not pos.half_sold and high >= pos.buy_price * 1.10:
                sell_price = pos.buy_price * 1.10
                half_shares = pos.shares // 2
                if half_shares > 0:
                    proceeds = half_shares * sell_price * (1 - COMMISSION - TAX)
                    half_cost = half_shares * pos.buy_price * (1 + COMMISSION)
                    cash += proceeds
                    pos.half_sold = True
                    pos.half_sold_pnl = (sell_price / pos.buy_price - 1) - COMMISSION * 2 - TAX
                    pos.shares -= half_shares
                    day_pnl += proceeds - half_cost

            # 트레일링 (반매도 후 -8%)
            if pos.half_sold and low <= pos.peak_price * 0.92:
                exit_reason = "trailing"
                sell_price = pos.peak_price * 0.92
            elif low <= stop_price:
                exit_reason = "stop"
                sell_price = stop_price
            elif high >= pos.target:
                exit_reason = "target"
                sell_price = pos.target
            elif pos.days_held >= 15 and not pos.half_sold:
                if close_d < pos.buy_price * 1.03:
                    exit_reason = "time_stop"
                    sell_price = close_d
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
                cash += proceeds
                pnl_pct_net = pnl_pct - COMMISSION * 2 - TAX
                day_pnl += pnl

                trades.append({
                    "ticker": pos.ticker, "name": pos.name,
                    "buy_date": pos.buy_date, "sell_date": date,
                    "buy_price": pos.buy_price, "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net,
                    "pnl_won": pnl,
                    "exit": exit_reason, "days": pos.days_held,
                    "grade": pos.grade, "conviction": pos.conviction,
                    "counter": pos.counter, "freshness": pos.freshness,
                    "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입 ──
        held_tickers = {p.ticker for p in positions}
        regime, max_slots = get_kospi_regime(kospi_df, date)
        available_slots = max_slots - len(positions)
        daily_entries = 0

        if available_slots > 0:
            signals = scan_signals(data_dict, day_idx_map)

            for sig in signals:
                if daily_entries >= MAX_DAILY_ENTRY:
                    break
                if available_slots <= 0:
                    break
                if sig["ticker"] in held_tickers:
                    continue

                ticker = sig["ticker"]
                df = data_dict[ticker]
                idx = day_idx_map[ticker]

                if idx + 1 >= len(df):
                    continue
                next_open = float(df.iloc[idx + 1]["open"])
                if next_open <= 0:
                    continue

                buy_price = next_open * (1 + SLIPPAGE)

                # 확신 판정
                conviction = judge_conviction(sig["grade"], sig["vol_ratio"])
                conviction_stats[conviction] += 1

                # 포지션 사이징 (확신 반영)
                alloc = calc_position_size(cash, sig["grade"], sig["freshness"], conviction, mode)
                alloc = min(alloc, cash)
                if alloc < 100000:
                    continue

                shares = int(alloc / buy_price)
                if shares <= 0:
                    continue

                actual_cost = shares * buy_price * (1 + COMMISSION)
                if actual_cost > cash:
                    continue
                cash -= actual_cost

                atr = sig["atr"]
                stop = max(buy_price - atr * 2, buy_price * 0.93)
                stop = max(stop, buy_price * 0.90)
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
                    conviction=conviction,
                    peak_price=buy_price,
                )
                positions.append(pos)
                held_tickers.add(ticker)
                daily_entries += 1
                available_slots -= 1

        # ── 3. 일일 마감 ──
        position_value = sum(
            pos.shares * float(data_dict[pos.ticker].iloc[day_idx_map[pos.ticker]]["close"])
            for pos in positions if pos.ticker in day_idx_map
        )
        equity = cash + position_value

        daily_results.append(DayResult(
            date=date, equity=equity, cash=cash,
            positions=len(positions), daily_pnl=day_pnl,
        ))

    # 미청산 포지션 강제 청산
    for pos in positions:
        if pos.ticker in day_idx_map:
            last_close = float(data_dict[pos.ticker].iloc[day_idx_map[pos.ticker]]["close"])
        else:
            df_t = data_dict[pos.ticker]
            valid = df_t[df_t.index <= dates[-1]]
            last_close = float(valid.iloc[-1]["close"]) if len(valid) > 0 else pos.buy_price
        pnl_pct = (last_close / pos.buy_price - 1) - COMMISSION * 2 - TAX
        trades.append({
            "ticker": pos.ticker, "name": pos.name,
            "buy_date": pos.buy_date, "sell_date": dates[-1],
            "buy_price": pos.buy_price, "sell_price": last_close,
            "pnl_pct": pnl_pct,
            "pnl_won": pos.shares * last_close - pos.allocated,
            "exit": "force_close", "days": pos.days_held,
            "grade": pos.grade, "conviction": pos.conviction,
            "counter": pos.counter, "freshness": pos.freshness,
            "allocated": pos.allocated,
        })

    return trades, daily_results, conviction_stats


# ── 리포트 ──

def report(trades, daily_results, label, conviction_stats=None):
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

    equities = [d.equity for d in daily_results]
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100
    mdd = np.min(dd_pct)

    final_equity = equities[-1] if equities else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    # 일간 수익률 → Sharpe
    daily_returns = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

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
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  청산: 목표{exits.get('target',0)} 손절{exits.get('stop',0)} "
          f"트레일{exits.get('trailing',0)} 시간{exits.get('time_stop',0)} "
          f"타임아웃{exits.get('timeout',0)}")
    print(f"  평균 보유: {np.mean([t['days'] for t in trades]):.1f}일")

    # 확신별 분석 (P1 모드)
    if conviction_stats and sum(conviction_stats.values()) > 0:
        print(f"\n  확신 분포:")
        total_c = sum(conviction_stats.values())
        for level in ["HIGH", "MID", "LOW"]:
            cnt = conviction_stats.get(level, 0)
            pct = cnt / total_c * 100 if total_c > 0 else 0
            # 확신별 승률
            level_trades = [t for t in trades if t.get("conviction") == level]
            if level_trades:
                level_wins = len([t for t in level_trades if t["pnl_pct"] > 0])
                level_wr = level_wins / len(level_trades) * 100
                level_avg = np.mean([t["pnl_pct"] for t in level_trades]) * 100
                level_alloc_avg = np.mean([t["allocated"] for t in level_trades]) / 1e6
                print(f"    {level:>4}: {cnt:>3}건 ({pct:>4.1f}%) | "
                      f"승률 {level_wr:.0f}% | 평균 {level_avg:+.1f}% | "
                      f"평균배분 {level_alloc_avg:.1f}백만")
            else:
                print(f"    {level:>4}: {cnt:>3}건 ({pct:>4.1f}%)")

    # 월별 수익
    monthly = defaultdict(float)
    for t in trades:
        key = t["sell_date"].strftime("%Y-%m")
        monthly[key] += t["pnl_pct"] * 100

    print(f"\n  월별 수익률:")
    for m in sorted(monthly.keys()):
        bar_len = int(abs(monthly[m]) / 2)
        bar = "+" * bar_len if monthly[m] > 0 else "-" * bar_len
        print(f"    {m}: {monthly[m]:+6.1f}% {bar}")

    return {
        "label": label,
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pf": pf,
        "total_return": total_return,
        "mdd": mdd,
        "sharpe": sharpe,
        "final_equity": final_equity,
    }


# ── 메인 ──

def main():
    print("=" * 60)
    print("  Phase 1 확신 사이징 백테스트")
    print("  v10.3 기존 (C_new) vs 확신 비대칭 (P1)")
    print("=" * 60)
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  초기 자본: {INITIAL_CAPITAL / 1e8:.1f}억")
    print(f"  슬리피지: {SLIPPAGE * 100:.1f}%")
    print(f"  확신 배수: HIGH {CONVICTION_HIGH_MULT}x / MID 1.0x / LOW {CONVICTION_LOW_MULT}x")
    print(f"  확신 판정: Grade(S/A) + vol_ratio<0.8 → HIGH")
    print(f"             Grade(B) + vol_ratio>1.2 → LOW")
    print(f"  손절: -7% 고정 (변경 없음)")

    print("\n데이터 로딩...")
    data_dict = load_parquets()
    name_map = load_name_map()
    kospi_df = load_kospi_index()
    print(f"  {len(data_dict)}종목 로드")

    results = []

    # 1. 기존 C_new (확신 무시)
    print(f"\n[C_new] 기존 v10.3 실행 중...")
    trades_base, daily_base, stats_base = run_backtest(data_dict, name_map, "C_new", kospi_df)
    r1 = report(trades_base, daily_base, "C_new) 기존 v10.3 (균등 사이징)", stats_base)
    if r1:
        results.append(r1)

    # 2. Phase 1 확신 사이징
    print(f"\n[P1] Phase 1 확신 사이징 실행 중...")
    trades_p1, daily_p1, stats_p1 = run_backtest(data_dict, name_map, "P1", kospi_df)
    r2 = report(trades_p1, daily_p1, "P1) 확신 사이징 (HIGH 1.5x / LOW 0.5x)", stats_p1)
    if r2:
        results.append(r2)

    # 비교 테이블
    if len(results) == 2:
        base, p1 = results[0], results[1]

        print(f"\n{'=' * 70}")
        print(f"  종합 비교: C_new (기존) vs P1 (확신 사이징)")
        print(f"{'=' * 70}")
        print(f"  {'지표':<18} {'C_new (기존)':>14} {'P1 (확신)':>14} {'변화':>12}")
        print(f"  {'-' * 58}")
        for key, label, fmt in [
            ("trades", "거래 수", "{:.0f}"),
            ("win_rate", "승률 (%)", "{:.1f}"),
            ("pf", "Profit Factor", "{:.2f}"),
            ("total_return", "총 수익률 (%)", "{:+.1f}"),
            ("mdd", "MDD (%)", "{:.1f}"),
            ("sharpe", "Sharpe", "{:.2f}"),
            ("avg_win", "평균 수익 (%)", "{:+.1f}"),
            ("avg_loss", "평균 손실 (%)", "{:.1f}"),
        ]:
            v1 = base[key]
            v2 = p1[key]
            diff = v2 - v1
            sign = "+" if diff > 0 else ""
            print(f"  {label:<18} {fmt.format(v1):>14} {fmt.format(v2):>14} {sign}{fmt.format(diff):>10}")

        print(f"\n  판정:")
        pf_better = p1["pf"] >= base["pf"]
        mdd_safe = p1["mdd"] >= base["mdd"] - 1.0  # MDD 1%p 이내 악화 허용
        ret_better = p1["total_return"] >= base["total_return"]
        sharpe_better = p1["sharpe"] >= base["sharpe"]

        wins = sum([pf_better, mdd_safe, ret_better, sharpe_better])
        print(f"    PF 개선:       {'PASS' if pf_better else 'FAIL'} ({base['pf']:.2f} → {p1['pf']:.2f})")
        print(f"    MDD 안전:      {'PASS' if mdd_safe else 'FAIL'} ({base['mdd']:.1f}% → {p1['mdd']:.1f}%)")
        print(f"    수익률 개선:   {'PASS' if ret_better else 'FAIL'} ({base['total_return']:+.1f}% → {p1['total_return']:+.1f}%)")
        print(f"    Sharpe 개선:   {'PASS' if sharpe_better else 'FAIL'} ({base['sharpe']:.2f} → {p1['sharpe']:.2f})")

        if wins >= 3:
            print(f"\n  결론: PASS ({wins}/4) — Phase 1 확신 사이징 채택 가능")
        elif wins >= 2:
            print(f"\n  결론: 부분 개선 ({wins}/4) — 파라미터 미세 조정 후 재검토")
        else:
            print(f"\n  결론: FAIL ({wins}/4) — 확신 사이징 효과 없음, 기존 유지")


if __name__ == "__main__":
    main()
