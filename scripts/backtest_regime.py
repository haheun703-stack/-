"""
수급 레짐 전환 백테스트
=======================
핵심 가설: "MOMENTUM 레짐에서 RSI/freshness 제약을 완화하면 수익이 개선된다"

3-way 비교:
  C_new    = 기존 v11.2 (KOSPI 레짐 캡 + 균등 사이징)
  Regime_A = MOMENTUM freshness 완화 (진입만 늘림)
  Regime_B = Regime_A + MOMENTUM 전용 청산 (수급이탈 즉시탈출)

사용법:
  python -u -X utf8 scripts/backtest_regime.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from src.regime_detector import detect_supply_regime

# ── 설정 ──
START_DATE = "2025-03-01"
END_DATE = "2026-02-13"
INITIAL_CAPITAL = 100_000_000
MAX_POSITIONS = 5
MAX_DAILY_ENTRY = 2
SLIPPAGE = 0.005
COMMISSION = 0.00015
TAX = 0.0018

# MOMENTUM 전용 파라미터 (Regime_B용)
MOMENTUM_STOP_PCT = -0.035       # -3.5% 손절
MOMENTUM_TRAIL_PCT = -0.05       # -5% 트레일링
MOMENTUM_MAX_HOLD = 7            # 최대 7일
MOMENTUM_TARGET_PCT = 0.08       # +8% 익절

# Regime_C: 슬롯 분리 파라미터
NORMAL_ALLOC_PCT = 0.70          # NORMAL 풀 자금 비율
MOMENTUM_ALLOC_PCT = 0.25        # MOMENTUM 풀 자금 비율
# 나머지 5%는 현금 예비
MOMENTUM_MAX_SLOTS = 2           # MOMENTUM 전용 슬롯


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
    conviction: str
    regime_at_entry: str = "NORMAL"
    regime_score: float = 0.0
    peak_price: float = 0.0
    half_sold: bool = False
    half_sold_pnl: float = 0.0
    days_held: int = 0
    supply_bad_days: int = 0  # 수급 악화 연속일수 (Regime_D용)


@dataclass
class DayResult:
    date: pd.Timestamp
    equity: float
    cash: float
    positions: int
    daily_pnl: float = 0.0


# ── 데이터 로딩 ──

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


# ── 유틸 함수 ──

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
    """기존 freshness 계산 (NORMAL 모드)."""
    if counter <= 0 or counter >= 16 or vol_c > 2.0:
        return 0.0
    if counter >= 8 and rsi > 65:
        return 0.0
    c = 1.15 if counter <= 3 else (1.00 if counter <= 7 else 0.70)
    r = 1.00 if rsi <= 55 else (0.90 if rsi <= 65 else 0.75)
    v = 1.05 if vol_c < 0.7 else (1.00 if vol_c <= 1.3 else 0.85)
    return round(c * r * v, 3)


def calc_freshness_regime(counter, rsi, vol_c, regime):
    """레짐 인식 freshness 계산.

    NORMAL: 기존과 동일
    MOMENTUM: RSI 제약 완화, 거래량 폭증 허용
    """
    if counter <= 0:
        return 0.0

    if regime == "NORMAL":
        # 기존 로직 그대로
        if counter >= 16 or vol_c > 2.0:
            return 0.0
        if counter >= 8 and rsi > 65:
            return 0.0
        c = 1.15 if counter <= 3 else (1.00 if counter <= 7 else 0.70)
        r = 1.00 if rsi <= 55 else (0.90 if rsi <= 65 else 0.75)
        v = 1.05 if vol_c < 0.7 else (1.00 if vol_c <= 1.3 else 0.85)
    else:  # MOMENTUM
        # ★ RSI 제약 해제 — 수급이 살아있으면 RSI 높아도 진입
        if counter >= 20:
            return 0.0
        if counter >= 12 and rsi > 75:
            return 0.0
        c = 1.15 if counter <= 5 else (1.00 if counter <= 10 else 0.80)
        r = 1.00  # RSI 감점 없음!
        v = 1.10 if vol_c > 1.5 else 1.00  # 거래량 폭증은 보너스

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


# ── 시그널 스캔 ──

def scan_signals(data_dict, day_idx_map, regime_mode=False, regime_threshold=0.55):
    """시그널 스캔.

    regime_mode=False: 기존 freshness (C_new)
    regime_mode=True:  레짐 인식 freshness (Regime_A/B)
    regime_threshold: MOMENTUM 판정 임계값 (기본 0.55)
    """
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

        # 레짐 판별
        if regime_mode:
            regime_signal = detect_supply_regime(row, threshold=regime_threshold)
            regime = regime_signal.regime
            regime_score = regime_signal.score
            freshness = calc_freshness_regime(counter, rsi, vol_c, regime)
        else:
            regime = "NORMAL"
            regime_score = 0.0
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
            "vol_ratio": vol_c,
            "regime": regime,
            "regime_score": regime_score,
        })

    signals.sort(key=lambda s: s["rank"], reverse=True)
    return signals


# ── 포지션 사이징 ──

def calc_position_size(capital, grade, freshness):
    base = capital / MAX_POSITIONS
    grade_mult = {"S": 1.2, "A": 1.0, "B": 0.8}.get(grade, 1.0)
    if freshness >= 1.10:
        fresh_mult = 1.0
    elif freshness >= 0.90:
        fresh_mult = 0.8
    else:
        fresh_mult = 0.5
    return base * grade_mult * fresh_mult


# ── 백테스트 루프 ──

def run_backtest(data_dict, name_map, mode, kospi_df):
    """
    mode:
      C_new    = 기존 (regime_mode=False, 기존 청산)
      Regime_A = regime_mode=True, 기존 청산
      Regime_B = regime_mode=True, MOMENTUM 전용 청산
    """
    regime_mode = mode in ("Regime_A", "Regime_B")
    dual_exit = mode == "Regime_B"

    ref_ticker = "005930"
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= START_DATE) & (ref_df.index <= END_DATE)]

    cash = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades = []
    daily_results = []
    regime_stats = {"MOMENTUM": 0, "NORMAL": 0}

    for date in dates:
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        day_pnl = 0.0

        # ── 1. 기존 보유 관리 ──
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

            # ── MOMENTUM 전용 청산 (Regime_B) ──
            if dual_exit and pos.regime_at_entry == "MOMENTUM":
                # M1: 수급 이탈 즉시 탈출
                inst_5d = float(row.get('inst_net_5d', 0) or 0)
                foreign_5d = float(row.get('foreign_net_5d', 0) or 0)
                smart_flow = inst_5d + foreign_5d

                if smart_flow < 0:
                    exit_reason = "supply_exit"
                    sell_price = close_d
                # M2: 타이트 트레일링 -5%
                elif pos.peak_price > 0 and low <= pos.peak_price * (1 + MOMENTUM_TRAIL_PCT):
                    exit_reason = "momentum_trail"
                    sell_price = pos.peak_price * (1 + MOMENTUM_TRAIL_PCT)
                # M3: 짧은 보유 (7일)
                elif pos.days_held >= MOMENTUM_MAX_HOLD:
                    exit_reason = "momentum_timeout"
                    sell_price = close_d
                # M4: 타이트 손절 -3.5%
                elif low <= pos.buy_price * (1 + MOMENTUM_STOP_PCT):
                    exit_reason = "momentum_stop"
                    sell_price = pos.buy_price * (1 + MOMENTUM_STOP_PCT)
                # M5: +8% 익절
                elif high >= pos.buy_price * (1 + MOMENTUM_TARGET_PCT):
                    exit_reason = "momentum_target"
                    sell_price = pos.buy_price * (1 + MOMENTUM_TARGET_PCT)

            else:
                # ── NORMAL 기존 청산 ──
                atr_stop = pos.buy_price - pos.buy_price * 0.07
                stop_price = max(pos.stop_loss, atr_stop)
                stop_price = max(stop_price, pos.buy_price * 0.90)

                # 1차 익절: +10% → 반매도
                if not pos.half_sold and high >= pos.buy_price * 1.10:
                    sell_price_half = pos.buy_price * 1.10
                    half_shares = pos.shares // 2
                    if half_shares > 0:
                        proceeds = half_shares * sell_price_half * (1 - COMMISSION - TAX)
                        half_cost = half_shares * pos.buy_price * (1 + COMMISSION)
                        cash += proceeds
                        pos.half_sold = True
                        pos.half_sold_pnl = (sell_price_half / pos.buy_price - 1) - COMMISSION * 2 - TAX
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

                trades.append({
                    "ticker": pos.ticker, "name": pos.name,
                    "buy_date": pos.buy_date, "sell_date": date,
                    "buy_price": pos.buy_price, "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net,
                    "pnl_won": pnl,
                    "exit": exit_reason, "days": pos.days_held,
                    "grade": pos.grade,
                    "regime": pos.regime_at_entry,
                    "regime_score": pos.regime_score,
                    "counter": pos.counter, "freshness": pos.freshness,
                    "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입 ──
        held_tickers = {p.ticker for p in positions}
        kospi_regime, max_slots = get_kospi_regime(kospi_df, date)
        available_slots = max_slots - len(positions)
        daily_entries = 0

        if available_slots > 0:
            signals = scan_signals(data_dict, day_idx_map, regime_mode=regime_mode)

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

                # 포지션 사이징
                alloc = calc_position_size(cash, sig["grade"], sig["freshness"])
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

                regime_stats[sig["regime"]] += 1

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
                    conviction="MID",
                    regime_at_entry=sig["regime"],
                    regime_score=sig["regime_score"],
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
            "grade": pos.grade,
            "regime": pos.regime_at_entry,
            "regime_score": pos.regime_score,
            "counter": pos.counter, "freshness": pos.freshness,
            "allocated": pos.allocated,
        })

    return trades, daily_results, regime_stats


def _normal_exit_logic(pos, row, cash, day_pnl):
    """NORMAL 기존 청산 로직 (C_new 동일). 공용 함수."""
    high = float(row["high"])
    low = float(row["low"])
    close_d = float(row["close"])

    exit_reason = None
    sell_price = 0.0

    atr_stop = pos.buy_price - pos.buy_price * 0.07
    stop_price = max(pos.stop_loss, atr_stop)
    stop_price = max(stop_price, pos.buy_price * 0.90)

    # 1차 익절: +10% → 반매도
    if not pos.half_sold and high >= pos.buy_price * 1.10:
        sell_price_half = pos.buy_price * 1.10
        half_shares = pos.shares // 2
        if half_shares > 0:
            proceeds = half_shares * sell_price_half * (1 - COMMISSION - TAX)
            half_cost = half_shares * pos.buy_price * (1 + COMMISSION)
            cash += proceeds
            pos.half_sold = True
            pos.half_sold_pnl = (sell_price_half / pos.buy_price - 1) - COMMISSION * 2 - TAX
            pos.shares -= half_shares
            day_pnl += proceeds - half_cost

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

    return exit_reason, sell_price, cash, day_pnl


def _is_supply_bad(row):
    """수급 악화 판별: 스마트머니(기관+외인) 5일 순매도."""
    inst_5d = float(row.get('inst_net_5d', 0) or 0)
    foreign_5d = float(row.get('foreign_net_5d', 0) or 0)
    return (inst_5d + foreign_5d) < 0


def run_backtest_defense(data_dict, name_map, kospi_df, exit_mode="A",
                         b_time_stop=8, b_timeout=10):
    """Regime_D: 이중 청산 모드 백테스트.

    진입: C_new 100% 동일 (변경 없음)
    청산: 수급 악화 시 보조 청산 레이어 추가

    exit_mode:
      A  = 수급 악화 시 손절선 타이트닝 (-7% → -4%)
      B  = 수급 악화 시 보유일 축소 (b_time_stop, b_timeout)
      C  = 수급 악화 2일 연속 시 즉시 청산
      AB = A + B 콤보
    b_time_stop: 방식 B 수급 악화 시 time_stop 기준일 (기본 8)
    b_timeout: 방식 B 수급 악화 시 timeout 기준일 (기본 10)
    """
    ref_ticker = "005930"
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= START_DATE) & (ref_df.index <= END_DATE)]

    cash = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades = []
    daily_results = []
    supply_exit_count = 0  # 수급 기반 청산 횟수

    for date in dates:
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        day_pnl = 0.0

        # ── 1. 기존 보유 관리 + 수급 방어 청산 ──
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

            # ★ 수급 상태 체크
            supply_bad = _is_supply_bad(row)
            if supply_bad:
                pos.supply_bad_days += 1
            else:
                pos.supply_bad_days = 0

            exit_reason = None
            sell_price = 0.0

            # ── 방식 C: 수급 악화 2일 연속 → 즉시 청산 ──
            if exit_mode in ("C",) and pos.supply_bad_days >= 2:
                exit_reason = "supply_exit"
                sell_price = close_d
                supply_exit_count += 1

            if not exit_reason:
                # ── 손절선 계산 (방식 A/AB: 수급 악화 시 타이트닝) ──
                if exit_mode in ("A", "AB") and supply_bad:
                    atr_stop = pos.buy_price * 0.96   # -4% (기존 -7%에서 3%p 타이트)
                    stop_price = max(pos.stop_loss, atr_stop)
                    stop_price = max(stop_price, pos.buy_price * 0.93)  # floor -7% (기존 -10%)
                else:
                    atr_stop = pos.buy_price - pos.buy_price * 0.07  # -7%
                    stop_price = max(pos.stop_loss, atr_stop)
                    stop_price = max(stop_price, pos.buy_price * 0.90)

                # ── 1차 익절: +10% → 반매도 (C_new 동일) ──
                if not pos.half_sold and high >= pos.buy_price * 1.10:
                    sell_price_half = pos.buy_price * 1.10
                    half_shares = pos.shares // 2
                    if half_shares > 0:
                        proceeds = half_shares * sell_price_half * (1 - COMMISSION - TAX)
                        half_cost = half_shares * pos.buy_price * (1 + COMMISSION)
                        cash += proceeds
                        pos.half_sold = True
                        pos.half_sold_pnl = (sell_price_half / pos.buy_price - 1) - COMMISSION * 2 - TAX
                        pos.shares -= half_shares
                        day_pnl += proceeds - half_cost

                # ── 보유일 기준 (방식 B/AB: 수급 악화 시 축소) ──
                if exit_mode in ("B", "AB") and supply_bad:
                    time_stop_days = b_time_stop
                    timeout_days = b_timeout
                else:
                    time_stop_days = 15
                    timeout_days = 20

                # ── 트레일링 (반매도 후 -8%) ──
                if pos.half_sold and low <= pos.peak_price * 0.92:
                    exit_reason = "trailing"
                    sell_price = pos.peak_price * 0.92
                elif low <= stop_price:
                    exit_reason = "stop"
                    sell_price = stop_price
                    if exit_mode in ("A", "AB") and supply_bad:
                        exit_reason = "stop_tight"  # 수급 타이트닝 손절 구분
                        supply_exit_count += 1
                elif high >= pos.target:
                    exit_reason = "target"
                    sell_price = pos.target
                elif pos.days_held >= time_stop_days and not pos.half_sold:
                    if close_d < pos.buy_price * 1.03:
                        exit_reason = "time_stop"
                        sell_price = close_d
                        if exit_mode in ("B", "AB") and supply_bad:
                            exit_reason = "time_stop_early"
                            supply_exit_count += 1
                elif pos.days_held >= timeout_days:
                    exit_reason = "timeout"
                    sell_price = close_d
                    if exit_mode in ("B", "AB") and supply_bad:
                        exit_reason = "timeout_early"
                        supply_exit_count += 1

            if exit_reason:
                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                cost = pos.shares * pos.buy_price
                pnl = proceeds - cost + (pos.half_sold_pnl * pos.allocated * 0.5 if pos.half_sold else 0)
                pnl_pct = (sell_price / pos.buy_price - 1) if not pos.half_sold else (
                    (pos.half_sold_pnl + (sell_price / pos.buy_price - 1)) / 2
                )
                cash += proceeds
                pnl_pct_net = pnl_pct - COMMISSION * 2 - TAX

                trades.append({
                    "ticker": pos.ticker, "name": pos.name,
                    "buy_date": pos.buy_date, "sell_date": date,
                    "buy_price": pos.buy_price, "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net,
                    "pnl_won": pnl,
                    "exit": exit_reason, "days": pos.days_held,
                    "grade": pos.grade,
                    "regime": "NORMAL",
                    "regime_score": 0.0,
                    "counter": pos.counter, "freshness": pos.freshness,
                    "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입: C_new 100% 동일 ──
        held_tickers = {p.ticker for p in positions}
        kospi_regime, max_slots = get_kospi_regime(kospi_df, date)
        available_slots = max_slots - len(positions)
        daily_entries = 0

        if available_slots > 0:
            signals = scan_signals(data_dict, day_idx_map, regime_mode=False)

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
                alloc = calc_position_size(cash, sig["grade"], sig["freshness"])
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
                    conviction="MID",
                    regime_at_entry="NORMAL",
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
            "grade": pos.grade,
            "regime": "NORMAL",
            "regime_score": 0.0,
            "counter": pos.counter, "freshness": pos.freshness,
            "allocated": pos.allocated,
        })

    return trades, daily_results, {"MOMENTUM": 0, "NORMAL": len(trades)}, supply_exit_count


def run_backtest_split(data_dict, name_map, kospi_df,
                       regime_threshold=0.55, momentum_slots=2,
                       momentum_alloc=0.25):
    """Regime_C: NORMAL 슬롯 + MOMENTUM 슬롯 완전 분리.

    핵심: NORMAL과 MOMENTUM이 서로 간섭하지 않음.
    MOMENTUM 진입은 레짐 완화 freshness, 청산은 기존 C_new 동일.
    """
    ref_ticker = "005930"
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= START_DATE) & (ref_df.index <= END_DATE)]

    # 자금 분리
    normal_alloc_pct = 1.0 - momentum_alloc - 0.05  # 5% 예비 유지
    normal_cash = float(INITIAL_CAPITAL * normal_alloc_pct)
    momentum_cash = float(INITIAL_CAPITAL * momentum_alloc)
    reserve_cash = float(INITIAL_CAPITAL * 0.05)

    normal_positions: list[Position] = []
    momentum_positions: list[Position] = []
    trades = []
    daily_results = []
    regime_stats = {"MOMENTUM": 0, "NORMAL": 0}

    for date in dates:
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        day_pnl = 0.0

        # ═══ 1. 보유 관리: NORMAL 풀 ═══
        closed_n = []
        for pos in normal_positions:
            if pos.ticker not in day_idx_map:
                continue
            idx = day_idx_map[pos.ticker]
            df = data_dict[pos.ticker]
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            pos.days_held += 1
            pos.peak_price = max(pos.peak_price, float(row["high"]))

            exit_reason, sell_price, normal_cash, day_pnl = _normal_exit_logic(
                pos, row, normal_cash, day_pnl
            )

            if exit_reason:
                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                cost = pos.shares * pos.buy_price
                pnl = proceeds - cost + (pos.half_sold_pnl * pos.allocated * 0.5 if pos.half_sold else 0)
                pnl_pct = (sell_price / pos.buy_price - 1) if not pos.half_sold else (
                    (pos.half_sold_pnl + (sell_price / pos.buy_price - 1)) / 2
                )
                normal_cash += proceeds
                pnl_pct_net = pnl_pct - COMMISSION * 2 - TAX

                trades.append({
                    "ticker": pos.ticker, "name": pos.name,
                    "buy_date": pos.buy_date, "sell_date": date,
                    "buy_price": pos.buy_price, "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net, "pnl_won": pnl,
                    "exit": exit_reason, "days": pos.days_held,
                    "grade": pos.grade, "regime": "NORMAL",
                    "regime_score": 0.0,
                    "counter": pos.counter, "freshness": pos.freshness,
                    "allocated": pos.allocated,
                })
                closed_n.append(pos)
        for c in closed_n:
            normal_positions.remove(c)

        # ═══ 2. 보유 관리: MOMENTUM 풀 (같은 C_new 청산 로직) ═══
        closed_m = []
        for pos in momentum_positions:
            if pos.ticker not in day_idx_map:
                continue
            idx = day_idx_map[pos.ticker]
            df = data_dict[pos.ticker]
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            pos.days_held += 1
            pos.peak_price = max(pos.peak_price, float(row["high"]))

            exit_reason, sell_price, momentum_cash, day_pnl = _normal_exit_logic(
                pos, row, momentum_cash, day_pnl
            )

            if exit_reason:
                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                cost = pos.shares * pos.buy_price
                pnl = proceeds - cost + (pos.half_sold_pnl * pos.allocated * 0.5 if pos.half_sold else 0)
                pnl_pct = (sell_price / pos.buy_price - 1) if not pos.half_sold else (
                    (pos.half_sold_pnl + (sell_price / pos.buy_price - 1)) / 2
                )
                momentum_cash += proceeds
                pnl_pct_net = pnl_pct - COMMISSION * 2 - TAX

                trades.append({
                    "ticker": pos.ticker, "name": pos.name,
                    "buy_date": pos.buy_date, "sell_date": date,
                    "buy_price": pos.buy_price, "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net, "pnl_won": pnl,
                    "exit": exit_reason, "days": pos.days_held,
                    "grade": pos.grade, "regime": "MOMENTUM",
                    "regime_score": pos.regime_score,
                    "counter": pos.counter, "freshness": pos.freshness,
                    "allocated": pos.allocated,
                })
                closed_m.append(pos)
        for c in closed_m:
            momentum_positions.remove(c)

        # ═══ 3. 신규 진입: NORMAL 풀 (기존 C_new) ═══
        all_held = {p.ticker for p in normal_positions} | {p.ticker for p in momentum_positions}
        kospi_regime, max_slots = get_kospi_regime(kospi_df, date)
        normal_available = max_slots - len(normal_positions)
        daily_entries_n = 0

        if normal_available > 0:
            normal_signals = scan_signals(data_dict, day_idx_map, regime_mode=False)
            for sig in normal_signals:
                if daily_entries_n >= MAX_DAILY_ENTRY:
                    break
                if normal_available <= 0:
                    break
                if sig["ticker"] in all_held:
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
                alloc = calc_position_size(normal_cash, sig["grade"], sig["freshness"])
                alloc = min(alloc, normal_cash)
                if alloc < 100000:
                    continue
                shares = int(alloc / buy_price)
                if shares <= 0:
                    continue
                actual_cost = shares * buy_price * (1 + COMMISSION)
                if actual_cost > normal_cash:
                    continue
                normal_cash -= actual_cost

                atr = sig["atr"]
                stop = max(buy_price - atr * 2, buy_price * 0.93)
                stop = max(stop, buy_price * 0.90)

                regime_stats["NORMAL"] += 1
                pos = Position(
                    ticker=ticker, name=name_map.get(ticker, ticker),
                    buy_date=date, buy_price=buy_price, shares=shares,
                    allocated=actual_cost, stop_loss=stop, target=sig["target"],
                    grade=sig["grade"], freshness=sig["freshness"],
                    counter=sig["counter"], conviction="MID",
                    regime_at_entry="NORMAL", regime_score=0.0,
                    peak_price=buy_price,
                )
                normal_positions.append(pos)
                all_held.add(ticker)
                daily_entries_n += 1
                normal_available -= 1

        # ═══ 4. 신규 진입: MOMENTUM 풀 (레짐 완화 freshness) ═══
        momentum_available = momentum_slots - len(momentum_positions)
        daily_entries_m = 0

        # KOSPI CRISIS이면 MOMENTUM도 진입 안 함
        if momentum_available > 0 and kospi_regime != "CRISIS":
            momentum_signals = scan_signals(data_dict, day_idx_map, regime_mode=True,
                                           regime_threshold=regime_threshold)
            # MOMENTUM으로 판별된 시그널만 필터
            momentum_only = [s for s in momentum_signals if s["regime"] == "MOMENTUM"]

            for sig in momentum_only:
                if daily_entries_m >= 1:  # MOMENTUM은 하루 1건 제한
                    break
                if momentum_available <= 0:
                    break
                if sig["ticker"] in all_held:
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
                # MOMENTUM 풀 사이징: 풀 자금 / 슬롯 수
                alloc = momentum_cash / momentum_slots
                alloc = min(alloc, momentum_cash)
                if alloc < 100000:
                    continue
                shares = int(alloc / buy_price)
                if shares <= 0:
                    continue
                actual_cost = shares * buy_price * (1 + COMMISSION)
                if actual_cost > momentum_cash:
                    continue
                momentum_cash -= actual_cost

                atr = sig["atr"]
                stop = max(buy_price - atr * 2, buy_price * 0.93)
                stop = max(stop, buy_price * 0.90)

                regime_stats["MOMENTUM"] += 1
                pos = Position(
                    ticker=ticker, name=name_map.get(ticker, ticker),
                    buy_date=date, buy_price=buy_price, shares=shares,
                    allocated=actual_cost, stop_loss=stop, target=sig["target"],
                    grade=sig["grade"], freshness=sig["freshness"],
                    counter=sig["counter"], conviction="MID",
                    regime_at_entry="MOMENTUM", regime_score=sig["regime_score"],
                    peak_price=buy_price,
                )
                momentum_positions.append(pos)
                all_held.add(ticker)
                daily_entries_m += 1
                momentum_available -= 1

        # ═══ 5. 일일 마감 ═══
        normal_value = sum(
            pos.shares * float(data_dict[pos.ticker].iloc[day_idx_map[pos.ticker]]["close"])
            for pos in normal_positions if pos.ticker in day_idx_map
        )
        momentum_value = sum(
            pos.shares * float(data_dict[pos.ticker].iloc[day_idx_map[pos.ticker]]["close"])
            for pos in momentum_positions if pos.ticker in day_idx_map
        )
        equity = normal_cash + momentum_cash + reserve_cash + normal_value + momentum_value

        daily_results.append(DayResult(
            date=date, equity=equity,
            cash=normal_cash + momentum_cash + reserve_cash,
            positions=len(normal_positions) + len(momentum_positions),
            daily_pnl=day_pnl,
        ))

    # 미청산 포지션 강제 청산
    for pool, pool_label in [(normal_positions, "NORMAL"), (momentum_positions, "MOMENTUM")]:
        for pos in pool:
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
                "grade": pos.grade, "regime": pool_label,
                "regime_score": pos.regime_score,
                "counter": pos.counter, "freshness": pos.freshness,
                "allocated": pos.allocated,
            })

    return trades, daily_results, regime_stats


# ── 리포트 ──

def report(trades, daily_results, label, regime_stats=None):
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

    exit_parts = []
    for key in ["target", "stop", "trailing", "time_stop", "timeout",
                 "supply_exit", "momentum_trail", "momentum_timeout",
                 "momentum_stop", "momentum_target", "force_close"]:
        if exits.get(key, 0) > 0:
            exit_parts.append(f"{key}:{exits[key]}")
    print(f"  청산: {' / '.join(exit_parts)}")
    print(f"  평균 보유: {np.mean([t['days'] for t in trades]):.1f}일")

    # 레짐별 분석
    if regime_stats and sum(regime_stats.values()) > 0:
        print(f"\n  레짐 분포 (진입 시):")
        total_r = sum(regime_stats.values())
        for level in ["MOMENTUM", "NORMAL"]:
            cnt = regime_stats.get(level, 0)
            pct = cnt / total_r * 100 if total_r > 0 else 0
            level_trades = [t for t in trades if t.get("regime") == level]
            if level_trades:
                level_wins = len([t for t in level_trades if t["pnl_pct"] > 0])
                level_wr = level_wins / len(level_trades) * 100
                level_avg = np.mean([t["pnl_pct"] for t in level_trades]) * 100
                level_pf_wins = sum(t["pnl_pct"] for t in level_trades if t["pnl_pct"] > 0)
                level_pf_losses = abs(sum(t["pnl_pct"] for t in level_trades if t["pnl_pct"] <= 0))
                level_pf = level_pf_wins / level_pf_losses if level_pf_losses > 0 else float("inf")
                print(f"    {level:>10}: {cnt:>3}건 ({pct:>4.1f}%) | "
                      f"승률 {level_wr:.0f}% | 평균 {level_avg:+.1f}% | PF {level_pf:.2f}")
            else:
                print(f"    {level:>10}: {cnt:>3}건 ({pct:>4.1f}%)")

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


def compare_results(base, test, test_label):
    """base vs test 비교 + 4-criteria 판정."""
    print(f"\n  {'지표':<18} {'C_new (기존)':>14} {test_label:>14} {'변화':>12}")
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
        v2 = test[key]
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"  {label:<18} {fmt.format(v1):>14} {fmt.format(v2):>14} {sign}{fmt.format(diff):>10}")

    pf_better = test["pf"] >= base["pf"]
    mdd_safe = test["mdd"] >= base["mdd"] - 1.0
    ret_better = test["total_return"] >= base["total_return"]
    sharpe_better = test["sharpe"] >= base["sharpe"]

    wins = sum([pf_better, mdd_safe, ret_better, sharpe_better])
    print(f"\n  판정:")
    print(f"    PF 개선:       {'PASS' if pf_better else 'FAIL'} ({base['pf']:.2f} → {test['pf']:.2f})")
    print(f"    MDD 안전:      {'PASS' if mdd_safe else 'FAIL'} ({base['mdd']:.1f}% → {test['mdd']:.1f}%)")
    print(f"    수익률 개선:   {'PASS' if ret_better else 'FAIL'} ({base['total_return']:+.1f}% → {test['total_return']:+.1f}%)")
    print(f"    Sharpe 개선:   {'PASS' if sharpe_better else 'FAIL'} ({base['sharpe']:.2f} → {test['sharpe']:.2f})")

    if wins >= 3:
        print(f"\n  ★ 결론: PASS ({wins}/4) — 채택 가능")
    elif wins >= 2:
        print(f"\n  결론: 부분 개선 ({wins}/4) — 파라미터 미세 조정 후 재검토")
    else:
        print(f"\n  결론: FAIL ({wins}/4) — 효과 없음, 기존 유지")
    return wins


# ── 메인 ──

def analyze_momentum_detail(trades, label):
    """MOMENTUM 풀 거래 상세 분석."""
    m_trades = [t for t in trades if t.get("regime") == "MOMENTUM"]
    if not m_trades:
        return None

    m_wins = [t["pnl_pct"] for t in m_trades if t["pnl_pct"] > 0]
    m_losses = [t["pnl_pct"] for t in m_trades if t["pnl_pct"] <= 0]
    m_pf = sum(m_wins) / abs(sum(m_losses)) if m_losses and sum(m_losses) != 0 else float("inf")
    m_wr = len(m_wins) / len(m_trades) * 100
    m_avg = np.mean([t["pnl_pct"] for t in m_trades]) * 100
    m_avg_days = np.mean([t["days"] for t in m_trades])

    # 레짐 스코어 분포
    scores = [t["regime_score"] for t in m_trades]
    avg_score = np.mean(scores) if scores else 0

    return {
        "label": label,
        "trades": len(m_trades),
        "pf": m_pf,
        "win_rate": m_wr,
        "avg_pnl": m_avg,
        "avg_days": m_avg_days,
        "avg_score": avg_score,
    }


def main():
    print("=" * 70)
    print("  D_B 미세 조정 스윕")
    print("  수급 악화 시 보유일 축소 — time_stop × timeout 조합 탐색")
    print("=" * 70)
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  초기 자본: {INITIAL_CAPITAL / 1e8:.1f}억 / 슬리피지: {SLIPPAGE * 100:.1f}%")
    print(f"  진입: C_new 100% 동일, 청산만 변경")
    print(f"  기존 D_B: time_stop 15→8, timeout 20→10 (2/4 PASS)")

    # ── 스윕 설정 ──
    # time_stop: 수급 악화 시 +3% 미만이면 탈출하는 기준일
    # timeout: 수급 악화 시 무조건 탈출하는 기준일
    SWEEP = [
        # (label, time_stop, timeout)
        ("B_10_12", 10, 12),    # 약간 완화
        ("B_10_14", 10, 14),    # 더 완화
        ("B_10_16", 10, 16),    # 거의 기존 수준
        ("B_12_15", 12, 15),    # time_stop만 약간 앞당김
        ("B_12_18", 12, 18),
        ("B_8_10",   8, 10),    # 기존 D_B (재확인)
        ("B_8_12",   8, 12),    # time_stop 유지, timeout 완화
        ("B_8_14",   8, 14),
        ("B_6_8",    6,  8),    # 더 공격적
        ("B_6_10",   6, 10),
    ]

    print(f"\n  {len(SWEEP)}개 조합 스윕")

    print("\n데이터 로딩...")
    data_dict = load_parquets()
    name_map = load_name_map()
    kospi_df = load_kospi_index()
    print(f"  {len(data_dict)}종목 로드\n")

    # ── 1. Baseline ──
    print("[C_new] 기존 v11.2 현행 실행 중...")
    trades_base, daily_base, stats_base = run_backtest(data_dict, name_map, "C_new", kospi_df)
    base = report(trades_base, daily_base, "C_new) v11.2 현행", stats_base)

    # ── 2. 스윕 실행 ──
    sweep_results = []

    for label, ts, to in SWEEP:
        print(f"\n[{label}] time_stop={ts}, timeout={to} 실행 중...")
        trades_d, daily_d, stats_d, supply_exits = run_backtest_defense(
            data_dict, name_map, kospi_df, exit_mode="B",
            b_time_stop=ts, b_timeout=to,
        )
        res = report(trades_d, daily_d, f"{label}) ts={ts} to={to}", stats_d)
        sweep_results.append({
            "label": label, "ts": ts, "to": to,
            "res": res, "trades": trades_d, "supply_exits": supply_exits,
        })

    # ── 3. 요약표 ──
    print(f"\n{'=' * 95}")
    print(f"  D_B 미세 조정 스윕 요약표")
    print(f"{'=' * 95}")

    header = (f"  {'설정':<10} {'ts':>3} {'to':>3} │ {'거래':>4} {'PF':>6} {'수익률':>8} "
              f"{'MDD':>7} {'Sharpe':>7} │ {'승률':>6} {'평균손':>7} {'수급청산':>6} │ {'판정':>5}")
    print(header)
    print(f"  {'-' * 88}")

    # Baseline
    print(f"  {'C_new':<10} {15:>3} {20:>3} │ {base['trades']:>4} {base['pf']:>6.2f} "
          f"{base['total_return']:>+7.1f}% {base['mdd']:>6.1f}% {base['sharpe']:>7.2f} │ "
          f"{base['win_rate']:>5.1f}% {base['avg_loss']:>6.1f}% {'─':>6} │ {'기준':>5}")

    best_score = 0
    best_entry = None

    for s in sweep_results:
        r = s["res"]
        se = s["supply_exits"]

        pf_ok = r["pf"] >= base["pf"]
        mdd_ok = r["mdd"] >= base["mdd"] - 1.0
        ret_ok = r["total_return"] >= base["total_return"]
        sharpe_ok = r["sharpe"] >= base["sharpe"]
        score = sum([pf_ok, mdd_ok, ret_ok, sharpe_ok])

        verdict = f"{score}/4"
        if score >= 3:
            verdict = f"★{score}/4"

        print(f"  {s['label']:<10} {s['ts']:>3} {s['to']:>3} │ {r['trades']:>4} {r['pf']:>6.2f} "
              f"{r['total_return']:>+7.1f}% {r['mdd']:>6.1f}% {r['sharpe']:>7.2f} │ "
              f"{r['win_rate']:>5.1f}% {r['avg_loss']:>6.1f}% {se:>6} │ {verdict:>5}")

        if score > best_score or (score == best_score and best_entry and
                                   r["total_return"] > best_entry["res"]["total_return"]):
            best_score = score
            best_entry = s

    # ── 4. 최적 설정 상세 비교 ──
    if best_entry:
        print(f"\n{'=' * 95}")
        print(f"  최적 설정: {best_entry['label']} (time_stop={best_entry['ts']}, timeout={best_entry['to']})")
        print(f"{'=' * 95}")
        compare_results(base, best_entry["res"], best_entry["label"])

        # 수급 청산 상세
        trades_d = best_entry["trades"]
        supply_related = [
            t for t in trades_d
            if t["exit"] in ("time_stop_early", "timeout_early")
        ]
        if supply_related:
            sr_wins = [t for t in supply_related if t["pnl_pct"] > 0]
            sr_losses = [t for t in supply_related if t["pnl_pct"] <= 0]
            sr_avg = np.mean([t["pnl_pct"] for t in supply_related]) * 100
            sr_avg_days = np.mean([t["days"] for t in supply_related])
            print(f"\n  수급 조기 청산 상세:")
            print(f"    {len(supply_related)}건: {len(sr_wins)}승 / {len(sr_losses)}패, "
                  f"평균 {sr_avg:+.1f}%, 평균 {sr_avg_days:.1f}일 보유")

            sr_tickers = {t["ticker"] for t in supply_related}
            base_same = [t for t in trades_base if t["ticker"] in sr_tickers]
            if base_same:
                base_avg = np.mean([t["pnl_pct"] for t in base_same]) * 100
                base_days = np.mean([t["days"] for t in base_same])
                print(f"    C_new 동일 종목: 평균 {base_avg:+.1f}%, 평균 {base_days:.1f}일")
                print(f"    → 효과: {sr_avg - base_avg:+.1f}%p")


if __name__ == "__main__":
    main()
