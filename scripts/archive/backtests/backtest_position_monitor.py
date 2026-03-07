"""
동적 목표가 재판정 백테스트

비교:
  A) 고정 규칙: 손절 -7%(ATR), 익절 +10% 반매도+트레일, 시간 15일
  B) 동적 목표가: 매일 4축(RSI/MACD/BB/수급) 기반 목표가 재계산 → 판정

공통:
  - 진입: v10.1 TRIX + Gate 시그널 (기존 backtest_v2와 동일)
  - KOSPI 레짐 캡 (C_new 방식)
  - 1억 초기자본, 최대 5종목, 슬리피지 0.5%

기간: 2025-03-01 ~ 2026-02-27
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# ── 설정 ──
START_DATE = "2025-03-01"
END_DATE = "2026-02-27"
INITIAL_CAPITAL = 100_000_000
MAX_POSITIONS = 5
MAX_DAILY_ENTRY = 2
SLIPPAGE = 0.005
COMMISSION = 0.00015
TAX = 0.0018

# 동적 목표가 설정 (settings.yaml position_monitor와 동일)
DYN_CFG = {
    "add_threshold": 1.08,
    "hold_threshold": 1.03,
    "partial_sell_threshold": 1.01,
    "full_sell_threshold": 0.97,
    "hard_stop_loss_pct": -8.0,
    "profit_take_pct": 12.0,
    "total_adj_max": 0.15,
    "total_adj_min": -0.15,
    "rsi": {
        "extreme_overbought": 75,
        "override_multiplier": 1.03,
        "overbought": 70,
        "overbought_adj": -0.02,
        "oversold": 30,
        "oversold_adj": 0.03,
        "near_oversold": 40,
        "near_oversold_adj": 0.015,
    },
    "macd": {
        "golden_cross_adj": 0.02,
        "dead_cross_adj": -0.03,
        "bearish_divergence_adj": -0.05,
        "bullish_divergence_adj": 0.02,
        "divergence_lookback": 20,
    },
    "bollinger": {
        "upper_breach": 100,
        "upper_breach_adj": 0.01,
        "lower_breach": 5,
        "lower_breach_adj": -0.05,
    },
    "supply": {
        "dual_buy_3d_adj": 0.02,
        "foreign_only_3d_adj": 0.015,
        "inst_only_3d_adj": 0.01,
        "dual_sell_3d_adj": -0.02,
        "foreign_sell_3d_adj": -0.015,
    },
}


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
    peak_price: float = 0.0
    half_sold: bool = False
    half_sold_pnl: float = 0.0
    days_held: int = 0


# ═══════════════════════════════════════════
# 데이터 로딩 (backtest_v2와 동일)
# ═══════════════════════════════════════════

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
    import re
    name_map = {}
    csv_dir = Path("stock_data_daily")
    if csv_dir.exists():
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
        return ("BULL", 5) if not pd.isna(rv_pct) and rv_pct < 0.50 else ("CAUTION", 3)
    elif close > ma60:
        return "BEAR", 2
    else:
        return "CRISIS", 0


# ═══════════════════════════════════════════
# 시그널 스캔 (backtest_v2와 동일)
# ═══════════════════════════════════════════

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


def scan_signals(data_dict, day_idx_map):
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

        h252 = df["high"].iloc[max(0, idx - 252):idx + 1].max()
        if h252 > 0 and (close / h252 - 1) < -0.20:
            continue
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
            "ticker": ticker, "close": close, "stop": stop,
            "target": target, "rr": rr, "zone": zone, "rank": rank,
            "grade": grade, "counter": counter, "freshness": freshness,
            "rsi": rsi, "atr": atr,
        })

    signals.sort(key=lambda s: s["rank"], reverse=True)
    return signals


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


# ═══════════════════════════════════════════
# 동적 목표가 4축 계산 (parquet 데이터만 사용)
# ═══════════════════════════════════════════

def calc_dynamic_target(df, idx, buy_price, current_price):
    """
    parquet 데이터로 4축(RSI/MACD/BB/수급) 동적 조정.
    base_target = max(진입시 목표가, 현재가×1.08) → 업사이드 추정
    """
    row = df.iloc[idx]
    base_target = max(buy_price * 1.10, current_price * 1.08)

    adj = 0.0
    cfg = DYN_CFG

    # ── 1. RSI ──
    rsi = float(row.get("rsi_14", 50) or 50)
    rsi_cfg = cfg["rsi"]
    rsi_override = None
    if rsi >= rsi_cfg["extreme_overbought"]:
        # 극과열: base_target을 현재가×1.03으로 강제
        base_target = current_price * rsi_cfg["override_multiplier"]
        rsi_override = True
    elif rsi >= rsi_cfg["overbought"]:
        adj += rsi_cfg["overbought_adj"]
    elif rsi <= rsi_cfg["oversold"]:
        adj += rsi_cfg["oversold_adj"]
    elif rsi <= rsi_cfg["near_oversold"]:
        adj += rsi_cfg["near_oversold_adj"]

    # ── 2. MACD ──
    if "macd_histogram" in df.columns and idx >= 1:
        hist_now = float(df["macd_histogram"].iloc[idx] or 0)
        hist_prev = float(df["macd_histogram"].iloc[idx - 1] or 0)
        macd_cfg = cfg["macd"]

        if hist_prev < 0 and hist_now > 0:
            adj += macd_cfg["golden_cross_adj"]
        elif hist_prev > 0 and hist_now < 0:
            adj += macd_cfg["dead_cross_adj"]

        # 다이버전스 (20일)
        lb = macd_cfg["divergence_lookback"]
        if idx >= lb:
            closes = df["close"].values[idx - lb + 1:idx + 1]
            macds = df["macd_histogram"].values[idx - lb + 1:idx + 1]
            if len(closes) == lb and len(macds) == lb:
                price_peak = int(np.argmax(closes))
                macd_peak = int(np.argmax(macds))
                if price_peak > macd_peak and macds[-1] < macds[-2] and hist_now > 0:
                    adj += macd_cfg["bearish_divergence_adj"]

                price_trough = int(np.argmin(closes))
                macd_trough = int(np.argmin(macds))
                if price_trough > macd_trough and macds[-1] > macds[-2] and hist_now < 0:
                    adj += macd_cfg["bullish_divergence_adj"]

    # ── 3. 볼린저 ──
    if "bb_position" in df.columns:
        bb_pos = float(row.get("bb_position", 50) or 50)
        bb_cfg = cfg["bollinger"]
        if bb_pos > bb_cfg["upper_breach"]:
            adj += bb_cfg["upper_breach_adj"]
        elif bb_pos < bb_cfg["lower_breach"]:
            adj += bb_cfg["lower_breach_adj"]

    # ── 4. 수급 ──
    sup_cfg = cfg["supply"]
    f_streak = int(row.get("foreign_consecutive_buy", 0) or 0)

    # 기관 스트릭 계산 (inst_net_streak 없으면 기관합계에서)
    i_streak = 0
    if "inst_net_streak" in df.columns:
        i_streak = int(row.get("inst_net_streak", 0) or 0)
    elif "기관합계" in df.columns and idx >= 4:
        inst_vals = df["기관합계"].values[max(0, idx - 4):idx + 1]
        if len(inst_vals) > 0:
            sign = 1 if inst_vals[-1] > 0 else -1
            cnt = 0
            for v in reversed(inst_vals):
                if (v > 0 and sign > 0) or (v < 0 and sign < 0):
                    cnt += sign
                else:
                    break
            i_streak = cnt

    if f_streak >= 3 and i_streak >= 3:
        adj += sup_cfg["dual_buy_3d_adj"]
    elif f_streak >= 3:
        adj += sup_cfg["foreign_only_3d_adj"]
    elif i_streak >= 3:
        adj += sup_cfg["inst_only_3d_adj"]
    elif f_streak <= -3 and i_streak <= -3:
        adj += sup_cfg["dual_sell_3d_adj"]
    elif f_streak <= -3:
        adj += sup_cfg["foreign_sell_3d_adj"]

    # ── 클램프 + 최종 ──
    adj = max(cfg["total_adj_min"], min(cfg["total_adj_max"], adj))
    final_target = base_target * (1 + adj)

    return final_target, adj, rsi_override is not None


def dynamic_action(final_target, current_price, pnl_pct):
    """동적 목표가 기반 판정."""
    cfg = DYN_CFG

    # 하드 스톱
    if pnl_pct <= cfg["hard_stop_loss_pct"]:
        return "FULL_SELL"

    if current_price <= 0:
        return "HOLD"

    ratio = final_target / current_price

    # 이익실현 보장
    if pnl_pct >= cfg["profit_take_pct"]:
        if ratio >= cfg["add_threshold"]:
            return "HOLD"
        return "PARTIAL_SELL"

    if ratio >= cfg["add_threshold"]:
        return "ADD"
    if ratio >= cfg["hold_threshold"]:
        return "HOLD"
    if ratio >= cfg["partial_sell_threshold"]:
        return "PARTIAL_SELL"
    return "FULL_SELL"


# ═══════════════════════════════════════════
# 백테스트 메인
# ═══════════════════════════════════════════

def run_backtest(data_dict, name_map, mode="A", kospi_df=None):
    """
    mode:
      A = 고정 규칙 (trail -8%, time 15d) — 기존 D모드 벤치마크
      B = 동적 목표가 4축 (매일 재판정)
      C = 하이브리드 (고정 손절 + 동적 익절)
      D = 하이브리드 강화 (C + 타이트 손절 -6% + MACD 데드크로스 긴급탈출)
    """
    ref_ticker = "005930"
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= START_DATE) & (ref_df.index <= END_DATE)]

    cash = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades = []
    equity_curve = []

    for date in dates:
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

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
            pnl_pct = (close_d / pos.buy_price - 1) * 100

            if mode == "A":
                # ── 고정 규칙 (기존 D모드) ──
                trail_pct = 0.92
                time_limit = 15

                atr_stop = pos.buy_price - pos.buy_price * 0.07
                stop_price = max(pos.stop_loss, atr_stop)
                stop_price = max(stop_price, pos.buy_price * 0.90)

                if not pos.half_sold and high >= pos.buy_price * 1.10:
                    sell_price_half = pos.buy_price * 1.10
                    half_shares = pos.shares // 2
                    if half_shares > 0:
                        proceeds = half_shares * sell_price_half * (1 - COMMISSION - TAX)
                        cash += proceeds
                        pos.half_sold = True
                        pos.half_sold_pnl = (sell_price_half / pos.buy_price - 1) - COMMISSION * 2 - TAX
                        pos.shares -= half_shares

                if pos.half_sold and low <= pos.peak_price * trail_pct:
                    exit_reason = "trailing"
                    sell_price = pos.peak_price * trail_pct
                elif low <= stop_price:
                    exit_reason = "stop"
                    sell_price = stop_price
                elif high >= pos.target:
                    exit_reason = "target"
                    sell_price = pos.target
                elif pos.days_held >= time_limit and not pos.half_sold:
                    if close_d < pos.buy_price * 1.03:
                        exit_reason = "time_stop"
                        sell_price = close_d
                elif pos.days_held >= 20:
                    exit_reason = "timeout"
                    sell_price = close_d

            elif mode == "B":
                # ── 동적 목표가 4축 ──
                final_target, adj_total, rsi_overridden = calc_dynamic_target(
                    df, idx, pos.buy_price, close_d,
                )
                action = dynamic_action(final_target, close_d, pnl_pct)

                if action == "FULL_SELL":
                    exit_reason = "dyn_full_sell"
                    sell_price = close_d
                elif action == "PARTIAL_SELL" and not pos.half_sold:
                    half_shares = pos.shares // 2
                    if half_shares > 0:
                        proceeds = half_shares * close_d * (1 - COMMISSION - TAX)
                        cash += proceeds
                        pos.half_sold = True
                        pos.half_sold_pnl = (close_d / pos.buy_price - 1) - COMMISSION * 2 - TAX
                        pos.shares -= half_shares
                elif action == "PARTIAL_SELL" and pos.half_sold:
                    exit_reason = "dyn_partial_close"
                    sell_price = close_d

                if not exit_reason and pos.days_held >= 25:
                    exit_reason = "timeout"
                    sell_price = close_d

            elif mode in ("C", "D"):
                # ── 하이브리드: 고정 손절 + 동적 익절 ──
                # 손절: 고정 ATR 기반 (A모드와 동일)
                if mode == "D":
                    hard_stop_mult = 0.94  # -6% 타이트
                    time_limit = 12
                else:
                    hard_stop_mult = 0.93  # -7%
                    time_limit = 15

                atr_stop = pos.buy_price * hard_stop_mult
                stop_price = max(pos.stop_loss, atr_stop)
                stop_price = max(stop_price, pos.buy_price * 0.90)  # 캡 -10%

                # 1) 고정 손절 (우선)
                if low <= stop_price:
                    exit_reason = "stop"
                    sell_price = stop_price
                else:
                    # 2) 동적 목표가로 익절 판단
                    final_target, adj_total, rsi_overridden = calc_dynamic_target(
                        df, idx, pos.buy_price, close_d,
                    )
                    action = dynamic_action(final_target, close_d, pnl_pct)

                    # D모드: MACD 데드크로스 + 수익 중이면 긴급탈출
                    if mode == "D" and pnl_pct > 0:
                        if "macd_histogram" in df.columns and idx >= 1:
                            h_now = float(df["macd_histogram"].iloc[idx] or 0)
                            h_prev = float(df["macd_histogram"].iloc[idx - 1] or 0)
                            if h_prev > 0 and h_now < 0:
                                exit_reason = "macd_exit"
                                sell_price = close_d

                    if not exit_reason:
                        if action == "FULL_SELL":
                            exit_reason = "dyn_full_sell"
                            sell_price = close_d
                        elif action == "PARTIAL_SELL" and not pos.half_sold:
                            half_shares = pos.shares // 2
                            if half_shares > 0:
                                proceeds = half_shares * close_d * (1 - COMMISSION - TAX)
                                cash += proceeds
                                pos.half_sold = True
                                pos.half_sold_pnl = (close_d / pos.buy_price - 1) - COMMISSION * 2 - TAX
                                pos.shares -= half_shares
                        elif action == "PARTIAL_SELL" and pos.half_sold:
                            exit_reason = "dyn_partial_close"
                            sell_price = close_d

                        # 반매도 후 트레일링 스톱 (고정 규칙 활용)
                        if not exit_reason and pos.half_sold:
                            trail_pct = 0.92 if mode == "C" else 0.93
                            if low <= pos.peak_price * trail_pct:
                                exit_reason = "trailing"
                                sell_price = pos.peak_price * trail_pct

                    # 시간 손절
                    if not exit_reason and pos.days_held >= time_limit and not pos.half_sold:
                        if close_d < pos.buy_price * 1.03:
                            exit_reason = "time_stop"
                            sell_price = close_d
                    if not exit_reason and pos.days_held >= 20:
                        exit_reason = "timeout"
                        sell_price = close_d

            if exit_reason:
                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                cash += proceeds

                pnl_pct_sell = (sell_price / pos.buy_price - 1)
                if pos.half_sold:
                    pnl_pct_net = (pos.half_sold_pnl + pnl_pct_sell) / 2
                else:
                    pnl_pct_net = pnl_pct_sell - COMMISSION * 2 - TAX

                trades.append({
                    "ticker": pos.ticker,
                    "name": name_map.get(pos.ticker, pos.ticker),
                    "buy_date": pos.buy_date,
                    "sell_date": date,
                    "buy_price": pos.buy_price,
                    "sell_price": sell_price,
                    "pnl_pct": pnl_pct_net,
                    "exit": exit_reason,
                    "days": pos.days_held,
                    "grade": pos.grade,
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

                alloc = calc_position_size(cash, sig["grade"], sig["freshness"])
                alloc = min(alloc, cash * 0.30, cash)
                buy_price = sig["close"] * (1 + SLIPPAGE)
                shares = int(alloc / buy_price)
                if shares <= 0:
                    continue

                cost = shares * buy_price * (1 + COMMISSION)
                if cost > cash:
                    shares = int(cash / (buy_price * (1 + COMMISSION)))
                    cost = shares * buy_price * (1 + COMMISSION)
                if shares <= 0:
                    continue

                cash -= cost
                positions.append(Position(
                    ticker=sig["ticker"],
                    name=name_map.get(sig["ticker"], sig["ticker"]),
                    buy_date=date,
                    buy_price=buy_price,
                    shares=shares,
                    allocated=cost,
                    stop_loss=sig["stop"],
                    target=sig["target"],
                    grade=sig["grade"],
                    freshness=sig["freshness"],
                    counter=sig["counter"],
                    peak_price=buy_price,
                ))
                held_tickers.add(sig["ticker"])
                available_slots -= 1
                daily_entries += 1

        # ── equity 계산 ──
        port_value = 0.0
        for pos in positions:
            if pos.ticker in day_idx_map:
                idx = day_idx_map[pos.ticker]
                c = float(data_dict[pos.ticker].iloc[idx]["close"])
                port_value += pos.shares * c
        equity = cash + port_value
        equity_curve.append({"date": date, "equity": equity, "positions": len(positions)})

    # 잔여 포지션 강제 청산
    for pos in positions:
        last_idx = len(data_dict[pos.ticker]) - 1
        close_d = float(data_dict[pos.ticker].iloc[last_idx]["close"])
        proceeds = pos.shares * close_d * (1 - COMMISSION - TAX)
        cash += proceeds
        pnl_pct_sell = (close_d / pos.buy_price - 1)
        if pos.half_sold:
            pnl_pct_net = (pos.half_sold_pnl + pnl_pct_sell) / 2
        else:
            pnl_pct_net = pnl_pct_sell - COMMISSION * 2 - TAX
        trades.append({
            "ticker": pos.ticker,
            "name": name_map.get(pos.ticker, pos.ticker),
            "buy_date": pos.buy_date,
            "sell_date": dates[-1] if len(dates) > 0 else pos.buy_date,
            "buy_price": pos.buy_price,
            "sell_price": close_d,
            "pnl_pct": pnl_pct_net,
            "exit": "force_close",
            "days": pos.days_held,
            "grade": pos.grade,
            "allocated": pos.allocated,
        })

    return trades, equity_curve


def summarize(trades, equity_curve, label):
    if not trades:
        print(f"\n[{label}] 거래 없음")
        return

    n = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n * 100
    avg_win = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) * 100 if losses else 0
    avg_days = np.mean([t["days"] for t in trades])

    gross_win = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    # equity 기반 수익률/MDD
    eq_df = pd.DataFrame(equity_curve)
    if not eq_df.empty:
        total_return = (eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["dd"] = (eq_df["equity"] / eq_df["peak"] - 1) * 100
        mdd = eq_df["dd"].min()
    else:
        total_return = 0
        mdd = 0

    # 엑싯 분포
    exit_dist = {}
    for t in trades:
        ex = t["exit"]
        exit_dist[ex] = exit_dist.get(ex, 0) + 1

    print(f"\n{'='*60}")
    print(f"  [{label}] 백테스트 결과 ({START_DATE} ~ {END_DATE})")
    print(f"{'='*60}")
    print(f"  총 거래: {n}건  |  승: {len(wins)}  |  패: {len(losses)}")
    print(f"  승률: {win_rate:.1f}%  |  PF: {pf:.2f}")
    print(f"  평균 수익(W): {avg_win:+.2f}%  |  평균 손실(L): {avg_loss:+.2f}%")
    print(f"  평균 보유일: {avg_days:.1f}일")
    print(f"  총 수익률: {total_return:+.1f}%  |  MDD: {mdd:.1f}%")
    print(f"  최종 자산: {eq_df['equity'].iloc[-1]:,.0f}원" if not eq_df.empty else "")
    print(f"  엑싯 분포: {exit_dist}")
    print(f"{'='*60}")

    return {
        "label": label, "trades": n, "wins": len(wins),
        "win_rate": win_rate, "pf": pf, "avg_win": avg_win,
        "avg_loss": avg_loss, "avg_days": avg_days,
        "total_return": total_return, "mdd": mdd,
        "exit_dist": exit_dist,
    }


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("데이터 로딩 중...")
    data_dict = load_parquets()
    name_map = load_name_map()
    kospi_df = load_kospi_index()
    print(f"  종목 수: {len(data_dict)}, KOSPI 인덱스: {'있음' if kospi_df is not None else '없음'}")

    # A) 고정 규칙
    print("\n[A] 고정 규칙 백테스트 실행 중...")
    trades_a, eq_a = run_backtest(data_dict, name_map, mode="A", kospi_df=kospi_df)
    res_a = summarize(trades_a, eq_a, "A: 고정 규칙 (트레일+시간손절)")

    # B) 동적 목표가
    print("\n[B] 동적 목표가 백테스트 실행 중...")
    trades_b, eq_b = run_backtest(data_dict, name_map, mode="B", kospi_df=kospi_df)
    res_b = summarize(trades_b, eq_b, "B: 동적 목표가 4축")

    # C) 하이브리드 (고정 손절 + 동적 익절)
    print("\n[C] 하이브리드 백테스트 실행 중...")
    trades_c, eq_c = run_backtest(data_dict, name_map, mode="C", kospi_df=kospi_df)
    res_c = summarize(trades_c, eq_c, "C: 하이브리드 (고정손절+동적익절)")

    # D) 하이브리드 강화 (타이트 손절 + MACD 긴급탈출)
    print("\n[D] 하이브리드 강화 백테스트 실행 중...")
    trades_d, eq_d = run_backtest(data_dict, name_map, mode="D", kospi_df=kospi_df)
    res_d = summarize(trades_d, eq_d, "D: 하이브리드+ (타이트손절+MACD탈출)")

    # 비교 요약
    results = [r for r in [res_a, res_b, res_c, res_d] if r]
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print(f"  4모드 비교 요약 ({START_DATE} ~ {END_DATE})")
        print(f"{'='*80}")
        headers = [r["label"].split(":")[0].strip() for r in results]
        print(f"  {'항목':<16}", end="")
        for h in headers:
            print(f" {h:>14}", end="")
        print()
        print(f"  {'-'*16}", end="")
        for _ in results:
            print(f" {'-'*14}", end="")
        print()

        rows = [
            ("총 거래", "trades", "{:>13d}건"),
            ("승률", "win_rate", "{:>13.1f}%"),
            ("Profit Factor", "pf", "{:>14.2f}"),
            ("평균수익(W)", "avg_win", "{:>13.2f}%"),
            ("평균손실(L)", "avg_loss", "{:>13.2f}%"),
            ("평균보유일", "avg_days", "{:>13.1f}d"),
            ("총수익률", "total_return", "{:>13.1f}%"),
            ("MDD", "mdd", "{:>13.1f}%"),
        ]

        for label, key, fmt in rows:
            print(f"  {label:<16}", end="")
            vals = [r[key] for r in results]
            for v in vals:
                print(f" {fmt.format(v)}", end="")
            # 최적 표시
            if key in ("win_rate", "pf", "total_return"):
                best = max(range(len(vals)), key=lambda i: vals[i])
                print(f"  ← {headers[best]}", end="")
            elif key in ("avg_loss", "mdd"):
                best = max(range(len(vals)), key=lambda i: vals[i])  # 가장 작은 음수 = 최선
                print(f"  ← {headers[best]}", end="")
            print()

        print(f"{'='*80}")
