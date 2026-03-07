"""
v12.3 MACRO GATE 백테스트 — C_new(기존) vs E(MACRO GATE) 비교

backtest_v2.py의 C_new 모드를 기반으로,
US Overnight cross_regime + grade + shock_type에 의한
매수 게이트 효과를 정량 검증한다.

사용법:
    python scripts/backtest_macro_gate.py
    python scripts/backtest_macro_gate.py --start 2025-11-01 --end 2026-02-28
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

# ── 기존 backtest_v2 모듈 재활용 ──
from scripts.backtest_v2 import (
    load_parquets, load_name_map, load_kospi_index,
    scan_signals, calc_position_size, calc_trix_counter,
    get_kospi_regime, KOSPI_REGIME_SLOTS,
    Position, DayResult,
    INITIAL_CAPITAL, MAX_POSITIONS, MAX_DAILY_ENTRY,
    SLIPPAGE, COMMISSION, TAX,
)

# ── US Daily Parquet (MACRO GATE 재구성용) ──
US_DAILY_PATH = Path("data/us_market/us_daily.parquet")


def load_us_daily():
    """US daily parquet → 날짜별 cross_regime + grade + shock_type 재구성."""
    if not US_DAILY_PATH.exists():
        print(f"  ⚠ US daily parquet 없음: {US_DAILY_PATH}")
        return pd.DataFrame()
    df = pd.read_parquet(US_DAILY_PATH)
    df = df.sort_index()
    return df


def reconstruct_cross_regime(us_df, date):
    """
    해당 날짜의 cross_regime 재구성.
    (전일 미국장 종가 = 한국장 당일 아침 기준)
    """
    # 한국 날짜 기준으로 전일 미국장 데이터 사용
    # us_df 인덱스는 US 거래일 → date 이전 가장 최근 거래일
    prev_dates = us_df.index[us_df.index < date]
    if len(prev_dates) == 0:
        return "GREEN", "NEUTRAL", "NONE"

    us_date = prev_dates[-1]
    row = us_df.loc[us_date]

    spy_ret = row.get("spy_ret_1d", 0) or 0  # 이미 % 단위 (예: -1.34)
    tnx_bp = row.get("tnx_change_bp", 0) or 0
    vix_level = row.get("vix_close", 0) or 0
    vix_zscore = row.get("vix_zscore", 0) or 0
    soxx_ret = row.get("soxx_ret_1d", 0) or 0
    uso_ret = row.get("uso_ret_1d", 0) or 0
    gld_ret = row.get("gld_ret_1d", 0) or 0
    copx_ret = row.get("copx_ret_1d", 0) or 0
    ewy_ret = row.get("ewy_ret_1d", 0) or 0

    # spy_ret는 %단위 (예: -1.34)
    spy_th = -1.0   # -1%
    tnx_th = 5.0    # 5bp (하지만 tnx_change_bp가 소수점 형태)

    # tnx_change_bp 단위 확인 — us_overnight_signal.py에서는 직접 계산
    # us_daily.parquet의 tnx_change_bp는 어떤 단위?
    # engine.py line 159: tnx_chg = latest["tnx_change_bp"]
    # parquet에서: tnx_change_bp 컬럼

    spy_pct = spy_ret / 100.0 if abs(spy_ret) > 1 else spy_ret  # normalize

    # Cross regime 판정 (engine.py 기반)
    # NOTE: parquet의 tnx_change_bp는 %p 단위 (0.05 = 5bp)
    # settings의 tnx_threshold_bp=5.0은 500bp라 버그 → 올바른 값 0.05 사용
    tnx_th_fixed = 0.05   # 5bp in %p units (settings의 5.0은 버그)
    spy_down = spy_pct <= spy_th / 100.0     # SPY -1% 이하
    spy_up = spy_pct >= abs(spy_th) / 100.0  # SPY +1% 이상
    tnx_up = tnx_bp >= tnx_th_fixed          # TNX +5bp 이상
    tnx_down = tnx_bp <= -tnx_th_fixed       # TNX -5bp 이하
    tnx_surge = tnx_bp >= tnx_th_fixed * 2   # TNX +10bp 이상

    cross_regime = "NEUTRAL"
    if spy_down and tnx_up:
        cross_regime = "VIGILANTE_VETO"   # 주식↓ + 금리↑ = RED
    elif spy_down and tnx_down:
        cross_regime = "RISK_OFF"         # 주식↓ + 금리↓
    elif spy_down:
        cross_regime = "CORRECTION"       # 단순 하락
    elif spy_up and not tnx_surge:
        cross_regime = "NORMAL_RALLY"
    elif spy_up and tnx_surge:
        cross_regime = "OVERHEAT_WARNING"
    elif tnx_surge:
        cross_regime = "RATE_SURGE"

    # ── 3색 신호등 ──
    color = "GREEN"
    if cross_regime in ("VIGILANTE_VETO", "RATE_SURGE"):
        color = "RED"
    elif cross_regime in ("RISK_OFF", "CORRECTION", "OVERHEAT_WARNING"):
        color = "YELLOW"

    # ── Grade 재구성 (간이: L1 score 기반) ──
    # 실제 combined_score를 정확히 재구성하기 어려우므로
    # spy_ret + ewy_ret + vix 기반 간이 판정
    score_proxy = ewy_ret * 2.5 + spy_ret * 1.5  # weighted
    if vix_level >= 25 and vix_zscore >= 1.5:
        score_proxy -= 15  # VIX 공포
    if vix_level >= 30:
        score_proxy -= 10

    if score_proxy >= 50:
        grade = "STRONG_BULL"
    elif score_proxy >= 20:
        grade = "MILD_BULL"
    elif score_proxy > -20:
        grade = "NEUTRAL"
    elif score_proxy > -50:
        grade = "MILD_BEAR"
    else:
        grade = "STRONG_BEAR"

    # STRONG_BEAR → RED 오버라이드
    if grade == "STRONG_BEAR":
        color = "RED"
    elif grade == "MILD_BEAR" and color == "GREEN":
        color = "YELLOW"

    # ── Shock type 재구성 (간이) ──
    shock_scores = {"GEOPOLITICAL": 0, "RATE": 0, "LIQUIDITY": 0, "EARNINGS": 0}
    if abs(uso_ret) > 3:
        shock_scores["GEOPOLITICAL"] += 35
    if abs(gld_ret) > 1.5:
        shock_scores["GEOPOLITICAL"] += 15
    if vix_level >= 25 and vix_zscore >= 1.5:
        shock_scores["GEOPOLITICAL"] += 20

    if abs(tnx_bp) >= 10:
        shock_scores["RATE"] += 40
    if cross_regime in ("VIGILANTE_VETO", "RATE_SURGE"):
        shock_scores["RATE"] += 20

    if vix_zscore >= 2.0:
        shock_scores["LIQUIDITY"] += 30
    if copx_ret <= -3:
        shock_scores["LIQUIDITY"] += 25

    if soxx_ret <= -4 and spy_pct > -0.02:
        shock_scores["EARNINGS"] += 45
    elif soxx_ret <= -5:
        shock_scores["EARNINGS"] += 30

    high_scores = [k for k, v in shock_scores.items() if v >= 40]
    if len(high_scores) >= 2:
        shock_type = "COMPOUND"
    elif high_scores:
        shock_type = high_scores[0]
    elif max(shock_scores.values()) >= 25:
        shock_type = max(shock_scores, key=shock_scores.get)
    else:
        shock_type = "NONE"

    # COMPOUND → GREEN이라도 제한
    if shock_type == "COMPOUND" and color == "GREEN":
        color = "YELLOW"  # COMPOUND는 GREEN→YELLOW 격상

    return color, grade, shock_type


def run_backtest_with_gate(data_dict, name_map, kospi_df, us_df,
                           start_date, end_date, use_macro_gate=False):
    """C_new + (선택적) MACRO GATE 백테스트."""
    ref_ticker = "005930"
    ref_df = data_dict[ref_ticker]
    dates = ref_df.index[(ref_df.index >= start_date) & (ref_df.index <= end_date)]

    capital = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    trades = []
    daily_results = []

    # MACRO GATE 통계
    gate_stats = {"GREEN": 0, "YELLOW": 0, "RED": 0,
                  "blocked_days": 0, "limited_days": 0,
                  "blocked_signals": 0, "limited_signals": 0}

    for date in dates:
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                ad = df.index[loc[0]]
                if abs((ad - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        day_pnl = 0.0

        # ── 1. 보유 종목 관리 ──
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

            # C_new 모드: 단순 손절/목표/타임아웃
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
                pnl_pct = (sell_price / pos.buy_price - 1) - COMMISSION * 2 - TAX
                cash += proceeds
                day_pnl += proceeds - pos.allocated

                trades.append({
                    "ticker": pos.ticker, "name": pos.name,
                    "buy_date": pos.buy_date, "sell_date": date,
                    "buy_price": pos.buy_price, "sell_price": sell_price,
                    "pnl_pct": pnl_pct,
                    "pnl_won": proceeds - pos.allocated,
                    "exit": exit_reason, "days": pos.days_held,
                    "grade": pos.grade, "counter": pos.counter,
                    "freshness": pos.freshness, "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입 ──
        held_tickers = {p.ticker for p in positions}

        # KOSPI 레짐 캡 (C_new)
        regime, max_slots = get_kospi_regime(kospi_df, date)
        available_slots = max_slots - len(positions)
        daily_entries = 0

        # ── MACRO GATE ──
        gate_cap = None  # None = 무제한
        gate_pos_mult = 1.0

        if use_macro_gate and len(us_df) > 0:
            color, us_grade, shock_type = reconstruct_cross_regime(us_df, date)
            gate_stats[color] += 1

            if color == "RED":
                gate_cap = 0
                gate_pos_mult = 0.0
                gate_stats["blocked_days"] += 1
            elif color == "YELLOW":
                gate_cap = 1  # yellow_max_survivors
                gate_pos_mult = 0.5
                gate_stats["limited_days"] += 1

        if available_slots > 0 and (gate_cap is None or gate_cap > 0):
            signals = scan_signals(data_dict, day_idx_map)

            # MACRO GATE cap 적용
            if gate_cap is not None and len(signals) > gate_cap:
                gate_stats["limited_signals"] += len(signals) - gate_cap
                signals = signals[:gate_cap]

            for sig in signals:
                if daily_entries >= MAX_DAILY_ENTRY:
                    break
                if available_slots <= 0:
                    break
                if sig["ticker"] in held_tickers:
                    continue
                if gate_cap is not None and daily_entries >= gate_cap:
                    break

                ticker = sig["ticker"]
                df = data_dict[ticker]
                idx = day_idx_map[ticker]

                if idx + 1 >= len(df):
                    continue
                next_open = float(df.iloc[idx + 1]["open"])
                if next_open <= 0:
                    continue

                buy_price = next_open * (1 + SLIPPAGE)

                alloc = calc_position_size(cash, sig["grade"], sig["freshness"], "C_new")
                alloc = min(alloc, cash)

                # MACRO GATE 포지션 배수
                alloc *= gate_pos_mult

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
                    ticker=ticker, name=name_map.get(ticker, ticker),
                    buy_date=date, buy_price=buy_price,
                    shares=shares, allocated=actual_cost,
                    stop_loss=stop, target=target,
                    grade=sig["grade"], freshness=sig["freshness"],
                    counter=sig["counter"], peak_price=buy_price,
                )
                positions.append(pos)
                held_tickers.add(ticker)
                daily_entries += 1
                available_slots -= 1
        elif gate_cap == 0:
            # RED: 시그널 있었으면 전부 차단
            blocked_sigs = scan_signals(data_dict, day_idx_map)
            gate_stats["blocked_signals"] += len(blocked_sigs)

        # ── 3. 일일 마감 ──
        position_value = sum(
            pos.shares * float(data_dict[pos.ticker].iloc[day_idx_map[pos.ticker]]["close"])
            for pos in positions if pos.ticker in day_idx_map
        )
        equity = cash + position_value

        daily_results.append(DayResult(
            date=date, equity=equity, cash=cash,
            positions=len(positions), daily_pnl=day_pnl
        ))

    # 미청산 강제 청산
    for pos in positions:
        last_close = pos.buy_price
        if pos.ticker in day_idx_map:
            last_close = float(data_dict[pos.ticker].iloc[day_idx_map[pos.ticker]]["close"])
        pnl_pct = (last_close / pos.buy_price - 1) - COMMISSION * 2 - TAX
        trades.append({
            "ticker": pos.ticker, "name": pos.name,
            "buy_date": pos.buy_date, "sell_date": dates[-1] if len(dates) > 0 else pd.Timestamp(end_date),
            "buy_price": pos.buy_price, "sell_price": last_close,
            "pnl_pct": pnl_pct, "pnl_won": pos.shares * last_close - pos.allocated,
            "exit": "force_close", "days": pos.days_held,
            "grade": pos.grade, "counter": pos.counter,
            "freshness": pos.freshness, "allocated": pos.allocated,
        })

    return trades, daily_results, gate_stats


def report(trades, daily_results, label, gate_stats=None):
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
          f"타임아웃{exits.get('timeout',0)} 미청산{exits.get('force_close',0)}")
    print(f"  평균 보유: {np.mean([t['days'] for t in trades]):.1f}일")

    print(f"\n  월별 수익률:")
    for m in sorted(monthly.keys()):
        val = monthly[m]
        bar = "█" * int(abs(val) / 2) if val > 0 else "░" * int(abs(val) / 2)
        sign = "+" if val > 0 else ""
        print(f"    {m}: {sign}{val:6.1f}% {bar}")

    if gate_stats:
        total_days = gate_stats["GREEN"] + gate_stats["YELLOW"] + gate_stats["RED"]
        print(f"\n  MACRO GATE 통계:")
        print(f"    🟢 GREEN: {gate_stats['GREEN']}일 ({gate_stats['GREEN']/total_days*100:.0f}%)" if total_days > 0 else "")
        print(f"    🟡 YELLOW: {gate_stats['YELLOW']}일 ({gate_stats['YELLOW']/total_days*100:.0f}%)" if total_days > 0 else "")
        print(f"    🔴 RED: {gate_stats['RED']}일 ({gate_stats['RED']/total_days*100:.0f}%)" if total_days > 0 else "")
        print(f"    차단 시그널: {gate_stats['blocked_signals']}건")
        print(f"    제한 시그널: {gate_stats['limited_signals']}건")

    return {
        "label": label,
        "trades": len(trades),
        "win_rate": win_rate,
        "pf": pf,
        "total_return": total_return,
        "mdd": mdd,
        "final_equity": final_equity,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-02-28")
    args = parser.parse_args()

    print(f"MACRO GATE 백테스트 — {args.start} ~ {args.end}")
    print("=" * 58)

    print("\n데이터 로딩...")
    data_dict = load_parquets()
    name_map = load_name_map()
    kospi_df = load_kospi_index()
    us_df = load_us_daily()

    print(f"  종목: {len(data_dict)}개")
    if kospi_df is not None:
        print(f"  KOSPI: {kospi_df.index[0].strftime('%Y-%m-%d')} ~ {kospi_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  US Daily: {us_df.index[0].strftime('%Y-%m-%d')} ~ {us_df.index[-1].strftime('%Y-%m-%d')}" if len(us_df) > 0 else "  US Daily: 없음")
    print(f"  기간: {args.start} ~ {args.end}")

    # KOSPI 레짐 분포
    if kospi_df is not None:
        regime_counts = {"BULL": 0, "CAUTION": 0, "BEAR": 0, "CRISIS": 0}
        ref_df = data_dict["005930"]
        test_dates = ref_df.index[(ref_df.index >= args.start) & (ref_df.index <= args.end)]
        for d in test_dates:
            r, _ = get_kospi_regime(kospi_df, d)
            regime_counts[r] += 1
        total_days = sum(regime_counts.values())
        print(f"\n  KOSPI 레짐 분포 ({total_days}일):")
        for r_name in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
            cnt = regime_counts[r_name]
            pct = cnt / total_days * 100 if total_days > 0 else 0
            slots = KOSPI_REGIME_SLOTS[r_name]
            print(f"    {r_name:>8} ({slots}슬롯): {cnt:>3}일 ({pct:>5.1f}%)")

    # US cross_regime 분포 미리 보기
    if len(us_df) > 0:
        color_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
        for d in test_dates:
            c, _, _ = reconstruct_cross_regime(us_df, d)
            color_counts[c] += 1
        total_days = sum(color_counts.values())
        print(f"\n  MACRO GATE 신호등 분포 ({total_days}일):")
        for c_name, emoji in [("GREEN", "🟢"), ("YELLOW", "🟡"), ("RED", "🔴")]:
            cnt = color_counts[c_name]
            pct = cnt / total_days * 100 if total_days > 0 else 0
            print(f"    {emoji} {c_name:>7}: {cnt:>3}일 ({pct:>5.1f}%)")

    results = []

    # ── A) C_new (기존: KOSPI 레짐만, MACRO GATE 없음) ──
    print("\n[1/2] C_new 실행 중...")
    trades_cnew, daily_cnew, stats_cnew = run_backtest_with_gate(
        data_dict, name_map, kospi_df, us_df,
        args.start, args.end, use_macro_gate=False
    )
    r1 = report(trades_cnew, daily_cnew, "C_new (기존: KOSPI 레짐만)")
    results.append(r1)

    # ── B) E (C_new + MACRO GATE) ──
    print("\n[2/2] E (MACRO GATE) 실행 중...")
    trades_e, daily_e, stats_e = run_backtest_with_gate(
        data_dict, name_map, kospi_df, us_df,
        args.start, args.end, use_macro_gate=True
    )
    r2 = report(trades_e, daily_e, "E (C_new + MACRO GATE)", gate_stats=stats_e)
    results.append(r2)

    # ── 비교 ──
    print(f"\n{'=' * 58}")
    print("  C_new vs E 비교")
    print(f"{'=' * 58}")
    if r1 and r2:
        metrics = [
            ("총 거래", "trades", "건", 0),
            ("승률", "win_rate", "%", 1),
            ("Profit Factor", "pf", "", 2),
            ("총 수익률", "total_return", "%", 1),
            ("MDD", "mdd", "%", 1),
            ("평균 수익", "avg_win", "%", 2),
            ("평균 손실", "avg_loss", "%", 2),
        ]
        print(f"  {'지표':<14} {'C_new':>10} {'E(Gate)':>10} {'차이':>10}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
        for label, key, unit, dec in metrics:
            v1 = r1.get(key, 0)
            v2 = r2.get(key, 0)
            diff = v2 - v1
            fmt = f".{dec}f"
            sign = "+" if diff > 0 else ""
            print(f"  {label:<14} {v1:>9{fmt}}{unit} {v2:>9{fmt}}{unit} {sign}{diff:>8{fmt}}{unit}")

    # 손절 비교 (가장 중요)
    if trades_cnew and trades_e:
        stops_cnew = [t for t in trades_cnew if t["exit"] == "stop"]
        stops_e = [t for t in trades_e if t["exit"] == "stop"]
        print(f"\n  손절 건수: C_new {len(stops_cnew)}건 → E {len(stops_e)}건 ({len(stops_e)-len(stops_cnew):+d})")

        # MACRO GATE가 차단한 날의 시그널 중 실제 손실인 것이 몇 개인지
        if stats_e["blocked_signals"] > 0:
            print(f"  RED 차단 시그널: {stats_e['blocked_signals']}건 (손실 회피)")


if __name__ == "__main__":
    main()
