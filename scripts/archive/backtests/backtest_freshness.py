"""
v10.1 Freshness 3축 백테스트 — 기존(v8.7) vs 신규(v10.1) 비교

기간: 2025-03-01 ~ 2026-02-13
대상: parquet 101종목

전략:
  매일 스캔 → 진입 시그널 발생 시 다음날 시가 매수
  목표가 도달 시 익절, 손절가 도달 시 손절
  최대 보유일 20일 초과 시 강제 청산

비교:
  A) v8.7 기존: T2 트리거면 진입 (TRIX 방향 무시)
  B) v10.1 신규: TRIX GC + Freshness 필터 적용
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# ── 설정 ──
START_DATE = "2025-03-01"
END_DATE = "2026-02-13"
MAX_HOLD_DAYS = 20
COMMISSION = 0.00015  # 편도 수수료 0.015%
TAX = 0.0018  # 매도 세금 0.18%


def load_parquets() -> dict[str, pd.DataFrame]:
    pq_dir = Path("data/processed")
    data = {}
    for f in sorted(pq_dir.glob("*.parquet")):
        ticker = f.stem
        if len(ticker) == 6 and ticker.isdigit():
            df = pd.read_parquet(f)
            if len(df) > 252:
                data[ticker] = df
    return data


def calc_trix_counter(df: pd.DataFrame, idx: int) -> int:
    trix = df.get("trix")
    sig = df.get("trix_signal")
    if trix is None or sig is None:
        return 0
    t_now, s_now = trix.iloc[idx], sig.iloc[idx]
    if pd.isna(t_now) or pd.isna(s_now):
        return 0
    current_above = t_now > s_now
    counter = 0
    for i in range(idx, max(idx - 60, -1), -1):
        t, s = trix.iloc[i], sig.iloc[i]
        if pd.isna(t) or pd.isna(s):
            break
        if current_above and t > s:
            counter += 1
        elif not current_above and t <= s:
            counter -= 1
        else:
            break
    return counter


def calc_freshness(counter: int, rsi: float, vol_c: float) -> float:
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
    high = df["high"].values
    low = df["low"].values
    plus_sum = sum(max(0, high[i] - high[i - 1]) for i in range(idx - period + 1, idx + 1))
    minus_sum = sum(max(0, low[i - 1] - low[i]) for i in range(idx - period + 1, idx + 1))
    return plus_sum, minus_sum


def scan_day(data_dict: dict, day_idx_map: dict, date: pd.Timestamp, mode: str):
    """하루 스캔 — mode='v87' 또는 'v101'"""
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

        # v8 Gate: ADX >= 15
        if adx < 15:
            continue

        # DI 방향
        plus_di, minus_di = calc_di(df, idx)
        if plus_di < minus_di and adx > 20:
            continue

        # Trigger 판정 (T1/T2/T3 간이)
        trix_above = trix_val > trix_sig
        macd_above = macd > macd_sig

        # T2: Volume/RSI (간이: 거래량 변화 + RSI 구간)
        vol = df["volume"].values
        vol_5 = np.mean(vol[max(0, idx - 4):idx + 1])
        vol_20 = np.mean(vol[max(0, idx - 19):idx + 1])
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

        # 트리거: TRIX or MACD 크로스 + 거래량/RSI 조건
        trigger = "none"
        if trix_above and macd_above:
            trigger = "confirm"
        elif trix_above:
            trigger = "impulse"
        elif macd_above and 35 <= rsi <= 65 and vol_ratio > 0.8:
            trigger = "T2"
        else:
            trigger = "none"

        if trigger == "none":
            continue

        # ATR 기반 목표/손절
        atr_series = df["high"].values - df["low"].values
        atr = np.mean(atr_series[max(0, idx - 13):idx + 1])
        if atr <= 0:
            atr = close * 0.02

        low_10 = df["low"].iloc[max(0, idx - 9):idx + 1].min()
        stop = max(low_10 - 0.5 * atr, close - 3 * atr)
        high_60 = df["high"].iloc[max(0, idx - 59):idx + 1].max()
        target = max(high_60, close + 1.5 * atr)

        risk = close - stop
        reward = target - close
        rr = reward / risk if risk > 0 else 0

        if rr < 1.0:
            continue

        # Zone Score (간이)
        ma20 = np.mean(df["close"].values[max(0, idx - 19):idx + 1])
        ma60 = float(row.get("sma_60", 0) or 0)
        ma120 = float(row.get("sma_120", 0) or 0)
        zone = 0.0
        if close > ma20: zone += 0.2
        if ma60 > 0 and close > ma60: zone += 0.2
        if ma120 > 0 and close > ma120: zone += 0.2
        if trix_above: zone += 0.2
        if 35 <= rsi <= 55: zone += 0.2

        # ── v10.1 필터 ──
        if mode == "v101":
            counter = calc_trix_counter(df, idx)
            vol_c = vol_5 / vol_20 if vol_20 > 0 else 1.0
            freshness = calc_freshness(counter, rsi, vol_c)

            if freshness == 0.0:
                continue

            # K11: 폭락
            h252 = df["high"].iloc[max(0, idx - 252):idx + 1].max()
            drawdown = (close / h252 - 1) * 100 if h252 > 0 else 0
            if drawdown < -20:
                continue

            # K12: MA120 하회
            if ma120 > 0 and close < ma120:
                continue

            rank = rr * zone * freshness
        else:
            # v8.7: TRIX 방향 무시, 기존 방식
            rank = rr * zone
            counter = 0
            freshness = 1.0

        signals.append({
            "ticker": ticker,
            "date": date,
            "close": close,
            "entry": close,  # 다음날 시가로 대체
            "stop": stop,
            "target": target,
            "rr": rr,
            "zone": zone,
            "rank": rank,
            "trigger": trigger,
            "counter": counter,
            "freshness": freshness,
            "rsi": rsi,
        })

    # 상위 3종목만
    signals.sort(key=lambda s: s["rank"], reverse=True)
    return signals[:3]


def simulate_trade(df: pd.DataFrame, entry_idx: int, stop: float, target: float):
    """진입 다음날 시가 매수 → 목표/손절/최대보유일 청산"""
    if entry_idx + 1 >= len(df):
        return None

    buy_price = float(df.iloc[entry_idx + 1]["open"])
    if buy_price <= 0:
        return None

    # 목표/손절을 매수가 기준으로 재계산
    risk_pct = (buy_price - stop) / buy_price
    reward_pct = (target - buy_price) / buy_price
    adj_stop = buy_price * (1 - min(risk_pct, 0.08))
    adj_target = buy_price * (1 + min(reward_pct, 0.15))

    for d in range(1, MAX_HOLD_DAYS + 1):
        if entry_idx + 1 + d >= len(df):
            break

        day = df.iloc[entry_idx + 1 + d]
        low = float(day["low"])
        high = float(day["high"])
        close_d = float(day["close"])

        # 손절 우선
        if low <= adj_stop:
            sell_price = adj_stop
            pnl_gross = (sell_price / buy_price) - 1
            pnl_net = pnl_gross - COMMISSION * 2 - TAX
            return {"pnl": pnl_net, "days": d, "exit": "stop"}

        # 목표 도달
        if high >= adj_target:
            sell_price = adj_target
            pnl_gross = (sell_price / buy_price) - 1
            pnl_net = pnl_gross - COMMISSION * 2 - TAX
            return {"pnl": pnl_net, "days": d, "exit": "target"}

    # 최대 보유일 초과 → 종가 청산
    last_idx = min(entry_idx + 1 + MAX_HOLD_DAYS, len(df) - 1)
    sell_price = float(df.iloc[last_idx]["close"])
    pnl_gross = (sell_price / buy_price) - 1
    pnl_net = pnl_gross - COMMISSION * 2 - TAX
    return {"pnl": pnl_net, "days": MAX_HOLD_DAYS, "exit": "timeout"}


def run_backtest(data_dict: dict, mode: str):
    """전체 기간 백테스트"""
    # 날짜별 인덱스 매핑
    ref_df = list(data_dict.values())[0]
    dates = ref_df.index[ref_df.index >= START_DATE]
    dates = dates[dates <= END_DATE]

    trades = []
    daily_positions = 0

    for date in dates:
        # 각 종목의 해당 날짜 인덱스
        day_idx_map = {}
        for ticker, df in data_dict.items():
            loc = df.index.get_indexer([date], method="pad")
            if loc[0] >= 0:
                actual_date = df.index[loc[0]]
                if abs((actual_date - date).days) <= 3:
                    day_idx_map[ticker] = loc[0]

        signals = scan_day(data_dict, day_idx_map, date, mode)

        for sig in signals:
            ticker = sig["ticker"]
            df = data_dict[ticker]
            idx = day_idx_map[ticker]

            result = simulate_trade(df, idx, sig["stop"], sig["target"])
            if result:
                trades.append({
                    "ticker": ticker,
                    "date": date,
                    "trigger": sig["trigger"],
                    "counter": sig["counter"],
                    "freshness": sig["freshness"],
                    "rsi": sig["rsi"],
                    **result,
                })

    return trades


def report(trades: list, label: str):
    if not trades:
        print(f"\n=== {label}: 거래 없음 ===")
        return {}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = sum(pnls) * 100
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # 누적 수익률 + MDD
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    mdd = np.min(dd) * 100

    # Exit 분포
    exits = defaultdict(int)
    for t in trades:
        exits[t["exit"]] += 1

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  총 거래: {len(trades)}건")
    print(f"  승률: {win_rate:.1f}% ({len(wins)}승 / {len(losses)}패)")
    print(f"  평균 수익: +{avg_win:.2f}% | 평균 손실: {avg_loss:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  총 수익률: {total_return:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  청산 분포: 목표 {exits['target']} | 손절 {exits['stop']} | 타임아웃 {exits['timeout']}")
    print(f"  평균 보유일: {np.mean([t['days'] for t in trades]):.1f}일")

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "pf": pf,
        "total_return": total_return,
        "mdd": mdd,
    }


def main():
    print("데이터 로딩...")
    data_dict = load_parquets()
    print(f"  {len(data_dict)}종목 로드 완료")

    print(f"\n백테스트 기간: {START_DATE} ~ {END_DATE}")

    # A: v8.7 기존 (TRIX 방향 무시)
    print("\n[A] v8.7 기존 스캔 실행 중...")
    trades_v87 = run_backtest(data_dict, "v87")
    r1 = report(trades_v87, "v8.7 기존 (TRIX 방향 무시, T2 허용)")

    # B: v10.1 Freshness
    print("\n[B] v10.1 Freshness 스캔 실행 중...")
    trades_v101 = run_backtest(data_dict, "v101")
    r2 = report(trades_v101, "v10.1 Freshness (TRIX GC 필수 + 3축)")

    # 비교
    if r1 and r2:
        print(f"\n{'=' * 55}")
        print(f"  비교 요약")
        print(f"{'=' * 55}")
        print(f"  {'지표':<15} {'v8.7':>10} {'v10.1':>10} {'변화':>10}")
        print(f"  {'-' * 45}")
        print(f"  {'거래 수':<15} {r1['trades']:>10} {r2['trades']:>10} {r2['trades']-r1['trades']:>+10}")
        print(f"  {'승률':<15} {r1['win_rate']:>9.1f}% {r2['win_rate']:>9.1f}% {r2['win_rate']-r1['win_rate']:>+9.1f}%")
        print(f"  {'PF':<15} {r1['pf']:>10.2f} {r2['pf']:>10.2f} {r2['pf']-r1['pf']:>+10.2f}")
        print(f"  {'총 수익률':<15} {r1['total_return']:>9.1f}% {r2['total_return']:>9.1f}% {r2['total_return']-r1['total_return']:>+9.1f}%")
        print(f"  {'MDD':<15} {r1['mdd']:>9.1f}% {r2['mdd']:>9.1f}% {r2['mdd']-r1['mdd']:>+9.1f}%")

    # 상위 거래 예시
    if trades_v101:
        print(f"\n  v10.1 상위 수익 거래 Top 5:")
        top = sorted(trades_v101, key=lambda t: t["pnl"], reverse=True)[:5]
        for t in top:
            print(f"    {t['date'].strftime('%Y-%m-%d')} {t['ticker']} "
                  f"+{t['pnl']*100:.1f}% ({t['exit']}, {t['days']}일, "
                  f"GC+{t['counter']}, F{t['freshness']:.2f})")

    if trades_v101:
        print(f"\n  v10.1 최악 손실 거래 Top 5:")
        worst = sorted(trades_v101, key=lambda t: t["pnl"])[:5]
        for t in worst:
            print(f"    {t['date'].strftime('%Y-%m-%d')} {t['ticker']} "
                  f"{t['pnl']*100:.1f}% ({t['exit']}, {t['days']}일, "
                  f"GC+{t['counter']}, F{t['freshness']:.2f})")


if __name__ == "__main__":
    main()
