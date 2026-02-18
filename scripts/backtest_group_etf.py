"""
그룹 ETF 순환매 백테스트 — TIGER 현대차그룹플러스 (138540)

핵심 아이디어:
  ETF는 올라가는데 개별 종목이 눌려있는 구간 포착
  z_20 < -0.8 (그룹 내 언더퍼폼) + z_5 > z_20 (단기 반등 시작)

2가지 모드 비교:
  A) 순수 평균회귀 (z-score only)
  B) 추세 병행 (z-score + TRIX/MACD 트렌드 필터)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# ── 설정 ──
START_DATE = "2025-03-01"
END_DATE = "2026-02-13"
INITIAL_CAPITAL = 40_000_000  # 4천만원 (1억의 40%)
MAX_POSITIONS = 2
SLIPPAGE = 0.005
COMMISSION = 0.00015
TAX = 0.0018

# 비중 가중 배분 계수
WEIGHT_MULT = {
    "005380": 1.2,   # 현대차
    "000270": 1.1,   # 기아
    "012330": 1.0,   # 현대모비스
    "004020": 0.8,   # 현대제철
    "086280": 0.8,   # 현대글로비스
}

DATA_DIR = Path("data/group_etf")
PQ_DIR = Path("data/processed")


@dataclass
class Position:
    ticker: str
    name: str
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    allocated: float
    z20_entry: float
    days_held: int = 0


def load_data():
    """ETF + 구성종목 데이터 로드"""
    # ETF
    etf_path = DATA_DIR / "etf_138540.csv"
    etf = pd.read_csv(etf_path, index_col="Date", parse_dates=True).sort_index()

    # 멤버
    with open(DATA_DIR / "members.json", "r", encoding="utf-8") as f:
        members = json.load(f)

    # 종목 데이터
    stocks = {}
    for m in members:
        ticker = m["ticker"]
        pq_path = PQ_DIR / f"{ticker}.parquet"
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            stocks[ticker] = df
        else:
            # CSV fallback (현대글로비스)
            csv_path = DATA_DIR / f"glovis_{ticker}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True).sort_index()
                stocks[ticker] = df

    return etf, members, stocks


def calc_rsi(series, period=14):
    """RSI 계산"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_trix(close, span=12, signal_span=9):
    """TRIX + Signal 계산"""
    ema1 = close.ewm(span=span).mean()
    ema2 = ema1.ewm(span=span).mean()
    ema3 = ema2.ewm(span=span).mean()
    trix = ema3.pct_change() * 100
    trix_signal = trix.ewm(span=signal_span).mean()
    return trix, trix_signal


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD + Signal 계산"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal


def prepare_indicators(etf, stocks, members):
    """상대강도 지표 + 추세 지표 전처리

    반환: {ticker: DataFrame with columns:
      close, ret_20, ret_5, etf_ret_20, etf_ret_5,
      rel_ret_20, rel_ret_5, rsi, trix, trix_signal, macd, macd_signal,
      ma60, high_60}
    """
    # ETF 수익률
    etf["ret_20"] = etf["close"].pct_change(20) * 100
    etf["ret_5"] = etf["close"].pct_change(5) * 100
    etf["ma20"] = etf["close"].rolling(20).mean()

    result = {}
    for m in members:
        ticker = m["ticker"]
        if ticker not in stocks:
            continue

        df = stocks[ticker].copy()
        close = df["close"]

        # 수익률
        df["ret_20"] = close.pct_change(20) * 100
        df["ret_5"] = close.pct_change(5) * 100

        # RSI
        df["rsi"] = calc_rsi(close)
        df["rsi_prev"] = df["rsi"].shift(1)

        # TRIX
        if "trix" in df.columns and "trix_signal" in df.columns:
            pass  # parquet에 이미 있음
        else:
            df["trix"], df["trix_signal"] = calc_trix(close)

        # MACD
        if "macd" in df.columns and "macd_signal" in df.columns:
            pass
        else:
            df["macd"], df["macd_signal"] = calc_macd(close)

        # MA60, 60일 고점
        df["ma60"] = close.rolling(60).mean()
        df["high_60"] = close.rolling(60).max()

        result[ticker] = df

    return result


def calc_z_scores(stock_indicators, etf, date, members):
    """특정 날짜의 5종목 z-score 계산

    반환: {ticker: {z_20, z_5, rel_ret_20, rel_ret_5, ...}}
    """
    scores = {}

    # ETF 수익률 (해당 날짜)
    etf_loc = etf.index.get_indexer([date], method="pad")
    if etf_loc[0] < 0:
        return {}
    etf_idx = etf_loc[0]
    etf_row = etf.iloc[etf_idx]
    if abs((etf.index[etf_idx] - date).days) > 3:
        return {}

    etf_ret_20 = etf_row.get("ret_20", np.nan)
    etf_ret_5 = etf_row.get("ret_5", np.nan)
    etf_close = etf_row["close"]
    etf_ma20 = etf_row.get("ma20", np.nan)

    if pd.isna(etf_ret_20) or pd.isna(etf_ret_5):
        return {}

    # 각 종목 상대 수익률
    raw = {}
    for m in members:
        ticker = m["ticker"]
        if ticker not in stock_indicators:
            continue
        df = stock_indicators[ticker]
        loc = df.index.get_indexer([date], method="pad")
        if loc[0] < 0:
            continue
        idx = loc[0]
        if abs((df.index[idx] - date).days) > 3:
            continue

        row = df.iloc[idx]
        ret_20 = row.get("ret_20", np.nan)
        ret_5 = row.get("ret_5", np.nan)
        if pd.isna(ret_20) or pd.isna(ret_5):
            continue

        rel_20 = ret_20 - etf_ret_20
        rel_5 = ret_5 - etf_ret_5

        raw[ticker] = {
            "close": float(row["close"]),
            "ret_20": float(ret_20),
            "ret_5": float(ret_5),
            "rel_ret_20": float(rel_20),
            "rel_ret_5": float(rel_5),
            "rsi": float(row.get("rsi", 50)),
            "rsi_prev": float(row.get("rsi_prev", 50)),
            "trix": float(row.get("trix", 0)),
            "trix_signal": float(row.get("trix_signal", 0)),
            "macd": float(row.get("macd", 0)),
            "macd_signal": float(row.get("macd_signal", 0)),
            "ma60": float(row.get("ma60", 0)),
            "high_60": float(row.get("high_60", 0)),
            "idx": idx,
        }

    if len(raw) < 3:  # 최소 3종목 필요
        return {}

    # Z-score 계산
    rel_20_vals = [v["rel_ret_20"] for v in raw.values()]
    rel_5_vals = [v["rel_ret_5"] for v in raw.values()]

    mean_20, std_20 = np.mean(rel_20_vals), np.std(rel_20_vals)
    mean_5, std_5 = np.mean(rel_5_vals), np.std(rel_5_vals)

    if std_20 < 0.01:
        std_20 = 1.0
    if std_5 < 0.01:
        std_5 = 1.0

    for ticker, data in raw.items():
        data["z_20"] = (data["rel_ret_20"] - mean_20) / std_20
        data["z_5"] = (data["rel_ret_5"] - mean_5) / std_5
        data["etf_close"] = float(etf_close)
        data["etf_ma20"] = float(etf_ma20) if not pd.isna(etf_ma20) else 0

    return raw


C_EXCLUDE = {"012330", "086280"}  # 현대모비스, 현대글로비스 (0승)


def check_entry(data, mode="A", ticker=""):
    """진입 조건 체크

    mode A: 순수 평균회귀
    mode B: 추세 병행 (TRIX/MACD > signal)
    mode C: A에서 문제종목 제거 + TRIX 개선방향 필터

    반환: (bool, 사유)
    """
    # 조건 1: ETF 추세 건재
    if data["etf_close"] <= data["etf_ma20"]:
        return False, "ETF<MA20"

    # 모드 C: 문제종목 제거
    if mode == "C" and ticker in C_EXCLUDE:
        return False, "excluded"

    # 조건 2: 개별 종목 눌림
    if data["z_20"] >= -0.8:
        return False, f"z20={data['z_20']:.2f}>=−0.8"
    if data["rel_ret_20"] >= -3.0:
        return False, f"rel20={data['rel_ret_20']:.1f}%>=−3%"

    # 조건 3: 반전 시그널
    reversal = False
    if data["z_5"] > data["z_20"]:
        reversal = True
    elif data["rsi_prev"] < 40 and data["rsi"] > data["rsi_prev"]:
        reversal = True
    if not reversal:
        return False, "no_reversal"

    # 조건 4: 바닥 확인
    if data["ma60"] > 0 and data["close"] < data["ma60"]:
        return False, "close<MA60"
    if data["high_60"] > 0:
        drop = (data["close"] / data["high_60"] - 1) * 100
        if drop < -25:
            return False, f"drop={drop:.0f}%<−25%"

    # 모드 B: 추세 필터 (TRIX/MACD > signal)
    if mode == "B":
        trix_ok = data["trix"] > data["trix_signal"]
        macd_ok = data["macd"] > data["macd_signal"]
        if not (trix_ok or macd_ok):
            return False, "no_trend(TRIX/MACD)"

    # 모드 C: TRIX 개선방향 필터 (아직 아래여도 괜찮지만, 상승 전환 중이어야)
    if mode == "C":
        trix_improving = data["trix"] > data["trix_signal"] or data["z_5"] > data["z_20"] + 0.3
        if not trix_improving:
            return False, "trix_not_improving"

    return True, "OK"


def check_exit(pos, z_scores, stock_data, etf_close, etf_ma20):
    """청산 조건 체크

    반환: (exit_reason or None, sell_price)
    """
    ticker = pos.ticker
    data = z_scores.get(ticker)
    close = stock_data.get("close", pos.buy_price) if stock_data else pos.buy_price

    # ETF 붕괴
    if etf_close < etf_ma20:
        return "etf_collapse", close

    if data:
        # 익절: z_20 >= 0 (ETF 수준 따라잡음)
        if data["z_20"] >= 0:
            return "z_recovery", close

    # 손절: -7%
    if close <= pos.buy_price * 0.93:
        return "stop_loss", pos.buy_price * 0.93

    # 시간: 20일 내 z_20 > -0.3 미도달
    if pos.days_held >= 20:
        if data and data["z_20"] < -0.3:
            return "time_stop", close
        elif not data:
            return "time_stop", close

    # 최대 30일
    if pos.days_held >= 30:
        return "max_time", close

    return None, 0.0


def run_backtest(etf, stock_indicators, members, mode="A"):
    """백테스트 실행"""
    name_map = {m["ticker"]: m["name"] for m in members}

    # 날짜 범위 (ETF 기준)
    dates = etf.index[(etf.index >= START_DATE) & (etf.index <= END_DATE)]

    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    trades = []
    daily_results = []

    # 일별 레짐 기록
    regime_log = {"active": 0, "paused": 0}

    for date in dates:
        # z-score 계산
        z_scores = calc_z_scores(stock_indicators, etf, date, members)
        if not z_scores:
            daily_results.append({"date": date, "equity": cash, "positions": 0})
            continue

        # ETF 상태
        sample = next(iter(z_scores.values()))
        etf_close = sample["etf_close"]
        etf_ma20 = sample["etf_ma20"]
        etf_active = etf_close > etf_ma20

        if etf_active:
            regime_log["active"] += 1
        else:
            regime_log["paused"] += 1

        day_pnl = 0.0

        # ── 1. 보유 종목 청산 체크 ──
        closed = []
        for pos in positions:
            pos.days_held += 1
            stock_data = z_scores.get(pos.ticker)
            exit_reason, sell_price = check_exit(
                pos, z_scores, stock_data, etf_close, etf_ma20
            )

            if exit_reason:
                if stock_data:
                    sell_price = max(sell_price, stock_data["close"] * (1 - SLIPPAGE))
                else:
                    sell_price = sell_price * (1 - SLIPPAGE)

                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                pnl_pct = (sell_price / pos.buy_price - 1) - COMMISSION * 2 - TAX

                cash += proceeds
                day_pnl += proceeds - pos.allocated

                trades.append({
                    "ticker": pos.ticker,
                    "name": name_map.get(pos.ticker, pos.ticker),
                    "buy_date": pos.buy_date,
                    "sell_date": date,
                    "buy_price": pos.buy_price,
                    "sell_price": sell_price,
                    "pnl_pct": pnl_pct,
                    "exit": exit_reason,
                    "days": pos.days_held,
                    "z20_entry": pos.z20_entry,
                    "z20_exit": stock_data["z_20"] if stock_data else np.nan,
                    "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입 ──
        if etf_active and len(positions) < MAX_POSITIONS:
            held_tickers = {p.ticker for p in positions}
            candidates = []

            for ticker, data in z_scores.items():
                if ticker in held_tickers:
                    continue
                ok, reason = check_entry(data, mode=mode, ticker=ticker)
                if ok:
                    candidates.append((ticker, data))

            # z_20 가장 낮은 순 (가장 많이 빠진 종목 우선)
            candidates.sort(key=lambda x: x[1]["z_20"])

            for ticker, data in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break

                # 포지션 사이징
                base_alloc = INITIAL_CAPITAL / MAX_POSITIONS
                w_mult = WEIGHT_MULT.get(ticker, 1.0)
                alloc = base_alloc * w_mult
                alloc = min(alloc, cash)

                if alloc < 100000:
                    continue

                # 다음 봉 시가 매수 (당일 종가 기준 시뮬레이션)
                buy_price = data["close"] * (1 + SLIPPAGE)
                shares = int(alloc / buy_price)
                if shares <= 0:
                    continue

                actual_cost = shares * buy_price * (1 + COMMISSION)
                if actual_cost > cash:
                    continue

                cash -= actual_cost

                positions.append(Position(
                    ticker=ticker,
                    name=name_map.get(ticker, ticker),
                    buy_date=date,
                    buy_price=buy_price,
                    shares=shares,
                    allocated=actual_cost,
                    z20_entry=data["z_20"],
                ))

        # ── 3. 일일 마감 ──
        pos_value = 0
        for pos in positions:
            data = z_scores.get(pos.ticker)
            if data:
                pos_value += pos.shares * data["close"]
            else:
                pos_value += pos.allocated  # fallback

        equity = cash + pos_value
        daily_results.append({"date": date, "equity": equity, "positions": len(positions)})

    # 미청산 강제 청산
    for pos in positions:
        last_data = z_scores.get(pos.ticker) if z_scores else None
        if last_data:
            sell_price = last_data["close"]
        else:
            sell_price = pos.buy_price
        pnl_pct = (sell_price / pos.buy_price - 1) - COMMISSION * 2 - TAX
        trades.append({
            "ticker": pos.ticker,
            "name": name_map.get(pos.ticker, pos.ticker),
            "buy_date": pos.buy_date,
            "sell_date": dates[-1],
            "buy_price": pos.buy_price,
            "sell_price": sell_price,
            "pnl_pct": pnl_pct,
            "exit": "force_close",
            "days": pos.days_held,
            "z20_entry": pos.z20_entry,
            "z20_exit": last_data["z_20"] if last_data else np.nan,
            "allocated": pos.allocated,
        })

    return trades, daily_results, regime_log


def report(trades, daily_results, regime_log, label, members):
    """결과 리포트"""
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

    # 자본 곡선 MDD
    equities = [d["equity"] for d in daily_results]
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100
    mdd = np.min(dd_pct)

    final_equity = equities[-1] if equities else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    exits = defaultdict(int)
    for t in trades:
        exits[t["exit"]] += 1

    print(f"\n{'=' * 58}")
    print(f"  {label}")
    print(f"{'=' * 58}")
    print(f"  초기 자본: {INITIAL_CAPITAL / 1e4:,.0f}만원 → 최종: {final_equity / 1e4:,.0f}만원")
    print(f"  총 수익률: {total_return:+.1f}%")
    print(f"  총 거래: {len(trades)}건")
    print(f"  승률: {win_rate:.1f}% ({len(wins)}승 / {len(losses)}패)")
    print(f"  평균 수익: +{avg_win:.2f}% | 평균 손실: {avg_loss:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  평균 보유: {np.mean([t['days'] for t in trades]):.1f}일")
    print(f"  청산: z회복{exits.get('z_recovery',0)} 손절{exits.get('stop_loss',0)} "
          f"시간{exits.get('time_stop',0)} ETF붕괴{exits.get('etf_collapse',0)} "
          f"최대{exits.get('max_time',0)} 강제{exits.get('force_close',0)}")

    # 레짐
    active = regime_log["active"]
    paused = regime_log["paused"]
    total_d = active + paused
    print(f"\n  ETF MA20 기준: 활성 {active}일 ({active/total_d*100:.0f}%) | "
          f"중지 {paused}일 ({paused/total_d*100:.0f}%)")

    # 종목별 분석
    print(f"\n  종목별 성과:")
    name_map = {m["ticker"]: m["name"] for m in members}
    for m in members:
        ticker = m["ticker"]
        t_trades = [t for t in trades if t["ticker"] == ticker]
        if not t_trades:
            print(f"    {name_map[ticker]:>10}: 거래 없음")
            continue
        t_pnls = [t["pnl_pct"] for t in t_trades]
        t_wins = [p for p in t_pnls if p > 0]
        t_avg_z = np.mean([t["z20_entry"] for t in t_trades])
        print(f"    {name_map[ticker]:>10}: {len(t_trades)}건, "
              f"승률 {len(t_wins)/len(t_trades)*100:.0f}%, "
              f"평균z진입 {t_avg_z:.2f}, "
              f"순PnL {sum(t_pnls)*100:+.1f}%")

    # z_20 진입 분포
    z_entries = [t["z20_entry"] for t in trades if not pd.isna(t["z20_entry"])]
    if z_entries:
        print(f"\n  z_20 진입 분포:")
        print(f"    평균: {np.mean(z_entries):.2f}, 중앙: {np.median(z_entries):.2f}")
        print(f"    범위: [{min(z_entries):.2f}, {max(z_entries):.2f}]")

    # 최대 수익/손실 거래
    best = max(trades, key=lambda t: t["pnl_pct"])
    worst = min(trades, key=lambda t: t["pnl_pct"])
    print(f"\n  최대 수익: {best['name']}({best['ticker']}) "
          f"{best['pnl_pct']*100:+.1f}% ({best['exit']}, {best['days']}일)")
    print(f"  최대 손실: {worst['name']}({worst['ticker']}) "
          f"{worst['pnl_pct']*100:+.1f}% ({worst['exit']}, {worst['days']}일)")

    # 월별 수익
    monthly = defaultdict(float)
    for t in trades:
        key = t["sell_date"].strftime("%Y-%m")
        monthly[key] += t["pnl_pct"] * 100

    print(f"\n  월별 수익률:")
    for m_key in sorted(monthly.keys()):
        bar = "+" * int(abs(monthly[m_key]) / 2) if monthly[m_key] > 0 else "-" * int(abs(monthly[m_key]) / 2)
        print(f"    {m_key}: {monthly[m_key]:+6.1f}% {bar}")

    return {
        "label": label,
        "trades": len(trades),
        "win_rate": win_rate,
        "pf": pf,
        "total_return": total_return,
        "mdd": mdd,
        "final_equity": final_equity,
        "avg_days": np.mean([t["days"] for t in trades]),
    }


def main():
    print("데이터 로딩...")
    etf, members, stocks = load_data()
    print(f"  ETF: {len(etf)}행")
    print(f"  종목: {', '.join(m['name'] for m in members)}")

    print("\n지표 계산...")
    stock_indicators = prepare_indicators(etf, stocks, members)
    print(f"  {len(stock_indicators)}종목 지표 완료")

    results = []

    for mode, label in [
        ("A", "A) 순수 평균회귀 (z-score only, 5종목)"),
        ("B", "B) 추세 병행 (z-score + TRIX/MACD, 5종목)"),
        ("C", "C) 정제 (3종목 + TRIX개선방향)"),
    ]:
        print(f"\n[{mode}] {label} 실행 중...")
        trades, daily, regime = run_backtest(etf, stock_indicators, members, mode)
        r = report(trades, daily, regime, label, members)
        if r:
            results.append(r)

    # 비교
    if len(results) >= 2:
        print(f"\n{'=' * 75}")
        print(f"  종합 비교")
        print(f"{'=' * 75}")
        print(f"  {'모드':<45} {'거래':>5} {'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>8}")
        print(f"  {'-' * 70}")
        for r in results:
            print(f"  {r['label']:<45} {r['trades']:>5} "
                  f"{r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['total_return']:>+7.1f}% {r['mdd']:>7.1f}%")

        # 최적 모드 판정
        valid = [r for r in results if r["pf"] > 1.0]
        if valid:
            best = max(valid, key=lambda r: r["pf"])
            print(f"\n  PF 기준 최적: {best['label']}")

        # v10.1 참고 비교
        print(f"\n  참고: v10.1 C_new모드 (84종목, KOSPI레짐) 결과")
        print(f"    거래 82건, 승률 50.0%, PF 1.78, 수익률 +21.7%, MDD -4.5%")
        print(f"    → 그룹ETF(40%) + v10.1(60%) 조합 시 분산 효과 기대")


if __name__ == "__main__":
    main()
