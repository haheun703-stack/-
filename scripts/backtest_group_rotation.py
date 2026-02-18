"""
3층 순환매 백테스트 — 현대차+삼성 그룹 + EWY

Layer 1 (Market): EWY > MA20 → 진입 허용
Layer 2 (Group): 그룹 ETF > MA20 + 그룹당 최대 2종목 + 눌림그룹 우선
Layer 3 (Stock): z-score 평균회귀 (z_20 < -0.8 + 반전 시그널)

4가지 모드:
  A) 현대차그룹 단독 (3종목)
  B) 삼성그룹 단독 (4종목)
  C) 현대차+삼성 통합 (7종목, L2 적용)
  D) 3층 완성 (L1: EWY + L2 + L3)

목표: D모드 거래 20건+ AND PF 1.5+ AND MDD -7% 이내
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
INITIAL_CAPITAL = 40_000_000  # 4천만원
MAX_POSITIONS = 3
SLIPPAGE = 0.005
COMMISSION = 0.00015
TAX = 0.0018

MAX_PER_GROUP = 2  # Mode C/D: 그룹당 최대 보유
SAMSUNG_EXCLUDE = {"032830"}  # 삼성생명 (삼성물산과 r=0.985)

# ETF 비중 기반 가중 배분
WEIGHT_MULT = {
    "005380": 1.2,   # 현대차 (30.84%)
    "000270": 1.1,   # 기아 (24.4%)
    "004020": 0.9,   # 현대제철 (7.56%)
    "005930": 1.2,   # 삼성전자 (32.45%)
    "028260": 1.0,   # 삼성물산 (16.11%)
    "006400": 0.9,   # 삼성SDI (9.6%)
    "000810": 0.9,   # 삼성화재 (7.11%)
}

GROUP_PENALTY_MAX = 0.3  # Level 2: 눌림그룹 우선 가중치

DATA_DIR = Path("data/group_rotation")
PQ_DIR = Path("data/processed")


@dataclass
class Position:
    ticker: str
    name: str
    group: str
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    allocated: float
    z20_entry: float
    days_held: int = 0


# ── 데이터 로드 ──

def load_all_data():
    """그룹 멤버 + ETF + EWY + 종목 데이터 로드"""
    with open(DATA_DIR / "members.json", "r", encoding="utf-8") as f:
        groups = json.load(f)

    # 삼성생명 제외
    for gname, gdata in groups.items():
        gdata["members"] = [m for m in gdata["members"]
                            if m["ticker"] not in SAMSUNG_EXCLUDE]

    # 그룹 ETF
    etfs = {}
    for gname, gdata in groups.items():
        etfs[gname] = pd.read_csv(
            DATA_DIR / gdata["etf_file"],
            index_col="Date", parse_dates=True
        ).sort_index()

    # EWY
    ewy = pd.read_csv(
        DATA_DIR / "etf_ewy.csv",
        index_col="Date", parse_dates=True
    ).sort_index()

    # 종목 데이터 (parquet)
    stocks = {}
    for gname, gdata in groups.items():
        for m in gdata["members"]:
            ticker = m["ticker"]
            pq = PQ_DIR / f"{ticker}.parquet"
            if pq.exists():
                stocks[ticker] = pd.read_parquet(pq)

    return groups, etfs, ewy, stocks


# ── 지표 계산 ──

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_trix(close, span=12, signal_span=9):
    ema1 = close.ewm(span=span).mean()
    ema2 = ema1.ewm(span=span).mean()
    ema3 = ema2.ewm(span=span).mean()
    trix = ema3.pct_change() * 100
    trix_signal = trix.ewm(span=signal_span).mean()
    return trix, trix_signal


def prepare_indicators(groups, etfs, ewy, stocks):
    """ETF + EWY + 종목 지표 전처리"""
    # ETF 지표
    for gname, etf in etfs.items():
        etf["ret_20"] = etf["close"].pct_change(20) * 100
        etf["ret_5"] = etf["close"].pct_change(5) * 100
        etf["ma20"] = etf["close"].rolling(20).mean()

    # EWY 지표
    ewy["ret_20"] = ewy["close"].pct_change(20) * 100
    ewy["ma20"] = ewy["close"].rolling(20).mean()

    # 종목 지표
    indicators = {}
    for gname, gdata in groups.items():
        for m in gdata["members"]:
            ticker = m["ticker"]
            if ticker not in stocks:
                continue

            df = stocks[ticker].copy()
            close = df["close"]

            df["ret_20"] = close.pct_change(20) * 100
            df["ret_5"] = close.pct_change(5) * 100
            df["rsi"] = calc_rsi(close)
            df["rsi_prev"] = df["rsi"].shift(1)

            if "trix" not in df.columns or "trix_signal" not in df.columns:
                df["trix"], df["trix_signal"] = calc_trix(close)

            df["ma60"] = close.rolling(60).mean()
            df["high_60"] = close.rolling(60).max()

            indicators[ticker] = df

    return indicators


# ── Z-Score 계산 ──

def get_etf_state(etf, date):
    """ETF의 close/ma20/ret_20 조회 (pad 방식)"""
    loc = etf.index.get_indexer([date], method="pad")
    if loc[0] < 0:
        return None
    idx = loc[0]
    if abs((etf.index[idx] - date).days) > 5:
        return None
    row = etf.iloc[idx]
    return {
        "close": float(row["close"]),
        "ma20": float(row.get("ma20", 0)),
        "ret_20": float(row.get("ret_20", 0)),
    }


def calc_group_z_scores(stock_indicators, etf, date, members):
    """특정 그룹의 종목별 z-score 계산

    반환: {ticker: {z_20, z_5, close, rel_ret_20, rel_ret_5, rsi, ...}}
    """
    etf_state = get_etf_state(etf, date)
    if not etf_state:
        return {}

    etf_ret_20 = etf_state["ret_20"]
    if pd.isna(etf_ret_20):
        return {}

    # ETF 5일 수익률
    loc = etf.index.get_indexer([date], method="pad")
    etf_ret_5 = float(etf.iloc[loc[0]].get("ret_5", 0))

    # 종목별 상대 수익률
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

        raw[ticker] = {
            "close": float(row["close"]),
            "ret_20": float(ret_20),
            "ret_5": float(ret_5),
            "rel_ret_20": float(ret_20) - etf_ret_20,
            "rel_ret_5": float(ret_5) - etf_ret_5,
            "rsi": float(row.get("rsi", 50)),
            "rsi_prev": float(row.get("rsi_prev", 50)),
            "trix": float(row.get("trix", 0)),
            "trix_signal": float(row.get("trix_signal", 0)),
            "ma60": float(row.get("ma60", 0)),
            "high_60": float(row.get("high_60", 0)),
        }

    if len(raw) < 2:
        return {}

    # Z-score
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

    return raw


# ── 진입/청산 로직 ──

def check_entry(data, etf_state):
    """진입 4조건 체크

    1. 그룹 ETF > MA20
    2. z_20 < -0.8 AND rel_ret_20 < -3%
    3. z_5 > z_20 OR RSI 반전
    4. close > MA60 AND 60일 낙폭 < 25%
    """
    # 조건 1: ETF 추세 건재
    if not etf_state or etf_state["close"] <= etf_state["ma20"]:
        return False, "ETF<MA20"

    # 조건 2: 눌림
    if data["z_20"] >= -0.8:
        return False, f"z20={data['z_20']:.2f}"
    if data["rel_ret_20"] >= -3.0:
        return False, f"rel20={data['rel_ret_20']:.1f}%"

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
            return False, f"drop={drop:.0f}%"

    return True, "OK"


def check_exit(pos, z_data, stock_close, etf_state, ewy_state=None, mode="A"):
    """청산 조건 체크"""
    # 그룹 ETF 붕괴
    if etf_state and etf_state["close"] < etf_state["ma20"]:
        return "etf_collapse", stock_close

    # Mode D: EWY 붕괴
    if mode == "D" and ewy_state and ewy_state["close"] < ewy_state["ma20"]:
        return "ewy_collapse", stock_close

    # z 회복 (ETF 수준 따라잡음)
    if z_data and z_data["z_20"] >= 0:
        return "z_recovery", stock_close

    # 손절 -7%
    if stock_close <= pos.buy_price * 0.93:
        return "stop_loss", pos.buy_price * 0.93

    # 시간 정지: 20일 내 z_20 > -0.3 미도달
    if pos.days_held >= 20:
        if z_data and z_data["z_20"] < -0.3:
            return "time_stop", stock_close
        elif not z_data:
            return "time_stop", stock_close

    # 최대 30일
    if pos.days_held >= 30:
        return "max_time", stock_close

    return None, 0.0


# ── 메인 백테스트 루프 ──

def get_trading_dates(mode, etfs):
    """모드별 거래일 산출"""
    if mode == "A":
        idx = etfs["hyundai"].index
    elif mode == "B":
        idx = etfs["samsung"].index
    else:
        idx = etfs["hyundai"].index.intersection(etfs["samsung"].index)
    return idx[(idx >= START_DATE) & (idx <= END_DATE)]


def run_backtest(mode, groups, etfs, ewy, stock_indicators):
    """백테스트 실행

    mode A: 현대차 단독
    mode B: 삼성 단독
    mode C: 통합 (L2: 그룹 선택)
    mode D: 3층 (L1: EWY + L2 + L3)
    """
    # 활성 그룹
    if mode == "A":
        active_groups = ["hyundai"]
    elif mode == "B":
        active_groups = ["samsung"]
    else:
        active_groups = ["hyundai", "samsung"]

    # 멤버 목록 구축
    all_members = []
    for gname in active_groups:
        for m in groups[gname]["members"]:
            all_members.append({**m, "group": gname})

    name_map = {m["ticker"]: m["name"] for m in all_members}

    # 거래일
    dates = get_trading_dates(mode, etfs)

    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    trades = []
    daily_results = []
    regime_log = {"active": 0, "paused": 0, "ewy_block": 0}

    last_group_z = {}

    for date in dates:
        # ETF 상태 조회
        etf_states = {}
        for gname in active_groups:
            etf_states[gname] = get_etf_state(etfs[gname], date)

        ewy_state = get_etf_state(ewy, date) if mode == "D" else None

        # Level 1: EWY 필터 (mode D only)
        ewy_ok = True
        if mode == "D" and ewy_state:
            ewy_ok = ewy_state["close"] > ewy_state["ma20"]
            if not ewy_ok:
                regime_log["ewy_block"] += 1

        # 그룹별 Z-score
        group_z = {}
        for gname in active_groups:
            members_g = [m for m in all_members if m["group"] == gname]
            z = calc_group_z_scores(stock_indicators, etfs[gname], date, members_g)
            if z:
                group_z[gname] = z

        if group_z:
            last_group_z = group_z

        # 레짐 추적
        any_active = any(
            etf_states.get(g) and etf_states[g]["close"] > etf_states[g]["ma20"]
            for g in active_groups
        )
        if any_active:
            regime_log["active"] += 1
        else:
            regime_log["paused"] += 1

        # ── 1. 청산 체크 ──
        closed = []
        for pos in positions:
            pos.days_held += 1
            gname = pos.group
            etf_st = etf_states.get(gname)
            z_data = group_z.get(gname, {}).get(pos.ticker)
            stock_close = z_data["close"] if z_data else pos.buy_price

            exit_reason, sell_price = check_exit(
                pos, z_data, stock_close, etf_st, ewy_state, mode
            )

            if exit_reason:
                sell_price = sell_price * (1 - SLIPPAGE)
                proceeds = pos.shares * sell_price * (1 - COMMISSION - TAX)
                pnl_pct = (sell_price / pos.buy_price - 1) - COMMISSION * 2 - TAX
                cash += proceeds

                trades.append({
                    "ticker": pos.ticker,
                    "name": name_map.get(pos.ticker, pos.ticker),
                    "group": gname,
                    "buy_date": pos.buy_date,
                    "sell_date": date,
                    "buy_price": pos.buy_price,
                    "sell_price": sell_price,
                    "pnl_pct": pnl_pct,
                    "exit": exit_reason,
                    "days": pos.days_held,
                    "z20_entry": pos.z20_entry,
                    "z20_exit": z_data["z_20"] if z_data else np.nan,
                    "allocated": pos.allocated,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── 2. 신규 진입 ──
        if len(positions) < MAX_POSITIONS and ewy_ok:
            held_tickers = {p.ticker for p in positions}
            held_groups = defaultdict(int)
            for p in positions:
                held_groups[p.group] += 1

            # 진입 후보 수집
            candidates = []
            for gname in active_groups:
                etf_st = etf_states.get(gname)
                if not etf_st:
                    continue

                # Mode C/D: 그룹당 최대 보유 체크
                if mode in ("C", "D") and held_groups[gname] >= MAX_PER_GROUP:
                    continue

                z = group_z.get(gname, {})
                for ticker, data in z.items():
                    if ticker in held_tickers:
                        continue
                    ok, reason = check_entry(data, etf_st)
                    if ok:
                        candidates.append((ticker, data, gname))

            # Level 2 정렬: 눌림그룹 우선
            if mode in ("C", "D") and len(active_groups) > 1:
                # 그룹 상대강도 계산
                group_rels = {}
                for gname in active_groups:
                    etf_st = etf_states.get(gname)
                    if etf_st and not pd.isna(etf_st["ret_20"]):
                        if mode == "D" and ewy_state and not pd.isna(ewy_state["ret_20"]):
                            group_rels[gname] = etf_st["ret_20"] - ewy_state["ret_20"]
                        else:
                            group_rels[gname] = etf_st["ret_20"]

                if len(group_rels) >= 2:
                    min_rel = min(group_rels.values())
                    max_rel = max(group_rels.values())
                    spread = max_rel - min_rel if max_rel != min_rel else 1.0

                    def sort_key(item):
                        _, d, g = item
                        # 덜 눌린 그룹에 패널티 → 눌린 그룹 우선
                        penalty = (group_rels.get(g, 0) - min_rel) / spread * GROUP_PENALTY_MAX
                        return d["z_20"] + penalty

                    candidates.sort(key=sort_key)
                else:
                    candidates.sort(key=lambda x: x[1]["z_20"])
            else:
                candidates.sort(key=lambda x: x[1]["z_20"])

            # 진입 실행
            for ticker, data, gname in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if mode in ("C", "D"):
                    grp_count = sum(1 for p in positions if p.group == gname)
                    if grp_count >= MAX_PER_GROUP:
                        continue

                alloc = INITIAL_CAPITAL / MAX_POSITIONS * WEIGHT_MULT.get(ticker, 1.0)
                alloc = min(alloc, cash)
                if alloc < 100_000:
                    continue

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
                    group=gname,
                    buy_date=date,
                    buy_price=buy_price,
                    shares=shares,
                    allocated=actual_cost,
                    z20_entry=data["z_20"],
                ))

        # ── 3. 일일 마감 ──
        pos_value = 0
        for pos in positions:
            z_data = group_z.get(pos.group, {}).get(pos.ticker)
            if z_data:
                pos_value += pos.shares * z_data["close"]
            else:
                pos_value += pos.allocated
        equity = cash + pos_value
        daily_results.append({"date": date, "equity": equity, "positions": len(positions)})

    # 미청산 강제 청산
    for pos in positions:
        z_data = last_group_z.get(pos.group, {}).get(pos.ticker)
        sell_price = z_data["close"] if z_data else pos.buy_price
        pnl_pct = (sell_price / pos.buy_price - 1) - COMMISSION * 2 - TAX
        trades.append({
            "ticker": pos.ticker,
            "name": name_map.get(pos.ticker, pos.ticker),
            "group": pos.group,
            "buy_date": pos.buy_date,
            "sell_date": dates[-1] if len(dates) > 0 else pos.buy_date,
            "buy_price": pos.buy_price,
            "sell_price": sell_price,
            "pnl_pct": pnl_pct,
            "exit": "force_close",
            "days": pos.days_held,
            "z20_entry": pos.z20_entry,
            "z20_exit": z_data["z_20"] if z_data else np.nan,
            "allocated": pos.allocated,
        })

    return trades, daily_results, regime_log


# ── 리포트 ──

def report(trades, daily_results, regime_log, label, groups, active_group_names):
    """결과 리포트 출력 + 통계 반환"""
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

    # MDD
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

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  초기: {INITIAL_CAPITAL / 1e4:,.0f}만원 → 최종: {final_equity / 1e4:,.0f}만원 ({total_return:+.1f}%)")
    print(f"  거래: {len(trades)}건 | 승률: {win_rate:.1f}% ({len(wins)}승/{len(losses)}패)")
    print(f"  평균 수익: +{avg_win:.2f}% | 평균 손실: {avg_loss:.2f}%")
    print(f"  Profit Factor: {pf:.2f} | MDD: {mdd:.1f}%")
    print(f"  평균 보유: {np.mean([t['days'] for t in trades]):.1f}일")

    # 청산 분류
    exit_parts = []
    for key in ["z_recovery", "stop_loss", "time_stop", "etf_collapse", "ewy_collapse", "max_time", "force_close"]:
        if exits.get(key, 0) > 0:
            exit_parts.append(f"{key}:{exits[key]}")
    print(f"  청산: {', '.join(exit_parts)}")

    # 레짐
    active = regime_log["active"]
    paused = regime_log["paused"]
    ewy_block = regime_log.get("ewy_block", 0)
    total_d = active + paused
    if total_d > 0:
        print(f"  레짐: 활성 {active}일({active/total_d*100:.0f}%) "
              f"중지 {paused}일({paused/total_d*100:.0f}%)"
              + (f" EWY차단 {ewy_block}일" if ewy_block else ""))

    # 그룹별 분석 (C/D 모드)
    if len(active_group_names) > 1:
        print(f"\n  그룹별 성과:")
        for gname in active_group_names:
            g_trades = [t for t in trades if t["group"] == gname]
            if not g_trades:
                print(f"    {gname}: 거래 없음")
                continue
            g_pnls = [t["pnl_pct"] for t in g_trades]
            g_wins = [p for p in g_pnls if p > 0]
            g_wr = len(g_wins) / len(g_pnls) * 100
            g_pf = (sum(max(p, 0) for p in g_pnls) /
                    abs(sum(min(p, 0) for p in g_pnls)))  \
                if any(p < 0 for p in g_pnls) else float("inf")
            print(f"    {gname:>10}: {len(g_trades)}건 승률{g_wr:.0f}% "
                  f"PF{g_pf:.2f} 순PnL{sum(g_pnls)*100:+.1f}%")

    # 종목별 분석
    print(f"\n  종목별 성과:")
    name_map = {}
    for gname in active_group_names:
        for m in groups[gname]["members"]:
            name_map[m["ticker"]] = m["name"]

    for gname in active_group_names:
        for m in groups[gname]["members"]:
            ticker = m["ticker"]
            t_trades = [t for t in trades if t["ticker"] == ticker]
            if not t_trades:
                print(f"    {name_map[ticker]:>10}: 거래 없음")
                continue
            t_pnls = [t["pnl_pct"] for t in t_trades]
            t_wins = [p for p in t_pnls if p > 0]
            t_avg_z = np.mean([t["z20_entry"] for t in t_trades])
            print(f"    {name_map[ticker]:>10}: {len(t_trades)}건 "
                  f"승률{len(t_wins)/len(t_trades)*100:.0f}% "
                  f"z진입{t_avg_z:.2f} "
                  f"순PnL{sum(t_pnls)*100:+.1f}%")

    # 월별 수익
    monthly = defaultdict(float)
    for t in trades:
        key = t["sell_date"].strftime("%Y-%m")
        monthly[key] += t["pnl_pct"] * 100

    print(f"\n  월별 수익률:")
    for m_key in sorted(monthly.keys()):
        bar_len = int(abs(monthly[m_key]) / 1.5)
        bar = "+" * bar_len if monthly[m_key] > 0 else "-" * bar_len
        print(f"    {m_key}: {monthly[m_key]:+6.1f}% {bar}")

    # 최대/최소 거래
    best = max(trades, key=lambda t: t["pnl_pct"])
    worst = min(trades, key=lambda t: t["pnl_pct"])
    print(f"\n  최대 수익: {best['name']}({best['group']}) "
          f"{best['pnl_pct']*100:+.1f}% ({best['exit']}, {best['days']}일)")
    print(f"  최대 손실: {worst['name']}({worst['group']}) "
          f"{worst['pnl_pct']*100:+.1f}% ({worst['exit']}, {worst['days']}일)")

    return {
        "label": label,
        "trades": len(trades),
        "win_rate": win_rate,
        "pf": pf,
        "total_return": total_return,
        "mdd": mdd,
        "avg_days": np.mean([t["days"] for t in trades]),
        "final_equity": final_equity,
    }


# ── 메인 ──

def main():
    print("=" * 62)
    print("  3층 순환매 백테스트 — 현대차+삼성 그룹 + EWY")
    print("=" * 62)
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  자본: {INITIAL_CAPITAL/1e4:,.0f}만원 | 최대 {MAX_POSITIONS}종목 | 슬리피지 {SLIPPAGE*100}%")

    print("\n데이터 로딩...")
    groups, etfs, ewy, stocks = load_all_data()
    for gname, gdata in groups.items():
        names = [m["name"] for m in gdata["members"]]
        print(f"  {gname}: {', '.join(names)} ({len(names)}종목)")
    print(f"  EWY: {len(ewy)}행")

    print("\n지표 계산...")
    stock_indicators = prepare_indicators(groups, etfs, ewy, stocks)
    print(f"  {len(stock_indicators)}종목 지표 완료")

    modes = [
        ("A", "A) 현대차그룹 단독 (3종목)", ["hyundai"]),
        ("B", "B) 삼성그룹 단독 (4종목)", ["samsung"]),
        ("C", "C) 통합 L2:그룹선택+L3:종목선택 (7종목)", ["hyundai", "samsung"]),
        ("D", "D) 3층완성 L1:EWY+L2+L3 (7종목)", ["hyundai", "samsung"]),
    ]

    results = []
    for mode_code, label, active_groups in modes:
        print(f"\n[{mode_code}] 실행 중...")
        t, d, r = run_backtest(mode_code, groups, etfs, ewy, stock_indicators)
        stat = report(t, d, r, label, groups, active_groups)
        if stat:
            results.append(stat)

    # ── 종합 비교 ──
    if len(results) >= 2:
        print(f"\n{'=' * 78}")
        print(f"  종합 비교")
        print(f"{'=' * 78}")
        print(f"  {'모드':<42} {'거래':>5} {'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>7}")
        print(f"  {'-' * 72}")
        for r in results:
            print(f"  {r['label']:<42} {r['trades']:>5} "
                  f"{r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['total_return']:>+7.1f}% {r['mdd']:>6.1f}%")

        # 목표 달성 체크 (D모드 기준)
        d_result = next((r for r in results if r["label"].startswith("D)")), None)
        if d_result:
            print(f"\n  D모드 목표 체크:")
            checks = [
                ("거래 20건+", d_result["trades"] >= 20),
                ("PF 1.5+", d_result["pf"] >= 1.5),
                ("MDD -7% 이내", d_result["mdd"] >= -7.0),
            ]
            all_pass = True
            for name, passed in checks:
                mark = "O" if passed else "X"
                print(f"    [{mark}] {name}")
                if not passed:
                    all_pass = False

            if all_pass:
                print(f"\n  >>> D모드 채택 가능! <<<")
            else:
                print(f"\n  >>> 파라미터 조정 필요 <<<")

        # v10.1 참고
        print(f"\n  참고: v10.1 C_new모드 (84종목, KOSPI레짐)")
        print(f"    거래 82건, 승률 50.0%, PF 1.78, +21.7%, MDD -4.5%")
        print(f"    → 그룹순환(40%) + v10.1(60%) 조합 시 분산 효과 기대")


if __name__ == "__main__":
    main()
