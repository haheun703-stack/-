#!/usr/bin/env python
"""서킷브레이커 프로토콜 백테스트.

KOSPI -5% 이상 폭락 이벤트 감지 → 첫 반등일 진입 → 5일 보유.
3-way 비교: A(건전주) vs B(무조건) vs C(수급확인).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# ── 상수 ──────────────────────────────────────────────
SLIPPAGE = 0.005
COMMISSION = 0.00015
TAX = 0.0018
CB_THRESHOLD = -5.0       # KOSPI 일간 -5% 이상 = CB 이벤트
CLUSTER_GAP = 5            # 5거래일 이내 연속 폭락 = 같은 클러스터
NUM_PICKS = 5              # CB 모드에서 매수할 종목 수
HOLD_DAYS = 5              # 보유 기간
ALLOC_PER_SLOT = 0.125     # 슬롯당 12.5% (2슬롯 = 25%)
STOP_LOSS = -0.05          # CB 전용 손절 -5%
INITIAL_CAPITAL = 100_000_000


@dataclass
class CBEvent:
    crash_dates: list          # 폭락일 리스트
    first_rebound: pd.Timestamp  # 클러스터 이후 첫 양봉일
    pre_crash_date: pd.Timestamp  # 첫 폭락 D-5 (건전성 판단)
    kospi_drop: float          # 최대 낙폭 (%)
    label: str = ""


@dataclass
class CBTrade:
    ticker: str
    name: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str           # timeout / sma20_recovery / stop_loss
    pnl_pct: float             # 순수익률 (수수료+세금 반영)
    crash_depth: float         # 폭락 폭 (%)
    pre_counter: int           # 폭락 전 counter
    pre_freshness: float       # 폭락 전 freshness
    variant: str               # A / B / C


# ── 데이터 로드 ───────────────────────────────────────

def load_parquets():
    pq_dir = Path("data/processed")
    data = {}
    name_map = {}

    csv_dir = Path("stock_data_daily")
    for f in csv_dir.glob("*.csv"):
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]

    for f in sorted(pq_dir.glob("*.parquet")):
        ticker = f.stem
        if len(ticker) == 6 and ticker.isdigit():
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
            if len(df) > 252:
                data[ticker] = df
    return data, name_map


def load_kospi():
    df = pd.read_csv("data/kospi_index.csv", index_col="Date", parse_dates=True)
    df["ret"] = df["close"].pct_change() * 100
    return df


# ── CB 이벤트 감지 ────────────────────────────────────

def detect_cb_events(kospi_df):
    """KOSPI -5% 이상 낙폭 이벤트 클러스터 감지."""
    crashes = kospi_df[kospi_df["ret"] <= CB_THRESHOLD].copy()
    if crashes.empty:
        return []

    # 클러스터링: CLUSTER_GAP 거래일 이내 = 같은 이벤트
    events = []
    used = set()

    for date in crashes.index:
        if date in used:
            continue
        cluster = [date]
        used.add(date)

        for d2 in crashes.index:
            if d2 != date and d2 not in used:
                # 거래일 기준 gap 계산
                idx1 = kospi_df.index.get_loc(date)
                idx2 = kospi_df.index.get_loc(d2)
                if 0 < idx2 - idx1 <= CLUSTER_GAP:
                    cluster.append(d2)
                    used.add(d2)

        cluster.sort()

        # 첫 폭락 D-5
        first_crash_idx = kospi_df.index.get_loc(cluster[0])
        pre_crash_idx = max(0, first_crash_idx - 5)
        pre_crash_date = kospi_df.index[pre_crash_idx]

        # 마지막 폭락 이후 첫 양봉일
        last_crash_idx = kospi_df.index.get_loc(cluster[-1])
        first_rebound = None
        for i in range(last_crash_idx + 1, len(kospi_df)):
            if kospi_df.iloc[i]["ret"] > 0:
                first_rebound = kospi_df.index[i]
                break

        if first_rebound is None:
            continue  # 반등 없으면 스킵

        # 최대 낙폭
        max_drop = min(kospi_df.loc[cluster, "ret"])

        event = CBEvent(
            crash_dates=cluster,
            first_rebound=first_rebound,
            pre_crash_date=pre_crash_date,
            kospi_drop=round(max_drop, 1),
            label=f"{cluster[0].strftime('%Y-%m-%d')} ({len(cluster)}일)",
        )
        events.append(event)

    return events


# ── Freshness 계산 ────────────────────────────────────

def calc_freshness(counter, rsi, vol_c):
    if counter <= 0 or counter >= 16:
        return 0.0
    if vol_c > 2.0:
        return 0.0
    if counter >= 8 and rsi > 65:
        return 0.0
    c = 1.15 if counter <= 3 else (1.00 if counter <= 7 else 0.70)
    r = 1.00 if rsi <= 55 else (0.90 if rsi <= 65 else 0.75)
    v = 1.05 if vol_c < 0.7 else (1.00 if vol_c <= 1.3 else 0.85)
    return round(c * r * v, 3)


# ── 종목 선정 ─────────────────────────────────────────

def select_cb_candidates(data_dict, name_map, event, mode="healthy"):
    """CB 이벤트에 대해 종목 후보 선정.

    mode:
      healthy  - 폭락 전 SMA20↑ + counter 1~10 + freshness > 0
      random   - 거래량 상위 (시총 프록시)
      supply   - healthy + 반등일 수급 양호
    """
    pre_date = event.pre_crash_date
    entry_date = event.first_rebound
    candidates = []

    for ticker, df in data_dict.items():
        # pre_crash_date 데이터
        pre_mask = df.index <= pre_date
        if pre_mask.sum() < 5:
            continue
        pre_idx = df.index[pre_mask][-1]
        row_pre = df.loc[pre_idx]

        # entry_date 데이터
        entry_mask = df.index == entry_date
        if not entry_mask.any():
            # entry_date가 정확히 없으면 가장 가까운 이후 날짜
            future = df.index[df.index >= entry_date]
            if len(future) == 0:
                continue
            entry_idx = future[0]
        else:
            entry_idx = entry_date
        row_entry = df.loc[entry_idx]

        entry_price = float(row_entry["close"])
        pre_price = float(row_pre["close"])
        crash_depth = (entry_price - pre_price) / pre_price * 100

        # 공통 데이터
        pre_counter = int(row_pre.get("days_above_sma20", 0) or 0)
        pre_rsi = float(row_pre.get("rsi_14", 50) or 50)
        pre_vol = float(row_pre.get("volume", 0) or 0)
        pre_vol_ma20 = float(row_pre.get("volume_ma20", 1) or 1)
        pre_vol_c = pre_vol / pre_vol_ma20 if pre_vol_ma20 > 0 else 1
        pre_sma20 = float(row_pre.get("sma_20", 0) or 0)
        pre_close = float(row_pre["close"])
        pre_above_sma20 = pre_close > pre_sma20 if pre_sma20 > 0 else False
        pre_fresh = calc_freshness(pre_counter, pre_rsi, pre_vol_c)

        # 반등일 수급
        entry_inst = float(row_entry.get("inst_net_5d", 0) or 0)
        entry_foreign = float(row_entry.get("foreign_net_5d", 0) or 0)
        entry_smart = entry_inst + entry_foreign
        entry_vol = float(row_entry.get("volume", 0) or 0)

        rec = {
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "entry_price": entry_price,
            "entry_date": entry_idx,
            "crash_depth": round(crash_depth, 2),
            "pre_counter": pre_counter,
            "pre_rsi": round(pre_rsi, 1),
            "pre_freshness": pre_fresh,
            "pre_above_sma20": pre_above_sma20,
            "entry_smart": entry_smart,
            "entry_vol": entry_vol,
        }

        if mode == "healthy":
            # 폭락 전 SMA20↑ + counter 1~10 + freshness > 0
            if not pre_above_sma20:
                continue
            if pre_counter < 1 or pre_counter > 10:
                continue
            if pre_fresh <= 0:
                continue
            # 스코어: 폭락 폭 × 0.6 + freshness × 0.4
            rec["cb_score"] = abs(crash_depth) * 0.6 + pre_fresh * 0.4
            candidates.append(rec)

        elif mode == "random":
            # 거래량 기준 랭킹 (시총 프록시)
            rec["cb_score"] = entry_vol
            candidates.append(rec)

        elif mode == "supply":
            # healthy 조건 + 수급 양호
            if not pre_above_sma20:
                continue
            if pre_counter < 1 or pre_counter > 10:
                continue
            if pre_fresh <= 0:
                continue
            if entry_smart <= 0:
                continue
            rec["cb_score"] = abs(crash_depth) * 0.6 + pre_fresh * 0.4
            candidates.append(rec)

    # 스코어 내림차순 정렬
    candidates.sort(key=lambda x: x["cb_score"], reverse=True)
    return candidates


# ── 거래 시뮬레이션 ───────────────────────────────────

def simulate_cb_trades(candidates, data_dict, event, variant_name):
    """선정된 후보 종목으로 CB 거래 시뮬레이션."""
    trades = []

    for cand in candidates[:NUM_PICKS]:
        ticker = cand["ticker"]
        df = data_dict[ticker]
        entry_date = cand["entry_date"]

        # entry_date의 인덱스 위치
        if entry_date not in df.index:
            continue
        entry_idx = df.index.get_loc(entry_date)

        entry_price = cand["entry_price"]
        buy_cost = entry_price * (1 + SLIPPAGE + COMMISSION)

        # 보유 기간 동안 청산 조건 체크
        exit_date = None
        exit_price = None
        exit_reason = None

        for d in range(1, HOLD_DAYS + 1):
            if entry_idx + d >= len(df):
                break

            row = df.iloc[entry_idx + d]
            close_d = float(row["close"])
            low_d = float(row["low"])
            high_d = float(row["high"])
            sma20 = float(row.get("sma_20", 0) or 0)

            # 1. 손절 체크 (장중 저가 기준)
            stop_price = buy_cost * (1 + STOP_LOSS)
            if low_d <= stop_price:
                exit_date = df.index[entry_idx + d]
                exit_price = stop_price
                exit_reason = "stop_loss"
                break

            # 2. SMA20 회복 (종가 기준)
            if sma20 > 0 and close_d > sma20:
                exit_date = df.index[entry_idx + d]
                exit_price = close_d
                exit_reason = "sma20_recovery"
                break

            # 3. 시간 제한
            if d == HOLD_DAYS:
                exit_date = df.index[entry_idx + d]
                exit_price = close_d
                exit_reason = "timeout"
                break

        if exit_date is None:
            # 데이터 부족 — 마지막 날 종가로 청산
            last_idx = min(entry_idx + HOLD_DAYS, len(df) - 1)
            exit_date = df.index[last_idx]
            exit_price = float(df.iloc[last_idx]["close"])
            exit_reason = "data_end"

        # 순수익률 계산
        sell_proceeds = exit_price * (1 - SLIPPAGE - COMMISSION - TAX)
        pnl_pct = (sell_proceeds - buy_cost) / buy_cost * 100

        trade = CBTrade(
            ticker=ticker,
            name=cand["name"],
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_pct=round(pnl_pct, 2),
            crash_depth=cand["crash_depth"],
            pre_counter=cand["pre_counter"],
            pre_freshness=cand["pre_freshness"],
            variant=variant_name,
        )
        trades.append(trade)

    return trades


# ── 결과 출력 ─────────────────────────────────────────

def print_event_results(event, all_variant_trades):
    crash_str = ", ".join(d.strftime("%Y-%m-%d") for d in event.crash_dates)
    print(f"\n{'='*70}")
    print(f"[이벤트] {event.label}")
    print(f"  폭락일: {crash_str}")
    print(f"  KOSPI 최대 낙폭: {event.kospi_drop:+.1f}%")
    print(f"  진입일: {event.first_rebound.strftime('%Y-%m-%d')}")
    print(f"  D-5 기준: {event.pre_crash_date.strftime('%Y-%m-%d')}")

    for variant_name, trades in all_variant_trades:
        if not trades:
            print(f"\n  [{variant_name}] 후보 없음")
            continue

        pnls = [t.pnl_pct for t in trades]
        avg_pnl = np.mean(pnls)
        wins = sum(1 for p in pnls if p > 0)

        print(f"\n  [{variant_name}] {len(trades)}종목, "
              f"평균:{avg_pnl:+.2f}%, 승률:{wins}/{len(trades)}")

        for t in trades:
            exit_d = (t.exit_date - t.entry_date).days
            print(f"    {t.name:>12s} {t.pnl_pct:+6.2f}% "
                  f"(폭락:{t.crash_depth:+.0f}%, c:{t.pre_counter}, "
                  f"fresh:{t.pre_freshness:.2f}, "
                  f"청산:{t.exit_reason}, {exit_d}일)")


def print_summary(all_trades):
    print(f"\n{'='*70}")
    print("=== 전체 합산 ===")
    print(f"{'Variant':>12s} {'건수':>4s} {'평균수익':>8s} {'승률':>6s} "
          f"{'총수익(만)':>10s} {'최악':>7s}")
    print("-" * 55)

    for variant in ["A_건전주", "B_무조건", "C_수급확인"]:
        trades = [t for t in all_trades if t.variant == variant]
        if not trades:
            print(f"{variant:>12s}  후보 없음")
            continue

        pnls = [t.pnl_pct for t in trades]
        avg = np.mean(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) * 100

        # 총수익 (1억 기준, 슬롯당 12.5%)
        total_profit = sum(
            INITIAL_CAPITAL * ALLOC_PER_SLOT * p / 100 for p in pnls
        )
        worst = min(pnls)

        print(f"{variant:>12s} {len(trades):>4d} {avg:>+7.2f}% "
              f"{win_rate:>5.1f}% {total_profit/10000:>+10,.0f} "
              f"{worst:>+6.2f}%")

    # 판정
    print()
    for variant in ["A_건전주", "C_수급확인"]:
        trades = [t for t in all_trades if t.variant == variant]
        if not trades:
            continue
        pnls = [t.pnl_pct for t in trades]
        bench_trades = [t for t in all_trades if t.variant == "B_무조건"]
        bench_pnls = [t.pnl_pct for t in bench_trades] if bench_trades else [0]

        c1 = sum(1 for p in pnls if p > 0) / len(pnls) > 0.60
        c2 = np.mean(pnls) > 2.0
        c3 = min(pnls) > -5.0
        c4 = np.mean(pnls) > np.mean(bench_pnls)

        passed = sum([c1, c2, c3, c4])
        verdict = "PASS" if passed >= 3 else "FAIL"

        print(f"  [{variant}] 판정: {verdict} ({passed}/4)")
        print(f"    승률>60%: {'O' if c1 else 'X'} ({sum(1 for p in pnls if p > 0)}/{len(pnls)})")
        print(f"    평균>+2%: {'O' if c2 else 'X'} ({np.mean(pnls):+.2f}%)")
        print(f"    최악>-5%: {'O' if c3 else 'X'} ({min(pnls):+.2f}%)")
        print(f"    vs B_무조건: {'O' if c4 else 'X'} "
              f"({np.mean(pnls):+.2f}% vs {np.mean(bench_pnls):+.2f}%)")


# ── 메인 ──────────────────────────────────────────────

def main():
    print("=== 서킷브레이커 프로토콜 백테스트 ===")
    print(f"CB 기준: KOSPI 일간 {CB_THRESHOLD}% 이하")
    print(f"진입: 첫 반등일 종가 / 청산: {HOLD_DAYS}일 or SMA20 회복 or {STOP_LOSS*100:.0f}% 손절")
    print(f"슬리피지: {SLIPPAGE*100:.1f}% / 수수료+세금: {(COMMISSION*2+TAX)*100:.3f}%")

    data_dict, name_map = load_parquets()
    kospi_df = load_kospi()

    print(f"\n종목: {len(data_dict)}개")
    print(f"KOSPI: {kospi_df.index[0].strftime('%Y-%m-%d')} ~ "
          f"{kospi_df.index[-1].strftime('%Y-%m-%d')}")

    events = detect_cb_events(kospi_df)
    print(f"\nCB 이벤트: {len(events)}건")

    all_trades = []

    for event in events:
        # 3-way 종목 선정
        cands_a = select_cb_candidates(data_dict, name_map, event, "healthy")
        cands_b = select_cb_candidates(data_dict, name_map, event, "random")
        cands_c = select_cb_candidates(data_dict, name_map, event, "supply")

        # 거래 시뮬레이션
        trades_a = simulate_cb_trades(cands_a, data_dict, event, "A_건전주")
        trades_b = simulate_cb_trades(cands_b, data_dict, event, "B_무조건")
        trades_c = simulate_cb_trades(cands_c, data_dict, event, "C_수급확인")

        # 이벤트별 결과 출력
        print_event_results(event, [
            ("A_건전주", trades_a),
            ("B_무조건", trades_b),
            ("C_수급확인", trades_c),
        ])

        all_trades.extend(trades_a + trades_b + trades_c)

        # 후보 수 요약
        print(f"\n  후보 수: A={len(cands_a)}, B={len(cands_b)}, C={len(cands_c)}")

    # 전체 합산
    print_summary(all_trades)


if __name__ == "__main__":
    main()
