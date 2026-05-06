"""바톤터치 백테스트 — 연기금 선행 매집 → 외인 후속 진입 패턴

시나리오:
  1) 연기금 N일 연속 순매수 (매집 중)
  2) 외인이 당일 대량 진입 (바톤터치 발생)
  3) D+1 양봉 확인 → 진입
  매도: 외인+기관 3일 연속 매도 OR 손절 -10%

비교:
  A) 연기금 3일+ → 외인 50억+ 신규  (순차)
  B) 연기금 3일+ → 외인 100억+ 신규 (강한 순차)
  C) 연기금 5일+ → 외인 50억+ 신규  (장기 매집 후)
  D) 연기금 3일+ → 쌍끌이(외인+기관 동시) (릴레이+동반)
  E) 외인 3일+ → 연기금 신규 진입    (역방향)
  F) 기준선: 연기금 단독 (비교용)
  G) 기준선: 외인 단독 (비교용)
"""
import os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/investor_flow/investor_daily.db"
CSV_DIR = Path("stock_data_daily")


def load_all_data():
    """수급 + 가격 전체 로딩."""
    conn = sqlite3.connect(DB_PATH)
    supply = pd.read_sql_query(
        """
        SELECT date, ticker,
               SUM(CASE WHEN investor = '외국인'   THEN net_val ELSE 0 END) / 1e8 AS fgn,
               SUM(CASE WHEN investor = '기관합계' THEN net_val ELSE 0 END) / 1e8 AS inst,
               SUM(CASE WHEN investor = '연기금'   THEN net_val ELSE 0 END) / 1e8 AS pension,
               SUM(CASE WHEN investor = '개인'     THEN net_val ELSE 0 END) / 1e8 AS retail
        FROM investor_daily
        GROUP BY date, ticker
        """,
        conn,
    )
    conn.close()
    print(f"수급: {len(supply):,}행 / {supply['ticker'].nunique()}종목")

    # 가격 로딩
    prices = {}
    for f in CSV_DIR.glob("*.csv"):
        parts = f.stem.split("_")
        ticker = parts[-1] if len(parts) >= 2 else parts[0]
        if len(ticker) != 6 or not ticker.isdigit():
            continue
        try:
            df = pd.read_csv(f, header=0)
            if len(df.columns) < 6 or len(df) < 60:
                continue
            df = df.iloc[:, :6]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = df["date"].astype(str).str.replace("-", "")
            df = df.sort_values("date").reset_index(drop=True)
            prices[ticker] = df.set_index("date")
        except Exception:
            continue
    print(f"가격: {len(prices)}종목")

    trading_days = sorted(supply["date"].unique())
    print(f"거래일: {len(trading_days)}일 ({trading_days[0]}~{trading_days[-1]})")
    return supply, prices, trading_days


def detect_baton_touch(supply_df, trading_days, conditions):
    """바톤터치 시그널 감지.

    conditions: dict with keys:
      pension_days: 연기금 연속 순매수 최소 일수
      fgn_min: 외인 당일 최소 순매수 (억)
      fgn_prev_neg: 외인이 이전에 매수 안 했어야 하는지 (True/False)
      inst_min: 기관 당일 최소 (optional, 0이면 무시)
      reverse: True면 역방향 (외인→연기금)
    """
    pension_days = conditions.get("pension_days", 3)
    fgn_min = conditions.get("fgn_min", 50)
    fgn_prev_neg = conditions.get("fgn_prev_neg", False)
    inst_min = conditions.get("inst_min", 0)
    reverse = conditions.get("reverse", False)

    day_map = {d: i for i, d in enumerate(trading_days)}
    signals = []

    tickers = supply_df["ticker"].unique()
    supply_grouped = {t: g.sort_values("date") for t, g in supply_df.groupby("ticker")}

    for ticker, tdf in supply_grouped.items():
        tdf = tdf.set_index("date")
        dates = sorted(tdf.index)

        for di, date in enumerate(dates):
            if di < pension_days:
                continue

            row = tdf.loc[date]

            if not reverse:
                # 정방향: 연기금 N일 선행 → 외인 당일 진입
                # 연기금 N일 연속 순매수 체크
                prev_dates = dates[max(0, di - pension_days):di]
                pension_streak = all(
                    tdf.loc[d, "pension"] > 0 for d in prev_dates if d in tdf.index
                )
                if not pension_streak:
                    continue

                # 외인 당일 대량 진입
                if row["fgn"] < fgn_min:
                    continue

                # 외인이 이전에 매수 안 했어야 하는 조건
                if fgn_prev_neg:
                    prev_fgn_avg = np.mean([tdf.loc[d, "fgn"] for d in prev_dates if d in tdf.index])
                    if prev_fgn_avg > 10:  # 이미 매수 중이었으면 skip
                        continue

                # 기관 동반 조건
                if inst_min > 0 and row["inst"] < inst_min:
                    continue

            else:
                # 역방향: 외인 N일 선행 → 연기금 당일 진입
                prev_dates = dates[max(0, di - pension_days):di]
                fgn_streak = all(
                    tdf.loc[d, "fgn"] > 0 for d in prev_dates if d in tdf.index
                )
                if not fgn_streak:
                    continue
                if row["pension"] < fgn_min:  # 여기서 fgn_min을 연기금 최소로 재활용
                    continue

            signals.append({
                "ticker": ticker,
                "date": date,
                "fgn": round(float(row["fgn"]), 1),
                "inst": round(float(row["inst"]), 1),
                "pension": round(float(row["pension"]), 1),
            })

    return signals


def simulate_trades(signals, prices, trading_days):
    """시그널 → D+1 양봉 진입 → 수급이탈/손절 매도."""
    day_map = {d: i for i, d in enumerate(trading_days)}
    trades = []

    # 수급 데이터 다시 로딩 (매도 판단용)
    conn = sqlite3.connect(DB_PATH)
    supply_all = pd.read_sql_query(
        """
        SELECT date, ticker,
               SUM(CASE WHEN investor = '외국인'   THEN net_val ELSE 0 END) / 1e8 AS fgn,
               SUM(CASE WHEN investor = '기관합계' THEN net_val ELSE 0 END) / 1e8 AS inst
        FROM investor_daily
        GROUP BY date, ticker
        """,
        conn,
    )
    conn.close()
    supply_dict = {}
    for _, r in supply_all.iterrows():
        key = (r["date"], r["ticker"])
        supply_dict[key] = {"fgn": r["fgn"], "inst": r["inst"]}

    for sig in signals:
        ticker = sig["ticker"]
        date = sig["date"]
        if ticker not in prices:
            continue
        pdf = prices[ticker]

        idx = day_map.get(date, -1)
        if idx < 0 or idx + 1 >= len(trading_days):
            continue

        # D+1 양봉 확인
        next_date = trading_days[idx + 1]
        if next_date not in pdf.index:
            continue
        nd = pdf.loc[next_date]
        if nd["close"] <= nd["open"]:
            continue  # 음봉 = 스킵

        entry_price = nd["close"]
        entry_date = next_date

        # 매도 시뮬레이션
        sell_streak = 0
        exit_date = ""
        exit_price = 0
        exit_reason = ""

        for j in range(idx + 2, min(idx + 60, len(trading_days))):
            d = trading_days[j]
            if d not in pdf.index:
                continue

            cur_close = pdf.loc[d, "close"]

            # 손절 -10%
            loss = (cur_close / entry_price - 1) * 100 if entry_price > 0 else 0
            if loss <= -10:
                exit_date = d
                exit_price = cur_close
                exit_reason = "손절-10%"
                break

            # 외인+기관 3일 연속 매도
            s = supply_dict.get((d, ticker), {"fgn": 0, "inst": 0})
            if s["fgn"] + s["inst"] < -10:
                sell_streak += 1
            else:
                sell_streak = 0

            if sell_streak >= 3:
                exit_date = d
                exit_price = cur_close
                exit_reason = "수급이탈3일"
                break

        if not exit_date:
            # 60일 내 매도 안 됨 → 마지막 날 강제 청산
            last_d = trading_days[min(idx + 59, len(trading_days) - 1)]
            if last_d in pdf.index:
                exit_date = last_d
                exit_price = pdf.loc[last_d, "close"]
                exit_reason = "만기청산"
            else:
                continue

        ret = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
        hold = day_map.get(exit_date, 0) - day_map.get(entry_date, 0)

        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "exit_reason": exit_reason,
            "return_pct": round(ret, 2),
            "hold_days": hold,
            "fgn": sig["fgn"],
            "pension": sig["pension"],
        })

    return trades


def print_results(name, trades):
    """결과 요약."""
    if not trades:
        print(f"  {name:30s}  시그널 0건")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = len(df[df["return_pct"] > 0])
    losses = n - wins
    avg_ret = df["return_pct"].mean()
    avg_win = df[df["return_pct"] > 0]["return_pct"].mean() if wins > 0 else 0
    avg_loss = df[df["return_pct"] <= 0]["return_pct"].mean() if losses > 0 else 0
    wr = wins / n * 100 if n > 0 else 0
    pf = abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 and avg_loss != 0 else 999
    avg_hold = df["hold_days"].mean()

    print(f"  {name:30s}  {n:>4}건  WR {wr:>5.1f}%  avg {avg_ret:>+6.2f}%  "
          f"PF {pf:>5.2f}  보유 {avg_hold:>4.1f}일  "
          f"이익 {avg_win:>+6.2f}% / 손실 {avg_loss:>+6.2f}%")
    return {"name": name, "n": n, "wr": wr, "avg_ret": avg_ret, "pf": pf,
            "avg_hold": avg_hold, "avg_win": avg_win, "avg_loss": avg_loss}


def main():
    supply, prices, trading_days = load_all_data()

    SEP = "=" * 110
    print(f"\n{SEP}")
    print(f"  바톤터치 백테스트 — 연기금 선행 → 외인 후속 패턴")
    print(f"  기간: {trading_days[0]} ~ {trading_days[-1]} ({len(trading_days)}거래일)")
    print(SEP)

    scenarios = [
        ("A) 연3일+펜션 → 외50억+신규", {"pension_days": 3, "fgn_min": 50, "fgn_prev_neg": True}),
        ("B) 연3일+펜션 → 외100억+신규", {"pension_days": 3, "fgn_min": 100, "fgn_prev_neg": True}),
        ("C) 연5일+펜션 → 외50억+신규", {"pension_days": 5, "fgn_min": 50, "fgn_prev_neg": True}),
        ("D) 연3일+펜션 → 쌍끌이(외50+기50)", {"pension_days": 3, "fgn_min": 50, "inst_min": 50}),
        ("E) 연3일+펜션 → 외50억+(기존매수OK)", {"pension_days": 3, "fgn_min": 50, "fgn_prev_neg": False}),
        ("F) 역: 외3일+ → 연기금20억+신규", {"pension_days": 3, "fgn_min": 20, "reverse": True}),
    ]

    print(f"\n  {'시나리오':30s}  {'건수':>4}  {'승률':>7}  {'평균수익':>8}  "
          f"{'PF':>6}  {'보유':>6}  {'이익/손실':>20}")
    print(f"  {'─' * 105}")

    results = []
    for name, conds in scenarios:
        sigs = detect_baton_touch(supply, trading_days, conds)
        trades = simulate_trades(sigs, prices, trading_days)
        r = print_results(name, trades)
        if r:
            results.append(r)

    # 비교 기준선
    print(f"\n  [비교 기준선 — 어제 백테스트 결과]")
    print(f"  {'연기금 단독 (E유형)':30s}    65건  WR  55.4%  avg  +9.43%  PF  4.96")
    print(f"  {'외인 단독 (D유형)':30s}   189건  WR  50.8%  avg  +6.78%  PF  3.41")
    print(f"  {'쌍끌이 동시 (A유형)':30s}   157건  WR  54.8%  avg  +7.43%  PF  3.13")

    # 최강 시나리오
    if results:
        best = max(results, key=lambda x: x["pf"])
        print(f"\n  {'★ 최강 바톤터치':30s}  → {best['name']}")
        print(f"     PF {best['pf']:.2f} / 승률 {best['wr']:.1f}% / 평균 {best['avg_ret']:+.2f}%")

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
