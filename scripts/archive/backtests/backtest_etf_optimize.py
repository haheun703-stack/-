"""ETF 3축 백테스트 — 드로다운 방어력 분석 + 파라미터 그리드 최적화.

사용법:
  python -u -X utf8 scripts/backtest_etf_optimize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 기존 백테스트 엔진 재사용
from scripts.backtest_etf_rotation import (
    load_all_etf_data, load_kospi, calc_regime, calc_sector_momentum,
    Portfolio, SECTOR_ETFS, LEVERAGE_ETFS, INDEX_ETFS, REGIME_ALLOC,
    COST_PER_TRADE,
)


def run_scenario(
    etf_data: dict,
    kospi: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    *,
    stop_loss: float = -5.0,
    rotation_freq: int = 1,     # 1=매일, 5=주간
    momentum_weights: tuple = (0.2, 0.5, 0.3),  # 5d, 20d, 60d
    max_sector: int = 3,
    momentum_top_n: int = 5,
    regime_alloc: dict = None,
    label: str = "default",
) -> dict:
    """단일 시나리오 백테스트 실행."""
    alloc_map = regime_alloc or REGIME_ALLOC
    portfolio = Portfolio(initial_cash=100_000_000)
    prev_regime = "CAUTION"
    last_rotation_day = 0

    for i, date in enumerate(trading_days):
        regime, above_ma20, above_ma60 = calc_regime(kospi, date)
        alloc = alloc_map[regime]

        # 가격 맵
        price_map = {}
        open_map = {}
        for code in etf_data:
            df = etf_data[code]
            today = df[df.index == date]
            if not today.empty:
                price_map[code] = float(today["close"].iloc[0])
                open_map[code] = float(today["open"].iloc[0])

        # 모멘텀 (커스텀 가중)
        momentum = calc_sector_momentum_custom(etf_data, date, momentum_weights)

        # 레짐 전환 → 포지션 정리
        if regime != prev_regime:
            portfolio.sell_by_axis("leverage", open_map, date, f"레짐 {prev_regime}→{regime}")
            if regime in ("BEAR", "CRISIS"):
                portfolio.sell_by_axis("sector", open_map, date, f"레짐 {regime}")
            if regime == "CRISIS":
                portfolio.sell_by_axis("index", open_map, date, "CRISIS")

        # 손절 체크
        for code in list(portfolio.positions.keys()):
            if code in price_map:
                pos = portfolio.positions[code]
                pnl = (price_map[code] / pos["avg_price"] - 1) * 100
                if pnl <= stop_loss:
                    portfolio.sell(code, open_map.get(code, price_map[code]), date, f"손절 {pnl:.1f}%")

        # 레버리지 보유일 체크 (5일)
        for code in [c for c, p in portfolio.positions.items() if p["axis"] == "leverage"]:
            entry = portfolio.positions[code]["entry_date"]
            hold_days = len(trading_days[(trading_days >= entry) & (trading_days <= date)])
            if hold_days >= 5:
                portfolio.sell(code, open_map.get(code, 0), date, f"보유일 {hold_days}일")

        # 섹터 로테이션 (빈도 제한)
        days_since_rotation = i - last_rotation_day
        do_rotation = (days_since_rotation >= rotation_freq) and (alloc["sector"] > 0) and momentum

        if do_rotation:
            last_rotation_day = i
            top_sectors = sorted(
                momentum.keys(), key=lambda s: momentum[s]["score"], reverse=True,
            )[:momentum_top_n]

            # 순위 이탈 청산
            for code in [c for c, p in portfolio.positions.items() if p["axis"] == "sector"]:
                sector_of = SECTOR_ETFS.get(code, ("", ""))[1]
                if sector_of not in top_sectors:
                    portfolio.sell(code, open_map.get(code, 0), date, "순위 이탈")

            # 신규 매수
            sector_count = sum(1 for p in portfolio.positions.values() if p["axis"] == "sector")
            slots = max_sector - sector_count
            if slots > 0:
                per_wt = alloc["sector"] / max_sector
                for sector in top_sectors:
                    if slots <= 0:
                        break
                    info = momentum[sector]
                    code = info["code"]
                    if code in portfolio.positions:
                        continue
                    if info["sm_type"] == "none" and info["supply_score"] < 50:
                        continue
                    price = open_map.get(code, 0)
                    if price > 0:
                        portfolio.buy(code, price, per_wt, "sector", date, info["name"])
                        slots -= 1

        # 레버리지
        lev_holding = any(p["axis"] == "leverage" for p in portfolio.positions.values())
        if alloc["leverage"] > 0 and not lev_holding:
            lev_info = LEVERAGE_ETFS.get(regime)
            if lev_info:
                code, name, mult = lev_info
                price = open_map.get(code, 0)
                if price > 0:
                    portfolio.buy(code, price, alloc["leverage"], "leverage", date, name)

        # 지수 ETF
        if alloc["index"] > 0:
            if above_ma20 and above_ma60:
                ma_adj = 1.0
            elif above_ma20:
                ma_adj = 0.8
            elif above_ma60:
                ma_adj = 0.6
            else:
                ma_adj = 0.4
            adj_pct = alloc["index"] * ma_adj
            for code, (name, w) in INDEX_ETFS.items():
                target = adj_pct * w
                if code not in portfolio.positions and target > 1:
                    price = open_map.get(code, 0)
                    if price > 0:
                        portfolio.buy(code, price, target, "index", date, name)

        portfolio.record_daily(date, etf_data)
        prev_regime = regime

    # 결과 계산
    df_val = pd.DataFrame(portfolio.daily_values)
    df_val["date"] = pd.to_datetime(df_val["date"])
    df_val.set_index("date", inplace=True)
    df_val["return_pct"] = (df_val["total_value"] / 100_000_000 - 1) * 100

    peak = df_val["total_value"].cummax()
    dd = (df_val["total_value"] - peak) / peak * 100
    mdd = dd.min()

    trades = pd.DataFrame(portfolio.trade_log)
    sells = trades[trades["action"] == "SELL"] if not trades.empty else pd.DataFrame()

    if not sells.empty and "pnl_pct" in sells.columns:
        wins = sells[sells["pnl_pct"] > 0]
        losses = sells[sells["pnl_pct"] <= 0]
        win_rate = len(wins) / len(sells) * 100
        gp = wins["pnl_pct"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0.01
        pf = gp / max(gl, 0.01)
        avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    else:
        win_rate = pf = avg_win = avg_loss = 0

    total_ret = df_val["return_pct"].iloc[-1]
    daily_ret = df_val["total_value"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    return {
        "label": label,
        "total_return": round(total_ret, 2),
        "mdd": round(mdd, 2),
        "sharpe": round(sharpe, 2),
        "trades": len(sells),
        "win_rate": round(win_rate, 1),
        "pf": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "daily_values": df_val,
    }


def calc_sector_momentum_custom(
    etf_data: dict, date: pd.Timestamp, weights: tuple
) -> dict:
    """커스텀 가중 모멘텀."""
    w5, w20, w60 = weights
    results = {}

    for code, (name, sector) in SECTOR_ETFS.items():
        if code not in etf_data:
            continue
        df = etf_data[code]
        hist = df[df.index < date]
        if len(hist) < 60:
            continue

        close = hist["close"]
        c = close.iloc[-1]
        ret_5d = (c / close.iloc[-5] - 1) * 100
        ret_20d = (c / close.iloc[-20] - 1) * 100
        ret_60d = (c / close.iloc[-60] - 1) * 100

        mom_score = ret_5d * w5 + ret_20d * w20 + ret_60d * w60

        vol = hist["volume"]
        avg5 = vol.iloc[-5:].mean()
        avg20 = vol.iloc[-20:].mean()
        vr = avg5 / max(avg20, 1)

        if vr > 1.5 and ret_5d > 0:
            sm = "smart_money"
        elif vr > 1.3:
            sm = "theme_money"
        else:
            sm = "none"

        recent = hist.iloc[-5:]
        up = (recent["close"] > recent["open"]).sum()
        supply = (up / 5) * 60 + min(vr / 2, 1.0) * 40

        results[sector] = {
            "code": code, "name": name,
            "5d": ret_5d, "20d": ret_20d, "60d": ret_60d,
            "score": mom_score, "sm_type": sm, "supply_score": supply,
        }

    for rank, sector in enumerate(
        sorted(results.keys(), key=lambda s: results[s]["score"], reverse=True), 1
    ):
        results[sector]["rank"] = rank

    return results


def drawdown_defense_analysis(
    strategy_values: pd.DataFrame,
    kospi: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
):
    """KOSPI 하락 구간에서 전략 방어력 분석."""
    # KOSPI 일별 수익률
    k = kospi.reindex(trading_days)
    k = k.dropna(subset=["close"])
    k["return"] = k["close"].pct_change()
    k["cum_return"] = (1 + k["return"]).cumprod() - 1
    k["peak"] = k["close"].cummax()
    k["dd"] = (k["close"] - k["peak"]) / k["peak"] * 100

    # 드로다운 구간 식별 (KOSPI -3% 이상 하락)
    in_dd = False
    dd_periods = []
    dd_start = None

    for date in k.index:
        if k.loc[date, "dd"] < -3:
            if not in_dd:
                dd_start = date
                in_dd = True
        else:
            if in_dd:
                dd_periods.append((dd_start, date))
                in_dd = False
    if in_dd:
        dd_periods.append((dd_start, k.index[-1]))

    # 각 드로다운 구간에서 전략 vs KOSPI 비교
    print(f"\n{'='*70}")
    print(f"  KOSPI 하락 구간 방어력 분석")
    print(f"{'='*70}")
    print(f"  {'구간':>25} | {'KOSPI':>8} | {'전략':>8} | {'방어':>8}")
    print(f"  {'-'*25}+{'-'*10}+{'-'*10}+{'-'*10}")

    total_kospi_dd = 0
    total_strat_dd = 0

    for dd_start, dd_end in dd_periods:
        # KOSPI 수익률
        k_slice = k.loc[dd_start:dd_end, "close"]
        if len(k_slice) < 2:
            continue
        k_ret = (k_slice.iloc[-1] / k_slice.iloc[0] - 1) * 100

        # 전략 수익률
        s_slice = strategy_values.loc[dd_start:dd_end, "total_value"]
        if len(s_slice) < 2:
            continue
        s_ret = (s_slice.iloc[-1] / s_slice.iloc[0] - 1) * 100

        defense = s_ret - k_ret  # 양수면 전략이 더 방어

        total_kospi_dd += k_ret
        total_strat_dd += s_ret

        period_str = f"{dd_start.strftime('%m/%d')}~{dd_end.strftime('%m/%d')}"
        print(f"  {period_str:>25} | {k_ret:>+7.2f}% | {s_ret:>+7.2f}% | {defense:>+7.2f}%p")

    if dd_periods:
        total_defense = total_strat_dd - total_kospi_dd
        print(f"  {'-'*25}+{'-'*10}+{'-'*10}+{'-'*10}")
        print(f"  {'합계':>25} | {total_kospi_dd:>+7.2f}% | {total_strat_dd:>+7.2f}% | {total_defense:>+7.2f}%p")
    else:
        print(f"  (KOSPI -3% 이상 드로다운 구간 없음)")

    return dd_periods


def main():
    print("=" * 70)
    print("  ETF 3축 백테스트 최적화")
    print("=" * 70)

    # 데이터 로드
    print("\n데이터 로드...")
    etf_data = load_all_etf_data()
    kospi = load_kospi()

    start = pd.Timestamp("2025-06-01")
    end = pd.Timestamp("2026-02-27")
    kodex_dates = etf_data["069500"].index
    trading_days = kodex_dates[(kodex_dates >= start) & (kodex_dates <= end)]
    print(f"  {len(etf_data)}종 ETF, {len(trading_days)}거래일\n")

    # ── 시나리오 정의 ──
    scenarios = [
        # A: 현재 (베이스라인)
        {
            "label": "A 현재설정",
            "stop_loss": -5.0,
            "rotation_freq": 1,
            "momentum_weights": (0.2, 0.5, 0.3),
        },
        # B: 손절 완화 (-7%)
        {
            "label": "B 손절-7%",
            "stop_loss": -7.0,
            "rotation_freq": 1,
            "momentum_weights": (0.2, 0.5, 0.3),
        },
        # C: 주간 로테이션
        {
            "label": "C 주간로테이션",
            "stop_loss": -5.0,
            "rotation_freq": 5,
            "momentum_weights": (0.2, 0.5, 0.3),
        },
        # D: 손절-7% + 주간 + 모멘텀 조정
        {
            "label": "D 종합개선",
            "stop_loss": -7.0,
            "rotation_freq": 5,
            "momentum_weights": (0.3, 0.4, 0.3),
        },
        # E: 손절-8% + 주간 + 모멘텀20d강화
        {
            "label": "E 중기모멘텀",
            "stop_loss": -8.0,
            "rotation_freq": 5,
            "momentum_weights": (0.15, 0.55, 0.3),
        },
        # F: 손절-7% + 격주
        {
            "label": "F 격주로테이션",
            "stop_loss": -7.0,
            "rotation_freq": 10,
            "momentum_weights": (0.2, 0.5, 0.3),
        },
    ]

    results = []
    for sc in scenarios:
        label = sc.pop("label")
        print(f"  ▶ {label} 실행 중...", end=" ", flush=True)
        r = run_scenario(etf_data, kospi, trading_days, label=label, **sc)
        results.append(r)
        print(f"수익 {r['total_return']:+.1f}% | PF {r['pf']:.2f} | MDD {r['mdd']:.1f}%")

    # ── 결과 비교 테이블 ──
    print(f"\n{'='*70}")
    print(f"  파라미터 그리드 테스트 결과")
    print(f"{'='*70}")
    print(f"  {'시나리오':>16} | {'수익률':>8} | {'MDD':>7} | {'PF':>5} | {'Sharpe':>6} | {'매매':>4} | {'승률':>6}")
    print(f"  {'-'*16}+{'-'*10}+{'-'*9}+{'-'*7}+{'-'*8}+{'-'*6}+{'-'*8}")

    best_pf = max(r["pf"] for r in results)
    for r in results:
        marker = " ★" if r["pf"] == best_pf else ""
        print(f"  {r['label']:>16} | {r['total_return']:>+7.2f}% | {r['mdd']:>6.2f}% | "
              f"{r['pf']:>5.2f} | {r['sharpe']:>6.2f} | {r['trades']:>4} | {r['win_rate']:>5.1f}%{marker}")

    # ── 최고 PF 시나리오의 드로다운 방어력 분석 ──
    best = max(results, key=lambda r: r["pf"])
    print(f"\n  ★ 최적 시나리오: {best['label']}")
    print(f"    PF {best['pf']:.2f} | 수익 {best['total_return']:+.1f}% | MDD {best['mdd']:.1f}% | Sharpe {best['sharpe']:.2f}")

    drawdown_defense_analysis(best["daily_values"], kospi, trading_days)

    # ── 베이스라인(A)도 드로다운 분석 ──
    baseline = results[0]
    print(f"\n  (참고: 베이스라인 A의 드로다운 방어력)")
    drawdown_defense_analysis(baseline["daily_values"], kospi, trading_days)

    # ── 월별 수익률 비교 (최적 vs 베이스) ──
    print(f"\n{'='*70}")
    print(f"  월별 수익률 비교")
    print(f"{'='*70}")
    print(f"  {'월':>8} | {'A 현재':>8} | {'★최적':>8} | {'차이':>8}")
    print(f"  {'-'*8}+{'-'*10}+{'-'*10}+{'-'*10}")

    a_monthly = baseline["daily_values"]["return_pct"].resample("ME").last().diff()
    a_monthly.iloc[0] = baseline["daily_values"]["return_pct"].resample("ME").last().iloc[0]

    b_monthly = best["daily_values"]["return_pct"].resample("ME").last().diff()
    b_monthly.iloc[0] = best["daily_values"]["return_pct"].resample("ME").last().iloc[0]

    for date in a_monthly.index:
        a_val = a_monthly.get(date, 0)
        b_val = b_monthly.get(date, 0) if date in b_monthly.index else 0
        diff = b_val - a_val
        print(f"  {date.strftime('%Y-%m'):>8} | {a_val:>+7.2f}% | {b_val:>+7.2f}% | {diff:>+7.2f}%p")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
