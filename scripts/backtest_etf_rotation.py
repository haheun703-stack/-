"""ETF 3축 로테이션 백테스트 엔진.

KOSPI 레짐 기반 비중 배분 + 섹터 모멘텀 로테이션 + 레버리지 타이밍 + 지수 ETF.
2025-05-01 ~ 2026-02-27 (60일 워밍업 후) 일별 시뮬레이션.

사용법:
  python -u -X utf8 scripts/backtest_etf_rotation.py
  python -u -X utf8 scripts/backtest_etf_rotation.py --start 2025-06-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ETF_DIR = PROJECT_ROOT / "data" / "sector_rotation" / "etf_daily"
KOSPI_PATH = PROJECT_ROOT / "data" / "kospi_index.csv"

# ─── 섹터 ETF 유니버스 (백테스트 대상) ───
# TIGER 22종 중 섹터 카테고리만 (그룹/시장 제외 = 18종)
SECTOR_ETFS = {
    "157500": ("TIGER 증권", "증권"),
    "091220": ("TIGER 은행", "은행"),
    "140710": ("TIGER 보험", "보험"),
    "091230": ("TIGER 반도체", "반도체"),
    "305540": ("TIGER 2차전지테마", "2차전지"),
    "364970": ("TIGER 바이오TOP10", "바이오"),
    "143860": ("TIGER 헬스케어", "헬스케어"),
    "139260": ("TIGER 200 IT", "IT"),
    "139220": ("TIGER 200 건설", "건설"),
    "139270": ("TIGER 200 금융", "금융"),
    "139250": ("TIGER 200 에너지화학", "에너지화학"),
    "139240": ("TIGER 200 철강소재", "철강소재"),
    "157490": ("TIGER 소프트웨어", "소프트웨어"),
    "228810": ("TIGER 미디어컨텐츠", "미디어"),
    "300610": ("TIGER K게임", "게임"),
    "365000": ("TIGER 인터넷TOP10", "인터넷"),
    "463250": ("TIGER K방산&우주", "방산"),
    "494670": ("TIGER 조선TOP10", "조선"),
}

LEVERAGE_ETFS = {
    "BULL":   ("122630", "KODEX 레버리지", 2.0),
    "BEAR":   ("114800", "KODEX 인버스", -1.0),
    "CRISIS": ("252670", "KODEX 200선물인버스2X", -2.0),
}

INDEX_ETFS = {
    "069500": ("KODEX 200", 0.7),
    "278530": ("KODEX MSCI Korea TR", 0.3),
}

# 레짐별 비중 매트릭스
REGIME_ALLOC = {
    "BULL":    {"sector": 40, "leverage": 20, "index": 30, "cash": 10},
    "CAUTION": {"sector": 30, "leverage":  0, "index": 30, "cash": 40},
    "BEAR":    {"sector":  0, "leverage": 15, "index": 15, "cash": 70},
    "CRISIS":  {"sector":  0, "leverage": 20, "index":  0, "cash": 80},
}

# 비용
SLIPPAGE_PCT = 0.003       # 0.3% 편도
COMMISSION_PCT = 0.00015   # 0.015% 편도
COST_PER_TRADE = SLIPPAGE_PCT + COMMISSION_PCT  # ~0.315% 편도


# ─── 데이터 로더 ───

def load_all_etf_data() -> dict[str, pd.DataFrame]:
    """전체 ETF parquet 로드."""
    all_codes = set()
    all_codes.update(SECTOR_ETFS.keys())
    for _, (code, _, _) in LEVERAGE_ETFS.items():
        all_codes.add(code)
    all_codes.update(INDEX_ETFS.keys())

    data = {}
    for code in all_codes:
        path = ETF_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # close 필수
            if "close" in df.columns and len(df) > 0:
                data[code] = df
    return data


def load_kospi() -> pd.DataFrame:
    """KOSPI 인덱스 로드."""
    df = pd.read_csv(KOSPI_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


# ─── KOSPI 레짐 계산 ───

def calc_regime(kospi: pd.DataFrame, date: pd.Timestamp) -> tuple[str, bool, bool]:
    """해당 날짜의 KOSPI 레짐 판단. 전일 종가 기준 (look-ahead bias 방지)."""
    prev = kospi[kospi.index < date]
    if len(prev) < 60:
        return "CAUTION", True, True

    close = prev["close"]
    last_close = close.iloc[-1]
    ma20 = close.iloc[-20:].mean()
    ma60 = close.iloc[-60:].mean()
    above_ma20 = last_close > ma20
    above_ma60 = last_close > ma60

    # RV20 (20일 수익률 표준편차의 백분위)
    returns = close.pct_change().dropna()
    rv20 = returns.iloc[-20:].std() * np.sqrt(252)
    # 전체 기간 백분위
    all_rv = returns.rolling(20).std().dropna() * np.sqrt(252)
    rv_pct = (all_rv < rv20).mean()

    if above_ma20 and rv_pct < 0.5:
        regime = "BULL"
    elif above_ma20:
        regime = "CAUTION"
    elif above_ma60:
        regime = "BEAR"
    else:
        regime = "CRISIS"

    return regime, above_ma20, above_ma60


# ─── 섹터 모멘텀 계산 ───

def calc_sector_momentum(etf_data: dict, date: pd.Timestamp) -> dict:
    """각 섹터 ETF의 모멘텀 계산 (5d/20d/60d 수익률)."""
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

        ret_5d = (c / close.iloc[-5] - 1) * 100 if len(hist) >= 5 else 0
        ret_20d = (c / close.iloc[-20] - 1) * 100 if len(hist) >= 20 else 0
        ret_60d = (c / close.iloc[-60] - 1) * 100 if len(hist) >= 60 else 0

        # 가중 모멘텀 점수
        mom_score = ret_5d * 0.2 + ret_20d * 0.5 + ret_60d * 0.3

        # 스마트 머니 프록시: 최근 5일 평균 거래량이 20일 평균의 1.5배 이상 + 양봉
        vol = hist["volume"]
        avg_vol_5 = vol.iloc[-5:].mean()
        avg_vol_20 = vol.iloc[-20:].mean()
        vol_ratio = avg_vol_5 / max(avg_vol_20, 1)

        if vol_ratio > 1.5 and ret_5d > 0:
            sm_type = "smart_money"
        elif vol_ratio > 1.3:
            sm_type = "theme_money"
        else:
            sm_type = "none"

        # 수급 점수 프록시: 5일 중 양봉 비율 + 거래량 가중
        recent_5 = hist.iloc[-5:]
        up_days = (recent_5["close"] > recent_5["open"]).sum()
        supply_score = (up_days / 5) * 60 + min(vol_ratio / 2, 1.0) * 40

        results[sector] = {
            "code": code,
            "name": name,
            "5d": ret_5d,
            "20d": ret_20d,
            "60d": ret_60d,
            "score": mom_score,
            "sm_type": sm_type,
            "supply_score": supply_score,
        }

    # 모멘텀 순위
    sorted_sectors = sorted(results.keys(), key=lambda s: results[s]["score"], reverse=True)
    for rank, sector in enumerate(sorted_sectors, 1):
        results[sector]["rank"] = rank

    return results


# ─── 포트폴리오 관리 ───

class Portfolio:
    """백테스트 포트폴리오 관리."""

    def __init__(self, initial_cash: float = 100_000_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}  # {code: {"qty": int, "avg_price": float, "axis": str}}
        self.trade_log: list[dict] = []
        self.daily_values: list[dict] = []

    def buy(self, code: str, price: float, weight_pct: float, axis: str, date, name: str = ""):
        """매수. weight_pct 기준으로 금액 결정."""
        total_value = self.total_value_at(code, price)
        target_amount = total_value * (weight_pct / 100)

        # 이미 보유 시 스킵
        if code in self.positions:
            return

        # 슬리피지 반영
        buy_price = price * (1 + COST_PER_TRADE)
        qty = int(target_amount / buy_price)
        if qty <= 0:
            return

        cost = qty * buy_price
        if cost > self.cash:
            qty = int(self.cash / buy_price)
            cost = qty * buy_price

        if qty <= 0:
            return

        self.cash -= cost
        self.positions[code] = {
            "qty": qty,
            "avg_price": buy_price,
            "axis": axis,
            "entry_date": date,
            "name": name,
        }
        self.trade_log.append({
            "date": date, "code": code, "name": name, "action": "BUY",
            "price": price, "qty": qty, "amount": cost, "axis": axis,
        })

    def sell(self, code: str, price: float, date, reason: str = ""):
        """전량 매도."""
        if code not in self.positions:
            return

        pos = self.positions[code]
        sell_price = price * (1 - COST_PER_TRADE)
        proceeds = pos["qty"] * sell_price
        pnl_pct = (sell_price / pos["avg_price"] - 1) * 100

        self.cash += proceeds
        self.trade_log.append({
            "date": date, "code": code, "name": pos.get("name", ""),
            "action": "SELL", "price": price, "qty": pos["qty"],
            "amount": proceeds, "pnl_pct": round(pnl_pct, 2),
            "axis": pos["axis"], "reason": reason,
        })
        del self.positions[code]

    def total_value_at(self, ref_code: str = "", ref_price: float = 0) -> float:
        """현재 총 자산가치 (마지막 기록 기준)."""
        if self.daily_values:
            return self.daily_values[-1]["total_value"]
        return self.cash

    def record_daily(self, date, etf_data: dict):
        """일별 총 자산 기록."""
        position_value = 0
        for code, pos in self.positions.items():
            if code in etf_data:
                df = etf_data[code]
                price_data = df[df.index <= date]
                if not price_data.empty:
                    current_price = price_data["close"].iloc[-1]
                    position_value += pos["qty"] * current_price

        total = self.cash + position_value
        self.daily_values.append({
            "date": date,
            "total_value": total,
            "cash": self.cash,
            "position_value": position_value,
            "position_count": len(self.positions),
        })

    def sell_by_axis(self, axis: str, price_map: dict, date, reason: str = ""):
        """특정 축의 포지션 전량 청산."""
        codes_to_sell = [c for c, p in self.positions.items() if p["axis"] == axis]
        for code in codes_to_sell:
            price = price_map.get(code, 0)
            if price > 0:
                self.sell(code, price, date, reason)


# ─── 백테스트 엔진 ───

def run_backtest(start_date: str = "2025-06-01", end_date: str = "2026-02-27") -> dict:
    """ETF 3축 로테이션 백테스트 실행."""
    print("=" * 60)
    print("  ETF 3축 로테이션 백테스트")
    print(f"  기간: {start_date} ~ {end_date}")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    etf_data = load_all_etf_data()
    kospi = load_kospi()
    print(f"  ETF {len(etf_data)}종 로드 완료")
    print(f"  KOSPI {len(kospi)}일 로드 완료")

    # 거래일 리스트 생성
    # KODEX 200 기준 거래일
    kodex_dates = etf_data["069500"].index
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_days = kodex_dates[(kodex_dates >= start) & (kodex_dates <= end)]

    print(f"  백테스트 거래일: {len(trading_days)}일\n")

    # 포트폴리오 초기화
    portfolio = Portfolio(initial_cash=100_000_000)

    prev_regime = "CAUTION"
    prev_sector_holdings = set()  # 이전에 보유했던 섹터
    prev_sector_codes = set()     # 이전 섹터 ETF 코드

    # 섹터 로테이션 설정
    MAX_SECTOR_HOLDINGS = 3
    MOMENTUM_TOP_N = 5
    STOP_LOSS_PCT = -5.0
    LEVERAGE_MAX_HOLD = 5
    leverage_entry_date = None

    print("[2/3] 백테스트 실행...")

    for i, date in enumerate(trading_days):
        # ── 1. 레짐 판단 ──
        regime, above_ma20, above_ma60 = calc_regime(kospi, date)

        # ── 2. 비중 배분 ──
        alloc = REGIME_ALLOC[regime]

        # ── 3. 현재가 맵 (오늘 종가) ──
        price_map = {}
        open_map = {}
        for code in etf_data:
            df = etf_data[code]
            today_data = df[df.index == date]
            if not today_data.empty:
                price_map[code] = float(today_data["close"].iloc[0])
                open_map[code] = float(today_data["open"].iloc[0])

        # ── 4. 섹터 모멘텀 ──
        momentum = calc_sector_momentum(etf_data, date)

        # ── 5. 레짐 전환 → 포지션 정리 ──
        if regime != prev_regime:
            # 레짐 변경 시 레버리지 청산
            portfolio.sell_by_axis("leverage", open_map, date, f"레짐 전환 {prev_regime}→{regime}")

            # BEAR/CRISIS → 섹터 ETF 전량 청산
            if regime in ("BEAR", "CRISIS"):
                portfolio.sell_by_axis("sector", open_map, date, f"레짐 {regime} 전환")

            # CRISIS → 지수 ETF 전량 청산
            if regime == "CRISIS":
                portfolio.sell_by_axis("index", open_map, date, "CRISIS 레짐 전환")

        # ── 6. 기존 포지션 손절 체크 ──
        codes_to_sell = []
        for code, pos in portfolio.positions.items():
            if code in price_map:
                pnl = (price_map[code] / pos["avg_price"] - 1) * 100
                if pnl <= STOP_LOSS_PCT:
                    codes_to_sell.append((code, f"손절 {pnl:.1f}%"))

        for code, reason in codes_to_sell:
            portfolio.sell(code, open_map.get(code, price_map.get(code, 0)), date, reason)

        # ── 7. 레버리지 보유일 체크 ──
        lev_codes = [c for c, p in portfolio.positions.items() if p["axis"] == "leverage"]
        for code in lev_codes:
            entry = portfolio.positions[code]["entry_date"]
            hold_days = len(trading_days[(trading_days >= entry) & (trading_days <= date)])
            if hold_days >= LEVERAGE_MAX_HOLD:
                portfolio.sell(code, open_map.get(code, 0), date, f"보유일 초과 {hold_days}일")

        # ── 8. 섹터 ETF 로테이션 (BULL/CAUTION만) ──
        if alloc["sector"] > 0 and momentum:
            # 상위 N개 섹터 필터
            top_sectors = sorted(
                momentum.keys(),
                key=lambda s: momentum[s]["score"],
                reverse=True,
            )[:MOMENTUM_TOP_N]

            # 현재 보유 중인 섹터 ETF
            current_sector_codes = {
                c for c, p in portfolio.positions.items() if p["axis"] == "sector"
            }

            # 순위 이탈 종목 청산 (모멘텀 TOP N 밖으로)
            for code in list(current_sector_codes):
                pos_info = portfolio.positions.get(code, {})
                name = pos_info.get("name", "")
                # 해당 코드의 섹터 찾기
                sector_of_code = SECTOR_ETFS.get(code, ("", ""))[1]
                if sector_of_code not in top_sectors:
                    portfolio.sell(code, open_map.get(code, 0), date, "모멘텀 순위 이탈")

            # 신규 매수 (빈 슬롯)
            current_sector_count = sum(
                1 for p in portfolio.positions.values() if p["axis"] == "sector"
            )
            slots = MAX_SECTOR_HOLDINGS - current_sector_count

            if slots > 0:
                per_etf_weight = alloc["sector"] / MAX_SECTOR_HOLDINGS
                for sector in top_sectors:
                    if slots <= 0:
                        break
                    info = momentum[sector]
                    code = info["code"]

                    # 이미 보유 중이면 스킵
                    if code in portfolio.positions:
                        continue

                    # 스마트머니 + 수급 간이 필터
                    if info["sm_type"] == "none" and info["supply_score"] < 50:
                        continue

                    price = open_map.get(code, 0)
                    if price > 0:
                        portfolio.buy(
                            code, price, per_etf_weight,
                            axis="sector", date=date, name=info["name"],
                        )
                        slots -= 1

        # ── 9. 레버리지 매매 ──
        lev_holding = any(p["axis"] == "leverage" for p in portfolio.positions.values())
        if alloc["leverage"] > 0 and not lev_holding:
            lev_info = LEVERAGE_ETFS.get(regime)
            if lev_info:
                lev_code, lev_name, lev_mult = lev_info
                price = open_map.get(lev_code, 0)
                if price > 0:
                    portfolio.buy(
                        lev_code, price, alloc["leverage"],
                        axis="leverage", date=date, name=lev_name,
                    )

        # ── 10. 지수 ETF ──
        if alloc["index"] > 0:
            # MA 보정
            if above_ma20 and above_ma60:
                ma_adj = 1.0
            elif above_ma20:
                ma_adj = 0.8
            elif above_ma60:
                ma_adj = 0.6
            else:
                ma_adj = 0.4

            adjusted_index_pct = alloc["index"] * ma_adj

            for code, (name, inner_weight) in INDEX_ETFS.items():
                target_pct = adjusted_index_pct * inner_weight
                if code not in portfolio.positions and target_pct > 1:
                    price = open_map.get(code, 0)
                    if price > 0:
                        portfolio.buy(
                            code, price, target_pct,
                            axis="index", date=date, name=name,
                        )

        # ── 일별 기록 ──
        portfolio.record_daily(date, etf_data)
        prev_regime = regime

        # 진행 표시
        if (i + 1) % 50 == 0 or i == 0 or i == len(trading_days) - 1:
            val = portfolio.daily_values[-1]
            ret = (val["total_value"] / portfolio.initial_cash - 1) * 100
            print(f"  {date.strftime('%Y-%m-%d')} | {regime:8s} | "
                  f"총자산: {val['total_value']:>14,.0f}원 ({ret:+.1f}%) | "
                  f"포지션: {val['position_count']}개")

    # ── 3. 결과 분석 ──
    print("\n[3/3] 결과 분석...")
    results = analyze_results(portfolio)
    print_report(results, start_date, end_date)

    return results


def analyze_results(portfolio: Portfolio) -> dict:
    """백테스트 결과 분석."""
    df = pd.DataFrame(portfolio.daily_values)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 수익률 곡선
    df["return_pct"] = (df["total_value"] / portfolio.initial_cash - 1) * 100
    df["daily_return"] = df["total_value"].pct_change()

    # 최대 낙폭 (MDD)
    peak = df["total_value"].cummax()
    dd = (df["total_value"] - peak) / peak * 100
    mdd = dd.min()

    # 매매 분석
    trades = pd.DataFrame(portfolio.trade_log)
    sell_trades = trades[trades["action"] == "SELL"] if not trades.empty else pd.DataFrame()

    if not sell_trades.empty and "pnl_pct" in sell_trades.columns:
        win_trades = sell_trades[sell_trades["pnl_pct"] > 0]
        win_rate = len(win_trades) / len(sell_trades) * 100

        gross_profit = sell_trades[sell_trades["pnl_pct"] > 0]["pnl_pct"].sum()
        gross_loss = abs(sell_trades[sell_trades["pnl_pct"] < 0]["pnl_pct"].sum())
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_win = win_trades["pnl_pct"].mean() if len(win_trades) > 0 else 0
        avg_loss = sell_trades[sell_trades["pnl_pct"] <= 0]["pnl_pct"].mean() if len(sell_trades[sell_trades["pnl_pct"] <= 0]) > 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0

    total_return = df["return_pct"].iloc[-1]

    # 축별 트레이드
    axis_stats = {}
    if not sell_trades.empty:
        for axis in ["sector", "leverage", "index"]:
            axis_sells = sell_trades[sell_trades["axis"] == axis]
            if len(axis_sells) > 0:
                axis_wins = axis_sells[axis_sells["pnl_pct"] > 0]
                axis_stats[axis] = {
                    "trades": len(axis_sells),
                    "win_rate": len(axis_wins) / len(axis_sells) * 100 if len(axis_sells) > 0 else 0,
                    "avg_pnl": axis_sells["pnl_pct"].mean(),
                    "total_pnl": axis_sells["pnl_pct"].sum(),
                }

    # 연환산 수익률
    days = len(df)
    annual_return = (1 + total_return / 100) ** (252 / max(days, 1)) - 1
    # Sharpe
    daily_std = df["daily_return"].std()
    sharpe = (df["daily_return"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    return {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "mdd_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": len(sell_trades),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "axis_stats": axis_stats,
        "daily_values": df,
        "trades": trades if not trades.empty else pd.DataFrame(),
    }


def print_report(results: dict, start: str, end: str):
    """백테스트 결과 출력."""
    print("\n" + "=" * 60)
    print("  ETF 3축 로테이션 백테스트 결과")
    print(f"  기간: {start} ~ {end}")
    print("=" * 60)

    print(f"\n  총 수익률:       {results['total_return_pct']:>+8.2f}%")
    print(f"  연환산 수익률:   {results['annual_return_pct']:>+8.2f}%")
    print(f"  MDD:             {results['mdd_pct']:>8.2f}%")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']:>8.2f}")
    print(f"\n  총 매매 횟수:    {results['total_trades']:>5}건")
    print(f"  승률:            {results['win_rate_pct']:>7.1f}%")
    print(f"  Profit Factor:   {results['profit_factor']:>8.2f}")
    print(f"  평균 수익:       {results['avg_win_pct']:>+8.2f}%")
    print(f"  평균 손실:       {results['avg_loss_pct']:>+8.2f}%")

    if results["axis_stats"]:
        print(f"\n  {'축':>10} | {'매매':>4} | {'승률':>6} | {'평균P&L':>8} | {'누적P&L':>8}")
        print(f"  {'-'*10}+{'-'*6}+{'-'*8}+{'-'*10}+{'-'*10}")
        for axis, stats in results["axis_stats"].items():
            print(f"  {axis:>10} | {stats['trades']:>4} | {stats['win_rate']:>5.1f}% | "
                  f"{stats['avg_pnl']:>+7.2f}% | {stats['total_pnl']:>+7.1f}%")

    # 월별 수익률
    df = results["daily_values"]
    monthly = df["return_pct"].resample("ME").last().diff()
    monthly.iloc[0] = df["return_pct"].resample("ME").last().iloc[0]

    print(f"\n  월별 수익률:")
    for date, ret in monthly.items():
        bar = "█" * int(abs(ret) * 2) if not pd.isna(ret) else ""
        sign = "+" if ret >= 0 else ""
        print(f"    {date.strftime('%Y-%m')} | {sign}{ret:.2f}% {bar}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ETF 3축 로테이션 백테스트")
    parser.add_argument("--start", default="2025-06-01", help="시작일 (기본 2025-06-01, 워밍업 고려)")
    parser.add_argument("--end", default="2026-02-27", help="종료일")
    args = parser.parse_args()
    run_backtest(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
