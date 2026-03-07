"""
ETF 3축 백테스트: Standard vs Predator 모드 비교
=================================================
ETF 일봉 데이터(1년) + 모멘텀 히스토리(6개월)로 시뮬레이션.

Standard: 격주 리밸런싱, 모멘텀 Top-3 균등 배분
Predator: 가속도 기반 선제 매수, 확신 집중, 이벤트 트리거

Usage:
    python -u -X utf8 scripts/backtest_etf_predator.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================
# 데이터 로더
# ============================================================

ETF_DIR = PROJECT_ROOT / "data" / "sector_rotation" / "etf_daily"
MOMENTUM_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "momentum" / "momentum_history.parquet"
KOSPI_PATH = PROJECT_ROOT / "data" / "kospi_index.csv"
UNIVERSE_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "etf_universe.json"


def load_etf_prices() -> dict[str, pd.DataFrame]:
    """모든 ETF 일봉 로드."""
    prices = {}
    for p in ETF_DIR.glob("*.parquet"):
        code = p.stem
        df = pd.read_parquet(p)
        if "close" in df.columns and len(df) > 60:
            prices[code] = df[["close", "volume"]].copy()
    return prices


def load_sector_map() -> dict[str, dict]:
    """ETF 코드 → 섹터 매핑."""
    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        universe = json.load(f)
    mapping = {}
    for sector, info in universe.items():
        code = info["etf_code"]
        mapping[code] = {"sector": sector, "name": info["etf_name"]}
    return mapping


def load_momentum_history() -> pd.DataFrame:
    """모멘텀 히스토리 로드."""
    return pd.read_parquet(MOMENTUM_PATH)


def load_kospi() -> pd.DataFrame:
    """KOSPI 인덱스 로드."""
    df = pd.read_csv(KOSPI_PATH, index_col=0, parse_dates=True)
    return df


# ============================================================
# 모멘텀 계산기 (price 기반 — 히스토리 없는 기간용)
# ============================================================

def calc_momentum_from_prices(
    prices: dict[str, pd.DataFrame],
    sector_map: dict[str, dict],
    date: pd.Timestamp,
) -> dict[str, dict]:
    """ETF 가격에서 모멘텀 지표 직접 계산.

    Returns:
        {sector: {rank, ret_5, ret_20, ret_60, momentum_score, code}}
    """
    sectors = {}
    for code, info in sector_map.items():
        if code not in prices:
            continue
        df = prices[code]
        mask = df.index <= date
        if mask.sum() < 60:
            continue
        sub = df.loc[mask]

        close_now = sub["close"].iloc[-1]
        ret_5 = (close_now / sub["close"].iloc[-6] - 1) * 100 if len(sub) >= 6 else 0
        ret_20 = (close_now / sub["close"].iloc[-21] - 1) * 100 if len(sub) >= 21 else 0
        ret_60 = (close_now / sub["close"].iloc[-61] - 1) * 100 if len(sub) >= 61 else 0

        # 가중 모멘텀 스코어
        mom_score = ret_5 * 0.2 + ret_20 * 0.5 + ret_60 * 0.3

        sectors[info["sector"]] = {
            "code": code,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "ret_60": ret_60,
            "momentum_score": mom_score,
        }

    # 순위 매기기
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]["momentum_score"], reverse=True)
    for i, (sector, data) in enumerate(sorted_sectors, 1):
        data["rank"] = i

    return sectors


# ============================================================
# KOSPI 레짐 계산
# ============================================================

def calc_regime(kospi: pd.DataFrame, date: pd.Timestamp) -> str:
    """KOSPI 레짐 판정."""
    mask = kospi.index <= date
    if mask.sum() < 60:
        return "CAUTION"
    sub = kospi.loc[mask]
    close = sub["close"].iloc[-1]
    ma20 = sub["close"].rolling(20).mean().iloc[-1]
    ma60 = sub["close"].rolling(60).mean().iloc[-1]

    # 실현변동성
    returns = sub["close"].pct_change().dropna()
    rv20 = returns.iloc[-20:].std() * np.sqrt(252) * 100 if len(returns) >= 20 else 20
    rv_percentile = (returns.rolling(252).std().rank(pct=True) * 100).iloc[-1] if len(returns) >= 252 else 50

    if close > ma20 and rv_percentile < 50:
        return "BULL"
    elif close > ma20:
        return "CAUTION"
    elif close > ma60:
        return "BEAR"
    else:
        return "CRISIS"


# ============================================================
# 포트폴리오 시뮬레이터
# ============================================================

REGIME_ALLOCATION = {
    "BULL":    {"sector": 40, "leverage": 20, "index": 30, "cash": 10},
    "CAUTION": {"sector": 30, "leverage": 0,  "index": 30, "cash": 40},
    "BEAR":    {"sector": 0,  "leverage": 15, "index": 15, "cash": 70},
    "CRISIS":  {"sector": 0,  "leverage": 20, "index": 0,  "cash": 80},
}

SLIPPAGE_PCT = 0.3  # 편도 슬리피지


class PortfolioSimulator:
    """포트폴리오 시뮬레이터 (Standard / Predator 공용)."""

    def __init__(self, initial_capital: float = 100_000_000, mode: str = "standard"):
        self.mode = mode
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions: dict[str, dict] = {}  # {code: {shares, entry_price, sector, weight}}
        self.cash = initial_capital
        self.history: list[dict] = []
        self.trades: list[dict] = []
        self.total_value_history: list[float] = []

        # Standard 설정
        self.rebalance_freq = 10  # 거래일
        self.top_n = 3
        self.stop_loss_pct = -7.0

        # Predator 설정
        self.conviction_weights = {"HIGH": 60, "MID": 30, "LOW": 10}
        self.conviction_thresholds = {"HIGH": 70, "MID": 45, "LOW": 25}

    def get_total_value(self, prices: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        """총 평가액."""
        total = self.cash
        for code, pos in self.positions.items():
            if code in prices:
                mask = prices[code].index <= date
                if mask.sum() > 0:
                    current_price = prices[code].loc[mask, "close"].iloc[-1]
                    total += pos["shares"] * current_price
        return total

    def rebalance_standard(
        self,
        date: pd.Timestamp,
        momentum: dict[str, dict],
        prices: dict[str, pd.DataFrame],
        sector_pct: float,
    ):
        """Standard: 모멘텀 Top-N 균등 배분."""
        # Top N 선정
        sorted_sectors = sorted(momentum.items(), key=lambda x: x[1].get("rank", 99))
        top_sectors = sorted_sectors[:self.top_n]

        # 전체 청산
        self._close_all_positions(prices, date, "리밸런싱")

        # 새 포지션 진입
        total_value = self.cash
        per_sector_pct = sector_pct / max(len(top_sectors), 1)

        for sector, data in top_sectors:
            code = data.get("code", "")
            if not code or code not in prices:
                continue
            mask = prices[code].index <= date
            if mask.sum() == 0:
                continue
            price = prices[code].loc[mask, "close"].iloc[-1]
            buy_price = price * (1 + SLIPPAGE_PCT / 100)

            target_value = total_value * per_sector_pct / 100
            shares = int(target_value / buy_price)
            if shares <= 0:
                continue

            cost = shares * buy_price
            if cost > self.cash:
                shares = int(self.cash / buy_price)
                cost = shares * buy_price

            self.cash -= cost
            self.positions[code] = {
                "shares": shares,
                "entry_price": buy_price,
                "sector": sector,
                "weight": per_sector_pct,
            }
            self.trades.append({
                "date": str(date.date()),
                "action": "BUY",
                "code": code,
                "sector": sector,
                "price": buy_price,
                "shares": shares,
                "value": cost,
                "reason": f"Standard Top-{data.get('rank', '?')}",
            })

    def rebalance_predator(
        self,
        date: pd.Timestamp,
        current_momentum: dict[str, dict],
        prev_momentum: dict[str, dict],
        prices: dict[str, pd.DataFrame],
        sector_pct: float,
        supply_data: dict = None,
        keep_existing: bool = False,
    ):
        """Predator: 가속도 + 확신 집중 배분.

        keep_existing: True면 기존 보유 중 여전히 상위인 포지션은 유지 (불필요 회전 방지)
        """
        supply_data = supply_data or {}

        # 가속도 계산
        accel_scores = {}
        for sector, cur in current_momentum.items():
            prev = prev_momentum.get(sector, {})
            cur_rank = cur.get("rank", 20)
            prev_rank = prev.get("rank", cur_rank)
            rank_change = prev_rank - cur_rank  # 양수 = 상승

            ret_5 = cur.get("ret_5", 0)
            mom_score = cur.get("momentum_score", 0)

            # 가속도 점수
            rank_norm = min(max((rank_change + 10) / 20 * 100, 0), 100)
            ret_norm = min(max((ret_5 + 10) / 30 * 100, 0), 100)
            mom_norm = min(max(mom_score, 0), 100)
            rank_bonus = max(0, (15 - cur_rank)) / 14 * 100

            accel_score = rank_norm * 0.35 + ret_norm * 0.25 + mom_norm * 0.25 + rank_bonus * 0.15

            # 수급 보너스
            sd = supply_data.get(sector, {})
            if sd.get("foreign_cum", 0) > 0 and sd.get("inst_cum", 0) > 0:
                accel_score = min(accel_score + 15, 100)
            elif sd.get("foreign_cum", 0) > 0:
                accel_score = min(accel_score + 8, 100)

            accel_scores[sector] = {
                "score": accel_score,
                "rank_change": rank_change,
                "code": cur.get("code", ""),
            }

        # Top N by 가속도
        sorted_accel = sorted(accel_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top_sectors = sorted_accel[:self.top_n]

        # 확신도 레벨 판정
        convictions = []
        for sector, data in top_sectors:
            score = data["score"]
            if score >= self.conviction_thresholds["HIGH"]:
                level = "HIGH"
            elif score >= self.conviction_thresholds["MID"]:
                level = "MID"
            else:
                level = "LOW"
            convictions.append({
                "sector": sector,
                "code": data["code"],
                "level": level,
                "score": score,
                "rank_change": data["rank_change"],
            })

        # 비대칭 배분 비중 계산
        raw_weights = [self.conviction_weights[c["level"]] for c in convictions]
        total_raw = sum(raw_weights) or 1

        # keep_existing: 기존 보유 중 새 Top-N에 포함된 코드는 유지
        new_target_codes = {c["code"] for c in convictions}
        if keep_existing:
            # 기존 보유 중 새 타겟에 없는 것만 청산
            to_close = [code for code in list(self.positions.keys()) if code not in new_target_codes]
            for code in to_close:
                pos = self.positions[code]
                if code in prices:
                    mask = prices[code].index <= date
                    if mask.sum() > 0:
                        price = prices[code].loc[mask, "close"].iloc[-1]
                        sell_price = price * (1 - SLIPPAGE_PCT / 100)
                        proceeds = pos["shares"] * sell_price
                        self.cash += proceeds
                        self.trades.append({
                            "date": str(date.date()),
                            "action": "SELL",
                            "code": code,
                            "sector": pos["sector"],
                            "price": sell_price,
                            "shares": pos["shares"],
                            "value": proceeds,
                            "reason": "가속도 이탈 → 교체",
                        })
                del self.positions[code]
        else:
            self._close_all_positions(prices, date, "프레데터 리밸런싱")

        total_value = self.get_total_value(prices, date)
        for i, conv in enumerate(convictions):
            code = conv["code"]
            if not code or code not in prices:
                continue

            # 이미 보유 중이면 유지 (추가 매수 안 함)
            if code in self.positions:
                continue

            mask = prices[code].index <= date
            if mask.sum() == 0:
                continue
            price = prices[code].loc[mask, "close"].iloc[-1]
            buy_price = price * (1 + SLIPPAGE_PCT / 100)

            weight_pct = sector_pct * raw_weights[i] / total_raw
            target_value = total_value * weight_pct / 100
            shares = int(target_value / buy_price)
            if shares <= 0:
                continue

            cost = shares * buy_price
            if cost > self.cash:
                shares = int(self.cash / buy_price)
                cost = shares * buy_price
            if shares <= 0:
                continue

            self.cash -= cost
            self.positions[code] = {
                "shares": shares,
                "entry_price": buy_price,
                "sector": conv["sector"],
                "weight": weight_pct,
            }
            self.trades.append({
                "date": str(date.date()),
                "action": "BUY",
                "code": code,
                "sector": conv["sector"],
                "price": buy_price,
                "shares": shares,
                "value": cost,
                "reason": f"Predator {conv['level']}({conv['score']:.0f}) Δrank={conv['rank_change']:+d}",
            })

    def check_stop_loss(self, prices: dict[str, pd.DataFrame], date: pd.Timestamp):
        """손절 체크."""
        to_close = []
        for code, pos in self.positions.items():
            if code not in prices:
                continue
            mask = prices[code].index <= date
            if mask.sum() == 0:
                continue
            current_price = prices[code].loc[mask, "close"].iloc[-1]
            pnl_pct = (current_price / pos["entry_price"] - 1) * 100
            if pnl_pct <= self.stop_loss_pct:
                to_close.append((code, current_price, pnl_pct))

        for code, price, pnl in to_close:
            pos = self.positions[code]
            sell_price = price * (1 - SLIPPAGE_PCT / 100)
            proceeds = pos["shares"] * sell_price
            self.cash += proceeds
            self.trades.append({
                "date": str(date.date()),
                "action": "STOP_LOSS",
                "code": code,
                "sector": pos["sector"],
                "price": sell_price,
                "shares": pos["shares"],
                "value": proceeds,
                "reason": f"손절 {pnl:.1f}%",
            })
            del self.positions[code]

    def check_event_trigger(
        self,
        prices: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        momentum: dict[str, dict],
    ) -> bool:
        """이벤트 트리거 체크 (Predator 전용).

        일간 +5% 이상 급등 섹터 발견 시 True.
        """
        for sector, data in momentum.items():
            code = data.get("code", "")
            if not code or code not in prices:
                continue
            mask = prices[code].index <= date
            if mask.sum() < 2:
                continue
            sub = prices[code].loc[mask]
            ret_1d = (sub["close"].iloc[-1] / sub["close"].iloc[-2] - 1) * 100
            if ret_1d >= 5.0:
                return True
        return False

    def _close_all_positions(self, prices: dict[str, pd.DataFrame], date: pd.Timestamp, reason: str):
        """전 포지션 청산."""
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            if code in prices:
                mask = prices[code].index <= date
                if mask.sum() > 0:
                    price = prices[code].loc[mask, "close"].iloc[-1]
                    sell_price = price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = pos["shares"] * sell_price
                    self.cash += proceeds
                    self.trades.append({
                        "date": str(date.date()),
                        "action": "SELL",
                        "code": code,
                        "sector": pos["sector"],
                        "price": sell_price,
                        "shares": pos["shares"],
                        "value": proceeds,
                        "reason": reason,
                    })
        self.positions.clear()

    def record_daily(self, date: pd.Timestamp, prices: dict[str, pd.DataFrame], regime: str):
        """일별 기록."""
        total = self.get_total_value(prices, date)
        self.total_value_history.append(total)
        self.history.append({
            "date": str(date.date()),
            "total_value": total,
            "cash": self.cash,
            "positions": len(self.positions),
            "regime": regime,
            "return_pct": (total / self.initial_capital - 1) * 100,
        })


# ============================================================
# 백테스트 실행
# ============================================================

def run_backtest(mode: str, prices: dict, sector_map: dict, momentum_hist: pd.DataFrame, kospi: pd.DataFrame):
    """백테스트 실행.

    Args:
        mode: "standard" or "predator"
    """
    sim = PortfolioSimulator(initial_capital=100_000_000, mode=mode)

    # 거래일 리스트 (모멘텀 히스토리 기간)
    dates = sorted(momentum_hist.index.unique())
    if len(dates) < 20:
        print(f"  ⚠️ 데이터 부족: {len(dates)}일")
        return None

    # 이전 모멘텀 저장 (프레데터용)
    prev_momentum = {}
    last_rebalance_idx = -999
    rebalance_freq = 10  # 격주

    print(f"\n{'='*50}")
    print(f"🎯 {mode.upper()} 모드 백테스트")
    print(f"{'='*50}")
    print(f"기간: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}거래일)")

    for i, date in enumerate(dates):
        # 현재 모멘텀 (히스토리에서 직접 추출)
        day_data = momentum_hist.loc[date]
        current_momentum = {}
        for _, row in day_data.iterrows():
            sector = row["sector"]
            current_momentum[sector] = {
                "code": row["etf_code"],
                "rank": int(row["rank"]),
                "ret_5": row["ret_5"],
                "ret_20": row["ret_20"],
                "ret_60": row["ret_60"],
                "momentum_score": row["momentum_score"],
            }

        # 레짐 판정
        regime = calc_regime(kospi, date)
        alloc = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION["CAUTION"])
        sector_pct = alloc["sector"]

        # 손절 체크 (매일)
        sim.check_stop_loss(prices, date)

        # 리밸런싱 판단
        should_rebalance = False
        days_since = i - last_rebalance_idx

        if days_since >= rebalance_freq:
            should_rebalance = True

        # Predator: 이벤트 트리거 (쿨다운 5일 이상이어야 발동)
        if mode == "predator" and not should_rebalance and days_since >= 5:
            if sim.check_event_trigger(prices, date, current_momentum):
                should_rebalance = True

        # 리밸런싱 실행
        if should_rebalance and sector_pct > 0 and prev_momentum:
            if mode == "standard":
                sim.rebalance_standard(date, current_momentum, prices, sector_pct)
            elif mode == "predator":
                # Predator: 기존 보유 중 여전히 상위면 유지 (불필요 회전 방지)
                sim.rebalance_predator(
                    date, current_momentum, prev_momentum, prices, sector_pct,
                    keep_existing=True,
                )
            last_rebalance_idx = i

        # BEAR/CRISIS 레짐에서 섹터 비중 0% → 전량 청산
        if sector_pct == 0 and sim.positions:
            sim._close_all_positions(prices, date, f"레짐 {regime}: 섹터 비중 0%")

        # 일별 기록
        sim.record_daily(date, prices, regime)
        prev_momentum = current_momentum.copy()

    # 마지막 날 청산
    if sim.positions:
        sim._close_all_positions(prices, dates[-1], "백테스트 종료")
        sim.record_daily(dates[-1], prices, regime)

    return sim


# ============================================================
# 결과 분석
# ============================================================

def analyze_results(sim: PortfolioSimulator, mode: str) -> dict:
    """백테스트 결과 분석."""
    if not sim or not sim.history:
        return {}

    values = [h["total_value"] for h in sim.history]
    returns = pd.Series(values).pct_change().dropna()

    # 수익률
    total_return = (values[-1] / values[0] - 1) * 100

    # MDD
    peak = pd.Series(values).cummax()
    dd = (pd.Series(values) - peak) / peak * 100
    mdd = dd.min()

    # 거래 분석
    trades = sim.trades
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] in ["SELL", "STOP_LOSS"]]

    # PF 계산
    profits = []
    losses = []
    for i, sell in enumerate(sell_trades):
        # 매칭되는 매수 찾기
        matching_buys = [b for b in buy_trades if b["code"] == sell["code"] and b["date"] <= sell["date"]]
        if matching_buys:
            buy = matching_buys[-1]
            pnl = (sell["price"] / buy["price"] - 1) * 100
            if pnl > 0:
                profits.append(pnl)
            else:
                losses.append(abs(pnl))

    pf = sum(profits) / sum(losses) if losses else float("inf")
    win_rate = len(profits) / (len(profits) + len(losses)) * 100 if (profits or losses) else 0

    # 샤프 비율
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0

    # 레짐 분포
    regimes = pd.Series([h["regime"] for h in sim.history])
    regime_dist = regimes.value_counts(normalize=True) * 100

    # 손절 횟수
    stop_losses = len([t for t in trades if t["action"] == "STOP_LOSS"])

    # 리밸런싱 횟수
    rebalance_count = len([t for t in buy_trades])

    result = {
        "mode": mode,
        "total_return_pct": round(total_return, 2),
        "mdd_pct": round(mdd, 2),
        "pf": round(pf, 2),
        "win_rate_pct": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "total_trades": len(profits) + len(losses),
        "wins": len(profits),
        "losses": len(losses),
        "avg_profit_pct": round(np.mean(profits), 2) if profits else 0,
        "avg_loss_pct": round(np.mean(losses), 2) if losses else 0,
        "stop_losses": stop_losses,
        "buy_trades": len(buy_trades),
        "regime_dist": regime_dist.to_dict(),
        "final_value": round(values[-1]),
    }
    return result


def print_comparison(std_result: dict, pred_result: dict):
    """결과 비교 출력."""
    print(f"\n{'='*70}")
    print(f"📊 ETF 3축 백테스트 결과: Standard vs Predator")
    print(f"{'='*70}")

    metrics = [
        ("수익률", "total_return_pct", "%", True),
        ("MDD", "mdd_pct", "%", False),
        ("Profit Factor", "pf", "", True),
        ("승률", "win_rate_pct", "%", True),
        ("샤프 비율", "sharpe", "", True),
        ("총 거래수", "total_trades", "건", False),
        ("승리", "wins", "건", True),
        ("패배", "losses", "건", False),
        ("평균 수익", "avg_profit_pct", "%", True),
        ("평균 손실", "avg_loss_pct", "%", False),
        ("손절 횟수", "stop_losses", "건", False),
        ("매수 횟수", "buy_trades", "건", False),
        ("최종 평가액", "final_value", "원", True),
    ]

    print(f"\n{'메트릭':<16} {'Standard':>14} {'Predator':>14} {'차이':>12} {'우위':>6}")
    print(f"{'─'*16} {'─'*14} {'─'*14} {'─'*12} {'─'*6}")

    for name, key, unit, higher_better in metrics:
        s_val = std_result.get(key, 0)
        p_val = pred_result.get(key, 0)
        diff = p_val - s_val

        if key == "final_value":
            s_str = f"{s_val:>12,}{unit}"
            p_str = f"{p_val:>12,}{unit}"
            d_str = f"{diff:>+10,}{unit}"
        else:
            s_str = f"{s_val:>12.2f}{unit}"
            p_str = f"{p_val:>12.2f}{unit}"
            d_str = f"{diff:>+10.2f}{unit}"

        if higher_better:
            winner = "🟢P" if diff > 0 else "🔵S" if diff < 0 else "─"
        else:
            winner = "🟢P" if diff < 0 else "🔵S" if diff > 0 else "─"

        # MDD, 손실은 작을수록 좋음 (음수 방향)
        if key == "mdd_pct":
            winner = "🟢P" if p_val > s_val else "🔵S" if p_val < s_val else "─"

        print(f"{name:<16} {s_str:>14} {p_str:>14} {d_str:>12} {winner:>6}")

    # 레짐 분포
    print(f"\n📈 레짐 분포:")
    for regime in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
        s_pct = std_result.get("regime_dist", {}).get(regime, 0)
        print(f"  {regime}: {s_pct:.1f}%")

    # 결론
    print(f"\n{'='*70}")
    s_score = 0
    p_score = 0
    for name, key, unit, higher_better in metrics[:5]:  # 핵심 5개 지표만
        s_val = std_result.get(key, 0)
        p_val = pred_result.get(key, 0)
        if key == "mdd_pct":
            if p_val > s_val:
                p_score += 1
            else:
                s_score += 1
        elif higher_better:
            if p_val > s_val:
                p_score += 1
            else:
                s_score += 1
        else:
            if p_val < s_val:
                p_score += 1
            else:
                s_score += 1

    if p_score > s_score:
        print(f"🏆 프레데터 승리! ({p_score}/{p_score+s_score} 핵심 지표 우위)")
    elif s_score > p_score:
        print(f"🏆 스탠다드 승리! ({s_score}/{p_score+s_score} 핵심 지표 우위)")
    else:
        print(f"🤝 무승부 ({p_score} vs {s_score})")
    print(f"{'='*70}")


def print_trade_log(sim: PortfolioSimulator, mode: str, last_n: int = 20):
    """거래 로그 출력."""
    print(f"\n📋 {mode.upper()} 최근 {last_n}건 거래:")
    for t in sim.trades[-last_n:]:
        emoji = {"BUY": "🟢", "SELL": "🔴", "STOP_LOSS": "🚨"}.get(t["action"], "⚪")
        print(f"  {emoji} {t['date']} {t['action']:<10} {t['sector']:<8} "
              f"{t['price']:>10,.0f}원 ×{t['shares']:>4} = {t['value']:>12,.0f}원 | {t['reason']}")


# ============================================================
# 메인
# ============================================================

def main():
    print("🔄 데이터 로딩...")
    prices = load_etf_prices()
    sector_map = load_sector_map()
    momentum_hist = load_momentum_history()
    kospi = load_kospi()

    print(f"  ETF: {len(prices)}종목")
    print(f"  섹터: {len(sector_map)}개")
    print(f"  모멘텀 기간: {momentum_hist.index.min().date()} ~ {momentum_hist.index.max().date()}")
    print(f"  KOSPI: {kospi.index[0].date()} ~ {kospi.index[-1].date()}")

    # Standard 모드
    sim_standard = run_backtest("standard", prices, sector_map, momentum_hist, kospi)
    std_result = analyze_results(sim_standard, "standard")

    # Predator 모드
    sim_predator = run_backtest("predator", prices, sector_map, momentum_hist, kospi)
    pred_result = analyze_results(sim_predator, "predator")

    # 비교 출력
    print_comparison(std_result, pred_result)

    # 거래 로그
    if sim_standard:
        print_trade_log(sim_standard, "standard")
    if sim_predator:
        print_trade_log(sim_predator, "predator")

    # 결과 저장
    result_path = PROJECT_ROOT / "data" / "etf_backtest_predator.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "standard": std_result,
            "predator": pred_result,
            "backtest_date": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과 저장: {result_path}")


if __name__ == "__main__":
    main()
