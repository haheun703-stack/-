"""Alpha Engine V2 — STEP 6-2: V1 vs V2(5축) vs V2(4팩터) 3-Way 비교

3가지 스코어링 방식을 동일 조건에서 비교:
  V1:     기존 고정 가중치 (S1=0.30, S2=0.20, S3=0.20, S4=0.15, S5=0.15)
  V2_5ax: 레짐별 5축 재가중치 (STEP 2)
  V2_4f:  4팩터 통합 SD+M+V+Q (STEP 6-1)

실행:
  python -u -X utf8 scripts/v2_unified_backtest.py

출력:
  data/v2_migration/v2_vs_v8_comparison.json
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alpha.factors.regime_weighted_scorer import RegimeWeightedScorer
from src.alpha.factors.unified_scorer import UnifiedV2Scorer
from src.alpha.models import AlphaRegimeLevel
from src.regime_gate import RegimeGate
from src.v8_scorers import ScoringEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_GATE_TO_ALPHA = {
    "favorable": AlphaRegimeLevel.BULL,
    "neutral": AlphaRegimeLevel.CAUTION,
    "caution": AlphaRegimeLevel.BEAR,
    "hostile": AlphaRegimeLevel.CRISIS,
}


@dataclass
class Pos:
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    atr: float
    highest: float = 0.0
    hold_days: int = 0


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    hold_days: int
    exit_reason: str


class ThreeWayBacktester:
    """V1 vs V2(5축) vs V2(4팩터) 비교"""

    def __init__(self, config: dict):
        bt = config["backtest"]
        self.initial_capital = bt["initial_capital"]
        self.max_positions = bt["max_positions"]
        self.commission = bt["commission_rate"]
        self.slippage = bt["slippage_rate"]
        self.tax = bt.get("tax_rate", 0.0018)

        exit_cfg = config.get("quant_engine", {}).get("exit", {})
        self.max_hold = exit_cfg.get("max_hold_days", 10)
        self.stop_mult = bt["trailing_stop_atr_mult"]

        self.v1_scorer = ScoringEngine(config)
        self.v2_5ax_scorer = RegimeWeightedScorer(config)
        self.v2_4f_scorer = UnifiedV2Scorer(config)
        self.regime_gate = RegimeGate(config)

        # 등급 기반 필터: tradeable (B 이상) 사용
        self.use_grade_filter = True

    def run(self, data_dict: dict, mode: str = "v1") -> dict:
        """mode: 'v1' | 'v2_5ax' | 'v2_4f'"""
        self.cash = self.initial_capital
        positions: list[Pos] = []
        trades: list[Trade] = []
        equity: list[float] = []
        regime_counts = {"BULL": 0, "CAUTION": 0, "BEAR": 0, "CRISIS": 0}

        first_ticker = list(data_dict.keys())[0]
        all_dates = data_dict[first_ticker].index
        start_idx = 200
        end_idx = len(all_dates)

        for idx in tqdm(range(start_idx, end_idx), desc=f"  {mode}", leave=False):
            date_str = str(all_dates[idx].date())

            # 레짐 판정
            regime_state = self.regime_gate.detect(data_dict, idx)
            alpha_level = _GATE_TO_ALPHA.get(
                regime_state.regime, AlphaRegimeLevel.CAUTION
            )
            regime_counts[alpha_level.value] += 1

            # 포지션 관리
            for pos in list(positions):
                df = data_dict.get(pos.ticker)
                if df is None or idx >= len(df):
                    continue

                row = df.iloc[idx]
                close, high, low = row["close"], row["high"], row["low"]
                pos.hold_days += 1

                if high > pos.highest:
                    pos.highest = high

                # 손절
                if low <= pos.stop_loss:
                    self._sell(pos, pos.stop_loss, date_str, "stop", positions, trades)
                    continue

                if close / pos.entry_price - 1 <= -0.08:
                    self._sell(pos, close, date_str, "pct_stop", positions, trades)
                    continue

                if pos.hold_days >= self.max_hold:
                    self._sell(pos, close, date_str, "max_hold", positions, trades)
                    continue

                trailing = pos.highest - pos.atr * self.stop_mult
                if close <= trailing and close > pos.entry_price:
                    self._sell(pos, close, date_str, "trail", positions, trades)
                    continue

            # CRISIS → 매수 금지
            if alpha_level == AlphaRegimeLevel.CRISIS:
                equity.append(self._calc_equity(positions, data_dict, idx))
                continue

            # 레짐별 최대 포지션
            regime_max = {
                AlphaRegimeLevel.BULL: self.max_positions,
                AlphaRegimeLevel.CAUTION: max(1, self.max_positions - 2),
                AlphaRegimeLevel.BEAR: max(1, self.max_positions - 3),
            }.get(alpha_level, self.max_positions)

            # 매수
            if len(positions) < regime_max and idx + 1 < end_idx:
                held = {p.ticker for p in positions}
                candidates = []

                for ticker, df in data_dict.items():
                    if ticker in held or idx >= len(df) or idx + 1 >= len(df):
                        continue

                    row = df.iloc[idx]
                    atr = row.get("atr_14", 0)
                    if atr <= 0 or pd.isna(atr):
                        continue

                    vol_20 = df["volume"].iloc[max(0, idx - 19):idx + 1].mean()
                    close = row["close"]
                    if close * vol_20 < 500_000_000:
                        continue

                    # 스코어링
                    if mode == "v2_4f":
                        gr = self.v2_4f_scorer.score(row, ticker, alpha_level)
                    elif mode == "v2_5ax":
                        gr = self.v2_5ax_scorer.score(row, alpha_level)
                    else:
                        gr = self.v1_scorer.score_all(row)

                    if gr.tradeable:  # B 등급 이상만 매수
                        candidates.append({
                            "ticker": ticker, "score": gr.total_score,
                            "grade": gr.grade, "atr": atr,
                            "next_open": df["open"].iloc[idx + 1],
                        })

                candidates.sort(key=lambda x: x["score"], reverse=True)

                for c in candidates:
                    if len(positions) >= regime_max:
                        break

                    entry = c["next_open"] * (1 + self.slippage)
                    stop = entry - c["atr"] * self.stop_mult

                    risk_amt = self.cash * 0.01
                    stop_dist = c["atr"] * self.stop_mult
                    if stop_dist <= 0:
                        continue
                    shares = int(risk_amt / stop_dist)

                    max_invest = self.cash * 0.20
                    if shares * entry > max_invest:
                        shares = int(max_invest / entry)
                    if shares <= 0:
                        continue

                    cost = entry * shares * (1 + self.commission)
                    if cost > self.cash:
                        shares = int(self.cash / (entry * (1 + self.commission)))
                        if shares <= 0:
                            continue
                        cost = entry * shares * (1 + self.commission)

                    self.cash -= cost
                    positions.append(Pos(
                        ticker=c["ticker"], entry_date=date_str,
                        entry_price=entry, shares=shares,
                        stop_loss=stop, atr=c["atr"], highest=entry,
                    ))

            equity.append(self._calc_equity(positions, data_dict, idx))

        # 잔여 청산
        if positions:
            last_date = str(all_dates[end_idx - 1].date())
            for pos in list(positions):
                df = data_dict.get(pos.ticker)
                if df is not None and end_idx - 1 < len(df):
                    self._sell(pos, df["close"].iloc[end_idx - 1], last_date,
                               "end", positions, trades)

        return self._metrics(mode, trades, equity, regime_counts)

    def _calc_equity(self, positions, data_dict, idx):
        hv = sum(
            data_dict[p.ticker]["close"].iloc[idx] * p.shares
            for p in positions
            if p.ticker in data_dict and idx < len(data_dict[p.ticker])
        )
        return self.cash + hv

    def _sell(self, pos, price, date, reason, positions, trades):
        actual = price * (1 - self.slippage)
        sell_comm = actual * pos.shares * self.commission
        buy_comm = pos.entry_price * pos.shares * self.commission
        tax = actual * pos.shares * self.tax
        net_cost = sell_comm + buy_comm + tax
        pnl = ((actual - pos.entry_price) / pos.entry_price * 100
               - net_cost / (pos.entry_price * pos.shares) * 100)
        self.cash += actual * pos.shares - sell_comm - tax
        trades.append(Trade(
            ticker=pos.ticker, entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=actual, shares=pos.shares,
            pnl_pct=round(pnl, 2), hold_days=pos.hold_days, exit_reason=reason,
        ))
        if pos in positions:
            positions.remove(pos)

    def _metrics(self, name, trades, equity, regime_counts):
        res = {"name": name, "trades": len(trades), "regime": regime_counts}
        if not trades:
            res.update(sharpe=0, pf=0, mdd=0, win_rate=0, total_return=0,
                       avg_hold=0, avg_win=0, avg_loss=0)
            return res

        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        res["win_rate"] = round(len(wins) / len(pnls) * 100, 1)
        res["avg_win"] = round(np.mean(wins), 2) if wins else 0
        res["avg_loss"] = round(np.mean(losses), 2) if losses else 0
        res["avg_hold"] = round(np.mean([t.hold_days for t in trades]), 1)

        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 0.001
        res["pf"] = round(gp / gl, 2)

        if equity:
            eq = pd.Series(equity, dtype=float)
            res["total_return"] = round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2)
            dr = eq.pct_change().dropna()
            if len(dr) > 1 and dr.std() > 0:
                res["sharpe"] = round(dr.mean() / dr.std() * np.sqrt(252), 2)
            else:
                res["sharpe"] = 0
            peak = eq.cummax()
            dd = (eq - peak) / peak
            res["mdd"] = round(dd.min() * 100, 2)
        else:
            res.update(sharpe=0, mdd=0, total_return=0)

        return res


def load_data() -> dict:
    d = {}
    for f in sorted((PROJECT_ROOT / "data" / "processed").glob("*.parquet")):
        df = pd.read_parquet(f)
        if len(df) > 200:
            d[f.stem] = df
    return d


def main():
    logger.info("=" * 70)
    logger.info("Alpha Engine V2 — STEP 6-2: 3-Way 비교 (V1 vs V2_5ax vs V2_4f)")
    logger.info("=" * 70)

    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("데이터 로딩...")
    data_dict = load_data()
    logger.info("  %d종목 로딩 완료", len(data_dict))

    bt = ThreeWayBacktester(config)

    # V1
    logger.info("\n--- V1 (기존 고정 가중치) ---")
    r_v1 = bt.run(data_dict, mode="v1")

    # V2 5축
    logger.info("\n--- V2_5ax (레짐별 5축 재가중치) ---")
    r_v2_5ax = bt.run(data_dict, mode="v2_5ax")

    # V2 4팩터
    logger.info("\n--- V2_4f (4팩터 통합: SD+M+V+Q) ---")
    r_v2_4f = bt.run(data_dict, mode="v2_4f")

    # 결과 저장
    output_dir = PROJECT_ROOT / "data" / "v2_migration"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {"v1_fixed": r_v1, "v2_5axis": r_v2_5ax, "v2_4factor": r_v2_4f}

    with open(output_dir / "v2_vs_v8_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 비교 테이블
    print("\n" + "=" * 80)
    print("  V1 vs V2(5축) vs V2(4팩터) 비교 결과")
    print("=" * 80)
    print(f"  {'지표':<12} {'V1(기존)':>12} {'V2(5축)':>12} {'V2(4팩터)':>12}  {'V1→4f':>10}")
    print("  " + "-" * 65)

    metrics = ["sharpe", "pf", "mdd", "win_rate", "trades", "total_return", "avg_hold"]
    labels = ["Sharpe", "PF", "MDD(%)", "승률(%)", "거래(건)", "수익률(%)", "보유(일)"]

    for label, m in zip(labels, metrics):
        v1 = r_v1.get(m, 0)
        v5 = r_v2_5ax.get(m, 0)
        v4 = r_v2_4f.get(m, 0)
        diff = v4 - v1
        sign = "+" if diff > 0 else ""
        print(f"  {label:<12} {v1:>12.2f} {v5:>12.2f} {v4:>12.2f}  {sign}{diff:>9.2f}")

    print("=" * 80)

    # 판정
    v2_pass = (
        r_v2_4f.get("pf", 0) >= 1.3
        and r_v2_4f.get("mdd", -999) >= -15.0
    )

    v2_better = (
        r_v2_4f.get("pf", 0) >= r_v1.get("pf", 0)
        and r_v2_4f.get("mdd", -999) >= r_v1.get("mdd", -999)
    )

    print(f"\n  절대 기준: PF >= 1.3 AND MDD >= -15% → {'PASS' if v2_pass else 'FAIL'}")
    print(f"  상대 기준: V2 >= V1 (PF+MDD) → {'V2 우위' if v2_better else 'V1 유지'}")

    logger.info("\n저장: %s", output_dir / "v2_vs_v8_comparison.json")


if __name__ == "__main__":
    main()
