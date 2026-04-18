"""
Alpha Engine V2 — STEP 2-3: 레짐별 가중치 A/B 비교 백테스트

V1 (기존 고정 가중치) vs V2 (레짐별 동적 가중치) 비교.

V1: S1=0.30, S2=0.20, S3=0.20, S4=0.15, S5=0.15 (고정)
V2: 레짐(BULL/CAUTION/BEAR/CRISIS)별 Sharpe 비율 기반 가중치

실행:
  python -u -X utf8 scripts/v2_regime_weight_backtest.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alpha.factors.regime_weighted_scorer import RegimeWeightedScorer  # noqa: E402
from src.alpha.models import AlphaRegimeLevel  # noqa: E402
from src.regime_gate import RegimeGate  # noqa: E402
from src.v8_scorers import ScoringEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# RegimeGate → Alpha 레짐 매핑
_GATE_TO_ALPHA = {
    "favorable": AlphaRegimeLevel.BULL,
    "neutral": AlphaRegimeLevel.CAUTION,
    "caution": AlphaRegimeLevel.BEAR,
    "hostile": AlphaRegimeLevel.CRISIS,
}


@dataclass
class SimplePosition:
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    atr_value: float
    highest_price: float = 0.0
    hold_days: int = 0


@dataclass
class SimpleTrade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    hold_days: int
    exit_reason: str


class ABBacktester:
    """V1 vs V2 A/B 비교 백테스트"""

    def __init__(self, config: dict):
        bt = config["backtest"]
        self.initial_capital = bt["initial_capital"]
        self.max_positions = bt["max_positions"]
        self.commission_rate = bt["commission_rate"]
        self.slippage_rate = bt["slippage_rate"]
        self.tax_rate = bt.get("tax_rate", 0.0018)

        exit_cfg = config.get("quant_engine", {}).get("exit", {})
        self.max_hold_days = exit_cfg.get("max_hold_days", 10)
        self.atr_stop_mult = bt["trailing_stop_atr_mult"]

        self.v1_scorer = ScoringEngine(config)
        self.v2_scorer = RegimeWeightedScorer(config)
        self.regime_gate = RegimeGate(config)
        self.score_threshold = 0.45  # 0.35→0.45: V2 품질 희석 방지

    def run(self, data_dict: dict, use_v2: bool = False) -> dict:
        """
        백테스트 실행.

        Args:
            data_dict: {ticker: DataFrame}
            use_v2: True면 V2 레짐별 가중치, False면 V1 고정
        """
        label = "V2_regime" if use_v2 else "V1_fixed"
        self.cash = self.initial_capital
        positions: list[SimplePosition] = []
        trades: list[SimpleTrade] = []
        equity_curve: list[float] = []
        regime_counts = {"BULL": 0, "CAUTION": 0, "BEAR": 0, "CRISIS": 0}

        first_ticker = list(data_dict.keys())[0]
        all_dates = data_dict[first_ticker].index
        start_idx = 200
        end_idx = len(all_dates)

        for idx in tqdm(range(start_idx, end_idx), desc=f"  {label}", leave=False):
            date_str = str(all_dates[idx].date())

            # ── 0. 레짐 판정 ──
            regime_state = self.regime_gate.detect(data_dict, idx)
            alpha_level = _GATE_TO_ALPHA.get(
                regime_state.regime, AlphaRegimeLevel.CAUTION
            )
            regime_counts[alpha_level.value] += 1

            # ── 1. 포지션 관리 ──
            for pos in list(positions):
                df = data_dict.get(pos.ticker)
                if df is None or idx >= len(df):
                    continue

                row = df.iloc[idx]
                close, high, low = row["close"], row["high"], row["low"]
                pos.hold_days += 1

                if high > pos.highest_price:
                    pos.highest_price = high

                # 손절
                if low <= pos.stop_loss:
                    self._sell(pos, pos.stop_loss, date_str, "stop_loss",
                               positions, trades)
                    continue

                pct_loss = close / pos.entry_price - 1
                if pct_loss <= -0.08:
                    self._sell(pos, close, date_str, "pct_stop",
                               positions, trades)
                    continue

                if pos.hold_days >= self.max_hold_days:
                    self._sell(pos, close, date_str, "max_hold",
                               positions, trades)
                    continue

                trailing = pos.highest_price - pos.atr_value * self.atr_stop_mult
                if close <= trailing and pct_loss > 0:
                    self._sell(pos, close, date_str, "trailing",
                               positions, trades)
                    continue

            # ── 2. CRISIS → 신규 매수 금지 ──
            if alpha_level == AlphaRegimeLevel.CRISIS:
                holdings_value = sum(
                    data_dict[p.ticker]["close"].iloc[idx] * p.shares
                    for p in positions
                    if p.ticker in data_dict and idx < len(data_dict[p.ticker])
                )
                equity_curve.append(self.cash + holdings_value)
                continue

            # ── 3. 레짐별 최대 포지션 ──
            regime_max = {
                AlphaRegimeLevel.BULL: self.max_positions,
                AlphaRegimeLevel.CAUTION: max(1, self.max_positions - 2),
                AlphaRegimeLevel.BEAR: max(1, self.max_positions - 3),
            }.get(alpha_level, self.max_positions)

            # ── 4. 신규 매수 ──
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

                    # V1 vs V2 스코어링
                    if use_v2:
                        grade_result = self.v2_scorer.score(row, alpha_level)
                    else:
                        grade_result = self.v1_scorer.score_all(row)

                    if grade_result.total_score >= self.score_threshold:
                        candidates.append({
                            "ticker": ticker,
                            "score": grade_result.total_score,
                            "atr": atr,
                            "next_open": df["open"].iloc[idx + 1],
                        })

                candidates.sort(key=lambda x: x["score"], reverse=True)

                for c in candidates:
                    if len(positions) >= regime_max:
                        break

                    entry_price = c["next_open"] * (1 + self.slippage_rate)
                    stop_loss = entry_price - c["atr"] * self.atr_stop_mult

                    risk_amount = self.cash * 0.01
                    stop_dist = c["atr"] * self.atr_stop_mult
                    if stop_dist <= 0:
                        continue
                    shares = int(risk_amount / stop_dist)

                    max_invest = self.cash * 0.20
                    if shares * entry_price > max_invest:
                        shares = int(max_invest / entry_price)

                    if shares <= 0:
                        continue

                    cost = entry_price * shares * (1 + self.commission_rate)
                    if cost > self.cash:
                        shares = int(
                            self.cash / (entry_price * (1 + self.commission_rate))
                        )
                        if shares <= 0:
                            continue
                        cost = entry_price * shares * (1 + self.commission_rate)

                    self.cash -= cost
                    positions.append(SimplePosition(
                        ticker=c["ticker"],
                        entry_date=date_str,
                        entry_price=entry_price,
                        shares=shares,
                        stop_loss=stop_loss,
                        atr_value=c["atr"],
                        highest_price=entry_price,
                    ))

            # 에쿼티
            holdings_value = sum(
                data_dict[p.ticker]["close"].iloc[idx] * p.shares
                for p in positions
                if p.ticker in data_dict and idx < len(data_dict[p.ticker])
            )
            equity_curve.append(self.cash + holdings_value)

        # 잔여 강제 청산
        if positions:
            last_date = str(all_dates[end_idx - 1].date())
            for pos in list(positions):
                df = data_dict.get(pos.ticker)
                if df is not None and end_idx - 1 < len(df):
                    self._sell(pos, df["close"].iloc[end_idx - 1], last_date,
                               "end_of_test", positions, trades)

        return self._calc_metrics(label, trades, equity_curve, regime_counts)

    def _sell(self, pos, exit_price, date_str, reason, positions, trades):
        actual_price = exit_price * (1 - self.slippage_rate)
        sell_comm = actual_price * pos.shares * self.commission_rate
        buy_comm = pos.entry_price * pos.shares * self.commission_rate
        tax = actual_price * pos.shares * self.tax_rate
        net_cost = sell_comm + buy_comm + tax
        pnl_pct = ((actual_price - pos.entry_price) / pos.entry_price * 100
                    - net_cost / (pos.entry_price * pos.shares) * 100)

        self.cash += actual_price * pos.shares - sell_comm - tax

        trades.append(SimpleTrade(
            ticker=pos.ticker, entry_date=pos.entry_date, exit_date=date_str,
            entry_price=pos.entry_price, exit_price=actual_price,
            shares=pos.shares, pnl_pct=round(pnl_pct, 2),
            hold_days=pos.hold_days, exit_reason=reason,
        ))
        if pos in positions:
            positions.remove(pos)

    def _calc_metrics(self, name, trades, equity_curve, regime_counts):
        result = {
            "name": name,
            "trades": len(trades),
            "regime_distribution": regime_counts,
        }

        if not trades:
            result.update(sharpe=0, pf=0, mdd=0, win_rate=0,
                          total_return=0, avg_hold_days=0)
            return result

        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result["win_rate"] = round(len(wins) / len(pnls) * 100, 1)
        result["avg_win"] = round(np.mean(wins), 2) if wins else 0
        result["avg_loss"] = round(np.mean(losses), 2) if losses else 0
        result["avg_hold_days"] = round(np.mean([t.hold_days for t in trades]), 1)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        result["pf"] = round(gross_profit / gross_loss, 2)

        if equity_curve:
            eq = pd.Series(equity_curve, dtype=float)
            result["total_return"] = round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2)

            daily_ret = eq.pct_change().dropna()
            if len(daily_ret) > 1 and daily_ret.std() > 0:
                result["sharpe"] = round(
                    daily_ret.mean() / daily_ret.std() * np.sqrt(252), 2
                )
            else:
                result["sharpe"] = 0

            peak = eq.cummax()
            dd = (eq - peak) / peak
            result["mdd"] = round(dd.min() * 100, 2)
        else:
            result.update(sharpe=0, mdd=0, total_return=0)

        return result


def load_data() -> dict:
    processed_dir = Path("data/processed")
    data = {}
    for fpath in sorted(processed_dir.glob("*.parquet")):
        df = pd.read_parquet(fpath)
        if len(df) > 200:
            data[fpath.stem] = df
    return data


def main():
    logger.info("=" * 60)
    logger.info("Alpha Engine V2 — STEP 2-3: V1 vs V2 A/B 비교")
    logger.info("=" * 60)

    with open("config/settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("데이터 로딩 중...")
    data_dict = load_data()
    logger.info(f"  {len(data_dict)}종목 로딩 완료")

    bt = ABBacktester(config)

    # V1: 기존 고정 가중치
    logger.info("\n--- V1 (고정 가중치) ---")
    v1_result = bt.run(data_dict, use_v2=False)
    logger.info(
        f"  V1: Sharpe={v1_result['sharpe']} PF={v1_result['pf']} "
        f"MDD={v1_result['mdd']}% 거래={v1_result['trades']}건 "
        f"수익={v1_result['total_return']}%"
    )

    # V2: 레짐별 동적 가중치
    logger.info("\n--- V2 (레짐별 가중치) ---")
    v2_result = bt.run(data_dict, use_v2=True)
    logger.info(
        f"  V2: Sharpe={v2_result['sharpe']} PF={v2_result['pf']} "
        f"MDD={v2_result['mdd']}% 거래={v2_result['trades']}건 "
        f"수익={v2_result['total_return']}%"
    )

    # 결과 저장
    output_dir = Path("data/v2_migration")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {"v1_fixed": v1_result, "v2_regime": v2_result}

    with open(output_dir / "regime_weight_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 비교 테이블
    print("\n" + "=" * 70)
    print("  V1 vs V2 비교 결과")
    print("=" * 70)
    print(f"  {'지표':<15} {'V1(고정)':>12} {'V2(레짐)':>12} {'차이':>12}")
    print("  " + "-" * 55)

    metrics = ["sharpe", "pf", "mdd", "win_rate", "trades", "total_return"]
    labels = ["Sharpe", "PF", "MDD(%)", "승률(%)", "거래(건)", "수익률(%)"]

    for label, m in zip(labels, metrics):
        v1 = v1_result.get(m, 0)
        v2 = v2_result.get(m, 0)
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"  {label:<15} {v1:>12.2f} {v2:>12.2f} {sign}{diff:>11.2f}")

    print("=" * 70)

    # 판정
    v2_better = (
        v2_result.get("sharpe", 0) > v1_result.get("sharpe", 0)
        and v2_result.get("mdd", 0) > v1_result.get("mdd", 0)  # MDD는 음수, 큰게 좋음
    )

    if v2_better:
        print("\n  V2 PASS — Sharpe 개선 + MDD 개선. STEP 2-4 진행 가능.")
    else:
        print("\n  V2 NOT PASS — 추가 가중치 조정 필요.")

    # 레짐 분포
    print(f"\n  레짐 분포: {v2_result.get('regime_distribution', {})}")


if __name__ == "__main__":
    main()
