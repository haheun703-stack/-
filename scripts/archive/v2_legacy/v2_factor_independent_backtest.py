"""
Alpha Engine V2 — STEP 1: 기존 5축 독립 팩터 백테스트

목적: S1~S5 각 축의 독립적 엣지를 검증한다.
      이 결과가 나와야 V2 가중치를 데이터로 결정할 수 있다.

로직:
  - 84종목 parquet 데이터 로딩
  - 매일 전종목에 대해 5개 스코어러 각각 계산
  - 단일 팩터 모드: 해당 팩터 점수 상위 N종목 매수
  - 합산 모드 (기준선): 가중합 점수 상위 N종목 매수
  - 공통 청산: ATR 트레일링 + 고정 손절 + 최대 보유일
  - 출력: 각 축별 Sharpe, PF, MDD, 승률, 거래횟수

실행:
  python -u -X utf8 scripts/v2_factor_independent_backtest.py
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

from src.v8_scorers import ScoringEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 데이터 모델
# ═══════════════════════════════════════════════════════

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
    factor_score: float = 0.0


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


@dataclass
class FactorResult:
    name: str
    sharpe: float = 0.0
    pf: float = 0.0
    mdd: float = 0.0
    win_rate: float = 0.0
    trades: int = 0
    total_return: float = 0.0
    avg_hold_days: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


# ═══════════════════════════════════════════════════════
# 경량 백테스트 엔진
# ═══════════════════════════════════════════════════════

class FactorBacktester:
    """단일 팩터 독립 백테스트 엔진"""

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

        self.scorer = ScoringEngine(config)

        # 매수 임계값: 스코어 >= threshold인 종목만 후보
        self.score_threshold = 0.35  # C등급 이상

    def run(
        self,
        data_dict: dict,
        factor_name: str,
        factor_index: int | None = None,
    ) -> FactorResult:
        """
        단일 팩터 백테스트 실행.

        Args:
            data_dict: {ticker: DataFrame}
            factor_name: 팩터 이름
            factor_index: 0~4 (None이면 합산 모드)
        """
        self.cash = self.initial_capital
        positions: list[SimplePosition] = []
        trades: list[SimpleTrade] = []
        equity_curve: list[float] = []

        first_ticker = list(data_dict.keys())[0]
        all_dates = data_dict[first_ticker].index
        start_idx = 200
        end_idx = len(all_dates)

        for idx in tqdm(
            range(start_idx, end_idx),
            desc=f"  {factor_name}",
            leave=False,
        ):
            date_str = str(all_dates[idx].date())

            # ── 1. 포지션 관리 ──
            for pos in list(positions):
                df = data_dict.get(pos.ticker)
                if df is None or idx >= len(df):
                    continue

                row = df.iloc[idx]
                close = row["close"]
                high = row["high"]
                low = row["low"]
                pos.hold_days += 1

                # 최고가 갱신
                if high > pos.highest_price:
                    pos.highest_price = high

                # 1a. 절대 손절 (ATR 기반)
                if low <= pos.stop_loss:
                    self._sell(pos, pos.stop_loss, date_str, "stop_loss",
                               positions, trades)
                    continue

                # 1b. 고정 퍼센트 손절 (-8%)
                pct_loss = close / pos.entry_price - 1
                if pct_loss <= -0.08:
                    self._sell(pos, close, date_str, "pct_stop",
                               positions, trades)
                    continue

                # 1c. 최대 보유일 초과
                if pos.hold_days >= self.max_hold_days:
                    self._sell(pos, close, date_str, "max_hold",
                               positions, trades)
                    continue

                # 1d. 트레일링 스탑 (최고가 - ATR*mult, 수익 구간만)
                trailing = pos.highest_price - pos.atr_value * self.atr_stop_mult
                if close <= trailing and pct_loss > 0:
                    self._sell(pos, close, date_str, "trailing",
                               positions, trades)
                    continue

            # ── 2. 신규 매수 ──
            if len(positions) < self.max_positions and idx + 1 < end_idx:
                held = {p.ticker for p in positions}
                candidates = []

                for ticker, df in data_dict.items():
                    if ticker in held or idx >= len(df) or idx + 1 >= len(df):
                        continue

                    row = df.iloc[idx]

                    # ATR 확인
                    atr = row.get("atr_14", 0)
                    if atr <= 0 or pd.isna(atr):
                        continue

                    # 유동성 필터: 20일 평균 거래대금 > 5억
                    vol_20 = df["volume"].iloc[max(0, idx - 19):idx + 1].mean()
                    close = row["close"]
                    if close * vol_20 < 500_000_000:
                        continue

                    # 스코어 계산
                    score_result = self.scorer.score_all(row)

                    if factor_index is not None:
                        # 단일 팩터 모드: 해당 팩터 raw score만 사용
                        factor_score = score_result.scores[factor_index].score
                    else:
                        # 합산 모드: 가중합 사용
                        factor_score = score_result.total_score

                    if factor_score >= self.score_threshold:
                        candidates.append({
                            "ticker": ticker,
                            "score": factor_score,
                            "close": close,
                            "atr": atr,
                            "next_open": df["open"].iloc[idx + 1],
                        })

                # 점수 높은 순 정렬
                candidates.sort(key=lambda x: x["score"], reverse=True)

                for c in candidates:
                    if len(positions) >= self.max_positions:
                        break

                    entry_price = c["next_open"] * (1 + self.slippage_rate)
                    stop_loss = entry_price - c["atr"] * self.atr_stop_mult

                    # ATR 기반 포지션 사이징 (리스크 1%)
                    risk_amount = self.cash * 0.01
                    stop_dist = c["atr"] * self.atr_stop_mult
                    if stop_dist <= 0:
                        continue
                    shares = int(risk_amount / stop_dist)

                    # 최대 단일 종목 20%
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
                        factor_score=c["score"],
                    ))

            # ── 3. 에쿼티 기록 ──
            holdings_value = sum(
                data_dict[p.ticker]["close"].iloc[idx] * p.shares
                for p in positions
                if p.ticker in data_dict and idx < len(data_dict[p.ticker])
            )
            equity_curve.append(self.cash + holdings_value)

        # 잔여 포지션 강제 청산
        if positions:
            last_date = str(all_dates[end_idx - 1].date())
            for pos in list(positions):
                df = data_dict.get(pos.ticker)
                if df is not None and end_idx - 1 < len(df):
                    close = df["close"].iloc[end_idx - 1]
                    self._sell(pos, close, last_date, "end_of_test",
                               positions, trades)

        return self._calc_metrics(factor_name, trades, equity_curve)

    def _sell(
        self,
        pos: SimplePosition,
        exit_price: float,
        date_str: str,
        reason: str,
        positions: list,
        trades: list,
    ):
        """매도 실행 (수수료+세금 반영)"""
        actual_price = exit_price * (1 - self.slippage_rate)
        sell_comm = actual_price * pos.shares * self.commission_rate
        buy_comm = pos.entry_price * pos.shares * self.commission_rate
        tax = actual_price * pos.shares * self.tax_rate
        net_cost = sell_comm + buy_comm + tax
        pnl_pct = ((actual_price - pos.entry_price) / pos.entry_price * 100
                    - net_cost / (pos.entry_price * pos.shares) * 100)

        proceeds = actual_price * pos.shares - sell_comm - tax
        self.cash += proceeds

        trades.append(SimpleTrade(
            ticker=pos.ticker,
            entry_date=pos.entry_date,
            exit_date=date_str,
            entry_price=pos.entry_price,
            exit_price=actual_price,
            shares=pos.shares,
            pnl_pct=round(pnl_pct, 2),
            hold_days=pos.hold_days,
            exit_reason=reason,
        ))

        if pos in positions:
            positions.remove(pos)

    def _calc_metrics(
        self,
        name: str,
        trades: list[SimpleTrade],
        equity_curve: list[float],
    ) -> FactorResult:
        """성과 지표 계산"""
        result = FactorResult(name=name, trades=len(trades))

        if not trades:
            return result

        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        result.avg_hold_days = np.mean([t.hold_days for t in trades])

        # Profit Factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        result.pf = round(gross_profit / gross_loss, 2)

        # Equity curve metrics
        if equity_curve:
            eq = pd.Series(equity_curve, dtype=float)
            result.total_return = round(
                (eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2
            )

            # Daily returns → Sharpe
            daily_ret = eq.pct_change().dropna()
            if len(daily_ret) > 1 and daily_ret.std() > 0:
                result.sharpe = round(
                    daily_ret.mean() / daily_ret.std() * np.sqrt(252), 2
                )

            # MDD
            peak = eq.cummax()
            dd = (eq - peak) / peak
            result.mdd = round(dd.min() * 100, 2)

        return result


# ═══════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════

def load_data() -> dict:
    """processed 디렉토리에서 전종목 데이터 로딩"""
    processed_dir = Path("data/processed")
    data = {}
    for fpath in sorted(processed_dir.glob("*.parquet")):
        df = pd.read_parquet(fpath)
        if len(df) > 200:
            data[fpath.stem] = df
    return data


def main():
    logger.info("=" * 60)
    logger.info("Alpha Engine V2 — STEP 1: 독립 팩터 백테스트")
    logger.info("=" * 60)

    with open("config/settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("데이터 로딩 중...")
    data_dict = load_data()
    logger.info(f"  {len(data_dict)}종목 로딩 완료")

    # 6가지 테스트 정의
    tests = [
        ("S1_energy_exhaustion", 0),
        ("S2_valuation", 1),
        ("S3_ou_reversion", 2),
        ("S4_momentum_decel", 3),
        ("S5_supply_demand", 4),
        ("current_combined", None),
    ]

    results = {}
    bt = FactorBacktester(config)

    for test_name, factor_idx in tests:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"테스트: {test_name} (factor_idx={factor_idx})")
        logger.info(f"{'─' * 50}")

        result = bt.run(data_dict, test_name, factor_idx)

        results[test_name] = {
            "sharpe": result.sharpe,
            "pf": result.pf,
            "mdd": result.mdd,
            "win_rate": round(result.win_rate, 1),
            "trades": result.trades,
            "total_return": result.total_return,
            "avg_hold_days": round(result.avg_hold_days, 1),
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
        }

        logger.info(
            f"  결과: Sharpe={result.sharpe} | PF={result.pf} | "
            f"MDD={result.mdd}% | 승률={result.win_rate:.1f}% | "
            f"거래={result.trades}건 | 수익={result.total_return}%"
        )

    # 결과 저장
    output_dir = Path("data/v2_migration")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "factor_independent_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n결과 저장: {output_path}")

    # 요약 테이블
    print("\n" + "=" * 82)
    print("  독립 팩터 백테스트 결과 요약")
    print("=" * 82)
    print(
        f"  {'팩터':<25} {'Sharpe':>7} {'PF':>6} {'MDD':>8} "
        f"{'승률':>7} {'거래':>6} {'수익률':>8}"
    )
    print("  " + "-" * 78)

    for name, r in results.items():
        print(
            f"  {name:<25} {r['sharpe']:>7.2f} {r['pf']:>6.2f} "
            f"{r['mdd']:>7.2f}% {r['win_rate']:>6.1f}% "
            f"{r['trades']:>5}건 {r['total_return']:>7.2f}%"
        )

    print("=" * 82)

    # 핵심 인사이트
    print("\n  핵심 인사이트:")

    sorted_by_sharpe = sorted(
        results.items(), key=lambda x: x[1]["sharpe"], reverse=True
    )
    print(f"  - 최고 Sharpe: {sorted_by_sharpe[0][0]} ({sorted_by_sharpe[0][1]['sharpe']})")
    print(f"  - 최저 Sharpe: {sorted_by_sharpe[-1][0]} ({sorted_by_sharpe[-1][1]['sharpe']})")

    combined = results.get("current_combined", {})
    for name, r in results.items():
        if name != "current_combined":
            if r["pf"] >= 1.3:
                print(f"  - {name}: PF {r['pf']} >= 1.3 — 독립적 엣지 확인")
            if r["sharpe"] > combined.get("sharpe", 0):
                print(
                    f"  - {name}: 단독 Sharpe({r['sharpe']}) > "
                    f"합산({combined.get('sharpe', 0)}) — 가중치 상향 근거"
                )


if __name__ == "__main__":
    main()
