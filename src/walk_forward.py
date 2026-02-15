"""
Walk-Forward 검증 모듈

논문 기반 (데이터 스누핑 논문):
  - 18,410개 규칙 → SPA Test/StepM → 과최적화 방지
  - Train 2Y / Test 6M 롤링 → Out-of-Sample 성과 검증

구조:
  전체 기간을 Train/Test 윈도우로 분할하여
  각 구간별 독립적 백테스트 실행 후 OOS 성과 집계.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WFWindow:
    """Walk-Forward 단일 윈도우"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_trades: int = 0
    test_trades: int = 0
    train_return_pct: float = 0.0
    test_return_pct: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    efficiency_ratio: float = 0.0  # test_sharpe / train_sharpe


class WalkForwardValidator:
    """Train/Test 롤링 Walk-Forward 검증기"""

    def __init__(
        self,
        train_days: int = 504,   # 2년 (252 x 2)
        test_days: int = 126,    # 6개월
        step_days: int = 126,    # 6개월씩 이동
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.windows: list[WFWindow] = []

    def generate_windows(self, dates: pd.DatetimeIndex) -> list[WFWindow]:
        """날짜 인덱스에서 Train/Test 윈도우 생성"""
        self.windows = []
        total = len(dates)
        min_required = self.train_days + self.test_days

        if total < min_required:
            logger.warning(
                f"데이터 부족: {total}일 < 최소 {min_required}일. "
                f"Walk-Forward 불가."
            )
            return []

        window_id = 0
        start = 0

        while start + min_required <= total:
            train_start_idx = start
            train_end_idx = start + self.train_days - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_days - 1, total - 1)

            wf = WFWindow(
                window_id=window_id,
                train_start=str(dates[train_start_idx].date()),
                train_end=str(dates[train_end_idx].date()),
                test_start=str(dates[test_start_idx].date()),
                test_end=str(dates[test_end_idx].date()),
            )
            self.windows.append(wf)

            window_id += 1
            start += self.step_days

        logger.info(f"Walk-Forward: {len(self.windows)}개 윈도우 생성 "
                     f"(Train {self.train_days}일 / Test {self.test_days}일)")
        return self.windows

    def evaluate_window(
        self,
        window: WFWindow,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ):
        """단일 윈도우의 Train/Test 성과 평가"""
        equity_df = equity_df.copy()
        if "date" in equity_df.columns:
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.set_index("date")

        # Train 구간
        train_eq = equity_df.loc[
            window.train_start:window.train_end, "portfolio_value"
        ]
        # Test 구간
        test_eq = equity_df.loc[
            window.test_start:window.test_end, "portfolio_value"
        ]

        # Train 성과
        if len(train_eq) > 1:
            train_ret = train_eq.pct_change().dropna()
            window.train_return_pct = round(
                (train_eq.iloc[-1] / train_eq.iloc[0] - 1) * 100, 2
            )
            if train_ret.std() > 0:
                window.train_sharpe = round(
                    (train_ret.mean() - 0.03 / 252) / train_ret.std() * np.sqrt(252),
                    3,
                )

        # Test 성과
        if len(test_eq) > 1:
            test_ret = test_eq.pct_change().dropna()
            window.test_return_pct = round(
                (test_eq.iloc[-1] / test_eq.iloc[0] - 1) * 100, 2
            )
            if test_ret.std() > 0:
                window.test_sharpe = round(
                    (test_ret.mean() - 0.03 / 252) / test_ret.std() * np.sqrt(252),
                    3,
                )

        # 효율 비율 (OOS/IS)
        if window.train_sharpe != 0:
            window.efficiency_ratio = round(
                window.test_sharpe / window.train_sharpe, 3
            )

        # 거래 수
        if not trades_df.empty and "entry_date" in trades_df.columns:
            trades_copy = trades_df.copy()
            trades_copy["entry_date"] = pd.to_datetime(trades_copy["entry_date"])

            window.train_trades = len(
                trades_copy[
                    (trades_copy["entry_date"] >= window.train_start)
                    & (trades_copy["entry_date"] <= window.train_end)
                ]
            )
            window.test_trades = len(
                trades_copy[
                    (trades_copy["entry_date"] >= window.test_start)
                    & (trades_copy["entry_date"] <= window.test_end)
                ]
            )

    def run(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> dict:
        """전체 Walk-Forward 검증 실행"""
        windows = self.generate_windows(dates)

        if not windows:
            return {"status": "insufficient_data", "windows": []}

        for w in windows:
            self.evaluate_window(w, equity_df, trades_df)

        return self.compile_results()

    def compile_results(self) -> dict:
        """Walk-Forward 검증 결과 집계"""
        if not self.windows:
            return {"status": "no_windows", "windows": []}

        test_returns = [w.test_return_pct for w in self.windows]
        test_sharpes = [w.test_sharpe for w in self.windows]
        efficiencies = [
            w.efficiency_ratio for w in self.windows if w.efficiency_ratio != 0
        ]

        # OOS 수익률 양수 비율
        positive_oos = sum(1 for r in test_returns if r > 0)

        result = {
            "status": "completed",
            "n_windows": len(self.windows),
            "avg_test_return_pct": round(np.mean(test_returns), 2),
            "avg_test_sharpe": round(np.mean(test_sharpes), 3),
            "avg_efficiency_ratio": (
                round(np.mean(efficiencies), 3) if efficiencies else 0
            ),
            "positive_oos_rate": round(positive_oos / len(self.windows) * 100, 1),
            "total_test_trades": sum(w.test_trades for w in self.windows),
            "windows": [
                {
                    "id": w.window_id,
                    "train": f"{w.train_start}~{w.train_end}",
                    "test": f"{w.test_start}~{w.test_end}",
                    "train_ret": w.train_return_pct,
                    "test_ret": w.test_return_pct,
                    "train_sharpe": w.train_sharpe,
                    "test_sharpe": w.test_sharpe,
                    "efficiency": w.efficiency_ratio,
                }
                for w in self.windows
            ],
        }

        return result

    def print_results(self, results: dict):
        """Walk-Forward 결과 콘솔 출력"""
        if results["status"] != "completed":
            logger.info(f"Walk-Forward: {results['status']}")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward 검증 리포트")
        logger.info(f"{'='*60}")
        logger.info(f"  윈도우: {results['n_windows']}개 "
                     f"(Train {self.train_days}일 / Test {self.test_days}일)")
        logger.info(f"  OOS 평균 수익률: {results['avg_test_return_pct']:.2f}%")
        logger.info(f"  OOS 평균 Sharpe: {results['avg_test_sharpe']:.3f}")
        logger.info(f"  효율 비율 (OOS/IS): {results['avg_efficiency_ratio']:.3f}")
        logger.info(f"  OOS 양수 비율: {results['positive_oos_rate']:.1f}%")
        logger.info(f"  총 OOS 거래: {results['total_test_trades']}건")
        logger.info(f"{'─'*60}")

        for w in results["windows"]:
            eff_str = f"{w['efficiency']:.2f}" if w["efficiency"] else "N/A"
            emoji = "+" if w["test_ret"] > 0 else "-"
            logger.info(
                f"  [{w['id']}] Train {w['train_ret']:+.1f}% "
                f"(S={w['train_sharpe']:.2f}) | "
                f"Test {w['test_ret']:+.1f}% "
                f"(S={w['test_sharpe']:.2f}) | "
                f"Eff={eff_str} {emoji}"
            )

        logger.info(f"{'='*60}")

        # 과최적화 경고
        if results["avg_efficiency_ratio"] < 0.5:
            logger.warning(
                "과최적화 의심: OOS/IS 효율 비율 < 0.5. "
                "파라미터 완화 또는 특성 재검토 필요."
            )


# ══════════════════════════════════════════════════════════════
# v4.5 Bootstrap 검증 (Adaptive Strategy 반영)
# ══════════════════════════════════════════════════════════════


@dataclass
class BootstrapResult:
    """Bootstrap 검증 결과"""
    method: str                      # "shuffle" / "block"
    n_iterations: int                # 반복 수
    actual_metric: float             # 실제 전략 성과
    metric_name: str                 # "sharpe" / "total_return" / "win_rate"
    bootstrap_mean: float            # Bootstrap 평균
    bootstrap_std: float             # Bootstrap 표준편차
    p_value: float                   # p-value (실제가 Bootstrap보다 나을 확률)
    ci_lower: float                  # 95% 신뢰구간 하한
    ci_upper: float                  # 95% 신뢰구간 상한
    is_significant: bool             # p < 0.05 여부


class BootstrapValidator:
    """Shuffle + Block Bootstrap 검증기

    Shuffle Bootstrap:
      거래 수익률을 무작위 셔플하여 재계산.
      전략의 진입 타이밍이 랜덤보다 나은지 검증.

    Block Bootstrap:
      연속된 거래 블록 단위로 셔플하여 자기상관 보존.
      시계열 의존성을 유지하면서 검증.
    """

    def __init__(self, n_iterations: int = 1000, block_size: int = 10, seed: int = 42):
        self.n_iterations = n_iterations
        self.block_size = block_size
        self.rng = np.random.RandomState(seed)

    def _calc_metrics(self, returns: np.ndarray) -> dict:
        """수익률 배열에서 핵심 메트릭 계산"""
        if len(returns) == 0:
            return {"sharpe": 0.0, "total_return": 0.0, "win_rate": 0.0, "mdd": 0.0}

        total_return = float(np.sum(returns))
        win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0

        # Sharpe (거래 단위)
        std = float(np.std(returns))
        sharpe = float(np.mean(returns) / std) if std > 0 else 0.0

        # MDD (누적 수익 기반)
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        mdd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "sharpe": round(sharpe, 4),
            "total_return": round(total_return, 4),
            "win_rate": round(win_rate, 4),
            "mdd": round(mdd, 4),
        }

    def shuffle_bootstrap(
        self,
        trade_returns: np.ndarray,
        metric_name: str = "sharpe",
    ) -> BootstrapResult:
        """
        Shuffle Bootstrap: 거래 수익률을 무작위 셔플.

        Args:
            trade_returns: 각 거래의 수익률(%) 배열
            metric_name: 평가 메트릭 ("sharpe", "total_return", "win_rate")
        """
        actual = self._calc_metrics(trade_returns)[metric_name]
        bootstrap_values = []

        for _ in range(self.n_iterations):
            shuffled = self.rng.permutation(trade_returns)
            val = self._calc_metrics(shuffled)[metric_name]
            bootstrap_values.append(val)

        bs_arr = np.array(bootstrap_values)
        p_value = float(np.mean(bs_arr >= actual))

        return BootstrapResult(
            method="shuffle",
            n_iterations=self.n_iterations,
            actual_metric=actual,
            metric_name=metric_name,
            bootstrap_mean=round(float(np.mean(bs_arr)), 4),
            bootstrap_std=round(float(np.std(bs_arr)), 4),
            p_value=round(p_value, 4),
            ci_lower=round(float(np.percentile(bs_arr, 2.5)), 4),
            ci_upper=round(float(np.percentile(bs_arr, 97.5)), 4),
            is_significant=p_value < 0.05,
        )

    def block_bootstrap(
        self,
        trade_returns: np.ndarray,
        metric_name: str = "sharpe",
    ) -> BootstrapResult:
        """
        Block Bootstrap: 연속 블록 단위로 리샘플링 (자기상관 보존).

        Args:
            trade_returns: 각 거래의 수익률(%) 배열
            metric_name: 평가 메트릭
        """
        n = len(trade_returns)
        actual = self._calc_metrics(trade_returns)[metric_name]
        bootstrap_values = []

        n_blocks = max(1, n // self.block_size)

        for _ in range(self.n_iterations):
            # 블록 시작점을 랜덤 선택하여 n개 수익률 재구성
            sampled = []
            while len(sampled) < n:
                start = self.rng.randint(0, max(1, n - self.block_size + 1))
                block = trade_returns[start: start + self.block_size]
                sampled.extend(block)
            sampled = np.array(sampled[:n])

            val = self._calc_metrics(sampled)[metric_name]
            bootstrap_values.append(val)

        bs_arr = np.array(bootstrap_values)
        p_value = float(np.mean(bs_arr >= actual))

        return BootstrapResult(
            method="block",
            n_iterations=self.n_iterations,
            actual_metric=actual,
            metric_name=metric_name,
            bootstrap_mean=round(float(np.mean(bs_arr)), 4),
            bootstrap_std=round(float(np.std(bs_arr)), 4),
            p_value=round(p_value, 4),
            ci_lower=round(float(np.percentile(bs_arr, 2.5)), 4),
            ci_upper=round(float(np.percentile(bs_arr, 97.5)), 4),
            is_significant=p_value < 0.05,
        )

    def run(self, trades_df: pd.DataFrame) -> dict:
        """전체 Bootstrap 검증 실행.

        Args:
            trades_df: 거래 로그 (pnl_pct 컬럼 필요)

        Returns:
            dict: shuffle + block 결과 (sharpe, total_return, win_rate 각각)
        """
        if trades_df.empty or "pnl_pct" not in trades_df.columns:
            return {"status": "no_trades"}

        returns = trades_df["pnl_pct"].values / 100  # % → 비율

        results = {"status": "completed", "n_trades": len(returns)}

        for metric in ["sharpe", "total_return", "win_rate"]:
            shuffle_res = self.shuffle_bootstrap(returns, metric)
            block_res = self.block_bootstrap(returns, metric)

            results[f"shuffle_{metric}"] = {
                "actual": shuffle_res.actual_metric,
                "bs_mean": shuffle_res.bootstrap_mean,
                "p_value": shuffle_res.p_value,
                "significant": shuffle_res.is_significant,
                "ci": [shuffle_res.ci_lower, shuffle_res.ci_upper],
            }
            results[f"block_{metric}"] = {
                "actual": block_res.actual_metric,
                "bs_mean": block_res.bootstrap_mean,
                "p_value": block_res.p_value,
                "significant": block_res.is_significant,
                "ci": [block_res.ci_lower, block_res.ci_upper],
            }

        return results

    def print_results(self, results: dict):
        """Bootstrap 검증 결과 콘솔 출력"""
        if results.get("status") != "completed":
            logger.info(f"Bootstrap: {results.get('status', 'unknown')}")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Bootstrap 검증 리포트 ({results['n_trades']}건)")
        logger.info(f"{'='*60}")

        for metric in ["sharpe", "total_return", "win_rate"]:
            logger.info(f"\n  [{metric.upper()}]")
            for method in ["shuffle", "block"]:
                key = f"{method}_{metric}"
                r = results.get(key, {})
                sig = "***" if r.get("significant") else ""
                ci = r.get("ci", [0, 0])
                logger.info(
                    f"    {method:>7}: actual={r.get('actual', 0):.4f} | "
                    f"bs_mean={r.get('bs_mean', 0):.4f} | "
                    f"p={r.get('p_value', 1):.3f} {sig} | "
                    f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
                )

        logger.info(f"\n{'='*60}")
        logger.info("  *** = p < 0.05 (통계적 유의)")
        logger.info(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# v4.6 몬테카를로 시뮬레이션 (거래 순서 셔플 기반)
# ══════════════════════════════════════════════════════════════


@dataclass
class MonteCarloResult:
    """몬테카를로 시뮬레이션 결과"""
    n_simulations: int
    n_trades: int
    # 에쿼티 커브 분포
    final_equity_mean: float
    final_equity_median: float
    final_equity_p5: float          # 5th percentile (worst case)
    final_equity_p95: float         # 95th percentile (best case)
    # MDD 분포
    mdd_mean_pct: float
    mdd_median_pct: float
    mdd_p95_pct: float              # 95th percentile worst-case MDD
    # 파산 확률
    ruin_probability: float         # 자본 50% 이하로 하락할 확률
    ruin_threshold_pct: float       # 파산 임계값 (기본 -50%)
    # 수익 분포
    positive_pct: float             # 양수 수익 시뮬레이션 비율
    return_mean_pct: float
    return_p5_pct: float            # 5th percentile 수익률
    return_p95_pct: float           # 95th percentile 수익률


class MonteCarloSimulator:
    """거래 순서 셔플 기반 몬테카를로 시뮬레이션.

    거래 수익률을 무작위로 재배치하여 N개의 에쿼티 커브를 생성.
    이를 통해:
      - MDD 분포 (95th percentile worst-case)
      - 파산 확률 (Probability of Ruin)
      - 수익 분포의 신뢰구간
    을 도출.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        ruin_threshold_pct: float = -50.0,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.ruin_threshold_pct = ruin_threshold_pct
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _build_equity_curve(
        trade_pnl_amounts: np.ndarray,
        initial_capital: float,
    ) -> np.ndarray:
        """거래 PnL 금액 배열로 에쿼티 커브 구축."""
        equity = np.empty(len(trade_pnl_amounts) + 1)
        equity[0] = initial_capital
        for i, pnl in enumerate(trade_pnl_amounts):
            equity[i + 1] = equity[i] + pnl
        return equity

    @staticmethod
    def _calc_mdd_pct(equity: np.ndarray) -> float:
        """에쿼티 커브에서 MDD(%) 계산."""
        peak = np.maximum.accumulate(equity)
        drawdown_pct = (equity - peak) / np.where(peak > 0, peak, 1) * 100
        return float(np.min(drawdown_pct))

    def run(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 50_000_000,
    ) -> dict:
        """몬테카를로 시뮬레이션 실행.

        Args:
            trades_df: 거래 로그 (pnl 컬럼 필요)
            initial_capital: 초기 자본

        Returns:
            dict: 시뮬레이션 결과
        """
        if trades_df.empty or "pnl" not in trades_df.columns:
            return {"status": "no_trades"}

        trade_pnls = trades_df["pnl"].values.astype(float)
        n_trades = len(trade_pnls)

        final_equities = []
        mdd_values = []
        ruin_count = 0
        ruin_level = initial_capital * (1 + self.ruin_threshold_pct / 100)

        for _ in range(self.n_simulations):
            shuffled = self.rng.permutation(trade_pnls)
            equity = self._build_equity_curve(shuffled, initial_capital)

            final_equities.append(equity[-1])
            mdd_values.append(self._calc_mdd_pct(equity))

            # 파산 체크: 에쿼티가 한 번이라도 임계값 이하로 떨어졌는가
            if np.min(equity) <= ruin_level:
                ruin_count += 1

        final_arr = np.array(final_equities)
        mdd_arr = np.array(mdd_values)
        returns_pct = (final_arr / initial_capital - 1) * 100

        result = MonteCarloResult(
            n_simulations=self.n_simulations,
            n_trades=n_trades,
            final_equity_mean=round(float(np.mean(final_arr))),
            final_equity_median=round(float(np.median(final_arr))),
            final_equity_p5=round(float(np.percentile(final_arr, 5))),
            final_equity_p95=round(float(np.percentile(final_arr, 95))),
            mdd_mean_pct=round(float(np.mean(mdd_arr)), 2),
            mdd_median_pct=round(float(np.median(mdd_arr)), 2),
            mdd_p95_pct=round(float(np.percentile(mdd_arr, 5)), 2),  # 5th = worst
            ruin_probability=round(ruin_count / self.n_simulations * 100, 2),
            ruin_threshold_pct=self.ruin_threshold_pct,
            positive_pct=round(float(np.mean(returns_pct > 0)) * 100, 1),
            return_mean_pct=round(float(np.mean(returns_pct)), 2),
            return_p5_pct=round(float(np.percentile(returns_pct, 5)), 2),
            return_p95_pct=round(float(np.percentile(returns_pct, 95)), 2),
        )

        return {
            "status": "completed",
            "result": result,
        }

    def print_results(self, results: dict):
        """몬테카를로 시뮬레이션 결과 콘솔 출력."""
        if results.get("status") != "completed":
            logger.info(f"Monte Carlo: {results.get('status', 'unknown')}")
            return

        r = results["result"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Monte Carlo Simulation ({r.n_simulations}회, {r.n_trades}건)")
        logger.info(f"{'='*60}")

        logger.info(f"  [에쿼티 분포]")
        logger.info(f"    평균: {r.final_equity_mean:,.0f}원")
        logger.info(f"    중앙값: {r.final_equity_median:,.0f}원")
        logger.info(f"    5th percentile (worst): {r.final_equity_p5:,.0f}원")
        logger.info(f"    95th percentile (best): {r.final_equity_p95:,.0f}원")

        logger.info(f"  [수익률 분포]")
        logger.info(f"    평균: {r.return_mean_pct:+.2f}%")
        logger.info(f"    5th~95th: [{r.return_p5_pct:+.2f}%, {r.return_p95_pct:+.2f}%]")
        logger.info(f"    양수 수익 비율: {r.positive_pct:.1f}%")

        logger.info(f"  [MDD 분포]")
        logger.info(f"    평균 MDD: {r.mdd_mean_pct:.2f}%")
        logger.info(f"    중앙값 MDD: {r.mdd_median_pct:.2f}%")
        logger.info(f"    95th worst-case MDD: {r.mdd_p95_pct:.2f}%")

        logger.info(f"  [파산 확률]")
        logger.info(f"    임계값: 자본 {r.ruin_threshold_pct:.0f}% 이하")
        logger.info(f"    파산 확률: {r.ruin_probability:.2f}%")

        # 판정
        if r.ruin_probability > 10:
            logger.warning(f"    파산 확률 {r.ruin_probability:.1f}% > 10%: 리스크 과다")
        elif r.ruin_probability > 5:
            logger.warning(f"    파산 확률 {r.ruin_probability:.1f}% > 5%: 주의 필요")
        else:
            logger.info(f"    파산 확률 양호 (<= 5%)")

        if r.mdd_p95_pct < -30:
            logger.warning(f"    worst-case MDD {r.mdd_p95_pct:.1f}% < -30%: 심각한 하락 가능")

        logger.info(f"{'='*60}")
