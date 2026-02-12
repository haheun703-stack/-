"""
Step 6: backtest_engine.py — v2.1 듀얼 트리거 백테스트 엔진

v2.0 → v2.1 변경:
- 분할 매수: Impulse(40%) → Confirm(40%) → Breakout(20%)
- 모드별 손절: Impulse(-3% 빠른 컷) vs Confirm(-5% 여유)
- 보유 종목 돌파 시 추가 매수 (Trigger-3)
- 거래 로그에 trigger_type 기록
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .signal_engine import SignalEngine, TriggerType
from .position_sizer import PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """보유 포지션"""
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    target_price: float
    atr_value: float
    grade: str
    bes_score: float
    trigger_type: str = "confirm"   # impulse / confirm / breakout
    stop_loss_pct: float = 0.05     # 모드별 손절 비율
    highest_price: float = 0.0
    trailing_stop: float = 0.0
    partial_sold: bool = False
    # 분할 매수 추적
    total_stages_entered: int = 1   # 몇 차 매수까지 들어갔나
    max_investment: float = 0.0     # 최대 투입 가능 금액


@dataclass
class Trade:
    """완료된 거래 기록"""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str
    grade: str
    bes_score: float
    commission: float
    trigger_type: str = "confirm"


class BacktestEngine:
    """v2.1 듀얼 트리거 백테스트 엔진"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        bt = self.config["backtest"]
        self.initial_capital = bt["initial_capital"]
        self.max_positions = bt["max_positions"]
        self.commission_rate = bt["commission_rate"]
        self.slippage_rate = bt["slippage_rate"]
        self.partial_tp_pct = bt["partial_take_profit_pct"]
        self.trend_exit_days = bt["trend_exit_consecutive_days"]
        self.atr_stop_mult = bt["trailing_stop_atr_mult"]

        self.signal_engine = SignalEngine(config_path)
        self.position_sizer = PositionSizer(self.config)

        # 상태
        self.cash = self.initial_capital
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self.signal_log: list[dict] = []

    def load_data(self) -> dict:
        """processed 디렉토리에서 전종목 데이터 로딩"""
        processed_dir = Path("data/processed")
        data = {}
        for fpath in sorted(processed_dir.glob("*.parquet")):
            data[fpath.stem] = pd.read_parquet(fpath)
        logger.info(f"데이터 로딩: {len(data)}종목")
        return data

    # ──────────────────────────────────────────────
    # 매수 (분할 매수 지원)
    # ──────────────────────────────────────────────

    def _execute_buy(self, signal: dict, next_open: float,
                     stage_pct: float = 1.0) -> Optional[Position]:
        """
        시그널 기반 매수.
        stage_pct: 이 단계에서 사용할 비중 (Impulse=0.4, Confirm=0.4, Breakout=0.2)
        """
        entry_price = next_open * (1 + self.slippage_rate)

        current_risk = sum(
            p.shares * (p.entry_price - p.stop_loss)
            for p in self.positions
        )

        sizing = self.position_sizer.calculate(
            account_balance=self.cash,
            entry_price=entry_price,
            atr_value=signal["atr_value"],
            grade_ratio=signal["position_ratio"],
            current_portfolio_risk=current_risk,
            stage_pct=stage_pct,  # 분할 비중 전달
        )

        if sizing["shares"] <= 0:
            return None

        commission = entry_price * sizing["shares"] * self.commission_rate
        total_cost = sizing["investment"] + commission

        if total_cost > self.cash:
            affordable = int((self.cash - commission) / entry_price)
            if affordable <= 0:
                return None
            sizing["shares"] = affordable
            sizing["investment"] = affordable * entry_price
            commission = sizing["investment"] * self.commission_rate
            total_cost = sizing["investment"] + commission

        self.cash -= total_cost

        # 모드별 손절선
        stop_price = signal["stop_loss"]
        trigger_type = signal.get("trigger_type", "confirm")
        stop_pct = signal.get("stop_loss_pct", 0.05)

        pos = Position(
            ticker=signal["ticker"],
            entry_date=str(signal["date"]),
            entry_price=entry_price,
            shares=sizing["shares"],
            stop_loss=stop_price,
            target_price=signal["target_price"],
            atr_value=signal["atr_value"],
            grade=signal["grade"],
            bes_score=signal.get("zone_score", signal.get("bes_score", 0)),
            trigger_type=trigger_type,
            stop_loss_pct=stop_pct,
            highest_price=entry_price,
            trailing_stop=entry_price - signal["atr_value"] * self.atr_stop_mult,
        )
        self.positions.append(pos)

        mode_emoji = "⚡" if trigger_type == "impulse" else ("🎯" if trigger_type == "confirm" else "🚀")
        logger.debug(f"  {mode_emoji} 매수: {signal['ticker']} {sizing['shares']}주 @{int(entry_price):,} "
                     f"[{trigger_type}] Zone={pos.bes_score:.2f} {signal['grade']}등급")
        return pos

    def _execute_add(self, pos: Position, df: pd.DataFrame, idx: int,
                     trigger_result, stage_pct: float = 0.20):
        """기존 포지션에 추가 매수 (Breakout 트리거)"""
        if idx + 1 >= len(df):
            return

        next_open = df["open"].iloc[idx + 1]
        entry_price = next_open * (1 + self.slippage_rate)

        # 추가 매수 수량 계산 (전체 포지션의 stage_pct만큼)
        add_amount = self.cash * 0.3 * stage_pct  # 잔고의 30% × stage_pct
        add_shares = int(add_amount / entry_price)

        if add_shares <= 0:
            return

        commission = entry_price * add_shares * self.commission_rate
        total_cost = entry_price * add_shares + commission

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        # 평균 단가 갱신
        total_shares = pos.shares + add_shares
        avg_price = (pos.entry_price * pos.shares + entry_price * add_shares) / total_shares
        pos.entry_price = avg_price
        pos.shares = total_shares
        pos.total_stages_entered += 1

        # 트레일링 스탑 재설정
        pos.trailing_stop = max(pos.trailing_stop, entry_price - pos.atr_value * self.atr_stop_mult)

        logger.debug(f"  🚀 추가매수: {pos.ticker} +{add_shares}주 @{int(entry_price):,} "
                     f"[breakout] 총 {pos.shares}주")

    # ──────────────────────────────────────────────
    # 매도
    # ──────────────────────────────────────────────

    def _execute_sell(self, pos: Position, exit_price: float,
                      exit_date: str, reason: str, shares: Optional[int] = None):
        """포지션 매도"""
        sell_shares = shares if shares else pos.shares
        actual_price = exit_price * (1 - self.slippage_rate)
        commission = actual_price * sell_shares * self.commission_rate
        gross_pnl = (actual_price - pos.entry_price) * sell_shares
        net_pnl = gross_pnl - commission - (pos.entry_price * sell_shares * self.commission_rate)
        pnl_pct = (actual_price / pos.entry_price - 1) * 100

        self.cash += actual_price * sell_shares - commission

        hold_days = 0
        try:
            hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(pos.entry_date)).days
        except Exception:
            pass

        trade = Trade(
            ticker=pos.ticker,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=actual_price,
            shares=sell_shares,
            pnl=round(net_pnl),
            pnl_pct=round(pnl_pct, 2),
            hold_days=hold_days,
            exit_reason=reason,
            grade=pos.grade,
            bes_score=pos.bes_score,
            commission=round(commission),
            trigger_type=pos.trigger_type,
        )
        self.trades.append(trade)

        if shares and shares < pos.shares:
            pos.shares -= sell_shares
            pos.partial_sold = True
        else:
            self.positions.remove(pos)

        emoji = "🔴" if net_pnl < 0 else "🟢"
        logger.debug(f"  {emoji} 매도: {pos.ticker} {sell_shares}주 @{int(actual_price):,} "
                     f"({reason}) [{pos.trigger_type}] PnL={pnl_pct:+.1f}%")

    # ──────────────────────────────────────────────
    # 보유 종목 관리
    # ──────────────────────────────────────────────

    def _manage_positions(self, data_dict: dict, idx: int, date_str: str):
        """모드별 손절/익절/트레일링 체크"""
        for pos in list(self.positions):
            if pos.ticker not in data_dict:
                continue
            df = data_dict[pos.ticker]
            if idx >= len(df):
                continue

            row = df.iloc[idx]
            close = row["close"]
            high = row["high"]
            low = row["low"]

            # 1. 모드별 손절 체크
            #    Impulse: 스윙 저점 or -3% (타이트)
            #    Confirm: ATR 스탑 or -5% (여유)
            if low <= pos.stop_loss:
                self._execute_sell(pos, pos.stop_loss, date_str, f"stop_loss_{pos.trigger_type}")
                continue

            # 퍼센트 기반 손절도 별도 체크 (당일 종가 기준)
            pct_loss = (close / pos.entry_price - 1)
            if pct_loss <= -pos.stop_loss_pct:
                self._execute_sell(pos, close, date_str, f"pct_stop_{pos.trigger_type}")
                continue

            # 2. 목표가 도달 → 50% 익절
            if not pos.partial_sold and high >= pos.target_price:
                partial_shares = pos.shares // 2
                if partial_shares > 0:
                    self._execute_sell(pos, pos.target_price, date_str,
                                       "partial_target", shares=partial_shares)

            # 3. 트레일링 스탑 갱신
            if high > pos.highest_price:
                pos.highest_price = high
                pos.trailing_stop = high - pos.atr_value * self.atr_stop_mult

            # 4. 트레일링 스탑 히트
            if pos.partial_sold and low <= pos.trailing_stop:
                self._execute_sell(pos, pos.trailing_stop, date_str, "trailing_stop")
                continue

            # 5. 추세 이탈
            if "sma_60" in df.columns and "sma_120" in df.columns:
                start_check = max(0, idx - self.trend_exit_days + 1)
                trend_broken = all(
                    df["sma_60"].iloc[i] < df["sma_120"].iloc[i]
                    for i in range(start_check, idx + 1)
                    if not pd.isna(df["sma_60"].iloc[i])
                )
                if trend_broken and (idx - start_check + 1) >= self.trend_exit_days:
                    self._execute_sell(pos, close, date_str, "trend_exit")
                    continue

    # ──────────────────────────────────────────────
    # 포트폴리오 가치
    # ──────────────────────────────────────────────

    def _calc_portfolio_value(self, data_dict: dict, idx: int) -> float:
        holdings = 0
        for pos in self.positions:
            if pos.ticker in data_dict:
                df = data_dict[pos.ticker]
                if idx < len(df):
                    holdings += df["close"].iloc[idx] * pos.shares
        return self.cash + holdings

    # ──────────────────────────────────────────────
    # 메인 백테스트 루프
    # ──────────────────────────────────────────────

    def run(self, data_dict: dict) -> dict:
        """
        v2.1 백테스트 루프.

        매일:
        1. 보유 종목 관리 (모드별 손절/익절/트레일링)
        2. 보유 종목 돌파 체크 (Trigger-3 추가 매수)
        3. 전종목 스캔 (Trigger-1/2 신규 진입)
        4. 에쿼티 커브 기록
        """
        first_ticker = list(data_dict.keys())[0]
        all_dates = data_dict[first_ticker].index
        start_idx = 200

        logger.info(f"백테스트 v2.1 시작: {all_dates[start_idx]} ~ {all_dates[-1]}")
        logger.info(f"  초기자본: {self.initial_capital:,}원 | 종목수: {len(data_dict)} | 듀얼트리거 ON")

        for idx in tqdm(range(start_idx, len(all_dates)), desc="🧪 백테스트 v2.1"):
            date_str = str(all_dates[idx].date()) if hasattr(all_dates[idx], "date") else str(all_dates[idx])

            # ── 1. 보유 종목 관리 ──
            self._manage_positions(data_dict, idx, date_str)

            # ── 2. 보유 종목 Breakout 체크 (Trigger-3 추가 매수) ──
            held_tickers = {p.ticker for p in self.positions}
            if held_tickers:
                breakouts = self.signal_engine.scan_breakout(data_dict, idx, held_tickers)
                for bo in breakouts:
                    # 해당 포지션 찾기
                    pos_list = [p for p in self.positions if p.ticker == bo["ticker"]]
                    if pos_list and pos_list[0].total_stages_entered < 3:
                        self._execute_add(
                            pos_list[0],
                            data_dict[bo["ticker"]],
                            idx,
                            bo["trigger"],
                            stage_pct=bo["trigger"].entry_stage_pct,
                        )

            # ── 3. 신규 진입 (Trigger-1 또는 Trigger-2) ──
            if len(self.positions) < self.max_positions:
                signals = self.signal_engine.scan_universe(data_dict, idx)

                for sig in signals:
                    self.signal_log.append({
                        "date": date_str,
                        "ticker": sig["ticker"],
                        "zone_score": sig["zone_score"],
                        "grade": sig["grade"],
                        "trigger_type": sig["trigger_type"],
                        "trigger_confidence": sig.get("trigger_confidence", 0),
                        "entry_price": sig["entry_price"],
                        "rr_ratio": sig["risk_reward_ratio"],
                    })

                held_tickers = {p.ticker for p in self.positions}

                for sig in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    if sig["ticker"] in held_tickers:
                        continue

                    # Impulse: 손익비 1.5 이상이면 OK (빠른 진입)
                    # Confirm: 손익비 2.0 이상 (안전 진입)
                    min_rr = 1.5 if sig["trigger_type"] == "impulse" else 2.0
                    if sig["risk_reward_ratio"] < min_rr:
                        continue

                    if idx + 1 < len(all_dates):
                        next_df = data_dict.get(sig["ticker"])
                        if next_df is not None and idx + 1 < len(next_df):
                            next_open = next_df["open"].iloc[idx + 1]
                            self._execute_buy(
                                sig, next_open,
                                stage_pct=sig.get("entry_stage_pct", 0.40),
                            )
                            held_tickers.add(sig["ticker"])

            # ── 4. 에쿼티 커브 ──
            portfolio_value = self._calc_portfolio_value(data_dict, idx)
            self.equity_curve.append({
                "date": date_str,
                "portfolio_value": round(portfolio_value),
                "cash": round(self.cash),
                "n_positions": len(self.positions),
                "positions": [f"{p.ticker}[{p.trigger_type[0]}]" for p in self.positions],
            })

        # ── 잔여 포지션 강제 청산 ──
        last_date = str(all_dates[-1].date()) if hasattr(all_dates[-1], "date") else str(all_dates[-1])
        for pos in list(self.positions):
            if pos.ticker in data_dict:
                close = data_dict[pos.ticker]["close"].iloc[-1]
                self._execute_sell(pos, close, last_date, "backtest_end")

        return self._compile_results()

    # ──────────────────────────────────────────────
    # 결과 정리
    # ──────────────────────────────────────────────

    def _compile_results(self) -> dict:
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        equity_df = pd.DataFrame(self.equity_curve)
        signals_df = pd.DataFrame(self.signal_log)

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        if not trades_df.empty:
            trades_df.to_csv(results_dir / "trades_log.csv", index=False, encoding="utf-8-sig")
        if not equity_df.empty:
            equity_df.to_csv(results_dir / "daily_equity.csv", index=False, encoding="utf-8-sig")
        if not signals_df.empty:
            signals_df.to_csv(results_dir / "signals_log.csv", index=False, encoding="utf-8-sig")

        stats = self._calc_stats(trades_df, equity_df)

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 백테스트 v2.1 완료!")
        logger.info(f"   총 거래: {stats['total_trades']}건")
        logger.info(f"   ├ Impulse: {stats['trigger_breakdown'].get('impulse', {}).get('count', 0)}건 "
                     f"(승률 {stats['trigger_breakdown'].get('impulse', {}).get('win_rate', 0):.1f}%)")
        logger.info(f"   └ Confirm: {stats['trigger_breakdown'].get('confirm', {}).get('count', 0)}건 "
                     f"(승률 {stats['trigger_breakdown'].get('confirm', {}).get('win_rate', 0):.1f}%)")
        logger.info(f"   승률: {stats['win_rate']:.1f}%")
        logger.info(f"   평균 손익비: 1:{stats['avg_rr_ratio']:.2f}")
        logger.info(f"   총 수익률: {stats['total_return_pct']:.1f}%")
        logger.info(f"   CAGR: {stats['cagr_pct']:.1f}%")
        logger.info(f"   MDD: {stats['max_drawdown_pct']:.1f}%")
        logger.info(f"   Sharpe: {stats['sharpe_ratio']:.2f}")
        logger.info(f"{'='*60}")

        return {
            "stats": stats,
            "trades_df": trades_df,
            "equity_df": equity_df,
            "signals_df": signals_df,
        }

    def _calc_stats(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        stats = {
            "total_trades": 0, "win_rate": 0.0, "avg_rr_ratio": 0.0,
            "total_return_pct": 0.0, "cagr_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "avg_hold_days": 0, "total_commission": 0,
            "profit_factor": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "grade_breakdown": {}, "trigger_breakdown": {},
        }

        if trades_df.empty or equity_df.empty:
            return stats

        stats["total_trades"] = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        stats["win_rate"] = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        stats["avg_hold_days"] = int(trades_df["hold_days"].mean())
        stats["total_commission"] = int(trades_df["commission"].sum())
        stats["avg_win_pct"] = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        stats["avg_loss_pct"] = losses["pnl_pct"].mean() if len(losses) > 0 else 0

        avg_win = abs(wins["pnl_pct"].mean()) if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 1
        stats["avg_rr_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

        total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
        stats["profit_factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else 0

        final_value = equity_df["portfolio_value"].iloc[-1]
        stats["total_return_pct"] = round((final_value / self.initial_capital - 1) * 100, 1)

        n_years = len(equity_df) / 252
        if n_years > 0 and final_value > 0:
            stats["cagr_pct"] = round(((final_value / self.initial_capital) ** (1 / n_years) - 1) * 100, 1)

        equity_series = equity_df["portfolio_value"]
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        stats["max_drawdown_pct"] = round(drawdown.min(), 1)

        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess = daily_returns.mean() - 0.03 / 252
            stats["sharpe_ratio"] = round(excess / daily_returns.std() * np.sqrt(252), 2)

        # 등급별 분석
        for grade in ["A", "B", "C"]:
            gt = trades_df[trades_df["grade"] == grade]
            if len(gt) > 0:
                gw = gt[gt["pnl"] > 0]
                stats["grade_breakdown"][grade] = {
                    "count": len(gt),
                    "win_rate": round(len(gw) / len(gt) * 100, 1),
                    "avg_pnl_pct": round(gt["pnl_pct"].mean(), 2),
                }

        # 🔥 트리거별 분석 (v2.1 신규)
        if "trigger_type" in trades_df.columns:
            for ttype in ["impulse", "confirm", "breakout"]:
                tt = trades_df[trades_df["trigger_type"] == ttype]
                if len(tt) > 0:
                    tw = tt[tt["pnl"] > 0]
                    stats["trigger_breakdown"][ttype] = {
                        "count": len(tt),
                        "win_rate": round(len(tw) / len(tt) * 100, 1),
                        "avg_pnl_pct": round(tt["pnl_pct"].mean(), 2),
                        "avg_hold_days": int(tt["hold_days"].mean()),
                        "total_pnl": int(tt["pnl"].sum()),
                    }

        return stats
