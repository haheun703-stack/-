"""
Alpha Engine — L1 REGIME + L4 RISK 통합 인터페이스

모든 외부 호출은 이 클래스를 통해서만 접근.
enabled=False 시 모든 메서드가 안전 기본값 반환 → 기존 시스템 동작 변경 없음.
"""

from __future__ import annotations

import logging

import pandas as pd

from .models import (
    AlphaRegimeLevel,
    ExitSignal,
    RegimeParams,
    VetoDecision,
)
from .regime import AlphaRegime
from .risk_manager import AlphaRiskManager

logger = logging.getLogger(__name__)

# disabled 시 사용할 기본 파라미터 (기존 시스템과 동일 효과)
_DEFAULT_PARAMS = RegimeParams(
    max_positions=5,
    position_pct=0.40,
    stop_mult=2.0,
    cash_min_pct=0.0,
)


class AlphaEngine:
    """L1 REGIME + L4 RISK 통합 퀀트 엔진

    Usage (backtest):
        alpha = AlphaEngine(config)
        alpha.reset(initial_capital)

        for each day:
            level, params = alpha.get_regime(regime_state)
            alpha.update_equity(equity)

            for each signal:
                veto = alpha.veto_buy(level, params, capital, cash, ...)
                if veto.vetoed: skip

            for each position:
                exit = alpha.check_exits(pos_data, df, idx)
                if exit: sell

            portfolio_exit = alpha.check_portfolio()
            alpha.end_of_day(daily_pnl_pct)
    """

    def __init__(self, config: dict):
        self.enabled = config.get("alpha_engine", {}).get("enabled", False)
        self._config = config

        if self.enabled:
            self.regime = AlphaRegime(config)
            self.risk = AlphaRiskManager(config)
            logger.info("Alpha Engine 활성화 — L1 REGIME + L4 RISK")
        else:
            self.regime = None
            self.risk = None

    def reset(self, initial_capital: float) -> None:
        """백테스트 시작 시 상태 초기화"""
        if self.enabled:
            self.regime.reset()
            self.risk.reset(initial_capital)

    # ──────────────────────────────────────────────
    # L1: Regime
    # ──────────────────────────────────────────────

    def get_regime(
        self, regime_state=None,
    ) -> tuple[AlphaRegimeLevel, RegimeParams]:
        """현재 레짐 + 파라미터 반환.

        Args:
            regime_state: RegimeGate.detect() 결과 (백테스트)
                          None이면 라이브 모드 (JSON 파일 읽기)

        Returns:
            (AlphaRegimeLevel, RegimeParams)
        """
        if not self.enabled:
            return AlphaRegimeLevel.CAUTION, _DEFAULT_PARAMS

        if regime_state is not None:
            level = self.regime.detect_backtest(regime_state)
        else:
            level = self.regime.detect_live()

        params = self.regime.get_params()
        return level, params

    # ──────────────────────────────────────────────
    # L4: VETO
    # ──────────────────────────────────────────────

    def veto_buy(
        self,
        regime_level: AlphaRegimeLevel,
        regime_params: RegimeParams,
        capital: float,
        cash: float,
        num_positions: int,
        investment_amount: float,
    ) -> VetoDecision:
        """매수 VETO 검사.

        disabled → 항상 통과 (VetoDecision.pass_through()).
        """
        if not self.enabled:
            return VetoDecision.pass_through()

        return self.risk.veto_buy(
            regime_level=regime_level,
            regime_params=regime_params,
            capital=capital,
            cash=cash,
            num_positions=num_positions,
            investment_amount=investment_amount,
        )

    # ──────────────────────────────────────────────
    # L4: Exit Rules (X1-X5)
    # ──────────────────────────────────────────────

    def check_exits(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr_value: float,
        hold_days: int,
        partial_sold: bool = False,
        df: pd.DataFrame | None = None,
        idx: int = 0,
    ) -> ExitSignal | None:
        """X1-X5 청산 규칙 검사.

        disabled → None (기존 청산 로직 그대로 사용).
        """
        if not self.enabled:
            return None

        return self.risk.check_exit_rules(
            entry_price=entry_price,
            current_price=current_price,
            highest_price=highest_price,
            atr_value=atr_value,
            hold_days=hold_days,
            partial_sold=partial_sold,
            df=df,
            idx=idx,
        )

    # ──────────────────────────────────────────────
    # L4: Portfolio Defense
    # ──────────────────────────────────────────────

    def check_portfolio(self) -> ExitSignal | None:
        """포트폴리오 레벨 방어선 검사.

        disabled → None.
        """
        if not self.enabled or self.risk is None:
            return None

        equity = self.risk._peak_equity  # 현재 추적 중인 최고치 사용
        return self.risk.check_portfolio_risk(equity)

    # ──────────────────────────────────────────────
    # Daily Update
    # ──────────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        """MDD 추적용 자산 갱신 (매일 호출)"""
        if self.enabled and self.risk:
            self.risk.update_peak_equity(equity)

    def end_of_day(self, daily_pnl_pct: float) -> None:
        """일일 마감 시 PnL 기록"""
        if self.enabled and self.risk:
            self.risk.update_daily_pnl(daily_pnl_pct)
