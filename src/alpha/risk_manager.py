"""
Alpha L4 RISK Manager — VETO 시스템 + X1-X5 청산 규칙 + 포트폴리오 방어선

ALPHA_ENGINE_SPECIFICATION.docx 기반:
- VETO: 8개 조건 중 하나라도 해당 → 매수 거부 (L4만이 거부 가능)
- X1-X5: 우선순위 X1 > X5 > X3 > X4 > X2
- 포트폴리오: 일일 -3%, 주간 -5%, MDD -15% 서킷브레이커
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .models import (
    AlphaRegimeLevel,
    ExitRuleType,
    ExitSignal,
    RegimeParams,
    VetoDecision,
    VetoReason,
)

logger = logging.getLogger(__name__)


class AlphaRiskManager:
    """L4 리스크 관리 — VETO + Exit Rules + Portfolio Defense"""

    def __init__(self, config: dict):
        risk_cfg = config.get("alpha_engine", {}).get("risk", {})

        # 포트폴리오 방어선
        self.daily_loss_limit = risk_cfg.get("daily_loss_limit", -0.03)
        self.weekly_loss_limit = risk_cfg.get("weekly_loss_limit", -0.05)
        self.mdd_circuit_breaker = risk_cfg.get("mdd_circuit_breaker", -0.15)

        # X1-X5 파라미터
        self.x1_stop_mult = risk_cfg.get("x1_stop_mult", 2.0)
        self.x2_consecutive_days = risk_cfg.get("x2_consecutive_days", 3)
        self.x3_trailing_mult = risk_cfg.get("x3_trailing_mult", 2.5)
        self.x4_max_hold_days = risk_cfg.get("x4_max_hold_days", 10)
        self.x4_min_pnl_pct = risk_cfg.get("x4_min_pnl_pct", 0.02)
        self.x5_target_mult = risk_cfg.get("x5_target_mult", 4.0)
        self.x5_partial_pct = risk_cfg.get("x5_partial_pct", 0.50)

        # Swing Philosophy (SW-1~SW-6)
        sw = config.get("swing_philosophy", {})
        self._swing_enabled = sw.get("enabled", False)
        self._dynamic_stop = sw.get("dynamic_stop", {})
        self._dynamic_time = sw.get("dynamic_time", {})
        self._scenario_exit = sw.get("scenario_exit", {})
        self._config = config

        # MDD 추적
        self._peak_equity = 0.0
        self._mdd_triggered = False

        # 주간 PnL 추적 (5거래일 롤링)
        self._daily_pnl_history: deque[float] = deque(maxlen=5)

        # 일일 PnL
        self._daily_pnl_pct = 0.0

    def reset(self, initial_capital: float) -> None:
        """백테스트 시작 시 상태 초기화"""
        self._peak_equity = initial_capital
        self._mdd_triggered = False
        self._daily_pnl_history.clear()
        self._daily_pnl_pct = 0.0

    # ──────────────────────────────────────────────
    # VETO System — 매수 거부 판정
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
        """8개 VETO 조건 검사. 하나라도 해당 → vetoed=True.

        Args:
            regime_level: 현재 Alpha 레짐
            regime_params: 레짐별 운영 파라미터
            capital: 전체 자본 (현금 + 평가액)
            cash: 가용 현금
            num_positions: 현재 보유 종목 수
            investment_amount: 이번 매수 예상 투입 금액
        """
        reasons: list[VetoReason] = []
        details: list[str] = []

        # V1: CRISIS 레짐 → 매수 금지 (SW-3: contrarian 모드 시 우회)
        if regime_level == AlphaRegimeLevel.CRISIS:
            sw = self._config.get("swing_philosophy", {})
            contrarian_slots = sw.get("contrarian", {}).get("CRISIS", {}).get("slots", 0)
            if self._swing_enabled and contrarian_slots > 0:
                logger.info("V1 CRISIS 패스: swing_philosophy contrarian 모드 (%d슬롯)", contrarian_slots)
            else:
                reasons.append(VetoReason.V1_CRISIS)
                details.append("CRISIS 레짐 — 전면 매수 금지")

        # V2: 매수 후 현금 비중이 최소 기준 미달
        if capital > 0:
            cash_after = cash - investment_amount
            cash_pct_after = cash_after / capital
            if cash_pct_after < regime_params.cash_min_pct:
                reasons.append(VetoReason.V2_CASH_MIN)
                details.append(
                    f"매수 후 현금 {cash_pct_after:.1%} < 최소 {regime_params.cash_min_pct:.0%}"
                )

        # V3: 최대 동시 보유 초과
        if num_positions >= regime_params.max_positions:
            reasons.append(VetoReason.V3_MAX_POS)
            details.append(
                f"보유 {num_positions}종목 >= 최대 {regime_params.max_positions}"
            )

        # V4: MDD 서킷브레이커 발동
        if self._mdd_triggered:
            reasons.append(VetoReason.V4_MDD_CIRCUIT)
            details.append("MDD 서킷브레이커 활성 — 전면 매수 중단")

        # V5: 일일 손실 한도 초과
        if self._daily_pnl_pct <= self.daily_loss_limit:
            reasons.append(VetoReason.V5_DAILY_LOSS)
            details.append(
                f"일일 손실 {self._daily_pnl_pct:.2%} <= {self.daily_loss_limit:.0%}"
            )

        # V6: 주간 손실 한도 초과
        weekly_pnl = sum(self._daily_pnl_history) if self._daily_pnl_history else 0.0
        if weekly_pnl <= self.weekly_loss_limit:
            reasons.append(VetoReason.V6_WEEKLY_LOSS)
            details.append(
                f"주간 손실 {weekly_pnl:.2%} <= {self.weekly_loss_limit:.0%}"
            )

        # V7: 종목당 비중 초과
        if capital > 0 and investment_amount / capital > regime_params.position_pct:
            reasons.append(VetoReason.V7_POS_SIZE)
            details.append(
                f"투입 비중 {investment_amount / capital:.1%} > {regime_params.position_pct:.0%}"
            )

        # V8: STOP.signal 활성 (라이브용, 백테스트에서는 해당 없음)
        if Path("STOP.signal").exists():
            reasons.append(VetoReason.V8_STOP_SIGNAL)
            details.append("STOP.signal 파일 존재")

        vetoed = len(reasons) > 0
        if vetoed:
            logger.info(
                "  VETO 발동: %s",
                " | ".join(r.value for r in reasons),
            )

        return VetoDecision(
            vetoed=vetoed,
            reasons=reasons,
            details="; ".join(details),
        )

    # ──────────────────────────────────────────────
    # X1-X5 개별종목 청산 규칙
    # ──────────────────────────────────────────────

    def check_exit_rules(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr_value: float,
        hold_days: int,
        partial_sold: bool,
        df: pd.DataFrame | None = None,
        idx: int = 0,
        regime: str = "",
    ) -> ExitSignal | None:
        """X1-X5 우선순위 순서로 청산 규칙 검사.

        우선순위: X1(Hard Stop) > X5(Target) > X3(Trailing) > X4(Time) > X6(Scenario) > X2(Flow)

        Args:
            entry_price: 진입가
            current_price: 현재가 (당일 종가 또는 저가)
            highest_price: 보유 중 최고가
            atr_value: ATR(14) 값
            hold_days: 보유 일수
            partial_sold: 이미 부분 청산했는지 여부
            df: 종목 DataFrame (X2 수급 체크용)
            idx: 현재 인덱스
            regime: 현재 레짐 ("BULL"/"CAUTION"/"BEAR"/"CRISIS"), SW-1/SW-2용

        Returns:
            ExitSignal if triggered, None otherwise
        """
        if atr_value <= 0:
            return None

        # X1: Hard Stop — 절대 손절선 (SW-1: 레짐별 동적 배수)
        if self._swing_enabled and regime:
            stop_mult = self._dynamic_stop.get(regime, self.x1_stop_mult)
        else:
            stop_mult = self.x1_stop_mult
        hard_stop = entry_price - atr_value * stop_mult
        if current_price <= hard_stop:
            return ExitSignal(
                triggered=True,
                rule=ExitRuleType.X1_HARD_STOP,
                exit_price=current_price,
                details=f"종가 {current_price:,.0f} <= Hard Stop {hard_stop:,.0f}",
            )

        # X5: 목표 도달 → 50% 부분 청산
        target = entry_price + atr_value * self.x5_target_mult
        if current_price >= target and not partial_sold:
            return ExitSignal(
                triggered=True,
                rule=ExitRuleType.X5_TARGET,
                exit_price=current_price,
                partial_pct=self.x5_partial_pct,
                details=f"목표 {target:,.0f} 도달, {self.x5_partial_pct:.0%} 부분청산",
            )

        # X3: 트레일링 스탑 — 수익 구간에서만 (highest - ATR × x3_mult)
        if current_price > entry_price:
            trailing_stop = highest_price - atr_value * self.x3_trailing_mult
            if current_price <= trailing_stop:
                return ExitSignal(
                    triggered=True,
                    rule=ExitRuleType.X3_TRAILING,
                    exit_price=current_price,
                    details=f"트레일링 {trailing_stop:,.0f} 이탈 (최고 {highest_price:,.0f})",
                )

        # X4: 시간 청산 — 장기 보유 + 수익 미미 (SW-2: 레짐별 동적)
        pnl_pct = (current_price / entry_price - 1) if entry_price > 0 else 0
        if self._swing_enabled and regime:
            time_cfg = self._dynamic_time.get(regime, {})
            max_hold = time_cfg.get("max_days", self.x4_max_hold_days)
            min_pnl = time_cfg.get("min_pnl_pct", self.x4_min_pnl_pct)
        else:
            max_hold = self.x4_max_hold_days
            min_pnl = self.x4_min_pnl_pct
        if hold_days >= max_hold and pnl_pct < min_pnl:
            return ExitSignal(
                triggered=True,
                rule=ExitRuleType.X4_TIME_EXIT,
                exit_price=current_price,
                details=f"{hold_days}일 보유, 수익 {pnl_pct:.1%} < {min_pnl:.0%} (한도 {max_hold}일)",
            )

        # X6: 시나리오 무효화 청산 (SW-6)
        if self._swing_enabled and self._scenario_exit.get("enabled"):
            x6 = self._check_scenario_exit(
                entry_price, current_price, hold_days, pnl_pct,
            )
            if x6:
                return x6

        # X2: 수급 이탈 — N일 연속 기관+외국인 순매도
        if df is not None and idx >= self.x2_consecutive_days:
            if self._check_flow_exit(df, idx):
                return ExitSignal(
                    triggered=True,
                    rule=ExitRuleType.X2_FLOW_EXIT,
                    exit_price=current_price,
                    details=f"{self.x2_consecutive_days}일 연속 수급 이탈",
                )

        return None

    def _check_flow_exit(self, df: pd.DataFrame, idx: int) -> bool:
        """X2: 기관+외국인 N일 연속 순매도 체크"""
        # 외국인/기관 컬럼 찾기
        foreign_col = None
        inst_col = None
        for col in df.columns:
            if col in ("외국인합계", "foreign_net"):
                foreign_col = col
            if col in ("기관합계", "institution_net"):
                inst_col = col

        if foreign_col is None and inst_col is None:
            return False

        consecutive = 0
        for i in range(self.x2_consecutive_days):
            check_idx = idx - i
            if check_idx < 0 or check_idx >= len(df):
                return False

            row = df.iloc[check_idx]
            foreign_val = row.get(foreign_col, 0) if foreign_col else 0
            inst_val = row.get(inst_col, 0) if inst_col else 0

            if pd.isna(foreign_val):
                foreign_val = 0
            if pd.isna(inst_val):
                inst_val = 0

            # 기관+외국인 합산이 음수 → 매도
            if (foreign_val + inst_val) < 0:
                consecutive += 1
            else:
                return False

        return consecutive >= self.x2_consecutive_days

    def _check_scenario_exit(
        self,
        entry_price: float,
        current_price: float,
        hold_days: int,
        pnl_pct: float,
    ) -> ExitSignal | None:
        """X6: 시나리오 무효화 시 유예 후 청산.

        active_scenarios.json에서 활성 시나리오를 로드하여,
        진입 시나리오가 비활성화되었고, 유예일 경과 + 수익 기준 미달이면 청산.
        (현재는 시나리오 매핑이 없으므로, sell_monitor.py에서
         ticker별 scenario_id를 전달하여 사용)
        """
        import json
        grace_days = self._scenario_exit.get("grace_days", 2)
        min_pnl = self._scenario_exit.get("min_pnl_for_hold", 0.05)

        # 수익이 기준 이상이면 시나리오 무효화되어도 보유
        if pnl_pct >= min_pnl:
            return None

        try:
            path = Path("data/scenarios/active_scenarios.json")
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            scenarios = data.get("scenarios", [])

            # 활성 시나리오가 하나도 없으면 무효화 간주
            if not scenarios:
                if hold_days >= grace_days:
                    return ExitSignal(
                        triggered=True,
                        rule=ExitRuleType.X6_SCENARIO_EXIT,
                        exit_price=current_price,
                        details=f"시나리오 전체 무효화, {hold_days}일 보유, 수익 {pnl_pct:.1%} < {min_pnl:.0%}",
                    )
        except Exception as e:
            logger.debug("X6 시나리오 체크 실패: %s", e)

        return None

    # ──────────────────────────────────────────────
    # 포트폴리오 레벨 방어선
    # ──────────────────────────────────────────────

    def update_daily_pnl(self, daily_pnl_pct: float) -> None:
        """일일 PnL 기록 (매일 장마감 후 호출)"""
        self._daily_pnl_pct = daily_pnl_pct
        self._daily_pnl_history.append(daily_pnl_pct)

    def update_peak_equity(self, equity: float) -> None:
        """MDD 추적용 최고 자산 갱신"""
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            mdd = (equity - self._peak_equity) / self._peak_equity
            if mdd <= self.mdd_circuit_breaker:
                if not self._mdd_triggered:
                    logger.critical(
                        "MDD 서킷브레이커 발동! MDD=%.2f%% (한도 %.0f%%)",
                        mdd * 100, self.mdd_circuit_breaker * 100,
                    )
                self._mdd_triggered = True

    def check_portfolio_risk(self, equity: float) -> ExitSignal | None:
        """포트폴리오 레벨 방어선 검사.

        Returns:
            XP_MDD → 전량 청산
            XP_WEEKLY → 1/3 감축
            XP_DAILY → 신규매수 중단 (VETO로 처리, 여기서는 알림만)
            None → 이상 없음
        """
        # MDD -15% → 서킷브레이커 (전량 청산)
        if self._mdd_triggered:
            return ExitSignal(
                triggered=True,
                rule=ExitRuleType.XP_MDD,
                details=f"MDD 서킷브레이커 — 전량 청산",
            )

        # 주간 -5% → 전체 1/3 감축
        weekly_pnl = sum(self._daily_pnl_history) if self._daily_pnl_history else 0.0
        if weekly_pnl <= self.weekly_loss_limit:
            return ExitSignal(
                triggered=True,
                rule=ExitRuleType.XP_WEEKLY,
                partial_pct=0.33,  # 1/3 감축
                details=f"주간 손실 {weekly_pnl:.2%} — 1/3 감축",
            )

        return None

    @property
    def is_daily_loss_exceeded(self) -> bool:
        """일일 손실 한도 초과 여부 (VETO V5용)"""
        return self._daily_pnl_pct <= self.daily_loss_limit

    @property
    def is_mdd_triggered(self) -> bool:
        return self._mdd_triggered
