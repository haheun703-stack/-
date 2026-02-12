"""
Step 7: position_sizer.py — ATR 기반 동적 포지션 사이징

포지션 크기 = (계좌 × 리스크%) / (ATR × 배수)
등급별 조정: A=100%, B=67%, C=33%
"""

import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """ATR 기반 리스크 균등화 포지션 사이징"""

    def __init__(self, config: dict):
        bt = config["backtest"]
        self.max_risk_pct = bt["max_risk_pct"]
        self.max_single_pct = bt["max_single_position_pct"]
        self.max_portfolio_risk_pct = bt["max_portfolio_risk_pct"]
        self.atr_stop_mult = bt["trailing_stop_atr_mult"]

    def calculate(
        self,
        account_balance: float,
        entry_price: float,
        atr_value: float,
        grade_ratio: float,
        current_portfolio_risk: float = 0.0,
        stage_pct: float = 1.0,
    ) -> dict:
        """
        매수 수량 계산.

        Args:
            account_balance: 현재 계좌 잔고
            entry_price: 예상 진입가
            atr_value: ATR(14) 값
            grade_ratio: 등급별 비율 (A=1.0, B=0.67, C=0.33)
            current_portfolio_risk: 현재 포트폴리오 리스크 금액 합계
            stage_pct: 분할 매수 비중 (Impulse=0.4, Confirm=0.4, Breakout=0.2)

        Returns:
            {shares, investment, risk_amount, stop_loss, target, pct_of_account}
        """
        if atr_value <= 0 or entry_price <= 0 or account_balance <= 0:
            return self._empty_result()

        stop_distance = atr_value * self.atr_stop_mult

        # 기본 수량: 리스크 금액 / 손절 거리
        risk_amount = account_balance * self.max_risk_pct
        raw_shares = int(risk_amount / stop_distance)

        # 등급별 조정 × 분할 비중 조정
        adjusted_shares = int(raw_shares * grade_ratio * stage_pct)

        # 최대 투입 한도 (계좌의 40%)
        max_by_pct = int(account_balance * self.max_single_pct / entry_price)
        final_shares = min(adjusted_shares, max_by_pct)

        # 포트폴리오 리스크 한도 체크 (총 6%)
        max_portfolio_risk = account_balance * self.max_portfolio_risk_pct
        remaining_risk = max_portfolio_risk - current_portfolio_risk
        if remaining_risk <= 0:
            final_shares = 0
        else:
            max_by_risk = int(remaining_risk / stop_distance)
            final_shares = min(final_shares, max_by_risk)

        final_shares = max(final_shares, 0)

        # 한국 주식은 1주 단위
        investment = final_shares * entry_price
        actual_risk = final_shares * stop_distance
        stop_loss = entry_price - stop_distance
        target = entry_price + atr_value * 5.0  # ATR × 5 목표

        pct_of_account = investment / account_balance * 100 if account_balance > 0 else 0

        return {
            "shares": final_shares,
            "investment": int(investment),
            "risk_amount": int(actual_risk),
            "stop_loss": int(stop_loss),
            "target": int(target),
            "stop_distance": int(stop_distance),
            "pct_of_account": round(pct_of_account, 1),
        }

    @staticmethod
    def _empty_result():
        return {
            "shares": 0,
            "investment": 0,
            "risk_amount": 0,
            "stop_loss": 0,
            "target": 0,
            "stop_distance": 0,
            "pct_of_account": 0.0,
        }
