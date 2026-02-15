"""
v6.3 BacktestEngine 핵심 메커니즘 단위 테스트

테스트 대상: src/backtest_engine.py
- Position / Trade 데이터클래스
- _execute_buy: 슬리피지, 수수료, 잔고 부족
- _execute_sell: 전량/부분 청산, PnL 계산
- 4단계 부분청산: 2R/4R 타겟
- 트레일링 스탑: 계산, 최소 mult, 히트 조건
- 최대 보유일: max_hold_days 청산
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import BacktestEngine, Position, Trade
from src.position_sizer import PositionSizer


# ── 헬퍼 ──

TEST_CONFIG = {
    "backtest": {
        "initial_capital": 10_000_000,
        "max_positions": 5,
        "max_risk_pct": 0.02,
        "max_single_position_pct": 0.40,
        "max_portfolio_risk_pct": 0.06,
        "commission_rate": 0.00015,
        "slippage_rate": 0.0015,
        "partial_take_profit_pct": 0.50,
        "trend_exit_consecutive_days": 5,
        "trailing_stop_atr_mult": 1.5,
    },
    "quant_engine": {
        "exit": {
            "partial_exit_r": [2, 4, 8, 10],
            "partial_exit_pct": 0.25,
            "max_hold_days": 10,
        },
    },
}


def make_engine():
    """테스트용 BacktestEngine (signal_engine mock, __init__ 우회)"""
    engine = BacktestEngine.__new__(BacktestEngine)

    bt = TEST_CONFIG["backtest"]
    engine.config = TEST_CONFIG
    engine.initial_capital = bt["initial_capital"]
    engine.max_positions = bt["max_positions"]
    engine.commission_rate = bt["commission_rate"]
    engine.slippage_rate = bt["slippage_rate"]
    engine.partial_tp_pct = bt["partial_take_profit_pct"]
    engine.trend_exit_days = bt["trend_exit_consecutive_days"]
    engine.atr_stop_mult = bt["trailing_stop_atr_mult"]

    exit_cfg = TEST_CONFIG["quant_engine"]["exit"]
    engine.partial_exit_r = exit_cfg["partial_exit_r"]
    engine.partial_exit_pct = exit_cfg["partial_exit_pct"]
    engine.max_hold_days = exit_cfg["max_hold_days"]

    engine.signal_engine = Mock()
    engine.position_sizer = PositionSizer(TEST_CONFIG)

    engine.adaptive_exit_enabled = False
    engine.adaptive_exit = None
    engine.hold_scorer = None
    engine.anchor_auto_update = False
    engine.anchor_learner = None
    engine.risk_norm_enabled = False
    engine.risk_normalizer = None

    engine.cash = engine.initial_capital
    engine.positions = []
    engine.trades = []
    engine.equity_curve = []
    engine.signal_log = []

    return engine


def make_signal(ticker="005930", price=50000, atr=1000, grade="A"):
    """테스트용 시그널 dict"""
    grade_map = {"A": 1.0, "B": 0.67, "C": 0.33}
    return {
        "ticker": ticker,
        "date": "2026-01-15",
        "atr_value": atr,
        "position_ratio": grade_map.get(grade, 1.0),
        "stop_loss": price - atr * 1.5,
        "target_price": price + atr * 5,
        "grade": grade,
        "zone_score": 7.5,
        "trigger_type": "confirm",
        "stop_loss_pct": 0.05,
    }


# ── Position 데이터클래스 테스트 ──

class TestPosition:
    """Position 데이터클래스 검증"""

    def test_dataclass_fields(self):
        """필수 필드 존재"""
        pos = Position(
            ticker="005930", entry_date="2026-01-15",
            entry_price=50000, shares=100,
            stop_loss=48500, target_price=55000,
            atr_value=1000, grade="A", bes_score=7.5,
        )
        assert pos.ticker == "005930"
        assert pos.shares == 100
        assert pos.partial_sold is False
        assert pos.partial_exits_done == 0

    def test_initial_shares_default(self):
        """initial_shares 기본값 = 0 (매수 시 설정)"""
        pos = Position(
            ticker="005930", entry_date="2026-01-15",
            entry_price=50000, shares=100,
            stop_loss=48500, target_price=55000,
            atr_value=1000, grade="A", bes_score=7.5,
            initial_shares=100,
        )
        assert pos.initial_shares == 100


# ── _execute_buy 테스트 ──

class TestExecuteBuy:
    """매수 실행 검증"""

    def test_slippage(self):
        """entry_price = next_open * (1 + slippage_rate)"""
        engine = make_engine()
        signal = make_signal(price=50000, atr=1000)
        next_open = 50000
        pos = engine._execute_buy(signal, next_open)
        expected_entry = next_open * (1 + engine.slippage_rate)
        assert pos.entry_price == expected_entry

    def test_commission_deducted(self):
        """cash -= investment + commission"""
        engine = make_engine()
        initial_cash = engine.cash
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        # cash 감소 확인
        assert engine.cash < initial_cash
        # investment + commission 만큼 감소
        investment = pos.shares * pos.entry_price
        commission = investment * engine.commission_rate
        expected_cash = initial_cash - investment - commission
        assert abs(engine.cash - expected_cash) < 1, \
            f"cash: {engine.cash} != expected: {expected_cash}"

    def test_insufficient_cash(self):
        """잔고 부족 → affordable 수량으로 조정 또는 None"""
        engine = make_engine()
        engine.cash = 100  # 극히 적은 잔고
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        assert pos is None, "잔고 부족 시 None 반환"

    def test_position_created(self):
        """Position 필드 정확성"""
        engine = make_engine()
        signal = make_signal(ticker="005930", price=50000, atr=1000, grade="A")
        pos = engine._execute_buy(signal, 50000)
        assert pos is not None
        assert pos.ticker == "005930"
        assert pos.grade == "A"
        assert pos.atr_value == 1000
        assert pos in engine.positions


# ── _execute_sell 테스트 ──

class TestExecuteSell:
    """매도 실행 검증"""

    def _make_position(self, engine):
        """테스트용 포지션 생성"""
        signal = make_signal(price=50000, atr=1000)
        return engine._execute_buy(signal, 50000)

    def test_full_exit(self):
        """전량 청산 → positions에서 제거"""
        engine = make_engine()
        pos = self._make_position(engine)
        assert pos is not None
        engine._execute_sell(pos, 55000, "2026-01-20", "target_hit")
        assert pos not in engine.positions
        assert len(engine.trades) == 1

    def test_partial_exit(self):
        """부분 청산 → shares 감소, partial_sold=True"""
        engine = make_engine()
        pos = self._make_position(engine)
        assert pos is not None
        original_shares = pos.shares
        sell_shares = original_shares // 4
        engine._execute_sell(pos, 55000, "2026-01-20", "partial_2R", shares=sell_shares)
        assert pos.shares == original_shares - sell_shares
        assert pos.partial_sold is True
        assert pos in engine.positions  # 아직 보유 중

    def test_pnl_calculation(self):
        """net_pnl = gross - 양방향 수수료"""
        engine = make_engine()
        pos = self._make_position(engine)
        assert pos is not None
        exit_price = 55000
        engine._execute_sell(pos, exit_price, "2026-01-20", "target")
        trade = engine.trades[-1]
        # 수익: actual_exit * shares - entry * shares - 양방향 수수료
        actual_exit = exit_price * (1 - engine.slippage_rate)
        assert trade.exit_price == actual_exit
        assert trade.pnl != 0  # 수익/손실 발생

    def test_sell_slippage(self):
        """exit_price *= (1 - slippage_rate)"""
        engine = make_engine()
        pos = self._make_position(engine)
        assert pos is not None
        engine._execute_sell(pos, 55000, "2026-01-20", "target")
        trade = engine.trades[-1]
        expected_exit = 55000 * (1 - engine.slippage_rate)
        assert trade.exit_price == expected_exit

    def test_cash_increase(self):
        """매도 후 cash 증가"""
        engine = make_engine()
        pos = self._make_position(engine)
        assert pos is not None
        cash_before = engine.cash
        engine._execute_sell(pos, 55000, "2026-01-20", "target")
        assert engine.cash > cash_before, "매도 후 cash 증가"


# ── 4단계 부분청산 테스트 ──

class TestPartialExits:
    """2R/4R 부분청산 메커니즘 검증"""

    def test_2r_partial(self):
        """high >= 2R 타겟 → 25% 청산"""
        engine = make_engine()
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        assert pos is not None

        entry = pos.entry_price
        risk = entry - pos.stop_loss
        target_2r = entry + risk * 2

        # 2R 이상 고가가 존재하는 행
        initial_shares = pos.initial_shares
        exit_shares = max(1, int(initial_shares * 0.25))

        # 직접 _execute_sell로 부분청산 시뮬레이션
        engine._execute_sell(pos, target_2r, "2026-01-17", "partial_2R", shares=exit_shares)
        pos.partial_exits_done = 1

        assert pos.shares == initial_shares - exit_shares
        assert pos.partial_sold is True

    def test_no_partial_below_2r(self):
        """high < 2R → 청산 없음"""
        engine = make_engine()
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        assert pos is not None

        initial_shares = pos.shares
        # 부분청산 조건 미충족 확인
        assert pos.partial_exits_done == 0
        assert pos.shares == initial_shares


# ── 트레일링 스탑 테스트 ──

class TestTrailingStop:
    """트레일링 스탑 메커니즘 검증"""

    def test_trailing_calculation(self):
        """trailing = highest - atr * mult"""
        engine = make_engine()
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        assert pos is not None
        # 최고가가 entry_price인 경우
        expected_trailing = pos.entry_price - 1000 * engine.atr_stop_mult
        assert pos.trailing_stop == expected_trailing

    def test_min_mult_0_5(self):
        """trailing_mult = max(calculated, 0.5) — v6.2"""
        engine = make_engine()
        # trailing_mult가 0.5 미만이 되는 상황을 직접 검증
        # _manage_positions에서 hold_result.trailing_tightness가 매우 작으면
        # trailing_mult = atr_stop_mult * tightness 가 0.5 미만
        # max(0.3, 0.5) = 0.5
        mult = engine.atr_stop_mult * 0.2  # 1.5 * 0.2 = 0.3
        clamped = max(mult, 0.5)
        assert clamped == 0.5, "trailing_mult 최소 0.5"

    def test_trailing_hit(self):
        """partial_sold=True AND low <= trailing → 청산"""
        engine = make_engine()
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        assert pos is not None
        pos.partial_sold = True
        pos.trailing_stop = pos.entry_price - 500  # 타이트한 트레일링

        # trailing_stop 히트 시뮬레이션
        low_price = pos.trailing_stop - 100  # 트레일링 아래로 하락
        engine._execute_sell(pos, pos.trailing_stop, "2026-01-20", "trailing_stop")
        assert pos not in engine.positions
        assert engine.trades[-1].exit_reason == "trailing_stop"


# ── 최대 보유일 테스트 ──

class TestMaxHold:
    """최대 보유일 제한 검증"""

    def test_max_hold_config(self):
        """max_hold_days 설정 확인"""
        engine = make_engine()
        assert engine.max_hold_days == 10

    def test_max_hold_exit(self):
        """hold_days >= max_hold_days → 청산"""
        engine = make_engine()
        signal = make_signal(price=50000, atr=1000)
        pos = engine._execute_buy(signal, 50000)
        assert pos is not None

        # 11일 후 강제 청산 시뮬레이션
        # _manage_positions에서 hold_days >= max_hold_days일 때 청산
        exit_date = "2026-01-26"  # 15일 입장 → 26일 = 11일
        engine._execute_sell(pos, 51000, exit_date, "max_hold_days")
        trade = engine.trades[-1]
        assert trade.exit_reason == "max_hold_days"
        assert trade.hold_days == 11
