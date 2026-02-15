"""
v6.3 PositionSizer 단위 테스트

테스트 대상: src/position_sizer.py
- 기본 공식: (잔고 × 리스크%) / (ATR × mult) × grade × stage × vol_weight
- 한도: max_single_pct(40%), max_portfolio_risk_pct(6%)
- 엣지 케이스: ATR=0, 잔고=0, 가격=0
- 결과 구조: 필수 필드 존재
"""

import pytest

from src.position_sizer import PositionSizer


# ── 테스트용 config ──

TEST_CONFIG = {
    "backtest": {
        "max_risk_pct": 0.02,
        "max_single_position_pct": 0.40,
        "max_portfolio_risk_pct": 0.06,
        "trailing_stop_atr_mult": 1.5,
    },
}


def make_sizer():
    return PositionSizer(TEST_CONFIG)


# ── 기본 공식 테스트 ──

class TestBasicSizing:
    """포지션 사이징 기본 공식 검증"""

    def test_basic_formula(self):
        """잔고=10M, ATR=1000, A등급(1.0), stage=0.4"""
        sizer = make_sizer()
        result = sizer.calculate(
            account_balance=10_000_000,
            entry_price=50000,
            atr_value=1000,
            grade_ratio=1.0,
            stage_pct=0.4,
        )
        # raw = (10M * 0.02) / (1000 * 1.5) = 133
        # adjusted = 133 * 1.0 * 0.4 = 53
        assert result["shares"] == 53, f"기본 공식 → 53주, 실제: {result['shares']}"

    def test_grade_ratio_effect(self):
        """A(1.0) > B(0.67) > C(0.33) 비례 — max_single_pct 캡 미적용 조건"""
        sizer = make_sizer()
        # ATR=5000 → raw=int(200000/7500)=26, 캡(80)에 걸리지 않음
        shares_a = sizer.calculate(10_000_000, 50000, 5000, 1.0)["shares"]
        shares_b = sizer.calculate(10_000_000, 50000, 5000, 0.67)["shares"]
        shares_c = sizer.calculate(10_000_000, 50000, 5000, 0.33)["shares"]
        assert shares_a > shares_b > shares_c, \
            f"A({shares_a}) > B({shares_b}) > C({shares_c}) 불성립"

    def test_stage_pct_effect(self):
        """Impulse(0.4) == Confirm(0.4) > Breakout(0.2)"""
        sizer = make_sizer()
        impulse = sizer.calculate(10_000_000, 50000, 1000, 1.0, stage_pct=0.4)["shares"]
        confirm = sizer.calculate(10_000_000, 50000, 1000, 1.0, stage_pct=0.4)["shares"]
        breakout = sizer.calculate(10_000_000, 50000, 1000, 1.0, stage_pct=0.2)["shares"]
        assert impulse == confirm, "Impulse == Confirm"
        assert impulse > breakout, "Impulse > Breakout"

    def test_vol_weight_half(self):
        """vol_weight=0.5 → shares 절반 — max_single_pct 캡 미적용 조건"""
        sizer = make_sizer()
        # ATR=5000 → raw 26주, 캡에 걸리지 않아 vol_weight 효과 정확히 반영
        full = sizer.calculate(10_000_000, 50000, 5000, 1.0, stage_pct=1.0,
                               vol_normalized_weight=1.0)["shares"]
        half = sizer.calculate(10_000_000, 50000, 5000, 1.0, stage_pct=1.0,
                               vol_normalized_weight=0.5)["shares"]
        assert half == int(full * 0.5), f"vol=0.5 → shares 절반: {half} vs {int(full*0.5)}"


# ── 한도 테스트 ──

class TestLimits:
    """포지션 한도 제약 검증"""

    def test_max_single_pct(self):
        """40% 한도 초과 시 클리핑"""
        sizer = make_sizer()
        # 매우 작은 ATR → 큰 raw_shares → max_single_pct로 제한
        result = sizer.calculate(
            account_balance=10_000_000,
            entry_price=1000,  # 저가
            atr_value=10,      # 작은 ATR
            grade_ratio=1.0,
            stage_pct=1.0,
        )
        max_shares = int(10_000_000 * 0.40 / 1000)  # 4000주
        assert result["shares"] <= max_shares, \
            f"40% 한도 초과: {result['shares']} > {max_shares}"

    def test_portfolio_risk_limit(self):
        """이미 5.5% 리스크 → 남은 0.5%만 할당"""
        sizer = make_sizer()
        balance = 10_000_000
        current_risk = balance * 0.055  # 5.5%
        result = sizer.calculate(
            account_balance=balance,
            entry_price=50000,
            atr_value=1000,
            grade_ratio=1.0,
            current_portfolio_risk=current_risk,
        )
        remaining_risk = balance * 0.06 - current_risk  # 50,000원
        max_by_risk = int(remaining_risk / (1000 * 1.5))  # 33주
        assert result["shares"] <= max_by_risk, \
            f"포트폴리오 리스크 한도 초과: {result['shares']} > {max_by_risk}"

    def test_portfolio_risk_full(self):
        """6% 도달 → shares=0"""
        sizer = make_sizer()
        balance = 10_000_000
        current_risk = balance * 0.06  # 이미 6% 꽉 참
        result = sizer.calculate(
            account_balance=balance,
            entry_price=50000,
            atr_value=1000,
            grade_ratio=1.0,
            current_portfolio_risk=current_risk,
        )
        assert result["shares"] == 0, "포트폴리오 리스크 6% → shares=0"

    def test_zero_atr(self):
        """ATR=0 → empty result"""
        sizer = make_sizer()
        result = sizer.calculate(10_000_000, 50000, 0, 1.0)
        assert result["shares"] == 0

    def test_zero_balance(self):
        """잔고=0 → empty result"""
        sizer = make_sizer()
        result = sizer.calculate(0, 50000, 1000, 1.0)
        assert result["shares"] == 0

    def test_zero_price(self):
        """가격=0 → empty result"""
        sizer = make_sizer()
        result = sizer.calculate(10_000_000, 0, 1000, 1.0)
        assert result["shares"] == 0


# ── 결과 구조 테스트 ──

class TestResultStructure:
    """결과 딕셔너리 필드 검증"""

    def test_all_fields(self):
        """필수 필드 7개 존재"""
        sizer = make_sizer()
        result = sizer.calculate(10_000_000, 50000, 1000, 1.0)
        expected_keys = {
            "shares", "investment", "risk_amount",
            "stop_loss", "target", "stop_distance", "pct_of_account",
        }
        assert expected_keys == set(result.keys()), \
            f"누락 필드: {expected_keys - set(result.keys())}"

    def test_investment_calculation(self):
        """investment = shares × entry_price"""
        sizer = make_sizer()
        result = sizer.calculate(10_000_000, 50000, 1000, 1.0)
        expected_invest = result["shares"] * 50000
        assert result["investment"] == expected_invest

    def test_stop_loss_calculation(self):
        """stop_loss = entry - ATR × mult"""
        sizer = make_sizer()
        result = sizer.calculate(10_000_000, 50000, 1000, 1.0)
        expected_sl = int(50000 - 1000 * 1.5)
        assert result["stop_loss"] == expected_sl, \
            f"stop_loss: {result['stop_loss']} != {expected_sl}"
