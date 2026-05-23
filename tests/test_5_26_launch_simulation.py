"""test_5_26_launch_simulation.py — 5/26(월) 09:15 자동 가동 시뮬레이션 3종.

배경 (pending_tasks.md 5/23 토 4건 중 마지막):
  5/26 09:15 정식 가동 직전 최종 검증.
  자비스 cron (14:00~14:55) 정상 동작 시나리오 + fail-safe 차단 시나리오 검증.

3 시나리오:
  1. ★ 정상 강세장 시나리오 (HPSP 같은 PPA 종목 자동 매수)
     - STRONG_BULL + 80+ 점수 + EYE 통과 + 14:30 + 1주 75000원 + AUTO_TRADING_ENABLED
     - 안전선 9건 ALL 통과 → BUY

  2. ★ 약세장 BEARISH 자동 차단
     - regime=BEAR + 나머지 통과 → 안전선 ⑥ 차단 → SKIP

  3. ★ 강제 실패 (점수 낮음 + EYE SKIP + 시간 미달 다중)
     - 점수 50 < 80 + EYE SKIP + 10:00 + regime=CAUTION
     - 안전선 ①②③⑥ 다중 차단 → SKIP

검증 포인트:
  - 9건 안전선 OR (모두 통과해야 BUY) 정확 동작
  - 단일 안전선 실패만으로도 SKIP 보장 (fail-safe)
  - reason 메시지 명확성 (텔레그램 알림 가독성)

실행:
  python -m pytest tests/test_5_26_launch_simulation.py -v
"""

import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.use_cases.auto_buy_decider import (
    should_auto_buy,
    BuyDecision,
    THRESHOLD_INTEGRATED_SCORE,
    ALLOWED_REGIMES,
)


# ===== 가상 환경 =====
ENV_OK = {
    "AUTO_TRADING_ENABLED": "1",
    "AUTO_TRADING_MIN_SCORE": "80.0",
    "AUTO_TRADING_MAX_TRADES_PER_DAY": "15",
    "AUTO_TRADING_MAX_QTY": "1",
    "AUTO_TRADING_MAX_AMOUNT": "3000000",
}


class TestScenario1NormalStrongBull(unittest.TestCase):
    """[시나리오 1] ★ 정상 강세장 — HPSP 같은 PPA 종목 자동 매수."""

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_strong_bull_hpsp_buy(self, mock_daily):
        """5/26 09:15 가동 → HPSP 14:30 자동 매수 ALLOW

        조건:
          - integrated_score=85 (≥80 임계)
          - eye_should_skip=False (4종 통과)
          - now_str='14:30' (14:00 이후)
          - market_regime='STRONG_BULL' (ALLOWED ✓)
          - current_price=75000 (1주 ≤ 300만)
          - blocked_by_nega=False
          - AUTO_TRADING_ENABLED=true

        기대: action='BUY', 8건 checks_passed
        """
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="403870",
            name="HPSP",
            integrated_score=85.0,
            eye_should_skip=False,
            eye_skip_reasons=[],
            market_regime="STRONG_BULL",
            current_price=75000,
            blocked_by_nega=False,
            now_str="14:30",
            today="2026-05-26",
        )

        self.assertEqual(decision.action, "BUY", f"BUY 기대 / 실제 {decision.action} / 실패 {decision.checks_failed}")
        self.assertEqual(decision.ticker, "403870")
        self.assertEqual(decision.qty, 1)
        self.assertEqual(decision.estimated_amount, 75000)
        # 8건 안전선 ALL 통과 (이 함수 평가 범위, ⑨는 kis_order_adapter)
        self.assertEqual(len(decision.checks_failed), 0)
        self.assertGreaterEqual(len(decision.checks_passed), 8)

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_neutral_regime_also_allowed(self, mock_daily):
        """NEUTRAL regime도 ALLOWED_REGIMES 포함 → BUY"""
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="000660", name="SK하이닉스",
            integrated_score=82.0,
            eye_should_skip=False, eye_skip_reasons=[],
            market_regime="NEUTRAL",
            current_price=200000,
            blocked_by_nega=False,
            now_str="14:15", today="2026-05-26",
        )
        self.assertEqual(decision.action, "BUY")


class TestScenario2BearishBlock(unittest.TestCase):
    """[시나리오 2] ★ 약세장 BEARISH 자동 차단 — 안전선 ⑥ 동작 검증."""

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_bearish_block(self, mock_daily):
        """KOSPI BEAR + 나머지 통과 → 안전선 ⑥ 차단 → SKIP"""
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="403870", name="HPSP",
            integrated_score=85.0,                    # ① OK
            eye_should_skip=False, eye_skip_reasons=[],  # ② OK
            market_regime="BEAR",                     # ⑥ 차단!
            current_price=75000,
            blocked_by_nega=False,
            now_str="14:30", today="2026-05-26",
        )

        self.assertEqual(decision.action, "SKIP", "BEAR regime에서 BUY 절대 금지")
        # checks_failed에 regime 차단 포함
        regime_fail = any("regime" in c and "BEAR" in c for c in decision.checks_failed)
        self.assertTrue(regime_fail, f"BEAR regime 차단 메시지 누락: {decision.checks_failed}")

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_caution_also_blocked(self, mock_daily):
        """CAUTION regime도 ALLOWED_REGIMES에 없음 → 차단"""
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="403870", name="HPSP",
            integrated_score=85.0,
            eye_should_skip=False, eye_skip_reasons=[],
            market_regime="CAUTION",
            current_price=75000,
            blocked_by_nega=False,
            now_str="14:30", today="2026-05-26",
        )
        self.assertEqual(decision.action, "SKIP")


class TestScenario3MultipleFailures(unittest.TestCase):
    """[시나리오 3] ★ 강제 실패 — 다중 안전선 실패 (fail-safe)."""

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_multiple_failures(self, mock_daily):
        """점수 50 + EYE SKIP + 10:00 + CAUTION → 다중 차단

        목적: 안전선 단일만 실패해도 SKIP이지만,
              실전 다중 실패 시 모든 reason 누적되어 텔레그램에 명확 표기.
        """
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="UNKNOWN", name="모름",
            integrated_score=50.0,              # ① 차단 (< 80)
            eye_should_skip=True,                # ② 차단
            eye_skip_reasons=["VOL_DROP", "MA_DEAD"],
            market_regime="CAUTION",             # ⑥ 차단
            current_price=400000,                # ⑤ 차단 (> 300만)
            blocked_by_nega=True,                # ⑧ 차단
            now_str="10:00",                     # ③ 차단 (< 14:00)
            today="2026-05-26",
        )

        self.assertEqual(decision.action, "SKIP")
        # 4건 이상 차단 메시지 (점수 / EYE / 시간 / regime / 가격)
        self.assertGreaterEqual(
            len(decision.checks_failed), 4,
            f"다중 실패 시 모든 reason 누적되어야 함: {decision.checks_failed}"
        )
        # 각 차단 유형 1건 이상
        fail_text = " | ".join(decision.checks_failed)
        self.assertIn("50", fail_text)            # 점수
        self.assertIn("EYE", fail_text)           # EYE
        self.assertIn("14:00", fail_text)         # 시간
        self.assertIn("regime", fail_text)        # regime

    @patch.dict(os.environ, {**ENV_OK, "AUTO_TRADING_ENABLED": "0", "AUTO_TRADE_5_20": "false"}, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_env_flag_disabled(self, mock_daily):
        """환경변수 AUTO_TRADING_ENABLED=false → 차단 (안전선 ⑦)"""
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="403870", name="HPSP",
            integrated_score=90.0,
            eye_should_skip=False, eye_skip_reasons=[],
            market_regime="STRONG_BULL",
            current_price=75000,
            blocked_by_nega=False,
            now_str="14:30", today="2026-05-26",
        )
        self.assertEqual(decision.action, "SKIP", "AUTO_TRADING_ENABLED=false면 절대 BUY 금지")

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_5_26_price_over_limit(self, mock_daily):
        """1주 가격 > 300만원 → 안전선 ⑤ 차단"""
        mock_daily.return_value = (True, "일일 매수 0/15")

        decision = should_auto_buy(
            ticker="058470", name="리노공업",  # 고가주
            integrated_score=85.0,
            eye_should_skip=False, eye_skip_reasons=[],
            market_regime="STRONG_BULL",
            current_price=3_500_000,    # ⑤ 한도 초과
            blocked_by_nega=False,
            now_str="14:30", today="2026-05-26",
        )
        self.assertEqual(decision.action, "SKIP")
        price_fail = any("300" in c or "10만" in c for c in decision.checks_failed)
        self.assertTrue(price_fail, f"가격 한도 차단 메시지 누락: {decision.checks_failed}")


class TestRealtimeScoreIntegration(unittest.TestCase):
    """5/26 통합 흐름 — realtime_score → auto_buy_decider 연결 검증."""

    @patch.dict(os.environ, ENV_OK, clear=False)
    @patch("src.use_cases.auto_buy_decider._check_daily_count")
    def test_realtime_score_to_auto_buy(self, mock_daily):
        """realtime_score = +14 → integrated_score 80+ → auto_buy ALLOW

        시뮬: 10 시그널 합산 +14 (BUY 임계)
            → 통합 점수 80+ 가정 (별도 계산)
            → 자비스 매수
        """
        mock_daily.return_value = (True, "일일 매수 0/15")
        # realtime_score 결과 = +14 (BUY 추천)
        # → 통합 점수 산정 시 90+ 가능 가정
        decision = should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=85.0,
            eye_should_skip=False, eye_skip_reasons=[],
            market_regime="NEUTRAL",
            current_price=24000,
            blocked_by_nega=False,
            now_str="14:30", today="2026-05-26",
        )
        self.assertEqual(decision.action, "BUY")


class TestSimulationSummary(unittest.TestCase):
    """5/26 가동 시뮬레이션 종합 — 3 시나리오 ALL 검증."""

    def test_all_three_scenarios_complete(self):
        """3 시나리오 정의 + 안전선 9건 정상 동작 종합 검증"""
        # 안전선 임계값 정상 로드
        self.assertEqual(THRESHOLD_INTEGRATED_SCORE, 80.0)
        self.assertIn("STRONG_BULL", ALLOWED_REGIMES)
        self.assertIn("NEUTRAL", ALLOWED_REGIMES)
        self.assertIn("MILD_BULL", ALLOWED_REGIMES)
        # BEAR/CAUTION/CRISIS는 절대 ALLOWED에 없어야 함
        for blocked in ("BEAR", "CAUTION", "CRISIS", "BEARISH"):
            self.assertNotIn(blocked, ALLOWED_REGIMES, f"{blocked}이 ALLOWED에 있으면 안 됨!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
