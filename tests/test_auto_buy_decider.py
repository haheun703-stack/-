"""test_auto_buy_decider.py — 자비스 안전선 9건 유닛 테스트 (2026-05-21)

배경:
  5/21 commit c8518b7로 MAX_AMOUNT/MAX_QTY/MAX_DAILY_BUYS를 .env 동적 로드로 일반화.
  ENV_FLAG_ENABLED="AUTO_TRADING_ENABLED"로 변경 (레거시 AUTO_TRADE_5_20 호환).
  5/22 14:00 자비스 cron 가동 직전 의도 vs 행동 검증.

검증 시나리오:
  1. 시간 14:00 미달 (13:55) → SKIP
  2. 점수 100 + 14:00 + regime OK + 모든 조건 충족 → BUY
  3. 점수 85 (< 90) → SKIP (안전선 ① 미달)
  4. EYE 필터 SKIP → SKIP (안전선 ②)
  5. regime CAUTION → SKIP (안전선 ⑥)
  6. 1주 가격 350만 (> MAX_AMOUNT 300만) → SKIP (안전선 ⑤)
  7. AUTO_TRADING_ENABLED=0 + AUTO_TRADE_5_20=true (레거시) → 통과
  8. AUTO_TRADING_ENABLED=0 + AUTO_TRADE_5_20=false → SKIP (안전선 ⑦)
  9. NEGA 차단 → SKIP (안전선 ⑧)

실행:
  python -m pytest tests/test_auto_buy_decider.py -v
  또는: python tests/test_auto_buy_decider.py
"""

import sys
import os
import unittest
from unittest.mock import patch

# 프로젝트 루트 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAutoBuyDecider(unittest.TestCase):
    """자비스 should_auto_buy() 안전선 9건 검증"""

    def setUp(self):
        """각 테스트 전 환경변수 초기화"""
        # AUTO_TRADING_ENABLED 기본 1 (.env 일치)
        os.environ["AUTO_TRADING_ENABLED"] = "1"
        os.environ["AUTO_TRADE_5_20"] = "false"  # 레거시 비활성
        os.environ["AUTO_TRADING_MAX_QTY"] = "1"
        os.environ["AUTO_TRADING_MAX_AMOUNT"] = "3000000"
        os.environ["AUTO_TRADING_MAX_TRADES_PER_DAY"] = "15"

        # auto_buy_decider 모듈 강제 reload (환경변수 다시 읽기)
        import importlib
        from src.use_cases import auto_buy_decider
        importlib.reload(auto_buy_decider)
        self.decider = auto_buy_decider

    def test_1_time_before_1400_skip(self):
        """[1] 시간 13:55 (14:00 미달) → SKIP"""
        d = self.decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=54500,
            now_str="13:55", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("진입 시간 미달" in c for c in d.checks_failed))
        print("OK [1] 13:55 -> SKIP (time before 14:00)")

    def test_2_all_pass_buy(self):
        """[2] 모든 안전선 통과 → BUY"""
        d = self.decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=54500,
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "BUY", f"실패 사유: {d.checks_failed}")
        self.assertEqual(d.qty, 1)
        self.assertEqual(d.estimated_price, 54500)
        print("OK [2] all pass -> BUY (점수 100, 14:00, MILD_BULL, 54500원)")

    def test_3_score_below_90_skip(self):
        """[3] 점수 85 (< 90) → SKIP (안전선 ① 미달)"""
        d = self.decider.should_auto_buy(
            ticker="010060", name="OCI홀딩스",
            integrated_score=85.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=277500,
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("점수 85" in c for c in d.checks_failed))
        print("OK [3] score=85 < 90 -> SKIP (안전선 ①)")

    def test_4_eye_skip(self):
        """[4] EYE 필터 SKIP → SKIP (안전선 ②)"""
        d = self.decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=True,
            eye_skip_reasons=["RSI 80 과열", "거래량 부족"],
            market_regime="MILD_BULL", current_price=54500,
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("EYE SKIP" in c for c in d.checks_failed))
        print("OK [4] EYE SKIP -> SKIP (안전선 ②)")

    def test_5_regime_caution_skip(self):
        """[5] regime CAUTION → SKIP (안전선 ⑥)"""
        d = self.decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="CAUTION", current_price=54500,
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("CAUTION" in c for c in d.checks_failed))
        print("OK [5] regime=CAUTION -> SKIP (안전선 ⑥)")

    def test_6_price_over_max_amount_skip(self):
        """[6] 1주 350만 (> MAX_AMOUNT 300만) → SKIP (안전선 ⑤)

        commit c8518b7 검증: MAX_AMOUNT=3000000 .env 동적 로드"""
        d = self.decider.should_auto_buy(
            ticker="298040", name="효성중공업",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=3500000,  # 350만 > 300만
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("3,500,000" in c or "300만" in c or "위반" in c
                            for c in d.checks_failed))
        print("OK [6] 1주 350만 > MAX_AMOUNT 300만 -> SKIP (안전선 ⑤)")

    def test_7_legacy_auto_trade_5_20_compat(self):
        """[7] AUTO_TRADING_ENABLED=0 + AUTO_TRADE_5_20=true (레거시 호환) → 통과

        commit c8518b7 검증: 레거시 토글 호환 분기"""
        os.environ["AUTO_TRADING_ENABLED"] = "0"
        os.environ["AUTO_TRADE_5_20"] = "true"
        import importlib
        from src.use_cases import auto_buy_decider
        importlib.reload(auto_buy_decider)
        d = auto_buy_decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=54500,
            now_str="14:00", today="2026-05-21"
        )
        # ⑦ 안전선 통과 (레거시 호환) → 다른 안전선 다 통과 시 BUY
        self.assertEqual(d.action, "BUY", f"레거시 호환 실패: {d.checks_failed}")
        print("OK [7] AUTO_TRADING_ENABLED=0 + AUTO_TRADE_5_20=true -> BUY (레거시 호환)")

    def test_8_all_toggle_off_skip(self):
        """[8] AUTO_TRADING_ENABLED=0 + AUTO_TRADE_5_20=false → SKIP (안전선 ⑦)"""
        os.environ["AUTO_TRADING_ENABLED"] = "0"
        os.environ["AUTO_TRADE_5_20"] = "false"
        import importlib
        from src.use_cases import auto_buy_decider
        importlib.reload(auto_buy_decider)
        d = auto_buy_decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=54500,
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("AUTO_TRADING_ENABLED" in c or "AUTO_TRADE_5_20" in c
                            for c in d.checks_failed))
        print("OK [8] 모든 토글 OFF -> SKIP (안전선 ⑦)")

    def test_9_nega_blocked_skip(self):
        """[9] 막내 NEGA 차단 → SKIP (안전선 ⑧)"""
        d = self.decider.should_auto_buy(
            ticker="067310", name="하나마이크론",
            integrated_score=100.0, eye_should_skip=False, eye_skip_reasons=[],
            market_regime="MILD_BULL", current_price=54500,
            blocked_by_nega=True,
            now_str="14:00", today="2026-05-21"
        )
        self.assertEqual(d.action, "SKIP")
        self.assertTrue(any("NEGA" in c for c in d.checks_failed))
        print("OK [9] NEGA blocked -> SKIP (안전선 ⑧)")


class TestEnvDynamicLoad(unittest.TestCase):
    """환경변수 동적 로드 검증 (commit c8518b7)"""

    def test_10_max_amount_dynamic_load(self):
        """[10] MAX_AMOUNT 환경변수 변경 → 코드 즉시 반영"""
        os.environ["AUTO_TRADING_MAX_AMOUNT"] = "5000000"  # 500만
        import importlib
        from src.use_cases import auto_buy_decider
        importlib.reload(auto_buy_decider)
        self.assertEqual(auto_buy_decider.MAX_AMOUNT, 5_000_000)

        # 다시 300만으로 복원
        os.environ["AUTO_TRADING_MAX_AMOUNT"] = "3000000"
        importlib.reload(auto_buy_decider)
        self.assertEqual(auto_buy_decider.MAX_AMOUNT, 3_000_000)
        print("OK [10] MAX_AMOUNT .env 동적 로드 OK")

    def test_11_max_trades_per_day_dynamic_load(self):
        """[11] MAX_DAILY_BUYS 환경변수 변경 → 코드 즉시 반영"""
        os.environ["AUTO_TRADING_MAX_TRADES_PER_DAY"] = "10"
        import importlib
        from src.use_cases import auto_buy_decider
        importlib.reload(auto_buy_decider)
        self.assertEqual(auto_buy_decider.MAX_DAILY_BUYS, 10)

        os.environ["AUTO_TRADING_MAX_TRADES_PER_DAY"] = "15"
        importlib.reload(auto_buy_decider)
        self.assertEqual(auto_buy_decider.MAX_DAILY_BUYS, 15)
        print("OK [11] MAX_DAILY_BUYS .env 동적 로드 OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  auto_buy_decider.py 11 unit tests")
    print("  자비스 안전선 9건 + .env 동적 로드 (commit c8518b7) 검증")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
