"""test_adaptive_reentry.py — MVP-4 자동 재진입 단위 테스트.

배경 (퐝가님 5/23 흐름):
  3중 검증 게이트 ALL 통과해야 재진입:
    1. MVP-3 받침 시그널
    2. STEP 5 ★★★ 이상
    3. 자비스 9 안전선

5/17 자기반성 #1: import + 함수 호출 + 통합 흐름 검증.

검증 시나리오:
  [3 게이트 평가]
   1. 3 게이트 ALL 통과 → trigger=True
   2. 받침 실패 → trigger=False
   3. STEP5 ★★ (3 미만) → trigger=False
   4. 자비스 실패 → trigger=False

  [중복 진입 방지]
   5. MVP-2 큐 활성 → already_in_queue=True + trigger=False (위임)

  [자동 매수 실행]
   6. AUTO_REENTRY=0 → 알림만 (success=False, 에러 메시지)
   7. AUTO_REENTRY=1 + trigger + buy 성공 → success=True + order_id
   8. AUTO_REENTRY=1 + trigger + buy 실패 → success=False + error

  [방어]
   9. KILL_SWITCH 발동 시 모든 평가 스킵

  [포맷]
  10. trigger 포맷 (3 게이트 + 매수 정보)
  11. already_in_queue 포맷 (위임 안내)
  12. 미트리거 포맷 (실패 게이트 명시)

  [pool 스캔]
  13. scan_pool_for_reentry — trigger 우선 정렬

  [import]
  14. 모듈 + 상수 노출

실행:
  python -m pytest tests/test_adaptive_reentry.py -v
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_support_signal(trigger: bool):
    """가짜 SupportSignal."""
    from src.use_cases.support_pattern_detector import SupportSignal
    sig = SupportSignal(ticker="240810")
    sig.trigger = trigger
    if trigger:
        sig.reasons_pass = ["받침 통과"]
    else:
        sig.reasons_fail = ["아래꼬리 부족"]
    return sig


def _mock_broker(current_price: int = 22_000, buy_success: bool = True, order_id: str = "REORD001"):
    broker = MagicMock()
    broker.fetch_price.return_value = {"output": {"stck_prpr": str(current_price)}}
    if buy_success:
        order_obj = MagicMock()
        order_obj.order_id = order_id
        broker.buy_market.return_value = order_obj
    else:
        broker.buy_market.side_effect = Exception("KIS 매수 실패")
    return broker


class TestAdaptiveReentry(unittest.TestCase):

    def setUp(self):
        # 모든 관련 모듈 reload
        for mod_name in (
            "src.use_cases.adaptive_reentry",
            "src.use_cases.support_pattern_detector",
            "src.use_cases.adaptive_buy_queue",
        ):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)

        import src.use_cases.adaptive_reentry as mvp4
        import src.use_cases.support_pattern_detector as mvp3
        import src.use_cases.adaptive_buy_queue as mvp2
        self.mvp4 = mvp4
        self.mvp3 = mvp3
        self.mvp2 = mvp2

        # KILL_SWITCH 격리 (모든 모듈 동일 경로)
        kill_path = tmp_path / "kill_switch.flag"
        mvp4.KILL_SWITCH_PATH = kill_path
        mvp3.KILL_SWITCH_PATH = kill_path
        mvp2.KILL_SWITCH_PATH = kill_path
        mvp2.QUEUE_PATH = tmp_path / "adaptive_buy_queue.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _step5_pool(self, stars: int = 3, upside: float = 5.0) -> dict:
        return {"240810": {"stars": stars, "upside": upside}}

    def _jarvis_pass(self) -> callable:
        return lambda ticker: {"pass": True, "failed": []}

    def _jarvis_fail(self) -> callable:
        return lambda ticker: {"pass": False, "failed": ["매크로 BEARISH", "시간대 불일치"]}

    # ─────────────────────────────────────────────────────────
    # [3 게이트 평가]
    # ─────────────────────────────────────────────────────────

    def test_01_all_pass(self):
        """3 게이트 ALL 통과 → trigger=True."""
        broker = _mock_broker()
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(True)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810", "원익IPS",
                step5_pool=self._step5_pool(stars=4, upside=5.0),
                jarvis_safety_check=self._jarvis_pass(),
            )

        self.assertTrue(dec.trigger)
        self.assertTrue(dec.support_pass)
        self.assertTrue(dec.step5_pass)
        self.assertTrue(dec.jarvis_pass)
        self.assertEqual(dec.step5_stars, 4)
        self.assertGreater(dec.target_qty, 0)

    def test_02_support_fail(self):
        """받침 실패 → trigger=False."""
        broker = _mock_broker()
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(False)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_pass(),
            )
        self.assertFalse(dec.trigger)
        self.assertFalse(dec.support_pass)
        self.assertTrue(dec.step5_pass)
        self.assertTrue(dec.jarvis_pass)

    def test_03_step5_below_min(self):
        """STEP5 ★★ (3 미만) → trigger=False."""
        broker = _mock_broker()
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(True)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(stars=2),
                jarvis_safety_check=self._jarvis_pass(),
            )
        self.assertFalse(dec.trigger)
        self.assertFalse(dec.step5_pass)
        self.assertEqual(dec.step5_stars, 2)

    def test_04_jarvis_fail(self):
        """자비스 안전선 실패 → trigger=False."""
        broker = _mock_broker()
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(True)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_fail(),
            )
        self.assertFalse(dec.trigger)
        self.assertFalse(dec.jarvis_pass)
        self.assertEqual(len(dec.jarvis_failed_checks), 2)

    # ─────────────────────────────────────────────────────────
    # [중복 진입 방지]
    # ─────────────────────────────────────────────────────────

    def test_05_already_in_queue_skipped(self):
        """MVP-2 큐 활성 → already_in_queue=True + trigger=False."""
        # MVP-2에 등록
        self.mvp2.register_buy_queue("240810", 25_000, 3_000_000, name="원익IPS")

        broker = _mock_broker()
        # 받침 등 평가는 호출도 안 됨
        with patch.object(self.mvp3, "detect_support_pattern") as mock_support:
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_pass(),
            )

        self.assertTrue(dec.already_in_queue)
        self.assertFalse(dec.trigger)
        # 받침 검사 호출 안 됨 (위임)
        mock_support.assert_not_called()

    # ─────────────────────────────────────────────────────────
    # [자동 매수 실행]
    # ─────────────────────────────────────────────────────────

    def test_06_auto_reentry_off_alert_only(self):
        """AUTO_REENTRY=0 → 알림만."""
        self.assertFalse(self.mvp4.AUTO_REENTRY)

        broker = _mock_broker()
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(True)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_pass(),
            )
            self.assertTrue(dec.trigger)
            self.assertFalse(dec.auto_reentry_eligible)

            result = self.mvp4.execute_auto_reentry(broker, dec)
            self.assertFalse(result["success"])
            self.assertIn("AUTO_REENTRY=0", result["error"])
            broker.buy_market.assert_not_called()

    def test_07_auto_reentry_on_success(self):
        """AUTO_REENTRY=1 + 매수 성공 → success=True + order_id."""
        self.mvp4.AUTO_REENTRY = True

        broker = _mock_broker(current_price=22_000, buy_success=True, order_id="REORD007")
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(True)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_pass(),
            )
            self.assertTrue(dec.auto_reentry_eligible)

            result = self.mvp4.execute_auto_reentry(broker, dec)
            self.assertTrue(result["success"])
            self.assertEqual(result["order_id"], "REORD007")
            self.assertEqual(result["price"], 22_000)
            broker.buy_market.assert_called_once()

    def test_08_auto_reentry_on_failure(self):
        """AUTO_REENTRY=1 + 매수 실패 → success=False + error."""
        self.mvp4.AUTO_REENTRY = True

        broker = _mock_broker(buy_success=False)
        with patch.object(self.mvp3, "detect_support_pattern", return_value=_mock_support_signal(True)):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_pass(),
            )
            result = self.mvp4.execute_auto_reentry(broker, dec)
            self.assertFalse(result["success"])
            self.assertIn("KIS 매수 실패", result["error"])

    # ─────────────────────────────────────────────────────────
    # [방어]
    # ─────────────────────────────────────────────────────────

    def test_09_kill_switch_blocks_all(self):
        """KILL_SWITCH 발동 시 모든 평가 스킵."""
        self.mvp4.KILL_SWITCH_PATH.write_text("active", encoding="utf-8")

        broker = _mock_broker()
        with patch.object(self.mvp3, "detect_support_pattern") as mock_support:
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool=self._step5_pool(),
                jarvis_safety_check=self._jarvis_pass(),
            )

        self.assertFalse(dec.trigger)
        self.assertTrue(any("KILL_SWITCH" in c for c in dec.jarvis_failed_checks))
        mock_support.assert_not_called()

    # ─────────────────────────────────────────────────────────
    # [포맷]
    # ─────────────────────────────────────────────────────────

    def test_10_format_trigger(self):
        """trigger 포맷 — 3 게이트 + 매수 정보."""
        from src.use_cases.adaptive_reentry import ReentryDecision
        dec = ReentryDecision(
            ticker="240810", name="원익IPS",
            support_pass=True, step5_pass=True, jarvis_pass=True,
            step5_stars=4, step5_upside=5.0,
            trigger=True, auto_reentry_eligible=False,
            target_price=22_000, target_qty=45, target_amount=1_000_000,
        )
        msg = self.mvp4.format_reentry_for_telegram(dec)
        self.assertIn("원익IPS", msg)
        self.assertIn("받침: ✓", msg)
        self.assertIn("STEP5: ✓", msg)
        self.assertIn("자비스: ✓", msg)
        self.assertIn("45주", msg)
        self.assertIn("알림만", msg)

    def test_11_format_already_in_queue(self):
        from src.use_cases.adaptive_reentry import ReentryDecision
        dec = ReentryDecision(ticker="240810", name="원익IPS", already_in_queue=True)
        msg = self.mvp4.format_reentry_for_telegram(dec)
        self.assertIn("스킵", msg)
        self.assertIn("위임", msg)

    def test_12_format_no_trigger(self):
        from src.use_cases.adaptive_reentry import ReentryDecision
        dec = ReentryDecision(
            ticker="240810", name="원익IPS",
            support_pass=False, step5_pass=True, jarvis_pass=True,
        )
        msg = self.mvp4.format_reentry_for_telegram(dec)
        self.assertIn("미트리거", msg)
        self.assertIn("받침", msg)

    # ─────────────────────────────────────────────────────────
    # [pool 스캔]
    # ─────────────────────────────────────────────────────────

    def test_13_pool_scan(self):
        """scan_pool — trigger 우선 정렬."""
        broker = _mock_broker()

        def fake_support(b, t, **kwargs):
            sig = _mock_support_signal(t == "BBBB")  # BBBB만 trigger
            return sig

        with patch.object(self.mvp3, "detect_support_pattern", side_effect=fake_support):
            results = self.mvp4.scan_pool_for_reentry(
                broker,
                [("AAAA", "AAA종목"), ("BBBB", "BBB종목"), ("CCCC", "CCC종목")],
                step5_pool={
                    "AAAA": {"stars": 4, "upside": 3.0},
                    "BBBB": {"stars": 4, "upside": 5.0},
                    "CCCC": {"stars": 4, "upside": 4.0},
                },
                jarvis_safety_check=self._jarvis_pass(),
            )

        self.assertEqual(len(results), 3)
        # trigger=True인 BBBB가 첫 번째
        self.assertEqual(results[0].ticker, "BBBB")
        self.assertTrue(results[0].trigger)

    # ─────────────────────────────────────────────────────────
    # [import]
    # ─────────────────────────────────────────────────────────

    def test_14_imports(self):
        from src.use_cases.adaptive_reentry import (
            evaluate_reentry,
            execute_auto_reentry,
            scan_pool_for_reentry,
            format_reentry_for_telegram,
            ReentryDecision,
            AUTO_REENTRY,
            REENTRY_MAX_AMOUNT,
            STEP5_MIN_STARS,
        )
        self.assertTrue(callable(evaluate_reentry))
        self.assertEqual(STEP5_MIN_STARS, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
