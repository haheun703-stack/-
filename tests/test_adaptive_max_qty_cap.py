"""test_adaptive_max_qty_cap.py — 1주차 안전 1주 cap 검증.

배경 (5/24 퐝가님 지시):
  "1주씩 사는 걸로 진행하자, 페이퍼 X 실전 진행"
  → 차트영웅 max-qty 1과 동일 안전망: ADAPTIVE_SPLIT_MAX_QTY=1, ADAPTIVE_REENTRY_MAX_QTY=1

검증:
  1. MVP-2: SPLIT_MAX_QTY=1 시 3단계 모두 qty=1 (가용 현금 충분해도)
  2. MVP-2: SPLIT_MAX_QTY=0 (무제한) 시 기존 계산 유지
  3. MVP-4: REENTRY_MAX_QTY=1 시 target_qty=1
  4. MVP-4: REENTRY_MAX_QTY=0 시 기존 계산 유지

실행:
  python -m pytest tests/test_adaptive_max_qty_cap.py -v
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMaxQtyCap(unittest.TestCase):

    def setUp(self):
        for mod_name in (
            "src.use_cases.adaptive_buy_queue",
            "src.use_cases.adaptive_reentry",
            "src.use_cases.support_pattern_detector",
        ):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)

        import src.use_cases.adaptive_buy_queue as mvp2
        import src.use_cases.adaptive_reentry as mvp4
        self.mvp2 = mvp2
        self.mvp4 = mvp4

        kill_path = tmp_path / "kill_switch.flag"
        mvp2.KILL_SWITCH_PATH = kill_path
        mvp4.KILL_SWITCH_PATH = kill_path
        mvp2.QUEUE_PATH = tmp_path / "adaptive_buy_queue.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_01_mvp2_qty_cap_1(self):
        """SPLIT_MAX_QTY=1 → 3단계 모두 qty=1 (가용 현금 충분해도)."""
        self.mvp2.SPLIT_MAX_QTY = 1   # 1주 cap

        # 천장 10,000원 + 가용 300만 → 무제한이면 각 단계 90/100주
        result = self.mvp2.register_buy_queue("240810", 10_000, 3_000_000)
        self.assertTrue(result["success"])

        for stage in result["stages"]:
            self.assertEqual(stage["qty"], 1, f"L{stage['level']} qty={stage['qty']} 1 아님")

    def test_02_mvp2_qty_unlimited(self):
        """SPLIT_MAX_QTY=0 (기본) → 기존 가용 현금 기반 계산."""
        self.mvp2.SPLIT_MAX_QTY = 0  # 무제한

        result = self.mvp2.register_buy_queue("240810", 10_000, 3_000_000)
        self.assertTrue(result["success"])

        # L1: 천장 10,000 × 0.90 = 9,000, 가용 30% = 90만 (cap 100만 미만) → 100주
        s1 = result["stages"][0]
        self.assertEqual(s1["qty"], 900_000 // 9_000)  # 100주
        # cap 동작 안 함 → qty > 1
        self.assertGreater(s1["qty"], 1)

    def test_03_mvp4_qty_cap_1(self):
        """REENTRY_MAX_QTY=1 → target_qty=1."""
        from src.use_cases.support_pattern_detector import SupportSignal
        self.mvp4.REENTRY_MAX_QTY = 1

        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": "22000"}}

        sig = SupportSignal(ticker="240810")
        sig.trigger = True
        sig.reasons_pass = ["받침"]

        import src.use_cases.support_pattern_detector as mvp3
        with patch.object(mvp3, "detect_support_pattern", return_value=sig):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool={"240810": {"stars": 4, "upside": 5.0}},
                jarvis_safety_check=lambda t: {"pass": True, "failed": []},
            )

        self.assertTrue(dec.trigger)
        # REENTRY_MAX_AMOUNT=1,000,000 / 22,000 = 45주 가능, 하지만 cap 1주
        self.assertEqual(dec.target_qty, 1)

    def test_04_mvp4_qty_unlimited(self):
        """REENTRY_MAX_QTY=0 (기본) → 가용 금액 기반."""
        from src.use_cases.support_pattern_detector import SupportSignal
        self.mvp4.REENTRY_MAX_QTY = 0  # 무제한

        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": "22000"}}

        sig = SupportSignal(ticker="240810")
        sig.trigger = True

        import src.use_cases.support_pattern_detector as mvp3
        with patch.object(mvp3, "detect_support_pattern", return_value=sig):
            dec = self.mvp4.evaluate_reentry(
                broker, "240810",
                step5_pool={"240810": {"stars": 4, "upside": 5.0}},
                jarvis_safety_check=lambda t: {"pass": True, "failed": []},
            )

        # 1,000,000 / 22,000 = 45주
        self.assertEqual(dec.target_qty, 45)


if __name__ == "__main__":
    unittest.main(verbosity=2)
