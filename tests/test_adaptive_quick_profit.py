"""test_adaptive_quick_profit.py — MVP-2.5 단계별 빠른 익절 단위 테스트.

배경 (퐝가님 5/24 결단):
  적응형 1~2개월 보유 → 너무 느림. L1/L2/L3 각 단계 매수 후 +7% 도달 시
  즉시 단계별 독립 익절 → 3~10일 회전 + 단타 복리.

5/17 자기반성 #1: import + 함수 호출 + 상태 머신 검증.

검증 시나리오:
  [매수 시 quick_profit_target 자동 계산]
   1. MVP-2 매수 체결 → quick_profit_target = actual_price × 1.07

  [+7% 도달 트리거]
   2. 현재가 ≥ target → status QUICK_SOLD + sold_price 기록
   3. 현재가 < target → 변동 없음 (FILLED 유지)
   4. 여러 stage 중 도달한 것만 익절 (L1만 도달 → L2/L3 FILLED 유지)

  [방어]
   5. KILL_SWITCH 발동 시 익절 정지
   6. ENABLED=0 시 익절 비활성
   7. actual_qty=0 시 익절 거부

  [상태 머신]
   8. QUICK_SOLD stage는 다음 cron에서 재트리거 X (중복 방지)
   9. reset_quick_sold_for_reentry → PENDING 복원

  [포맷]
  10. format 메시지 (수익 % + 절대 금액)

  [import]
  11. 모듈 + 상수 노출

실행:
  python -m pytest tests/test_adaptive_quick_profit.py -v
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_broker(current_price: int, sell_success: bool = True, order_id: str = "QSELL001"):
    broker = MagicMock()
    broker.fetch_price.return_value = {"output": {"stck_prpr": str(current_price)}}
    if sell_success:
        order_obj = MagicMock()
        order_obj.order_id = order_id
        broker.sell_limit.return_value = order_obj
    else:
        broker.sell_limit.side_effect = Exception("KIS 매도 실패")
    return broker


class TestAdaptiveQuickProfit(unittest.TestCase):

    def setUp(self):
        for mod_name in (
            "src.use_cases.adaptive_buy_queue",
            "src.use_cases.adaptive_quick_profit",
        ):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)

        import src.use_cases.adaptive_buy_queue as mvp2
        import src.use_cases.adaptive_quick_profit as mvp25
        self.mvp2 = mvp2
        self.mvp25 = mvp25

        kill_path = tmp_path / "kill_switch.flag"
        mvp2.KILL_SWITCH_PATH = kill_path
        mvp25.KILL_SWITCH_PATH = kill_path
        mvp2.QUEUE_PATH = tmp_path / "adaptive_buy_queue.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _setup_filled_stage(self, ticker: str, peak: int, actual_price: int, qty: int = 1):
        """MVP-2 큐 등록 + 첫 stage FILLED 상태로 강제 변경."""
        self.mvp2.register_buy_queue(ticker, peak, 3_000_000)
        raw = self.mvp2._load_queues_raw()
        s = raw["queues"][ticker]["stages"][0]
        s["status"] = self.mvp2.STATUS_FILLED
        s["actual_price"] = actual_price
        s["actual_qty"] = qty
        s["quick_profit_target"] = int(actual_price * 1.07)
        self.mvp2._save_queues_raw(raw)
        return s

    # ─────────────────────────────────────────────────────────
    # [매수 시 quick_profit_target 자동 계산]
    # ─────────────────────────────────────────────────────────

    def test_01_buy_auto_sets_quick_profit_target(self):
        """MVP-2 매수 체결 → quick_profit_target = actual_price × 1.07."""
        self.mvp2.AUTO_BUY = True
        self.mvp2.QUICK_PROFIT_PCT = 7.0

        self.mvp2.register_buy_queue("240810", 25_000, 3_000_000)

        # L1 가격 도달 시뮬 (target=22,500)
        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": "22000"}}
        order_obj = MagicMock()
        order_obj.order_id = "BUY001"
        broker.buy_limit.return_value = order_obj

        triggers = self.mvp2.check_and_trigger_queues(broker)

        # L1 FILLED + quick_profit_target = 22,500 × 1.07 = 24,075
        entry = self.mvp2.get_queue_status("240810")
        s = entry["stages"][0]
        self.assertEqual(s["status"], "FILLED")
        expected_target = int(s["actual_price"] * 1.07)
        self.assertEqual(s["quick_profit_target"], expected_target)

    # ─────────────────────────────────────────────────────────
    # [+7% 도달 트리거]
    # ─────────────────────────────────────────────────────────

    def test_02_quick_profit_trigger_at_7pct(self):
        """현재가 ≥ target → QUICK_SOLD."""
        self._setup_filled_stage("240810", 25_000, actual_price=22_500, qty=1)
        # target = 22,500 × 1.07 = 24,075

        broker = _mock_broker(current_price=24_075)  # 정확히 도달
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 1)
        t = triggers[0]
        self.assertEqual(t["level"], 1)
        self.assertEqual(t["actual_buy_price"], 22_500)
        self.assertEqual(t["sold_price"], 24_075)
        self.assertAlmostEqual(t["profit_pct"], 7.0, places=1)

        # 상태 QUICK_SOLD
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "QUICK_SOLD")

    def test_03_no_trigger_below_target(self):
        """현재가 < target → 변동 없음 (FILLED 유지)."""
        self._setup_filled_stage("240810", 25_000, actual_price=22_500, qty=1)

        broker = _mock_broker(current_price=23_500)  # target 24,075 미달
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 0)
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "FILLED")

    def test_04_only_triggered_stage_sold(self):
        """L1만 도달 → L1만 익절, L2/L3는 FILLED 유지."""
        # 3 stage 모두 FILLED 시뮬 (서로 다른 매수가)
        self.mvp2.register_buy_queue("240810", 25_000, 3_000_000)
        raw = self.mvp2._load_queues_raw()
        prices = [22_500, 20_000, 17_500]
        for i, p in enumerate(prices):
            s = raw["queues"]["240810"]["stages"][i]
            s["status"] = "FILLED"
            s["actual_price"] = p
            s["actual_qty"] = 1
            s["quick_profit_target"] = int(p * 1.07)
        self.mvp2._save_queues_raw(raw)
        # target: L1=24,075 / L2=21,400 / L3=18,725

        # 현재가 21,500 → L2만 도달 (L1 24,075 미달, L3는 이미 도달)
        # 실제: L2(21,400)와 L3(18,725) 모두 도달
        broker = _mock_broker(current_price=21_500)
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        # L2, L3 도달, L1 미도달
        self.assertEqual(len(triggers), 2)
        sold_levels = sorted([t["level"] for t in triggers])
        self.assertEqual(sold_levels, [2, 3])

        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "FILLED")       # L1
        self.assertEqual(entry["stages"][1]["status"], "QUICK_SOLD")   # L2
        self.assertEqual(entry["stages"][2]["status"], "QUICK_SOLD")   # L3

    # ─────────────────────────────────────────────────────────
    # [방어]
    # ─────────────────────────────────────────────────────────

    def test_05_kill_switch_blocks(self):
        """KILL_SWITCH 발동 시 익절 정지."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        self.mvp25.KILL_SWITCH_PATH.write_text("active", encoding="utf-8")

        broker = _mock_broker(current_price=30_000)  # 충분히 도달
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 0)
        broker.fetch_price.assert_not_called()

    def test_06_disabled_no_trigger(self):
        """ENABLED=0 시 익절 비활성."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        self.mvp25.QUICK_PROFIT_ENABLED = False

        broker = _mock_broker(current_price=30_000)
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 0)

    def test_07_zero_qty_refused(self):
        """actual_qty=0 시 익절 거부."""
        s = self._setup_filled_stage("240810", 25_000, 22_500, 1)
        # qty 강제로 0
        raw = self.mvp2._load_queues_raw()
        raw["queues"]["240810"]["stages"][0]["actual_qty"] = 0
        self.mvp2._save_queues_raw(raw)

        broker = _mock_broker(current_price=30_000)
        result = self.mvp25.execute_quick_sell(broker, "240810", raw["queues"]["240810"]["stages"][0])
        self.assertFalse(result["success"])
        self.assertIn("actual_qty", result["error"])

    # ─────────────────────────────────────────────────────────
    # [상태 머신]
    # ─────────────────────────────────────────────────────────

    def test_08_quick_sold_no_double_trigger(self):
        """QUICK_SOLD stage는 다음 cron에서 재트리거 X."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)

        # 첫 cron — 익절
        broker = _mock_broker(current_price=24_075)
        triggers1 = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers1), 1)

        # 다음 cron — 같은 가격 다시 와도 트리거 X
        triggers2 = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers2), 0)

    def test_09_reset_for_reentry(self):
        """reset_quick_sold_for_reentry → PENDING 복원 + 재진입 가능."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)

        broker = _mock_broker(current_price=24_075)
        self.mvp25.check_quick_profit_triggers(broker)
        # QUICK_SOLD 상태
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "QUICK_SOLD")

        # reset
        self.assertTrue(self.mvp25.reset_quick_sold_for_reentry("240810"))
        entry = self.mvp2.get_queue_status("240810")
        s = entry["stages"][0]
        self.assertEqual(s["status"], "PENDING")
        self.assertEqual(s["actual_price"], 0)
        self.assertEqual(s["quick_profit_target"], 0)

    # ─────────────────────────────────────────────────────────
    # [포맷]
    # ─────────────────────────────────────────────────────────

    def test_10_format_telegram(self):
        """포맷 메시지 — 수익 % + 절대 금액."""
        trigger = {
            "ticker": "240810",
            "name": "원익IPS",
            "level": 1,
            "actual_buy_price": 22_500,
            "sold_price": 24_075,
            "profit_pct": 7.0,
            "qty": 1,
            "order_id": "QSELL999",
            "current_price": 24_080,
        }
        msg = self.mvp25.format_quick_profit_for_telegram(trigger)
        self.assertIn("원익IPS", msg)
        self.assertIn("L1", msg)
        self.assertIn("22,500", msg)
        self.assertIn("24,075", msg)
        self.assertIn("+7.00%", msg)
        self.assertIn("1,575", msg)  # 수익 (24,075 - 22,500) × 1
        self.assertIn("QSELL999", msg)

    # ─────────────────────────────────────────────────────────
    # [import]
    # ─────────────────────────────────────────────────────────

    def test_11_imports(self):
        from src.use_cases.adaptive_quick_profit import (
            check_quick_profit_triggers,
            execute_quick_sell,
            format_quick_profit_for_telegram,
            reset_quick_sold_for_reentry,
            QUICK_PROFIT_ENABLED,
            QUICK_PROFIT_PCT,
            QUICK_PROFIT_RATIO,
        )
        self.assertTrue(callable(check_quick_profit_triggers))
        self.assertEqual(QUICK_PROFIT_PCT, 7.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
