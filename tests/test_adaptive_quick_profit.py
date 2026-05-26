"""test_adaptive_quick_profit.py — MVP-2.5 v2 Trailing Quick Profit 단위 테스트.

배경 (퐝가님 5/24 v2 보강):
  "15%까지 가도 +7%만 먹고 나오는 건 멍청한 룰"
  → Trailing 보강: +7% 도달 시 ARMED → 고점 추적 → -2% 꺾임 시 매도

5/17 자기반성 #1: import + 함수 호출 + 상태 머신 흐름 검증.

검증 시나리오 (v2):
  [FILLED → ARMED 전환]
   1. +7% 도달 → ARMED 상태 + trailing_peak = 현재가 (매도 X)
   2. +7% 미달 → FILLED 유지 (변동 없음)
   3. broker.sell_market 호출 안 됨 (ARMED 시 매도 X)

  [ARMED → 고점 추적]
   4. ARMED + 현재가 > trailing_peak → peak 갱신 (매도 X)
   5. ARMED + 현재가 == trailing_peak → 매도 X
   6. ARMED + 현재가 ≤ peak × 0.98 → SOLD (시장가 매도)

  [실제 시나리오]
   7. 매수 22,500 → 24,075(+7%) ARMED → 25,800 peak → 25,284(-2%) SOLD
       → 최종 +12.4% (단순 +7% 대비 +5.4%p 추가)
   8. 매수 22,500 → 24,075(+7%) ARMED → 23,593(-2%) 즉시 SOLD (반락)
       → +4.86% (안전망: 빠르게 꺾이면 손실 적게)

  [방어]
   9. KILL_SWITCH 발동 시 정지
  10. ENABLED=0 시 비활성
  11. 여러 단계 독립 (L1 ARMED, L2 FILLED 동시)

  [상태 머신]
  12. QUICK_SOLD 후 중복 트리거 X
  13. reset_quick_sold_for_reentry → PENDING 복원

  [포맷]
  14. ARMED 이벤트 포맷
  15. SOLD 이벤트 포맷 (peak/sold/gap 표기)

  [import]
  16. v2 모듈 + 신규 상수 노출

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
        broker.sell_market.return_value = order_obj
    else:
        broker.sell_market.side_effect = Exception("KIS 매도 실패")
    return broker


class TestTrailingQuickProfit(unittest.TestCase):

    def setUp(self):
        # 5/26: 기존 sell_market 호출 시나리오 유지 (지정가는 별도 테스트)
        os.environ["ADAPTIVE_SELL_USE_LIMIT"] = "0"
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
        """MVP-2 큐 등록 + 첫 stage FILLED 상태로 강제."""
        self.mvp2.register_buy_queue(ticker, peak, 3_000_000)
        raw = self.mvp2._load_queues_raw()
        s = raw["queues"][ticker]["stages"][0]
        s["status"] = self.mvp2.STATUS_FILLED
        s["actual_price"] = actual_price
        s["actual_qty"] = qty
        s["quick_profit_target"] = int(actual_price * 1.07)
        self.mvp2._save_queues_raw(raw)
        return s

    def _setup_armed_stage(self, ticker: str, peak: int, actual_price: int,
                            trailing_peak: int, qty: int = 1):
        """ARMED 상태로 강제 설정."""
        self._setup_filled_stage(ticker, peak, actual_price, qty)
        raw = self.mvp2._load_queues_raw()
        s = raw["queues"][ticker]["stages"][0]
        s["status"] = self.mvp2.STATUS_QUICK_ARMED
        s["trailing_peak"] = trailing_peak
        self.mvp2._save_queues_raw(raw)
        return s

    # ─────────────────────────────────────────────────────────
    # [FILLED → ARMED 전환]
    # ─────────────────────────────────────────────────────────

    def test_01_filled_to_armed_at_7pct(self):
        """+7% 도달 → ARMED (매도 X)."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        broker = _mock_broker(current_price=24_075)  # 정확히 +7%

        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["event"], "ARMED")
        self.assertEqual(triggers[0]["trailing_peak"], 24_075)
        # 매도 X
        broker.sell_market.assert_not_called()

        entry = self.mvp2.get_queue_status("240810")
        s = entry["stages"][0]
        self.assertEqual(s["status"], "QUICK_ARMED")
        self.assertEqual(s["trailing_peak"], 24_075)
        self.assertIsNotNone(s.get("trailing_armed_at"))

    def test_02_below_7pct_no_change(self):
        """+7% 미달 → FILLED 유지."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        broker = _mock_broker(current_price=23_500)  # target 24,075 미달

        triggers = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers), 0)
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "FILLED")

    def test_03_armed_no_sell_call(self):
        """ARMED 시 broker.sell_market 호출 X (매도 안 함)."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        broker = _mock_broker(current_price=25_000)  # +11% (충분히 ARMED)

        self.mvp25.check_quick_profit_triggers(broker)
        broker.sell_market.assert_not_called()

    # ─────────────────────────────────────────────────────────
    # [ARMED → 고점 추적]
    # ─────────────────────────────────────────────────────────

    def test_04_armed_peak_update(self):
        """현재가 > trailing_peak → peak 갱신."""
        self._setup_armed_stage("240810", 25_000, 22_500, trailing_peak=24_075, qty=1)
        broker = _mock_broker(current_price=24_500)  # 더 오름

        triggers = self.mvp25.check_quick_profit_triggers(broker)
        # 갱신만 (이벤트 발송 X, 너무 시끄러움)
        self.assertEqual(len(triggers), 0)

        entry = self.mvp2.get_queue_status("240810")
        s = entry["stages"][0]
        self.assertEqual(s["status"], "QUICK_ARMED")
        self.assertEqual(s["trailing_peak"], 24_500)  # 갱신됨

    def test_05_armed_same_price_no_change(self):
        """현재가 == trailing_peak → 변동 없음."""
        self._setup_armed_stage("240810", 25_000, 22_500, trailing_peak=25_000, qty=1)
        broker = _mock_broker(current_price=25_000)  # 동일

        triggers = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers), 0)
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["trailing_peak"], 25_000)

    def test_06_armed_sells_on_2pct_drop(self):
        """현재가 ≤ peak × 0.98 → SOLD."""
        self._setup_armed_stage("240810", 25_000, 22_500, trailing_peak=25_000, qty=1)
        # 25,000 × 0.98 = 24,500 → 24,500 도달 시 매도
        broker = _mock_broker(current_price=24_500, order_id="TRAIL001")

        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 1)
        t = triggers[0]
        self.assertEqual(t["event"], "SOLD")
        self.assertEqual(t["trailing_peak"], 25_000)
        self.assertEqual(t["sold_price"], 24_500)
        # 수익 = (24,500 - 22,500) / 22,500 ≈ +8.89%
        self.assertAlmostEqual(t["profit_pct"], 8.89, places=1)
        # 고점 대비 = (25,000 - 22,500) / 22,500 = +11.11%
        self.assertAlmostEqual(t["peak_pct"], 11.11, places=1)

        broker.sell_market.assert_called_once()
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "QUICK_SOLD")

    # ─────────────────────────────────────────────────────────
    # [실제 시나리오: 다단계 cron 시뮬]
    # ─────────────────────────────────────────────────────────

    def test_07_full_cycle_15pct_then_drop(self):
        """매수 22,500 → 25,800 천장 → 25,284 꺾임 = +12.4%."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)

        # cron 1: 24,075 (+7%) → ARMED
        broker = _mock_broker(current_price=24_075)
        self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(self.mvp2.get_queue_status("240810")["stages"][0]["status"], "QUICK_ARMED")

        # cron 2: 25,000 → peak 갱신
        broker = _mock_broker(current_price=25_000)
        self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(self.mvp2.get_queue_status("240810")["stages"][0]["trailing_peak"], 25_000)

        # cron 3: 25,800 → peak 갱신 (천장)
        broker = _mock_broker(current_price=25_800)
        self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(self.mvp2.get_queue_status("240810")["stages"][0]["trailing_peak"], 25_800)

        # cron 4: 25,284 (peak × 0.98 = 25,284) → 매도
        broker = _mock_broker(current_price=25_284, order_id="FINAL001")
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["event"], "SOLD")
        # 수익 = (25,284 - 22,500) / 22,500 = +12.37%
        self.assertAlmostEqual(triggers[0]["profit_pct"], 12.37, places=1)
        # 고점 = (25,800 - 22,500) / 22,500 = +14.67%
        self.assertAlmostEqual(triggers[0]["peak_pct"], 14.67, places=1)

    def test_08_armed_quick_reversal_safety(self):
        """매수 22,500 → 24,075 ARMED 직후 23,593 즉시 반락 = +4.86% (안전망)."""
        self._setup_filled_stage("240810", 25_000, 22_500, 1)

        # cron 1: 24,075 ARMED
        broker = _mock_broker(current_price=24_075)
        self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(self.mvp2.get_queue_status("240810")["stages"][0]["trailing_peak"], 24_075)

        # cron 2: 23,593 (24,075 × 0.98) → 즉시 매도
        broker = _mock_broker(current_price=23_593)
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["event"], "SOLD")
        # 수익 = (23,593 - 22,500) / 22,500 ≈ +4.86%
        self.assertAlmostEqual(triggers[0]["profit_pct"], 4.86, places=1)

    # ─────────────────────────────────────────────────────────
    # [방어]
    # ─────────────────────────────────────────────────────────

    def test_09_kill_switch_does_NOT_block_sell(self):
        """P0-4 정책 (5/25 commit 7af8a17): KILL_SWITCH 활성에도 매도는 계속.

        이유: KILL_SWITCH는 자비스 매수 차단용. 매도(보호)는 항상 진행해야 시드 보호.
        매도가 차단되면 -5% 손절, +7% trailing 등 보호 안전망 모두 무력화.
        """
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        self.mvp25.KILL_SWITCH_PATH.write_text("active", encoding="utf-8")

        broker = _mock_broker(current_price=30_000)
        triggers = self.mvp25.check_quick_profit_triggers(broker)
        # KILL_SWITCH 무관 — 매도 트리거 정상 발화 (보유 보호)
        self.assertEqual(len(triggers), 1)
        broker.fetch_price.assert_called()

    def test_10_disabled_no_trigger(self):
        self._setup_filled_stage("240810", 25_000, 22_500, 1)
        self.mvp25.QUICK_PROFIT_ENABLED = False

        broker = _mock_broker(current_price=30_000)
        triggers = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers), 0)

    def test_11_multi_stage_independent(self):
        """L1 ARMED + L2 FILLED 동시에 → 각자 독립 처리."""
        self.mvp2.register_buy_queue("240810", 25_000, 3_000_000)
        raw = self.mvp2._load_queues_raw()
        # L1: 22,500 매수, ARMED 상태 (trailing_peak 24,500)
        s1 = raw["queues"]["240810"]["stages"][0]
        s1["status"] = "QUICK_ARMED"
        s1["actual_price"] = 22_500
        s1["actual_qty"] = 1
        s1["quick_profit_target"] = 24_075
        s1["trailing_peak"] = 24_500
        # L2: 20,000 매수, FILLED 상태 (target 21,400)
        s2 = raw["queues"]["240810"]["stages"][1]
        s2["status"] = "FILLED"
        s2["actual_price"] = 20_000
        s2["actual_qty"] = 1
        s2["quick_profit_target"] = 21_400
        self.mvp2._save_queues_raw(raw)

        # 현재가 21,800 → L2: +7% 도달 (ARMED 전환), L1: peak 24,500 × 0.98 = 24,010 → 21,800 < 24,010 → SOLD
        broker = _mock_broker(current_price=21_800)
        triggers = self.mvp25.check_quick_profit_triggers(broker)

        # L1 SOLD + L2 ARMED = 2 트리거
        self.assertEqual(len(triggers), 2)
        events = sorted([(t["level"], t["event"]) for t in triggers])
        self.assertEqual(events, [(1, "SOLD"), (2, "ARMED")])

    # ─────────────────────────────────────────────────────────
    # [상태 머신]
    # ─────────────────────────────────────────────────────────

    def test_12_quick_sold_no_double_trigger(self):
        """QUICK_SOLD 후 중복 트리거 X."""
        import os
        os.environ["ADAPTIVE_SELL_USE_LIMIT"] = "0"
        self._setup_armed_stage("240810", 25_000, 22_500, 25_000, 1)
        broker = _mock_broker(current_price=24_500)

        # 첫 번째 → SOLD
        triggers1 = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers1), 1)

        # 두 번째 → 변동 없음
        triggers2 = self.mvp25.check_quick_profit_triggers(broker)
        self.assertEqual(len(triggers2), 0)

    def test_13_reset_for_reentry(self):
        """reset → PENDING + trailing 필드 초기화."""
        import os
        os.environ["ADAPTIVE_SELL_USE_LIMIT"] = "0"
        self._setup_armed_stage("240810", 25_000, 22_500, 25_000, 1)
        broker = _mock_broker(current_price=24_500)
        self.mvp25.check_quick_profit_triggers(broker)

        self.assertTrue(self.mvp25.reset_quick_sold_for_reentry("240810"))
        entry = self.mvp2.get_queue_status("240810")
        s = entry["stages"][0]
        self.assertEqual(s["status"], "PENDING")
        self.assertEqual(s["trailing_peak"], 0)
        self.assertIsNone(s["trailing_armed_at"])

    # ─────────────────────────────────────────────────────────
    # [포맷]
    # ─────────────────────────────────────────────────────────

    def test_14_format_armed(self):
        trigger = {
            "ticker": "240810", "name": "원익IPS", "level": 1,
            "event": "ARMED",
            "actual_buy_price": 22_500, "current_price": 24_075,
            "profit_pct_so_far": 7.0, "trailing_peak": 24_075,
        }
        msg = self.mvp25.format_quick_profit_for_telegram(trigger)
        self.assertIn("원익IPS", msg)
        self.assertIn("추적 시작", msg)
        self.assertIn("Trailing", msg)
        self.assertIn("24,075", msg)

    def test_15_format_sold(self):
        trigger = {
            "ticker": "240810", "name": "원익IPS", "level": 1,
            "event": "SOLD",
            "actual_buy_price": 22_500, "trailing_peak": 25_800,
            "sold_price": 25_284, "profit_pct": 12.37, "peak_pct": 14.67,
            "qty": 1, "order_id": "TRAIL999",
        }
        msg = self.mvp25.format_quick_profit_for_telegram(trigger)
        self.assertIn("원익IPS", msg)
        self.assertIn("체결", msg)
        self.assertIn("22,500", msg)
        self.assertIn("25,800", msg)
        self.assertIn("25,284", msg)
        self.assertIn("+12.37%", msg)
        self.assertIn("+14.67%", msg)
        self.assertIn("TRAIL999", msg)

    # ─────────────────────────────────────────────────────────
    # [import]
    # ─────────────────────────────────────────────────────────

    def test_16_imports(self):
        from src.use_cases.adaptive_quick_profit import (
            check_quick_profit_triggers,
            execute_trailing_sell,
            execute_quick_sell,  # v1 호환
            format_quick_profit_for_telegram,
            reset_quick_sold_for_reentry,
            QUICK_PROFIT_ENABLED,
            QUICK_PROFIT_PCT,
            TRAILING_DROP_PCT,
        )
        from src.use_cases.adaptive_buy_queue import STATUS_QUICK_ARMED
        self.assertTrue(callable(check_quick_profit_triggers))
        self.assertEqual(QUICK_PROFIT_PCT, 7.0)
        self.assertEqual(TRAILING_DROP_PCT, 2.0)
        self.assertEqual(STATUS_QUICK_ARMED, "QUICK_ARMED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
