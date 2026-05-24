"""test_adaptive_buy_queue.py — MVP-2 분할매수 큐 단위 테스트.

배경 (퐝가님 5/23 흐름 + 5/24 작업):
  6단계 흐름의 [4단계 분할매수] 검증.
  -10% / -20% / -30% 3단계 큐 등록 + 도달 시 자동 매수 / 알림.

5/17 자기반성 #1 적용: import + 함수 호출 + 상태 머신 흐름 모두 검증.

검증 시나리오:
  [큐 등록]
   1. 신규 등록 + 3단계 가격/수량 계산 검증
   2. 평단가 효과 ≈ 천장 × 79% 수학적 검증
   3. 가용 현금 부족 (10만 미만) → 거부
   4. peak_price=0 → 거부
   5. 3종목 한도 초과 → 거부 (4번째 거부)
   6. 동일 종목 재등록 (천장 갱신) → 덮어쓰기 허용

  [가격 도달 트리거]
   7. 현재가 ≤ L1 target (-10%) → L1만 TRIGGERED
   8. 현재가 ≤ L3 target (-30%) → L1/L2/L3 모두 TRIGGERED (한 번에)
   9. 현재가 > L1 target → 모두 PENDING 유지
  10. ADAPTIVE_AUTO_BUY=0 (기본) → TRIGGERED 상태 (알림만)
  11. ADAPTIVE_AUTO_BUY=1 + 매수 성공 → FILLED + order_id
  12. ADAPTIVE_AUTO_BUY=1 + 매수 실패 → FAILED + error

  [방어]
  13. KILL_SWITCH 발동 시 등록 거부
  14. KILL_SWITCH 발동 시 트리거 정지 (빈 리스트)
  15. 만료 (QUEUE_EXPIRY_DAYS 초과) → EXPIRED

  [텔레그램 포맷]
  16. TRIGGERED 포맷 (천장/현재가/지정가/배정)
  17. FILLED 포맷 (주문 ID 포함)
  18. EXPIRED 포맷

  [유틸]
  19. get_queue_status / clear_queue 동작

실행:
  python -m pytest tests/test_adaptive_buy_queue.py -v
"""

import sys
import os
import json
import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_broker(current_price: int, buy_success: bool = True, order_id: str = "ORD123"):
    """가짜 broker — fetch_price + buy_limit."""
    broker = MagicMock()
    broker.fetch_price.return_value = {
        "output": {"stck_prpr": str(current_price)},
    }
    if buy_success:
        order_obj = MagicMock()
        order_obj.order_id = order_id
        broker.buy_limit.return_value = order_obj
    else:
        broker.buy_limit.side_effect = Exception("KIS 매수 실패")
    return broker


class TestAdaptiveBuyQueue(unittest.TestCase):
    """MVP-2 분할매수 큐 검증."""

    def setUp(self):
        """각 테스트 격리 — tmp dir에 큐 파일."""
        # 모듈 reload하여 임계값 재읽기
        import importlib
        if "src.use_cases.adaptive_buy_queue" in sys.modules:
            del sys.modules["src.use_cases.adaptive_buy_queue"]

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # 모듈의 QUEUE_PATH + KILL_SWITCH_PATH를 tmp로 패치
        import src.use_cases.adaptive_buy_queue as mod
        self.mod = mod
        mod.QUEUE_PATH = self.tmp_path / "adaptive_buy_queue.json"
        mod.KILL_SWITCH_PATH = self.tmp_path / "kill_switch.flag"

    def tearDown(self):
        self.tmpdir.cleanup()

    # ─────────────────────────────────────────────────────────
    # [큐 등록] 시나리오 1~6
    # ─────────────────────────────────────────────────────────

    def test_01_register_basic_3_stages(self):
        """신규 등록 + 3단계 가격/수량 계산 검증."""
        result = self.mod.register_buy_queue(
            ticker="240810", peak_price=25_000, available_cash=3_000_000, name="원익IPS"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["ticker"], "240810")
        self.assertEqual(len(result["stages"]), 3)

        # L1: 천장 × 0.90 = 22,500, 가용 30% = 900,000
        s1 = result["stages"][0]
        self.assertEqual(s1["level"], 1)
        self.assertEqual(s1["target_price"], 22_500)
        self.assertEqual(s1["alloc_amount"], 900_000)
        self.assertEqual(s1["qty"], 900_000 // 22_500)  # 40
        self.assertEqual(s1["status"], "PENDING")

        # L3: 천장 × 0.70 = 17,500, 가용 40% = 1,000,000 (cap)
        s3 = result["stages"][2]
        self.assertEqual(s3["target_price"], 17_500)
        # 40% = 1,200,000 > MAX_AMOUNT 1,000,000 → cap
        self.assertEqual(s3["alloc_amount"], 1_000_000)

    def test_02_avg_price_effect_math(self):
        """평단가 효과 ≈ 천장 × 79% 수학적 검증."""
        peak = 100_000
        cash = 10_000_000   # 충분
        # MAX_AMOUNT cap을 없애기 위해 큰 cash로
        result = self.mod.register_buy_queue("TEST01", peak, cash)
        self.assertTrue(result["success"])

        # 평단가 효과 검증 (실제 매수 수량 기반)
        total_cost = sum(s["target_price"] * s["qty"] for s in result["stages"])
        total_qty = sum(s["qty"] for s in result["stages"])
        avg_price = total_cost / total_qty if total_qty else 0
        # 천장 79% ± 2% 허용 (qty 정수 반올림 오차)
        self.assertAlmostEqual(avg_price / peak, 0.79, delta=0.02)

    def test_03_reject_low_cash(self):
        """가용 현금 부족 (10만 미만) → 거부."""
        result = self.mod.register_buy_queue("240810", 25_000, 50_000)
        self.assertFalse(result["success"])
        self.assertIn("가용 현금 부족", result["error"])

    def test_04_reject_zero_peak(self):
        """peak_price=0 → 거부."""
        result = self.mod.register_buy_queue("240810", 0, 3_000_000)
        self.assertFalse(result["success"])
        self.assertIn("peak_price 부적합", result["error"])

    def test_05_reject_4th_position(self):
        """3종목 한도 초과 → 4번째 거부."""
        for i, ticker in enumerate(["AAAA", "BBBB", "CCCC"], 1):
            r = self.mod.register_buy_queue(ticker, 10_000, 1_000_000)
            self.assertTrue(r["success"], f"{ticker} 등록 실패: {r}")

        # 4번째
        r4 = self.mod.register_buy_queue("DDDD", 10_000, 1_000_000)
        self.assertFalse(r4["success"])
        self.assertIn("3종목 한도", r4["error"])

    def test_06_reregister_same_ticker(self):
        """동일 종목 재등록 (천장 갱신) → 덮어쓰기."""
        r1 = self.mod.register_buy_queue("240810", 25_000, 3_000_000, name="원익IPS")
        self.assertTrue(r1["success"])
        self.assertFalse(r1.get("is_update", False))

        # 천장 갱신 재등록
        r2 = self.mod.register_buy_queue("240810", 30_000, 3_000_000, name="원익IPS")
        self.assertTrue(r2["success"])
        self.assertTrue(r2.get("is_update"))
        self.assertEqual(r2["peak_price"], 30_000)

        # 활성 종목 개수 1 유지
        queues = self.mod.load_queues()
        self.assertEqual(len(queues), 1)

    # ─────────────────────────────────────────────────────────
    # [가격 도달 트리거] 시나리오 7~12
    # ─────────────────────────────────────────────────────────

    def test_07_trigger_L1_only(self):
        """현재가 ≤ L1 target (-10%) → L1만 TRIGGERED."""
        self.mod.register_buy_queue("240810", 25_000, 3_000_000, name="원익IPS")

        # 현재가 22,000 (L1=22,500 도달, L2=20,000 미달)
        broker = _mock_broker(22_000)
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["level"], 1)
        self.assertIn(triggers[0]["status"], ("TRIGGERED", "FILLED"))

        # L2, L3는 여전히 PENDING
        entry = self.mod.get_queue_status("240810")
        self.assertEqual(entry["stages"][1]["status"], "PENDING")
        self.assertEqual(entry["stages"][2]["status"], "PENDING")

    def test_08_trigger_all_3_stages(self):
        """현재가 ≤ L3 target (-30%) → L1/L2/L3 모두 한 번에."""
        self.mod.register_buy_queue("240810", 25_000, 3_000_000)

        # 현재가 17,000 → L1(22,500)/L2(20,000)/L3(17,500) 모두 도달
        broker = _mock_broker(17_000)
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 3)
        levels = sorted([t["level"] for t in triggers])
        self.assertEqual(levels, [1, 2, 3])

    def test_09_no_trigger_above_L1(self):
        """현재가 > L1 target → 모두 PENDING 유지."""
        self.mod.register_buy_queue("240810", 25_000, 3_000_000)

        # 현재가 23,000 > L1(22,500)
        broker = _mock_broker(23_000)
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 0)
        entry = self.mod.get_queue_status("240810")
        for stage in entry["stages"]:
            self.assertEqual(stage["status"], "PENDING")

    def test_10_auto_buy_off_triggered_only(self):
        """ADAPTIVE_AUTO_BUY=0 (기본) → TRIGGERED 상태 (알림만)."""
        # 기본값 AUTO_BUY=False
        self.assertFalse(self.mod.AUTO_BUY)

        self.mod.register_buy_queue("240810", 25_000, 3_000_000)
        broker = _mock_broker(22_000)
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["status"], "TRIGGERED")
        # buy_limit 호출되지 않음
        broker.buy_limit.assert_not_called()

    def test_11_auto_buy_on_success_filled(self):
        """ADAPTIVE_AUTO_BUY=1 + 매수 성공 → FILLED + order_id."""
        self.mod.AUTO_BUY = True  # 강제 활성화

        self.mod.register_buy_queue("240810", 25_000, 3_000_000)
        broker = _mock_broker(22_000, buy_success=True, order_id="KISORD001")
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["status"], "FILLED")
        self.assertEqual(triggers[0]["order_id"], "KISORD001")
        broker.buy_limit.assert_called_once()

    def test_12_auto_buy_on_failure_failed(self):
        """ADAPTIVE_AUTO_BUY=1 + 매수 실패 → FAILED + error."""
        self.mod.AUTO_BUY = True

        self.mod.register_buy_queue("240810", 25_000, 3_000_000)
        broker = _mock_broker(22_000, buy_success=False)
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["status"], "FAILED")
        self.assertIn("KIS 매수 실패", triggers[0]["error"])

    # ─────────────────────────────────────────────────────────
    # [방어] 시나리오 13~15
    # ─────────────────────────────────────────────────────────

    def test_13_kill_switch_blocks_register(self):
        """KILL_SWITCH 발동 시 등록 거부."""
        self.mod.KILL_SWITCH_PATH.write_text("active", encoding="utf-8")
        result = self.mod.register_buy_queue("240810", 25_000, 3_000_000)
        self.assertFalse(result["success"])
        self.assertIn("KILL_SWITCH", result["error"])

    def test_14_kill_switch_blocks_trigger(self):
        """KILL_SWITCH 발동 시 트리거 정지 (빈 리스트)."""
        self.mod.register_buy_queue("240810", 25_000, 3_000_000)
        # 등록 후 KILL_SWITCH 발동
        self.mod.KILL_SWITCH_PATH.write_text("active", encoding="utf-8")

        broker = _mock_broker(17_000)  # L3 도달 가격
        triggers = self.mod.check_and_trigger_queues(broker)

        self.assertEqual(len(triggers), 0)
        # broker도 호출되지 않음
        broker.fetch_price.assert_not_called()

    def test_15_expiry_after_60_days(self):
        """만료 (QUEUE_EXPIRY_DAYS 초과) → EXPIRED."""
        self.mod.register_buy_queue("240810", 25_000, 3_000_000, name="원익IPS")

        # registered_at 강제로 61일 전으로 변경
        raw = self.mod._load_queues_raw()
        old_date = (datetime.now() - timedelta(days=61)).isoformat(timespec="seconds")
        raw["queues"]["240810"]["registered_at"] = old_date
        self.mod._save_queues_raw(raw)

        broker = _mock_broker(22_000)
        triggers = self.mod.check_and_trigger_queues(broker)

        # EXPIRED 이벤트 1개
        expired_events = [t for t in triggers if t.get("event") == "EXPIRED"]
        self.assertEqual(len(expired_events), 1)
        # 상태도 EXPIRED로 변경
        entry = self.mod.get_queue_status("240810")
        for stage in entry["stages"]:
            self.assertEqual(stage["status"], "EXPIRED")

    # ─────────────────────────────────────────────────────────
    # [텔레그램 포맷] 시나리오 16~18
    # ─────────────────────────────────────────────────────────

    def test_16_format_triggered(self):
        """TRIGGERED 포맷 (천장/현재가/지정가/배정 표기)."""
        trigger = {
            "ticker": "240810",
            "name": "원익IPS",
            "level": 1,
            "status": "TRIGGERED",
            "target_price": 22_500,
            "current_price": 22_000,
            "peak_price": 25_000,
            "qty": 40,
            "alloc_amount": 900_000,
        }
        msg = self.mod.format_trigger_for_telegram(trigger)
        self.assertIn("원익IPS", msg)
        self.assertIn("L1", msg)
        self.assertIn("25,000", msg)        # 천장
        self.assertIn("22,000", msg)        # 현재가
        self.assertIn("22,500", msg)        # 지정가
        self.assertIn("알림만", msg)

    def test_17_format_filled(self):
        """FILLED 포맷 (주문 ID 포함)."""
        trigger = {
            "ticker": "240810",
            "name": "원익IPS",
            "level": 2,
            "status": "FILLED",
            "target_price": 20_000,
            "current_price": 19_800,
            "peak_price": 25_000,
            "qty": 45,
            "alloc_amount": 900_000,
            "order_id": "KISORD002",
        }
        msg = self.mod.format_trigger_for_telegram(trigger)
        self.assertIn("체결", msg)
        self.assertIn("L2", msg)
        self.assertIn("KISORD002", msg)

    def test_18_format_expired(self):
        """EXPIRED 포맷."""
        trigger = {
            "ticker": "240810",
            "name": "원익IPS",
            "event": "EXPIRED",
            "registered_at": "2026-03-15T18:00:00",
        }
        msg = self.mod.format_trigger_for_telegram(trigger)
        self.assertIn("만료", msg)
        self.assertIn("원익IPS", msg)
        self.assertIn("2026-03-15", msg)

    # ─────────────────────────────────────────────────────────
    # [유틸] 시나리오 19
    # ─────────────────────────────────────────────────────────

    def test_19_get_and_clear(self):
        """get_queue_status / clear_queue 동작."""
        self.mod.register_buy_queue("240810", 25_000, 3_000_000)

        entry = self.mod.get_queue_status("240810")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["peak_price"], 25_000)

        # 없는 종목
        self.assertIsNone(self.mod.get_queue_status("999999"))

        # clear
        self.assertTrue(self.mod.clear_queue("240810"))
        self.assertIsNone(self.mod.get_queue_status("240810"))
        # 두 번째 clear는 False
        self.assertFalse(self.mod.clear_queue("240810"))

    # ─────────────────────────────────────────────────────────
    # [import 체인] 5/17 자기반성 #1
    # ─────────────────────────────────────────────────────────

    def test_20_imports(self):
        """import + 핵심 심볼 노출 검증."""
        from src.use_cases.adaptive_buy_queue import (
            register_buy_queue,
            check_and_trigger_queues,
            execute_auto_buy,
            get_queue_status,
            clear_queue,
            format_trigger_for_telegram,
            load_queues,
            QueueStage,
            STATUS_PENDING,
            STATUS_TRIGGERED,
            STATUS_FILLED,
            STATUS_EXPIRED,
            STATUS_FAILED,
        )
        self.assertTrue(callable(register_buy_queue))
        self.assertTrue(callable(check_and_trigger_queues))
        self.assertEqual(STATUS_PENDING, "PENDING")


if __name__ == "__main__":
    unittest.main(verbosity=2)
