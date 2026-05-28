"""test_adaptive_mvp1_mvp2_integration.py — MVP-1 매도 → MVP-2 큐 자동 등록 흐름.

배경 (5/17 자기반성 #1):
  "import OK = 동작 OK" 오판 금지. 실제 함수 호출 + main 흐름까지 검증.

검증 흐름:
  1. ADAPTIVE_AUTO_SELL=1 (강제 활성화)
  2. detect_peak_signal → trigger=True
  3. execute_auto_sell 호출:
     a. 시장가 매도 성공
     b. broker.get_available_cash() 호출
     c. register_buy_queue 자동 호출
     d. 큐 파일에 3단계 PENDING 저장
  4. 결과 dict에 queue_registered=True
  5. data/adaptive_buy_queue.json에 ticker 등록 확인

실행:
  python -m pytest tests/test_adaptive_mvp1_mvp2_integration.py -v
"""

import sys
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ohlcv_rows(peak_price: int, peak_days_ago: int, n_rows: int = 30) -> list[dict]:
    rows = []
    today = date.today()
    for i in range(n_rows):
        d = today - timedelta(days=i)
        if i == peak_days_ago:
            high = peak_price
        elif i < peak_days_ago:
            high = int(peak_price * (1 - 0.01 * (peak_days_ago - i)))
        else:
            high = int(peak_price * (1 - 0.015 * (i - peak_days_ago)))
        rows.append({
            "stck_bsop_date": d.strftime("%Y%m%d"),
            "stck_oprc": str(high - 200),
            "stck_hgpr": str(high),
            "stck_lwpr": str(high - 500),
            "stck_clpr": str(high - 100),
            "acml_vol": "100000",
        })
    return rows


class TestMVP1MVP2Integration(unittest.TestCase):

    def setUp(self):
        # 모듈 reload하여 임계값 재읽기
        for mod_name in (
            "src.use_cases.adaptive_position_manager",
            "src.use_cases.adaptive_buy_queue",
        ):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)

        import src.use_cases.adaptive_position_manager as mvp1
        import src.use_cases.adaptive_buy_queue as mvp2
        self.mvp1 = mvp1
        self.mvp2 = mvp2

        # tmp 경로로 격리
        mvp1.KILL_SWITCH_PATH = tmp_path / "kill_switch.flag"
        mvp2.KILL_SWITCH_PATH = tmp_path / "kill_switch.flag"
        mvp2.QUEUE_PATH = tmp_path / "adaptive_buy_queue.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_integration_sell_then_queue_register(self):
        """매도 성공 → MVP-2 큐 자동 등록 + 3단계 PENDING."""
        # AUTO_SELL 강제 활성화
        self.mvp1.AUTO_SELL = True

        # broker mock — 천장 25,000, 현재가 24,500 (-2% = trigger), 매도 성공, 가용 300만
        peak_price = 25_000
        current_price = 24_500
        rows = _make_ohlcv_rows(peak_price, peak_days_ago=2)

        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": str(current_price)}}
        broker.fetch_ohlcv.return_value = {"output2": rows}
        # P0-D (5/28): raw broker.create_market_sell_order fallback 차단됨 → adapter 메서드 mock 보강
        sell_order_mock = MagicMock()
        sell_order_mock.order_id = "SELL999"
        broker.sell_limit.return_value = sell_order_mock
        broker.sell_market.return_value = sell_order_mock
        broker.create_market_sell_order.return_value = {"output": {"ODNO": "SELL999"}}
        broker.get_available_cash.return_value = 3_000_000

        # MVP-1: 천장 감지
        sig = self.mvp1.detect_peak_signal(broker, "240810", current_price=current_price)
        self.assertTrue(sig.trigger, f"trigger=False: {sig.reasons_fail}")
        self.assertTrue(sig.auto_sell_eligible)
        self.assertEqual(sig.peak_price, peak_price)

        # MVP-1: 매도 실행 (MVP-2 큐 자동 등록 트리거)
        result = self.mvp1.execute_auto_sell(broker, sig, holdings_qty=10)
        self.assertTrue(result["success"])
        self.assertEqual(result["order_id"], "SELL999")
        self.assertTrue(
            result.get("queue_registered"),
            f"queue_registered=False: {result.get('queue_error')}",
        )

        # MVP-2: 큐 파일에 저장 확인
        entry = self.mvp2.get_queue_status("240810")
        self.assertIsNotNone(entry, "큐 파일에 종목 미등록")
        self.assertEqual(entry["peak_price"], peak_price)
        self.assertEqual(len(entry["stages"]), 3)

        # 3단계 모두 PENDING
        for stage in entry["stages"]:
            self.assertEqual(stage["status"], "PENDING")

        # L1 지정가 = 25,000 × 0.90 = 22,500
        self.assertEqual(entry["stages"][0]["target_price"], 22_500)

    def test_integration_sell_off_no_queue(self):
        """AUTO_SELL=0 시 매도 안 됨 → 큐 등록도 안 됨."""
        # 기본값 False
        self.assertFalse(self.mvp1.AUTO_SELL)

        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": "24500"}}
        broker.fetch_ohlcv.return_value = {
            "output2": _make_ohlcv_rows(25_000, peak_days_ago=2),
        }
        broker.get_available_cash.return_value = 3_000_000

        sig = self.mvp1.detect_peak_signal(broker, "240810", current_price=24_500)
        # auto_sell_eligible = False (AUTO_SELL=0)
        self.assertFalse(sig.auto_sell_eligible)

        result = self.mvp1.execute_auto_sell(broker, sig, holdings_qty=10)
        self.assertFalse(result["success"])

        # 큐 미등록
        self.assertIsNone(self.mvp2.get_queue_status("240810"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
