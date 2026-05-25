"""test_adaptive_position_manager.py — MVP-1 천장 감지 + -3% 트리거 단위 테스트.

배경 (퐝가님 5/23 흐름):
  6단계 흐름의 [3단계 천장 -3% 트리거 매도] 검증.
  4 차트 패턴 (디아이티/원익IPS/ISC/일진전기 등) 자동 감지.

5/17 자기반성 #1 적용: import + 함수 호출 + main 흐름 검증.

검증 시나리오:
  [현재가 위치 시나리오]
   1. 천장 -1.8% 위치 (디아이티 실측) → trigger=True
   2. 천장 -3% 정확 → trigger=True (경계)
   3. 천장 -3.5% → trigger=False (조정 시작)
   4. 천장 -20% (분할매수 영역) → trigger=False (MVP-2 영역)
   5. 현재가 = 천장 → trigger=True

  [천장 신선도 시나리오]
   6. 천장 5일 전 도달 → trigger=True (경계)
   7. 천장 6일 전 도달 → trigger=False (묵힘)
   8. 천장 1일 전 (어제) → trigger=True (가장 fresh)

  [방어 시나리오]
   9. KILL_SWITCH 발동 중 → trigger=False
  10. OHLCV 4건만 (부족) → trigger=False
  11. 현재가 fetch 실패 → trigger=False

  [텔레그램 포맷]
  12. 트리거 시 천장가 / 현재가 / % 모두 표기
  13. 미트리거 시 사유 명시

실행:
  python -m pytest tests/test_adaptive_position_manager.py -v
"""

import sys
import os
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.use_cases.adaptive_position_manager import (
    detect_peak_signal,
    format_peak_signal_for_telegram,
    PeakSignal,
    PEAK_TRIGGER_PCT,
    PEAK_LOOKBACK_DAYS,
    PEAK_FRESHNESS_DAYS,
)


def _make_ohlcv_rows(peak_price: int, peak_days_ago: int, n_rows: int = 30) -> list[dict]:
    """가짜 OHLCV 응답 생성 — 천장이 N일 전 도달."""
    rows = []
    today = date.today()
    for i in range(n_rows):
        d = today - timedelta(days=i)
        if i == peak_days_ago:
            high = peak_price
        elif i < peak_days_ago:
            # 천장 이후: 점진 하락
            high = int(peak_price * (1 - 0.01 * (peak_days_ago - i)))
        else:
            # 천장 이전: 상승 중
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


def _mock_broker(current_price: int, ohlcv_rows: list[dict]):
    """가짜 broker."""
    broker = MagicMock()
    broker.fetch_price.return_value = {
        "output": {"stck_prpr": str(current_price)}
    }
    broker.fetch_ohlcv.return_value = {"output2": ohlcv_rows}
    return broker


class TestPeakLocation(unittest.TestCase):
    """현재가 위치 시나리오."""

    def setUp(self):
        # KILL_SWITCH 비활성화 보장
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()

    def test_1_dait_minus_1_8pct(self):
        """[1] 디아이티 실측 케이스 — 천장 28,050 / 현재 27,550 (-1.8%) → 트리거"""
        rows = _make_ohlcv_rows(peak_price=28050, peak_days_ago=2)
        broker = _mock_broker(27550, rows)
        sig = detect_peak_signal(broker, "110990")
        self.assertTrue(sig.trigger, f"실패 사유: {sig.reasons_fail}")
        self.assertEqual(sig.peak_price, 28050)
        self.assertAlmostEqual(sig.pct_from_peak, -1.78, places=1)

    def test_2_exactly_minus_3pct(self):
        """[2] 천장 -3% 정확 (경계) → 트리거"""
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=2)
        broker = _mock_broker(97000, rows)  # 정확 -3%
        sig = detect_peak_signal(broker, "TEST")
        self.assertTrue(sig.trigger, f"실패 사유: {sig.reasons_fail}")
        self.assertEqual(sig.pct_from_peak, -3.0)

    def test_3_minus_3_5pct_no_trigger(self):
        """[3] 천장 -3.5% (조정 시작) → 미트리거"""
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=2)
        broker = _mock_broker(96500, rows)
        sig = detect_peak_signal(broker, "TEST")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("미진입" in r for r in sig.reasons_fail))

    def test_4_minus_20pct_for_split_buy(self):
        """[4] 천장 -20% (분할매수 영역, MVP-2) → 미트리거"""
        rows = _make_ohlcv_rows(peak_price=151500, peak_days_ago=2)
        broker = _mock_broker(121200, rows)  # -20%
        sig = detect_peak_signal(broker, "원익IPS")
        self.assertFalse(sig.trigger)
        # 분할매수 영역으로 명확히 분리됨
        self.assertLess(sig.pct_from_peak, -10)

    def test_5_at_peak(self):
        """[5] 현재가 = 천장 → 트리거"""
        rows = _make_ohlcv_rows(peak_price=50000, peak_days_ago=1)
        broker = _mock_broker(50000, rows)
        sig = detect_peak_signal(broker, "TEST")
        self.assertTrue(sig.trigger)
        self.assertEqual(sig.pct_from_peak, 0.0)


class TestPeakFreshness(unittest.TestCase):
    """천장 신선도 시나리오."""

    def setUp(self):
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()

    def test_6_peak_5days_ago_boundary(self):
        """[6] 천장 5일 전 도달 (경계) → 트리거 (default freshness=5)"""
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=5)
        broker = _mock_broker(98000, rows)  # -2%
        sig = detect_peak_signal(broker, "TEST")
        self.assertTrue(sig.trigger, f"실패: {sig.reasons_fail}")
        self.assertEqual(sig.days_since_peak, 5)

    def test_7_peak_6days_ago_no_trigger(self):
        """[7] 천장 6일 전 도달 → 미트리거 (묵힘)"""
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=6)
        broker = _mock_broker(98000, rows)
        sig = detect_peak_signal(broker, "TEST")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("묵힘" in r for r in sig.reasons_fail))

    def test_8_peak_yesterday(self):
        """[8] 천장 1일 전 (어제) → 트리거 (가장 fresh)"""
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=1)
        broker = _mock_broker(98500, rows)  # -1.5%
        sig = detect_peak_signal(broker, "TEST")
        self.assertTrue(sig.trigger)
        self.assertEqual(sig.days_since_peak, 1)


class TestDefenseScenarios(unittest.TestCase):
    """방어 시나리오 (안전선)."""

    def test_9_kill_switch_allows_sell_blocks_buy(self):
        """[9] P0-4 (5/25 보강) — KILL_SWITCH는 매수만 차단, 매도(천장 감지)는 계속.

        검수 P0-4: 기존엔 KILL_SWITCH 시 detect_peak_signal까지 차단되어
        꺾이는 순간 매도 안 되어 손실 확대 위험. P0-4 수정으로 매도 평가 계속.
        """
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        KILL_SWITCH_PATH.touch()
        try:
            rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=1)
            broker = _mock_broker(99000, rows)
            sig = detect_peak_signal(broker, "TEST")
            # P0-4 보장: KILL_SWITCH 시에도 매도(천장 감지) 계속
            self.assertTrue(
                sig.trigger,
                "P0-4 위반: KILL_SWITCH 시 매도 차단됨 (손실 확대 위험)"
            )
            # KILL_SWITCH가 reasons_fail에 들어가지 않아야 함
            self.assertFalse(
                any("KILL_SWITCH" in r for r in sig.reasons_fail),
                "P0-4 위반: KILL_SWITCH가 매도 차단 사유로 추가됨"
            )
        finally:
            KILL_SWITCH_PATH.unlink()

    def test_10_insufficient_ohlcv(self):
        """[10] OHLCV 4건만 → 미트리거"""
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=1, n_rows=4)
        broker = _mock_broker(99000, rows)
        sig = detect_peak_signal(broker, "TEST")
        self.assertFalse(sig.trigger)
        self.assertIn("OHLCV", sig.error or "")

    def test_11_price_fetch_failed(self):
        """[11] 현재가 fetch 실패 → 미트리거"""
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()
        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": "0"}}
        sig = detect_peak_signal(broker, "TEST")
        self.assertFalse(sig.trigger)
        self.assertEqual(sig.current_price, 0)


class TestTelegramFormat(unittest.TestCase):
    """텔레그램 알림 포맷."""

    def setUp(self):
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()

    def test_12_trigger_format(self):
        """[12] 트리거 시 천장가 / 현재가 / % 모두 표기"""
        rows = _make_ohlcv_rows(peak_price=28050, peak_days_ago=2)
        broker = _mock_broker(27550, rows)
        sig = detect_peak_signal(broker, "110990")
        msg = format_peak_signal_for_telegram(sig, name="디아이티")
        self.assertIn("디아이티", msg)
        self.assertIn("27,550", msg)        # 현재가
        self.assertIn("28,050", msg)        # 천장
        self.assertIn("-1.78", msg)         # 천장 대비

    def test_13_no_trigger_format(self):
        """[13] 미트리거 시 사유 명시"""
        rows = _make_ohlcv_rows(peak_price=100000, peak_days_ago=2)
        broker = _mock_broker(85000, rows)  # -15% (분할매수 영역)
        sig = detect_peak_signal(broker, "TEST")
        msg = format_peak_signal_for_telegram(sig)
        self.assertIn("미진입", msg)


class TestSoubujangPoolMatching(unittest.TestCase):
    """5/23 실측 — 6개 통과 종목의 천장 위치 검증."""

    def setUp(self):
        from src.use_cases.adaptive_position_manager import KILL_SWITCH_PATH
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()

    def test_dait_actual(self):
        """디아이티 — 천장 28,050 / 현재 27,550 = 트리거"""
        rows = _make_ohlcv_rows(peak_price=28050, peak_days_ago=2)
        broker = _mock_broker(27550, rows)
        sig = detect_peak_signal(broker, "110990")
        self.assertTrue(sig.trigger)

    def test_isc_actual(self):
        """ISC — 천장 292,500 / 현재 210,000 = 분할매수 영역 (-28%, 미트리거)"""
        rows = _make_ohlcv_rows(peak_price=292500, peak_days_ago=10)
        broker = _mock_broker(210000, rows)
        sig = detect_peak_signal(broker, "095340")
        self.assertFalse(sig.trigger)
        self.assertLess(sig.pct_from_peak, -25)

    def test_iljin_actual(self):
        """일진전기 — 천장 147,900 / 현재 108,300 = -26.8% 미트리거"""
        rows = _make_ohlcv_rows(peak_price=147900, peak_days_ago=10)
        broker = _mock_broker(108300, rows)
        sig = detect_peak_signal(broker, "103590")
        self.assertFalse(sig.trigger)


class TestImportChain(unittest.TestCase):
    """5/17 자기반성 #1: import + 함수 호출 + main 검증."""

    def test_imports(self):
        from src.use_cases.adaptive_position_manager import (
            detect_peak_signal,
            format_peak_signal_for_telegram,
            scan_holdings_for_peaks,
            execute_auto_sell,
            PeakSignal,
        )
        self.assertTrue(callable(detect_peak_signal))
        self.assertTrue(callable(format_peak_signal_for_telegram))
        self.assertTrue(callable(scan_holdings_for_peaks))
        self.assertTrue(callable(execute_auto_sell))

    def test_dataclass(self):
        sig = PeakSignal(ticker="TEST")
        self.assertEqual(sig.ticker, "TEST")
        self.assertEqual(sig.trigger, False)
        self.assertEqual(sig.reasons_pass, [])

    def test_env_defaults(self):
        """기본 임계값 검증 (퐝가님 5/23 흐름)."""
        self.assertEqual(PEAK_LOOKBACK_DAYS, 30)
        self.assertAlmostEqual(PEAK_TRIGGER_PCT, 0.97, places=2)
        self.assertEqual(PEAK_FRESHNESS_DAYS, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
