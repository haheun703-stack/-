"""test_support_pattern_detector.py — MVP-3 받침 패턴 감지 단위 테스트.

배경 (퐝가님 5/23 흐름):
  받침 패턴 = 어제 아래꼬리 + 오늘 양봉 + 거래량 폭증 (3 조건 ALL).

5/17 자기반성 #1 적용: import + 함수 호출 + 실제 OHLCV 시나리오 검증.

검증 시나리오:
  [3 조건 ALL 통과]
   1. 받침 확정 (어제 아래꼬리 2.5배 + 오늘 양봉 1.0 ATR + 거래량 3배)

  [개별 조건 실패]
   2. 어제 아래꼬리 부족 (1.5배 < 2.0배)
   3. 오늘 양봉 약함 (close ≤ open + 0.5 ATR)
   4. 거래량 부족 (1.5배 < 2.0배)
   5. 어제 아래꼬리 없음 (시·종가가 저가)

  [방어]
   6. OHLCV 부족 (7건 미만)
   7. KILL_SWITCH 발동
   8. 5일 평균 거래량 0 (휴장)
   9. body=0 (도지) 처리 (안전)

  [포맷]
  10. trigger 텔레그램 포맷 (아래꼬리/양봉/거래량 표기)
  11. 미트리거 포맷 (실패 사유)

  [pool 스캔]
  12. scan_pool_for_support — trigger 우선 정렬

  [import 체인]
  13. 모듈 import 검증

실행:
  python -m pytest tests/test_support_pattern_detector.py -v
"""

import sys
import os
import unittest
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_row(d_str: str, o: int, h: int, l: int, c: int, v: int) -> dict:
    return {
        "stck_bsop_date": d_str,
        "stck_oprc": str(o),
        "stck_hgpr": str(h),
        "stck_lwpr": str(l),
        "stck_clpr": str(c),
        "acml_vol": str(v),
    }


def _make_ohlcv_with_support(
    yesterday_shadow_ratio: float = 2.5,
    today_bullish_atr_mult: float = 1.0,
    today_volume_mult: float = 3.0,
    base_price: int = 25_000,
    base_vol: int = 100_000,
) -> list[dict]:
    """받침 패턴 OHLCV 시나리오 생성 (최신순)."""
    today = date.today()
    rows: list[dict] = []

    # 오늘 (index 0): 양봉 — close = open + ATR * mult
    # ATR ~ 500 가정 (단순 high-low)
    atr_approx = 500
    t_open = base_price - 100
    t_close = t_open + int(atr_approx * today_bullish_atr_mult)
    t_high = t_close + 200
    t_low = t_open - 100
    t_vol = int(base_vol * today_volume_mult)
    rows.append(_make_row(today.strftime("%Y%m%d"), t_open, t_high, t_low, t_close, t_vol))

    # 어제 (index 1): 아래꼬리 — body=200, low_shadow=body * ratio
    y_open = base_price - 200
    y_close = base_price          # 양봉 본체 200
    body = abs(y_close - y_open)  # 200
    y_low = min(y_open, y_close) - int(body * yesterday_shadow_ratio)
    y_high = max(y_open, y_close) + 100
    y_vol = base_vol
    rows.append(_make_row(
        (today - timedelta(days=1)).strftime("%Y%m%d"),
        y_open, y_high, y_low, y_close, y_vol,
    ))

    # 그제~7일 전: 일반 캔들 (ATR 계산용)
    for i in range(2, 10):
        d = today - timedelta(days=i)
        p = base_price - 100 * i
        rows.append(_make_row(
            d.strftime("%Y%m%d"),
            p - 50, p + 250, p - 250, p + 50, base_vol,
        ))

    return rows


class TestSupportPatternDetector(unittest.TestCase):

    def setUp(self):
        # 모듈 reload — KILL_SWITCH_PATH 격리
        if "src.use_cases.support_pattern_detector" in sys.modules:
            del sys.modules["src.use_cases.support_pattern_detector"]

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)

        import src.use_cases.support_pattern_detector as mod
        self.mod = mod
        mod.KILL_SWITCH_PATH = tmp_path / "kill_switch.flag"

    def tearDown(self):
        self.tmpdir.cleanup()

    # ─────────────────────────────────────────────────────────
    # [3 조건 ALL 통과]
    # ─────────────────────────────────────────────────────────

    def test_01_all_pass_trigger(self):
        """받침 확정 — 아래꼬리 2.5배 + 양봉 1.0 ATR + 거래량 3배."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 1.0, 3.0)
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertTrue(sig.trigger, f"reasons_fail={sig.reasons_fail}")
        self.assertGreaterEqual(sig.yesterday_shadow_ratio, 2.0)
        self.assertGreaterEqual(sig.volume_ratio, 2.0)
        self.assertEqual(len(sig.reasons_pass), 3)

    # ─────────────────────────────────────────────────────────
    # [개별 조건 실패]
    # ─────────────────────────────────────────────────────────

    def test_02_shadow_ratio_below(self):
        """어제 아래꼬리 1.5배 < 2.0배 → 실패."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(1.5, 1.0, 3.0)
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("아래꼬리" in r for r in sig.reasons_fail))

    def test_03_bullish_weak(self):
        """오늘 양봉 약함 (mult 0.2 → close ≤ threshold)."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 0.2, 3.0)
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("양봉" in r for r in sig.reasons_fail))

    def test_04_volume_below(self):
        """거래량 1.5배 < 2.0배 → 실패."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 1.0, 1.5)
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("거래량" in r for r in sig.reasons_fail))

    def test_05_no_low_shadow(self):
        """어제 시·종가가 저가와 동일 → 아래꼬리 없음."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 1.0, 3.0)
        # 어제 row 강제 변형 — low = min(open, close)
        y = rows[1]
        y["stck_lwpr"] = str(min(int(y["stck_oprc"]), int(y["stck_clpr"])))
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("아래꼬리 없음" in r for r in sig.reasons_fail))

    # ─────────────────────────────────────────────────────────
    # [방어]
    # ─────────────────────────────────────────────────────────

    def test_06_insufficient_ohlcv(self):
        """OHLCV 7건 미만 → 에러."""
        broker = MagicMock()
        broker.fetch_ohlcv.return_value = {"output2": _make_ohlcv_with_support()[:5]}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertIn("부족", sig.error or "")

    def test_07_kill_switch_blocks(self):
        """KILL_SWITCH 발동 시 정지."""
        self.mod.KILL_SWITCH_PATH.write_text("active", encoding="utf-8")
        broker = MagicMock()

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("KILL_SWITCH" in r for r in sig.reasons_fail))
        # broker도 호출 안 됨
        broker.fetch_ohlcv.assert_not_called()

    def test_08_zero_avg_volume(self):
        """5일 평균 거래량 0 (휴장)."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 1.0, 3.0)
        # 어제~5일전 거래량 0으로
        for i in range(1, 6):
            rows[i]["acml_vol"] = "0"
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertFalse(sig.trigger)
        self.assertTrue(any("거래량 0" in r for r in sig.reasons_fail))

    def test_09_doji_body_safe(self):
        """body=0 (도지)도 안전하게 처리 (division by zero 회피)."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 1.0, 3.0)
        # 어제 시가=종가 (도지)
        rows[1]["stck_oprc"] = rows[1]["stck_clpr"]
        broker.fetch_ohlcv.return_value = {"output2": rows}

        # 예외 없이 결과 나옴
        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertIsNotNone(sig)
        self.assertIsInstance(sig.yesterday_shadow_ratio, float)

    # ─────────────────────────────────────────────────────────
    # [포맷]
    # ─────────────────────────────────────────────────────────

    def test_10_format_trigger(self):
        """trigger 텔레그램 포맷 — 아래꼬리/양봉/거래량/MVP-4 안내."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(2.5, 1.0, 3.0)
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        self.assertTrue(sig.trigger)

        msg = self.mod.format_support_signal_for_telegram(sig, "원익IPS")
        self.assertIn("원익IPS", msg)
        self.assertIn("아래꼬리", msg)
        self.assertIn("ATR", msg)
        self.assertIn("폭증", msg)
        self.assertIn("MVP-4", msg)

    def test_11_format_no_trigger(self):
        """미트리거 — 실패 사유 표기."""
        broker = MagicMock()
        rows = _make_ohlcv_with_support(1.5, 1.0, 3.0)  # 아래꼬리 실패
        broker.fetch_ohlcv.return_value = {"output2": rows}

        sig = self.mod.detect_support_pattern(broker, "240810")
        msg = self.mod.format_support_signal_for_telegram(sig)
        self.assertIn("미트리거", msg)
        self.assertIn("아래꼬리", msg)

    # ─────────────────────────────────────────────────────────
    # [pool 스캔]
    # ─────────────────────────────────────────────────────────

    def test_12_pool_scan_sort(self):
        """scan_pool — trigger 우선 + 거래량 비율 높은 순."""
        broker = MagicMock()

        def fake_fetch(ticker, **kwargs):
            if ticker == "AAA":
                return {"output2": _make_ohlcv_with_support(1.5, 1.0, 3.0)}  # 미트리거
            elif ticker == "BBB":
                return {"output2": _make_ohlcv_with_support(2.5, 1.0, 5.0)}  # 트리거 + 5배
            else:
                return {"output2": _make_ohlcv_with_support(2.5, 1.0, 3.0)}  # 트리거 + 3배

        broker.fetch_ohlcv.side_effect = fake_fetch

        results = self.mod.scan_pool_for_support(broker, ["AAA", "BBB", "CCC"])
        self.assertEqual(len(results), 3)
        # BBB(5배 trigger) → CCC(3배 trigger) → AAA(미트리거)
        self.assertEqual(results[0].ticker, "BBB")
        self.assertEqual(results[1].ticker, "CCC")
        self.assertEqual(results[2].ticker, "AAA")

    # ─────────────────────────────────────────────────────────
    # [import]
    # ─────────────────────────────────────────────────────────

    def test_13_imports(self):
        from src.use_cases.support_pattern_detector import (
            detect_support_pattern,
            scan_pool_for_support,
            format_support_signal_for_telegram,
            SupportSignal,
            SHADOW_RATIO_MIN,
            BULLISH_ATR_MULT,
            VOLUME_MULT,
        )
        self.assertTrue(callable(detect_support_pattern))
        self.assertEqual(SHADOW_RATIO_MIN, 2.0)
        self.assertEqual(VOLUME_MULT, 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
