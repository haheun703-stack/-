"""test_entry_gates_volume_power.py — entry_gates.py 유닛 테스트 (2026-05-21)

배경:
  5/21 1주차 첫날 자비스 12회 cron 전부 매수 0건 사고.
  체결강도 게이트(C) 데이터 결측 + 임계 150 단타봇 기준 부적합.
  commit 5215a6b: KIS inquire-ccnl 직접 호출
  commit 174f296: threshold<=0 비활성화 분기 추가

  외부 AI 제안 유닛 테스트 패턴 적용 — 우리 시스템 핵심 모듈 검증.

검증 시나리오:
  1. threshold=0 (비활성화) → vp=0이어도 passed=True, source="disabled"
  2. threshold=0 (비활성화) → vp>0이어도 passed=True (로깅만)
  3. threshold=150 + vp=200 (충족) → passed=True, source="ccnl_tday_rltv"
  4. threshold=150 + vp=100 (미달) → passed=False
  5. threshold=150 + vp=0 source="none" (API 실패) → passed=False
  6. inquire-ccnl API 응답 정상 → vp 정확 추출
  7. inquire-ccnl API 실패 → fetch_price fallback 작동
  8. broker 예외 (네트워크 등) → source="none"

실행:
  python -m pytest tests/test_entry_gates_volume_power.py -v
  또는: python tests/test_entry_gates_volume_power.py
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.use_cases.entry_gates import (
    check_volume_power_gate,
    _fetch_volume_power,
)


class TestVolumePowerGateDisabled(unittest.TestCase):
    """C게이트 비활성화 분기 (threshold<=0, commit 174f296)"""

    def test_1_disabled_threshold_zero_with_vp_zero(self):
        """[1] threshold=0 + vp=0 (API 실패) → passed=True (비활성화 의도)"""
        broker = MagicMock()
        # inquire-ccnl 실패 시나리오 → vp=0 source="none"
        with patch("src.use_cases.entry_gates._fetch_volume_power", return_value=(0.0, "none")):
            result = check_volume_power_gate(broker, "067310", threshold=0)

        self.assertTrue(result["passed"], "threshold=0이면 vp=0이어도 통과해야 함")
        self.assertEqual(result["source"], "disabled")
        self.assertEqual(result["threshold"], 0)
        self.assertIn("비활성화", result["reason"])
        print("OK [1] threshold=0 + vp=0 -> passed=True (disabled)")

    def test_2_disabled_threshold_zero_with_vp_positive(self):
        """[2] threshold=0 + vp=112 (정상) → passed=True + 측정값 로깅"""
        broker = MagicMock()
        with patch("src.use_cases.entry_gates._fetch_volume_power", return_value=(112.66, "ccnl_tday_rltv")):
            result = check_volume_power_gate(broker, "067310", threshold=0)

        self.assertTrue(result["passed"])
        self.assertEqual(result["source"], "disabled")
        self.assertEqual(result["volume_power"], 112.66, "측정값은 로깅용으로 기록")
        print("OK [2] threshold=0 + vp=112.66 -> passed=True + vp logged")

    def test_3_disabled_threshold_negative(self):
        """[3] threshold=-1 (명시적 비활성화) → passed=True"""
        broker = MagicMock()
        with patch("src.use_cases.entry_gates._fetch_volume_power", return_value=(50.0, "ccnl_tday_rltv")):
            result = check_volume_power_gate(broker, "067310", threshold=-1)

        self.assertTrue(result["passed"], "threshold<0도 비활성화로 처리")
        self.assertEqual(result["source"], "disabled")
        print("OK [3] threshold=-1 -> passed=True (disabled)")


class TestVolumePowerGateActive(unittest.TestCase):
    """C게이트 활성 (threshold>0, 정상 평가)"""

    def test_4_active_passed(self):
        """[4] threshold=150 + vp=200 → passed=True"""
        broker = MagicMock()
        with patch("src.use_cases.entry_gates._fetch_volume_power", return_value=(200.0, "ccnl_tday_rltv")):
            result = check_volume_power_gate(broker, "067310", threshold=150)

        self.assertTrue(result["passed"])
        self.assertEqual(result["source"], "ccnl_tday_rltv")
        self.assertIn("강한 매수세", result["reason"])
        print("OK [4] threshold=150 + vp=200 -> passed=True (strong)")

    def test_5_active_failed(self):
        """[5] threshold=150 + vp=100 → passed=False (5/21 사고 시나리오)"""
        broker = MagicMock()
        with patch("src.use_cases.entry_gates._fetch_volume_power", return_value=(100.0, "ccnl_tday_rltv")):
            result = check_volume_power_gate(broker, "067310", threshold=150)

        self.assertFalse(result["passed"])
        self.assertIn("매수세 부족", result["reason"])
        print("OK [5] threshold=150 + vp=100 -> passed=False (weak)")

    def test_6_active_api_failure(self):
        """[6] threshold=150 + source='none' (API 실패) → passed=False"""
        broker = MagicMock()
        with patch("src.use_cases.entry_gates._fetch_volume_power", return_value=(0.0, "none")):
            result = check_volume_power_gate(broker, "067310", threshold=150)

        self.assertFalse(result["passed"])
        self.assertIn("fetch 실패", result["reason"])
        print("OK [6] threshold=150 + API failure -> passed=False")


class TestFetchVolumePowerKisAPI(unittest.TestCase):
    """_fetch_volume_power KIS inquire-ccnl 호출 (commit 5215a6b 검증)"""

    @patch("src.use_cases.entry_gates._rq", create=True)
    def test_7_ccnl_api_success(self, mock_rq):
        """[7] inquire-ccnl 정상 응답 → vp 정확 추출 + source='ccnl_tday_rltv'"""
        import requests as _rq_module
        # mock requests.get
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rt_cd": "0",
            "msg1": "정상처리",
            "output": [
                {"tday_rltv": "112.66", "stck_prpr": "54500"}
            ]
        }
        # entry_gates에서 import requests as _rq 형태로 사용
        with patch("requests.get", return_value=mock_response):
            broker = MagicMock()
            broker.access_token = "Bearer test"
            broker.api_key = "test_key"
            broker.api_secret = "test_secret"
            vp, source = _fetch_volume_power(broker, "067310")

        self.assertEqual(vp, 112.66)
        self.assertEqual(source, "ccnl_tday_rltv")
        print("OK [7] inquire-ccnl success -> vp=112.66 source=ccnl_tday_rltv")

    def test_8_ccnl_api_failure_fallback(self):
        """[8] inquire-ccnl 실패 → fetch_price fallback → tday_rltv 또는 0"""
        broker = MagicMock()
        broker.access_token = "Bearer test"
        broker.api_key = "test_key"
        broker.api_secret = "test_secret"

        # inquire-ccnl 호출 시 예외 → fallback으로 fetch_price 호출
        broker.fetch_price.return_value = {
            "output": {"tday_rltv": "75.5", "shnu_cntg_smtn": 0, "seln_cntg_smtn": 0}
        }

        with patch("requests.get", side_effect=Exception("Network error")):
            vp, source = _fetch_volume_power(broker, "067310")

        # fetch_price fallback에서 tday_rltv 추출
        self.assertEqual(vp, 75.5)
        self.assertEqual(source, "fetch_price_fallback")
        print("OK [8] inquire-ccnl failure -> fetch_price fallback (vp=75.5)")


if __name__ == "__main__":
    print("=" * 60)
    print("  entry_gates.py check_volume_power_gate 8 unit tests")
    print("  C gate disable + active + KIS API verification")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
