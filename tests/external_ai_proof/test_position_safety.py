"""test_position_safety.py — 외부 AI 제안 8개 유닛 테스트 (2026-05-21 검증용)

실행: python -m pytest tests/external_ai_proof/test_position_safety.py -v
또는: python tests/external_ai_proof/test_position_safety.py

원리:
  실제 KIS API 호출 X. 가짜 데이터(MagicMock)로 "이런 상황이면 코드가
  어떻게 동작해야 하는지"를 검증.

검증 시나리오 (8건):
  1. KIS에만 있는 종목 → 메모리 자동 등록 + 기본 SL 세팅
  2. 메모리에만 있는 종목 → 자동 제거 (매도 완료)
  3. 수량 불일치 → KIS 기준 덮어쓰기
  4. KIS API 실패(None) → 기존 메모리 보존 (절대 안전)
  5. SL=None → 기본 SL 자동 세팅
  6. 손실 -15.4% (킬라인 -5% 초과) → 시장가 매도
  7. 손실 -3% (킬라인 -5% 미만) → 매도 안 함
  8. dry_run=True → 로깅만, 매도 안 함
"""

import sys
import os
import copy
import unittest
from unittest.mock import MagicMock

# 같은 디렉토리의 position_safety 모듈 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from position_safety import sync_positions, enforce_sl, hard_kill_check


class FakeAutoTrader:
    """가짜 AutoTrader (테스트용). 실제 KIS API 호출 X."""

    def __init__(self):
        self._positions = {}
        self.kis = MagicMock()
        self._saved = False

    def _save_positions(self):
        self._saved = True

    def _record_trade(self, **kwargs):
        self.last_trade_record = kwargs


# position_safety 함수들을 FakeAutoTrader에 메서드로 부착
FakeAutoTrader.sync_positions = sync_positions
FakeAutoTrader.enforce_sl = enforce_sl
FakeAutoTrader.hard_kill_check = hard_kill_check


# ============================================================
# TestSyncPositions
# ============================================================

class TestSyncPositions(unittest.TestCase):
    """sync_positions() — KIS ↔ 메모리 동기화"""

    def test_1_kis_only_auto_register(self):
        """[1] KIS에만 있는 종목 → 자동 등록 + 기본 SL 세팅 (로킷헬스케어 사고 재현)"""
        trader = FakeAutoTrader()

        # 메모리: LG전자만 있음
        trader._positions = {
            "066570": {"name": "LG전자", "qty": 1, "buy_price": 196400,
                       "stop_loss": 190508, "take_profit": 206220}
        }

        # KIS: LG전자 + 로킷헬스케어 (메모리에 없는 거)
        trader.kis.fetch_balance.return_value = [
            {"code": "066570", "name": "LG전자", "qty": 1, "buy_price": 196400},
            {"code": "376900", "name": "로킷헬스케어", "qty": 1, "buy_price": 61600},
        ]

        trader.sync_positions()

        # 검증: 로킷헬스케어 자동 등록
        self.assertIn("376900", trader._positions,
                      "로킷헬스케어가 메모리에 자동 등록되어야 함")

        # 검증: 기본 SL 세팅 (-3%)
        rokit = trader._positions["376900"]
        expected_sl = round(61600 * 0.97)
        self.assertEqual(rokit["stop_loss"], expected_sl,
                         f"기본 SL이 {expected_sl}이어야 함")

        # 검증: 저장 호출
        self.assertTrue(trader._saved, "positions.json 저장 호출되어야 함")

        print("OK [1] KIS only -> auto register + SL")

    def test_2_memory_only_remove(self):
        """[2] 메모리에만 있는 종목 → 제거 (매도 완료 추정)"""
        trader = FakeAutoTrader()

        trader._positions = {
            "066570": {"name": "LG전자", "qty": 1, "buy_price": 196400},
            "000000": {"name": "켄코아에어로", "qty": 1, "buy_price": 50000},
            "111111": {"name": "아이씨티케이", "qty": 1, "buy_price": 30000},
        }

        # KIS: LG전자만 (나머지 2개 매도됨)
        trader.kis.fetch_balance.return_value = [
            {"code": "066570", "name": "LG전자", "qty": 1, "buy_price": 196400},
        ]

        trader.sync_positions()

        self.assertNotIn("000000", trader._positions, "켄코아 제거되어야 함")
        self.assertNotIn("111111", trader._positions, "아이씨티케이 제거되어야 함")
        self.assertIn("066570", trader._positions, "LG전자 유지되어야 함")

        print("OK [2] memory only -> remove")

    def test_3_qty_mismatch_overwrite(self):
        """[3] 수량 불일치 → KIS 기준 덮어쓰기 (부분 매도 후 동기화)"""
        trader = FakeAutoTrader()

        # 메모리: 2주
        trader._positions = {
            "066570": {"name": "LG전자", "qty": 2, "buy_price": 196400}
        }

        # KIS: 1주 (부분 매도됨)
        trader.kis.fetch_balance.return_value = [
            {"code": "066570", "name": "LG전자", "qty": 1, "buy_price": 196400},
        ]

        trader.sync_positions()

        self.assertEqual(trader._positions["066570"]["qty"], 1,
                         "KIS 기준 1주로 덮어쓰여야 함")

        print("OK [3] qty mismatch -> KIS overwrite")

    def test_4_api_failure_preserve(self):
        """[4] API 실패(None) → 기존 메모리 보존 (절대 안전)"""
        trader = FakeAutoTrader()

        trader._positions = {
            "066570": {"name": "LG전자", "qty": 1, "buy_price": 196400}
        }

        # API 실패
        trader.kis.fetch_balance.return_value = None

        original = copy.deepcopy(trader._positions)
        trader.sync_positions()

        # 메모리 변경 없어야 함
        self.assertEqual(trader._positions, original,
                         "API 실패 시 기존 데이터 보존 필수")

        print("OK [4] API failure -> preserve")


# ============================================================
# TestEnforceSL
# ============================================================

class TestEnforceSL(unittest.TestCase):
    """enforce_sl() — SL=None 자동 세팅"""

    def test_5_sl_none_auto_set(self):
        """[5] SL=None인 종목 → 기본 SL 자동 세팅, 이미 있는 종목은 건드리지 않음"""
        trader = FakeAutoTrader()

        trader._positions = {
            "219420": {"name": "링크제니시스", "qty": 1, "buy_price": 5030,
                       "stop_loss": None, "take_profit": None},
            "066570": {"name": "LG전자", "qty": 1, "buy_price": 196400,
                       "stop_loss": 190508, "take_profit": 206220},
        }

        trader.enforce_sl(default_sl_pct=0.03, default_tp_pct=0.05)

        # 검증: SL=None 종목 자동 세팅
        link = trader._positions["219420"]
        expected_sl = round(5030 * 0.97)
        self.assertEqual(link["stop_loss"], expected_sl)

        # 검증: 기존 SL 있는 종목 건드리지 않음
        lg = trader._positions["066570"]
        self.assertEqual(lg["stop_loss"], 190508,
                         "기존 SL이 있으면 덮어쓰면 안 됨")

        print("OK [5] SL=None -> default SL, existing preserved")


# ============================================================
# TestHardKillCheck
# ============================================================

class TestHardKillCheck(unittest.TestCase):
    """hard_kill_check() — 최후 방어선"""

    def test_6_kill_line_exceeded(self):
        """[6] 로킷헬스케어 -15.4% → 시장가 매도 (-5% 킬라인 초과)"""
        trader = FakeAutoTrader()

        trader.kis.fetch_balance.return_value = [
            {"code": "376900", "name": "로킷헬스케어", "qty": 1, "buy_price": 61600},
        ]
        trader.kis.fetch_price.return_value = 52100  # -15.4%
        trader.kis.sell_market.return_value = {"success": True}

        killed = trader.hard_kill_check(kill_pct=0.05, dry_run=False)

        # 검증: sell_market 호출됨
        trader.kis.sell_market.assert_called_once_with("376900", 1)

        # 검증: 결과 기록
        self.assertEqual(len(killed), 1)
        self.assertEqual(killed[0]["code"], "376900")
        self.assertEqual(killed[0]["action"], "KILLED")

        print("OK [6] -15.4% -> market sell")

    def test_7_within_kill_line(self):
        """[7] 루미르 -3% (킬라인 -5% 미만) → 매도 안 함"""
        trader = FakeAutoTrader()

        trader.kis.fetch_balance.return_value = [
            {"code": "474170", "name": "루미르", "qty": 1, "buy_price": 12620},
        ]
        trader.kis.fetch_price.return_value = 12241  # -3%

        killed = trader.hard_kill_check(kill_pct=0.05, dry_run=False)

        trader.kis.sell_market.assert_not_called()
        self.assertEqual(len(killed), 0, "킬라인 미만은 매도 안 함")

        print("OK [7] -3% within kill_line -> no sell")

    def test_8_dry_run_no_sell(self):
        """[8] dry_run=True → 매도 안 하고 로깅만"""
        trader = FakeAutoTrader()

        trader.kis.fetch_balance.return_value = [
            {"code": "376900", "name": "로킷헬스케어", "qty": 1, "buy_price": 61600},
        ]
        trader.kis.fetch_price.return_value = 52100  # -15.4%

        killed = trader.hard_kill_check(kill_pct=0.05, dry_run=True)

        # sell_market 호출 X
        trader.kis.sell_market.assert_not_called()

        # 결과는 dry_run으로 기록
        self.assertEqual(killed[0]["action"], "dry_run")

        print("OK [8] dry_run -> logged only, no sell")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  position_safety.py 8 unit tests")
    print("  No real KIS API call. Fake data verification.")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
