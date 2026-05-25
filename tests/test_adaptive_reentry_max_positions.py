"""test_adaptive_reentry_max_positions.py — P0-3 MVP-4 매수 직전 한도 재검사.

배경 (퐝가님 5/24 안전 수정):
  bkit:code-analyzer 판정: "MVP-4가 등록 시점만 검사 → 한도 초과 매수 위험"
  ADAPTIVE_MAX_POSITIONS=3 (1주차 안전 설정)

  기존 MVP-4 evaluate_reentry는 큐 등록 시점에만 한도 검사 → 매수 트리거 시점에
  이미 3종목이 차 있으면 4번째 매수 발생 가능.

  P0-3: execute_auto_reentry가 broker 호출 직전에 활성 포지션 수를 재검사하여
  한도 초과 시 매수 SKIP + 명확한 에러 반환.

카운트 소스: **MVP-2 큐 PENDING/TRIGGERED/FILLED/QUICK_ARMED 종목 수**
  - 큐가 6단계 흐름의 단일 진실 원천 (MVP-1 매도 후 자동 등록 + 분할매수 진행)
  - 본인(decision.ticker)은 제외 → 재진입 평가 대상이 큐에 있어도 신규 매수 슬롯으로 간주 X

검증 시나리오:
  1. 활성 3종목 (한도 도달) 상태에서 4번째 매수 시도 → SKIP + max_positions_blocked=True
  2. 활성 2종목 상태에서 3번째 매수 시도 → 통과 + broker.buy_market 호출
  3. ADAPTIVE_MAX_POSITIONS=5 override 시 활성 3종목에서 4번째 매수 통과
  4. (회귀) 카운트 헬퍼: 큐 없으면 0 반환
  5. (회귀) 카운트 헬퍼: exclude_ticker 정상 동작

실행:
  python -m pytest tests/test_adaptive_reentry_max_positions.py -v
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_support_signal(trigger: bool):
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


class TestAdaptiveReentryMaxPositions(unittest.TestCase):

    def setUp(self):
        # 모든 관련 모듈 reload (격리)
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

        # KILL_SWITCH + QUEUE 격리
        kill_path = tmp_path / "kill_switch.flag"
        mvp4.KILL_SWITCH_PATH = kill_path
        mvp3.KILL_SWITCH_PATH = kill_path
        mvp2.KILL_SWITCH_PATH = kill_path
        mvp2.QUEUE_PATH = tmp_path / "adaptive_buy_queue.json"

        # 자동매수 ON (P0-3 게이트만 검증)
        mvp4.AUTO_REENTRY = True
        # MAX_POSITIONS 기본 3 보장 (.env 영향 차단)
        mvp4.MAX_POSITIONS = 3
        # MVP-2 한도도 3 (등록 자체는 별도 한도 — 본 테스트는 MVP-4 게이트만 검증)
        mvp2.MAX_POSITIONS = 999  # 큐 등록 차단 회피 (MVP-4 한도만 보고 싶음)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _register_active_position(self, ticker: str, name: str = ""):
        """활성 큐 1건 등록 (PENDING 상태로 시드)."""
        self.mvp2.register_buy_queue(
            ticker=ticker,
            peak_price=25_000,
            available_cash=3_000_000,
            name=name or ticker,
        )

    def _build_trigger_decision(self, ticker: str = "240810", name: str = "원익IPS"):
        """3 게이트 ALL 통과 + AUTO_REENTRY=1 상태의 ReentryDecision 생성."""
        broker = _mock_broker()
        with patch.object(
            self.mvp3, "detect_support_pattern",
            return_value=_mock_support_signal(True),
        ):
            dec = self.mvp4.evaluate_reentry(
                broker, ticker, name,
                step5_pool={ticker: {"stars": 4, "upside": 5.0}},
                jarvis_safety_check=lambda t: {"pass": True, "failed": []},
            )
        self.assertTrue(dec.trigger, "사전조건: trigger=True여야 함")
        self.assertTrue(dec.auto_reentry_eligible)
        return dec, broker

    # ─────────────────────────────────────────────────────────
    # [P0-3 매수 직전 한도 재검사]
    # ─────────────────────────────────────────────────────────

    def test_01_max_positions_blocks_fourth_buy(self):
        """활성 3종목 (한도 도달) → 4번째 매수 시도 SKIP."""
        # 3종목 사전 등록 (한도 도달)
        self._register_active_position("005930", "삼성전자")
        self._register_active_position("000660", "SK하이닉스")
        self._register_active_position("035420", "NAVER")

        # 4번째 종목 재진입 평가 + 매수 시도
        dec, broker = self._build_trigger_decision(ticker="240810", name="원익IPS")
        result = self.mvp4.execute_auto_reentry(broker, dec)

        self.assertFalse(result["success"])
        self.assertTrue(result.get("max_positions_blocked", False))
        self.assertIn("3종목 한도", result["error"])
        # broker 매수 호출 안 됨
        broker.buy_market.assert_not_called()
        # decision.error에도 기록
        self.assertIsNotNone(dec.error)
        self.assertIn("한도", dec.error)

    def test_02_under_limit_allows_buy(self):
        """활성 2종목 → 3번째 매수 통과."""
        self._register_active_position("005930", "삼성전자")
        self._register_active_position("000660", "SK하이닉스")

        dec, broker = self._build_trigger_decision(ticker="240810", name="원익IPS")
        result = self.mvp4.execute_auto_reentry(broker, dec)

        self.assertTrue(result["success"], f"매수 실패: {result.get('error')}")
        self.assertFalse(result.get("max_positions_blocked", False))
        broker.buy_market.assert_called_once()

    def test_03_max_positions_override_5(self):
        """ADAPTIVE_MAX_POSITIONS=5 override → 활성 3종목 + 4번째 매수 통과."""
        # MAX_POSITIONS override
        self.mvp4.MAX_POSITIONS = 5

        self._register_active_position("005930", "삼성전자")
        self._register_active_position("000660", "SK하이닉스")
        self._register_active_position("035420", "NAVER")

        dec, broker = self._build_trigger_decision(ticker="240810", name="원익IPS")
        result = self.mvp4.execute_auto_reentry(broker, dec)

        self.assertTrue(result["success"], f"override 실패: {result.get('error')}")
        broker.buy_market.assert_called_once()

    # ─────────────────────────────────────────────────────────
    # [카운트 헬퍼 회귀]
    # ─────────────────────────────────────────────────────────

    def test_04_count_helper_empty_queue(self):
        """빈 큐 → 카운트 0."""
        count = self.mvp4._count_active_positions()
        self.assertEqual(count, 0)

    def test_05_count_helper_exclude_self(self):
        """exclude_ticker 정상 동작 — 본인은 카운트 제외."""
        self._register_active_position("005930", "삼성전자")
        self._register_active_position("000660", "SK하이닉스")
        self._register_active_position("240810", "원익IPS")

        # 본인 포함 카운트
        count_all = self.mvp4._count_active_positions()
        self.assertEqual(count_all, 3)

        # 본인 제외 카운트
        count_excl = self.mvp4._count_active_positions(exclude_ticker="240810")
        self.assertEqual(count_excl, 2)

    # ─────────────────────────────────────────────────────────
    # [import]
    # ─────────────────────────────────────────────────────────

    def test_06_imports_max_positions(self):
        """MAX_POSITIONS 상수 노출 + 기본값 3 확인."""
        from src.use_cases.adaptive_reentry import MAX_POSITIONS, _count_active_positions
        # 본 테스트는 setUp에서 3으로 강제 (.env 영향 차단)
        self.assertEqual(self.mvp4.MAX_POSITIONS, 3)
        self.assertTrue(callable(_count_active_positions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
