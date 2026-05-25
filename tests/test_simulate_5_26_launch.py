"""test_simulate_5_26_launch.py — 5/26 가동 시뮬레이션 3 시나리오.

배경 (5/24 일요일 작업):
  5/26(화) 09:15 적응형 매매법 4단계 동시 실전 가동 직전 검증.
  3 가지 시장 상황에서 통합 사이클이 정확히 동작하는지 확인.

5/17 자기반성 #1 적용: import OK ≠ 동작 OK 방지.

3 시나리오:
  1. ★ 정상 강세장 — 디아이티 천장 -1.8% + 일진전기 받침 패턴
     → MVP-1 매도 시그널 발동 (AUTO_SELL=0이라 알림만)
     → MVP-3 받침 감지
     → MVP-4 재진입 (AUTO_REENTRY=0이라 알림만)

  2. ★ 약세장 KILL_SWITCH — KOSPI BEARISH 발동
     → 4단계 모두 차단 (트리거 0건)
     → 시드 보전 검증

  3. ★ 큐 도달 트리거 — 원익IPS 천장 25,000 → 현재 22,000 (L1 -10% 도달)
     → MVP-2 큐 트리거 (TRIGGERED 상태, 알림만)
     → 큐 파일에 TRIGGERED 기록

실행:
  python -m pytest tests/test_simulate_5_26_launch.py -v
"""

import sys
import os
import json
import unittest
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ohlcv_for_peak(peak_price: int, peak_days_ago: int, n: int = 30) -> list[dict]:
    """천장 시나리오 OHLCV (최신순)."""
    today = date.today()
    rows = []
    for i in range(n):
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


def _make_ohlcv_support_pattern(base_price: int = 25_000) -> list[dict]:
    """받침 패턴 OHLCV (어제 아래꼬리 + 오늘 양봉 + 거래량 폭증)."""
    today = date.today()
    rows = []

    # 오늘: 양봉 close = open + ATR(약 500) × 1.0
    t_open = base_price - 100
    t_close = t_open + 500
    rows.append({
        "stck_bsop_date": today.strftime("%Y%m%d"),
        "stck_oprc": str(t_open),
        "stck_hgpr": str(t_close + 200),
        "stck_lwpr": str(t_open - 100),
        "stck_clpr": str(t_close),
        "acml_vol": "300000",  # 5일 평균 100k × 3배
    })

    # 어제: 아래꼬리 (body=200, low_shadow=500 → 2.5배)
    y_open = base_price - 200
    y_close = base_price
    rows.append({
        "stck_bsop_date": (today - timedelta(days=1)).strftime("%Y%m%d"),
        "stck_oprc": str(y_open),
        "stck_hgpr": str(y_close + 100),
        "stck_lwpr": str(y_open - 500),
        "stck_clpr": str(y_close),
        "acml_vol": "100000",
    })

    # 그제~10일전
    for i in range(2, 11):
        d = today - timedelta(days=i)
        p = base_price - 100 * i
        rows.append({
            "stck_bsop_date": d.strftime("%Y%m%d"),
            "stck_oprc": str(p - 50),
            "stck_hgpr": str(p + 250),
            "stck_lwpr": str(p - 250),
            "stck_clpr": str(p + 50),
            "acml_vol": "100000",
        })
    return rows


def _build_mock_broker(scenario: str):
    """시나리오별 mock broker."""
    broker = MagicMock(name=f"MockBroker_{scenario}")

    def fake_fetch_price(ticker):
        if scenario == "normal_bullish":
            # 디아이티(110990) 천장 23,400 × 0.97 = 22,698 → 현재 22,980 (-1.8%, 트리거)
            prices = {
                "110990": "22980",   # 디아이티 (MVP-1 트리거)
                "240810": "21000",   # 원익IPS (L1 -16% 영역)
                "103590": "18300",   # 일진전기 (MVP-3 받침 시나리오)
            }
        elif scenario == "queue_l1_hit":
            # 원익IPS L1 도달 (천장 25,000 → 22,000 = -12%)
            prices = {"240810": "22000"}
        else:
            prices = {}
        price = prices.get(ticker, "10000")
        return {"output": {"stck_prpr": price}}

    def fake_fetch_ohlcv(ticker, **kwargs):
        if scenario == "normal_bullish":
            if ticker == "110990":
                # 디아이티 천장 23,400 (2일 전)
                return {"output2": _make_ohlcv_for_peak(23_400, peak_days_ago=2)}
            elif ticker == "103590":
                # 일진전기 받침 패턴
                return {"output2": _make_ohlcv_support_pattern(base_price=19_000)}
            else:
                return {"output2": _make_ohlcv_for_peak(25_000, peak_days_ago=15)}  # 묵힘
        return {"output2": _make_ohlcv_for_peak(25_000, peak_days_ago=15)}

    broker.fetch_price.side_effect = fake_fetch_price
    broker.fetch_ohlcv.side_effect = fake_fetch_ohlcv
    broker.fetch_balance.return_value = {"output1": []}
    return broker


class TestSimulate526Launch(unittest.TestCase):

    def setUp(self):
        # 모든 적응형 모듈 reload
        for mod_name in (
            "src.use_cases.adaptive_position_manager",
            "src.use_cases.adaptive_buy_queue",
            "src.use_cases.support_pattern_detector",
            "src.use_cases.adaptive_reentry",
        ):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)

        import src.use_cases.adaptive_position_manager as mvp1
        import src.use_cases.adaptive_buy_queue as mvp2
        import src.use_cases.support_pattern_detector as mvp3
        import src.use_cases.adaptive_reentry as mvp4
        self.mvp1, self.mvp2, self.mvp3, self.mvp4 = mvp1, mvp2, mvp3, mvp4

        kill_path = tmp_path / "kill_switch.flag"
        for mod in (mvp1, mvp2, mvp3, mvp4):
            mod.KILL_SWITCH_PATH = kill_path
        mvp2.QUEUE_PATH = tmp_path / "adaptive_buy_queue.json"
        self.kill_path = kill_path

    def tearDown(self):
        self.tmpdir.cleanup()

    # ─────────────────────────────────────────────────────────
    # 시나리오 1: 정상 강세장
    # ─────────────────────────────────────────────────────────

    def test_scenario_1_normal_bullish_full_cycle(self):
        """정상: 디아이티 매도 알림 + 일진전기 받침 → 재진입 알림."""
        broker = _build_mock_broker("normal_bullish")

        # MVP-1: 디아이티 천장 -1.8% 트리거
        sig1 = self.mvp1.detect_peak_signal(broker, "110990")
        self.assertTrue(sig1.trigger, f"MVP-1 미트리거: {sig1.reasons_fail}")
        self.assertFalse(sig1.auto_sell_eligible)  # AUTO_SELL=0 (1주차 안전)

        # MVP-3: 일진전기 받침 트리거
        sig3 = self.mvp3.detect_support_pattern(broker, "103590")
        self.assertTrue(sig3.trigger, f"MVP-3 미트리거: {sig3.reasons_fail}")

        # MVP-4: 일진전기 재진입 평가 (AUTO_REENTRY=0이라 알림만)
        step5_pool = {"103590": {"stars": 5, "upside": 7.45}}
        jarvis_check = lambda t: {"pass": True, "failed": []}
        dec = self.mvp4.evaluate_reentry(
            broker, "103590", "일진전기",
            step5_pool=step5_pool,
            jarvis_safety_check=jarvis_check,
        )
        self.assertTrue(dec.trigger, "MVP-4 3 게이트 미통과")
        self.assertTrue(dec.support_pass)
        self.assertTrue(dec.step5_pass)
        self.assertTrue(dec.jarvis_pass)
        self.assertFalse(dec.auto_reentry_eligible)  # AUTO_REENTRY=0

        # 매수 시도 — 알림만 (success=False)
        result = self.mvp4.execute_auto_reentry(broker, dec)
        self.assertFalse(result["success"])
        self.assertIn("AUTO_REENTRY=0", result["error"])

    # ─────────────────────────────────────────────────────────
    # 시나리오 2: 약세장 KILL_SWITCH
    # ─────────────────────────────────────────────────────────

    def test_scenario_2_kill_switch_blocks_buys_allows_sells(self):
        """P0-4 (5/25 보강) — KILL_SWITCH 시 매수 3단계 차단, 매도(MVP-1)는 계속.

        검수 P0-4: 기존엔 KILL_SWITCH가 매도까지 정지 → 꺾이는 순간 손실 확대.
        수정: 매수(MVP-2/3/4)만 차단, 매도(MVP-1 천장 감지)는 손실 차단 목적으로 계속.
        """
        # KILL_SWITCH 활성화
        self.kill_path.write_text("BEARISH 발동 5/26 시뮬", encoding="utf-8")

        broker = _build_mock_broker("normal_bullish")  # 환경은 강세장이지만

        # MVP-1: P0-4 보장 — 매도(천장 감지) 계속 (KILL_SWITCH 무관)
        sig1 = self.mvp1.detect_peak_signal(broker, "110990")
        self.assertFalse(
            any("KILL_SWITCH" in r for r in sig1.reasons_fail),
            "P0-4 위반: KILL_SWITCH가 매도 차단 사유로 추가됨"
        )

        # MVP-2: 매수 큐 등록 차단 (정상 — 매수 차단)
        reg = self.mvp2.register_buy_queue("240810", 25_000, 3_000_000)
        self.assertFalse(reg["success"])
        self.assertIn("KILL_SWITCH", reg["error"])

        # MVP-3: 받침 패턴(매수 시그널) 차단 (정상)
        sig3 = self.mvp3.detect_support_pattern(broker, "103590")
        self.assertFalse(sig3.trigger)
        self.assertTrue(any("KILL_SWITCH" in r for r in sig3.reasons_fail))

        # MVP-4: 재진입(매수) 차단 (정상)
        dec = self.mvp4.evaluate_reentry(
            broker, "103590", "일진전기",
            step5_pool={"103590": {"stars": 5, "upside": 7.45}},
            jarvis_safety_check=lambda t: {"pass": True, "failed": []},
        )
        self.assertFalse(dec.trigger)
        self.assertTrue(any("KILL_SWITCH" in c for c in dec.jarvis_failed_checks))

    # ─────────────────────────────────────────────────────────
    # 시나리오 3: 큐 도달 트리거
    # ─────────────────────────────────────────────────────────

    def test_scenario_3_queue_L1_hit(self):
        """원익IPS L1 -10% 도달 → TRIGGERED 상태 (AUTO_BUY=0)."""
        # 사전 등록 (천장 25,000 + 가용 300만)
        reg = self.mvp2.register_buy_queue(
            "240810", peak_price=25_000, available_cash=3_000_000, name="원익IPS",
        )
        self.assertTrue(reg["success"])
        # 3단계 PENDING 확인
        for stage in reg["stages"]:
            self.assertEqual(stage["status"], "PENDING")

        # 가격 22,000 도달 (L1=22,500, L2=20,000)
        broker = _build_mock_broker("queue_l1_hit")
        triggers = self.mvp2.check_and_trigger_queues(broker)

        # L1만 도달
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["level"], 1)
        self.assertEqual(triggers[0]["status"], "TRIGGERED")  # AUTO_BUY=0

        # 큐 파일에 TRIGGERED 기록 확인
        entry = self.mvp2.get_queue_status("240810")
        self.assertEqual(entry["stages"][0]["status"], "TRIGGERED")
        self.assertEqual(entry["stages"][1]["status"], "PENDING")  # L2 미도달
        self.assertEqual(entry["stages"][2]["status"], "PENDING")  # L3 미도달

    # ─────────────────────────────────────────────────────────
    # 시나리오 4 (보너스): MVP-2 큐가 있으면 MVP-4 재진입 스킵
    # ─────────────────────────────────────────────────────────

    def test_scenario_4_mvp2_priority_over_mvp4(self):
        """원익IPS가 MVP-2 큐에 있으면 MVP-4 재진입 스킵 (중복 방지)."""
        # 사전 큐 등록
        self.mvp2.register_buy_queue("240810", 25_000, 3_000_000, name="원익IPS")

        broker = _build_mock_broker("normal_bullish")
        dec = self.mvp4.evaluate_reentry(
            broker, "240810", "원익IPS",
            step5_pool={"240810": {"stars": 4, "upside": 4.88}},
            jarvis_safety_check=lambda t: {"pass": True, "failed": []},
        )

        # already_in_queue=True + trigger=False (위임)
        self.assertTrue(dec.already_in_queue)
        self.assertFalse(dec.trigger)


if __name__ == "__main__":
    unittest.main(verbosity=2)
