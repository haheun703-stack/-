"""test_macro_signal_scorer.py — 옵션 C 매크로 시그널 통합 유닛 테스트 (2026-05-23).

배경 (5/23 옵션 C):
  realtime_score에 정보봇 intelligence_macro 매크로 시그널 추가.
  Perplexity 기반 매크로 (기재부 AI 데이터센터, 젠슨황 슈퍼사이클, FOMC 등) → ±3 가산.

5/17 자기반성 #1 적용:
  "import OK = 동작 OK" 오판 금지.
  함수 호출 + main 흐름 dry-run까지 검증.

검증 시나리오:
  1. Supabase 미연결 → score=0, n_signals=0
  2. 시그널 없음 → score=0
  3. ticker 매칭 + CRITICAL 호재 → +2~+3 (multiplier 1.5)
  4. sector 매칭 + WARNING 호재 → +1
  5. 글로벌만 + CRITICAL 악재 → -1 (multiplier 0.5)
  6. 호재+악재 혼재 → 자연 상쇄
  7. 신선도 stale (72h 초과) → score=0
  8. 신선도 half (24~72h) → score 절반
  9. tags 호재 키워드 (sentiment 약함) → 보조 부호
  10. realtime_score 통합 후 breakdown에 매크로_X 포함

실행:
  python -m pytest tests/test_macro_signal_scorer.py -v
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.use_cases.macro_signal_scorer import (
    calculate_macro_signal_score,
    _score_single_macro,
    _freshness_weight,
    _normalize_list,
    SEVERITY_BASE,
    MATCH_TICKER,
    MATCH_SECTOR,
    MATCH_GLOBAL,
)


def _now_iso(hours_ago: int = 0) -> str:
    """현재 시각 - hours_ago, ISO 포맷."""
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat()


class TestNormalizeList(unittest.TestCase):
    def test_jsonb_list(self):
        self.assertEqual(_normalize_list(["a", "b", "c"]), ["a", "b", "c"])

    def test_csv_string(self):
        self.assertEqual(_normalize_list("a, b ,c"), ["a", "b", "c"])

    def test_none(self):
        self.assertEqual(_normalize_list(None), [])

    def test_empty(self):
        self.assertEqual(_normalize_list([]), [])


class TestFreshnessWeight(unittest.TestCase):
    def test_fresh_under_24h(self):
        self.assertEqual(_freshness_weight(_now_iso(1)), 1.0)
        self.assertEqual(_freshness_weight(_now_iso(23)), 1.0)

    def test_stale_24_to_72h(self):
        self.assertEqual(_freshness_weight(_now_iso(48)), 0.5)
        self.assertEqual(_freshness_weight(_now_iso(70)), 0.5)

    def test_expired_over_72h(self):
        self.assertEqual(_freshness_weight(_now_iso(100)), 0.0)

    def test_none(self):
        self.assertEqual(_freshness_weight(None), 1.0)


class TestSingleMacroScore(unittest.TestCase):
    def test_ticker_match_critical_positive(self):
        """[3] ticker 직접 매칭 + CRITICAL 호재 → +2 × 1.5 = +3"""
        row = {
            "severity": "CRITICAL",
            "sentiment_score": 50,
            "tags": ["AI수혜"],
            "affected_tickers": ["005930"],
            "affected_sectors": ["반도체"],
            "published_at": _now_iso(2),
            "title": "기재부 AI 데이터센터",
        }
        score, mode, _ = _score_single_macro(row, "005930", "반도체")
        self.assertEqual(score, 3, "CRITICAL 호재 ticker 매칭 → +3")
        self.assertEqual(mode, "ticker")

    def test_sector_match_warning_positive(self):
        """[4] sector 매칭 + WARNING 호재 → +1 × 1.0 = +1"""
        row = {
            "severity": "WARNING",
            "sentiment_score": 40,
            "tags": ["반도체호재"],
            "affected_tickers": [],
            "affected_sectors": ["반도체"],
            "published_at": _now_iso(2),
            "title": "젠슨황 슈퍼사이클",
        }
        score, mode, _ = _score_single_macro(row, "005930", "반도체")
        self.assertEqual(score, 1)
        self.assertEqual(mode, "sector")

    def test_global_only_critical_negative(self):
        """[5] 글로벌 + CRITICAL 악재 → -2 × 0.5 = -1"""
        row = {
            "severity": "CRITICAL",
            "sentiment_score": -50,
            "tags": ["금리인상"],
            "affected_tickers": [],
            "affected_sectors": [],
            "published_at": _now_iso(2),
            "title": "FOMC 금리인상",
        }
        score, mode, _ = _score_single_macro(row, "005930", "반도체")
        self.assertEqual(score, -1)
        self.assertEqual(mode, "global")

    def test_stale_signal_half_weight(self):
        """[8] 24~72h 시그널 → fresh 0.5로 절반"""
        row = {
            "severity": "CRITICAL",
            "sentiment_score": 50,
            "tags": ["AI수혜"],
            "affected_tickers": ["005930"],
            "affected_sectors": [],
            "published_at": _now_iso(48),
            "title": "(stale) AI 호재",
        }
        score, _, _ = _score_single_macro(row, "005930", "반도체")
        # +2 × 1.5(ticker) × 0.5(stale) = +1.5 → round → +2
        self.assertEqual(score, 2)

    def test_expired_signal_zero(self):
        """[7] 72h 초과 → 0"""
        row = {
            "severity": "CRITICAL",
            "sentiment_score": 50,
            "tags": ["AI수혜"],
            "affected_tickers": ["005930"],
            "affected_sectors": [],
            "published_at": _now_iso(100),
            "title": "expired",
        }
        score, _, _ = _score_single_macro(row, "005930", "반도체")
        self.assertEqual(score, 0)

    def test_tag_only_positive(self):
        """[9] sentiment 약함 + tags 호재만 → 보조 부호 +1"""
        row = {
            "severity": "WARNING",
            "sentiment_score": 0,
            "tags": ["AI수혜", "데이터센터수혜"],
            "affected_tickers": ["005930"],
            "affected_sectors": [],
            "published_at": _now_iso(2),
            "title": "tag positive",
        }
        score, mode, _ = _score_single_macro(row, "005930", "반도체")
        # WARNING(1) × +1(tag) × 1.5(ticker) = 1.5 → +2
        self.assertEqual(score, 2)
        self.assertEqual(mode, "ticker")

    def test_neutral_info_zero(self):
        """INFO + sentiment 0 + 매칭 없음 → 0"""
        row = {
            "severity": "INFO",
            "sentiment_score": 0,
            "tags": [],
            "affected_tickers": [],
            "affected_sectors": [],
            "published_at": _now_iso(2),
            "title": "neutral",
        }
        score, _, _ = _score_single_macro(row, "005930", "")
        self.assertEqual(score, 0)


class TestCalculateMacroSignalScore(unittest.TestCase):
    """전체 흐름 테스트 (Supabase mock)."""

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_no_client(self, mock_client):
        """[1] Supabase 미연결 → score=0"""
        mock_client.return_value = None
        result = calculate_macro_signal_score("005930", sector="반도체")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["n_signals"], 0)
        self.assertEqual(result["matched_by"], "none")

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_no_signals(self, mock_client):
        """[2] 시그널 없음 → score=0"""
        client = MagicMock()
        client.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = []
        mock_client.return_value = client
        result = calculate_macro_signal_score("005930", sector="반도체")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["n_signals"], 0)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_mixed_signals_clamp(self, mock_client):
        """[6] 호재+악재 혼재 → 자연 상쇄 + 클램프 (-3 ~ +3)"""
        client = MagicMock()
        client.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {
                "severity": "CRITICAL",
                "sentiment_score": 50,
                "tags": ["AI수혜"],
                "affected_tickers": ["005930"],
                "affected_sectors": [],
                "published_at": _now_iso(2),
                "title": "호재",
            },
            {
                "severity": "WARNING",
                "sentiment_score": -40,
                "tags": ["금리인상"],
                "affected_tickers": [],
                "affected_sectors": ["반도체"],
                "published_at": _now_iso(2),
                "title": "악재",
            },
        ]
        mock_client.return_value = client
        result = calculate_macro_signal_score("005930", sector="반도체")
        # +3(ticker호재) + -1(sector악재) = +2, 클램프 안 걸림
        self.assertEqual(result["score"], 2)
        self.assertEqual(result["n_signals"], 2)
        self.assertEqual(result["matched_by"], "ticker")  # 우선순위
        self.assertEqual(result["n_effective"], 2)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_exception_safe(self, mock_client):
        """예외 발생 시 빈 리스트로 폴백"""
        mock_client.side_effect = Exception("DB down")
        result = calculate_macro_signal_score("005930", sector="반도체")
        self.assertEqual(result["score"], 0)


class TestRealtimeScoreIntegration(unittest.TestCase):
    """realtime_score.py에 매크로_X 시그널 추가 확인 (5/17 자기반성 적용)."""

    def test_import_chain(self):
        """[10] realtime_score → macro_signal_scorer import OK"""
        from src.use_cases.realtime_score import calculate_realtime_score
        self.assertTrue(callable(calculate_realtime_score))

    def test_breakdown_has_macro_x(self):
        """[10] breakdown 키에 매크로_X 포함 (Supabase mock)"""
        from src.use_cases import realtime_score as rs

        # 모든 외부 의존 mock
        broker = MagicMock()
        broker.fetch_price.return_value = {"output": {"stck_prpr": "70000"}}

        with patch("src.use_cases.realtime_score.calculate_entry_score") as mock_es, \
             patch("src.use_cases.realtime_score._fetch_intel_smart_money") as mock_sm, \
             patch("src.use_cases.realtime_score._fetch_intel_sniper") as mock_sn, \
             patch("src.use_cases.realtime_score._fetch_sector_momentum") as mock_sec, \
             patch("src.use_cases.realtime_score._score_macro_regime") as mock_mr, \
             patch("src.use_cases.realtime_score.calculate_ppa_score") as mock_ppa, \
             patch("src.use_cases.realtime_score.calculate_dart_score") as mock_dart, \
             patch("src.use_cases.realtime_score.calculate_macro_signal_score") as mock_mx, \
             patch("src.use_cases.realtime_score.calculate_edgar_score") as mock_edg:

            mock_es.return_value = {"score": 5, "reasoning": "es ok"}
            mock_sm.return_value = {"score": 2, "raw": 60, "reason": "sm"}
            mock_sn.return_value = {"score": 0, "raw": 30, "reason": "sn"}
            mock_sec.return_value = {"score": 1, "sector": "반도체", "reason": "sec"}
            mock_mr.return_value = {"score": 1, "regime": "NEUTRAL", "kospi_chg": 0.5, "reason": "mr"}
            mock_ppa.return_value = {"score": 0, "reason": "ppa none"}
            mock_dart.return_value = {"score": 0, "reason": "dart none"}
            mock_mx.return_value = {
                "score": 2,
                "n_signals": 3,
                "n_effective": 1,
                "matched_by": "sector",
                "top_signal": None,
                "reason": "매크로 호재 1건 (sector)",
                "breakdown": ["WARNING AI 호재 +2"],
            }
            mock_edg.return_value = {
                "score": 1,
                "n_signals": 2,
                "n_effective": 1,
                "matched_us": ["NVDA"],
                "breakdown": ["NVDA EARNING_BEAT × 1.0 = +1"],
                "reason": "EDGAR 1건 영향 +1",
            }

            result = rs.calculate_realtime_score(broker, "005930", current_price=70000)

            # 매크로_X / EDGAR breakdown 키 존재
            self.assertIn("매크로_X", result["breakdown"])
            self.assertIn("EDGAR_매핑", result["breakdown"])
            self.assertEqual(result["breakdown"]["매크로_X"]["score"], 2)
            self.assertEqual(result["breakdown"]["EDGAR_매핑"]["score"], 1)

            # 총점 = 5+2+0+1+1+0+0+2+1 = 12 → BUY
            self.assertEqual(result["total"], 12)
            self.assertEqual(result["recommend"], "BUY")

            # macro_x / edgar dict 반환에 포함
            self.assertIn("macro_x", result)
            self.assertIn("edgar", result)
            self.assertEqual(result["edgar"]["matched_us"], ["NVDA"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
