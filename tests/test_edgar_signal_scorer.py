"""test_edgar_signal_scorer.py — 옵션 D EDGAR 시그널 매핑 유닛 테스트 (2026-05-23).

배경 (5/23 옵션 D):
  EDGAR 미국 SEC 공시 (Form 4 임원 매수/매도, 10-Q 실적, 8-K 사건)
  → config/edgar_us_kr_mapping.yaml 사전 → 한국 매핑 종목 점수.

5/22 인사이트 (퐝가님):
  "AMD Lisa Su $55.7M 매도 같은 내부자 거래는 한국 메모리주에 영향"

검증 시나리오:
  1. 매핑 사전 로드 (NVDA → 005930/000660)
  2. 역인덱스 (000660 → [NVDA primary, AMD primary, ...])
  3. classify_insider_signal ($60M sell → INSIDER_SELL_MEGA)
  4. classify_insider_signal ($15M buy → INSIDER_BUY_LARGE)
  5. _freshness_weight (48h fresh / 168h half / 그 외 0)
  6. calculate_edgar_score 무매핑 종목 → 0
  7. calculate_edgar_score 매핑 있으나 시그널 없음 → 0
  8. NVDA INSIDER_BUY_MEGA + 000660 primary → +2 (round(2 × 1.0))
  9. AMD INSIDER_SELL_MEGA + 005930 secondary → -1 (round(-2 × 0.6) = -1)
  10. 여러 시그널 합산 + 클램프
  11. 예외 안전

실행:
  python -m pytest tests/test_edgar_signal_scorer.py -v
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.use_cases.edgar_signal_scorer import (
    _load_mapping,
    _build_reverse_index,
    _reverse_index_cached,
    get_us_mapping_for_kr,
    classify_insider_signal,
    _freshness_weight,
    calculate_edgar_score,
    EDGAR_BASE_SCORES,
    INSIDER_MEGA_USD,
    INSIDER_LARGE_USD,
)


def _now_iso(hours_ago: int = 0) -> str:
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat()


class TestMappingLoad(unittest.TestCase):
    def setUp(self):
        # lru_cache 비우기 (다른 테스트 영향 차단)
        _load_mapping.cache_clear()
        _reverse_index_cached.cache_clear()

    def test_mapping_loaded(self):
        mapping = _load_mapping()
        self.assertIn("NVDA", mapping)
        self.assertIn("TSLA", mapping)
        self.assertIn("AAPL", mapping)
        # 15사 확인
        self.assertGreaterEqual(len(mapping), 15)

    def test_nvda_primary_includes_hynix(self):
        mapping = _load_mapping()
        nvda = mapping["NVDA"]
        primary_tickers = [p["ticker"] for p in nvda["primary"]]
        self.assertIn("000660", primary_tickers)
        self.assertIn("005930", primary_tickers)


class TestReverseIndex(unittest.TestCase):
    def setUp(self):
        _load_mapping.cache_clear()
        _reverse_index_cached.cache_clear()

    def test_skh_has_multiple_us(self):
        """000660 SK하이닉스는 다수 미국 빅테크에 매핑됨"""
        mappings = get_us_mapping_for_kr("000660")
        us_tickers = {m["us_ticker"] for m in mappings}
        # NVDA, AMD, ASML, MU, AVGO 등 다수 포함
        for expected in ("NVDA", "AMD", "MU"):
            self.assertIn(expected, us_tickers, f"{expected} 매핑 누락")

    def test_lgenergy_tsla_primary(self):
        """373220 LG에너지솔루션 → TSLA primary (1.0)"""
        mappings = get_us_mapping_for_kr("373220")
        tsla = [m for m in mappings if m["us_ticker"] == "TSLA"]
        self.assertTrue(tsla)
        self.assertEqual(tsla[0]["tier"], "primary")
        self.assertEqual(tsla[0]["multiplier"], 1.0)

    def test_unmatched_kr_ticker(self):
        """매핑 없는 한국 ticker → 빈 리스트"""
        self.assertEqual(get_us_mapping_for_kr("999999"), [])


class TestInsiderSignalClassify(unittest.TestCase):
    def test_mega_sell(self):
        self.assertEqual(
            classify_insider_signal(60_000_000, "SELL"), "INSIDER_SELL_MEGA"
        )

    def test_mega_buy(self):
        self.assertEqual(
            classify_insider_signal(55_000_000, "BUY"), "INSIDER_BUY_MEGA"
        )

    def test_large_sell(self):
        self.assertEqual(
            classify_insider_signal(15_000_000, "S"), "INSIDER_SELL_LARGE"
        )

    def test_large_buy(self):
        self.assertEqual(
            classify_insider_signal(12_000_000, "P"), "INSIDER_BUY_LARGE"
        )

    def test_small_no_signal(self):
        self.assertEqual(classify_insider_signal(5_000_000, "SELL"), "")

    def test_threshold_boundary(self):
        self.assertEqual(
            classify_insider_signal(INSIDER_MEGA_USD, "SELL"), "INSIDER_SELL_MEGA"
        )
        self.assertEqual(
            classify_insider_signal(INSIDER_LARGE_USD, "BUY"), "INSIDER_BUY_LARGE"
        )


class TestFreshness(unittest.TestCase):
    def test_fresh(self):
        self.assertEqual(_freshness_weight(_now_iso(24)), 1.0)

    def test_stale(self):
        self.assertEqual(_freshness_weight(_now_iso(120)), 0.5)

    def test_expired(self):
        self.assertEqual(_freshness_weight(_now_iso(200)), 0.0)


class TestCalculateEdgarScore(unittest.TestCase):
    def setUp(self):
        _load_mapping.cache_clear()
        _reverse_index_cached.cache_clear()

    def test_no_mapping(self):
        result = calculate_edgar_score("999999")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["matched_us"], [])
        self.assertIn("매핑 없음", result["reason"])

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_mapping_but_no_signals(self, mock_client):
        client = MagicMock()
        # intelligence_edgar 빈 응답
        client.table.return_value.select.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = []
        mock_client.return_value = client

        result = calculate_edgar_score("000660")
        self.assertEqual(result["score"], 0)
        self.assertGreater(len(result["matched_us"]), 0)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_nvda_insider_buy_mega_to_skh(self, mock_client):
        """[8] NVDA INSIDER_BUY_MEGA → 000660 primary → +2"""
        client = MagicMock()
        client.table.return_value.select.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {
                "us_ticker": "NVDA",
                "signal_type": "INSIDER_BUY_MEGA",
                "filed_at": _now_iso(24),
            }
        ]
        mock_client.return_value = client

        result = calculate_edgar_score("000660")
        # +2 × 1.0(primary) × 1.0(fresh) = +2
        self.assertEqual(result["score"], 2)
        self.assertEqual(result["n_effective"], 1)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_amd_insider_sell_to_samsung_secondary(self, mock_client):
        """[9] AMD INSIDER_SELL_MEGA → 005930 secondary (0.6) → -1 (round)"""
        client = MagicMock()
        client.table.return_value.select.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {
                "us_ticker": "AMD",
                "signal_type": "INSIDER_SELL_MEGA",
                "filed_at": _now_iso(24),
            }
        ]
        mock_client.return_value = client

        result = calculate_edgar_score("005930")
        # -2 × 0.6 × 1.0 = -1.2 → round → -1
        self.assertEqual(result["score"], -1)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_amount_action_auto_classify(self, mock_client):
        """signal_type 없이 amount_usd + action 있으면 자동 분류"""
        client = MagicMock()
        client.table.return_value.select.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {
                "us_ticker": "NVDA",
                "amount_usd": 60_000_000,
                "action": "BUY",
                "filed_at": _now_iso(12),
            }
        ]
        mock_client.return_value = client

        result = calculate_edgar_score("000660")
        # INSIDER_BUY_MEGA +2 × 1.0 × 1.0 = +2
        self.assertEqual(result["score"], 2)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_multiple_signals_sum_clamp(self, mock_client):
        """[10] 여러 시그널 합산 + 클램프 (-3 ~ +3)"""
        client = MagicMock()
        client.table.return_value.select.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {"us_ticker": "NVDA", "signal_type": "INSIDER_BUY_MEGA", "filed_at": _now_iso(10)},
            {"us_ticker": "AMD",  "signal_type": "INSIDER_BUY_MEGA", "filed_at": _now_iso(20)},
            {"us_ticker": "MU",   "signal_type": "INSIDER_BUY_MEGA", "filed_at": _now_iso(30)},
        ]
        mock_client.return_value = client

        result = calculate_edgar_score("000660")
        # +2 × 1.0 × 1 + +2 × 1.0 × 1 + +2 × 1.0 × 1 = +6 → 클램프 +3
        self.assertEqual(result["score"], 3)

    @patch("src.adapters.quant_supabase_reader._get_client")
    def test_stale_signal_half_weight(self, mock_client):
        """48h~168h 시그널 → 가중치 절반"""
        client = MagicMock()
        client.table.return_value.select.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {"us_ticker": "NVDA", "signal_type": "GUIDANCE_UP", "filed_at": _now_iso(100)},
        ]
        mock_client.return_value = client

        result = calculate_edgar_score("000660")
        # +2 × 1.0 × 0.5 = +1
        self.assertEqual(result["score"], 1)

    def test_exception_safe(self):
        """예외 시 score=0"""
        with patch("src.adapters.quant_supabase_reader._get_client", side_effect=Exception("err")):
            result = calculate_edgar_score("000660")
            self.assertEqual(result["score"], 0)


class TestImportChain(unittest.TestCase):
    """import + main 흐름 검증 (5/17 자기반성 #1)."""

    def test_callable(self):
        self.assertTrue(callable(calculate_edgar_score))
        self.assertTrue(callable(classify_insider_signal))

    def test_base_scores_complete(self):
        for sig in (
            "INSIDER_SELL_LARGE", "INSIDER_SELL_MEGA",
            "INSIDER_BUY_LARGE", "INSIDER_BUY_MEGA",
            "EARNING_BEAT", "EARNING_MISS",
            "GUIDANCE_UP", "GUIDANCE_DOWN",
        ):
            self.assertIn(sig, EDGAR_BASE_SCORES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
