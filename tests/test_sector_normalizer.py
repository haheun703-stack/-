"""test_sector_normalizer.py — sector alias 정규화 유닛 테스트 (2026-05-23).

배경 (5/22 HPSP 사고):
  정보봇 dashboard_smart_money sector="반도체와반도체장비" →
  우리 sector_fire "AI반도체" 매칭 실패 → fire_score 0 시그널 손실.
  본 모듈로 정규화 후 매칭 → 사고 재발 방지.

5/17 자기반성 #1 적용: import + 함수 호출 + main 흐름 검증.

실행:
  python -m pytest tests/test_sector_normalizer.py -v
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.use_cases.sector_normalizer import (
    normalize_sector,
    list_canonical_sectors,
    is_canonical,
    _load_alias_map,
    _clean_key,
)


class TestNormalizeSector(unittest.TestCase):
    """핵심 정규화 케이스 (5/22 HPSP 사고 재발 방지)."""

    def setUp(self):
        _load_alias_map.cache_clear()

    def test_semiconductor_aliases(self):
        """반도체 계열 모두 AI반도체로 정규화"""
        for raw in (
            "반도체",
            "반도체장비",
            "반도체와반도체장비",
            "반도체 장비",
            "반도체및반도체장비",
            "메모리",
            "HBM",
            "semiconductor",
            "SEMICONDUCTOR",
            "Semiconductor",
        ):
            self.assertEqual(
                normalize_sector(raw), "AI반도체",
                f"'{raw}' should normalize to AI반도체"
            )

    def test_battery_aliases(self):
        for raw in ("이차전지", "배터리", "양극재", "battery"):
            self.assertEqual(normalize_sector(raw), "2차전지")

    def test_defense_aliases(self):
        for raw in ("방위산업", "우주항공", "defense"):
            self.assertEqual(normalize_sector(raw), "방산")

    def test_shipbuilding_aliases(self):
        for raw in ("조선", "해운", "shipbuilding"):
            self.assertEqual(normalize_sector(raw), "조선해운")

    def test_power_aliases(self):
        for raw in ("전력", "송배전", "변압기", "데이터센터전력"):
            self.assertEqual(normalize_sector(raw), "전력기기")

    def test_canonical_passthrough(self):
        """이미 canonical인 경우 그대로 반환"""
        for s in ("AI반도체", "2차전지", "방산", "전력기기"):
            self.assertEqual(normalize_sector(s), s)

    def test_unknown_passthrough(self):
        """등록되지 않은 sector → 원본 반환 (보수적)"""
        self.assertEqual(normalize_sector("미등록새섹터"), "미등록새섹터")

    def test_empty_and_none(self):
        self.assertEqual(normalize_sector(""), "")
        self.assertEqual(normalize_sector(None), "")

    def test_partial_match(self):
        """변형된 형태도 부분 매칭 (예: '반도체장비주', '조선해운주식')"""
        # alias_key "반도체장비"가 "반도체장비주" 안에 포함 → 매칭
        result = normalize_sector("반도체장비주")
        self.assertEqual(result, "AI반도체")

    def test_case_insensitive(self):
        self.assertEqual(normalize_sector("Defense"), "방산")
        self.assertEqual(normalize_sector("DEFENSE"), "방산")

    def test_whitespace_strip(self):
        self.assertEqual(normalize_sector("  반도체  "), "AI반도체")


class TestCleanKey(unittest.TestCase):
    def test_strip_lower(self):
        self.assertEqual(_clean_key("  Semiconductor  "), "semiconductor")

    def test_remove_separators(self):
        self.assertEqual(_clean_key("ai_semiconductor"), "aisemiconductor")
        self.assertEqual(_clean_key("반도체 장비"), "반도체장비")
        self.assertEqual(_clean_key("data-center"), "datacenter")

    def test_empty(self):
        self.assertEqual(_clean_key(""), "")
        self.assertEqual(_clean_key(None), "")


class TestCanonicalList(unittest.TestCase):
    def setUp(self):
        _load_alias_map.cache_clear()

    def test_has_core_sectors(self):
        sectors = list_canonical_sectors()
        # 핵심 섹터 모두 포함
        for expected in ("AI반도체", "2차전지", "방산", "조선해운", "전력기기"):
            self.assertIn(expected, sectors)

    def test_is_canonical(self):
        self.assertTrue(is_canonical("AI반도체"))
        self.assertTrue(is_canonical("방산"))
        self.assertFalse(is_canonical("반도체"))  # alias, canonical X
        self.assertFalse(is_canonical("미등록"))
        self.assertFalse(is_canonical(""))


class TestHpspScenario(unittest.TestCase):
    """5/22 HPSP 사고 재현 시나리오."""

    def test_5_22_hpsp_case(self):
        """
        5/22 HPSP 사례:
        - 정보봇 dashboard_smart_money sector="반도체" 또는 "반도체와반도체장비"
        - 우리 sector_fire에는 "AI반도체"로 등록
        - 정규화 후 매칭 → fire_score 정상 시그널 +2/+1
        """
        # 정보봇 측 raw sector
        sm_raw = "반도체와반도체장비"
        sniper_raw = "반도체"

        # 정규화
        sm_canonical = normalize_sector(sm_raw)
        sniper_canonical = normalize_sector(sniper_raw)

        # 모두 우리 표준 "AI반도체"로 매칭
        self.assertEqual(sm_canonical, "AI반도체")
        self.assertEqual(sniper_canonical, "AI반도체")
        self.assertEqual(sm_canonical, sniper_canonical)


if __name__ == "__main__":
    unittest.main(verbosity=2)
