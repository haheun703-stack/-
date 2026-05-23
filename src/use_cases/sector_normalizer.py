"""Sector Alias 정규화 — 정보봇/KRX/WICS 분류 → 우리 표준 sector 통일.

배경 (5/22 HPSP 사고, 5/23 sector_momentum 보강):
  정보봇 dashboard_smart_money/sniper의 sector는 KRX/WICS 분류
  (예: "반도체와반도체장비") 사용. 우리 sector_fire는 한글 합성어
  ("AI반도체") 사용. 문자열 직접 비교 시 매칭 실패 → fire_score 0으로
  처리되어 시그널 손실.

  본 모듈로 alias → canonical 정규화 후 매칭하여 5/22 사고 재발 방지.

사용:
  from src.use_cases.sector_normalizer import normalize_sector
  normalize_sector("반도체와반도체장비")  # → "AI반도체"
  normalize_sector("이차전지")           # → "2차전지"
  normalize_sector("AI반도체")          # → "AI반도체" (이미 canonical)
  normalize_sector("미등록섹터")         # → "미등록섹터" (원본 반환, 보수적)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALIAS_FILE = PROJECT_ROOT / "config" / "sector_alias.yaml"


@lru_cache(maxsize=1)
def _load_alias_map() -> dict[str, str]:
    """alias → canonical 역인덱스 (대소문자/공백 무시 정규화).

    Returns:
        {정규화된 alias: canonical_sector}
        예: {"반도체": "AI반도체", "반도체와반도체장비": "AI반도체", ...}
    """
    try:
        with open(ALIAS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        canonical_block = data.get("canonical_sectors", {}) or {}
        out: dict[str, str] = {}
        for canonical, info in canonical_block.items():
            # canonical 자체도 매핑 (스스로 → 스스로)
            out[_clean_key(canonical)] = canonical
            aliases = info.get("aliases", []) or []
            for alias in aliases:
                key = _clean_key(alias)
                if key:
                    out[key] = canonical
        return out
    except Exception as e:
        logger.warning("sector_alias.yaml 로드 실패: %s", e)
        return {}


def _clean_key(raw: str) -> str:
    """대소문자 무시 + 공백/특수문자 제거."""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    # 공백/하이픈/언더스코어 제거 (예: "반도체 장비" / "ai_semiconductor")
    for ch in (" ", "-", "_", "/"):
        s = s.replace(ch, "")
    return s


def normalize_sector(raw: Optional[str]) -> str:
    """sector 문자열을 우리 표준 canonical 섹터로 정규화.

    정규화 실패 시 원본 그대로 반환 (보수적 — 정보봇 미등록 새 sector
    잘못 매핑 방지).

    Args:
        raw: 입력 sector 문자열

    Returns:
        canonical sector or 원본 (대소문자/공백은 trim)
    """
    if not raw:
        return ""
    alias_map = _load_alias_map()
    key = _clean_key(raw)
    if key in alias_map:
        return alias_map[key]
    # 부분 포함 매칭 (예: "반도체장비주", "조선해운주식" 같은 변형)
    for alias_key, canonical in alias_map.items():
        if alias_key and (alias_key in key or key in alias_key) and len(alias_key) >= 3:
            return canonical
    return str(raw).strip()


def list_canonical_sectors() -> list[str]:
    """등록된 모든 canonical 섹터 리스트."""
    return sorted({v for v in _load_alias_map().values()})


def is_canonical(sector: str) -> bool:
    """입력 문자열이 우리 표준 canonical 섹터인지 검사."""
    if not sector:
        return False
    return sector.strip() in list_canonical_sectors()
