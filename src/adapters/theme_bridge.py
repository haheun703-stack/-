# -*- coding: utf-8 -*-
"""
Theme Bridge — 단타봇 shared/theme_service.py 크로스프로젝트 브릿지
=================================================================
단타봇이 관리하는 KIS 302개 테마 분류체계를 퀀트봇에서 사용할 수 있도록
크로스프로젝트 import를 처리한다.

Usage:
    from src.adapters.theme_bridge import get_themes, group_by_theme

    themes = get_themes("005930")          # ["3D낸드", "CXL", "HBM", ...]
    groups = group_by_theme(["005930", "000660"], min_count=2)
"""

import logging
import platform
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 단타봇 프로젝트 경로 (OS별) ──
if platform.system() == "Windows":
    _PROPHET_ROOT = Path("D:/Prophet_Agent_System_예언자")
else:
    _PROPHET_ROOT = Path("/home/ubuntu/bodyhunter")

# sys.path에 추가하여 shared.theme_service import 가능하게
_path_str = str(_PROPHET_ROOT)
if _path_str not in sys.path:
    sys.path.insert(0, _path_str)

# ── import 시도 ──
try:
    from shared.theme_service import (
        get_themes,
        get_theme_entries,
        get_codes,
        get_theme_size,
        is_noise_theme,
        group_by_theme,
        get_all_themes,
        get_raw_map,
        reload as theme_reload,
    )
    THEME_SERVICE_AVAILABLE = True
    logger.info("[ThemeBridge] shared.theme_service 로드 성공")
except ImportError:
    THEME_SERVICE_AVAILABLE = False
    logger.warning("[ThemeBridge] shared.theme_service 없음 — fallback 사용")

    def get_themes(code: str) -> list:
        return []

    def get_theme_entries(code: str) -> list:
        return []

    def get_codes(theme_name: str) -> list:
        return []

    def get_theme_size(theme_name: str) -> int:
        return 0

    def is_noise_theme(theme_name: str) -> bool:
        return False

    def group_by_theme(codes: list, min_count: int = 2, **kw) -> list:
        return []

    def get_all_themes() -> dict:
        return {}

    def get_raw_map() -> dict:
        return {}

    def theme_reload():
        pass
