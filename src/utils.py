"""공통 유틸리티 — JSON/YAML 로드, 프로젝트 경로 등."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json(path: str | Path, default: dict | list | None = None) -> dict | list | None:
    """JSON 파일 안전 로드.

    Args:
        path: 절대 경로 또는 DATA_DIR 상대 경로 (문자열)
        default: 파일이 없거나 파싱 실패 시 반환값
    """
    if default is None:
        default = {}

    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p

    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("JSON 로드 실패 %s: %s", p, e)
        return default


def save_json(path: str | Path, data: dict | list, indent: int = 2) -> None:
    """JSON 파일 안전 저장."""
    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
