"""
릴레이 엔진 설정 로더
========================
config/relay_sectors.yaml에서 섹터별 릴레이 설정 로드.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "relay_sectors.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "relay"


def load_relay_config() -> dict:
    """relay_sectors.yaml 전체 로드."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("relay_engine", {})


def get_sectors(config: dict = None) -> dict:
    """섹터 정의 딕셔너리 반환."""
    if config is None:
        config = load_relay_config()
    return config.get("sectors", {})


def get_common_rules(config: dict = None) -> dict:
    """공통 매매 규칙 반환."""
    if config is None:
        config = load_relay_config()
    return config.get("common_rules", {})


def get_all_us_tickers(config: dict = None) -> list[str]:
    """모든 섹터의 US 대장주 티커 목록 (중복 제거)."""
    sectors = get_sectors(config)
    tickers = set()
    for sec in sectors.values():
        for leader in sec.get("us_leaders", []):
            tickers.add(leader["ticker"])
    return sorted(tickers)
