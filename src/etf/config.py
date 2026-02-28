"""
ETF 3축 통합 전략 — 설정 + 유니버스
====================================
축1: 섹터 ETF (저격수) — 모멘텀 로테이션
축2: 레버리지/인버스 (포병) — 방향성 베팅
축3: 지수 ETF (보급부대) — 패시브 배분

settings.yaml의 etf_rotation 섹션에서 파라미터 로드.
"""

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
ETF_UNIVERSE_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "etf_universe.json"

# ============================================================
# 섹터 ETF 통합 유니버스 (TIGER + KODEX)
# ============================================================
# etf_universe.json의 TIGER ETF에 KODEX 추가분을 합산
# 동일 섹터에 TIGER+KODEX 둘 다 포함 → engine에서 composite 최고점 선택

# KODEX 추가 유니버스 (참조 코드 기반)
_KODEX_ADDITIONS = {
    "091160": {"name": "KODEX 반도체", "sector": "반도체"},
    "091170": {"name": "KODEX 은행", "sector": "은행"},
    "091180": {"name": "KODEX 자동차", "sector": "자동차"},
    "117700": {"name": "KODEX 건설", "sector": "건설"},
    "117460": {"name": "KODEX 에너지화학", "sector": "에너지화학"},
    "140700": {"name": "KODEX 보험", "sector": "보험"},
    "140710": {"name": "KODEX 운송", "sector": "운송"},
    "305720": {"name": "KODEX 2차전지산업", "sector": "2차전지"},
    "364970": {"name": "KODEX 조선", "sector": "조선"},
    "244580": {"name": "KODEX 바이오", "sector": "바이오"},
    "266360": {"name": "KODEX 핀테크", "sector": "핀테크"},
}

# 레버리지/인버스 ETF (지수 기본값)
LEVERAGE_ETF = {
    "BULL": {"code": "122630", "name": "KODEX 레버리지", "multiplier": 2.0},
    "BEAR": {"code": "114800", "name": "KODEX 인버스", "multiplier": -1.0},
    "CRISIS": {"code": "252670", "name": "KODEX 200선물인버스2X", "multiplier": -2.0},
}

# 섹터 레버리지 ETF — 유동성 기준 통과 종목만
# AUM 500억+ & 일거래대금 충분한 종목만 등록
# "포병이 저격수로 변한다" — 섹터 확신 시 정밀 타격
SECTOR_LEVERAGE_ETF = {
    "반도체": {"code": "488080", "name": "TIGER 반도체TOP10레버리지", "multiplier": 2.0},
    # 유동성 미달 (AUM/거래대금 부족) — 관찰 후 추가
    # "2차전지": {"code": "466940", "name": "KODEX 2차전지레버리지", "multiplier": 2.0},
    # "방산":    {"code": "472170", "name": "PLUS K방산레버리지", "multiplier": 2.0},
}

# 지수 ETF
INDEX_ETF = {
    "primary": {"code": "069500", "name": "KODEX 200", "weight": 0.7},
    "secondary": {"code": "278530", "name": "KODEX MSCI Korea TR", "weight": 0.3},
}


def load_settings() -> dict:
    """settings.yaml에서 etf_rotation 섹션 로드."""
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("etf_rotation", {})


def get_allocation(regime: str, settings: dict = None) -> dict:
    """현재 레짐에 따른 비중 배분 반환."""
    if settings is None:
        settings = load_settings()
    regime = regime.upper()
    alloc = settings.get("regime_allocation", {}).get(regime)
    if not alloc:
        raise ValueError(f"Unknown regime: {regime}")
    return alloc


def build_sector_universe() -> dict:
    """TIGER(etf_universe.json) + KODEX 통합 유니버스 구축."""
    universe = {}

    # 1) TIGER ETF (기존 etf_universe.json)
    if ETF_UNIVERSE_PATH.exists():
        with open(ETF_UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tiger_data = json.load(f)
        for sector_name, info in tiger_data.items():
            code = info["etf_code"]
            universe[code] = {
                "name": info["etf_name"],
                "sector": sector_name,
            }

    # 2) KODEX 추가 (중복 코드는 덮어쓰지 않음)
    for code, info in _KODEX_ADDITIONS.items():
        if code not in universe:
            universe[code] = info

    return universe


def get_leverage_etf(regime: str) -> dict | None:
    """레짐에 따른 레버리지/인버스 ETF 반환."""
    regime = regime.upper()
    return LEVERAGE_ETF.get(regime)
