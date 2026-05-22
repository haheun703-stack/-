"""PPA 화이트리스트 69종 자동 가중치 + 우리 jgis OHLCV 교차 검증.

배경 (5/22 퐝가님 지시):
  정보봇 100% 신뢰 X — 우리 시스템 (jgis OHLCV)으로 실제 강세 검증.
  정보봇 화이트리스트 = 종목 후보 / 우리 검증 = 진짜 가중치 결정.

화이트리스트 (config/ppa_whitelist.yaml, 정보봇 자료 기반):
  - PPA 35종 (1차): 발전사/데이터센터/전력기기/건설EPC/냉각공조/메모리
  - 2차 37종: 반도체후공정/전력기기확장/AI로봇/방산조선
  - 총 69종 (중복 제외)

카테고리별 weight (정보봇 정의):
  발전사 25 / 데이터센터 25 / power_grid_extended 22 / 전력기기 20
  semiconductor_backend 18 / ai_robotics 15 / 건설 15 / defense_shipbuilding 12
  냉각공조 10 / 메모리 5

교차 검증 (jgis OHLCV — 정보봇 vs 우리):
  3 시그널 평가 (각 통과 시 + 가중):
    1. MA20 위 (close > MA20) — 추세 강세
    2. RSI < 70 (과매수 X) — 안전 진입
    3. 외인+기관 5일 누적 > 0 (수급 유지)

최종 가중치 부여:
  3/3 통과: weight × 1.0 (max 25점 → 우리 시스템 검증 통과)
  2/3:      weight × 0.5
  1/3:      weight × 0.2
  0/3:      weight × 0 (정보봇만 추적, 우리 검증 실패 — 신뢰 X)

  realtime_score 9번째 시그널 추가:
  실제 점수 = min(weight × 매칭율 / 25, 1) × 3 → 0~3점

사용:
  from src.use_cases.ppa_whitelist_scorer import calculate_ppa_score
  result = calculate_ppa_score("403870")  # HPSP
  # → {score: +2, weight: 18, validation: "3/3", reason: "..."}
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WHITELIST_YAML = PROJECT_ROOT / "config" / "ppa_whitelist.yaml"
JGIS_OHLCV_DIR = Path("/home/ubuntu/jgis/stock_data_daily")

# 캐시
_whitelist_cache: dict | None = None


def _load_whitelist() -> dict[str, dict]:
    """yaml에서 화이트리스트 로드 (1회 캐시)."""
    global _whitelist_cache
    if _whitelist_cache is not None:
        return _whitelist_cache
    try:
        with open(WHITELIST_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _whitelist_cache = data.get("whitelist", {})
        return _whitelist_cache
    except Exception as e:
        logger.warning("ppa_whitelist.yaml 로드 실패: %s", e)
        _whitelist_cache = {}
        return _whitelist_cache


def is_whitelist_ticker(ticker: str) -> bool:
    """화이트리스트 종목 여부."""
    return str(ticker).zfill(6) in _load_whitelist()


def get_whitelist_info(ticker: str) -> dict[str, Any] | None:
    """화이트리스트 종목 정보 (name, category, weight, source)."""
    return _load_whitelist().get(str(ticker).zfill(6))


def _validate_via_jgis(ticker: str) -> dict[str, Any]:
    """jgis OHLCV로 실제 강세 검증 (우리 시스템 교차 확인).

    Returns:
        {
            "pass_count": int,                # 0~3
            "ma20_ok": bool,                   # close > MA20
            "rsi_ok": bool,                    # RSI < 70 (과매수 X)
            "supply_ok": bool,                  # 외인+기관 5일 매수 유지
            "data_available": bool,
            "details": dict,
        }
    """
    matches = list(JGIS_OHLCV_DIR.glob(f"*_{str(ticker).zfill(6)}.csv"))
    if not matches:
        return {"pass_count": 0, "ma20_ok": False, "rsi_ok": False,
                "supply_ok": False, "data_available": False,
                "details": {"reason": "jgis 파일 없음"}}

    try:
        with open(matches[0], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {"pass_count": 0, "ma20_ok": False, "rsi_ok": False,
                    "supply_ok": False, "data_available": False,
                    "details": {"reason": "OHLCV 빈 파일"}}

        latest = rows[-1]
        close = float(latest.get("Close", 0) or 0)
        ma20 = float(latest.get("MA20", 0) or 0)
        rsi = float(latest.get("RSI", 0) or 0)

        # 외인+기관 5일 누적
        recent_5 = rows[-5:] if len(rows) >= 5 else rows
        fg_5d = sum(float(r.get("Foreign_Net", 0) or 0) for r in recent_5)
        inst_5d = sum(float(r.get("Inst_Net", 0) or 0) for r in recent_5)

        ma20_ok = close > 0 and ma20 > 0 and close > ma20
        rsi_ok = 0 < rsi < 70
        supply_ok = (fg_5d + inst_5d) > 0

        pass_count = sum([ma20_ok, rsi_ok, supply_ok])

        return {
            "pass_count": pass_count,
            "ma20_ok": ma20_ok,
            "rsi_ok": rsi_ok,
            "supply_ok": supply_ok,
            "data_available": True,
            "details": {
                "close": close,
                "ma20": ma20,
                "rsi": round(rsi, 1),
                "foreign_5d": fg_5d,
                "inst_5d": inst_5d,
                "ma20_vs_close": round((close - ma20) / ma20 * 100, 2) if ma20 > 0 else 0,
            },
        }
    except Exception as e:
        logger.warning("jgis 검증 실패 %s: %s", ticker, e)
        return {"pass_count": 0, "ma20_ok": False, "rsi_ok": False,
                "supply_ok": False, "data_available": False,
                "details": {"error": str(e)}}


def calculate_ppa_score(ticker: str) -> dict[str, Any]:
    """PPA 화이트리스트 점수 + jgis 교차 검증.

    Returns:
        {
            "score": int,                       # 0 ~ +3 (realtime_score 통합용)
            "in_whitelist": bool,
            "whitelist_info": dict | None,
            "weight_base": int,                  # 카테고리 기본 weight
            "validation": dict,                  # jgis 검증 결과 (3 시그널)
            "weight_adjusted": float,             # 검증 후 실제 weight
            "reason": str,
        }
    """
    info = get_whitelist_info(ticker)
    if not info:
        return {
            "score": 0,
            "in_whitelist": False,
            "whitelist_info": None,
            "weight_base": 0,
            "validation": None,
            "weight_adjusted": 0,
            "reason": "PPA 화이트리스트 미포함",
        }

    weight_base = info.get("weight", 0)
    validation = _validate_via_jgis(ticker)

    if not validation["data_available"]:
        return {
            "score": 0,
            "in_whitelist": True,
            "whitelist_info": info,
            "weight_base": weight_base,
            "validation": validation,
            "weight_adjusted": 0,
            "reason": f"PPA 종목이지만 jgis 데이터 부재 — 검증 불가",
        }

    pass_count = validation["pass_count"]
    # 매칭율: 3/3 → 1.0, 2/3 → 0.5, 1/3 → 0.2, 0/3 → 0
    match_ratio = {3: 1.0, 2: 0.5, 1: 0.2, 0: 0.0}.get(pass_count, 0)

    weight_adjusted = weight_base * match_ratio
    # realtime_score 통합 점수: max 25점 → 0~3점 스케일
    score = min(int(weight_adjusted / 25 * 3), 3)

    cat = info.get("category", "unknown")
    name = info.get("name", ticker)

    reason = (
        f"PPA {cat} (weight {weight_base}) × 검증 {pass_count}/3 "
        f"(MA20위={validation['ma20_ok']}, RSI<70={validation['rsi_ok']}, "
        f"수급+={validation['supply_ok']}) → +{score}점"
    )

    return {
        "score": score,
        "in_whitelist": True,
        "whitelist_info": info,
        "weight_base": weight_base,
        "validation": validation,
        "weight_adjusted": round(weight_adjusted, 1),
        "reason": reason,
    }
