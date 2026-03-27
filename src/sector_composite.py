"""TIER2 — 섹터 통합 점수 엔진 (Sector Composite Scorer)

모멘텀 + 기관수급 + 상대강도 + 기술지표 → 섹터 컴포짓 점수(0~100)
BRAIN etf_sector ARM + SignalEngine sector_boost 입력으로 사용.

입력:
  - data/sector_rotation/sector_momentum.json  (모멘텀 엔진)
  - data/institutional_flow/sector_institutional_flow.json (기관수급)

출력:
  - data/sector_rotation/sector_composite.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MOMENTUM_PATH = DATA_DIR / "sector_rotation" / "sector_momentum.json"
FLOW_PATH = DATA_DIR / "institutional_flow" / "sector_institutional_flow.json"
OUTPUT_PATH = DATA_DIR / "sector_rotation" / "sector_composite.json"

# 가중치 (합계 100%)
WEIGHTS = {
    "momentum": 0.35,
    "institutional": 0.35,
    "relative_strength": 0.15,
    "technical": 0.15,
}

# 레짐 경계
REGIME_THRESHOLDS = {
    "STRONG_ROTATION": 75,
    "MODERATE_ROTATION": 60,
    "NEUTRAL": 40,
    "WEAK_ROTATION": 25,
    # < 25 = EXODUS
}


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        logger.warning("파일 없음: %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_rel_strength(val: float, floor: float = -20.0, ceil: float = 20.0) -> float:
    """상대강도(-20~+20) → 0~100 정규화."""
    clamped = max(floor, min(ceil, val))
    return (clamped - floor) / (ceil - floor) * 100


def _normalize_rsi(rsi: float) -> float:
    """RSI(0~100) → 기술적 점수(0~100).
    RSI 50 부근이 가장 건강 → 50 중심 가우시안 변환.
    극단(30 이하, 70 이상)은 페널티.
    """
    if rsi <= 30:
        return max(0, rsi * 1.0)          # 과매도: 30→30, 20→20
    elif rsi >= 70:
        return max(0, (100 - rsi) * 1.0)  # 과매수: 70→30, 80→20
    else:
        # 30~70 구간: 50이 최고(100), 30/70이 최저(30)
        dist = abs(rsi - 50)
        return 100 - (dist / 20) * 70  # 50→100, 30/70→30


def classify_regime(score: float) -> str:
    """컴포짓 점수 → 섹터 레짐."""
    if score >= REGIME_THRESHOLDS["STRONG_ROTATION"]:
        return "STRONG_ROTATION"
    elif score >= REGIME_THRESHOLDS["MODERATE_ROTATION"]:
        return "MODERATE_ROTATION"
    elif score >= REGIME_THRESHOLDS["NEUTRAL"]:
        return "NEUTRAL"
    elif score >= REGIME_THRESHOLDS["WEAK_ROTATION"]:
        return "WEAK_ROTATION"
    return "EXODUS"


def compute_sector_composite() -> dict[str, Any]:
    """섹터 컴포짓 점수 산출.

    Returns:
        {
            "computed_at": ...,
            "sectors": [...],           # 점수 내림차순
            "regime_summary": {...},    # 레짐별 카운트
            "strong_sectors": [...],    # STRONG_ROTATION 섹터명
            "exodus_sectors": [...],    # EXODUS 섹터명
        }
    """
    # ── 데이터 로드 ──
    momentum_raw = _load_json(MOMENTUM_PATH)
    flow_raw = _load_json(FLOW_PATH)

    if not momentum_raw or not flow_raw:
        logger.error("모멘텀 또는 기관수급 데이터 없음")
        return {}

    # 모멘텀: list → dict by sector name
    momentum_by_sector = {}
    for s in momentum_raw.get("sectors", []):
        momentum_by_sector[s["sector"]] = s

    # 기관수급: dict by sector name
    flow_by_sector = flow_raw.get("sectors", {})

    # ── 매칭 및 점수 산출 ──
    results = []

    for sector_name, mom in momentum_by_sector.items():
        flow = flow_by_sector.get(sector_name)

        # 모멘텀 점수 (이미 0~100)
        momentum_score = mom.get("momentum_score", 50)

        # 기관수급 점수 (이미 0~100)
        if flow:
            inst_score = flow.get("aggregate", {}).get("weighted_score", 50)
        else:
            inst_score = 50  # 수급 데이터 없으면 중립

        # 상대강도 → 0~100
        rel_str = mom.get("rel_strength", 0)
        rel_score = _normalize_rel_strength(rel_str)

        # RSI → 기술적 점수
        rsi = mom.get("rsi_14", 50)
        tech_score = _normalize_rsi(rsi)

        # 컴포짓 점수
        composite = (
            WEIGHTS["momentum"] * momentum_score
            + WEIGHTS["institutional"] * inst_score
            + WEIGHTS["relative_strength"] * rel_score
            + WEIGHTS["technical"] * tech_score
        )
        composite = round(max(0, min(100, composite)), 1)

        regime = classify_regime(composite)

        results.append({
            "sector": sector_name,
            "etf_code": mom.get("etf_code", ""),
            "composite_score": composite,
            "regime": regime,
            "momentum_score": round(momentum_score, 1),
            "institutional_score": round(inst_score, 1),
            "relative_strength_score": round(rel_score, 1),
            "technical_score": round(tech_score, 1),
            "momentum_rank": mom.get("rank", 0),
            "ret_5": mom.get("ret_5", 0),
            "ret_20": mom.get("ret_20", 0),
            "inst_5d_억": flow.get("aggregate", {}).get("inst_5d_억", 0) if flow else 0,
            "foreign_5d_억": flow.get("aggregate", {}).get("foreign_5d_억", 0) if flow else 0,
            "has_flow_data": flow is not None,
        })

    # 정렬: 컴포짓 점수 내림차순
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # 레짐 요약
    regime_counts = {}
    strong_sectors = []
    exodus_sectors = []
    for r in results:
        reg = r["regime"]
        regime_counts[reg] = regime_counts.get(reg, 0) + 1
        if reg == "STRONG_ROTATION":
            strong_sectors.append(r["sector"])
        elif reg == "EXODUS":
            exodus_sectors.append(r["sector"])

    output = {
        "computed_at": datetime.now().isoformat(),
        "momentum_date": momentum_raw.get("date", ""),
        "sector_count": len(results),
        "sectors": results,
        "regime_summary": regime_counts,
        "strong_sectors": strong_sectors,
        "exodus_sectors": exodus_sectors,
        "avg_composite": round(sum(r["composite_score"] for r in results) / len(results), 1) if results else 50,
    }

    # ── 저장 ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    tmp.replace(OUTPUT_PATH)

    logger.info(
        "[SectorComposite] %d섹터 | 평균 %.1f | STRONG %d | EXODUS %d",
        len(results), output["avg_composite"],
        len(strong_sectors), len(exodus_sectors),
    )

    return output


# ── 섹터 부스트 승수 (SignalEngine 연동용) ──

SECTOR_BOOST_MAP = {
    "STRONG_ROTATION": 1.15,
    "MODERATE_ROTATION": 1.05,
    "NEUTRAL": 1.00,
    "WEAK_ROTATION": 0.90,
    "EXODUS": 0.75,
}

SECTOR_MAP_PATH = DATA_DIR / "sector_rotation" / "sector_map.json"


def get_ticker_sector_boost() -> dict[str, dict]:
    """종목코드 → {sector, regime, boost} 룩업 생성.

    sector_map.json의 종목코드 → sector_composite.json의 레짐 매칭.

    Returns:
        {"005930": {"sector": "반도체", "regime": "WEAK_ROTATION", "boost": 0.90}, ...}
    """
    # sector_composite 로드
    comp = _load_json(OUTPUT_PATH)
    if not comp or "sectors" not in comp:
        return {}

    # 섹터명 → 레짐/부스트 매핑
    regime_by_sector = {}
    for s in comp["sectors"]:
        regime_by_sector[s["sector"]] = {
            "regime": s["regime"],
            "boost": SECTOR_BOOST_MAP.get(s["regime"], 1.0),
            "composite_score": s["composite_score"],
        }

    # sector_map에서 종목 → 섹터 역매핑
    sector_map = _load_json(SECTOR_MAP_PATH)
    if not sector_map:
        return {}

    result = {}
    for sector_name, info in sector_map.items():
        regime_info = regime_by_sector.get(sector_name)
        if not regime_info:
            continue
        for stock in info.get("stocks", []):
            code = stock.get("code", "")
            if code and code not in result:
                result[code] = {
                    "sector": sector_name,
                    "regime": regime_info["regime"],
                    "boost": regime_info["boost"],
                    "composite_score": regime_info["composite_score"],
                }
    return result


# ── 단독 실행 ──
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = compute_sector_composite()
    if not result:
        print("데이터 부족으로 컴포짓 산출 실패")
        sys.exit(1)

    print("=" * 60)
    print("  TIER2 섹터 컴포짓 점수")
    print("=" * 60)

    for s in result["sectors"]:
        tag = {
            "STRONG_ROTATION": "[강]",
            "MODERATE_ROTATION": "[중]",
            "NEUTRAL": "[중립]",
            "WEAK_ROTATION": "[약]",
            "EXODUS": "[이탈]",
        }.get(s["regime"], "[?]")

        bar_len = max(0, min(20, int(s["composite_score"] / 5)))
        bar = "+" * bar_len + "-" * (20 - bar_len)
        flow_tag = "" if s["has_flow_data"] else " (수급X)"
        print(
            f"  {tag:>4s} {s['sector']:>8s} | [{bar}] {s['composite_score']:5.1f} | "
            f"M:{s['momentum_score']:4.0f} I:{s['institutional_score']:4.0f} "
            f"R:{s['relative_strength_score']:4.0f} T:{s['technical_score']:4.0f}"
            f"{flow_tag}"
        )

    print(f"\n레짐 분포: {result['regime_summary']}")
    if result["strong_sectors"]:
        print(f"STRONG: {', '.join(result['strong_sectors'])}")
    if result["exodus_sectors"]:
        print(f"EXODUS: {', '.join(result['exodus_sectors'])}")
