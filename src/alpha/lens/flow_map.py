"""LENS 2: FLOW MAP — 섹터 자금흐름 → 종목 가중치 (STEP 7-3)

data/sector_rotation/investor_flow.json에서 섹터별 외인+기관 순매수를 읽어
Z-Score 기반으로 hot(부스트) / cold(패널티) 섹터를 판별한다.

hot  섹터 종목: position_ratio × 1.2~1.3
cold 섹터 종목: position_ratio × 0.7~0.8
"""

from __future__ import annotations

import logging
from statistics import mean, stdev

logger = logging.getLogger(__name__)


def compute(flow_data: dict, lens_cfg: dict) -> dict:
    """FLOW MAP 렌즈 계산.

    Args:
        flow_data: investor_flow.json 로드 결과
        lens_cfg: settings.yaml의 alpha_v2.lens 설정

    Returns:
        {
            "hot_sectors": [...],
            "cold_sectors": [...],
            "flow_direction": str,
            "sector_weight_adjustments": {sector: multiplier, ...}
        }
    """
    flow_cfg = lens_cfg.get("flow_map", {})
    hot_boost = flow_cfg.get("hot_boost", 1.2)
    cold_penalty = flow_cfg.get("cold_penalty", 0.8)
    top_n = flow_cfg.get("top_n", 3)

    sectors = flow_data.get("sectors", [])
    if not sectors:
        return _empty_result()

    # 섹터별 외인+기관 누적 순매수 합산
    scores = []
    for s in sectors:
        if s.get("category") != "sector":
            continue
        name = s.get("sector", "")
        f_cum = s.get("foreign_cum_bil", 0) or 0
        i_cum = s.get("inst_cum_bil", 0) or 0
        total = f_cum + i_cum
        scores.append({"sector": name, "total_flow": total})

    if len(scores) < 3:
        return _empty_result()

    # Z-Score 계산
    flows = [s["total_flow"] for s in scores]
    mu = mean(flows)
    sd = stdev(flows) if len(flows) > 1 else 1.0
    if sd < 0.01:
        sd = 1.0

    for s in scores:
        s["z"] = (s["total_flow"] - mu) / sd

    # 정렬: Z 내림차순
    scores.sort(key=lambda x: x["z"], reverse=True)

    hot = [s["sector"] for s in scores[:top_n]]
    cold = [s["sector"] for s in scores[-top_n:]]

    # 가중치 생성
    adjustments = {}
    for s in scores[:top_n]:
        adjustments[s["sector"]] = round(hot_boost, 2)
    for s in scores[-top_n:]:
        adjustments[s["sector"]] = round(cold_penalty, 2)

    # 자금 방향 판정
    hot_avg = mean([s["z"] for s in scores[:top_n]])
    cold_avg = mean([s["z"] for s in scores[-top_n:]])
    if hot_avg > 1.0:
        direction = "강한 섹터 쏠림"
    elif cold_avg < -1.0:
        direction = "방어적 자금 이동"
    else:
        direction = "균형 분산"

    return {
        "hot_sectors": hot,
        "cold_sectors": cold,
        "flow_direction": direction,
        "sector_weight_adjustments": adjustments,
    }


def _empty_result() -> dict:
    return {
        "hot_sectors": [],
        "cold_sectors": [],
        "flow_direction": "데이터 부족",
        "sector_weight_adjustments": {},
    }
