"""LENS 2: FLOW MAP — 섹터 자금흐름 → 종목 가중치 (STEP 7-3, 10-3 확장)

data/sector_rotation/investor_flow.json에서 섹터별 외인+기관 순매수를 읽어
Z-Score 기반으로 hot(부스트) / cold(패널티) 섹터를 판별한다.

STEP 10 확장: 시나리오 엔진 활성 시 해당 섹터 가중치를 추가 조정.
  hot_sectors  → +0.2 부스트
  cold_sectors → -0.2 패널티

hot  섹터 종목: position_ratio × 1.2~1.3
cold 섹터 종목: position_ratio × 0.7~0.8
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean, stdev

logger = logging.getLogger(__name__)

_ACTIVE_SCENARIOS_PATH = Path("data/scenarios/active_scenarios.json")
_CHAINS_PATH = Path("data/scenarios/scenario_chains.json")


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

    # ── STEP 10: 시나리오 가중치 합산 ──
    scenario_info = _apply_scenario_adjustments(adjustments)

    # hot/cold 재평가 (시나리오로 인해 새 섹터가 추가될 수 있음)
    all_hot = set(hot)
    all_cold = set(cold)
    for sector, mult in adjustments.items():
        if mult >= hot_boost and sector not in all_hot:
            all_hot.add(sector)
        elif mult <= cold_penalty and sector not in all_cold:
            all_cold.add(sector)

    return {
        "hot_sectors": list(all_hot),
        "cold_sectors": list(all_cold),
        "flow_direction": direction,
        "sector_weight_adjustments": adjustments,
        "active_scenarios": scenario_info,
    }


def _apply_scenario_adjustments(adjustments: dict) -> list[dict]:
    """active_scenarios.json + scenario_chains.json을 읽어
    현재 Phase의 hot/cold 섹터를 adjustments에 반영한다.

    Returns:
        LENS 컨텍스트용 시나리오 정보 리스트
    """
    try:
        with open(_ACTIVE_SCENARIOS_PATH, "r", encoding="utf-8") as f:
            active_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    try:
        with open(_CHAINS_PATH, "r", encoding="utf-8") as f:
            chains_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    active = active_data.get("scenarios", {})
    if not active:
        return []

    scenario_map = {s["id"]: s for s in chains_data.get("scenarios", [])}
    scenario_boost = 0.2
    scenario_penalty = 0.2
    info = []

    for sid, state in active.items():
        scenario = scenario_map.get(sid)
        if not scenario:
            continue

        phase_idx = state.get("current_phase", 1) - 1
        chain = scenario.get("chain", [])
        if phase_idx >= len(chain):
            phase_idx = len(chain) - 1
        current = chain[phase_idx]
        next_p = chain[phase_idx + 1] if phase_idx + 1 < len(chain) else None

        # hot 섹터 부스트
        for sector in current.get("hot_sectors", []):
            base = adjustments.get(sector, 1.0)
            adjustments[sector] = round(base + scenario_boost, 2)

        # cold 섹터 패널티
        for sector in current.get("cold_sectors", []):
            base = adjustments.get(sector, 1.0)
            adjustments[sector] = round(base - scenario_penalty, 2)

        info.append({
            "id": sid,
            "phase": state.get("current_phase", 1),
            "phase_name": current.get("name", ""),
            "hot_now": current.get("hot_sectors", []),
            "cold_now": current.get("cold_sectors", []),
            "next_phase": next_p.get("name", "") if next_p else None,
            "next_hot": next_p.get("hot_sectors", []) if next_p else [],
            "days_active": state.get("days_active", 0),
        })

        logger.info("시나리오 %s P%d → HOT %s (+%.1f), COLD %s (-%.1f)",
                     scenario["name"], state.get("current_phase", 1),
                     current.get("hot_sectors", []), scenario_boost,
                     current.get("cold_sectors", []), scenario_penalty)

    return info


def _empty_result() -> dict:
    return {
        "hot_sectors": [],
        "cold_sectors": [],
        "flow_direction": "데이터 부족",
        "sector_weight_adjustments": {},
    }
