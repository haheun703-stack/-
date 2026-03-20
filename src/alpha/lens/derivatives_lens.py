"""LENS 5: DERIVATIVES — 파생 시장 심리 컨텍스트 (STEP 10)

derivatives_signal.json을 읽어 파생시장 관점의 맥락을 제공한다.

3개 축:
  1. 선물 베이시스 (콘탱고/백워데이션) → 기관 포지셔닝
  2. 풋/콜 프록시 (인버스 vs 레버리지) → 시장 심리
  3. 레버리지 자금 흐름 → 프로그램매매/수급 방향
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DERIVATIVES_PATH = Path("data") / "derivatives" / "derivatives_signal.json"


def compute() -> dict:
    """파생 렌즈 계산.

    Returns:
        {
            "available": bool,
            "composite_score": float,       # -100 ~ +100
            "composite_grade": str,         # STRONG_BULL ~ STRONG_BEAR
            "basis_status": str,            # CONTANGO/BACKWARDATION/FLAT
            "put_call_status": str,         # EXTREME_BEARISH ~ EXTREME_BULLISH
            "put_call_reversal": str,       # BEARISH_EXHAUSTION / BULLISH_EXHAUSTION / ""
            "flow_direction": str,          # 롱 우세 / 숏 우세 / 중립
            "program_signal": str,          # 차익매수 추정 / 차익매도 추정 / 없음
            "warning": str,                 # 경고 메시지 (있을 때만)
        }
    """
    try:
        with open(_DERIVATIVES_PATH, "r", encoding="utf-8") as f:
            sig = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("LENS-5 DERIVATIVES: 시그널 로드 실패 — %s", e)
        return _default()

    composite = sig.get("composite", {})
    basis = sig.get("futures_basis", {})
    pc = sig.get("put_call_proxy", {})
    flow = sig.get("leverage_flow", {})

    if not composite:
        return _default()

    # 자금 흐름 방향
    net_5d = flow.get("net_flow_5d_억", 0)
    if net_5d > 10000:
        flow_dir = "롱 강세"
    elif net_5d > 0:
        flow_dir = "롱 우세"
    elif net_5d > -10000:
        flow_dir = "숏 우세"
    else:
        flow_dir = "숏 강세"

    # 프로그램매매 시그널
    if flow.get("program_buy_est"):
        prog = "차익매수 추정"
    elif flow.get("program_sell_est"):
        prog = "차익매도 추정"
    else:
        prog = "없음"

    # 경고 생성
    warnings = []
    if pc.get("reversal") == "BEARISH_EXHAUSTION":
        warnings.append("공포 정점 — 반등 가능성")
    elif pc.get("reversal") == "BULLISH_EXHAUSTION":
        warnings.append("탐욕 정점 — 조정 가능성")

    if basis.get("status") == "BACKWARDATION" and composite.get("score", 0) < -20:
        warnings.append("백워데이션 + 약세 — 하방 압력 주의")

    if flow.get("inverse_vol_z", 0) > 2.0:
        warnings.append("인버스 거래량 급증 — 헤지 수요 확대")

    result = {
        "available": True,
        "composite_score": composite.get("score", 0),
        "composite_grade": composite.get("grade", "NEUTRAL"),
        "basis_status": basis.get("status", "FLAT"),
        "put_call_status": pc.get("status", "NEUTRAL"),
        "put_call_reversal": pc.get("reversal", ""),
        "flow_direction": flow_dir,
        "program_signal": prog,
    }

    if warnings:
        result["warning"] = " | ".join(warnings)

    return result


def _default() -> dict:
    return {
        "available": False,
        "composite_score": 0,
        "composite_grade": "NEUTRAL",
        "basis_status": "FLAT",
        "put_call_status": "NEUTRAL",
        "put_call_reversal": "",
        "flow_direction": "중립",
        "program_signal": "없음",
    }
