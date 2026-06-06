"""FLOWX Market OS v1 engine policy map (2단계).

regime_router_v1의 route document를 읽어, 오늘 어떤 엔진이 어떤 모드로 허용되는지
단일 정책표로 만든다. 실주문/스케줄러/SAJANG 변경 없음. read-only 정책 산출물.

설계: docs/02-design/flowx_market_os_v1.md §1·§3, 진행 지시서 2단계 정책표.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.use_cases.regime_router_v1 import (
    HYSTERESIS_DAYS,
    PAPER_OPEN_DEFAULT,
    REGIME_R1,
    REGIME_R4,
    SELL_AUTOMATION_STATUS,
    run_router,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_DIR = PROJECT_ROOT / "data_store" / "policies"

POLICY_VERSION = "engine_policy_map_v1"

# 시장 종합 국면 (ticker별 C60을 보수적으로 통합)
MARKET_R4 = "R4_NORMAL_BULL"
MARKET_R1 = "R1_BEAR_RISK"
MARKET_DATA_UNAVAILABLE = "DATA_UNAVAILABLE"

# 엔진 모드 라벨 (설계도 §2)
MODE_PAPER_ONLY = "PAPER_ONLY"
MODE_ALLOWED_SHADOW = "ALLOWED_SHADOW"
MODE_SHADOW_ONLY = "SHADOW_ONLY"
MODE_BLOCKED = "BLOCKED"

# 국면별 엔진 정책표 (지시서 2단계).
# 가설C=R4 종목선별 / SmartEntry=장중 타이밍 / B·C로테이션·Event/DART=미검증 shadow / 매도=항상 BLOCKED
ENGINE_POLICY: dict[str, dict[str, str]] = {
    MARKET_R4: {
        "hypothesis_c": MODE_PAPER_ONLY,
        "smart_entry": MODE_ALLOWED_SHADOW,
        "bc_rotation": MODE_SHADOW_ONLY,
        "event_dart": MODE_SHADOW_ONLY,
        "sell": MODE_BLOCKED,
    },
    MARKET_R1: {
        "hypothesis_c": MODE_BLOCKED,
        "smart_entry": MODE_SHADOW_ONLY,
        "bc_rotation": MODE_SHADOW_ONLY,
        "event_dart": MODE_SHADOW_ONLY,
        "sell": MODE_BLOCKED,
    },
    MARKET_DATA_UNAVAILABLE: {
        "hypothesis_c": MODE_BLOCKED,
        "smart_entry": MODE_SHADOW_ONLY,
        "bc_rotation": MODE_SHADOW_ONLY,
        "event_dart": MODE_SHADOW_ONLY,
        "sell": MODE_BLOCKED,
    },
}

MARKET_REGIME_RULE = (
    "보수적 통합(설계도 §10): 기초자산 중 하나라도 데이터없음/R1(BEAR)이면 신규진입 차단. "
    "전부 R4(BULL)일 때만 강세 정책. 하루짜리 C60 깜빡임은 라우터 히스테리시스가 흡수."
)


def resolve_market_regime(routes: dict[str, dict]) -> tuple[str, dict[str, dict]]:
    """ticker별 route를 보수적으로 종합한다.

    안전 우선: 하나라도 데이터없음 → DATA_UNAVAILABLE, 하나라도 R1 → R1,
    전부 R4일 때만 R4. (설계도 §0 안전장치 먼저 / §10 약세면 전부 shadow)
    히스테리시스는 라우터 effective_regime이 이미 흡수하므로 여기선 hard_gate_regime만 본다.
    """
    per_ticker: dict[str, dict] = {}
    any_unavailable = False
    any_bear = False
    bull_count = 0

    for ticker, route in routes.items():
        if not route.get("data_available"):
            any_unavailable = True
            per_ticker[ticker] = {
                "data_available": False,
                "hard_gate_regime": None,
                "effective_regime": None,
                "in_hysteresis_window": False,
            }
            continue
        regime = route.get("hard_gate_regime")
        per_ticker[ticker] = {
            "data_available": True,
            "hard_gate_regime": regime,
            "c60_regime_raw": route.get("c60_regime_raw"),
            "effective_regime": route.get("effective_regime"),
            "in_hysteresis_window": bool(route.get("in_hysteresis_window")),
        }
        if regime == REGIME_R1:
            any_bear = True
        elif regime == REGIME_R4:
            bull_count += 1

    if any_unavailable:
        market = MARKET_DATA_UNAVAILABLE
    elif any_bear:
        market = MARKET_R1
    elif bull_count > 0:
        market = MARKET_R4
    else:
        market = MARKET_DATA_UNAVAILABLE
    return market, per_ticker


def _collect_shadow_advisories(routes: dict[str, dict]) -> list[dict]:
    """R0/R5 등 SHADOW_LABEL 수집. 엔진 전환권 0 (설계도 §3)."""
    advisories: list[dict] = []
    for ticker, route in routes.items():
        for label in route.get("shadow_labels") or []:
            advisories.append({"ticker": ticker, **label})
    return advisories


def build_policy_document(route_document: dict) -> dict:
    routes = route_document.get("routes", {}) or {}
    market_regime, per_ticker = resolve_market_regime(routes)
    engines = dict(ENGINE_POLICY[market_regime])
    advisories = _collect_shadow_advisories(routes)
    pending = [t for t, v in per_ticker.items() if v.get("in_hysteresis_window")]

    return {
        "version": POLICY_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": route_document.get("as_of_date"),
        "market_regime": market_regime,
        "market_regime_rule": MARKET_REGIME_RULE,
        "hysteresis": {
            "applied": True,
            "days": HYSTERESIS_DAYS,
            "source": "regime_router_v1.effective_regime",
            "pending_tickers": pending,
        },
        "engines": engines,
        "per_ticker": per_ticker,
        "shadow_advisories": advisories,
        "shadow_advisory_authority": False,
        "paper_open_allowed": PAPER_OPEN_DEFAULT,
        "sell_automation": SELL_AUTOMATION_STATUS,
        "safety": {
            "real_order": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "auto_promotion": False,
            "unverified_labels_can_switch_engine": False,
            "paper_open_default": PAPER_OPEN_DEFAULT,
        },
    }


def save_policy_document(document: dict, output_dir: Path = POLICY_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of_date = str(document.get("as_of_date") or datetime.now().strftime("%Y-%m-%d"))
    path = output_dir / f"policy_{as_of_date}.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_policy_map(
    days: int = 1300, prefer_remote: bool = True, write: bool = True
) -> tuple[dict, Path | None]:
    route_document, _ = run_router(days=days, prefer_remote=prefer_remote, write=write)
    policy = build_policy_document(route_document)
    path = save_policy_document(policy) if write else None
    return policy, path
