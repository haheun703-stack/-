"""FLOWX Market OS v1 regime router.

regime_monitor의 C60 보고서를 정책 JSON으로 번역한다.
실주문/스케줄러/SAJANG 변경은 없고, 다음 단계의 morning plan이 읽을 read-only 산출물만 만든다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.etf.regime_monitor import REGIME_BEAR, REGIME_BULL, run_all

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGIME_DIR = PROJECT_ROOT / "data_store" / "regime"

ROUTER_VERSION = "regime_router_v1"
HARD_GATE_SOURCE = "src.etf.regime_monitor:C60"
SELL_AUTOMATION_STATUS = "BLOCKED"
PAPER_OPEN_DEFAULT = False
OVERHEAT_DISTANCE_PCT = 50.0
HYSTERESIS_DAYS = 2  # 설계도 §3: C60 1일 휩쏘(깜빡임) 방지. 전환은 2거래일 연속 확인 후 정책 발동.

REGIME_R1 = "R1_BEAR_RISK"
REGIME_R4 = "R4_NORMAL_BULL"
REGIME_R0 = "R0_RISK_EVENT"
REGIME_R5 = "R5_OVERHEAT"


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _distance_pct(close: Any, ma60: Any) -> float | None:
    close_f = _round(close, 6)
    ma60_f = _round(ma60, 6)
    if close_f is None or ma60_f is None or ma60_f <= 0:
        return None
    return round((close_f / ma60_f - 1) * 100, 2)


def _effective_regime(current_regime: Any, days_in_regime: Any) -> tuple[Any, bool]:
    """히스테리시스 적용 국면. 설계도 §3.

    raw C60(monitor)은 매일 close vs MA60로 즉시 판정하므로 1일 급락 V자 휩쏘에
    BULL<->BEAR가 깜빡일 수 있다(2026-06-02 C60 휩쏘 리스크 실증). 정책 레이어는
    전환이 HYSTERESIS_DAYS(2거래일) 연속 확인될 때까지 직전 국면을 유지한다.
    monitor raw는 손대지 않는다(백테스트/lead-lag 복기 보존).

    반환: (effective_regime, confirmed)
      - confirmed=True : 현재 국면이 2거래일 이상 지속(확정)
      - confirmed=False: 전환 첫날(미확정) → 직전 국면 유지
    """
    if current_regime is None or days_in_regime is None:
        return current_regime, False
    try:
        days = int(days_in_regime)
    except (TypeError, ValueError):
        return current_regime, False
    if days >= HYSTERESIS_DAYS:
        return current_regime, True
    # 전환 첫날(days_in_regime==1): 직전 국면 = 현재의 반대값
    prev = REGIME_BULL if current_regime == REGIME_BEAR else REGIME_BEAR
    return prev, False


def _shadow_labels(report: dict) -> list[dict]:
    """미검증 국면 라벨. 엔진 권한 0, 설명/복기용."""
    labels: list[dict] = []
    obs = report.get("current_observations") or {}
    distance = _distance_pct(report.get("current_close"), report.get("current_ma60"))

    risk_reasons = []
    if obs.get("vol_cluster_warn"):
        risk_reasons.append("vol_cluster_warn")
    if obs.get("kospi_warn"):
        risk_reasons.append("kospi_warn")
    if risk_reasons:
        labels.append({
            "regime": REGIME_R0,
            "status": "SHADOW_LABEL",
            "reasons": risk_reasons,
            "engine_switch_authority": False,
        })

    if report.get("current_regime") == REGIME_BULL and distance is not None and distance >= OVERHEAT_DISTANCE_PCT:
        labels.append({
            "regime": REGIME_R5,
            "status": "SHADOW_LABEL",
            "close_vs_ma60_pct": distance,
            "engine_switch_authority": False,
        })

    return labels


def route_from_report(ticker: str, report: dict) -> dict:
    """단일 기초자산 C60 보고서를 Market OS route로 변환."""
    if report.get("rows", 0) == 0:
        return {
            "ticker": ticker,
            "data_available": False,
            "hard_gate_regime": None,
            "hard_gate_status": "DATA_UNAVAILABLE",
            "allow_new_entries": False,
            "allow_hypothesis_c": False,
            "smart_entry_observation": "SHADOW_ONLY",
            "paper_open_allowed": False,
            "sell_automation": SELL_AUTOMATION_STATUS,
            "reason": report.get("error", "no data"),
        }

    current = report.get("current_regime")
    days_in_regime = report.get("days_in_current_regime")
    effective, regime_confirmed = _effective_regime(current, days_in_regime)
    is_bull = effective == REGIME_BULL
    hard_regime = REGIME_R4 if is_bull else REGIME_R1

    return {
        "ticker": ticker,
        "name": report.get("name", ticker),
        "as_of_date": report.get("last_date"),
        "data_available": True,
        "hard_gate_source": HARD_GATE_SOURCE,
        "c60_regime_raw": current,
        "effective_regime": effective,
        "regime_confirmed": regime_confirmed,
        "in_hysteresis_window": not regime_confirmed,
        "hysteresis_days": HYSTERESIS_DAYS,
        "c60_regime": current,
        "hard_gate_regime": hard_regime,
        "hard_gate_status": "HARD_GATE",
        "current_close": report.get("current_close"),
        "current_ma60": report.get("current_ma60"),
        "close_vs_ma60_pct": _distance_pct(report.get("current_close"), report.get("current_ma60")),
        "days_in_current_regime": days_in_regime,
        "allow_new_entries": is_bull,
        "allow_hypothesis_c": is_bull,
        "smart_entry_observation": "ALLOWED_SHADOW" if is_bull else "SHADOW_ONLY",
        "paper_open_allowed": PAPER_OPEN_DEFAULT,
        "sell_automation": SELL_AUTOMATION_STATUS,
        "shadow_labels": _shadow_labels(report),
        "observation_gate_status": report.get("observation_gate_status", {}),
        "safety": {
            "real_order": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "auto_promotion": False,
            "unverified_labels_can_switch_engine": False,
        },
    }


def build_route_document(reports: dict[str, dict]) -> dict:
    routes = {ticker: route_from_report(ticker, report) for ticker, report in reports.items()}
    dates = [
        route.get("as_of_date")
        for route in routes.values()
        if route.get("data_available") and route.get("as_of_date")
    ]
    as_of_date = max(dates) if dates else datetime.now().strftime("%Y-%m-%d")

    return {
        "version": ROUTER_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "hard_gate_policy": "R1/R4 only. R0/R2/R3/R5 are shadow labels with zero engine authority.",
        "routes": routes,
        "global_safety": {
            "real_order": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "sell_automation": SELL_AUTOMATION_STATUS,
            "paper_open_default": PAPER_OPEN_DEFAULT,
        },
    }


def save_route_document(document: dict, output_dir: Path = REGIME_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of_date = str(document.get("as_of_date") or datetime.now().strftime("%Y-%m-%d"))
    path = output_dir / f"regime_{as_of_date}.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_router(days: int = 1300, prefer_remote: bool = True, write: bool = True) -> tuple[dict, Path | None]:
    reports = run_all(days=days, prefer_remote=prefer_remote)
    document = build_route_document(reports)
    path = save_route_document(document) if write else None
    return document, path
