"""한국 장기 사이클/밸류 레짐 판정 레이어.

영상/통화 내용의 핵심 가설을 실행 코드로 고정한다.

- 경기선행지수 순환변동치 102 이상: 과열 주의
- 경기선행지수 순환변동치 첫 감소: 장기 투자 주의보
- 선행/동행 순환변동치 데드크로스: 장기 위험 경보
- 경기선행지수 순환변동치 98 근처 + KOSPI PBR 1.0 이하: 장기 축적권

이 레이어는 월간/장기 레짐 관측용이며 C60 hard gate를 대체하지 않는다.
실주문, 스케줄러, SAJANG 변경 권한은 없다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ECOS_PATH = PROJECT_ROOT / "data" / "ecos_macro.json"
OUTPUT_DIR = PROJECT_ROOT / "data_store" / "regime"

VERSION = "kr_cycle_valuation_regime_v1"

LEADING_ALIASES = (
    "leading_cycle",
    "leading_cyclical",
    "leading_ci_cycle",
    "business_leading_cycle",
    "cli_leading",
)
COINCIDENT_ALIASES = (
    "coincident_cycle",
    "coincident_cyclical",
    "coincident_ci_cycle",
    "business_coincident_cycle",
    "cli_coincident",
)

OVERHEAT_LEVEL = 102.0
FLOOR_LEVEL = 98.0
PBR_VALUE_LEVEL = 1.0
PBR_STRONG_VALUE_LEVEL = 0.8

PHASE_DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
PHASE_RISK_OFF = "CYCLE_RISK_OFF"
PHASE_OVERHEAT_WARNING = "CYCLE_OVERHEAT_WARNING"
PHASE_ACCUMULATION = "CYCLE_ACCUMULATION"
PHASE_FLOOR_WATCH = "CYCLE_FLOOR_WATCH"
PHASE_NEUTRAL = "CYCLE_NEUTRAL"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _first_existing(data: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, dict[str, Any]]:
    for key in keys:
        val = data.get(key)
        if isinstance(val, dict):
            return key, val
    return None, {}


def _series_values(indicator: dict[str, Any]) -> list[float]:
    vals: list[float] = []
    for row in indicator.get("series") or []:
        try:
            vals.append(float(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue
    return vals


def _periods(indicator: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for row in indicator.get("series") or []:
        val = row.get("period")
        if val is not None:
            result.append(str(val))
    return result


def _latest_value(indicator: dict[str, Any]) -> float | None:
    if "value" in indicator:
        try:
            return float(indicator["value"])
        except (TypeError, ValueError):
            return None
    values = _series_values(indicator)
    return values[-1] if values else None


def _is_first_decline(values: list[float]) -> bool:
    """최근값이 첫 감소인지 보수적으로 판정.

    최소 3개 값이 필요하다. 직전까지는 상승/보합, 최신값은 하락이면 첫 감소로 본다.
    """
    if len(values) < 3:
        return False
    before_prev, prev, latest = values[-3], values[-2], values[-1]
    return latest < prev and prev >= before_prev


def _is_dead_cross(leading: list[float], coincident: list[float]) -> bool:
    if len(leading) < 2 or len(coincident) < 2:
        return False
    return leading[-2] >= coincident[-2] and leading[-1] < coincident[-1]


def _extract_inputs(snapshot: dict[str, Any]) -> dict[str, Any]:
    indicators = snapshot.get("indicators", {}) or {}
    leading_key, leading = _first_existing(indicators, LEADING_ALIASES)
    coincident_key, coincident = _first_existing(indicators, COINCIDENT_ALIASES)
    kospi = snapshot.get("kospi", {}) or {}

    try:
        pbr = float(kospi["pbr"]) if kospi.get("pbr") not in (None, "") else None
    except (TypeError, ValueError):
        pbr = None

    return {
        "leading_key": leading_key,
        "leading": leading,
        "leading_value": _latest_value(leading),
        "leading_series": _series_values(leading),
        "leading_periods": _periods(leading),
        "coincident_key": coincident_key,
        "coincident": coincident,
        "coincident_value": _latest_value(coincident),
        "coincident_series": _series_values(coincident),
        "coincident_periods": _periods(coincident),
        "kospi_pbr": pbr,
        "kospi_pbr_date": kospi.get("date"),
    }


def analyze_snapshot(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """ECOS/KOSPI 스냅샷으로 장기 사이클 레짐을 판정한다."""
    snapshot = snapshot if snapshot is not None else _load_json(ECOS_PATH)
    inputs = _extract_inputs(snapshot)

    missing = []
    if inputs["leading_value"] is None:
        missing.append("leading_cycle")
    if inputs["coincident_value"] is None:
        missing.append("coincident_cycle")
    if inputs["kospi_pbr"] is None:
        missing.append("kospi_pbr")

    leading_value = inputs["leading_value"]
    pbr = inputs["kospi_pbr"]
    leading_series = inputs["leading_series"]
    coincident_series = inputs["coincident_series"]

    first_decline = _is_first_decline(leading_series)
    dead_cross = _is_dead_cross(leading_series, coincident_series)
    overheat = leading_value is not None and leading_value >= OVERHEAT_LEVEL
    floor = leading_value is not None and leading_value <= FLOOR_LEVEL
    value_floor = pbr is not None and pbr <= PBR_VALUE_LEVEL
    strong_value_floor = pbr is not None and pbr <= PBR_STRONG_VALUE_LEVEL

    reasons: list[str] = []
    if missing:
        phase = PHASE_DATA_UNAVAILABLE
        reasons.append("필수 입력 부족: " + ", ".join(missing))
    elif dead_cross:
        phase = PHASE_RISK_OFF
        reasons.append("선행/동행 순환변동치 데드크로스 발생")
    elif floor and strong_value_floor:
        phase = PHASE_ACCUMULATION
        reasons.append(f"선행지수 {leading_value:.1f} ≤ {FLOOR_LEVEL:.0f}, KOSPI PBR {pbr:.2f} ≤ {PBR_STRONG_VALUE_LEVEL:.1f}")
    elif floor and value_floor:
        phase = PHASE_FLOOR_WATCH
        reasons.append(f"선행지수 {leading_value:.1f} ≤ {FLOOR_LEVEL:.0f}, KOSPI PBR {pbr:.2f} ≤ {PBR_VALUE_LEVEL:.1f}")
    elif value_floor:
        phase = PHASE_FLOOR_WATCH
        reasons.append(f"KOSPI PBR {pbr:.2f} ≤ {PBR_VALUE_LEVEL:.1f}")
    elif first_decline and overheat:
        phase = PHASE_RISK_OFF
        reasons.append("과열권에서 선행지수 첫 감소")
    elif first_decline:
        phase = PHASE_OVERHEAT_WARNING
        reasons.append("선행지수 첫 감소")
    elif overheat:
        phase = PHASE_OVERHEAT_WARNING
        reasons.append(f"선행지수 {leading_value:.1f} ≥ {OVERHEAT_LEVEL:.0f} 과열권")
    else:
        phase = PHASE_NEUTRAL
        reasons.append("장기 사이클 중립")

    return {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "data/ecos_macro.json",
        "phase": phase,
        "missing_inputs": missing,
        "signals": {
            "leading_key": inputs["leading_key"],
            "leading_value": leading_value,
            "leading_period": (inputs["leading_periods"][-1] if inputs["leading_periods"] else None),
            "coincident_key": inputs["coincident_key"],
            "coincident_value": inputs["coincident_value"],
            "coincident_period": (inputs["coincident_periods"][-1] if inputs["coincident_periods"] else None),
            "kospi_pbr": pbr,
            "kospi_pbr_date": inputs["kospi_pbr_date"],
            "leading_first_decline": first_decline,
            "leading_coincident_dead_cross": dead_cross,
            "leading_overheat_102": overheat,
            "leading_floor_98": floor,
            "pbr_value_floor_1x": value_floor,
            "pbr_strong_value_floor_0_8x": strong_value_floor,
        },
        "reasons": reasons,
        "bot_guidance": {
            "quant_bot": {
                "use": "장기투자/레버리지 비중 판단 보조",
                "hard_gate_authority": False,
                "note": "C60 가격 레짐을 대체하지 않는다.",
            },
            "daytrade_bot": {
                "use": "시장 온도 라벨 및 보유기간/추격강도 조절 참고",
                "hard_gate_authority": False,
                "note": "개별 종목 진입 신호로 사용 금지.",
            },
        },
        "safety": {
            "real_order": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "hard_gate_authority": False,
            "auto_promotion": False,
        },
    }


def save_report(report: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"kr_cycle_valuation_{date}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(write: bool = True) -> tuple[dict[str, Any], Path | None]:
    report = analyze_snapshot()
    path = save_report(report) if write else None
    return report, path
