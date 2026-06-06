from __future__ import annotations

from src.use_cases import kr_cycle_valuation_regime as kcv


def _indicator(values: list[float]) -> dict:
    return {
        "value": values[-1],
        "period": f"2026{len(values):02d}",
        "series": [{"period": f"2026{i + 1:02d}", "value": v} for i, v in enumerate(values)],
    }


def _snapshot(leading: list[float] | None, coincident: list[float] | None, pbr: float | None) -> dict:
    indicators = {}
    if leading is not None:
        indicators["leading_cycle"] = _indicator(leading)
    if coincident is not None:
        indicators["coincident_cycle"] = _indicator(coincident)
    kospi = {}
    if pbr is not None:
        kospi = {"pbr": pbr, "date": "20260605"}
    return {"indicators": indicators, "kospi": kospi}


def test_missing_inputs_are_reported_without_authority() -> None:
    report = kcv.analyze_snapshot({})

    assert report["phase"] == kcv.PHASE_DATA_UNAVAILABLE
    assert "leading_cycle" in report["missing_inputs"]
    assert report["safety"]["hard_gate_authority"] is False
    assert report["bot_guidance"]["daytrade_bot"]["hard_gate_authority"] is False


def test_overheat_first_decline_is_risk_off() -> None:
    report = kcv.analyze_snapshot(_snapshot([101.2, 103.5, 103.1], [100.0, 100.4, 100.8], 1.25))

    assert report["phase"] == kcv.PHASE_RISK_OFF
    assert report["signals"]["leading_first_decline"] is True
    assert report["signals"]["leading_overheat_102"] is True


def test_dead_cross_is_risk_off_even_without_102() -> None:
    report = kcv.analyze_snapshot(_snapshot([99.8, 100.1, 99.7], [99.5, 99.9, 99.9], 1.05))

    assert report["phase"] == kcv.PHASE_RISK_OFF
    assert report["signals"]["leading_coincident_dead_cross"] is True


def test_98_and_pbr_0_8_is_accumulation() -> None:
    report = kcv.analyze_snapshot(_snapshot([99.2, 98.5, 97.9], [97.0, 97.2, 97.3], 0.79))

    assert report["phase"] == kcv.PHASE_ACCUMULATION
    assert report["signals"]["leading_floor_98"] is True
    assert report["signals"]["pbr_strong_value_floor_0_8x"] is True


def test_98_and_pbr_1_is_floor_watch() -> None:
    report = kcv.analyze_snapshot(_snapshot([99.2, 98.5, 97.9], [97.0, 97.2, 97.3], 0.95))

    assert report["phase"] == kcv.PHASE_FLOOR_WATCH
    assert report["signals"]["pbr_value_floor_1x"] is True


def test_neutral_has_no_order_authority() -> None:
    report = kcv.analyze_snapshot(_snapshot([99.0, 100.0, 101.0], [99.0, 99.3, 99.5], 1.1))

    assert report["phase"] == kcv.PHASE_NEUTRAL
    assert report["safety"]["real_order"] is False
    assert report["safety"]["scheduler_changed"] is False
    assert report["safety"]["sajang_changed"] is False
