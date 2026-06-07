"""FLOWX paper training / shadow forward 장부 보강 검증 (6/7 사장님 지시).

검증 대상(보강 9필드만, tier/engine/regime/실주문 무변경):
  A 후보선정: candidate_score(=total_score 노출), reason[](확정값 재표현)
  B 가상진입: D0 종가 즉시 / D+1 시가 익일 backfill (둘 다 기록)

★실주문/PAPER_OPEN/매도/scheduler/SAJANG 무관. 순수 함수 + tmp 격리 테스트.
"""
from __future__ import annotations

import json

import pandas as pd

from src.use_cases.morning_plan_07 import _build_reason, _candidate_summary
from src.use_cases import smart_entry_adapter as sea
from src.use_cases.smart_entry_adapter import (
    _next_kr_trading_day,
    _to_payload,
    backfill_d1_open,
)


# ── A. candidate_score + reason (설명 전용, tier 미개입) ──

def test_build_reason_is_pure_reexpression_of_confirmed_values() -> None:
    row = {
        "_tier": "CORE",
        "_drop_context": "resilient_pullback",
        "_supply_state": "foreign_accumulation",
        "_floor_label": "바닥다지기후보",
        "total_score": 5,
    }
    sl = {
        "price_axis": {"weekly_open_state": "ABOVE", "half_year_open_state": "ABOVE"},
        "half_year_leader": {"half_year_leader_grade": "HY_LEADER_CORE"},
        "annual_overheat": {"annual_overheat_warning": True, "overheat_grade": "OVERHEAT_500"},
    }
    reason = _build_reason(row, "R4_NORMAL_BULL", sl)
    assert reason == [
        "C60_BULL", "CORE_TIER", "RESILIENT_PULLBACK", "FOREIGN_ACCUMULATION",
        "FLOOR:바닥다지기후보", "WEEKLY_OPEN_ABOVE", "HALF_YEAR_OPEN_ABOVE",
        "HY_LEADER_CORE", "OVERHEAT_500",
    ]


def test_build_reason_bear_and_empty_labels() -> None:
    row = {"_tier": "CONTROL"}
    reason = _build_reason(row, "R1_BEAR", None)
    assert reason == ["C60_BEAR", "CONTROL_TIER"]


def test_candidate_summary_exposes_score_without_changing_tier() -> None:
    row = {
        "ticker": "005930", "name": "삼성전자", "_tier": "WATCH",
        "close": 80000, "stop_loss": 77000, "target_price": 90000,
        "total_score": 7, "_floor_label": "x", "_drop_context": "y", "_supply_state": "z",
    }
    out = _candidate_summary(row, "obs", {"price_axis": {}}, "R4_NORMAL_BULL")
    assert out["tier"] == "WATCH"               # tier 그대로(_candidate_summary는 분류 안 함)
    assert out["candidate_score"] == 7          # total_score 노출
    assert "WATCH_TIER" in out["reason"]
    assert out["candidate_score"] != out["tier"]  # score는 tier 결정에 미사용


# ── B. D0/D1 가상진입 필드 + 다음 거래일 계산 ──

def test_next_kr_trading_day_skips_holiday_and_weekend() -> None:
    # 6/5(금) 다음 거래일 = 6/8(월). 6/6 현충일·6/7 일요일 건너뜀.
    assert _next_kr_trading_day("2026-06-05") == "2026-06-08"
    assert _next_kr_trading_day("2026-06-08") == "2026-06-09"
    assert _next_kr_trading_day(None) is None
    assert _next_kr_trading_day("bad-date") is None


def test_to_payload_records_both_entry_bases() -> None:
    row = {"ticker": "005930", "ref_close": 80000, "candidate_score": 3, "reason": ["C60_BULL"]}
    p = _to_payload(row, "SHADOW_OPEN", "2026-06-05")
    assert p["virtual_entry_price_d0_close"] == 80000
    assert p["virtual_entry_date_d0"] == "2026-06-05"
    assert p["entry_basis_d0"] == "D0_CLOSE"
    assert p["virtual_entry_price_d1_open"] is None
    assert p["virtual_entry_date_d1"] == "2026-06-08"
    assert p["entry_basis_d1"] == "D1_OPEN"
    assert p["d1_open_filled"] is False
    assert p["real_order"] is False
    assert p["candidate_score"] == 3 and p["reason"] == ["C60_BULL"]


# ── C. D1_OPEN backfill (OHLCV mock, tmp 격리) ──

def _write_ledger(tmp_path, d1_date):
    ledger = {
        "as_of_date": "2026-06-05",
        "shadow_entries": [{
            "ticker": "005930", "name": "삼성전자", "status": "SHADOW_OPEN",
            "entry_basis_d1": "D1_OPEN", "virtual_entry_date_d1": d1_date,
            "virtual_entry_price_d1_open": None, "d1_open_filled": False,
        }],
    }
    path = tmp_path / "shadow_entries_2026-06-05.json"
    path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    return path


def _patch_ohlcv(monkeypatch, df):
    import src.etf.samsung_single_leverage_shadow as sls
    import src.etf.c60_shadow as c60
    monkeypatch.setattr(sls, "load_daily_ohlcv", lambda *a, **k: df)
    monkeypatch.setattr(c60, "normalize_ohlcv", lambda x: x)


def test_backfill_fills_d1_open_when_data_present(tmp_path, monkeypatch) -> None:
    df = pd.DataFrame(
        {"open": [81200], "high": [82000], "low": [80000], "close": [81500], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-08")], name="date"),
    )
    _patch_ohlcv(monkeypatch, df)
    path = _write_ledger(tmp_path, "2026-06-08")
    result = backfill_d1_open("2026-06-05", prefer_remote=False, output_dir=tmp_path)
    assert result["filled"] == 1
    e = json.loads(path.read_text(encoding="utf-8"))["shadow_entries"][0]
    assert e["d1_open_filled"] is True
    assert e["virtual_entry_price_d1_open"] == 81200


def test_backfill_skips_when_d1_data_absent(tmp_path, monkeypatch) -> None:
    # 장부 D1=6/08인데 OHLCV엔 6/05까지만 → 채우지 않음(다음 기회).
    df = pd.DataFrame(
        {"open": [70000], "high": [71000], "low": [69000], "close": [70500], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-05")], name="date"),
    )
    _patch_ohlcv(monkeypatch, df)
    path = _write_ledger(tmp_path, "2026-06-08")
    result = backfill_d1_open("2026-06-05", prefer_remote=False, output_dir=tmp_path)
    assert result["filled"] == 0
    e = json.loads(path.read_text(encoding="utf-8"))["shadow_entries"][0]
    assert e["d1_open_filled"] is False
    assert e["virtual_entry_price_d1_open"] is None


def test_backfill_no_ledger_returns_zero(tmp_path) -> None:
    result = backfill_d1_open("2099-01-01", prefer_remote=False, output_dir=tmp_path)
    assert result["filled"] == 0 and result["reason"] == "no_ledger"
