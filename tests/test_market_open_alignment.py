"""MARKET_OPEN_REGIME 보조 재심사 레이어 테스트.

★검증 핵심: tier 자동 변경 없음(classify_tier SSOT 불변). 재심사 라벨/점수만 부착.
회귀 케이스(사장님 지시):
  - HPSP: semiconductor + CONTROL → RECHECK_CONTROL_TO_WATCH
  - LG화학: battery + CORE + 약정렬 → CORE_WEAK_ALIGNMENT_RECHECK
  - json 없음 → 현행 tier 유지(unavailable)
  - freshness stale → 자동 변경 없음(보류)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.market_open_alignment import (  # noqa: E402
    ACTION_CORE_WEAK,
    ACTION_RECHECK_CONTROL,
    STATUS_OK,
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    assess_alignment,
    load_market_regime,
    regime_summary,
)

REGIME = {
    "market_bias": "KOSPI",
    "etf_dominant": "semiconductor",
    "leading_themes": ["semiconductor", "AI반도체"],
    "avoid_themes": ["battery", "2차전지"],
    "sector_weights": {"semiconductor": 1.0},
    "freshness": {"ok": True},
}

REAL_SCHEMA_REGIME = {
    "market_bias": "KOSPI",
    "etf_dominant": True,
    "leading_themes": [
        {"key": "it", "label": "IT", "score": 68.0},
        {"key": "bio", "label": "BIO", "score": 58.0},
    ],
    "avoid_themes": [],
    "sector_weights": {
        "semiconductor": 1.315,
        "battery": 1.121,
    },
    "freshness": {"ok": True},
}


# ── 회귀 케이스 (사장님 지시) ──
def test_hpsp_control_semiconductor_recheck_to_watch():
    r = assess_alignment("CONTROL", "semiconductor", REGIME)
    assert r["status"] == STATUS_OK
    assert r["market_alignment_action"] == ACTION_RECHECK_CONTROL
    assert r["alignment_score"] > 0


def test_lgchem_core_battery_weak_alignment_recheck():
    r = assess_alignment("CORE", "battery", REGIME)
    assert r["market_alignment_action"] == ACTION_CORE_WEAK
    assert r["alignment_score"] < 0  # avoid 감점


def test_json_absent_keeps_current_tier():
    r = assess_alignment("CONTROL", "semiconductor", None)
    assert r["status"] == STATUS_UNAVAILABLE
    assert r["market_alignment_action"] is None


def test_freshness_stale_no_auto_change():
    stale = dict(REGIME)
    stale["freshness"] = {"ok": False}
    r = assess_alignment("CONTROL", "semiconductor", stale)
    assert r["status"] == STATUS_STALE
    assert r["market_alignment_action"] is None  # 자동 변경 없음


# ── 경계/대칭 케이스 ──
def test_core_aligned_no_recheck():
    r = assess_alignment("CORE", "semiconductor", REGIME)
    assert r["market_alignment_action"] is None  # 정렬 좋으면 재심사 없음
    assert r["alignment_score"] > 0


def test_core_weak_when_not_in_leading_even_if_not_avoid():
    # 주도축에 없으면(회피축이 아니어도) CORE 약정렬 재심사
    r = assess_alignment("CORE", "shipbuilding", REGIME)
    assert r["market_alignment_action"] == ACTION_CORE_WEAK


def test_control_not_aligned_no_recheck():
    # 주도축이 아니면 CONTROL→WATCH 재심사 안 함(올릴 근거 없음)
    r = assess_alignment("CONTROL", "shipbuilding", REGIME)
    assert r["market_alignment_action"] is None


def test_watch_has_score_but_no_action():
    # 재심사 라벨은 CORE/CONTROL 전용 — WATCH는 점수만, action 없음
    r = assess_alignment("WATCH", "semiconductor", REGIME)
    assert r["market_alignment_action"] is None
    assert r["alignment_score"] > 0


def test_theme_partial_match_korean():
    # 섹터 'AI반도체' ↔ 테마 '반도체' 부분 매칭
    regime = {"leading_themes": ["반도체"], "freshness": {"ok": True}}
    r = assess_alignment("CONTROL", "AI반도체", regime)
    assert r["in_leading"] is True
    assert r["market_alignment_action"] == ACTION_RECHECK_CONTROL


def test_no_sector_no_match():
    r = assess_alignment("CORE", None, REGIME)
    # 섹터 모르면 leading에도 없음 → CORE 약정렬로 분류(보수적), 단 점수 0
    assert r["in_leading"] is False
    assert r["alignment_score"] == 0.0


def test_real_schema_dict_themes_are_supported():
    r = assess_alignment("CORE", "바이오", REAL_SCHEMA_REGIME)
    assert r["status"] == STATUS_OK
    assert r["in_leading"] is True
    assert r["market_alignment_action"] is None


def test_control_recheck_when_sector_weight_strong_even_not_top_leading():
    # 단타봇 실데이터에서 반도체가 top leading 밖이어도 sector_weights가 강하면 재심사.
    r = assess_alignment("CONTROL", "semiconductor", REAL_SCHEMA_REGIME)
    assert r["weight_aligned"] is True
    assert r["market_alignment_action"] == ACTION_RECHECK_CONTROL


def test_core_weak_when_weight_below_alignment_threshold():
    r = assess_alignment("CORE", "battery", REAL_SCHEMA_REGIME)
    assert r["weight_aligned"] is False
    assert r["market_alignment_action"] == ACTION_CORE_WEAK


def test_real_schema_dict_avoid_theme_penalizes_core():
    regime = dict(REAL_SCHEMA_REGIME)
    regime["avoid_themes"] = [{"key": "battery", "label": "BATTERY", "score": -20.0}]
    r = assess_alignment("CORE", "battery", regime)
    assert r["in_avoid"] is True
    assert r["alignment_score"] < 0
    assert r["market_alignment_action"] == ACTION_CORE_WEAK


# ── graceful 로드 ──
def test_load_market_regime_absent(tmp_path):
    assert load_market_regime(tmp_path / "nope.json") is None


def test_load_market_regime_broken(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert load_market_regime(p) is None


def test_regime_summary_graceful():
    assert regime_summary(None)["available"] is False
    s = regime_summary(REGIME)
    assert s["available"] is True
    assert s["market_bias"] == "KOSPI"
    assert "semiconductor" in s["leading_themes"]


# ── morning_plan 통합: tier 불변 + market_alignment 부착 ──
def test_morning_plan_attaches_alignment_without_changing_tier():
    from src.use_cases.morning_plan_07 import build_plan_document

    picks = [{
        "ticker": "051910", "name": "LG화학", "_tier": "CORE",
        "close": 329000, "stop_loss": 1, "target_price": 1,
        "_floor_label": "바닥다지기후보", "_drop_context": "resilient_pullback",
        "_supply_state": "foreign_accumulation", "total_score": 2,
    }]
    control = [{
        "ticker": "403870", "name": "HPSP", "_tier": "CONTROL",
        "close": 57000, "stop_loss": 1, "target_price": 1,
        "_floor_label": "중립", "_drop_context": "resilient_pullback",
        "_supply_state": "foreign_accumulation", "total_score": 2,
    }]
    policy = {"market_regime": "R4_NORMAL_BULL",
              "engines": {"smart_entry": "ALLOWED_SHADOW"}, "as_of_date": "2026-06-09"}
    shadow_labels = {
        "051910": {"half_year_leader": {"sector": "battery"}},
        "403870": {"half_year_leader": {"sector": "semiconductor"}},
    }
    plan = build_plan_document(
        policy, picks, control, {"as_of_date": "2026-06-09"},
        shadow_labels=shadow_labels, market_regime=REGIME,
    )
    core = plan["tiers"]["CORE"][0]
    ctrl = plan["tiers"]["CONTROL"][0]
    # ★tier 자체는 SSOT 그대로(자동 변경 없음)
    assert core["tier"] == "CORE"
    assert ctrl["tier"] == "CONTROL"
    # 재심사 라벨만 부착
    assert core["market_alignment"]["market_alignment_action"] == ACTION_CORE_WEAK
    assert ctrl["market_alignment"]["market_alignment_action"] == ACTION_RECHECK_CONTROL
    assert plan["market_open_regime"]["available"] is True


def test_morning_plan_graceful_when_no_regime():
    from src.use_cases.morning_plan_07 import build_plan_document

    picks = [{
        "ticker": "051910", "name": "LG화학", "_tier": "CORE",
        "close": 1, "stop_loss": 1, "target_price": 1,
        "_floor_label": "바닥다지기후보", "_drop_context": "x",
        "_supply_state": "x", "total_score": 0,
    }]
    policy = {"market_regime": "R4_NORMAL_BULL",
              "engines": {"smart_entry": "ALLOWED_SHADOW"}, "as_of_date": "2026-06-09"}
    plan = build_plan_document(
        policy, picks, [], {"as_of_date": "2026-06-09"},
        shadow_labels={}, market_regime=None,
    )
    core = plan["tiers"]["CORE"][0]
    assert core["tier"] == "CORE"  # 현행 유지
    assert core["market_alignment"]["status"] == STATUS_UNAVAILABLE
    assert plan["market_open_regime"]["available"] is False
