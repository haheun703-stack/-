"""섹터 모멘텀 관측 라벨 테스트 (6/10 올바른 개선 ①단계).

★검증 핵심: classify_tier(SSOT) 무변경, 6/12 판정 무오염. 라벨은 관측뿐 —
tier/hard gate/진입/주문 어디에도 분기로 안 쓰임. graceful 로드 + best/worst 둘 다 기록.

회귀 케이스:
  - 강세 섹터(은행) 종목 → STRONG_SECTOR, in_strong_sector=True
  - 이탈 섹터(증권) 종목 → in_exodus_sector=True, EXODUS_SECTOR
  - 약세 섹터(2차전지)만 노출 → WEAK_SECTOR
  - 다중 섹터(증권+금융) → best/worst 정확히 분리
  - 미매핑 종목 → unmapped / json 없음 → unavailable / 날짜 불일치 → stale
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.sector_momentum_label import (  # noqa: E402
    STATUS_OK,
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    STATUS_UNMAPPED,
    assess_sector_momentum,
    load_sector_composite,
    load_stock_to_sector,
    sector_momentum_summary,
)

# 실데이터(2026-06-10) 축약 fixture
COMPOSITE = {
    "momentum_date": "2026-06-10",
    "sector_count": 6,
    "sectors": [
        {"sector": "은행", "regime": "STRONG_ROTATION", "composite_score": 79.0},
        {"sector": "IT", "regime": "STRONG_ROTATION", "composite_score": 76.0},
        {"sector": "금융", "regime": "NEUTRAL", "composite_score": 58.5},
        {"sector": "에너지화학", "regime": "NEUTRAL", "composite_score": 40.6},
        {"sector": "2차전지", "regime": "WEAK_ROTATION", "composite_score": 35.3},
        {"sector": "증권", "regime": "EXODUS", "composite_score": 23.3},
    ],
    "regime_summary": {"STRONG_ROTATION": 2, "NEUTRAL": 2, "WEAK_ROTATION": 1, "EXODUS": 1},
    "strong_sectors": ["은행", "IT"],
    "exodus_sectors": ["증권"],
}

S2S = {
    "091220": ["은행", "금융", "KRX300"],          # 강세
    "006800": ["증권", "금융", "KRX300"],          # 이탈+중립 (다중)
    "051910": ["2차전지", "에너지화학", "KRX300"],  # 약세 노출(LG화학형)
    "139260": ["IT", "KRX300"],                    # 강세
    "999999": ["존재하지않는섹터", "KRX300"],       # 미매핑
}


# ── 강세/이탈/약세 분류 ──
def test_strong_sector_stock():
    r = assess_sector_momentum("091220", COMPOSITE, S2S)
    assert r["status"] == STATUS_OK
    assert r["sector_strength_label"] == "STRONG_SECTOR"
    assert r["in_strong_sector"] is True
    assert r["in_exodus_sector"] is False
    assert r["best_sector"] == "은행"
    assert r["best_regime"] == "STRONG_ROTATION"


def test_exodus_sector_stock_multi():
    # 증권(EXODUS 23.3) + 금융(NEUTRAL 58.5) → best=금융, worst=증권, in_exodus=True
    r = assess_sector_momentum("006800", COMPOSITE, S2S)
    assert r["status"] == STATUS_OK
    assert r["in_exodus_sector"] is True
    assert r["best_sector"] == "금융"          # composite 최고가 대표
    assert r["worst_sector"] == "증권"          # composite 최저가 최악 노출
    assert r["worst_regime"] == "EXODUS"
    # best 기준 라벨은 금융=NEUTRAL → NEUTRAL_SECTOR, 단 worst/플래그로 이탈 노출 드러남
    assert r["sector_strength_label"] == "NEUTRAL_SECTOR"


def test_weak_sector_exposure_lgchem_form():
    # LG화학형: 2차전지(WEAK 35.3) + 에너지화학(NEUTRAL 40.6) → best=에너지화학 NEUTRAL,
    # worst=2차전지 WEAK. 강세/이탈 어디에도 없지만 worst_regime로 약세 노출 포착.
    r = assess_sector_momentum("051910", COMPOSITE, S2S)
    assert r["status"] == STATUS_OK
    assert r["in_strong_sector"] is False
    assert r["in_exodus_sector"] is False
    assert r["best_regime"] == "NEUTRAL"
    assert r["worst_regime"] == "WEAK_ROTATION"
    assert r["worst_sector"] == "2차전지"


def test_single_sector_best_equals_worst():
    r = assess_sector_momentum("139260", COMPOSITE, S2S)
    assert r["best_sector"] == r["worst_sector"] == "IT"
    assert r["sector_strength_label"] == "STRONG_SECTOR"


# ── graceful ──
def test_unmapped_stock():
    r = assess_sector_momentum("999999", COMPOSITE, S2S)
    assert r["status"] == STATUS_UNMAPPED
    assert r["sector_strength_label"] is None


def test_stock_not_in_s2s():
    r = assess_sector_momentum("000000", COMPOSITE, S2S)
    assert r["status"] == STATUS_UNMAPPED


def test_composite_none_unavailable():
    r = assess_sector_momentum("091220", None, S2S)
    assert r["status"] == STATUS_UNAVAILABLE
    assert r["sector_strength_label"] is None


def test_empty_sectors_unavailable():
    r = assess_sector_momentum("091220", {"momentum_date": "2026-06-10", "sectors": []}, S2S)
    assert r["status"] == STATUS_UNAVAILABLE


def test_stale_when_date_mismatch():
    r = assess_sector_momentum("091220", COMPOSITE, S2S, as_of_date="2026-06-11")
    assert r["status"] == STATUS_STALE
    # stale이어도 라벨은 부착(관측만)
    assert r["sector_strength_label"] == "STRONG_SECTOR"
    assert "stale" in (r["note"] or "")


def test_ok_when_date_matches():
    r = assess_sector_momentum("091220", COMPOSITE, S2S, as_of_date="2026-06-10")
    assert r["status"] == STATUS_OK


def test_broad_buckets_excluded():
    # KRX300만 있는 종목은 매칭 0 → unmapped (광범위 분류축은 섹터 강약 아님)
    r = assess_sector_momentum("X", COMPOSITE, {"X": ["KRX300", "KOSPI200"]}, None)
    assert r["status"] == STATUS_UNMAPPED


# ── 로드 graceful ──
def test_load_sector_composite_absent(tmp_path):
    assert load_sector_composite(tmp_path / "nope.json") is None


def test_load_sector_composite_broken(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid", encoding="utf-8")
    assert load_sector_composite(p) is None


def test_load_stock_to_sector_absent(tmp_path):
    assert load_stock_to_sector(tmp_path / "nope.json") == {}


def test_real_files_load():
    # 실제 data/sector_rotation/*.json 로드(존재 시) — 스키마 회귀 가드
    comp = load_sector_composite()
    s2s = load_stock_to_sector()
    if comp is not None:
        assert "sectors" in comp
    assert isinstance(s2s, dict)


# ── summary graceful ──
def test_summary_graceful():
    assert sector_momentum_summary(None)["available"] is False
    s = sector_momentum_summary(COMPOSITE)
    assert s["available"] is True
    assert s["strong_sectors"] == ["은행", "IT"]
    assert s["exodus_sectors"] == ["증권"]


# ── morning_plan 통합: tier 불변 + sector_momentum 부착 ──
def test_morning_plan_attaches_sector_without_changing_tier():
    from src.use_cases.morning_plan_07 import build_plan_document

    picks = [{
        "ticker": "051910", "name": "LG화학", "_tier": "CORE",
        "close": 329000, "stop_loss": 1, "target_price": 1,
        "_floor_label": "바닥다지기후보", "_drop_context": "resilient_pullback",
        "_supply_state": "foreign_accumulation", "total_score": 2,
    }]
    control = [{
        "ticker": "091220", "name": "KODEX은행", "_tier": "CONTROL",
        "close": 10000, "stop_loss": 1, "target_price": 1,
        "_floor_label": "중립", "_drop_context": "x",
        "_supply_state": "x", "total_score": 1,
    }]
    policy = {"market_regime": "R4_NORMAL_BULL",
              "engines": {"smart_entry": "ALLOWED_SHADOW"}, "as_of_date": "2026-06-10"}
    shadow_labels = {
        "051910": {"sector_momentum": assess_sector_momentum("051910", COMPOSITE, S2S, "2026-06-10")},
        "091220": {"sector_momentum": assess_sector_momentum("091220", COMPOSITE, S2S, "2026-06-10")},
    }
    plan = build_plan_document(
        policy, picks, control, {"as_of_date": "2026-06-10"},
        shadow_labels=shadow_labels, market_regime=None,
        sector_momentum=sector_momentum_summary(COMPOSITE),
    )
    core = plan["tiers"]["CORE"][0]
    ctrl = plan["tiers"]["CONTROL"][0]
    # ★tier 자체는 SSOT 그대로(자동 변경 없음)
    assert core["tier"] == "CORE"
    assert ctrl["tier"] == "CONTROL"
    # 섹터 라벨은 shadow_labels로 그대로 부착
    assert core["shadow_labels"]["sector_momentum"]["worst_regime"] == "WEAK_ROTATION"
    assert ctrl["shadow_labels"]["sector_momentum"]["sector_strength_label"] == "STRONG_SECTOR"
    # reason 토큰에도 라벨 반영(확정값 재표현)
    assert "STRONG_SECTOR" in ctrl["reason"]
    # plan 레벨 요약
    assert plan["sector_momentum"]["available"] is True
    assert "은행" in plan["sector_momentum"]["strong_sectors"]


def test_morning_plan_graceful_when_no_sector_data():
    from src.use_cases.morning_plan_07 import build_plan_document

    picks = [{
        "ticker": "051910", "name": "LG화학", "_tier": "CORE",
        "close": 1, "stop_loss": 1, "target_price": 1,
        "_floor_label": "x", "_drop_context": "x", "_supply_state": "x", "total_score": 0,
    }]
    policy = {"market_regime": "R4_NORMAL_BULL",
              "engines": {"smart_entry": "ALLOWED_SHADOW"}, "as_of_date": "2026-06-10"}
    plan = build_plan_document(
        policy, picks, [], {"as_of_date": "2026-06-10"},
        shadow_labels={}, market_regime=None, sector_momentum=None,
    )
    assert plan["tiers"]["CORE"][0]["tier"] == "CORE"
    assert plan["sector_momentum"]["available"] is False


# ── daily_review 통합: 라벨별 성과 집계 ──
def test_daily_review_aggregates_sector_labels():
    from src.use_cases.daily_review import build_label_performance

    rows = [
        {"data_available": True, "raw_fwd_pct": {"D+1": 2.0, "D+3": 3.0, "D+10": 5.0},
         "mfe_pct": 6.0, "shadow_labels": {
             "sector_momentum": {"status": "ok", "sector_strength_label": "STRONG_SECTOR",
                                 "best_regime": "STRONG_ROTATION", "worst_regime": "STRONG_ROTATION"}}},
        {"data_available": True, "raw_fwd_pct": {"D+1": -3.0, "D+3": -4.0, "D+10": -7.0},
         "mfe_pct": 1.0, "shadow_labels": {
             "sector_momentum": {"status": "ok", "sector_strength_label": "NEUTRAL_SECTOR",
                                 "best_regime": "NEUTRAL", "worst_regime": "EXODUS"}}},
    ]
    lp = build_label_performance(rows)
    assert "sector_strength" in lp
    assert "sector_best_regime" in lp
    assert "sector_worst_regime" in lp
    assert lp["sector_strength"]["STRONG_SECTOR"]["count"] == 1
    assert lp["sector_strength"]["STRONG_SECTOR"]["mean_d10"] == 5.0
    # worst_regime 축에서 EXODUS 노출 종목이 분리 집계됨(약세 노출 검증축)
    assert lp["sector_worst_regime"]["EXODUS"]["count"] == 1
    assert lp["sector_worst_regime"]["EXODUS"]["mean_d10"] == -7.0
