"""매크로 예실 트래커 — score/stance/market_impact 로직 + 적재 어댑터 테스트.

★검증 핵심:
  - 인플레/금리 지표(CPI 등): 상회=긴축=악재 / 하회=완화=호재 (단정 명확)
  - 경기/고용 지표(NFP·ISM 등): market_impact 중립(국면의존, 임의 단정 금지)
  - 발표 전(actual=None): surprise/score/stance/market_impact 전부 None
  - upsert는 dry_run 기본 (실제 외부 write 0)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src.macro.macro_forecast_tracker import (  # noqa: E402
    IMPACT_BAD,
    IMPACT_GOOD,
    IMPACT_NEUTRAL,
    STANCE_EASE,
    STANCE_NEUTRAL,
    STANCE_TIGHT,
    build_forecast_row,
    classify_market_impact,
    classify_stance,
    compute_surprise,
    compute_surprise_score,
    month_to_event_date,
    upsert_forecast_actual,
)


# ── surprise / score ──
def test_surprise_basic():
    assert compute_surprise(3.2, 3.0) == 0.2
    assert compute_surprise(2.8, 3.0) == -0.2
    assert compute_surprise(None, 3.0) is None  # 발표 전
    assert compute_surprise(3.0, None) is None


def test_score_normalized_by_sigma():
    # CPI typical_sigma=0.1 → 0.2 서프라이즈 = 2.0점
    assert compute_surprise_score("CPI_HEAD", 0.2) == 2.0
    assert compute_surprise_score("CPI_HEAD", -0.1) == -1.0
    assert compute_surprise_score("UNKNOWN", 0.2) is None
    assert compute_surprise_score("CPI_HEAD", None) is None


# ── stance (통화 함의) ──
def test_cpi_upside_is_hawkish():
    score = compute_surprise_score("CPI_HEAD", 0.2)  # +2.0
    assert classify_stance("CPI_HEAD", score) == STANCE_TIGHT


def test_cpi_downside_is_dovish():
    score = compute_surprise_score("CPI_HEAD", -0.2)  # -2.0
    assert classify_stance("CPI_HEAD", score) == STANCE_EASE


def test_cpi_inline_is_neutral():
    assert classify_stance("CPI_HEAD", 0.0) == STANCE_NEUTRAL


def test_unrate_upside_is_dovish():
    # 실업률 상회(actual>consensus) = 경기둔화 = 비둘기(완화) (hawkish_sign=-1)
    score = compute_surprise_score("UNRATE", 0.2)  # +2.0
    assert classify_stance("UNRATE", score) == STANCE_EASE


def test_context_dependent_indicator_stance_neutral():
    # NFP/ISM 등 hawkish_sign=0 → 통화 스탠스 중립
    score = compute_surprise_score("NFP", 50.0)
    assert classify_stance("NFP", score) == STANCE_NEUTRAL


# ── market_impact (증시 영향) ──
def test_cpi_upside_market_bad():
    score = compute_surprise_score("CPI_HEAD", 0.2)
    assert classify_market_impact("CPI_HEAD", score) == IMPACT_BAD  # 긴축=악재


def test_cpi_downside_market_good():
    score = compute_surprise_score("CPI_HEAD", -0.2)
    assert classify_market_impact("CPI_HEAD", score) == IMPACT_GOOD  # 완화=호재


def test_growth_indicator_market_impact_neutral():
    # 경기/고용은 국면의존 → 임의 단정 안 함(중립)
    score = compute_surprise_score("NFP", 80.0)
    assert classify_market_impact("NFP", score) == IMPACT_NEUTRAL
    assert classify_market_impact("ISM_MFG", compute_surprise_score("ISM_MFG", 3.0)) == IMPACT_NEUTRAL


# ── build_forecast_row ──
def test_row_pre_release_has_no_actual_fields():
    row = build_forecast_row("CPI_HEAD", "2026-06-11", consensus=3.0,
                             consensus_source="cleveland_nowcast", prior=2.9)
    assert row["indicator_code"] == "CPI_HEAD"
    assert row["region"] == "US"
    assert row["consensus"] == 3.0
    assert row["actual"] is None
    assert row["surprise"] is None and row["surprise_score"] is None
    assert row["stance"] is None and row["market_impact"] is None
    assert row["indicator_name_ko"] == "미국 소비자물가(헤드라인)"


def test_row_post_release_computes_all():
    row = build_forecast_row("CPI_HEAD", "2026-06-11", consensus=3.0,
                             consensus_source="cleveland_nowcast", actual=3.2, prior=2.9)
    assert row["surprise"] == 0.2
    assert row["surprise_score"] == 2.0
    assert row["stance"] == STANCE_TIGHT
    assert row["market_impact"] == IMPACT_BAD
    assert row["impact"] == 5  # CPI 별점


def test_row_unknown_code_graceful():
    row = build_forecast_row("WEIRD", "2026-06-11", consensus=1.0, actual=1.5)
    assert row["indicator_name_ko"] == "WEIRD"
    assert row["surprise"] == 0.5
    assert row["surprise_score"] is None  # 메타 없음 → 점수 None


# ── event_date 규약 (정보봇 P1-① 정합) ──
def test_month_to_event_date_is_data_month_first():
    # 월간 지표는 데이터 귀속월 1일(발표일 아님)
    assert month_to_event_date(2026, 6) == "2026-06-01"
    assert month_to_event_date(2026, 12) == "2026-12-01"


def test_build_row_rejects_malformed_event_date():
    # 잘못된 event_date 형식 → 적재 차단(데이터 품질 방어)
    with pytest.raises(ValueError):
        build_forecast_row("CPI_HEAD", "2026-6", consensus=3.0)  # 0패딩 없음
    with pytest.raises(ValueError):
        build_forecast_row("CPI_HEAD", "내일", consensus=3.0)


def test_build_row_accepts_data_month_event_date():
    # 6월 데이터(7월 발표)와 임박 5월 데이터가 서로 다른 행으로 구분됨
    jun = build_forecast_row("CPI_HEAD", month_to_event_date(2026, 6), consensus=0.124)
    may = build_forecast_row("CPI_HEAD", month_to_event_date(2026, 5), consensus=0.20, actual=0.18)
    assert jun["event_date"] == "2026-06-01" and jun["actual"] is None
    assert may["event_date"] == "2026-05-01" and may["surprise"] == round(0.18 - 0.20, 4)


# ── upsert 안전장치 ──
def test_upsert_dry_run_writes_nothing():
    rows = [build_forecast_row("CPI_HEAD", "2026-06-11", consensus=3.0, actual=3.2)]
    res = upsert_forecast_actual(rows)  # dry_run 기본 True
    assert res["dry_run"] is True
    assert res["written"] == 0
    assert res["rows"] == 1


def test_upsert_empty():
    res = upsert_forecast_actual([])
    assert res["written"] == 0
