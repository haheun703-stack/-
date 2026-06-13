"""four_signal CSV staleness 경고 (fx-liquidity P0-1 재발방지).

생산자 `four_signal_gate.save_daily_record`가 BAT/cron 어디에도 미배선(고아)이라 5/19 수동
실행 후 조용히 멈춘 것을 preflight가 자가 노출하는 read-only 경고. 매매 안전 게이트(preflight
checks)와 분리 — RESULT/카운트 불변(6/11 finality '게이트 분리 경고' 원칙).

핵심 불변식:
  1. stale(>= max_age_days)면 경고 문자열(마지막 날짜 포함), 아니면 None.
  2. 파일 자체가 없으면 None(휴면 정상 — 생성된 적 없는 상태와 구분 불가).
  3. 파일은 있는데 데이터 행이 없으면(손상) 경고.
  4. 순수 read-only — CSV를 쓰지 않는다.
"""
from __future__ import annotations

import datetime as dt

from src.macro.four_signal_gate import _last_record_date, stale_warning

_HEADER = "date,kospi_k,us10y,krw,fg_score,fg_rating,s1,s2,s3,s4,gate_score,gate_pass\n"


def _write_csv(path, last_date: str) -> None:
    path.write_text(
        _HEADER + f"{last_date},74.2,4.62,1508.66,62.17,greed,False,False,False,False,0,False\n",
        encoding="utf-8",
    )


def test_stale_detected(tmp_path):
    p = tmp_path / "fs.csv"
    _write_csv(p, "2026-05-19")  # 26일 전
    w = stale_warning(str(p), max_age_days=5, today=dt.date(2026, 6, 14))
    assert w is not None
    assert "stale" in w and "2026-05-19" in w


def test_fresh_no_warning(tmp_path):
    p = tmp_path / "fs.csv"
    _write_csv(p, "2026-06-13")  # 1일 전
    assert stale_warning(str(p), max_age_days=5, today=dt.date(2026, 6, 14)) is None


def test_boundary_exactly_max_age(tmp_path):
    p = tmp_path / "fs.csv"
    _write_csv(p, "2026-06-09")  # 정확히 5일 전 → >= max_age_days
    assert stale_warning(str(p), max_age_days=5, today=dt.date(2026, 6, 14)) is not None


def test_boundary_one_day_under(tmp_path):
    p = tmp_path / "fs.csv"
    _write_csv(p, "2026-06-10")  # 4일 전 → 경고 없음
    assert stale_warning(str(p), max_age_days=5, today=dt.date(2026, 6, 14)) is None


def test_missing_file_no_warning(tmp_path):
    # 파일 없음 = 생성된 적 없는 상태와 구분 불가 → 휴면 정상으로 간주, 경고 안 함
    assert stale_warning(str(tmp_path / "nope.csv"), today=dt.date(2026, 6, 14)) is None


def test_empty_data_rows_flagged(tmp_path):
    p = tmp_path / "fs.csv"
    p.write_text(_HEADER, encoding="utf-8")  # 헤더만, 데이터 0행
    w = stale_warning(str(p), today=dt.date(2026, 6, 14))
    assert w is not None
    assert "행 없음" in w or "파싱 불가" in w


def test_last_record_date_parsing(tmp_path):
    p = tmp_path / "fs.csv"
    _write_csv(p, "2026-05-19")
    assert _last_record_date(str(p)) == dt.date(2026, 5, 19)


def test_last_record_date_picks_last_row(tmp_path):
    # 여러 행 중 마지막 데이터 행을 골라야 한다
    p = tmp_path / "fs.csv"
    p.write_text(
        _HEADER
        + "2026-05-18,1,2,3,4,greed,F,F,F,F,0,False\n"
        + "2026-05-19,1,2,3,4,greed,F,F,F,F,0,False\n",
        encoding="utf-8",
    )
    assert _last_record_date(str(p)) == dt.date(2026, 5, 19)


def test_no_write_side_effect(tmp_path):
    # read-only 보장: 호출 후 파일 내용/수정시각 불변
    p = tmp_path / "fs.csv"
    _write_csv(p, "2026-05-19")
    before = p.read_bytes()
    stale_warning(str(p), today=dt.date(2026, 6, 14))
    assert p.read_bytes() == before
