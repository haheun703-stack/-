"""B-39 기타법인 DB 충전 회귀 테스트.

KIS 종목별 수급 API(FHKST01010900)에 기타법인이 없어 parquet 기타법인이
2026-04-17부터 전량 0으로 남던 결함의 수정(_fill_etc_corp) 검증.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from scripts import extend_parquet_data as epd


def _make_db(tmp_path, rows):
    """(date, ticker, investor, net_val) 행으로 investor_daily 스키마 재현."""
    db = tmp_path / "investor_daily.db"
    con = sqlite3.connect(db)
    con.execute(
        "create table investor_daily (date text, ticker text, name text, investor text, "
        "sell_vol int, buy_vol int, net_vol int, sell_val int, buy_val int, net_val int)"
    )
    con.executemany(
        "insert into investor_daily values (?,?,'',?,0,0,0,0,0,?)",
        [(d, t, inv, v) for d, t, inv, v in rows],
    )
    con.commit()
    con.close()
    return db


def test_fill_zero_and_keep_real(tmp_path, monkeypatch):
    """0만 채우고 실값은 불변, 다른 주체 값에 안 덮인다."""
    db = _make_db(tmp_path, [
        ("20260803", "005930", "기타법인", 111),
        ("20260804", "005930", "기타법인", -222),
        ("20260805", "005930", "기타법인", 0),
        ("20260804", "005930", "연기금", 999),
    ])
    monkeypatch.setattr(epd, "INVESTOR_DB_PATH", db)

    idx = pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-04", "2026-08-05"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0],
                       "기타법인": [555.0, 0.0, 0.0, 0.0]}, index=idx)

    n = epd._fill_etc_corp(df, "005930", None)

    assert n == 2  # 8/3, 8/4만 (8/1은 실값, 8/5는 DB도 0)
    assert df.at[pd.Timestamp("2026-08-01"), "기타법인"] == 555.0
    assert df.at[pd.Timestamp("2026-08-03"), "기타법인"] == 111.0
    assert df.at[pd.Timestamp("2026-08-04"), "기타법인"] == -222.0
    assert df.at[pd.Timestamp("2026-08-05"), "기타법인"] == 0.0


def test_nan_is_filled(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [("20260804", "005930", "기타법인", 300)])
    monkeypatch.setattr(epd, "INVESTOR_DB_PATH", db)

    df = pd.DataFrame({"기타법인": [float("nan")]},
                      index=pd.to_datetime(["2026-08-04"]))
    assert epd._fill_etc_corp(df, "005930", None) == 1
    assert df.at[pd.Timestamp("2026-08-04"), "기타법인"] == 300.0


def test_recent_window_excludes_old(tmp_path, monkeypatch):
    """일일 증분 창(ETC_CORP_FILL_DAYS) 밖 과거는 안 건드린다 — 백필 모드 전용."""
    db = _make_db(tmp_path, [
        ("20260701", "005930", "기타법인", 100),
        ("20260804", "005930", "기타법인", 200),
    ])
    monkeypatch.setattr(epd, "INVESTOR_DB_PATH", db)

    idx = pd.to_datetime(["2026-07-01", "2026-08-04"])
    df = pd.DataFrame({"기타법인": [0.0, 0.0]}, index=idx)

    n = epd._fill_etc_corp(df, "005930", epd._etc_corp_start("20260806"))

    assert n == 1
    assert df.at[pd.Timestamp("2026-07-01"), "기타법인"] == 0.0  # 창 밖
    assert df.at[pd.Timestamp("2026-08-04"), "기타법인"] == 200.0


def test_missing_db_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(epd, "INVESTOR_DB_PATH", tmp_path / "없음.db")
    df = pd.DataFrame({"기타법인": [0.0]}, index=pd.to_datetime(["2026-08-04"]))
    assert epd._fill_etc_corp(df, "005930", None) == 0
    assert df.at[pd.Timestamp("2026-08-04"), "기타법인"] == 0.0


def test_missing_column_is_noop(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [("20260804", "005930", "기타법인", 300)])
    monkeypatch.setattr(epd, "INVESTOR_DB_PATH", db)
    df = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-08-04"]))
    assert epd._fill_etc_corp(df, "005930", None) == 0
