"""test_signal_snapshot.py — 실시간 신호 SQLite snapshot 단위 테스트.

배경 (퐝가님 5/25 결단):
  5/26~5/28 3일 학습 모드 신규 모듈 검증.
  매 30분 cron snapshot → 5/29 결단 데이터 무결성 확보.

검증 시나리오:
  1. init_db + 멱등성 (재호출 안전)
  2. snapshot_signals 5종목 일괄 저장 + count 확인
  3. query_recent 시간 필터 (1시간 vs 24시간)
  4. summarize_by_ticker 통계 정확성
  5. 학습 모드 OFF 시 저장 0건

실행:
  python -m pytest tests/test_signal_snapshot.py -v
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.use_cases import signal_snapshot as ss


@pytest.fixture
def tmp_db(monkeypatch, tmp_path: Path):
    """임시 DB 경로 + 학습 모드 ON fixture."""
    db_path = tmp_path / "realtime_signals.db"
    monkeypatch.setattr(ss, "DB_PATH", db_path)
    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "1")
    return db_path


def _sample_candidates() -> list[dict]:
    """5종목 표본 (다양한 필드 패턴)."""
    return [
        {
            "ticker": "005930",
            "name": "삼성전자",
            "current_price": 71000,
            "signal_strength": 180.5,
            "volume_ratio": 1.8,
            "bullish_ratio": 0.62,
            "foreign_inst_buy": True,
            "time_slot": "MORNING",
            "c2_combo": {"ai_sector": True, "value_chain": False, "us_momentum": True, "intelligence": True},
            "peak_drop_pct": -3.2,
            "trailing_drop_pct": -1.5,
            "supply_outflow_days": 0,
            "in_queue": 0,
            "in_holdings": 1,
            "raw_data": {"note": "보유 중 모니터링"},
        },
        {
            "ticker": "000660",
            "name": "SK하이닉스",
            "current_price": 195000,
            "signal_strength": 220.0,
            "volume_ratio": 2.4,
            "bullish_ratio": 0.71,
            "foreign_inst_buy": True,
            "time_slot": "MORNING",
            "c2_combo": {"ai_sector": True, "value_chain": True},
            "peak_drop_pct": 0.0,
            "trailing_drop_pct": 0.0,
            "supply_outflow_days": 0,
            "in_queue": 0,
            "in_holdings": 0,
        },
        {
            "ticker": "067310",
            "name": "하나마이크론",
            "current_price": 38500,
            "signal_strength": 95.0,
            "volume_ratio": 0.8,
            "bullish_ratio": 0.40,
            "foreign_inst_buy": False,
            "time_slot": "NOON",
            "c2_combo": None,
            "peak_drop_pct": -8.5,
            "trailing_drop_pct": -4.2,
            "supply_outflow_days": 2,
            "in_queue": 1,
            "in_holdings": 0,
        },
        {
            "ticker": "012450",
            "name": "한화에어로스페이스",
            "current_price": 280000,
            "signal_strength": 155.3,
            "volume_ratio": 1.2,
            "bullish_ratio": 0.55,
            "foreign_inst_buy": True,
            "time_slot": "AFTERNOON",
            "c2_combo": {"intelligence": True},
            "peak_drop_pct": -1.0,
            "trailing_drop_pct": 0.0,
            "supply_outflow_days": 0,
            "in_queue": 0,
            "in_holdings": 0,
        },
        {
            "ticker": "247540",
            "name": "에코프로비엠",
            "current_price": 165000,
            # 일부 필드 누락 → 기본값 적용 검증
        },
    ]


# === 1. init_db 멱등성 ============================================

def test_init_db_idempotent(tmp_db: Path):
    """init_db 2회 호출 시 에러 없이 멱등."""
    ss.init_db()
    assert tmp_db.exists(), "DB 파일이 생성되어야 함"

    # 재호출 (테이블 + 인덱스 존재 상태)
    ss.init_db()

    with sqlite3.connect(str(tmp_db)) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_snapshots'"
        )
        assert cur.fetchone() is not None, "테이블 존재 확인"

        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name IN ('idx_snapshot_ticker_time', 'idx_snapshot_time')"
        )
        idx_names = {r[0] for r in cur.fetchall()}
        assert idx_names == {"idx_snapshot_ticker_time", "idx_snapshot_time"}, \
            f"두 인덱스 모두 존재해야 함: {idx_names}"


# === 2. snapshot_signals 5종목 일괄 저장 =====================================

def test_snapshot_signals_bulk_insert(tmp_db: Path):
    """5종목 일괄 저장 후 row 수 확인 + JSON 직렬화 검증."""
    ss.init_db()
    candidates = _sample_candidates()
    saved = ss.snapshot_signals(candidates, snapshot_at="2026-05-26T09:30:00")

    assert saved == 5, f"5건 저장 기대, 실제 {saved}"

    with sqlite3.connect(str(tmp_db)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT COUNT(*) AS n FROM signal_snapshots")
        assert cur.fetchone()["n"] == 5

        # JSON 직렬화 (c2_combo dict → str)
        cur = conn.execute(
            "SELECT c2_combo FROM signal_snapshots WHERE ticker='005930'"
        )
        row = cur.fetchone()
        c2 = json.loads(row["c2_combo"])
        assert c2["ai_sector"] is True
        assert c2["value_chain"] is False

        # foreign_inst_buy bool → int
        cur = conn.execute(
            "SELECT foreign_inst_buy FROM signal_snapshots WHERE ticker='067310'"
        )
        assert cur.fetchone()["foreign_inst_buy"] == 0

        # 누락 필드 기본값
        cur = conn.execute(
            "SELECT signal_strength, volume_ratio, supply_outflow_days "
            "FROM signal_snapshots WHERE ticker='247540'"
        )
        r = cur.fetchone()
        assert r["signal_strength"] == 0
        assert r["volume_ratio"] == 0
        assert r["supply_outflow_days"] == 0


# === 3. query_recent 시간 필터 =========================================

def test_query_recent_time_filter(tmp_db: Path):
    """1시간 / 24시간 윈도우 필터 동작 검증."""
    ss.init_db()
    now = datetime.now()

    # 3건: 30분 전 / 2시간 전 / 12시간 전
    times = [
        (now - timedelta(minutes=30)).isoformat(timespec="seconds"),
        (now - timedelta(hours=2)).isoformat(timespec="seconds"),
        (now - timedelta(hours=12)).isoformat(timespec="seconds"),
    ]
    for t in times:
        ss.snapshot_signals(
            [{"ticker": "005930", "name": "삼성전자", "current_price": 71000}],
            snapshot_at=t,
        )

    # 1시간 윈도우: 30분 전 1건만
    rows_1h = ss.query_recent(ticker="005930", hours=1)
    assert len(rows_1h) == 1, f"1시간 윈도우 기대 1건, 실제 {len(rows_1h)}"

    # 24시간 윈도우: 3건 모두
    rows_24h = ss.query_recent(ticker="005930", hours=24)
    assert len(rows_24h) == 3, f"24시간 윈도우 기대 3건, 실제 {len(rows_24h)}"

    # 정렬 (DESC) — 가장 최근(30분 전)이 첫 row
    assert rows_24h[0]["snapshot_at"] == times[0]
    assert rows_24h[-1]["snapshot_at"] == times[2]


# === 4. summarize_by_ticker 통계 정확성 =================================

def test_summarize_by_ticker(tmp_db: Path):
    """종목별 count + 평균 + 최신 시각 통계."""
    ss.init_db()
    now = datetime.now()

    # 005930: 3건 (sig 100, 200, 300 → 평균 200)
    base_iso = (now - timedelta(minutes=10)).isoformat(timespec="seconds")
    older_iso = (now - timedelta(minutes=40)).isoformat(timespec="seconds")
    oldest_iso = (now - timedelta(minutes=70)).isoformat(timespec="seconds")

    ss.snapshot_signals(
        [{"ticker": "005930", "name": "삼성전자", "current_price": 70000, "signal_strength": 300}],
        snapshot_at=base_iso,
    )
    ss.snapshot_signals(
        [{"ticker": "005930", "name": "삼성전자", "current_price": 69500, "signal_strength": 200}],
        snapshot_at=older_iso,
    )
    ss.snapshot_signals(
        [{"ticker": "005930", "name": "삼성전자", "current_price": 69000, "signal_strength": 100}],
        snapshot_at=oldest_iso,
    )
    # 000660: 1건
    ss.snapshot_signals(
        [{"ticker": "000660", "name": "SK하이닉스", "current_price": 200000, "signal_strength": 150}],
        snapshot_at=base_iso,
    )

    summary = ss.summarize_by_ticker(hours=24)

    assert "005930" in summary
    assert "000660" in summary
    assert summary["005930"]["count"] == 3
    assert summary["005930"]["avg_signal_strength"] == pytest.approx(200.0)
    assert summary["005930"]["latest_at"] == base_iso
    assert summary["005930"]["latest_price"] == 70000  # 가장 최신 row의 가격

    assert summary["000660"]["count"] == 1
    assert summary["000660"]["avg_signal_strength"] == pytest.approx(150.0)
    assert summary["000660"]["latest_price"] == 200000


# === 5. 학습 모드 OFF 시 no-op =============================================

def test_learning_mode_off_noop(monkeypatch, tmp_path: Path):
    """ADAPTIVE_DAILY_LEARNING_MODE != 1 → snapshot_signals 0 반환."""
    db_path = tmp_path / "realtime_signals.db"
    monkeypatch.setattr(ss, "DB_PATH", db_path)
    monkeypatch.delenv("ADAPTIVE_DAILY_LEARNING_MODE", raising=False)

    ss.init_db()  # init_db는 동작 (스키마 준비)
    saved = ss.snapshot_signals(_sample_candidates())
    assert saved == 0, f"학습 모드 OFF 시 0 반환 기대, 실제 {saved}"

    # 명시적 "0" 값도 OFF로 간주
    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "0")
    saved = ss.snapshot_signals(_sample_candidates())
    assert saved == 0, f"=0 시 0 반환 기대, 실제 {saved}"

    # DB가 비어있음 확인
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM signal_snapshots")
        assert cur.fetchone()[0] == 0
