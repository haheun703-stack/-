from __future__ import annotations

import inspect

import pandas as pd

from src.use_cases import half_year_leader_scanner as hls


def _ohlcv(rows: list[tuple]) -> pd.DataFrame:
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1000 for _ in rows],
        },
        index=idx,
    )


def _strong_df() -> pd.DataFrame:
    # H1(2026): 1/2 시가 70000 → 6/5 종가 95000(반기 시가 위·신고가 근처·20일 신고가)
    rows = [("2026-01-02", 70000, 71000, 69000, 70500)]
    price = 72000
    for d in range(1, 21):  # 5월 상승 구간(종가 ~91500까지, 전부 95000 미만)
        rows.append((f"2026-05-{d:02d}", price, price + 800, price - 300, price + 500))
        price += 1000
    # 마지막 봉: 20일 신고가 갱신·반기 신고가(close 95000 > 직전 모든 종가)
    rows.append(("2026-06-05", 93000, 95000, 92500, 95000))
    return _ohlcv(rows)


def _weak_df() -> pd.DataFrame:
    # 반기 시가 80000 아래로 마감(주도주 아님)
    return _ohlcv([
        ("2026-01-02", 80000, 80500, 79000, 80000),
        ("2026-06-05", 70000, 70500, 68000, 68000),
    ])


# ── compute_half_year_metrics ───────────────────────────────
def test_metrics_above_and_new_high() -> None:
    m = hls.compute_half_year_metrics(_strong_df(), kospi_half_ret=0.05)
    assert m["data_available"] is True
    assert m["above_half_year_open"] is True
    assert m["new_half_year_high_20d"] is True
    assert m["near_half_year_high"] is True
    assert m["rs_positive"] is True  # 종목 반기수익 >> KOSPI 5%


def test_metrics_below_half_open() -> None:
    m = hls.compute_half_year_metrics(_weak_df(), kospi_half_ret=0.05)
    assert m["above_half_year_open"] is False
    assert m["rs_positive"] is False


def test_metrics_empty_safe() -> None:
    assert hls.compute_half_year_metrics(pd.DataFrame())["data_available"] is False


# ── build_half_year_leader (점수/분류) ──────────────────────
def test_leader_core_full_score() -> None:
    m = hls.compute_half_year_metrics(_strong_df(), kospi_half_ret=0.05)
    rec = hls.build_half_year_leader("005930", "삼성전자", m, "반도체", sector_peer_sync_count=5)
    # 30+20+20+20+10 = 100
    assert rec["half_year_leader_score"] == 100
    assert rec["half_year_leader_grade"] == "HY_LEADER_CORE"


def test_leader_watch_without_sector_sync() -> None:
    m = hls.compute_half_year_metrics(_strong_df(), kospi_half_ret=0.05)
    # 섹터 동조 1개(자기뿐) → -20 → 80? 정확히 경계 확인: 30+20+20+0+10 = 80 → CORE
    rec = hls.build_half_year_leader("005930", "삼성", m, "반도체", sector_peer_sync_count=1)
    assert rec["half_year_leader_score"] == 80
    assert rec["half_year_leader_grade"] == "HY_LEADER_CORE"


def test_leader_not_leader_when_below() -> None:
    m = hls.compute_half_year_metrics(_weak_df(), kospi_half_ret=0.05)
    rec = hls.build_half_year_leader("000000", "약체", m, None, 0)
    assert rec["half_year_leader_grade"] in ("HY_NOT_LEADER", "HY_LEADER_WEAK")
    assert rec["half_year_leader_score"] < hls.WATCH_MIN


def test_leader_no_data_safe() -> None:
    rec = hls.build_half_year_leader("000000", "x", {"data_available": False}, None, 0)
    assert rec["data_available"] is False
    assert rec["half_year_leader_grade"] == "HY_NOT_LEADER"


# ── scan_half_year_leaders (섹터 동조 cross-sectional) ──────
def test_scan_sector_sync_counts_peers() -> None:
    items = [
        {"ticker": "A", "name": "A", "df": _strong_df(), "sector": "반도체"},
        {"ticker": "B", "name": "B", "df": _strong_df(), "sector": "반도체"},
        {"ticker": "C", "name": "C", "df": _strong_df(), "sector": "반도체"},
        {"ticker": "D", "name": "D", "df": _strong_df(), "sector": "반도체"},
        {"ticker": "E", "name": "E", "df": _weak_df(), "sector": "바이오"},
    ]
    records = hls.scan_half_year_leaders(items, kospi_df=None)
    by = {r["ticker"]: r for r in records}
    # 반도체 4개 모두 반기 시가 위 → 각자 동조 카운트 3(자기 제외) → +20 충족
    assert by["A"]["sector_peer_sync_count"] == 3
    assert by["A"]["half_year_leader_score"] >= hls.CORE_MIN - hls.PTS_RS_POSITIVE  # RS는 kospi None이라 0
    # 정렬: 점수 내림차순
    scores = [r["half_year_leader_score"] for r in records]
    assert scores == sorted(scores, reverse=True)


def test_leader_board_top_n() -> None:
    items = [{"ticker": f"T{i}", "name": f"T{i}", "df": _strong_df()} for i in range(25)]
    board = hls.build_leader_board(hls.scan_half_year_leaders(items), top_n=20)
    assert len(board) == 20


# ── loaders (실데이터 의존, 실패해도 None/빈dict) ───────────
def test_loaders_degrade_safely() -> None:
    # 존재하지 않는 경로 → None / 빈 dict
    from pathlib import Path
    assert hls.load_kospi_index(Path("nonexistent.csv")) is None
    assert hls.load_sector_map(Path("nonexistent.yaml")) == {}


# ── 안전선: 주문/매도/스케줄러 심볼 0 ───────────────────────
def test_source_has_no_order_or_sell_symbols() -> None:
    src = inspect.getsource(hls)
    forbidden = (
        "smart_sell", "SmartSellExecutor", "sell_brain", "owner_rule",
        "sell_market", "sell_limit", "order_intents_gate", "KisOrderAdapter",
        "place_order", "send_order", "create_market", "create_limit",
        "PAPER_OPEN", "scheduler", "run_adaptive_cycle",
    )
    for f in forbidden:
        assert f not in src, f"금지 심볼 발견: {f}"
