"""확신모델 역전 B안 — 과매집(crowding) 감점 검증.

설계: conviction-reversal-redesign_2026-06-18 §3·§4.1 / §6 사장님 승인 6/22.
배경: paper 63건 해부 — STRONG_ALPHA 쌍끌이 2종이 "연속 매수(과매집)" 길이를 무시하고
      AA를 강제 → 천장 진입. B안은 연속 ≥STRONG_CROWDING_DAYS면 AA 승격을 차단(→A).

검증:
  1. A안(현행): demote_if_crowded는 무조건 no-op → 현행 거동 100% 보존(freeze 무손상 핵심).
  2. B안: AA가 과매집(연속 ≥6일)이면 A로 강등, 초입(<6)은 AA 유지.
  3. 경계값(5/6). AA 외 등급 미적용. graceful(parquet 결손 → 0 → 미적용).
  4. PORTFOLIO_PATH 격리(A=paper_portfolio.json / B=paper_portfolio_b.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import paper_trading_unified as pt  # noqa: E402


# ── 1. A안: 무조건 no-op (현행 무손상) ─────────────────────────────────
def test_a_mode_is_noop(monkeypatch):
    monkeypatch.setattr(pt, "CONVICTION_MODE", "A")
    monkeypatch.setattr(pt, "crowding_streak", lambda t: 99)  # 극단 과매집이어도
    assert pt.demote_if_crowded("AA", "005930") == "AA"


# ── 2. B안: 과매집 차단 / 초입 유지 ───────────────────────────────────
def test_b_mode_blocks_crowded(monkeypatch):
    monkeypatch.setattr(pt, "CONVICTION_MODE", "B")
    monkeypatch.setattr(pt, "crowding_streak", lambda t: 7)
    assert pt.demote_if_crowded("AA", "005930") == "A"


def test_b_mode_keeps_fresh(monkeypatch):
    monkeypatch.setattr(pt, "CONVICTION_MODE", "B")
    monkeypatch.setattr(pt, "crowding_streak", lambda t: 2)
    assert pt.demote_if_crowded("AA", "005930") == "AA"


# ── 3. 경계값 (STRONG_CROWDING_DAYS=6) ────────────────────────────────
def test_b_mode_boundary(monkeypatch):
    monkeypatch.setattr(pt, "CONVICTION_MODE", "B")
    monkeypatch.setattr(pt, "crowding_streak", lambda t: pt.STRONG_CROWDING_DAYS)
    assert pt.demote_if_crowded("AA", "x") == "A"       # 6 = 차단
    monkeypatch.setattr(pt, "crowding_streak", lambda t: pt.STRONG_CROWDING_DAYS - 1)
    assert pt.demote_if_crowded("AA", "x") == "AA"      # 5 = 유지


# ── 4. AA 외 등급은 미적용 ────────────────────────────────────────────
def test_b_mode_only_affects_aa(monkeypatch):
    monkeypatch.setattr(pt, "CONVICTION_MODE", "B")
    monkeypatch.setattr(pt, "crowding_streak", lambda t: 9)
    assert pt.demote_if_crowded("A", "x") == "A"
    assert pt.demote_if_crowded("B", "x") == "B"


# ── 5. crowding_streak graceful (parquet 부재 → 0) ────────────────────
def test_crowding_streak_missing_parquet():
    assert pt.crowding_streak("__nonexistent_000000__") == 0


# ── 6. 포트폴리오 파일 격리 (현 프로세스 = 환경변수 미설정 = A안) ──────
def test_portfolio_path_isolated_by_mode():
    assert pt.CONVICTION_MODE in ("A", "B")
    if pt.CONVICTION_MODE == "B":
        assert pt.PORTFOLIO_PATH.name == "paper_portfolio_b.json"
    else:
        assert pt.PORTFOLIO_PATH.name == "paper_portfolio.json"
