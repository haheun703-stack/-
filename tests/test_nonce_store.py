"""RISK_ENGINE C-ii — PersistentNonceSet 영속 nonce 저장소.

핵심 불변식:
  1. 같은 프로세스 add→contains 작동.
  2. ★재시작/교차 인스턴스: 새 인스턴스가 파일에서 기존 nonce를 본다(재부팅 replay 차단).
  3. 만료(retention 초과) nonce는 검사에서 제외(토큰 자체가 만료라 replay 불가).
  4. graceful: 파일 글리치에 전면 차단(crash) 없이 인메모리로 계속.
  5. verify_gate_token과 결합 시 인스턴스 경계를 넘어 replay 차단.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from risk.config import KST
from risk.nonce_store import PersistentNonceSet
from risk.pre_trade_gate import GateRequest, evaluate_pre_trade, verify_gate_token

_T0 = datetime(2026, 6, 12, 10, 0, 0, tzinfo=KST)


def _store(tmp_path: Path, now: datetime = _T0, retention: int = 360) -> PersistentNonceSet:
    return PersistentNonceSet(
        path=tmp_path / "nonces.log", retention_sec=retention, now_fn=lambda: now
    )


# ── 1. add + contains ────────────────────────────────────────────────────────
def test_add_then_contains(tmp_path):
    s = _store(tmp_path)
    assert "abc" not in s
    s.add("abc")
    assert "abc" in s


# ── 2. ★재시작/교차 인스턴스 — 새 인스턴스가 파일에서 본다 ──────────────────
def test_persists_across_instances(tmp_path):
    a = _store(tmp_path, now=_T0)
    a.add("nonce-1")
    # 새 인스턴스(재부팅 가정), 같은 파일, 10초 후
    b = _store(tmp_path, now=_T0 + timedelta(seconds=10))
    assert "nonce-1" in b  # 재시작 후에도 replay 차단


# ── 3. 만료 nonce는 검사 제외 ────────────────────────────────────────────────
def test_expired_nonce_pruned(tmp_path):
    a = _store(tmp_path, now=_T0)
    a.add("old")
    # retention(360s) 초과한 시점의 새 인스턴스 → 만료로 제외
    b = _store(tmp_path, now=_T0 + timedelta(seconds=400))
    assert "old" not in b


# ── 4. 빈 nonce 무시 ─────────────────────────────────────────────────────────
def test_empty_nonce_ignored(tmp_path):
    s = _store(tmp_path)
    s.add("")
    assert "" not in s


# ── 5. graceful — 디렉토리가 파일 위치를 막아도 crash 없이 인메모리 동작 ──────
def test_graceful_on_unwritable(tmp_path, caplog):
    # path 위치에 '디렉토리'를 만들어 파일 쓰기를 실패시킨다(OSError 유발).
    bad = tmp_path / "nonces.log"
    bad.mkdir()
    s = PersistentNonceSet(path=bad, now_fn=lambda: _T0)
    # add는 파일 기록 실패해도 예외를 던지지 않고 인메모리에는 남는다
    s.add("mem-only")
    assert "mem-only" in s  # 같은 프로세스 replay는 여전히 차단


# ── 6. compact — 만료행 누적 시 파일 크기 제한(살아있는 행만 남김) ───────────
def test_compaction_drops_expired(tmp_path):
    path = tmp_path / "nonces.log"
    # 만료될 오래된 행 600개를 T0에 직접 기록
    old_a = PersistentNonceSet(path=path, retention_sec=360, now_fn=lambda: _T0)
    for i in range(600):
        old_a.add(f"old-{i}")
    # 임계(_COMPACT_THRESHOLD=500) 초과 만료행 → 새 인스턴스 refresh 시 compact
    fresh = PersistentNonceSet(
        path=path, retention_sec=360, now_fn=lambda: _T0 + timedelta(seconds=400)
    )
    fresh.add("new")  # 살아있는 행 1개
    # compact 후 파일에는 만료된 600행이 빠지고 살아있는 행만 남는다
    remaining = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all("old-" not in l for l in remaining), "만료행이 compact로 제거돼야 함"


# ── 7. ★verify_gate_token 통합 — 인스턴스 경계 넘어 replay 차단 ──────────────
def test_verify_blocks_replay_across_instances(tmp_path):
    key = "nonce-store-test-key"
    req = GateRequest(ticker="487240", sector="ETF",
                      proposed_size_krw=100000, equity_krw=20_000_000, adv20_krw=10_000_000_000)
    tok = evaluate_pre_trade(req, [], log_dir=tmp_path / "gatelog", hmac_key=key)
    assert tok.verdict == "PASS" and tok.signed

    path = tmp_path / "nonces.log"
    s1 = PersistentNonceSet(path=path)  # 실시간(토큰 방금 발급 → 미만료)
    assert verify_gate_token(tok, hmac_key=key, seen_nonces=s1) is True   # 1회차 통과

    # 새 인스턴스(재시작 가정) — 같은 파일을 보므로 같은 토큰은 replay로 거부
    s2 = PersistentNonceSet(path=path)
    assert verify_gate_token(tok, hmac_key=key, seen_nonces=s2) is False  # 2회차 차단
