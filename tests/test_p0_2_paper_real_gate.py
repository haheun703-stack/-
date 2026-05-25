"""P0-2 회귀 테스트 — REAL 모드 3중 안전 게이트.

배경 (5/24 bkit:code-analyzer 검수 P0-2):
  "paper 모드 실거래 가능 — run_adaptive_cycle.py:191-197 mock=True여도 ACC_NO가 실계좌면 위험"

수정 (5/25):
  scripts/run_adaptive_cycle.py:
    - validate_real_mode_gate() 헬퍼 신설
    - REAL 모드 진입 전 게이트 통과 못하면 자동 PAPER fallback + 텔레그램 경고

게이트:
  1) AUTO_TRADING_ENABLED=1 환경변수 명시
  2) KIS_ACC_NO 형식 검증 (- 제거 후 최소 8자 숫자)
  (CLI 인자 --real은 호출 측 검증)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# === validate_real_mode_gate 단위 테스트 ===

def test_gate_auto_trading_disabled(monkeypatch):
    """AUTO_TRADING_ENABLED 미설정 → 차단."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.delenv("AUTO_TRADING_ENABLED", raising=False)
    monkeypatch.setenv("KIS_ACC_NO", "47339014-01")

    passed, reason = validate_real_mode_gate()
    assert passed is False
    assert "AUTO_TRADING_ENABLED" in reason


def test_gate_auto_trading_zero(monkeypatch):
    """AUTO_TRADING_ENABLED=0 → 차단."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "0")
    monkeypatch.setenv("KIS_ACC_NO", "47339014-01")

    passed, reason = validate_real_mode_gate()
    assert passed is False
    assert "AUTO_TRADING_ENABLED" in reason


def test_gate_acc_no_missing(monkeypatch):
    """KIS_ACC_NO 미설정 → 차단."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.delenv("KIS_ACC_NO", raising=False)

    passed, reason = validate_real_mode_gate()
    assert passed is False
    assert "KIS_ACC_NO" in reason


def test_gate_acc_no_too_short(monkeypatch):
    """KIS_ACC_NO 8자 미만 → 차단."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("KIS_ACC_NO", "1234567")  # 7자

    passed, reason = validate_real_mode_gate()
    assert passed is False
    assert "KIS_ACC_NO" in reason


def test_gate_acc_no_non_numeric(monkeypatch):
    """KIS_ACC_NO 영문자 포함 → 차단."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("KIS_ACC_NO", "ABCD1234-01")

    passed, reason = validate_real_mode_gate()
    assert passed is False
    assert "KIS_ACC_NO" in reason


def test_gate_all_pass(monkeypatch):
    """3중 게이트 모두 통과 → REAL 모드 허용."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("KIS_ACC_NO", "47339014-01")

    passed, reason = validate_real_mode_gate()
    assert passed is True
    assert reason == ""


def test_gate_all_pass_no_dash(monkeypatch):
    """ACC_NO 대시 없어도 통과."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("KIS_ACC_NO", "4733901401")

    passed, reason = validate_real_mode_gate()
    assert passed is True


def test_gate_acc_no_8_digits_exact(monkeypatch):
    """ACC_NO 정확히 8자리 → 통과."""
    from scripts.run_adaptive_cycle import validate_real_mode_gate

    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setenv("KIS_ACC_NO", "12345678")

    passed, reason = validate_real_mode_gate()
    assert passed is True
