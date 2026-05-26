"""보호 종목 격리 시스템 단위 테스트 — 5/26 LS ELECTRIC 보호.

검증:
1. settings.yaml의 protected_tickers 로드
2. PROTECTED_TICKERS 환경변수 로드
3. scan_holdings_for_peaks가 보호 종목 제외
4. 후보 풀 보호 종목 제외
5. ticker zfill(6) 정규화
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.use_cases.adaptive_position_manager import (
    _load_protected_tickers,
    scan_holdings_for_peaks,
)


def test_settings_yaml_loads_protected(monkeypatch):
    """settings.yaml의 protected_tickers 로드."""
    monkeypatch.setenv("PROTECTED_TICKERS", "")
    protected = _load_protected_tickers()
    # 010120 LS ELECTRIC이 settings.yaml에 등록됨
    assert "010120" in protected


def test_env_overlay(monkeypatch):
    """PROTECTED_TICKERS env로 추가 등록."""
    monkeypatch.setenv("PROTECTED_TICKERS", "005930,000660")
    protected = _load_protected_tickers()
    assert "005930" in protected
    assert "000660" in protected
    # settings.yaml도 함께 로드
    assert "010120" in protected


def test_env_zfill_normalize(monkeypatch):
    """env의 5자리 ticker도 6자리로 정규화."""
    monkeypatch.setenv("PROTECTED_TICKERS", "5930")  # 4자리
    protected = _load_protected_tickers()
    assert "005930" in protected


def test_scan_excludes_protected(monkeypatch):
    """scan_holdings_for_peaks가 보호 종목 제외."""
    monkeypatch.setenv("PROTECTED_TICKERS", "")
    holdings = {
        "010120": {"quantity": 20, "avg_price": 283250},  # 보호
        "067310": {"quantity": 1, "avg_price": 30000},    # 일반
    }

    # detect_peak_signal mock — 호출 시 ticker별 PeakSignal 반환
    with patch("src.use_cases.adaptive_position_manager.detect_peak_signal") as mock_detect:
        mock_detect.return_value = MagicMock(
            trigger=False, pct_from_peak=0, ticker="X",
        )
        broker = MagicMock()
        results = scan_holdings_for_peaks(broker, holdings)

        # detect_peak_signal이 010120에 대해서는 호출되지 않아야 함
        called_tickers = [c.args[1] for c in mock_detect.call_args_list]
        assert "010120" not in called_tickers, "보호 종목 010120이 스캔됨!"
        assert "067310" in called_tickers, "일반 종목 067310이 누락됨"


def test_scan_protected_with_zfill(monkeypatch):
    """5자리 ticker로 보유 시에도 6자리 보호 매칭."""
    monkeypatch.setenv("PROTECTED_TICKERS", "")
    holdings = {
        "10120": {"quantity": 20, "avg_price": 283250},  # 5자리
    }
    with patch("src.use_cases.adaptive_position_manager.detect_peak_signal") as mock_detect:
        broker = MagicMock()
        results = scan_holdings_for_peaks(broker, holdings)
        # 5자리 10120 → 010120 normalize → 보호 → 스캔 X
        assert mock_detect.call_count == 0


def test_soubujang_pool_excludes_protected(monkeypatch):
    """후보 풀에 보호 종목 있어도 스캔 제외."""
    monkeypatch.setenv("PROTECTED_TICKERS", "")
    holdings = {}
    pool = {
        "010120": {"passed": True},  # 보호 + 통과
        "067310": {"passed": True},
        "005930": {"passed": False},  # passed=False
    }
    with patch("src.use_cases.adaptive_position_manager.detect_peak_signal") as mock_detect:
        mock_detect.return_value = MagicMock(trigger=False, pct_from_peak=0)
        broker = MagicMock()
        scan_holdings_for_peaks(broker, holdings, soubujang_pool=pool)
        called = [c.args[1] for c in mock_detect.call_args_list]
        assert "010120" not in called
        assert "067310" in called
        assert "005930" not in called  # passed=False


def test_no_protected_no_exclusion(monkeypatch):
    """보호 종목 0건이면 모든 보유 종목 스캔."""
    monkeypatch.setenv("PROTECTED_TICKERS", "")
    # settings.yaml의 010120은 항상 보호이므로 010120 없는 holdings로 검증
    holdings = {
        "067310": {"quantity": 1, "avg_price": 30000},
        "005930": {"quantity": 5, "avg_price": 70000},
    }
    with patch("src.use_cases.adaptive_position_manager.detect_peak_signal") as mock_detect:
        mock_detect.return_value = MagicMock(trigger=False, pct_from_peak=0)
        broker = MagicMock()
        scan_holdings_for_peaks(broker, holdings)
        called = [c.args[1] for c in mock_detect.call_args_list]
        assert "067310" in called
        assert "005930" in called
        assert len(called) == 2
