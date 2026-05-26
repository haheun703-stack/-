"""AI 동조 워치리스트 자동 추가 단위 테스트 (5/26)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.use_cases.ai_chain_auto_watchlist import (
    add_to_ai_chain_watchlist,
    load_ai_chain_watchlist,
    get_ai_chain_watchlist_tickers,
    format_added_for_telegram,
)


@pytest.fixture
def temp_watchlist(tmp_path, monkeypatch):
    """임시 워치리스트 파일."""
    p = tmp_path / "ai_chain_watchlist.json"
    monkeypatch.setattr("src.use_cases.ai_chain_auto_watchlist.WATCHLIST_PATH", p)
    return p


def test_empty_load(temp_watchlist):
    """파일 없으면 빈 dict."""
    d = load_ai_chain_watchlist()
    assert d["tickers"] == []


def test_add_new_tickers(temp_watchlist):
    """폭등 종목 신규 추가."""
    surge = [
        {"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
         "current_price": 250000, "change_pct": 16.9},
        {"ticker": "007810", "name": "코리아써키트", "sector": "AI반도체PCB",
         "current_price": 113000, "change_pct": 12.4},
    ]
    r = add_to_ai_chain_watchlist(surge)
    assert len(r["added"]) == 2
    assert r["total"] == 2
    # 파일 저장 확인
    d = load_ai_chain_watchlist()
    assert len(d["tickers"]) == 2
    assert d["tickers"][0]["ticker"] in ("095340", "007810")


def test_skip_protected(temp_watchlist):
    """보호 종목은 추가 X."""
    surge = [
        {"ticker": "010120", "name": "LS ELECTRIC", "sector": "AI산업소재",
         "current_price": 290000, "change_pct": 10.0},
        {"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
         "current_price": 250000, "change_pct": 16.9},
    ]
    r = add_to_ai_chain_watchlist(surge, protected_tickers={"010120"})
    assert len(r["added"]) == 1
    assert r["added"][0]["ticker"] == "095340"
    assert any(s["reason"] == "PROTECTED" for s in r["skipped"])


def test_skip_held(temp_watchlist):
    """보유 종목 스킵."""
    surge = [
        {"ticker": "067310", "name": "하나마이크론", "sector": "AI반도체",
         "current_price": 50000, "change_pct": 8.0},
    ]
    r = add_to_ai_chain_watchlist(surge, held_tickers={"067310"})
    assert len(r["added"]) == 0
    assert r["skipped"][0]["reason"] == "HELD"


def test_skip_already_listed(temp_watchlist):
    """이미 워치리스트에 있으면 중복 X."""
    surge1 = [{"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
               "current_price": 250000, "change_pct": 16.9}]
    add_to_ai_chain_watchlist(surge1)
    # 다시 추가
    r = add_to_ai_chain_watchlist(surge1)
    assert len(r["added"]) == 0
    assert r["skipped"][0]["reason"] == "ALREADY_LISTED"


def test_expiry_auto_removed(temp_watchlist):
    """만료된 항목 자동 제거."""
    # 만료된 데이터 미리 저장
    expired = {
        "tickers": [
            {"ticker": "095340", "name": "ISC",
             "added_at": "2026-04-01T10:00:00",
             "expires_at": "2026-04-08T10:00:00",  # 이미 지남
             "trigger_reason": "old"},
        ]
    }
    temp_watchlist.write_text(json.dumps(expired), encoding="utf-8")

    d = load_ai_chain_watchlist()
    # 만료로 제거
    assert len(d["tickers"]) == 0


def test_ticker_zfill_normalize(temp_watchlist):
    """5자리 ticker → 6자리 정규화."""
    surge = [{"ticker": "67310", "name": "하나마이크론", "sector": "x",
              "current_price": 50000, "change_pct": 10.0}]
    r = add_to_ai_chain_watchlist(surge)
    assert r["added"][0]["ticker"] == "067310"


def test_get_active_tickers(temp_watchlist):
    """활성 ticker만 반환."""
    surge = [
        {"ticker": "095340", "name": "ISC", "sector": "x",
         "current_price": 250000, "change_pct": 16.9},
        {"ticker": "007810", "name": "코리아써키트", "sector": "y",
         "current_price": 113000, "change_pct": 12.4},
    ]
    add_to_ai_chain_watchlist(surge)
    tks = get_ai_chain_watchlist_tickers()
    assert "095340" in tks
    assert "007810" in tks
    assert len(tks) == 2


def test_format_telegram(temp_watchlist):
    """텔레그램 포맷에 종목명/섹터/사유 포함."""
    result = {
        "added": [
            {"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
             "trigger_reason": "AI 동조 발화 (+16.9% — AI반도체검사)"},
        ],
        "skipped": [],
        "total": 1,
    }
    msg = format_added_for_telegram(result)
    assert "ISC" in msg
    assert "AI반도체검사" in msg
    assert "16.9" in msg


def test_format_telegram_empty():
    """추가 없음 → 짧은 메시지."""
    msg = format_added_for_telegram({"added": [], "skipped": [], "total": 0})
    assert "추가 없음" in msg
