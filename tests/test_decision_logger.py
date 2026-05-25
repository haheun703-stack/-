"""decision_logger 단위 테스트 — 5/26~5/28 3일 학습 모드."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_learning_mode_off_no_save(monkeypatch, tmp_path):
    """학습 모드 OFF (env 미설정) → 저장 0건."""
    from src.use_cases import decision_logger as dl

    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "0")
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path / "decision_log")

    saved = dl.log_decision("BUY", "110990", name="디아이티", current_price=27000, qty=1)
    assert saved is False
    assert not (tmp_path / "decision_log").exists()


def test_learning_mode_on_saves_jsonl(monkeypatch, tmp_path):
    """학습 모드 ON → JSON Lines 저장."""
    from src.use_cases import decision_logger as dl

    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "1")
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path / "decision_log")

    saved = dl.log_decision(
        "BUY", "240810", name="원익IPS",
        current_price=121700, qty=1, amount=121700, target_price=136350,
        signal_strength=169.08, volume_ratio=3.2, bullish_ratio=0.65,
        foreign_inst_buy=True,
        c2_combo={"ai_sector": True, "us_momentum": False},
        peak_drop_pct=-19.7,
        pass_reasons=["천장 -3% 진입", "체결강도 169"],
    )
    assert saved is True

    # 파일 존재 + JSON Lines 1줄
    log_dir = tmp_path / "decision_log"
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "BUY"
    assert record["ticker"] == "240810"
    assert record["name"] == "원익IPS"
    assert record["signals"]["S1_signal_strength"] == 169.08
    assert record["signals"]["S4_foreign_inst_buy"] is True
    assert record["signals"]["S6_c2_combo"] == {"ai_sector": True, "us_momentum": False}


def test_query_today_filter(monkeypatch, tmp_path):
    """type / ticker 필터 동작."""
    from src.use_cases import decision_logger as dl

    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "1")
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path / "decision_log")

    dl.log_decision("BUY", "110990", name="디아이티", current_price=27000)
    dl.log_decision("BUY", "240810", name="원익IPS", current_price=121700)
    dl.log_decision("SELL", "110990", name="디아이티", current_price=29000)
    dl.log_decision("SKIP", "095340", name="ISC", skip_reason="L1 미도달")

    all_today = dl.query_today()
    assert len(all_today) == 4

    buys = dl.query_today(decision_type="BUY")
    assert len(buys) == 2

    dait = dl.query_today(ticker="110990")
    assert len(dait) == 2
    assert dait[0]["ticker"] == "110990"


def test_summarize_today(monkeypatch, tmp_path):
    """summarize_today 통계 정확성."""
    from src.use_cases import decision_logger as dl

    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "1")
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path / "decision_log")

    dl.log_decision("BUY", "110990", name="디아이티")
    dl.log_decision("BUY", "240810", name="원익IPS")
    dl.log_decision("SELL", "110990", name="디아이티")
    dl.log_decision("SKIP", "095340", name="ISC", skip_reason="천장 미진입")

    summary = dl.summarize_today()
    assert summary["total"] == 4
    assert summary["by_type"]["BUY"] == 2
    assert summary["by_type"]["SELL"] == 1
    assert summary["by_type"]["SKIP"] == 1
    assert summary["by_ticker"]["110990"] == 2
    assert summary["buy_count"] == 2
    assert summary["sell_count"] == 1
    assert summary["skip_count"] == 1


def test_time_slot_labels():
    """_time_slot 시간대 라벨 (현재 시각 기준 OK 케이스)."""
    from src.use_cases.decision_logger import _time_slot

    slot = _time_slot()
    valid_slots = {"PRE_OPEN", "MORNING", "NOON", "LUNCH", "AFTERNOON", "CLOSE", "AFTER_CLOSE"}
    assert slot in valid_slots


def test_append_multiple_calls(monkeypatch, tmp_path):
    """동일 일자 append (10 records → 10 lines)."""
    from src.use_cases import decision_logger as dl

    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "1")
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path / "decision_log")

    for i in range(10):
        dl.log_decision(
            "QUEUE_TRIGGER" if i % 2 else "ALERT",
            f"00000{i}",
            name=f"종목{i}",
            current_price=10_000 + i * 100,
        )

    files = list((tmp_path / "decision_log").glob("*.json"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10


def test_korean_round_trip(monkeypatch, tmp_path):
    """한글 종목명 round-trip 보존."""
    from src.use_cases import decision_logger as dl

    monkeypatch.setenv("ADAPTIVE_DAILY_LEARNING_MODE", "1")
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path / "decision_log")

    dl.log_decision(
        "ALERT", "319400",
        name="현대무벡스",
        pass_reasons=["받침 패턴 ★★★", "외인+기관 동시 매수"],
        extra={"테마": "물류로봇", "비고": "한자 漢字 도 포함"},
    )

    records = dl.query_today()
    assert len(records) == 1
    assert records[0]["name"] == "현대무벡스"
    assert "받침 패턴 ★★★" in records[0]["pass_reasons"]
    assert records[0]["extra"]["테마"] == "물류로봇"
    assert records[0]["extra"]["비고"] == "한자 漢字 도 포함"
