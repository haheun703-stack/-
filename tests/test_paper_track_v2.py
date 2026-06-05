"""paper_track v2 ledger schema regression tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import paper_track


def _sample_price_df() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=120)
    close = pd.Series([10000 + i * 20 for i in range(len(dates))], index=dates, dtype=float)
    close.iloc[-5:] = [11900, 11820, 11940, 12020, 12100]
    df = pd.DataFrame(index=dates)
    df["open"] = close * 0.995
    df["high"] = close * 1.02
    df["low"] = close * 0.98
    df["close"] = close
    df["volume"] = 100000
    df["sma_5"] = df["close"].rolling(5, min_periods=1).mean()
    df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["sma_60"] = df["close"].rolling(60, min_periods=1).mean()
    df["rsi_14"] = 55.0
    df["adx_14"] = 23.0
    df["외국인합계"] = 1000
    df["기관합계"] = 800
    df["개인"] = -1200
    df["기타법인"] = 100
    return df


def test_paper_track_enriches_legacy_ledger(tmp_path: Path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    _sample_price_df().to_parquet(processed / "123456.parquet")
    monkeypatch.setattr(paper_track, "PROCESSED", processed)

    ledger = tmp_path / "paper_ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "_note": "legacy test ledger",
                "paper_trades": [
                    {
                        "id": "PAPER-TEST-123456",
                        "ticker": "123456",
                        "name": "테스트",
                        "sector": "테스트섹터",
                        "entry_date": "2026-05-29",
                        "entry_price": 11820,
                        "qty": 10,
                        "capital_won": 118200,
                        "stop_loss_price": 11000,
                        "stop_loss_pct": -6.9,
                        "hold_target_days": 10,
                        "thesis": "legacy",
                        "status": "PAPER_OPEN",
                        "real_order": False,
                        "tracking": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert paper_track.run(ledger, write=True, check_only=False) == 0
    data = json.loads(ledger.read_text(encoding="utf-8"))
    trade = data["paper_trades"][0]

    assert data["_schema_version"] == paper_track.SCHEMA_VERSION
    assert trade["schema_version"] == paper_track.SCHEMA_VERSION
    assert trade["candidate"]["ticker"] == "123456"
    assert "weekly_LONG_ALLOWED" in trade["candidate"]
    assert "daily_pullback_support" in trade["candidate"]
    assert "risk_reward" in trade["candidate"]
    assert trade["candidate"]["stock_flow_4_actor"]["hard_gate_used"] is False
    assert trade["entry"]["t0_close_price"] > 0
    assert trade["entry"]["t1_open_price"] is not None
    assert trade["tracking"]
    latest = trade["tracking"][-1]
    assert "MFE_pct" in latest
    assert "MAE_pct" in latest
    assert "exit_check" in latest


def test_paper_track_records_candidate_log(tmp_path: Path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    df = _sample_price_df()
    df["rsi_14"] = 80.0
    df.to_parquet(processed / "123456.parquet")
    monkeypatch.setattr(paper_track, "PROCESSED", processed)

    ledger = tmp_path / "paper_ledger.json"
    ledger.write_text(
        json.dumps({"_note": "candidate test ledger", "paper_trades": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert paper_track.run(
        ledger,
        write=True,
        check_only=False,
        candidate_source="manual",
        candidate_tickers=["123456"],
        candidate_label="manual_test",
        candidate_note="회피 후보 기록 테스트",
        candidate_asof="2026-05-29",
    ) == 0

    data = json.loads(ledger.read_text(encoding="utf-8"))
    log = data["candidate_log"][0]
    candidate = log["candidates"][0]

    assert log["total"] == 1
    assert log["avoid_count"] == 1
    assert candidate["ticker"] == "123456"
    assert candidate["decision"] == "회피"
    assert "overheated" in candidate["reason"]
    assert candidate["real_order"] is False
