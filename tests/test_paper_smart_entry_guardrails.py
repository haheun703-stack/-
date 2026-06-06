from __future__ import annotations

from scripts import paper_smart_entry as pse


def _ledger() -> dict:
    return {
        "candidate_log": [
            {
                "as_of_date": "2026-06-04",
                "source": "unit",
                "candidates": [
                    {
                        "ticker": "000001",
                        "name": "OLD",
                        "decision": "진입",
                        "date": "2026-06-04",
                        "floor_quality": {"label": "바닥다지기후보", "floor_quality_score": 1},
                        "market_context": {"drop_context": "normal"},
                        "supply_confirmation": {"supply_state": "foreign_accumulation"},
                        "risk_reward": {"entry_price": 1000},
                    }
                ],
            },
            {
                "as_of_date": "2026-06-05",
                "source": "unit",
                "candidates": [
                    {
                        "ticker": "000001",
                        "name": "NEW",
                        "decision": "진입",
                        "date": "2026-06-05",
                        "floor_quality": {"label": "바닥다지기후보", "floor_quality_score": 3},
                        "market_context": {"drop_context": "normal"},
                        "supply_confirmation": {"supply_state": "foreign_accumulation"},
                        "risk_reward": {"entry_price": 1100},
                    },
                    {
                        "ticker": "000002",
                        "name": "CTRL",
                        "decision": "진입",
                        "date": "2026-06-05",
                        "floor_quality": {"label": "관찰(위험)", "floor_quality_score": -1},
                        "market_context": {"drop_context": "stock_specific_drop"},
                        "supply_confirmation": {"supply_state": "distribution_warning"},
                        "risk_reward": {"entry_price": 900},
                    },
                ],
            },
        ]
    }


def test_candidate_map_uses_latest_as_of_only() -> None:
    cand_map = pse._candidate_feature_map(_ledger())

    assert cand_map["000001"]["name"] == "NEW"
    assert cand_map["000001"]["_log_as_of_date"] == "2026-06-05"


def test_record_control_pool_dedupes_by_candidate_as_of(monkeypatch) -> None:
    ledger = _ledger()
    control = [{
        "ticker": "000002",
        "name": "CTRL",
        "close": 900,
        "_floor_label": "관찰(위험)",
        "_drop_context": "stock_specific_drop",
        "_supply_state": "distribution_warning",
        "_as_of_date": "2026-06-05",
    }]

    class FakeLedgerPath:
        def write_text(self, text, encoding="utf-8") -> None:
            return None

    monkeypatch.setattr(pse, "LEDGER", FakeLedgerPath())
    added = pse.record_control_pool(control, ledger)
    added_again = pse.record_control_pool(control, ledger)

    assert added == 1
    assert added_again == 0
    assert ledger["shadow_control"][0]["date"] == "2026-06-05"
    assert ledger["shadow_control"][0]["as_of_date"] == "2026-06-05"


def test_record_entries_dedupes_same_day(monkeypatch) -> None:
    ledger = _ledger()
    writes = []

    class FakeLedgerPath:
        def write_text(self, text, encoding="utf-8") -> None:
            writes.append(text)

    monkeypatch.setattr(pse, "LEDGER", FakeLedgerPath())

    report = {
        "details": [
            {
                "ticker": "000001",
                "name": "NEW",
                "decision": "buy",
                "order_price": 1110,
                "trigger_time": "2026-06-08T09:30:00",
            }
        ]
    }

    opened = pse.record_entries(report, ledger, paper_open=False)
    opened_again = pse.record_entries(report, ledger, paper_open=False)

    assert opened == 1
    assert opened_again == 0
    row = ledger["shadow_observations"][0]
    assert row["ticker"] == "000001"
    assert row["status"] == "SHADOW_OPEN"
    assert row["candidate_as_of_date"] == "2026-06-05"
    assert len(writes) == 1
