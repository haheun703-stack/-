from __future__ import annotations

from src.entities.forward_paper_event import PaperEventType
from src.use_cases.forward_paper_portfolio_mapper import map_legacy_portfolio


def _snapshot() -> dict:
    return {
        "positions": {
            "005930": {
                "ticker": "005930", "name": "삼성전자", "entry_date": "2026-08-07",
                "avg_price": 70000, "qty": 2, "strategy": "AI_BRAIN", "grade": "A",
            }
        },
        "closed_trades": [{
            "ticker": "000660", "name": "SK하이닉스", "entry_date": "2026-08-01",
            "exit_date": "2026-08-08", "avg_price": 250000, "exit_price": 260000,
            "qty": 1, "strategy": "SCAN", "pnl_pct": 4.0, "exit_reason": "TARGET",
        }],
        "daily_equity": [{
            "date": "2026-08-08", "equity": 30_100_000, "capital": 20_000_000,
            "positions": 2, "stock_ratio": 33.6,
        }],
    }


def test_mapper_emits_fill_exit_and_mark_without_claiming_net_return() -> None:
    events = map_legacy_portfolio(
        _snapshot(), portfolio_id="paper_main_a",
        strategy_version="abc123", source_name="paper_portfolio.json",
    )
    types = [event.event_type for event in events]
    assert types.count(PaperEventType.FILL) == 2
    assert types.count(PaperEventType.EXIT) == 1
    assert types.count(PaperEventType.MARK) == 1

    exit_event = next(event for event in events if event.event_type == PaperEventType.EXIT)
    assert exit_event.net_return is None
    assert exit_event.metadata["source_pnl_pct"] == 4.0
    assert exit_event.metadata["cost_complete"] is False
    assert all(event.real_order is False for event in events)


def test_mapper_is_deterministic_and_does_not_duplicate_closed_fill() -> None:
    snapshot = _snapshot()
    snapshot["positions"]["000660"] = {
        "ticker": "000660", "name": "SK하이닉스", "entry_date": "2026-08-01",
        "avg_price": 250000, "qty": 1, "strategy": "SCAN",
    }
    first = map_legacy_portfolio(
        snapshot, portfolio_id="paper_main_a",
        strategy_version="abc123", source_name="paper_portfolio.json",
    )
    second = map_legacy_portfolio(
        snapshot, portfolio_id="paper_main_a",
        strategy_version="abc123", source_name="paper_portfolio.json",
    )
    assert [event.event_id for event in first] == [event.event_id for event in second]
    fills = [event for event in first if event.event_type == PaperEventType.FILL]
    assert len(fills) == 2
    hynix_fill = next(event for event in fills if event.ticker == "000660")
    assert hynix_fill.filled_quantity == 2
    assert hynix_fill.metadata["quantity_reconstructed_from_snapshot"] is True


def test_mapper_rejects_invalid_trade_identity() -> None:
    snapshot = {"positions": {"bad": {"ticker": "005930", "qty": 1}}}
    try:
        map_legacy_portfolio(
            snapshot, portfolio_id="paper_main_a",
            strategy_version="abc123", source_name="paper_portfolio.json",
        )
    except ValueError as exc:
        assert "invalid legacy paper trade identity" in str(exc)
    else:
        raise AssertionError("invalid position must be rejected")


def test_mapper_accepts_holdnav_entry_px_contract() -> None:
    snapshot = {
        "positions": {
            "003550": {
                "name": "LG",
                "ticker": "003550",
                "entry_date": "2026-08-06",
                "entry_px": 104500,
                "qty": 239,
                "strategy": "HOLDING_NAV",
            }
        }
    }

    events = map_legacy_portfolio(
        snapshot,
        portfolio_id="paper_holding_nav",
        strategy_version="test",
        source_name="paper_portfolio_holdnav.json",
    )

    assert len(events) == 1
    assert events[0].fill_price == 104500
    assert events[0].filled_quantity == 239
