import pytest

from src.entities.forward_paper_event import PaperEventType, PaperMarket
from src.use_cases.forward_paper_market_mapper import map_index_benchmarks, map_wave_book


def test_wave_book_maps_kr_fill_exit_and_marks_without_certified_net_return():
    snapshot = {
        "trades": [
            {"date": "2026-08-01", "side": "BUY", "name": "KODEX200", "price": 100, "qty": 10},
            {"date": "2026-08-04", "side": "SELL", "name": "KODEX200", "price": 110, "qty": 10,
             "pnl_pct": 10, "reason": "REGIME"},
        ],
        "daily_equity": [{"date": "2026-08-04", "equity": 1_099_000, "regime": "BULL", "holding": "CASH"}],
    }
    events = map_wave_book(snapshot, market="KR", strategy_version="abc", source_name="wave.json")
    fill = next(event for event in events if event.event_type == PaperEventType.FILL)
    exit_event = next(event for event in events if event.event_type == PaperEventType.EXIT)
    mark = next(event for event in events if event.event_type == PaperEventType.MARK)
    assert fill.market == PaperMarket.KR and fill.currency == "KRW" and fill.fee == pytest.approx(0.5)
    assert exit_event.trade_id == fill.trade_id
    assert exit_event.net_return is None and exit_event.metadata["declared_fee_net_return"] > 0
    assert exit_event.metadata["cost_complete"] is False
    assert mark.benchmark_id == "KODEX200" and mark.exposure == 0.0


def test_wave_book_maps_us_timezone_and_rejects_unmatched_sell():
    events = map_wave_book(
        {"trades": [{"date": "2026-08-01", "side": "BUY", "name": "SPY", "price": 500, "qty": 2}]},
        market="US", strategy_version="abc", source_name="wave_us.json",
    )
    assert events[0].market == PaperMarket.US and events[0].currency == "USD"
    assert events[0].event_at.endswith("-04:00")
    with pytest.raises(ValueError, match="no matching BUY"):
        map_wave_book(
            {"trades": [{"date": "2026-08-01", "side": "SELL", "name": "SPY", "price": 500, "qty": 2}]},
            market="US", strategy_version="abc", source_name="wave_us.json",
        )


def test_index_benchmarks_keep_market_currency_and_normalized_return():
    snapshot = {
        "initial_capital": 100_000_000,
        "benchmarks": {
            "KODEX200": {"symbol": "069500.KS", "market": "KR", "entry_date": "2026-06-22",
                         "entry_px": 100, "daily": [{"date": "2026-06-23", "px": 102, "return_pct": 2, "equity": 102_000_000}]},
            "SPY": {"symbol": "SPY", "market": "US", "entry_date": "2026-06-22",
                    "entry_px": 500, "daily": [{"date": "2026-06-23", "px": 505, "return_pct": 1, "equity": 101_000_000}]},
        },
    }
    events = map_index_benchmarks(snapshot, strategy_version="abc", source_name="index.json")
    spy_mark = next(event for event in events if event.ticker == "SPY" and event.event_type == PaperEventType.MARK)
    kr_fill = next(event for event in events if event.ticker == "069500.KS" and event.event_type == PaperEventType.FILL)
    assert spy_mark.market == PaperMarket.US and spy_mark.currency == "USD"
    assert spy_mark.benchmark_return == pytest.approx(0.01)
    assert kr_fill.currency == "KRW" and kr_fill.metadata["cost_complete"] is False
