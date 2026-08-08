import json

from src.alpha import position_sizer_v2
from src.alpha.position_sizer_v2 import PositionSizerV2


TEST_CONFIG = {
    "backtest": {
        "max_risk_pct": 0.02,
        "max_single_position_pct": 0.40,
        "max_portfolio_risk_pct": 0.06,
        "trailing_stop_atr_mult": 1.5,
    },
    "alpha_v2": {
        "sizing": {
            "use_kelly": True,
            "kelly_default": 0.5,
            "kelly_fraction": 0.5,
            "kelly_min_samples": 30,
            "kelly_floor": 0.0,
            "kelly_cap": 1.0,
            "kelly_fallback_odds": 1.0,
        },
    },
}


def make_sizer() -> PositionSizerV2:
    sizer = PositionSizerV2(TEST_CONFIG)
    sizer._accuracy = {}
    return sizer


def test_load_accuracy_supports_current_signals_schema(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "market_learning"
    data_dir.mkdir(parents=True)
    (data_dir / "signal_accuracy.json").write_text(
        json.dumps(
            {
                "signals": {
                    "pullback_scan": {
                        "total": 40,
                        "hit_rate": 60.0,
                        "avg_win": 3.0,
                        "avg_loss": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(position_sizer_v2, "PROJECT_ROOT", tmp_path)

    sizer = PositionSizerV2(TEST_CONFIG)

    assert "pullback_scan" in sizer._accuracy


def test_fractional_kelly_uses_empirical_win_loss_ratio():
    sizer = make_sizer()
    sizer._accuracy = {
        "pullback_scan": {
            "total": 100,
            "hit_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": 1.0,
        },
    }

    result = sizer.calculate(
        account_balance=10_000_000,
        entry_price=50_000,
        atr_value=5_000,
        grade_ratio=1.0,
        signal_source="pullback_scan",
    )

    assert result["kelly_multiplier"] == 0.233
    assert result["shares"] == 6


def test_fractional_kelly_defaults_when_samples_are_insufficient():
    sizer = make_sizer()
    sizer._accuracy = {
        "pullback_scan": {
            "total": 10,
            "hit_rate": 90.0,
            "avg_win": 5.0,
            "avg_loss": 1.0,
        },
    }

    result = sizer.calculate(
        account_balance=10_000_000,
        entry_price=50_000,
        atr_value=5_000,
        grade_ratio=1.0,
        signal_source="pullback_scan",
    )

    assert result["kelly_multiplier"] == 0.5
    assert result["shares"] == 13


def test_negative_kelly_edge_reduces_position_to_zero():
    sizer = make_sizer()
    sizer._accuracy = {
        "weak_source": {
            "total": 100,
            "hit_rate": 40.0,
            "avg_win": 1.0,
            "avg_loss": 2.0,
        },
    }

    result = sizer.calculate(
        account_balance=10_000_000,
        entry_price=50_000,
        atr_value=5_000,
        grade_ratio=1.0,
        signal_source="weak_source",
    )

    assert result["kelly_multiplier"] == 0.0
    assert result["shares"] == 0
    assert result["investment"] == 0
