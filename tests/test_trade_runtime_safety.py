from unittest.mock import MagicMock

import pytest


def test_runtime_guard_blocks_when_kill_switch_file_exists(tmp_path, monkeypatch):
    from src.utils import trade_runtime_safety as safety

    kill_switch = tmp_path / "KILL_SWITCH"
    kill_switch.write_text("test", encoding="utf-8")
    monkeypatch.setattr(safety, "KILL_SWITCH_PATHS", (kill_switch,))

    reasons = safety.runtime_order_block_reasons()
    assert "KILL_SWITCH exists" in reasons

    with pytest.raises(PermissionError, match="KILL_SWITCH exists"):
        safety.assert_runtime_orders_allowed()


def test_runtime_guard_blocks_when_paper_only_env_is_true(tmp_path, monkeypatch):
    from src.utils import trade_runtime_safety as safety

    monkeypatch.setattr(safety, "KILL_SWITCH_PATHS", (tmp_path / "missing",))
    monkeypatch.setenv("PAPER_ONLY", "true")

    with pytest.raises(PermissionError, match="PAPER_ONLY=true"):
        safety.assert_runtime_orders_allowed()


def test_kis_order_guard_uses_runtime_gate(tmp_path, monkeypatch):
    from src.utils import trade_runtime_safety as safety

    kill_switch = tmp_path / "KILL_SWITCH"
    kill_switch.write_text("test", encoding="utf-8")
    monkeypatch.setattr(safety, "KILL_SWITCH_PATHS", (kill_switch,))
    monkeypatch.setenv("MODEL", "REAL")
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "1")
    monkeypatch.setattr("mojito.KoreaInvestment", MagicMock())

    from src.adapters.kis_order_adapter import KisOrderAdapter

    adapter = KisOrderAdapter()
    with pytest.raises(PermissionError, match="RUNTIME-GUARD"):
        adapter._guard("005930", 1, price=70000, side="BUY")


def test_runtime_guard_allows_when_no_blocker(tmp_path, monkeypatch):
    from src.utils import trade_runtime_safety as safety

    monkeypatch.setattr(safety, "KILL_SWITCH_PATHS", (tmp_path / "missing",))
    for name in (
        "QUANT_AUTO_TRADE_DISABLED",
        "AUTO_TRADE_DISABLED",
        "AUTO_TRADING_DISABLED",
        "PAPER_ONLY",
        "QUANT_PAPER_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert safety.runtime_order_block_reasons() == []
    safety.assert_runtime_orders_allowed()
