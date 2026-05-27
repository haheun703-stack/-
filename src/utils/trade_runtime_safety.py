"""Runtime safety gate for live trading mutations.

This module is intentionally small and dependency-free so every live order
adapter can call it right before broker-side mutations.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KILL_SWITCH_PATHS = (
    PROJECT_ROOT / "data" / "KILL_SWITCH",
    PROJECT_ROOT / "data" / "kill_switch.flag",
)

TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def runtime_order_block_reasons() -> list[str]:
    """Return reasons live broker mutations must be blocked."""
    reasons: list[str] = []

    for name in (
        "QUANT_AUTO_TRADE_DISABLED",
        "AUTO_TRADE_DISABLED",
        "AUTO_TRADING_DISABLED",
        "PAPER_ONLY",
        "QUANT_PAPER_ONLY",
    ):
        if _env_true(name):
            reasons.append(f"{name}=true")

    for path in KILL_SWITCH_PATHS:
        if path.exists():
            reasons.append(f"{path.name} exists")

    return reasons


def live_orders_blocked() -> bool:
    return bool(runtime_order_block_reasons())


def assert_runtime_orders_allowed() -> None:
    """Raise when runtime controls require live orders to stay blocked."""
    reasons = runtime_order_block_reasons()
    if reasons:
        raise PermissionError(
            "[RUNTIME-GUARD] live broker order blocked: "
            + "; ".join(reasons)
        )
