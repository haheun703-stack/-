"""Read-only preflight for Quantum Master trading safety.

Usage:
  python tools/quant_preflight.py --expect blocked

The check intentionally avoids broker calls and does not print secrets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_batch(path: Path) -> str:
    for encoding in ("utf-8", "cp949", "mbcs"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue
    return path.read_text(errors="replace")


def _status(label: str, ok: bool, detail: str) -> tuple[bool, str]:
    mark = "PASS" if ok else "FAIL"
    return ok, f"[{mark}] {label}: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        choices=("blocked", "live"),
        default="blocked",
        help="Expected runtime state.",
    )
    args = parser.parse_args()

    _load_env(PROJECT_ROOT / ".env")

    from src.utils.trade_runtime_safety import runtime_order_block_reasons

    data_dir = PROJECT_ROOT / "data"
    upper_kill = data_dir / "KILL_SWITCH"
    lower_kill = data_dir / "kill_switch.flag"
    schedule_e = PROJECT_ROOT / "scripts" / "schedule_E_smart_entry.bat"
    schedule_p = PROJECT_ROOT / "scripts" / "schedule_P_daytrading.bat"

    runtime_reasons = runtime_order_block_reasons()
    schedule_e_text = _read_batch(schedule_e) if schedule_e.exists() else ""
    schedule_p_text = _read_batch(schedule_p) if schedule_p.exists() else ""
    smart_entry_command_live = "scripts/smart_entry_runner.py --live" in schedule_e_text

    checks: list[tuple[bool, str]] = []
    checks.append(
        _status(
            "runtime order gate",
            bool(runtime_reasons) if args.expect == "blocked" else not runtime_reasons,
            ", ".join(runtime_reasons) if runtime_reasons else "no runtime blockers",
        )
    )
    checks.append(
        _status(
            "data/KILL_SWITCH",
            upper_kill.exists() if args.expect == "blocked" else not upper_kill.exists(),
            "exists" if upper_kill.exists() else "absent",
        )
    )
    checks.append(
        _status(
            "legacy kill_switch.flag",
            True,
            "exists" if lower_kill.exists() else "absent",
        )
    )
    checks.append(
        _status(
            "AUTO_TRADING_ENABLED",
            (os.getenv("AUTO_TRADING_ENABLED", "0") != "1")
            if args.expect == "blocked"
            else os.getenv("AUTO_TRADING_ENABLED", "0") == "1",
            os.getenv("AUTO_TRADING_ENABLED", "0"),
        )
    )
    checks.append(
        _status(
            "MODEL",
            os.getenv("MODEL", "") in {"REAL", "MOCK", "PAPER", ""},
            os.getenv("MODEL", ""),
        )
    )
    checks.append(
        _status(
            "QM-E schedule live command",
            not smart_entry_command_live if args.expect == "blocked" else smart_entry_command_live,
            "live" if smart_entry_command_live else "dry-run/no-live",
        )
    )
    checks.append(
        _status(
            "QM-P missing script guard",
            "signal_logger_daytrading.py missing" in schedule_p_text
            and "daily_close_daytrading.py missing" in schedule_p_text,
            "guarded" if "missing" in schedule_p_text else "unguarded",
        )
    )

    ok_all = all(ok for ok, _ in checks)
    print(f"Quantum preflight expect={args.expect}")
    for _, line in checks:
        print(line)
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
