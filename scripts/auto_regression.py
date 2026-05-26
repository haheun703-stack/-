"""매일 자동 회귀 — 옵션 A (5/27 신규).

배경: 코드 변경 시 회귀 자동 검증 (매일 16:00 cron).
실패 시 텔레그램 즉시 알림 + 핵심 모듈 fail 매트릭스.

실행:
  PYTHONPATH=. ./venv/bin/python3.11 scripts/auto_regression.py

cron:
  0 16 * * 1-5 — 평일 16:00 (장 마감 30분 후)
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 핵심 모듈 (5/26 신규 + 통합)
CORE_TESTS = [
    "tests/test_adaptive_buy_queue.py",
    "tests/test_adaptive_entry_gates.py",
    "tests/test_adaptive_position_manager.py",
    "tests/test_adaptive_quick_profit.py",
    "tests/test_adaptive_time_exit.py",
    "tests/test_adaptive_trend_exit.py",
    "tests/test_ai_chain_auto_watchlist.py",
    "tests/test_ai_chain_detector.py",
    "tests/test_ai_chain_queue_auto_register.py",
    "tests/test_atr_dynamic_stop.py",
    "tests/test_gap_volatility_guard.py",
    "tests/test_momentum_chase.py",
    "tests/test_opening_call_gate.py",
    "tests/test_orderbook_gate.py",
    "tests/test_protected_tickers.py",
    "tests/test_supply_flow_gate.py",
    "tests/test_supply_zone_gate.py",
    "tests/test_vwap_gate.py",
    "tests/test_v8_gates.py",
    "tests/test_v8_pipeline.py",
    "tests/test_v8_triggers.py",
]


def run_pytest() -> tuple[bool, int, int, str]:
    """pytest 실행 → (성공/실패, passed, failed, output)."""
    cmd = [sys.executable, "-m", "pytest"] + CORE_TESTS + ["-q", "--tb=line", "--no-header"]
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", timeout=300,
        )
        output = result.stdout + result.stderr
        # passed/failed 파싱
        passed = failed = 0
        for line in output.split("\n"):
            if "passed" in line and "failed" not in line:
                # "281 passed in 2.64s"
                parts = line.strip().split()
                for p in parts:
                    if p.isdigit():
                        passed = int(p)
                        break
            elif "failed" in line:
                # "1 failed, 280 passed in 2.64s"
                import re
                m = re.search(r"(\d+) failed.*?(\d+) passed", line)
                if m:
                    failed = int(m.group(1))
                    passed = int(m.group(2))
        return result.returncode == 0, passed, failed, output[-3000:]
    except subprocess.TimeoutExpired:
        return False, 0, 0, "TIMEOUT (300s)"
    except Exception as e:
        return False, 0, 0, f"EXCEPTION: {e}"


def send_telegram_report(success: bool, passed: int, failed: int, output: str):
    """결과 텔레그램 발송."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.telegram_sender import send_message

        emoji = "✅" if success else "🚨"
        title = f"{emoji} [매일 자동 회귀] {datetime.now():%Y-%m-%d %H:%M}"

        if success:
            msg = (
                f"{title}\n"
                f"PASS: {passed}건 / FAIL: {failed}건\n"
                f"5/27~5/29 시스템 안정 ✓"
            )
        else:
            # 실패 시 핵심 fail만 표시
            fail_lines = [l for l in output.split("\n") if "FAILED" in l][:5]
            msg = (
                f"{title}\n"
                f"PASS: {passed}건 / FAIL: {failed}건 ★ 점검 필요\n\n"
                f"실패 (top 5):\n" + "\n".join(fail_lines[:5])
            )
        send_message(msg)
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")


def main():
    print(f"[자동 회귀] {datetime.now():%Y-%m-%d %H:%M:%S} 시작")
    success, passed, failed, output = run_pytest()
    print(f"PASS: {passed} / FAIL: {failed}")

    send_telegram_report(success, passed, failed, output)

    if not success:
        print("=== FAIL 상세 ===")
        for line in output.split("\n"):
            if "FAILED" in line or "Error" in line:
                print(line)
        sys.exit(1)
    else:
        print("✅ 모든 회귀 통과")
        sys.exit(0)


if __name__ == "__main__":
    main()
