"""§9 dry-run 단계 3 — quant_preflight.py --simulate-paper (S1~S6) 검증.

subprocess로 실제 preflight를 실행해 16/16 PASS + 회귀 격리를 확인한다.
한글 출력(cp949 회피)을 위해 PYTHONIOENCODING=utf-8 + cwd=project_root.

전제(로컬/VPS 재가동 심사 환경):
  - data/KILL_SWITCH 존재 (S6 차단=PASS)
  - .env ORDER_INTENTS_HMAC_KEY 32+자 (S2/S3 서명 PASS)
이 둘은 preflight expect=blocked의 본질적 가정과 동일하다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = PROJECT_ROOT / "tools" / "quant_preflight.py"


def _preconditions_met() -> bool:
    """§9 dry-run smoke 전제: data/KILL_SWITCH 존재 + .env HMAC 키 32+자.

    전제 미충족 환경(CI/타 머신)에서는 16/16이 성립하지 않으므로 skip한다
    (실패가 아니라 '검증 불가'로 분리 — code-analyzer P1-2).
    """
    data = PROJECT_ROOT / "data"
    kill = (data / "KILL_SWITCH").exists() or (data / "kill_switch.flag").exists()
    hmac_ok = False
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("ORDER_INTENTS_HMAC_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                hmac_ok = len(val) >= 32
                break
    return kill and hmac_ok


pytestmark = pytest.mark.skipif(
    not _preconditions_met(),
    reason="§9 dry-run 전제 미충족 (data/KILL_SWITCH + .env ORDER_INTENTS_HMAC_KEY 32+자 필요)",
)


def _run_preflight(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(PREFLIGHT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
        errors="replace",
    )


class TestSimulatePaper:
    """--simulate-paper S1~S6 시뮬레이션 가드."""

    def test_simulate_paper_passes_16_of_16(self):
        result = _run_preflight("--simulate-paper")
        assert result.returncode == 0, result.stdout
        assert "RESULT: PASS (16/16)" in result.stdout
        for s in ("S1", "S2", "S3", "S4", "S5", "S6"):
            assert f"[PASS] {s}" in result.stdout, f"{s} missing/FAIL:\n{result.stdout}"

    def test_s4_kis_paper_blocked(self):
        # KisOrderAdapter는 mode='live' 전용 → paper 전달 시 ValueError 차단
        result = _run_preflight("--simulate-paper")
        assert "[PASS] S4 KisAdapter mode=paper 차단" in result.stdout

    def test_s5_paper_live_blocked(self):
        # PaperOrderAdapter는 mode='paper' 전용 → live 전달 시 ValueError 차단
        result = _run_preflight("--simulate-paper")
        assert "[PASS] S5 PaperAdapter mode=live 차단" in result.stdout

    def test_s6_runtime_guard_blocks(self):
        # KILL_SWITCH 존재 시 차단(PermissionError) = PASS (fail-closed)
        result = _run_preflight("--simulate-paper")
        assert "[PASS] S6 runtime guard 차단" in result.stdout

    def test_simulate_paper_no_network_fast(self):
        # __new__ 우회 + paper lazy-init → mojito 토큰 발급 네트워크 0.
        # 네트워크 호출이 섞이면 수 초 이상 걸리므로, 정상 종료(RC=0)만 보장.
        result = _run_preflight("--simulate-paper")
        assert result.returncode == 0, result.stdout


class TestSimulatePaperRegressionIsolation:
    """비-simulate 실행은 S1~S6를 절대 포함하지 않음 (회귀 격리)."""

    def test_expect_blocked_excludes_s_guards(self):
        result = _run_preflight("--expect", "blocked")
        assert result.returncode == 0, result.stdout
        assert "RESULT: PASS" in result.stdout
        # 카운트 tail은 simulate 모드 전용 — 비-simulate 출력에 없어야 함
        assert "/16)" not in result.stdout
        # S1~S6 라벨 미포함
        for label in (
            "S1 paper",
            "S2 HMAC",
            "S3 intent",
            "S4 KisAdapter",
            "S5 PaperAdapter",
            "S6 runtime",
        ):
            assert label not in result.stdout, f"{label} leaked into non-simulate output"

    def test_expect_blocked_still_ten_guards(self):
        # 기존 10개 가드 라벨 유지 (대표 3개 확인)
        result = _run_preflight("--expect", "blocked")
        assert "[PASS] data/KILL_SWITCH:" in result.stdout
        assert "[PASS] ORDER_INTENTS_HMAC_KEY" in result.stdout
        assert "[PASS] raw mojito broker calls" in result.stdout
