"""EnvChecker — 5/20 자율 자동매매 가동 사전 점검 에이전트 (2026-05-19 신규)

배경:
  5/19 단타봇 D-Day 사고 — VERIFICATION_MODE=false 토글 OFF 상태로 가동돼서 매매 0건.
  같은 패턴 방지가 목표.

  사전에 .env / 사전 조건 파일 / cron 라인을 종합 점검하고,
  비활성/실패 항목을 사장님 텔레그램으로 즉시 보고한다.

점검 항목:
  1. 환경변수 12개 (MODEL, AUTO_TRADING_*, AUTO_TRADE_5_20, PAPER_MIRROR_MODE, KIS_*)
  2. 사전 조건 파일 (KILL_SWITCH, tomorrow_picks.json, owner_rule_positions.json)
  3. cron 라인 (5/20 06:00 / 16:00 / 14:* / 9-15:* — VPS 한정)

사용:
  python scripts/run_env_check.py           # 텔레그램 + 출력
  python scripts/run_env_check.py --no-tg   # 출력만
  FAIL 시 exit 1

cron 호출 시점:
  매일 06:00 (BAT-A 직전) — */5 06 * * * (단발)
  5/20 13:55             — 가동 5분 전 사전 경고
  5/20 14:00~14:55 매 5분 — auto_buy_executor와 동기
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"

# 점검 대상 환경변수 — (키, 기대값 또는 검증함수, 설명)
REQUIRED_ENV_VARS: list[tuple[str, object, str]] = [
    ("MODEL", "REAL", "실계좌 모드 (모의투자 차단)"),
    ("AUTO_TRADING_ENABLED", "1", "KIS 어댑터 가드"),
    ("AUTO_TRADE_5_20", "true", "5/20 출격 게이트"),
    ("PAPER_MIRROR_MODE", "true", "paper 시뮬 병행"),
    ("AUTO_TRADING_MAX_QTY", "1", "1주 한도"),
    ("AUTO_TRADING_MAX_AMOUNT", "100000", "10만원 한도"),
    ("AUTO_TRADING_MAX_TRADES_PER_DAY", "1", "일일 1건"),
    ("AUTO_TRADING_WHITELIST_ONLY", "0", "사장님 5/18 결단 (전종목 OK)"),
    ("AUTO_TRADING_TELEGRAM_ALERT", "1", "텔레그램 알림 ON"),
    ("KIS_APP_KEY", "_nonempty_min10", "KIS API 키 (길이 ≥10)"),
    ("KIS_APP_SECRET", "_nonempty_min10", "KIS API 시크릿 (길이 ≥10)"),
    ("KIS_ACC_NO", "_acc_format", "계좌번호 NNNNNNNN-NN 형식"),
]

# 사전 조건 파일
REQUIRED_FILES: list[tuple[str, Path, dict]] = [
    ("KILL_SWITCH",                  DATA_DIR / "KILL_SWITCH",
     {"must_exist_until": "2026-05-20 06:00",
      "must_absent_between": ("2026-05-20 06:00", "2026-05-20 16:00")}),
    ("tomorrow_picks.json",          DATA_DIR / "tomorrow_picks.json",
     {"must_exist": True, "max_age_hours": 24}),
    ("owner_rule_positions.json",    DATA_DIR / "owner_rule_positions.json",
     {"must_exist": True}),
]

# 5/20 cron 라인 키워드 (VPS crontab grep 검증)
REQUIRED_CRON_LINES: list[tuple[str, str]] = [
    ("KILL_SWITCH 자동 삭제 06:00", r"0\s+6\s+20\s+5"),
    ("KILL_SWITCH 자동 복구 16:00", r"0\s+16\s+20\s+5"),
    ("auto_buy_executor 14:00-14:55", r"\*/5\s+14\s+20\s+5"),
    ("owner_rule_monitor 9-15시", r"\*/5\s+9-15\s+20"),
]


def _validate_env_value(key: str, expected, actual: str) -> tuple[bool, str]:
    """환경변수 값 검증.

    Returns: (통과 여부, 메시지)
    """
    if actual is None or actual == "":
        return False, f"{key} 누락 또는 빈 값"

    if expected == "_nonempty_min10":
        if len(actual.strip()) < 10:
            return False, f"{key} 길이 부족 (len={len(actual)} < 10)"
        return True, f"{key} = ******** (len={len(actual)})"

    if expected == "_acc_format":
        # NNNNNNNN-NN 형식
        if not re.match(r"^\d{8}-\d{2}$", actual.strip()):
            return False, f"{key} = {actual} (기대 형식: NNNNNNNN-NN)"
        return True, f"{key} = {actual}"

    # 정확 일치 (대소문자 무시 일부 키)
    if key in ("AUTO_TRADE_5_20", "PAPER_MIRROR_MODE"):
        if actual.strip().lower() != str(expected).lower():
            return False, f"{key} = {actual} (기대: {expected})"
        return True, f"{key} = {actual}"

    if str(actual).strip() != str(expected):
        return False, f"{key} = {actual} (기대: {expected})"
    return True, f"{key} = {actual}"


def _check_kill_switch_with_grace(
    now: datetime, exists: bool | None = None
) -> tuple[bool, str]:
    """KILL_SWITCH 시간 기반 검증 + 06:00/16:00 cron 직후 10분 유예.

    유예 이유:
      5/20 06:00 정규 삭제 cron과 EnvChecker 06:00 cron이 동시 트리거 시
      EnvChecker가 막 삭제된 KILL_SWITCH를 다시 자동 활성화하는 사고 방지.
      마찬가지로 16:00 자동 복구 직후 10분도 검증 SKIP (cron 경합 방지).

    Args:
      now: 현재 시각
      exists: KILL_SWITCH 존재 여부 (None이면 파일 경로 확인)

    Returns:
      (통과 여부, 메시지)
    """
    if exists is None:
        exists = (DATA_DIR / "KILL_SWITCH").exists()

    h, m = now.hour, now.minute
    minutes = h * 60 + m
    is_active_hours = 6 * 60 <= minutes < 16 * 60  # 06:00~15:59
    is_grace_start = 6 * 60 <= minutes < 6 * 60 + 10  # 06:00~06:09 유예
    is_grace_end = 16 * 60 <= minutes < 16 * 60 + 10  # 16:00~16:09 유예

    if is_grace_start or is_grace_end:
        return True, "유예 구간 (06:00/16:00 ±10분, KILL_SWITCH 상태 검증 SKIP)"

    if is_active_hours:
        if exists:
            return False, "KILL_SWITCH 존재 — 가동 시간에 차단됨! 즉시 rm 필요"
        return True, "KILL_SWITCH 부재 — 가동 시간 정상"
    else:
        if not exists:
            return False, "KILL_SWITCH 부재 — 가동 외 시간에 자동매매 가드 풀림!"
        return True, "KILL_SWITCH 존재 — 가동 외 시간 정상 (가드 ON)"


def _check_file_condition(name: str, path: Path, cond: dict) -> tuple[bool, str]:
    """파일 조건 검증.

    Returns: (통과 여부, 메시지)
    """
    exists = path.exists()
    now = datetime.now()

    # KILL_SWITCH 특수 처리 (시간대별 존재/부재 요구 + 06:00/16:00 cron 직후 ±10분 유예)
    if name == "KILL_SWITCH":
        ok, msg = _check_kill_switch_with_grace(now, exists)
        return ok, msg

    # 일반 파일
    if cond.get("must_exist") and not exists:
        return False, f"{name} 부재 ({path})"

    if exists and cond.get("max_age_hours"):
        age_h = (time.time() - path.stat().st_mtime) / 3600
        max_age = cond["max_age_hours"]
        if age_h > max_age:
            return False, f"{name} 갱신 {age_h:.1f}시간 전 (>{max_age}h)"
        return True, f"{name} 갱신 {age_h:.1f}h 전"

    return True, f"{name} 존재"


def _check_vps_cron() -> list[tuple[str, bool, str]]:
    """VPS crontab 조회 (로컬 Windows에서는 SKIP).

    SSH 접속이 가능하면 crontab -l 실행 후 키워드 매칭.
    """
    results: list[tuple[str, bool, str]] = []

    # 로컬 Windows에서는 SKIP (정보용 경고만)
    if sys.platform.startswith("win"):
        for name, _pat in REQUIRED_CRON_LINES:
            results.append((name, True, "Windows 로컬 — VPS cron 검증 SKIP (warning)"))
        return results

    # Linux (VPS) — crontab -l 직접 호출
    try:
        out = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.STDOUT, timeout=5
        ).decode("utf-8", errors="replace")
    except Exception as e:
        for name, _pat in REQUIRED_CRON_LINES:
            results.append((name, False, f"crontab -l 실패: {e}"))
        return results

    for name, pat in REQUIRED_CRON_LINES:
        if re.search(pat, out):
            results.append((name, True, f"crontab 매칭 OK"))
        else:
            results.append((name, False, f"crontab에 패턴 [{pat}] 없음"))
    return results


class EnvChecker:
    """5/20 자율 자동매매 사전 점검 에이전트.

    check_all() 호출 시 dict 반환:
      {"status": "OK"|"FAIL",
       "passed": [...], "failed": [...], "warnings": [...],
       "timestamp": "...", "total": N, "ok_count": N, "fail_count": N}
    """

    def __init__(self):
        load_dotenv(ENV_PATH)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def check_env_vars(self) -> list[tuple[str, bool, str]]:
        """환경변수 12개 점검. Returns: [(name, passed, message), ...]"""
        results: list[tuple[str, bool, str]] = []
        for key, expected, desc in REQUIRED_ENV_VARS:
            actual = os.environ.get(key, "")
            ok, msg = _validate_env_value(key, expected, actual)
            results.append((f"{key} ({desc})", ok, msg))
        return results

    def check_files(self) -> list[tuple[str, bool, str]]:
        """사전 조건 파일 3종 점검."""
        results: list[tuple[str, bool, str]] = []
        for name, path, cond in REQUIRED_FILES:
            ok, msg = _check_file_condition(name, path, cond)
            results.append((name, ok, msg))
        return results

    def _check_kill_switch_with_grace(self) -> tuple[bool, str]:
        """KILL_SWITCH 유예 구간 포함 검증 (인스턴스 메서드 — 검증/외부 호출용).

        모듈 레벨 _check_kill_switch_with_grace(now, exists)에 위임.
        """
        return _check_kill_switch_with_grace(datetime.now())

    def check_cron(self) -> list[tuple[str, bool, str]]:
        """cron 4종 점검 (Windows는 SKIP-warning)."""
        return _check_vps_cron()

    def check_all(self) -> dict:
        """전체 점검. dict 반환."""
        env_results = self.check_env_vars()
        file_results = self.check_files()
        cron_results = self.check_cron()

        all_results = env_results + file_results + cron_results

        passed = [(n, m) for n, ok, m in all_results if ok]
        failed = [(n, m) for n, ok, m in all_results if not ok]

        # Windows 로컬 cron 검증은 warning (FAIL 아님)
        warnings: list[tuple[str, str]] = []
        if sys.platform.startswith("win"):
            warnings = [
                (n, m) for n, ok, m in cron_results
                if ok and "SKIP" in m
            ]
            # cron warning은 passed에서 제거하지 않음 (count는 OK 유지)

        status = "FAIL" if failed else "OK"
        # S1 표준 필드 — Reporter가 통일 표시 (5/19 자체 검수 S1)
        summary = f"{len(passed)}/{len(all_results)} 통과"
        if warnings:
            summary += f" (warnings {len(warnings)})"
        result = {
            "agent": "env_checker",
            "status": status,
            "summary": summary,
            "timestamp": self.timestamp,
            "total": len(all_results),
            "ok_count": len(passed),
            "fail_count": len(failed),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "env_results": env_results,
            "file_results": file_results,
            "cron_results": cron_results,
        }

        # Layer 7 — FAIL 시 KILL_SWITCH 자동 활성화 (5/19 1년 패턴 돌파)
        if result["status"] == "FAIL":
            from src.agents.kill_switch_manager import activate_kill_switch
            fail_summary = ", ".join(f[0] for f in result["failed"][:3])
            activate_kill_switch(
                reason=f"환경변수/파일 {len(result['failed'])}건 FAIL: {fail_summary}",
                source="EnvChecker",
                send_tg=True,  # 5/19 사장님 결단 C — KILL_SWITCH RED 단일 채널만 카톡
            )

        # latest.json 저장 (Reporter가 읽음)
        from src.agents.kill_switch_manager import save_worker_report
        save_worker_report("env_checker", result)

        return result

    def report_to_telegram(self, result: dict) -> None:
        """비활성/실패 항목 사장님 카톡.

        status=OK면 짧게 1줄, FAIL이면 상세.

        사장님 결단 C (2026-05-19): 도배 방지를 위해 디폴트 OFF.
        AGENT_TELEGRAM_ENABLED=true 시만 발송.
        KILL_SWITCH RED는 kill_switch_manager가 별도로 발송 (유일한 단일 채널).
        """
        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            logger.info(
                "[EnvChecker] 결과 logger.info만 (AGENT_TELEGRAM_ENABLED=false): %s — "
                "ok=%s fail=%s",
                result.get("status", "?"),
                result.get("ok_count", "?"),
                result.get("fail_count", "?"),
            )
            return

        try:
            from src.telegram_sender import send_message
        except Exception as e:
            logger.warning("telegram_sender import 실패: %s", e)
            return

        now_hhmm = datetime.now().strftime("%H:%M")
        if result["status"] == "OK":
            msg = (
                f"✅ [EnvChecker] {result['ok_count']}/{result['total']} 통과 ({now_hhmm})"
            )
            if result.get("warnings"):
                msg += f"\n  ⚠️ warnings {len(result['warnings'])}건 (Windows 로컬)"
        else:
            failed_lines = "\n".join(
                f"  ❌ {name}: {msg}" for name, msg in result["failed"]
            )
            msg = (
                f"🚨 [EnvChecker 실패 {result['fail_count']}건] 5/20 가동 위험!\n"
                f"{failed_lines}\n"
                f"  → 즉시 확인 부탁드립니다 사장님!"
            )
        try:
            send_message(msg)
        except Exception as e:
            logger.error("텔레그램 발송 실패: %s", e)


def main() -> int:
    """CLI 진입점 (run_env_check.py에서 호출)."""
    import argparse

    parser = argparse.ArgumentParser(description="EnvChecker — 5/20 가동 사전 점검")
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 OFF, 출력만")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    checker = EnvChecker()
    result = checker.check_all()

    # 콘솔 출력 (한국어)
    print("=" * 60)
    print(f"  EnvChecker 점검 결과 ({result['timestamp']})")
    print(f"  상태: {result['status']} | {result['ok_count']}/{result['total']} 통과")
    print("=" * 60)

    print("\n[환경변수 12종]")
    for name, ok, msg in result["env_results"]:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}")
        print(f"      → {msg}")

    print("\n[사전 조건 파일 3종]")
    for name, ok, msg in result["file_results"]:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {msg}")

    print("\n[cron 4종]")
    for name, ok, msg in result["cron_results"]:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {msg}")

    if result["failed"]:
        print("\n" + "=" * 60)
        print("  실패 요약:")
        for name, msg in result["failed"]:
            print(f"    ❌ {name}: {msg}")
        print("=" * 60)

    # 텔레그램 발송
    if not args.no_tg:
        checker.report_to_telegram(result)

    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
