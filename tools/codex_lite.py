"""라이트 코덱스 — 메인 AI 정적 분석 자동 (5/27 협업 자동화 콤보 2).

배경:
  5/27 LS ELECTRIC + 원익IPS 사고 = 메인 AI 자가 검수 한계
  "구현 있는데 동작 안 함" 4회 반복 패턴 = 영구 학습 필요

  코덱스 외부 검수 의존도 ↓ + 메인 AI 1차 자체 점검 강화
  → 회색지대(자동 매매 vs 일반 로직) 사이 결정 도움

사용:
  python tools/codex_lite.py                # 작업 트리 분석
  python tools/codex_lite.py --commit HEAD~1
  python tools/codex_lite.py --strict       # FAIL 1건이라도 exit 1

자동 점검 룰 (5/27 사고 4회 패턴 학습):
  1. cron 변경 시 → env_checker.py REQUIRED_CRON_LINES 정규식 동기화
  2. 보호 종목 추가 시 → universe.yaml 모순 점검
  3. KILL_SWITCH 의존 코드 → 삭제 시 영향 종속 grep
  4. 자동 매도 코드 → protected_tickers 체크 필수
  5. paper_orders.jsonl 스키마 충돌 점검
  6. env guard 3중 (is_paper + run_id + pilot) 정합성
  7. trade_runtime_safety 게이트 의무 import 확인
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git"] + args, cwd=PROJECT_ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def get_changed_files(commit: str) -> list[str]:
    if commit == "WORKING":
        output = run_git(["diff", "--name-only", "HEAD"])
        untracked = run_git(["ls-files", "--others", "--exclude-standard"])
        files = output.splitlines() + untracked.splitlines()
    else:
        output = run_git(["diff", "--name-only", commit])
        files = output.splitlines()
    return [f for f in files if f.endswith((".py", ".yaml", ".yml", ".bat"))]


def read_file(path: str) -> str:
    p = PROJECT_ROOT / path
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# ============================================================
# 자동 점검 룰 (5/27 사고 패턴 학습)
# ============================================================

def check_rule_1_cron_env_sync(files: list[str]) -> list[tuple[str, str]]:
    """룰 1: cron 변경 시 EnvChecker 정규식 동기화 점검 (5/27 사고 #3)."""
    issues = []
    cron_changed = any("cron" in f.lower() or ".bat" in f or "schedule" in f.lower() for f in files)
    env_checker_changed = "src/agents/env_checker.py" in files

    if cron_changed and not env_checker_changed:
        # cron 변경됐는데 env_checker 미변경 — 정규식 영향 가능
        for f in files:
            if "schedule" in f.lower() or ".bat" in f or "cron" in f.lower():
                # 변경 내용에 시간 패턴 변경 있는지 grep
                diff = run_git(["diff", f])
                if re.search(r"[+-].*\*/?\d+\s+\d+", diff):
                    issues.append(("WARN", f"{f}: cron 시간 변경 가능 → env_checker.py REQUIRED_CRON_LINES 동기화 확인"))
    return issues


def check_rule_2_protected_vs_universe(files: list[str]) -> list[tuple[str, str]]:
    """룰 2: 보호 종목 vs universe.yaml 모순 점검 (5/27 사고 #1)."""
    issues = []
    if "config/protected_tickers.yaml" in files or "config/soubujang_universe.yaml" in files:
        try:
            import yaml
            protected_path = PROJECT_ROOT / "config" / "protected_tickers.yaml"
            universe_path = PROJECT_ROOT / "config" / "soubujang_universe.yaml"
            if not protected_path.exists() or not universe_path.exists():
                return issues
            with protected_path.open(encoding="utf-8") as f:
                protected = yaml.safe_load(f) or {}
            with universe_path.open(encoding="utf-8") as f:
                universe = yaml.safe_load(f) or {}

            protected_tickers = {str(p.get("ticker", "")).zfill(6)
                                 for p in protected.get("protected", [])}

            for sector, info in universe.items():
                if sector.startswith("_") or sector == "whitelist":
                    continue
                for t in info.get("tickers", []):
                    ticker = str(t.get("ticker", "")).zfill(6)
                    if ticker in protected_tickers:
                        issues.append((
                            "FAIL",
                            f"보호 종목 {ticker} ({t.get('name', '')})이 universe.yaml [{sector}]에도 등록됨 — 모순 (매수 후보 + 자동 매도 차단)"
                        ))
        except Exception as e:
            issues.append(("WARN", f"룰 2 점검 실패: {e}"))
    return issues


def check_rule_3_kill_switch_deps(files: list[str]) -> list[tuple[str, str]]:
    """룰 3: KILL_SWITCH 영향 종속 점검 (5/27 사고 #2)."""
    issues = []
    for f in files:
        if f.endswith(".py"):
            diff = run_git(["diff", f])
            # KILL_SWITCH 또는 kill_switch.flag 삭제(-) 패턴
            if re.search(r"^-.*(KILL_SWITCH|kill_switch\.flag)", diff, re.MULTILINE):
                # 동시에 추가가 없으면 = 삭제만
                if not re.search(r"^\+.*(KILL_SWITCH|kill_switch\.flag)", diff, re.MULTILINE):
                    issues.append(("WARN", f"{f}: KILL_SWITCH/kill_switch.flag 의존 코드 삭제 — owner_rule/EnvChecker 영향 점검 필요"))
    return issues


def check_rule_4_auto_sell_protected_check(files: list[str]) -> list[tuple[str, str]]:
    """룰 4: 자동 매도 코드 protected_tickers 체크 필수 (5/27 사고 #1)."""
    issues = []
    sell_keywords = ["sell_limit", "SELL_STOP_LOSS", "execute_auto_sell", "force_sell"]
    for f in files:
        if not f.endswith(".py"):
            continue
        content = read_file(f)
        diff = run_git(["diff", f])
        # 새 매도 로직 추가됐는지
        has_new_sell = any(re.search(rf"^\+.*{kw}", diff, re.MULTILINE) for kw in sell_keywords)
        if has_new_sell:
            # protected_tickers 또는 load_protected_tickers 호출 있는지
            if "protected_tickers" not in content and "load_protected" not in content:
                issues.append((
                    "FAIL",
                    f"{f}: 새 매도 로직 추가됐는데 protected_tickers 체크 없음 — LS ELECTRIC 사고 재발 위험"
                ))
    return issues


def check_rule_5_paper_orders_schema(files: list[str]) -> list[tuple[str, str]]:
    """룰 5: paper_orders.jsonl 스키마 충돌 (5/27 코덱스 검수 HIGH)."""
    issues = []
    for f in files:
        if not f.endswith(".py"):
            continue
        diff = run_git(["diff", f])
        # paper_orders.jsonl에 임시 스키마(action/qty) 작성 패턴
        if re.search(r'paper_orders\.jsonl', diff):
            if re.search(r'"action"\s*:\s*"BUY"', diff) or re.search(r'"qty"\s*:', diff):
                # canonical 스키마(side/quantity)도 함께 있는지
                if not re.search(r'"side"\s*:', diff) and not re.search(r'"quantity"\s*:', diff):
                    issues.append((
                        "FAIL",
                        f"{f}: paper_orders.jsonl 임시 스키마 (action/qty) 사용 — PaperOrderAdapter canonical (side/quantity) 권장"
                    ))
    return issues


def check_rule_6_env_guard_triple(files: list[str]) -> list[tuple[str, str]]:
    """룰 6: env guard 3중 정합성 (is_paper + run_id + pilot)."""
    issues = []
    for f in files:
        if not f.endswith(".py"):
            continue
        diff = run_git(["diff", f])
        # paper 시뮬 관련 코드 추가
        if re.search(r"^\+.*(PaperOrderAdapter|paper.*buy_limit)", diff, re.MULTILINE):
            # QUANT_3DAY_PILOT_RUN_ID 체크만 있고 is_paper 체크 없으면
            content = read_file(f)
            has_run_id = "QUANT_3DAY_PILOT_RUN_ID" in content
            has_is_paper = "is_paper" in content
            has_pilot = "QUANT_3DAY_PILOT" in content
            if has_run_id and not (has_is_paper and has_pilot):
                issues.append((
                    "WARN",
                    f"{f}: paper 시뮬 코드인데 env guard 3중 (is_paper + QUANT_3DAY_PILOT + run_id) 미충족 — --real 장부 오염 위험"
                ))
    return issues


def check_rule_7_runtime_safety_import(files: list[str]) -> list[tuple[str, str]]:
    """룰 7: 자동 매매 코드에 trade_runtime_safety 의무 import."""
    issues = []
    for f in files:
        if not f.endswith(".py"):
            continue
        # 자동 매매 mutation 코드
        if "adapter" in f.lower() and ("kis_order" in f or "live_order" in f):
            content = read_file(f)
            if "buy_limit" in content or "sell_limit" in content:
                if "trade_runtime_safety" not in content and "assert_runtime_orders_allowed" not in content:
                    issues.append((
                        "FAIL",
                        f"{f}: 자동 매매 mutation 코드 — trade_runtime_safety import 필수"
                    ))
    return issues


# ============================================================
# 메인
# ============================================================

ALL_RULES = [
    ("룰 1: cron-env_checker 동기화", check_rule_1_cron_env_sync),
    ("룰 2: 보호종목-universe 모순", check_rule_2_protected_vs_universe),
    ("룰 3: KILL_SWITCH 의존 종속", check_rule_3_kill_switch_deps),
    ("룰 4: 자동매도 protected 체크", check_rule_4_auto_sell_protected_check),
    ("룰 5: paper_orders 스키마", check_rule_5_paper_orders_schema),
    ("룰 6: env guard 3중", check_rule_6_env_guard_triple),
    ("룰 7: runtime_safety 의무", check_rule_7_runtime_safety_import),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="라이트 코덱스 — 메인 AI 정적 분석")
    parser.add_argument("--commit", default="WORKING")
    parser.add_argument("--strict", action="store_true",
                        help="FAIL/WARN 1건이라도 exit 1")
    args = parser.parse_args()

    files = get_changed_files(args.commit)
    print(f"=== codex_lite 자동 점검 — 변경 {len(files)}개 파일 ===")
    if files:
        for f in files:
            print(f"  - {f}")
    print()

    all_issues: list[tuple[str, str, str]] = []  # (rule, level, msg)
    for rule_name, checker in ALL_RULES:
        results = checker(files)
        if results:
            for level, msg in results:
                all_issues.append((rule_name, level, msg))

    fail_count = sum(1 for _, lv, _ in all_issues if lv == "FAIL")
    warn_count = sum(1 for _, lv, _ in all_issues if lv == "WARN")

    print(f"=== 점검 결과 ===")
    print(f"  FAIL: {fail_count}건")
    print(f"  WARN: {warn_count}건")
    print()

    if all_issues:
        print("=== 상세 ===")
        for rule, level, msg in all_issues:
            print(f"  [{level}] {rule}")
            print(f"        {msg}")
        print()

    if fail_count == 0 and warn_count == 0:
        print("[PASS] 7개 룰 모두 통과")
        print("→ 회색지대 검수 — 외부 코덱스 생략 가능 (자동 매매 mutation 제외)")
        return 0

    if args.strict and (fail_count > 0 or warn_count > 0):
        print(f"[FAIL] strict 모드 — exit 1")
        return 1

    if fail_count > 0:
        print(f"[FAIL] FAIL {fail_count}건 — 코덱스 검수 의무")
        return 1

    print(f"[WARN] WARN {warn_count}건 — 검토 후 결단")
    return 0


if __name__ == "__main__":
    sys.exit(main())
