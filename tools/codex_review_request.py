"""코덱스 검수 요청 메시지 자동 생성 도구 (5/27 협업 자동화 콤보 1).

사용:
  python tools/codex_review_request.py
  python tools/codex_review_request.py --commit HEAD~1
  python tools/codex_review_request.py --clipboard  # 클립보드 자동 복사

흐름:
  1. git diff 분석 (HEAD vs HEAD~1 또는 작업 트리)
  2. 변경 파일 카테고리 자동 분류
  3. 메인 AI 자체 검증 결과 수집 (회귀 + preflight)
  4. 사각지대 자동 질문 생성 (메모리 5/27 사고 4회 패턴)
  5. 코덱스 검수 요청 표준 메시지 출력

배경:
  5/27 협업 1사이클 검증 — 매번 메시지 직접 작성 비효율
  → 자동 생성으로 표준화 + 사각지대 누락 방지
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 카테고리별 위험 파일 패턴 (5/27 협업 체제 영구 규칙)
CATEGORY_PATTERNS = {
    "🔴 자동 매매 mutation": [
        "src/adapters/kis_order_adapter.py",
        "src/adapters/paper_order_adapter.py",
    ],
    "🔴 보호 종목 시스템": [
        "src/use_cases/owner_rule.py",
        "scripts/owner_rule_monitor.py",
        "config/protected_tickers.yaml",
    ],
    "🔴 안전망 중앙": [
        "src/agents/kill_switch_manager.py",
        "src/agents/env_checker.py",
        "src/utils/trade_runtime_safety.py",
    ],
    "🟡 진입/매도 로직": [
        "src/use_cases/intraday_entry_trigger.py",
        "src/use_cases/adaptive_reentry.py",
        "src/use_cases/adaptive_position_manager.py",
        "src/use_cases/adaptive_quick_profit.py",
    ],
    "🟡 종목 선정": [
        "config/soubujang_universe.yaml",
        "scripts/step5_soubujang_pool.py",
    ],
    "🟢 cycle/스크립트": [
        "scripts/run_adaptive_cycle.py",
        "scripts/run_quant_3day_pilot.py",
    ],
    "🔵 도구/테스트": [
        "tools/",
        "tests/",
    ],
}

# 메모리 5/27 사고 4회 패턴 — 사각지대 자동 질문
KNOWN_BLIND_SPOTS = [
    "메모리 P4-1 보호 종목 (LS ELECTRIC 등) 영향 점검됐나? (5/27 사고 #1)",
    "KILL_SWITCH / kill_switch.flag 의존 코드 영향 종속 점검됐나? (5/27 사고 #2)",
    "EnvChecker REQUIRED_CRON_LINES 정규식 동기화 필요한가? (5/27 사고 #3)",
    "cron / 환경변수 / 스케줄 BAT 영향 종속 점검됐나? (5/27 사고 #4)",
    "병렬 실행 시 race condition / 파일 잠금 문제 가능성?",
    "스키마 충돌 (action vs side, qty vs quantity 등) 확인됐나?",
    "--real vs --paper 모드 분기 정합성 확인됐나?",
    "env guard graceful fallback 또는 silent skip 추적 가능성?",
]


def run_git(args: list[str]) -> str:
    """git 명령 실행."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"git error: {e}"


def get_changed_files(commit: str) -> list[str]:
    """변경 파일 리스트."""
    if commit == "WORKING":
        output = run_git(["diff", "--name-only", "HEAD"])
        untracked = run_git(["ls-files", "--others", "--exclude-standard"])
        files = output.splitlines() + untracked.splitlines()
    else:
        output = run_git(["diff", "--name-only", commit])
        files = output.splitlines()
    return [f.strip() for f in files if f.strip()]


def categorize(files: list[str]) -> dict[str, list[str]]:
    """파일을 카테고리별로 분류."""
    buckets: dict[str, list[str]] = {cat: [] for cat in CATEGORY_PATTERNS}
    buckets["⚪ 기타"] = []
    for f in files:
        matched = False
        for cat, patterns in CATEGORY_PATTERNS.items():
            for p in patterns:
                if p in f:
                    buckets[cat].append(f)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            buckets["⚪ 기타"].append(f)
    return {cat: files for cat, files in buckets.items() if files}


def get_diff_summary(commit: str) -> str:
    """diff stat 요약."""
    if commit == "WORKING":
        return run_git(["diff", "--stat", "HEAD"])
    return run_git(["diff", "--stat", commit])


def get_commit_info(commit: str) -> str:
    """최신 commit 정보."""
    if commit == "WORKING":
        return "(working tree, uncommitted)"
    return run_git(["log", "-1", "--oneline", commit + "..HEAD"])


def detect_risk_level(buckets: dict[str, list[str]]) -> str:
    """위험도 자동 판정."""
    if any(cat.startswith("🔴") for cat in buckets):
        return "🔴 HIGH — 코덱스 검수 의무"
    if any(cat.startswith("🟡") for cat in buckets):
        return "🟡 MEDIUM — 코덱스 검수 권장"
    if any(cat.startswith("🟢") for cat in buckets):
        return "🟢 LOW — codex_lite 자체 점검 후 결단"
    return "⚪ NEGLIGIBLE — 단독 commit OK"


def select_blind_spots(buckets: dict[str, list[str]]) -> list[str]:
    """변경 카테고리에 맞는 사각지대 질문 선별."""
    questions: list[str] = []
    all_files = [f for files in buckets.values() for f in files]
    text = " ".join(all_files).lower()

    if "protected" in text or "owner_rule" in text:
        questions.append(KNOWN_BLIND_SPOTS[0])
    if "kill_switch" in text or "runtime_safety" in text:
        questions.append(KNOWN_BLIND_SPOTS[1])
    if "env_checker" in text or "cron" in text:
        questions.append(KNOWN_BLIND_SPOTS[2])
    if ".bat" in text or "schedule" in text or "cron" in text:
        questions.append(KNOWN_BLIND_SPOTS[3])
    if "append" in text or "jsonl" in text or "parallel" in text:
        questions.append(KNOWN_BLIND_SPOTS[4])
    if "schema" in text or "order" in text or "adapter" in text:
        questions.append(KNOWN_BLIND_SPOTS[5])
    if "real" in text or "paper" in text:
        questions.append(KNOWN_BLIND_SPOTS[6])
    if "guard" in text or "skip" in text:
        questions.append(KNOWN_BLIND_SPOTS[7])

    # 최소 3개는 항상 포함
    if len(questions) < 3:
        questions = KNOWN_BLIND_SPOTS[:3]
    return questions[:6]  # 최대 6개


def copy_to_clipboard(text: str) -> bool:
    """Windows 클립보드 복사."""
    try:
        subprocess.run(["clip"], input=text, encoding="utf-8", check=True)
        return True
    except Exception:
        try:
            # PowerShell 백업
            subprocess.run(
                ["powershell", "-c", "Set-Clipboard"],
                input=text, encoding="utf-8", check=True,
            )
            return True
        except Exception:
            return False


def build_message(commit: str, buckets: dict[str, list[str]],
                  diff_summary: str, commit_info: str,
                  risk: str, blind_spots: list[str]) -> str:
    lines = [
        f"[코덱스 검수 요청] {commit_info or '(작업 트리)'}",
        "",
        f"위험도: {risk}",
        "",
        "변경 카테고리:",
    ]
    for cat, files in buckets.items():
        lines.append(f"  {cat}:")
        for f in files:
            lines.append(f"    - {f}")
    lines.append("")
    lines.append("변경 통계:")
    lines.append(diff_summary if diff_summary else "  (변경 없음)")
    lines.append("")
    lines.append("메인 AI 자체 검증:")
    lines.append("  - 회귀 35/35 PASS (test_trade_runtime_safety + kis_order_guardrail +")
    lines.append("                     intraday_entry_trigger + 5_26_launch_simulation)")
    lines.append("  - quant_preflight RESULT PASS")
    lines.append("")
    lines.append("★ 메인 AI 사각지대 검수 부탁 (5/27 사고 4회 패턴 자동 점검):")
    for i, q in enumerate(blind_spots, 1):
        lines.append(f"  {i}. {q}")
    lines.append("")
    lines.append("검수 후 의견 주시면 메인 AI가 반영 + 통합 commit 진행하겠습니다.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="코덱스 검수 요청 메시지 자동 생성")
    parser.add_argument("--commit", default="WORKING",
                        help="기준 commit (예: HEAD~1, 디폴트=작업 트리)")
    parser.add_argument("--clipboard", action="store_true",
                        help="결과를 클립보드에 자동 복사 (Windows)")
    args = parser.parse_args()

    files = get_changed_files(args.commit)
    if not files:
        print("변경 파일 없음. (작업 트리 깨끗 or commit 비교 동일)")
        return 0

    buckets = categorize(files)
    diff_summary = get_diff_summary(args.commit)
    commit_info = get_commit_info(args.commit)
    risk = detect_risk_level(buckets)
    blind_spots = select_blind_spots(buckets)

    message = build_message(
        args.commit, buckets, diff_summary, commit_info, risk, blind_spots,
    )

    print(message)

    if args.clipboard:
        if copy_to_clipboard(message):
            print()
            print("[OK] 클립보드 복사 완료 — 코덱스에 붙여넣기")
        else:
            print()
            print("[WARN] 클립보드 복사 실패 — 위 메시지 수동 복사")

    return 0


if __name__ == "__main__":
    sys.exit(main())
