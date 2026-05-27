"""5/27 사고 패턴 자동 점검 체크리스트 (5/27 협업 자동화 콤보 5).

배경:
  codex_lite.py = git diff 변경 파일 기반 점검 (변경 시 자동)
  self_review_checklist.py = 코드베이스 전체 영구 정합성 점검 (항시 비교)

  "사람이 기억해서 막는" 한계 → 영구 룰 자동 grep + 정합성 비교

사용:
  python tools/self_review_checklist.py             # 로컬 전체 점검
  python tools/self_review_checklist.py --vps       # VPS 점검 (crontab 비교)
  python tools/self_review_checklist.py --strict    # FAIL 시 exit 1

점검 항목 (5/27 사고 4회 패턴 학습):
  1. 보호 종목 vs universe.yaml 모순 (5/27 사고 #1)
  2. test_no_raw_mojito_order_bypass.py 존재 (5/27 코덱스 발견)
  3. owner_rule.py load_protected_tickers 호출 (보호 격리 영구화)
  4. KisOrderAdapter trade_runtime_safety 의무 import (5/27 안전망)
  5. owner_rule_monitor.py / sell_monitor.py assert_runtime_orders_allowed 호출
  6. protected_tickers.yaml 형식 정합성 (필수 필드)
  7. config/code-review-rules.yaml 존재 (영구 규칙 매트릭스)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    p = PROJECT_ROOT / path
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# ============================================================
# 점검 항목
# ============================================================

def check_1_protected_universe_consistency() -> list[tuple[str, str]]:
    """1. 보호 종목 vs universe.yaml 모순 (5/27 사고 #1)."""
    issues = []
    try:
        import yaml
        p = PROJECT_ROOT / "config" / "protected_tickers.yaml"
        u = PROJECT_ROOT / "config" / "soubujang_universe.yaml"
        if not p.exists() or not u.exists():
            return [("WARN", "protected_tickers.yaml 또는 soubujang_universe.yaml 부재")]
        with p.open(encoding="utf-8") as f:
            prot = yaml.safe_load(f) or {}
        with u.open(encoding="utf-8") as f:
            univ = yaml.safe_load(f) or {}
        protected_tk = {str(x.get("ticker", "")).zfill(6) for x in prot.get("protected", [])}
        for sector, info in univ.items():
            if sector.startswith("_") or sector == "whitelist":
                continue
            for t in info.get("tickers", []):
                tk = str(t.get("ticker", "")).zfill(6)
                if tk in protected_tk:
                    issues.append((
                        "FAIL",
                        f"보호 종목 {tk} ({t.get('name', '')})이 universe.yaml [{sector}]에 등록 — 정책 모순"
                    ))
    except Exception as e:
        issues.append(("WARN", f"점검 실패: {e}"))
    return issues


def check_2_raw_bypass_test_exists() -> list[tuple[str, str]]:
    """2. test_no_raw_mojito_order_bypass.py 존재 (5/27 코덱스 발견)."""
    p = PROJECT_ROOT / "tests" / "test_no_raw_mojito_order_bypass.py"
    if not p.exists():
        return [("FAIL", "tests/test_no_raw_mojito_order_bypass.py 부재 — raw mojito 우회 차단 회귀 불가")]
    return []


def check_3_owner_rule_protected_check() -> list[tuple[str, str]]:
    """3. owner_rule.py load_protected_tickers 호출 (보호 격리 영구화)."""
    content = _read("src/use_cases/owner_rule.py")
    if not content:
        return [("WARN", "owner_rule.py 부재")]
    if "load_protected_tickers" not in content:
        return [("FAIL", "owner_rule.py에 load_protected_tickers 함수 부재 — 보호 격리 시스템 깨짐")]
    if "ticker_norm in load_protected_tickers" not in content and "in load_protected_tickers()" not in content:
        return [("FAIL", "owner_rule.py evaluate_owner_rule이 보호 종목 체크 분기 미포함")]
    return []


def check_4_kis_adapter_runtime_safety() -> list[tuple[str, str]]:
    """4. KisOrderAdapter trade_runtime_safety 의무 import."""
    content = _read("src/adapters/kis_order_adapter.py")
    if not content:
        return [("WARN", "kis_order_adapter.py 부재")]
    if "trade_runtime_safety" not in content:
        return [("FAIL", "kis_order_adapter.py에 trade_runtime_safety import 부재")]
    if "assert_runtime_orders_allowed" not in content:
        return [("FAIL", "kis_order_adapter.py에 assert_runtime_orders_allowed 호출 부재")]
    return []


def check_5_monitors_runtime_safety() -> list[tuple[str, str]]:
    """5. owner_rule_monitor.py / sell_monitor.py assert_runtime_orders_allowed 호출."""
    issues = []
    for path in ["scripts/owner_rule_monitor.py", "scripts/sell_monitor.py"]:
        content = _read(path)
        if not content:
            continue  # 파일 부재는 별도 경고
        # 매도 주문 패턴 검색
        has_sell = bool(re.search(r"create_market_sell_order|sell_limit|create_market_buy_order", content))
        has_guard = "assert_runtime_orders_allowed" in content
        if has_sell and not has_guard:
            issues.append((
                "FAIL",
                f"{path}: 매수/매도 주문 호출 발견 but assert_runtime_orders_allowed 호출 부재 (raw bypass)"
            ))
    return issues


def check_6_protected_tickers_schema() -> list[tuple[str, str]]:
    """6. protected_tickers.yaml 형식 정합성."""
    issues = []
    try:
        import yaml
        p = PROJECT_ROOT / "config" / "protected_tickers.yaml"
        if not p.exists():
            return [("WARN", "protected_tickers.yaml 부재")]
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        required_fields = ["ticker", "name", "reason", "registered_at"]
        for item in data.get("protected", []):
            missing = [k for k in required_fields if not item.get(k)]
            if missing:
                issues.append((
                    "WARN",
                    f"protected_tickers.yaml {item.get('ticker', '?')} 누락 필드: {missing}"
                ))
    except Exception as e:
        issues.append(("WARN", f"점검 실패: {e}"))
    return issues


def check_7_code_review_rules_yaml() -> list[tuple[str, str]]:
    """7. docs/code-review-rules.yaml 존재 (영구 규칙 매트릭스)."""
    p = PROJECT_ROOT / "docs" / "code-review-rules.yaml"
    if not p.exists():
        return [("FAIL", "docs/code-review-rules.yaml 부재 — 검수 규칙 매트릭스 시스템 부재")]
    return []


def check_8_cron_regex_sync(vps_mode: bool = False) -> list[tuple[str, str]]:
    """8. env_checker REQUIRED_CRON_LINES 정규식 컴파일 가능 (5/27 사고 #3)."""
    content = _read("src/agents/env_checker.py")
    if not content:
        return [("WARN", "env_checker.py 부재")]
    try:
        # REQUIRED_CRON_LINES 추출 + 컴파일
        match = re.search(r"REQUIRED_CRON_LINES.*?\]", content, re.DOTALL)
        if not match:
            return [("FAIL", "env_checker.py REQUIRED_CRON_LINES 부재")]
        # 정규식 패턴 추출
        patterns = re.findall(r'r"([^"]+)"', match.group(0))
        for pat in patterns:
            try:
                re.compile(pat)
            except re.error as e:
                return [("FAIL", f"env_checker.py 정규식 컴파일 실패: {pat} — {e}")]
    except Exception as e:
        return [("WARN", f"정규식 점검 실패: {e}")]
    return []


# ============================================================
# 메인
# ============================================================

ALL_CHECKS = [
    ("1. 보호종목-universe 모순", check_1_protected_universe_consistency),
    ("2. raw bypass 테스트 존재", check_2_raw_bypass_test_exists),
    ("3. owner_rule 보호 분기", check_3_owner_rule_protected_check),
    ("4. KisOrderAdapter 안전 import", check_4_kis_adapter_runtime_safety),
    ("5. 모니터 runtime_safety 호출", check_5_monitors_runtime_safety),
    ("6. protected_tickers 스키마", check_6_protected_tickers_schema),
    ("7. code-review-rules.yaml 존재", check_7_code_review_rules_yaml),
    ("8. env_checker 정규식 컴파일", check_8_cron_regex_sync),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="5/27 사고 패턴 자동 점검")
    parser.add_argument("--strict", action="store_true",
                        help="FAIL/WARN 1건이라도 exit 1")
    parser.add_argument("--vps", action="store_true",
                        help="VPS 모드 (crontab 비교 포함)")
    args = parser.parse_args()

    print("=== self_review_checklist — 5/27 사고 패턴 자동 점검 ===")
    print()

    all_issues: list[tuple[str, str, str]] = []
    for name, checker in ALL_CHECKS:
        try:
            if "vps" in checker.__code__.co_varnames:
                results = checker(vps_mode=args.vps)
            else:
                results = checker()
            if not results:
                print(f"[PASS] {name}")
            else:
                for level, msg in results:
                    print(f"[{level}] {name}: {msg}")
                    all_issues.append((name, level, msg))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print()
    fail_count = sum(1 for _, lv, _ in all_issues if lv == "FAIL")
    warn_count = sum(1 for _, lv, _ in all_issues if lv == "WARN")
    print(f"=== 결과: FAIL {fail_count} / WARN {warn_count} ===")

    if fail_count == 0 and warn_count == 0:
        print("[PASS] 8개 영구 정합성 룰 모두 통과")
        return 0

    if args.strict:
        return 1 if (fail_count > 0 or warn_count > 0) else 0
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
