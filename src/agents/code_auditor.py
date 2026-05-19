"""CodeAuditor — 5/20 자율 자동매매 가동용 자동 검수 워커 (2026-05-19 신규).

배경: 1년 동안 메인 AI가 코드 작성 + 검수 둘 다 해서 매번 사장님이 잡아내야 했음.
     이제부터 자동 검수 워커가 분리되어 사장님 부담 0.

검수 대상 (퀀트봇 5/20 가동 핵심 7파일):
  1. scripts/auto_buy_executor.py
  2. scripts/owner_rule_monitor.py
  3. src/use_cases/paper_mirror.py
  4. src/adapters/paper_order_adapter.py
  5. src/adapters/kis_order_adapter.py
  6. src/use_cases/auto_buy_decider.py
  7. src/use_cases/owner_rule.py

검수 규칙 (단타봇 사고 + 1년 패턴 정리):
  CRITICAL — 환경변수 가드 / try-except / PermissionError 알림 / dead code
  HIGH     — 호가 단위 / enum 비교 / path 의존 / load_dotenv
  MEDIUM   — 로깅 / 숫자 단위 함정

호출 시점:
  - post-commit git hook (커밋 직후 자동)
  - 매일 18:00 (장마감 후)
  - 5/20 13:55 (가동 직전 최종 검수)
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 검수 대상 7파일 (5/20 가동 핵심) ──
AUDIT_TARGETS: list[str] = [
    "scripts/auto_buy_executor.py",
    "scripts/owner_rule_monitor.py",
    "src/use_cases/paper_mirror.py",
    "src/adapters/paper_order_adapter.py",
    "src/adapters/kis_order_adapter.py",
    "src/use_cases/auto_buy_decider.py",
    "src/use_cases/owner_rule.py",
]

# ── 진입점으로 간주되는 스크립트(scripts/ 산하 .py) ──
ENTRY_POINT_PREFIX = "scripts/"

# ── 자동매매 가드용 환경변수 (단타봇 사고 패턴 — 모두 있어야 안전) ──
REQUIRED_AUTO_TRADE_FLAGS = {
    "AUTO_TRADE_5_20",       # 5/20 출격 게이트
    "KILL_SWITCH",           # 파일 기반이라 env 로도 noop — 진입점에서는 file check
    "AUTO_TRADING_ENABLED",  # kis_order_adapter._guard
}

# ── KIS API 응답 단위 의심 키 ──
KIS_UNIT_SUSPECT_KEYS = {
    "pgtr_ntby_qty",   # 프로그램 순매수 — 수량인가 금액인가
    "prdy_ctrt",       # 등락률 — % 단위
    "acml_vol",        # 누적 거래량
    "tot_evlu_amt",    # 평가금액 — 원 단위
}


@dataclass
class Finding:
    severity: str   # CRITICAL / HIGH / MEDIUM
    file: str
    line: int
    rule: str
    msg: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "rule": self.rule,
            "msg": self.msg,
        }


@dataclass
class AuditResult:
    critical: list[Finding] = field(default_factory=list)
    high: list[Finding] = field(default_factory=list)
    medium: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.critical and not self.high

    def to_dict(self) -> dict:
        # S1 표준 필드 호환을 위해 카운트는 counts dict, summary는 audit_all에서 문자열로 덮어씀.
        return {
            "critical": [f.to_dict() for f in self.critical],
            "high": [f.to_dict() for f in self.high],
            "medium": [f.to_dict() for f in self.medium],
            "passed": self.passed,
            "counts": {
                "critical": len(self.critical),
                "high": len(self.high),
                "medium": len(self.medium),
            },
        }


class CodeAuditor:
    """AST + 정규식 정적 분석 기반 검수 워커."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT

    # ──────────────────────────────────────────
    # 퍼블릭 API
    # ──────────────────────────────────────────
    def audit_all(self) -> dict:
        """7파일 전체 검수."""
        result = AuditResult()
        for rel in AUDIT_TARGETS:
            findings = self.audit_file(rel)
            for f in findings:
                if f["severity"] == "CRITICAL":
                    result.critical.append(Finding(**f))
                elif f["severity"] == "HIGH":
                    result.high.append(Finding(**f))
                else:
                    result.medium.append(Finding(**f))
        result_dict = result.to_dict()

        # S1 표준 필드 — Reporter가 통일 표시 (5/19 자체 검수 S1)
        critical_count = len(result_dict.get("critical", []))
        high_count = len(result_dict.get("high", []))
        medium_count = len(result_dict.get("medium", []))
        if critical_count > 0:
            result_dict["status"] = "FAIL"
        elif high_count > 0 or medium_count > 0:
            result_dict["status"] = "WARN"
        else:
            result_dict["status"] = "OK"
        result_dict["agent"] = "code_auditor"
        result_dict["summary"] = (
            f"CRITICAL {critical_count} | HIGH {high_count} | MEDIUM {medium_count}"
        )

        # Layer 7 — CRITICAL 발견 시 KILL_SWITCH 자동 활성화
        if critical_count > 0:
            from src.agents.kill_switch_manager import activate_kill_switch
            first_critical = result_dict["critical"][0]
            activate_kill_switch(
                reason=f"코드 CRITICAL {critical_count}건: {first_critical.get('rule', 'unknown')}",
                source="CodeAuditor",
                send_tg=True,  # 5/19 사장님 결단 C — KILL_SWITCH RED 단일 채널만 카톡
            )

        # latest.json 저장
        from src.agents.kill_switch_manager import save_worker_report
        save_worker_report("code_auditor", result_dict)

        return result_dict

    def audit_file(self, file_path: str) -> list[dict]:
        """파일별 검수 — list[dict] 반환."""
        path = (self.root / file_path) if not Path(file_path).is_absolute() else Path(file_path)
        if not path.exists():
            return [{
                "severity": "CRITICAL",
                "file": str(file_path),
                "line": 0,
                "rule": "파일 부재",
                "msg": f"검수 대상 파일이 없음: {path}",
            }]

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            return [{
                "severity": "CRITICAL",
                "file": str(file_path),
                "line": 0,
                "rule": "파일 읽기 실패",
                "msg": f"{e}",
            }]

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            return [{
                "severity": "CRITICAL",
                "file": str(file_path),
                "line": e.lineno or 0,
                "rule": "SyntaxError",
                "msg": f"{e.msg}",
            }]

        rel = file_path.replace("\\", "/")
        lines = source.splitlines()

        findings: list[dict] = []
        # CRITICAL
        findings.extend(self._check_env_guards(rel, tree, source, lines))
        findings.extend(self._check_try_except_around_paper_mirror(rel, tree, source, lines))
        findings.extend(self._check_permission_error_pre_alert(rel, tree, source, lines))
        findings.extend(self._check_dead_code_archive_imports(rel, tree, source, lines))
        # HIGH
        findings.extend(self._check_tick_adjustment(rel, tree, source, lines))
        findings.extend(self._check_enum_fragile_comparison(rel, tree, source, lines))
        findings.extend(self._check_path_hardcoded(rel, tree, source, lines))
        findings.extend(self._check_load_dotenv_in_entrypoint(rel, tree, source, lines))
        # MEDIUM
        findings.extend(self._check_logging_around_orders(rel, tree, source, lines))
        findings.extend(self._check_kis_unit_assumption(rel, tree, source, lines))

        return findings

    def report_to_telegram(self, result: dict) -> None:
        """CRITICAL+HIGH 사장님 카톡. 0건이면 짧게 "코드 검수 OK".

        C5 도배 방지 (5/19 자체 검수):
        - CRITICAL>0 or HIGH>0: 즉시 카톡 (상세)
        - CRITICAL=0 AND HIGH=0 AND MEDIUM>0: 카톡 SKIP (logger.info만, daily summary용)
        - 전부 0: 짧은 OK 1줄

        사장님 결단 C (2026-05-19): 도배 방지를 위해 디폴트 OFF.
        AGENT_TELEGRAM_ENABLED=true 시만 발송.
        KILL_SWITCH RED는 kill_switch_manager가 별도로 발송 (유일한 단일 채널).
        """
        from datetime import datetime

        # 카운트 (가드 메시지에도 사용)
        n_crit = len(result.get("critical", []))
        n_high = len(result.get("high", []))
        n_med = len(result.get("medium", []))

        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            logger.info(
                "[CodeAuditor] 결과 logger.info만 (AGENT_TELEGRAM_ENABLED=false): "
                "CRITICAL=%d HIGH=%d MEDIUM=%d",
                n_crit, n_high, n_med,
            )
            return

        try:
            from src.telegram_sender import send_message
        except Exception as e:
            logger.warning("telegram_sender 임포트 실패: %s", e)
            return

        now = datetime.now().strftime("%H:%M")
        # result["summary"]가 S1 표준 적용 후 str로 바뀜 → 카운트는 critical/high/medium 리스트에서 직접 (위에서 이미 산출)

        # C5 — MEDIUM만 있을 때 카톡 SKIP (도배 방지)
        if n_crit == 0 and n_high == 0:
            if n_med > 0:
                logger.info(
                    "CodeAuditor: MEDIUM %d건 — daily summary용, 즉시 카톡 SKIP (도배 방지)",
                    n_med,
                )
                return
            # 전부 0 → 짧은 OK 1줄
            try:
                send_message(f"✅ [CodeAuditor] 7파일 검수 통과 ({now})")
            except Exception as e:
                logger.error("텔레그램 발송 실패: %s", e)
            return

        # CRITICAL 또는 HIGH 있음 — 상세 카톡
        lines_out = [f"🚨 [CodeAuditor 위험 발견] 5/20 가동 검토 필요! ({now})"]
        if n_crit:
            lines_out.append(f"  CRITICAL {n_crit}건:")
            for f in result.get("critical", [])[:5]:
                lines_out.append(f"    - {f['file']}:{f['line']} {f['rule']} — {f['msg'][:80]}")
        if n_high:
            lines_out.append(f"  HIGH {n_high}건:")
            for f in result.get("high", [])[:5]:
                lines_out.append(f"    - {f['file']}:{f['line']} {f['rule']} — {f['msg'][:80]}")
        if n_med:
            lines_out.append(f"  MEDIUM {n_med}건 (참고)")
        try:
            send_message("\n".join(lines_out))
        except Exception as e:
            logger.error("텔레그램 발송 실패: %s", e)

    # ──────────────────────────────────────────
    # CRITICAL 검사
    # ──────────────────────────────────────────
    def _check_env_guards(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """CRITICAL 1 — 환경변수 가드 누락 (KIS 주문 진입점).

        대상: scripts/ 산하 진입점 + kis_order_adapter._guard
        규칙: KIS 주문 호출 흐름에서 AUTO_TRADE_5_20 또는 AUTO_TRADING_ENABLED 가드가
              os.environ.get / os.getenv 비교로 등장해야 한다.
        """
        findings: list[dict] = []
        if not rel.startswith(ENTRY_POINT_PREFIX):
            return findings

        # KIS 주문 코드 경로가 있는가
        has_kis_order = bool(
            re.search(r"KisOrderAdapter|kis_order_adapter|create_(?:market|limit)_(?:buy|sell)_order", source)
        )
        if not has_kis_order:
            return findings

        # 가드 토글 비교 패턴
        guard_patterns = [
            r"os\.environ\.get\(\s*['\"]AUTO_TRADE_5_20['\"]",
            r"os\.getenv\(\s*['\"]AUTO_TRADE_5_20['\"]",
            r"os\.environ\.get\(\s*['\"]AUTO_TRADING_ENABLED['\"]",
            r"os\.getenv\(\s*['\"]AUTO_TRADING_ENABLED['\"]",
        ]
        matched = [p for p in guard_patterns if re.search(p, source)]
        if not matched:
            # 첫 KIS 호출 라인 찾기
            line_no = next(
                (i + 1 for i, ln in enumerate(lines)
                 if re.search(r"KisOrderAdapter\(\)|create_(?:market|limit)_(?:buy|sell)_order", ln)),
                1,
            )
            findings.append({
                "severity": "CRITICAL",
                "file": rel,
                "line": line_no,
                "rule": "환경변수 가드 누락",
                "msg": "KIS 주문 진입점인데 AUTO_TRADE_5_20 / AUTO_TRADING_ENABLED 가드 토글 부재",
            })

        # KILL_SWITCH 파일 가드 (auto_buy_executor / owner_rule_monitor)
        has_killswitch = "KILL_SWITCH" in source
        if not has_killswitch:
            findings.append({
                "severity": "CRITICAL",
                "file": rel,
                "line": 1,
                "rule": "KILL_SWITCH 가드 누락",
                "msg": "KIS 주문 진입점인데 KILL_SWITCH 파일 가드 부재",
            })
        return findings

    def _check_try_except_around_paper_mirror(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """CRITICAL 2 — paper mirror 호출이 실주문에 영향 0인지.

        paper_record_entry / paper_record_exit 호출은 try/except 안에 있어야 한다.
        """
        findings: list[dict] = []
        target_calls = {"paper_record_entry", "paper_record_exit"}

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.try_stack = 0
                self.unguarded: list[tuple[str, int]] = []

            def visit_Try(self, node: ast.Try):
                self.try_stack += 1
                self.generic_visit(node)
                self.try_stack -= 1

            def visit_Call(self, node: ast.Call):
                fname = None
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                if fname in target_calls and self.try_stack == 0:
                    self.unguarded.append((fname, node.lineno))
                self.generic_visit(node)

        v = Visitor()
        v.visit(tree)
        for fname, line_no in v.unguarded:
            findings.append({
                "severity": "CRITICAL",
                "file": rel,
                "line": line_no,
                "rule": "paper mirror try/except 누락",
                "msg": f"{fname}() 호출이 try/except 외부 — 실주문에 영향 0 보장 깨짐",
            })
        return findings

    def _check_permission_error_pre_alert(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """CRITICAL 3 — PermissionError 사전 카톡 경고 누락.

        kis_order_adapter._guard()가 PermissionError를 던지는데, 진입점에서
        주문 직전 사장님 카톡 예비 경고가 있어야 한다. (조용히 실패 방지)
        """
        findings: list[dict] = []
        if not rel.startswith(ENTRY_POINT_PREFIX):
            return findings

        # 실제로 buy_limit/sell_limit/buy_market/sell_market 호출하는지
        has_order_call = bool(
            re.search(r"\.(buy_limit|sell_limit|buy_market|sell_market)\s*\(", source)
        )
        if not has_order_call:
            return findings

        # 주문 직전·직후에 send_telegram / send_message 호출이 있는가
        has_pre_alert = bool(
            re.search(r"send_telegram|send_message", source)
        )
        if not has_pre_alert:
            # 첫 주문 호출 라인
            line_no = next(
                (i + 1 for i, ln in enumerate(lines)
                 if re.search(r"\.(buy_limit|sell_limit|buy_market|sell_market)\s*\(", ln)),
                1,
            )
            findings.append({
                "severity": "CRITICAL",
                "file": rel,
                "line": line_no,
                "rule": "PermissionError 사전 알림 누락",
                "msg": "KIS 주문 호출이 있는데 텔레그램 사전·사후 알림 부재 (조용한 실패 위험)",
            })
        return findings

    def _check_dead_code_archive_imports(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """CRITICAL 4 — archive 폐기 코드 import 잔재."""
        findings: list[dict] = []
        for node in ast.walk(tree):
            mod_name = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name and "archive" in alias.name.lower():
                        mod_name = alias.name
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module and "archive" in node.module.lower():
                    mod_name = node.module
            if mod_name:
                findings.append({
                    "severity": "CRITICAL",
                    "file": rel,
                    "line": getattr(node, "lineno", 1),
                    "rule": "archive import",
                    "msg": f"폐기 디렉터리 import: {mod_name} (CLAUDE.md LOCK 위반)",
                })
        return findings

    # ──────────────────────────────────────────
    # HIGH 검사
    # ──────────────────────────────────────────
    def _check_tick_adjustment(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """HIGH 5 — 호가 단위 미보정 가능성.

        AST 기반으로 ``X.create_limit_(buy|sell)_order(...)`` Call 노드를 직접 탐지.
        주석/docstring 등 문자열만 등장하는 경우는 무시.
        kis_order_adapter.py 본인은 _adjust_to_tick을 내부에서 호출하므로 통과.
        """
        findings: list[dict] = []
        if "_adjust_to_tick" in source:
            return findings

        suspicious_methods = {
            "create_limit_buy_order", "create_limit_sell_order",
        }
        first_line: Optional[int] = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in suspicious_methods:
                    first_line = node.lineno
                    break
        if first_line is not None:
            findings.append({
                "severity": "HIGH",
                "file": rel,
                "line": first_line,
                "rule": "호가 단위 미보정 의심",
                "msg": "create_limit_*_order 직접 호출 — _adjust_to_tick 사전 적용 불명확",
            })
        return findings

    def _check_enum_fragile_comparison(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """HIGH 6 — enum 비교 fragile (string 우회).

        예: str(order.status) == "OrderStatus.PENDING"
        """
        findings: list[dict] = []
        # str(...) == "OrderStatus." 패턴
        pattern = re.compile(r"str\([^)]+\)\s*==\s*['\"]OrderStatus\.")
        for i, ln in enumerate(lines, start=1):
            if pattern.search(ln):
                findings.append({
                    "severity": "HIGH",
                    "file": rel,
                    "line": i,
                    "rule": "enum 비교 fragile",
                    "msg": "str(...) == 'OrderStatus.X' fragile 비교 — OrderStatus.X 직접 비교 권장",
                })
        return findings

    def _check_path_hardcoded(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """HIGH 7 — hardcoded 절대 경로 의존 (cron cwd 다름).

        탐지: 'data/' 'logs/' 'config/' 같은 상대 경로 문자열을 Path 결합 없이 사용한 흔적.
        실제 검사는 PROJECT_ROOT 결합 패턴이 있는지 확인.
        """
        findings: list[dict] = []
        # PROJECT_ROOT 변수 정의 또는 Path(__file__) 사용 여부
        has_project_root = bool(
            re.search(r"PROJECT_ROOT\s*=\s*Path\s*\(\s*__file__\s*\)", source)
            or re.search(r"Path\s*\(\s*__file__\s*\)\s*\.resolve\(\)", source)
            or re.search(r"Path\s*\(\s*__file__\s*\)\.parent", source)
        )
        if has_project_root:
            return findings

        # 진입점이면서 data/ 또는 logs/ 문자열을 쓰는데 PROJECT_ROOT 결합 없음
        if rel.startswith(ENTRY_POINT_PREFIX):
            uses_data = bool(re.search(r"['\"](?:data|logs|config)/", source))
            if uses_data:
                line_no = next(
                    (i + 1 for i, ln in enumerate(lines) if re.search(r"['\"](?:data|logs|config)/", ln)),
                    1,
                )
                findings.append({
                    "severity": "HIGH",
                    "file": rel,
                    "line": line_no,
                    "rule": "path 의존",
                    "msg": "'data/' 등 상대 경로 사용 — PROJECT_ROOT(Path(__file__)) 결합 권장 (cron cwd 위험)",
                })
        return findings

    def _check_load_dotenv_in_entrypoint(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """HIGH 8 — 진입점에 load_dotenv 누락 (.env 미로드 위험)."""
        findings: list[dict] = []
        if not rel.startswith(ENTRY_POINT_PREFIX):
            return findings
        # KIS / 환경변수 사용 여부
        uses_env = bool(re.search(r"os\.environ\.get|os\.getenv|KisOrderAdapter", source))
        if not uses_env:
            return findings
        has_load = bool(re.search(r"load_dotenv\s*\(", source))
        if not has_load:
            findings.append({
                "severity": "HIGH",
                "file": rel,
                "line": 1,
                "rule": "load_dotenv 누락",
                "msg": "진입점에서 load_dotenv() 명시 호출 없음 — .env 미로드 위험",
            })
        return findings

    # ──────────────────────────────────────────
    # MEDIUM 검사
    # ──────────────────────────────────────────
    def _check_logging_around_orders(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """MEDIUM 9 — 매매 결정 직전·직후 logger.info 부재."""
        findings: list[dict] = []
        if not re.search(r"\.(buy_limit|sell_limit|buy_market|sell_market)\s*\(", source):
            return findings
        # logger 존재
        has_logger = bool(re.search(r"logger\s*=\s*logging\.getLogger", source))
        has_info_call = bool(re.search(r"logger\.(info|warning|error)\s*\(", source))
        if not (has_logger and has_info_call):
            findings.append({
                "severity": "MEDIUM",
                "file": rel,
                "line": 1,
                "rule": "매매 로깅 부재",
                "msg": "매매 호출 코드인데 logger.info/warning 사용 흔적 부재",
            })
        return findings

    def _check_kis_unit_assumption(self, rel: str, tree: ast.AST, source: str, lines: list[str]) -> list[dict]:
        """MEDIUM 10 — KIS 응답 키의 단위 가정 함정.

        owner_rule_monitor.py의 pgtr_eok 같은 변수 — pgtr_ntby_qty * 현재가 / 1e8 이
        실제로는 수량×원÷1e8이라 "억원" 추정인데 KIS 명세 불확실.
        """
        findings: list[dict] = []
        for key in KIS_UNIT_SUSPECT_KEYS:
            if key not in source:
                continue
            # 단위 가정 의심: 변수명에 _eok 또는 / 1e8 / / 100000000 사용
            suspicious = bool(
                re.search(rf"{key}.*?(?:/\s*1e8|/\s*100000000|_eok\b)", source, flags=re.DOTALL)
                or re.search(rf"_eok\b.*?{key}", source, flags=re.DOTALL)
            )
            if suspicious:
                line_no = next(
                    (i + 1 for i, ln in enumerate(lines) if key in ln),
                    1,
                )
                findings.append({
                    "severity": "MEDIUM",
                    "file": rel,
                    "line": line_no,
                    "rule": "KIS 단위 가정 의심",
                    "msg": f"{key} 응답의 단위(수량/원/억) 가정 — KIS 명세 재확인 필요",
                })
        return findings
