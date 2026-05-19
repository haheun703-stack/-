"""FlowMonitor — 자동매수 흐름 6단계 추적 에이전트 (2026-05-19 신규)

배경 (5/19 단타봇 D-Day 교훈):
  단타봇 5/19 자율 가동 중 매매 0건. 사장님이 직접 로그 까서 "왜 안 일어나는지"
  추적해야 했음. 같은 패턴을 퀀트봇 5/20 첫 가동에서 반복하지 않기 위한 가시화.

추적 대상 — auto_buy_executor 매매 흐름 8단계:
  1. 후보 풀: tomorrow_picks.json 9건 로드 OK?
  2. regime 로드: data/snapshots/{today}/_session.json → NEUTRAL/OVERHEAT?
  3. 안전선 평가: 9건 중 N건 통과 / 어떤 안전선이 가장 많이 막았나
  4. VWAP 게이트: 9건 중 N건 VWAP 정상
  5. integrated_score: TOP 3 점수 + 사유
  6. BUY 결정: 시도/성공/실패 + 실패 사유
  7. 카톡 발송: 텔레그램 메시지 발송 성공?
  8. positions.json 갱신: 매수 성공 시 갱신 확인?

데이터 소스:
  - /tmp/auto_buy_executor.log (VPS cron 출력)
  - data/tomorrow_picks.json
  - data/owner_rule_positions.json
  - data/snapshots/{today}/*_session.json (최신)

호출 시점 (5/20~5/23 cron, 매 5분 14:00~15:30):
  */5 14-15 * * 1-5 cd ~/quantum-master && python scripts/run_flow_monitor.py >> /tmp/flow_monitor.log 2>&1

매수 0건일 때 매 cron 텔레그램 도배를 막기 위해 matter_threshold 적용:
  - 14:55 마감 + 매수 0건 → 진단 카톡 1회
  - force=True → 무조건 발송
  - 그 외 단순 SKIP은 stdout/로그만 (텔레그램 X)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ── 경로 상수 ──
TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"
POSITIONS_PATH = PROJECT_ROOT / "data" / "owner_rule_positions.json"
SNAPSHOT_BASE = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_LOG = Path("/tmp/auto_buy_executor.log")
LOCAL_LOG_FALLBACK = PROJECT_ROOT / "logs" / "auto_buy_executor.log"
DEFAULT_GRADE = "강력 포착"
DEFAULT_TOP_N = 9

# 막힌 단계 식별자 (blocked_at_stage)
STAGE_CANDIDATES = "candidates"
STAGE_REGIME = "regime"
STAGE_SAFETY = "safety"
STAGE_VWAP = "vwap"
STAGE_SCORE = "score"
STAGE_BUY = "buy"
STAGE_TELEGRAM = "telegram"
STAGE_POSITIONS = "positions"
STAGE_NONE = "none"


@dataclass
class FlowTrace:
    """단일 cron 1회 흐름 추적 결과."""

    timestamp: str = ""
    log_path: str = ""
    log_found: bool = False
    log_mtime_str: str = ""

    # 단계별 결과
    candidates_loaded: int = 0
    candidates_expected: int = DEFAULT_TOP_N
    regime: str = "UNKNOWN"

    # 9건 평가 분포
    decisions: list[dict] = field(default_factory=list)  # [{ticker, name, score, action, reason}]
    safety_passed: int = 0
    vwap_passed: int = 0
    vwap_blocked: int = 0

    # TOP3
    top3_scores: list[tuple[str, float, str]] = field(default_factory=list)  # (name, score, reason)

    # 매수 결정
    buy_attempts: int = 0
    buy_success: int = 0
    buy_failed: int = 0
    buy_fail_reasons: list[str] = field(default_factory=list)

    # 외부 효과
    telegram_sent: bool = False
    positions_updated: bool = False
    positions_today_count: int = 0

    # 종합
    blocked_at_stage: str = STAGE_NONE
    diagnostic_msg: str = ""
    block_breakdown: dict[str, int] = field(default_factory=dict)  # {"점수 미달": 5, "VWAP 과열": 0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "log_path": self.log_path,
            "log_found": self.log_found,
            "log_mtime": self.log_mtime_str,
            "candidates_loaded": self.candidates_loaded,
            "candidates_expected": self.candidates_expected,
            "regime": self.regime,
            "safety_passed": self.safety_passed,
            "vwap_passed": self.vwap_passed,
            "vwap_blocked": self.vwap_blocked,
            "top3_scores": self.top3_scores,
            "buy_attempts": self.buy_attempts,
            "buy_success": self.buy_success,
            "buy_failed": self.buy_failed,
            "buy_fail_reasons": self.buy_fail_reasons,
            "telegram_sent": self.telegram_sent,
            "positions_updated": self.positions_updated,
            "positions_today_count": self.positions_today_count,
            "blocked_at_stage": self.blocked_at_stage,
            "diagnostic_msg": self.diagnostic_msg,
            "block_breakdown": self.block_breakdown,
            "decisions": self.decisions,
        }


class FlowMonitor:
    """auto_buy_executor 매매 흐름 추적 + 막힌 단계 카톡 진단."""

    # auto_buy_executor 로그 라인 정규식
    # 예: "  [SKIP] HPSP(403870) 점수 78 → 점수 78 < 90 | VWAP 정상"
    RE_DECISION = re.compile(
        r"\[(BUY|SKIP)\]\s+(.+?)\(([0-9A-Z]{6})\)\s+점수\s+(-?\d+(?:\.\d+)?)\s+→\s+(.+?)(?:\s+\|\s+(.+))?$"
    )
    # 예: "[regime] 1430_session.json → NEUTRAL"
    RE_REGIME = re.compile(r"\[regime\][^→]*→\s+(\w+)")
    # 예: "  자동매수 평가 (14:25 KST, regime=NEUTRAL)"
    RE_HEADER_REGIME = re.compile(r"regime=([A-Z_]+)")
    # 예: "  후보 9건, dry-run=False"
    RE_CANDIDATES = re.compile(r"후보\s+(\d+)건")
    # 예: "자동매수 성공 458650 @ 25000원 (주문번호 ...)"
    RE_BUY_SUCCESS = re.compile(r"자동매수 성공\s+([0-9A-Z]{6})\s+@\s+(\d+)원")
    # 예: "주문 실패 458650: ..."
    RE_BUY_FAIL = re.compile(r"주문 실패\s+([0-9A-Z]{6}):\s+(.+)")
    # 예: "주문 실행 예외 458650: ..."
    RE_BUY_EXC = re.compile(r"주문 실행 예외\s+([0-9A-Z]{6}):\s+(.+)")
    # 예: "텔레그램 발송 실패: ..."
    RE_TELEGRAM_FAIL = re.compile(r"텔레그램 발송 실패")
    # 평가 헤더 (cron 1회 경계)
    RE_SECTION_HEAD = re.compile(r"자동매수 평가")

    def __init__(
        self,
        log_path: Path | str | None = None,
        candidates_expected: int = DEFAULT_TOP_N,
    ):
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG
        # VPS /tmp/ 경로 미존재 시 로컬 logs/ fallback
        if not self.log_path.exists() and LOCAL_LOG_FALLBACK.exists():
            self.log_path = LOCAL_LOG_FALLBACK
        self.candidates_expected = candidates_expected

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────
    def trace_latest_run(self) -> dict[str, Any]:
        """최근 cron 1회 흐름 추적 → dict 반환."""
        trace = FlowTrace(
            timestamp=datetime.now().strftime("%H:%M"),
            log_path=str(self.log_path),
            candidates_expected=self.candidates_expected,
        )

        # 1) 후보 풀
        trace.candidates_loaded = self._count_candidates()
        if trace.candidates_loaded == 0:
            trace.blocked_at_stage = STAGE_CANDIDATES
            trace.diagnostic_msg = (
                "tomorrow_picks.json 강력포착 0건 — BAT-D 전날 갱신 점검 필요"
            )
            result = trace.to_dict()
            # S1 표준 필드 — early-return 경로 (5/19 자체 검수 S1)
            result["agent"] = "flow_monitor"
            result["status"] = "FAIL"  # blocked_at_stage == candidates
            result["summary"] = "매수 0 / 시도 0 / 후보 0 (BAT-D 미갱신 의심)"
            _post_process_trace(result)
            return result

        # 2) regime (snapshot 최신)
        trace.regime = self._load_latest_regime()

        # 3) 로그 파싱 (단계 3~7)
        log_lines = self._read_latest_section()
        trace.log_found = bool(log_lines)
        if log_lines:
            try:
                mtime = datetime.fromtimestamp(self.log_path.stat().st_mtime)
                trace.log_mtime_str = mtime.strftime("%H:%M:%S")
            except Exception:
                trace.log_mtime_str = ""

        self._parse_section(log_lines, trace)

        # 8) positions.json 오늘자 매수 카운트
        trace.positions_today_count = self._count_today_positions()
        trace.positions_updated = trace.positions_today_count > 0

        # 종합 판정: 막힌 단계 식별
        self._diagnose(trace)

        result = trace.to_dict()

        # S1 표준 필드 — Reporter가 통일 표시 (5/19 자체 검수 S1)
        result["agent"] = "flow_monitor"
        if result.get("buy_failed", 0) > 0 or result.get("blocked_at_stage") in (
            STAGE_CANDIDATES, STAGE_REGIME,
        ):
            result["status"] = "FAIL"
        elif result.get("buy_success", 0) > 0:
            result["status"] = "OK"
        else:
            # SKIP만 있는 경우(점수 미달, regime 차단 등)는 정상 흐름으로 본다
            result["status"] = "OK"
        result["summary"] = (
            f"매수 {result.get('buy_success', 0)} / "
            f"시도 {result.get('buy_attempts', 0)} / "
            f"후보 {result.get('candidates_loaded', 0)}"
        )

        _post_process_trace(result)
        return result

    def report_to_telegram(self, trace: dict[str, Any], force: bool = False) -> None:
        """매수 0건일 때 어디서 막혔는지 카톡.

        matter_threshold:
          - force=True → 무조건 발송
          - 14:55 이상 + buy_success=0 → 마감 진단 카톡
          - 그 외(단순 SKIP) → 발송 안 함 (도배 방지)

        사장님 결단 C (2026-05-19): 도배 방지를 위해 디폴트 OFF.
        AGENT_TELEGRAM_ENABLED=true 시만 발송 (force=True여도 무시).
        KILL_SWITCH RED는 kill_switch_manager가 별도로 발송 (유일한 단일 채널).
        """
        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            logger.info(
                "[FlowMonitor] 결과 logger.info만 (AGENT_TELEGRAM_ENABLED=false): "
                "buy_success=%s buy_failed=%s stage=%s ts=%s",
                trace.get("buy_success", 0),
                trace.get("buy_failed", 0),
                trace.get("blocked_at_stage", "?"),
                trace.get("timestamp", "?"),
            )
            return

        if not force and not self._matters(trace):
            logger.info(
                "[FlowMonitor] matter_threshold 미충족 — 텔레그램 SKIP (force=False, %s)",
                trace.get("timestamp"),
            )
            return

        msg = self._format_telegram(trace)
        try:
            from src.telegram_sender import send_message
            ok = send_message(msg)
            logger.info("[FlowMonitor] 텔레그램 발송 %s", "OK" if ok else "FAIL")
        except Exception as e:
            logger.warning("[FlowMonitor] 텔레그램 발송 예외: %s", e)

    def format_console(self, trace: dict[str, Any]) -> str:
        """stdout/로그용 콘솔 출력 — 매 cron 호출 시 항상 표시."""
        ts = trace["timestamp"]
        cl = trace["candidates_loaded"]
        sp = trace["safety_passed"]
        vp = trace["vwap_passed"]
        ba = trace["buy_attempts"]
        bs = trace["buy_success"]

        # 14:55 마감 여부
        is_close = ts >= "14:55"
        header = "🚨 [FlowMonitor " + ts + " 마감]" if is_close else f"🟢 [FlowMonitor {ts}]"

        lines: list[str] = []
        if is_close and bs == 0:
            lines.append(f"{header} 매수 0건 — {ba}회 시도 모두 SKIP")
            lines.append("  단계별 진단:")
            lines.append(f"    {'✅' if cl > 0 else '❌'} 후보 로드: {cl}건 {'정상' if cl > 0 else '없음'}")
            lines.append(f"    {'✅' if trace['regime'] != 'UNKNOWN' else '❌'} regime: {trace['regime']}")
            lines.append(
                f"    {'✅' if sp > 0 else '❌'} 안전선: {sp}/{cl} 통과"
            )
            lines.append(
                f"    {'✅' if vp > 0 else '⚠️'} VWAP: {vp}/{cl} 정상"
            )
            lines.append(f"  → {trace['diagnostic_msg']}")
        else:
            lines.append(f"{header} 후보 {cl} → 안전선 {sp}/{cl} 통과 → 매수 {bs}건")
            bb = trace.get("block_breakdown") or {}
            if bb:
                top_block = max(bb.items(), key=lambda kv: kv[1])
                others = ", ".join(f"{k} {v}건" for k, v in bb.items() if k != top_block[0])
                lines.append(
                    f"  ⏸️ 가장 많이 막힘: {top_block[0]} ({top_block[1]}건)"
                    + (f", {others}" if others else "")
                )
            top3 = trace.get("top3_scores") or []
            if top3:
                top_str = ", ".join(
                    f"{name} {int(score) if isinstance(score, (int, float)) else score}"
                    for name, score, _ in top3[:3]
                )
                reasons = [r for _, _, r in top3[:1]]
                reason_str = reasons[0] if reasons else ""
                lines.append(f"  TOP 3: {top_str}" + (f" — {reason_str}" if reason_str else ""))
            if not is_close:
                lines.append("  (다음 cron 재평가)")
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────
    def _count_candidates(self) -> int:
        if not TOMORROW_PICKS.exists():
            return 0
        try:
            data = json.loads(TOMORROW_PICKS.read_text(encoding="utf-8"))
            return sum(
                1 for p in data.get("picks", [])
                if p.get("grade") == DEFAULT_GRADE
            )
        except Exception as e:
            logger.warning("[FlowMonitor] tomorrow_picks 로드 실패: %s", e)
            return 0

    def _load_latest_regime(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        snap_dir = SNAPSHOT_BASE / today
        if not snap_dir.exists():
            return "UNKNOWN"
        files = sorted(snap_dir.glob("*_session.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return "UNKNOWN"
        try:
            data = json.loads(files[0].read_text(encoding="utf-8"))
            return data.get("regime", "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def _count_today_positions(self) -> int:
        if not POSITIONS_PATH.exists():
            return 0
        try:
            data = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
            today = datetime.now().strftime("%Y-%m-%d")
            return sum(
                1 for pos in data.get("positions", {}).values()
                if pos.get("entry_date") == today
            )
        except Exception:
            return 0

    def _read_latest_section(self) -> list[str]:
        """auto_buy_executor.log 마지막 '자동매수 평가' 섹션만 추출."""
        if not self.log_path.exists():
            return []
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("[FlowMonitor] 로그 읽기 실패 %s: %s", self.log_path, e)
            return []

        lines = content.splitlines()
        # 마지막 헤더 인덱스
        last_head = -1
        for i, line in enumerate(lines):
            if self.RE_SECTION_HEAD.search(line):
                last_head = i
        if last_head < 0:
            return lines[-200:]  # 헤더 없으면 마지막 200줄
        return lines[last_head:]

    def _parse_section(self, lines: list[str], trace: FlowTrace) -> None:
        """단일 cron 섹션을 줄별로 파싱."""
        if not lines:
            return

        block_counter: dict[str, int] = {}
        decisions: list[dict] = []

        for line in lines:
            # regime
            m = self.RE_REGIME.search(line)
            if m and trace.regime in ("UNKNOWN", ""):
                trace.regime = m.group(1)
                continue
            m = self.RE_HEADER_REGIME.search(line)
            if m and trace.regime in ("UNKNOWN", ""):
                trace.regime = m.group(1)
                continue

            # 후보 수 (헤더의 "후보 N건" — load_candidates 결과)
            m = self.RE_CANDIDATES.search(line)
            if m:
                try:
                    n = int(m.group(1))
                    # 로그에서 본 값이 실제 → tomorrow_picks 카운트보다 우선
                    if n > 0:
                        trace.candidates_loaded = n
                except Exception:
                    pass

            # 결정 (BUY/SKIP)
            m = self.RE_DECISION.search(line)
            if m:
                action = m.group(1)
                name = m.group(2).strip()
                ticker = m.group(3)
                try:
                    score = float(m.group(4))
                except Exception:
                    score = 0.0
                reason = m.group(5).strip()
                vwap_msg = (m.group(6) or "").strip()

                decisions.append({
                    "ticker": ticker,
                    "name": name,
                    "score": score,
                    "action": action,
                    "reason": reason,
                    "vwap": vwap_msg,
                })

                # VWAP 분석
                if vwap_msg:
                    if "정상" in vwap_msg or "눌림" in vwap_msg:
                        trace.vwap_passed += 1
                    elif "과열" in vwap_msg:
                        trace.vwap_blocked += 1
                    elif "SKIP" in vwap_msg:
                        # 오류로 PASS된 케이스 — 정상 측에 가산
                        trace.vwap_passed += 1

                # 안전선/사유 카운트
                if action == "BUY":
                    trace.safety_passed += 1
                else:
                    # reason 표준화: 점수/EYE/regime/안전선/VWAP
                    block_key = self._classify_block(reason, vwap_msg)
                    block_counter[block_key] = block_counter.get(block_key, 0) + 1

            # BUY 실행 결과
            if self.RE_BUY_SUCCESS.search(line):
                trace.buy_success += 1
                trace.buy_attempts += 1
            elif self.RE_BUY_FAIL.search(line) or self.RE_BUY_EXC.search(line):
                trace.buy_failed += 1
                trace.buy_attempts += 1
                fm = self.RE_BUY_FAIL.search(line) or self.RE_BUY_EXC.search(line)
                if fm:
                    trace.buy_fail_reasons.append(f"{fm.group(1)}: {fm.group(2)[:60]}")

            # DRY-RUN 매수 시뮬
            if "DRY-RUN: 매수 SKIP" in line:
                trace.buy_attempts += 1

            # 텔레그램 발송 실패
            if self.RE_TELEGRAM_FAIL.search(line):
                trace.telegram_sent = False

        # 결정 저장 + TOP3
        trace.decisions = decisions
        top3 = sorted(decisions, key=lambda d: d["score"], reverse=True)[:3]
        trace.top3_scores = [(d["name"], d["score"], d["reason"]) for d in top3]
        trace.block_breakdown = block_counter

        # 텔레그램: 실패 라인이 없었다면 일단 True로 가정 (cron이 try/except로 흡수)
        if trace.telegram_sent is False and not any("텔레그램" in ln for ln in lines):
            trace.telegram_sent = True

    @staticmethod
    def _classify_block(reason: str, vwap_msg: str) -> str:
        """SKIP 사유를 표준 카테고리로 분류."""
        r = (reason or "").strip()
        v = (vwap_msg or "").strip()
        if "과열" in v:
            return "VWAP 과열"
        if "점수" in r and ("<" in r or "미달" in r):
            return "점수 미달"
        if "EYE" in r or "필터" in r:
            return "EYE 필터"
        if "regime" in r.lower() or "CAUTION" in r or "BEAR" in r:
            return "regime 차단"
        if "AUTO_TRADE" in r:
            return "5/20 환경변수"
        if "일일" in r and "한도" in r:
            return "일일 1건 한도"
        if "10만" in r or "안전선 위반" in r:
            return "가격 10만 초과"
        if "시간" in r:
            return "14:00 미달"
        if "안전선" in r:
            return "안전선 미달"
        if "NEGA" in r:
            return "막내 NEGA"
        return "기타"

    def _diagnose(self, trace: FlowTrace) -> None:
        """blocked_at_stage + diagnostic_msg 결정."""
        if trace.candidates_loaded == 0:
            trace.blocked_at_stage = STAGE_CANDIDATES
            trace.diagnostic_msg = "tomorrow_picks 강력포착 0건 — BAT-D 갱신 점검"
            return
        if trace.regime in ("UNKNOWN", ""):
            trace.blocked_at_stage = STAGE_REGIME
            trace.diagnostic_msg = (
                "snapshot regime 미생성 — snapshot_session.py cron 점검"
            )
            return

        # decisions가 없으면 → 로그 자체가 비어있음 (cron 미실행 가능성)
        if not trace.decisions and trace.buy_attempts == 0:
            trace.blocked_at_stage = STAGE_SAFETY
            trace.diagnostic_msg = (
                f"auto_buy_executor 평가 결과 없음 — cron 실행/로그 경로({self.log_path}) 점검"
            )
            return

        if trace.buy_success > 0:
            trace.blocked_at_stage = STAGE_NONE
            trace.diagnostic_msg = f"매수 {trace.buy_success}건 성공"
            if not trace.positions_updated:
                trace.blocked_at_stage = STAGE_POSITIONS
                trace.diagnostic_msg = "매수 성공했으나 positions.json 미갱신 — 저장 로직 점검"
            return

        if trace.buy_failed > 0:
            trace.blocked_at_stage = STAGE_BUY
            trace.diagnostic_msg = (
                f"매수 시도 {trace.buy_failed}건 모두 실패: "
                + "; ".join(trace.buy_fail_reasons[:2])
            )
            return

        # SKIP만 있는 경우 — 가장 많이 막힌 단계 식별
        bb = trace.block_breakdown
        if not bb:
            trace.blocked_at_stage = STAGE_SAFETY
            trace.diagnostic_msg = "9건 모두 SKIP (사유 추출 실패)"
            return

        top_block = max(bb.items(), key=lambda kv: kv[1])
        category, count = top_block

        if category == "점수 미달":
            trace.blocked_at_stage = STAGE_SCORE
            top3 = trace.top3_scores
            top_score = max((s for _, s, _ in top3), default=0)
            trace.diagnostic_msg = (
                f"{count}건 점수 미달 (TOP {top_score:.0f} < 90). "
                "tomorrow_picks 갱신 결과 점수 낮음"
            )
        elif category == "VWAP 과열":
            trace.blocked_at_stage = STAGE_VWAP
            trace.diagnostic_msg = f"{count}건 VWAP 과열 차단 — 추격 보호 작동 중"
        elif category == "regime 차단":
            trace.blocked_at_stage = STAGE_REGIME
            trace.diagnostic_msg = f"regime {trace.regime} 차단 — CAUTION/BEAR 진입 보호"
        elif category == "5/20 환경변수":
            trace.blocked_at_stage = STAGE_SAFETY
            trace.diagnostic_msg = "AUTO_TRADE_5_20 != true — 환경변수 점검"
        elif category == "일일 1건 한도":
            trace.blocked_at_stage = STAGE_NONE
            trace.diagnostic_msg = "일일 1건 한도 도달 (정상)"
        else:
            trace.blocked_at_stage = STAGE_SAFETY
            trace.diagnostic_msg = f"가장 많이 막힘: {category} ({count}건)"

    @staticmethod
    def _matters(trace: dict[str, Any]) -> bool:
        """텔레그램 발송 임계값. 도배 방지.

        - 14:55 이상 + 매수 0건 + 진단 메시지 있음 → 발송
        - buy_failed > 0 → 발송 (주문 실패는 즉시 알림 가치)
        - positions_updated == False + buy_success > 0 → 발송 (저장 실패)
        - 그 외 → False
        """
        if trace.get("buy_success", 0) > 0 and not trace.get("positions_updated"):
            return True
        if trace.get("buy_failed", 0) > 0:
            return True
        ts = trace.get("timestamp", "00:00")
        if ts >= "14:55" and trace.get("buy_success", 0) == 0:
            return True
        # cron 자체 미실행 (decisions 0)
        if trace.get("blocked_at_stage") in (STAGE_CANDIDATES, STAGE_REGIME):
            return True
        return False

    @staticmethod
    def _format_telegram(trace: dict[str, Any]) -> str:
        ts = trace.get("timestamp", "??:??")
        cl = trace.get("candidates_loaded", 0)
        sp = trace.get("safety_passed", 0)
        vp = trace.get("vwap_passed", 0)
        bs = trace.get("buy_success", 0)
        bf = trace.get("buy_failed", 0)
        ba = trace.get("buy_attempts", 0)
        regime = trace.get("regime", "UNKNOWN")
        stage = trace.get("blocked_at_stage", STAGE_NONE)
        diag = trace.get("diagnostic_msg", "")

        if bs > 0 and not trace.get("positions_updated"):
            return (
                f"🚨 [FlowMonitor {ts}] 매수 성공 {bs}건 — positions.json 미갱신\n"
                f"  → 저장 로직 점검 필요\n"
                f"  진단: {diag}"
            )
        if bf > 0:
            reasons = "; ".join(trace.get("buy_fail_reasons", [])[:2])
            return (
                f"🚨 [FlowMonitor {ts}] 주문 실패 {bf}건/{ba}회 시도\n"
                f"  사유: {reasons}\n"
                f"  진단: {diag}"
            )

        is_close = ts >= "14:55"
        head = "🚨 [FlowMonitor " + ts + " 마감]" if is_close else f"🟢 [FlowMonitor {ts}]"

        lines: list[str] = [
            f"{head} 매수 0건 — {ba}회 시도 모두 SKIP",
            "  단계별 진단:",
            f"    {'✅' if cl > 0 else '❌'} 후보 로드: {cl}건",
            f"    {'✅' if regime not in ('UNKNOWN', '')  else '❌'} regime: {regime}",
            f"    {'✅' if sp > 0 else '❌'} 안전선: {sp}/{cl} 통과",
            f"    {'✅' if vp > 0 else '⚠️'} VWAP: {vp}/{cl} 정상",
        ]
        top3 = trace.get("top3_scores") or []
        if top3:
            top_str = ", ".join(
                f"{name} {int(score)}" for name, score, _ in top3[:3]
            )
            lines.append(f"  TOP 3: {top_str}")
        lines.append(f"  → 막힌 단계: {stage}")
        lines.append(f"  → {diag}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Layer 7 — 자동 KILL_SWITCH 통합 (5/20 가동 직전)
# ─────────────────────────────────────────────


def _check_critical_flow_failure(trace: dict) -> tuple[bool, str]:
    """매매 흐름의 치명적 실패 검출. (활성화 여부, 사유) 반환.

    활성화 조건 (보수적 — 정상 SKIP은 활성화 X):
    1. buy_failed > 0 (실제 KIS 주문 실패)
    2. blocked_at_stage == "candidates" (후보 풀 0건 — tomorrow_picks 자체 문제)
    3. blocked_at_stage == "regime" (snapshot 부재 — BAT 미실행 의심)
    4. positions_updated == False AND buy_success > 0 (매수 성공인데 저장 누락)
    5. telegram_sent == False (텔레그램 발송 실패)

    정상 SKIP은 활성화 X (안전선 점수 미달 같은 정당한 차단).
    """
    if trace.get("buy_failed", 0) > 0:
        return True, f"KIS 주문 실패 {trace['buy_failed']}건"
    if trace.get("buy_success", 0) > 0 and not trace.get("positions_updated", True):
        return True, "매수 성공인데 positions.json 미갱신"
    # 텔레그램 실패는 "실제 매매 시도/성공이 있었을 때"만 critical
    # (cron 첫 실행/매매 미발생 시 telegram_sent=False는 정상값)
    if (
        not trace.get("telegram_sent", True)
        and trace.get("buy_attempts", 0) > 0
    ):
        return True, "텔레그램 발송 실패"
    if trace.get("blocked_at_stage") == STAGE_CANDIDATES:
        return True, "후보 풀 0건 — tomorrow_picks 문제"
    # snapshot regime 부재는 매매 시도가 있었을 때만 critical
    # (장 시작 전 cron 호출에서는 정상)
    if (
        trace.get("blocked_at_stage") == STAGE_REGIME
        and trace.get("candidates_loaded", 0) > 0
        and trace.get("log_found")
    ):
        return True, "snapshot 부재 — BAT 미실행 의심"
    return False, ""


def _post_process_trace(trace: dict) -> None:
    """trace.to_dict() 결과 후처리 — Layer 7 KILL_SWITCH + Reporter 보고용 latest.json 저장.

    lazy import로 순환 import 방지. kill_switch_manager 부재 시 silent skip
    (다른 워커가 작성 중일 수 있음).
    """
    # 1) Layer 7 KILL_SWITCH 자동 활성화
    try:
        should_kill, kill_reason = _check_critical_flow_failure(trace)
        if should_kill:
            try:
                from src.agents.kill_switch_manager import activate_kill_switch
                activate_kill_switch(reason=kill_reason, source="FlowMonitor", send_tg=True)  # 5/19 결단 C — RED 단일 채널만
            except Exception as e:
                logger.warning("[FlowMonitor] activate_kill_switch 실패 (모듈 부재 가능): %s", e)
    except Exception as e:
        logger.warning("[FlowMonitor] critical_flow_failure 검사 실패: %s", e)

    # 2) latest.json 저장 (Reporter가 읽음)
    try:
        from src.agents.kill_switch_manager import save_worker_report
        save_worker_report("flow_monitor", trace)
    except Exception as e:
        logger.warning("[FlowMonitor] save_worker_report 실패 (모듈 부재 가능): %s", e)


__all__ = ["FlowMonitor", "FlowTrace"]
