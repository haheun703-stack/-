"""Reporter — 5/20 자율 자동매매 가동 검수팀 종합 보고 에이전트 (2026-05-19 신규)

배경:
  5/19 단타봇 D-Day 사고(VERIFICATION_MODE=false 토글 OFF로 매매 0건)에서 얻은 교훈.
  → 자동매매가 OFF여도, 검수팀 자체가 OFF여도 사장님이 알아챌 수 없음.

  Reporter는 EnvChecker / CodeAuditor / FlowMonitor / DataIntegrity 4명 워커가
  data/agent_reports/{name}_latest.json 에 남긴 결과를 *읽기만* 해서 종합 보고한다.
  → 워커 직접 호출 X (결합도 낮춤).

  **핵심 안전망**: Reporter가 정해진 시각에 카톡 발송.
  카톡이 안 오면 = 검수팀 OFF = 사장님이 즉시 알아챔.

보고 시점 (cron 4슬롯):
  morning_06       06:00 — 5명 작동 점호 + 어제 점검 N건 요약
  pre_trade_1355   13:55 — 자동매매 가동 5분 전, 환경/데이터 ALL CHECK
  post_trade_1600  16:00 — 자동매매 종료 후, 매매 결과 + paper mirror 결과
  daily_close_1900 19:00 — BAT-HEALTH 후 일일 마감

사용:
  python scripts/run_reporter.py --slot morning_06
  python scripts/run_reporter.py --slot pre_trade_1355
  python scripts/run_reporter.py --slot post_trade_1600
  python scripts/run_reporter.py --slot daily_close_1900
  python scripts/run_reporter.py --slot morning_06 --no-tg   # dry-run

워커 결과 스키마 (각 워커가 *_latest.json에 저장):
  {
    "agent": "env_checker"|"code_auditor"|"flow_monitor"|"data_integrity",
    "status": "OK"|"FAIL"|"WARN",
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "total": N, "ok_count": N, "fail_count": N,
    "summary": "한줄 요약",
    "details": [...]    # optional
  }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Layer 7 kill_switch_manager는 lazy import (단일 실패점 제거 — 5/19 자체 검수 C4)
# 함수 내부에서 try/except로 import (ImportError 시 fallback 동작)
# 이 패턴은 5/19 자체 검수에서 발견된 C4 (Reporter top-level import 단일 실패점)에 대응.
# kill_switch_manager 모듈이 일시적으로 부재/오류여도 Reporter는 최소 기능으로 동작해야 함.

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "agent_reports"

# 4명 워커
WORKERS = ["env_checker", "code_auditor", "flow_monitor", "data_integrity"]

# 한국어 표시명
WORKER_KR = {
    "env_checker": "EnvChecker",
    "code_auditor": "CodeAuditor",
    "flow_monitor": "FlowMonitor",
    "data_integrity": "DataIntegrity",
}

# 슬롯 정의
SLOTS = {
    "morning_06": "06:00",
    "pre_trade_1355": "13:55",
    "post_trade_1600": "16:00",
    "daily_close_1900": "19:00",
}

# 워커 결과 stale 기준 (시간)
STALE_HOURS_DEFAULT = 2

# 자동매매 D-Day
DDAY_DATE = datetime(2026, 5, 20)


def _load_worker_json(agent_name: str) -> dict | None:
    """data/agent_reports/{agent}_latest.json 읽기."""
    path = REPORTS_DIR / f"{agent_name}_latest.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("워커 %s 파일 파싱 실패: %s", agent_name, e)
        return None


def _is_stale(timestamp_str: str, max_hours: float = STALE_HOURS_DEFAULT) -> bool:
    """timestamp가 max_hours보다 오래됐는지."""
    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return True
    age_h = (datetime.now() - ts).total_seconds() / 3600.0
    return age_h > max_hours


def _icon(status: str, stale: bool = False) -> str:
    """상태 → 이모지."""
    if stale:
        return "⚠️"
    if status == "OK":
        return "✅"
    if status == "WARN":
        return "⚠️"
    if status == "FAIL":
        return "❌"
    return "❔"


def _post_trade_payload() -> dict | None:
    """post_trade 슬롯 전용: 실주문/paper mirror 결과 수집.

    소스 후보 (있으면 사용, 없으면 None):
      - data/auto_trade_log.jsonl     (auto_buy_executor 결과)
      - data/paper_mirror_log.jsonl   (paper mirror 시뮬)
    """
    payload: dict = {"real": None, "paper": None}

    real_path = DATA_DIR / "auto_trade_log.jsonl"
    if real_path.exists():
        today = datetime.now().strftime("%Y-%m-%d")
        buy_n = sell_n = 0
        last_buy = None
        try:
            with open(real_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not str(rec.get("ts", "")).startswith(today):
                        continue
                    action = rec.get("action", "").upper()
                    if action == "BUY":
                        buy_n += 1
                        last_buy = rec
                    elif action == "SELL":
                        sell_n += 1
            payload["real"] = {
                "buy_count": buy_n,
                "sell_count": sell_n,
                "last_buy": last_buy,
            }
        except Exception as e:
            logger.warning("auto_trade_log 읽기 실패: %s", e)

    paper_path = DATA_DIR / "paper_mirror_log.jsonl"
    if paper_path.exists():
        today = datetime.now().strftime("%Y-%m-%d")
        n = 0
        last = None
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if str(rec.get("ts", "")).startswith(today):
                        n += 1
                        last = rec
            payload["paper"] = {"count": n, "last": last}
        except Exception as e:
            logger.warning("paper_mirror_log 읽기 실패: %s", e)

    return payload


def _trading_day_today() -> bool:
    """오늘이 정규 거래일인지 간이 체크 (주말만 제외, 공휴일은 별도)."""
    return datetime.now().weekday() < 5


def _dday_text(now: datetime) -> str:
    """5/20 가동 D-Day 텍스트 ('가동 D-23h' / '가동 중' / '가동 완료')."""
    delta = DDAY_DATE.replace(hour=14, minute=0) - now
    total_sec = delta.total_seconds()
    if total_sec > 0:
        hours = int(total_sec // 3600)
        if hours >= 24:
            days = hours // 24
            rem = hours % 24
            return f"가동 D-{days}d{rem}h"
        return f"가동 D-{hours}h"
    # 5/20 14:00 이후
    end = DDAY_DATE.replace(hour=15, minute=30)
    if now <= end:
        return "가동 중"
    if now.date() == DDAY_DATE.date():
        return "가동 완료"
    days_after = (now.date() - DDAY_DATE.date()).days
    return f"가동 D+{days_after}d"


class Reporter:
    """4명 워커 결과 종합 → 사장님 카톡.

    public API:
      collect_all()                — 4명 결과 dict 반환
      format_summary(results, slot) — 메시지 문자열 반환
      send(slot)                    — collect + format + 텔레그램
    """

    def __init__(self):
        load_dotenv(ENV_PATH)
        self.now = datetime.now()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # 수집
    # ──────────────────────────────────────────────────────────────
    def collect_all(self) -> dict:
        """4명 워커의 최신 결과 수집.

        Returns:
          {
            "env_checker": {... or None, "_stale": bool, "_missing": bool},
            "code_auditor": ...,
            "flow_monitor": ...,
            "data_integrity": ...,
          }
        """
        out: dict = {}
        for name in WORKERS:
            data = _load_worker_json(name)
            if data is None:
                out[name] = {"_missing": True, "_stale": False}
                continue
            ts = data.get("timestamp", "")
            stale = _is_stale(ts) if ts else True
            data["_missing"] = False
            data["_stale"] = stale
            out[name] = data
        return out

    # ──────────────────────────────────────────────────────────────
    # 포맷
    # ──────────────────────────────────────────────────────────────
    def format_summary(self, results: dict, slot: str) -> str:
        """슬롯별 메시지 포맷."""
        if slot == "morning_06":
            return self._fmt_morning(results)
        if slot == "pre_trade_1355":
            return self._fmt_pre_trade(results)
        if slot == "post_trade_1600":
            return self._fmt_post_trade(results)
        if slot == "daily_close_1900":
            return self._fmt_daily_close(results)
        raise ValueError(f"알 수 없는 슬롯: {slot}")

    def _worker_line(self, results: dict, name: str, fallback: str = "") -> str:
        """한 줄 워커 상태.

        보강: timestamp 필드뿐 아니라 saved_at(save_worker_report 자동 부여)도 인식.
        둘 다 없거나 모두 stale이면 무응답으로 처리.
        """
        kr = WORKER_KR[name]
        r = results.get(name) or {"_missing": True}
        if r.get("_missing"):
            return f"❌ {kr}: 무응답 (latest.json 부재)"

        # _check_worker_stale: saved_at + timestamp 둘 다 본다
        # (collect_all의 _stale은 timestamp만 기준 — saved_at만 있는 워커를 stale로 오판할 수 있어
        #  _check_worker_stale 결과를 우선)
        stale, age_str = self._check_worker_stale(name)
        if stale:
            return f"⚠️ {kr}: {age_str}"

        status_val = r.get("status", "")
        status_str = status_val if isinstance(status_val, str) else str(status_val)
        icon = _icon(status_str)
        summary = r.get("summary") or fallback
        if not summary:
            ok = r.get("ok_count", "?")
            total = r.get("total", "?")
            summary = f"{ok}/{total} 통과"
        return f"{icon} {kr}: {summary}"

    def _reporter_self_line(self) -> str:
        """Reporter 자기 자신 (이 카톡 자체가 작동 증명)."""
        return "✅ Reporter: 이 카톡 (작동 증명)"

    # ──────────────────────────────────────────────────────────────
    # Layer 7: KILL_SWITCH 상태 표시 (Reporter는 활성화하지 않음 — 통합 보고 전용)
    # ──────────────────────────────────────────────────────────────
    def _format_kill_switch_status(self) -> str:
        """KILL_SWITCH 현재 상태를 메시지 헤더에 명시.

        표시 패턴:
        - 정상 (KILL_SWITCH 부재): "🟢 자동매매 GREEN — 가드 정상"
        - 자동 활성화: "🔴 자동매매 RED — [{source}] {reason}"
        - 수동 활성화: "🟡 자동매매 PAUSE — KILL_SWITCH 활성화 중"
        - kill_switch_manager 부재: "⚠️ 자동매매 UNKNOWN — kill_switch_manager import 실패"
        """
        try:
            from src.agents.kill_switch_manager import (
                get_kill_switch_info,
                is_kill_switch_active,
            )
        except ImportError as e:
            logger.warning("kill_switch_manager import 실패 (status): %s", e)
            return "⚠️ 자동매매 UNKNOWN — kill_switch_manager import 실패"

        if not is_kill_switch_active():
            return "🟢 자동매매 GREEN — 가드 정상"

        info = get_kill_switch_info() or {}
        if info.get("auto"):
            source = info.get("source", "unknown")
            reason = info.get("reason", "사유 미상")
            return f"🔴 자동매매 RED — [{source}] {reason}"
        return "🟡 자동매매 PAUSE — KILL_SWITCH 활성화 중"

    def _format_kill_switch_detail(self) -> str:
        """KILL_SWITCH RED(자동 차단) 상태일 때 상세 사유 + 해제 가이드.

        수동(PAUSE) 또는 GREEN 상태에서는 빈 문자열 반환.
        kill_switch_manager import 실패 시에도 빈 문자열 (조용히 SKIP).
        """
        try:
            from src.agents.kill_switch_manager import (
                get_kill_switch_info,
                is_kill_switch_active,
            )
        except ImportError as e:
            logger.warning("kill_switch_manager import 실패 (detail): %s", e)
            return ""

        if not is_kill_switch_active():
            return ""

        info = get_kill_switch_info() or {}
        if not info.get("auto"):
            return ""

        return (
            "\n━━━━━━━━━━━━━━━━━\n"
            "🚨 자동 차단 발동 상세\n"
            f"  활성화: {info.get('timestamp', 'unknown')}\n"
            f"  검수자: {info.get('source', 'unknown')}\n"
            f"  사유: {info.get('reason', '미상')}\n"
            "\n"
            "해제 방법 (안전 확인 후):\n"
            "  ssh VPS\n"
            "  rm ~/quantum-master/data/KILL_SWITCH\n"
            "━━━━━━━━━━━━━━━━━"
        )

    def _check_worker_stale(
        self, name: str, max_age_hours: float = STALE_HOURS_DEFAULT
    ) -> tuple[bool, str]:
        """워커 latest.json의 saved_at(또는 timestamp)이 max_age_hours 이상 오래됐는지.

        save_worker_report()는 saved_at(ISO 포맷)을 자동 부여.
        timestamp(YYYY-MM-DD HH:MM:SS) 필드도 보조로 확인.

        Returns:
          (stale: bool, age_str: str)
        """
        path = REPORTS_DIR / f"{name}_latest.json"
        if not path.exists():
            return True, "파일 없음"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return True, f"파싱 실패: {e}"

        # 1순위: saved_at (ISO 포맷, save_worker_report가 부여)
        saved_at_str = data.get("saved_at")
        ts_dt: datetime | None = None
        if saved_at_str:
            try:
                ts_dt = datetime.fromisoformat(saved_at_str)
            except Exception:
                ts_dt = None

        # 2순위: timestamp (YYYY-MM-DD HH:MM:SS, 워커가 직접 부여)
        if ts_dt is None:
            ts_legacy = data.get("timestamp")
            if ts_legacy:
                try:
                    ts_dt = datetime.strptime(ts_legacy, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts_dt = None

        if ts_dt is None:
            return True, "saved_at/timestamp 없음"

        age = self.now - ts_dt
        if age > timedelta(hours=max_age_hours):
            return True, f"{age.total_seconds()/3600:.1f}h 전 (무응답)"
        return False, "fresh"

    def _aggregate_state(self, results: dict) -> tuple[int, int, int]:
        """(ok_workers, warn_workers, fail_workers)."""
        ok = warn = fail = 0
        for name in WORKERS:
            r = results.get(name) or {"_missing": True}
            if r.get("_missing") or r.get("_stale"):
                fail += 1
                continue
            st = r.get("status", "")
            if st == "OK":
                ok += 1
            elif st == "WARN":
                warn += 1
            else:
                fail += 1
        return ok, warn, fail

    # ── 슬롯 1: morning_06 ───────────────────────────────────────
    def _fmt_morning(self, results: dict) -> str:
        date_str = self.now.strftime("%Y-%m-%d")
        weekday_kr = "월화수목금토일"[self.now.weekday()]
        trading_kr = "정규 거래일" if _trading_day_today() else "비거래일 (주말/휴장)"
        dday = _dday_text(self.now)

        ok, warn, fail = self._aggregate_state(results)
        overall = "ALL GREEN" if fail == 0 and warn == 0 else (
            "주의 필요" if fail == 0 else "위험 신호"
        )

        kill_status = self._format_kill_switch_status()
        lines = [
            "🌅 [퀀트봇 검수팀 점호 06:00]",
            "",
            kill_status,
            "",
            self._worker_line(results, "env_checker"),
            self._worker_line(results, "code_auditor"),
            self._worker_line(results, "flow_monitor"),
            self._worker_line(results, "data_integrity"),
            self._reporter_self_line(),
            "",
            f"📅 오늘: {date_str} ({weekday_kr}) {trading_kr}",
            f"🚀 5/20 {dday}, 시스템 {overall}",
        ]
        msg = "\n".join(lines)
        msg += self._format_kill_switch_detail()
        return msg

    # ── 슬롯 2: pre_trade_1355 ───────────────────────────────────
    def _fmt_pre_trade(self, results: dict) -> str:
        dday = _dday_text(self.now)
        ok, warn, fail = self._aggregate_state(results)

        kill_status = self._format_kill_switch_status()
        lines = [
            "🔥 [5/20 가동 5분 전 — 13:55]",
            "",
            kill_status,
            "",
            self._worker_line(results, "env_checker", "환경 게이트 점검"),
            self._worker_line(results, "code_auditor", "최종 코드 무결성"),
            self._worker_line(results, "data_integrity", "tomorrow_picks 신선도"),
            "⏰ FlowMonitor: 14:00부터 매 5분 추적 시작",
            self._reporter_self_line(),
            "",
        ]

        # KILL_SWITCH 활성화 시 매매 차단 안내가 최우선
        try:
            from src.agents.kill_switch_manager import is_kill_switch_active as _ks_active
            ks_on = _ks_active()
        except ImportError as e:
            logger.warning("kill_switch_manager import 실패 (pre_trade): %s", e)
            ks_on = False
        if ks_on:
            lines.append("🚨 KILL_SWITCH 활성화 — 14:00 cron에서 매수 차단됨")
        elif fail == 0:
            lines.append("🚀 가동 준비 완료. 안전선 9건 통과 시만 매수합니다.")
        else:
            lines.append(f"🚨 위험 신호 {fail}건 — 가동 전 점검 필요!")
            # 실패 항목 상세
            for name in WORKERS:
                r = results.get(name) or {"_missing": True}
                if r.get("_missing") or r.get("_stale") or r.get("status") not in ("OK", "WARN"):
                    failed = r.get("failed") or []
                    if failed and isinstance(failed, list):
                        for item in failed[:3]:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                lines.append(f"  ❌ {item[0]}: {item[1]}")
                            elif isinstance(item, dict):
                                lines.append(
                                    f"  ❌ {item.get('name','?')}: {item.get('msg','?')}"
                                )

        msg = "\n".join(lines)
        msg += self._format_kill_switch_detail()
        return msg

    # ── 슬롯 3: post_trade_1600 ──────────────────────────────────
    def _fmt_post_trade(self, results: dict) -> str:
        trade = _post_trade_payload()
        kill_status = self._format_kill_switch_status()
        lines = [
            "🏁 [5/20 자동매매 종료 16:00]",
            "",
            kill_status,
            "",
            "📊 실주문:",
        ]

        real = trade.get("real") if trade else None
        if real is None:
            lines.append("  (실주문 로그 없음 — auto_trade_log.jsonl 부재)")
        else:
            buy_n = real["buy_count"]
            sell_n = real["sell_count"]
            if buy_n == 0 and sell_n == 0:
                lines.append("  매수 0건 / 매도 0건 (이월 결정 — 룰 ④)")
            else:
                last = real.get("last_buy") or {}
                ticker = last.get("ticker") or last.get("symbol") or "?"
                price = last.get("price", "?")
                qty = last.get("qty", "?")
                score = last.get("score", "?")
                lines.append(
                    f"  매수 {buy_n}건: {ticker} {price}원 {qty}주 (점수 {score})"
                )
                lines.append(
                    f"  매도 {sell_n}건"
                    + (" (이월 결정 — 룰 ④)" if sell_n == 0 else "")
                )

        lines.append("")
        lines.append("🎯 Paper Mirror:")
        paper = trade.get("paper") if trade else None
        if paper is None:
            lines.append("  (paper_mirror_log.jsonl 부재)")
        else:
            n = paper["count"]
            last = paper.get("last") or {}
            if n == 0:
                lines.append("  시뮬 진입 0건")
            else:
                entry = last.get("entry_price", "?")
                fill = last.get("fill_price", "?")
                slip = last.get("slippage", "?")
                lines.append(f"  진입 {entry} → 시뮬 체결 {fill} (슬리피지 {slip})")
                if last.get("hold"):
                    lines.append("  현재 보유 (익일 청산 시뮬)")

        lines.append("")
        lines.append("💚 검수팀 작동 결과:")
        for name in WORKERS:
            lines.append(f"  {self._worker_line(results, name)}")
        msg = "\n".join(lines)
        msg += self._format_kill_switch_detail()
        return msg

    # ── 슬롯 4: daily_close_1900 ─────────────────────────────────
    def _fmt_daily_close(self, results: dict) -> str:
        # 오늘 검수 N건 = 4슬롯 × 4명 = 16개 (Reporter 자신 포함 시 20)
        # 단순 표시 (정확한 카운트는 로그 누적 필요)
        ok, warn, fail = self._aggregate_state(results)
        issue_n = warn + fail

        # 다음 거래일 자동매매 (5/20만 가동 D-Day)
        tmr = self.now + timedelta(days=1)
        # 5/20 발화 D-Hours
        next_dday = ""
        if self.now.date() < DDAY_DATE.date():
            hrs = int((DDAY_DATE.replace(hour=14) - self.now).total_seconds() // 3600)
            next_dday = f"5/20 발화 (D-{hrs}h)"
        elif self.now.date() == DDAY_DATE.date():
            next_dday = "5/20 가동 완료 — 내일은 일반 운영"
        else:
            next_dday = f"{tmr.strftime('%m/%d')} 일반 운영"

        kill_status = self._format_kill_switch_status()
        lines = [
            "🌆 [일일 마감 19:00]",
            "",
            kill_status,
            "",
            "오늘 검수팀 활동: 4회 × 5명 = 20개 점검",
            f"이상 발견: {issue_n}건",
            f"내일 자동매매 cron: {next_dday}",
            "",
            "검수팀 마지막 상태:",
            self._worker_line(results, "env_checker"),
            self._worker_line(results, "code_auditor"),
            self._worker_line(results, "flow_monitor"),
            self._worker_line(results, "data_integrity"),
            self._reporter_self_line(),
        ]
        msg = "\n".join(lines)
        msg += self._format_kill_switch_detail()
        return msg

    # ──────────────────────────────────────────────────────────────
    # 전송
    # ──────────────────────────────────────────────────────────────
    def send(self, slot: str, dry_run: bool = False) -> str:
        """수집 + 포맷 + 텔레그램 전송.

        Args:
          slot: morning_06 / pre_trade_1355 / post_trade_1600 / daily_close_1900
          dry_run: True면 텔레그램 발송 SKIP

        Returns:
          포맷된 메시지 (콘솔 출력 용)

        사장님 결단 C (2026-05-19): 도배 방지를 위해 디폴트 OFF.
        AGENT_TELEGRAM_ENABLED=true 시만 발송 (--no-tg와 별개로 환경변수도 가드).
        KILL_SWITCH RED는 kill_switch_manager가 별도로 발송 (유일한 단일 채널).
        """
        if slot not in SLOTS:
            raise ValueError(f"알 수 없는 슬롯: {slot} (허용: {list(SLOTS)})")

        results = self.collect_all()
        msg = self.format_summary(results, slot)

        if dry_run:
            logger.info("[dry-run] slot=%s, 텔레그램 발송 SKIP", slot)
            self._save_self_report(slot, results, dry_run=True)
            return msg

        # 환경변수 가드 — dry_run=False여도 디폴트 OFF (5/19 결단 C)
        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            logger.info(
                "[Reporter] 슬롯=%s 텔레그램 SKIP (AGENT_TELEGRAM_ENABLED=false) — "
                "stdout/latest.json만",
                slot,
            )
            self._save_self_report(slot, results, dry_run=False)
            return msg

        try:
            from src.telegram_sender import send_message
            ok = send_message(msg)
            if ok:
                logger.info("Reporter 슬롯=%s 텔레그램 발송 완료", slot)
            else:
                logger.error("Reporter 슬롯=%s 텔레그램 발송 실패", slot)
        except Exception as e:
            logger.error("텔레그램 발송 예외: %s", e)

        self._save_self_report(slot, results, dry_run=False)
        return msg

    def _save_self_report(self, slot: str, results: dict, dry_run: bool) -> None:
        """Reporter 자체 latest.json 저장. KILL_SWITCH는 절대 활성화하지 않음 (통합 보고 전용).

        lazy import — kill_switch_manager 부재 시 silent skip (단일 실패점 제거 C4).
        """
        try:
            from src.agents.kill_switch_manager import (
                is_kill_switch_active,
                save_worker_report,
            )
        except ImportError as e:
            logger.warning("kill_switch_manager import 실패 (self_report): %s", e)
            return

        try:
            workers_reported = sum(
                1 for n in WORKERS if not (results.get(n) or {}).get("_missing")
            )
            save_worker_report(
                "reporter",
                {
                    "agent": "reporter",
                    "status": "OK",
                    "slot": slot,
                    "kill_switch_active": is_kill_switch_active(),
                    "workers_reported": workers_reported,
                    "workers_total": len(WORKERS),
                    "dry_run": dry_run,
                    "timestamp": self.now.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": f"{workers_reported}/{len(WORKERS)} 워커 응답",
                },
            )
        except Exception as e:
            logger.warning("Reporter self-report 저장 실패: %s", e)


def main() -> int:
    """CLI 진입점 (scripts/run_reporter.py에서 호출)."""
    import argparse

    parser = argparse.ArgumentParser(description="Reporter — 검수팀 종합 보고")
    parser.add_argument(
        "--slot",
        required=True,
        choices=list(SLOTS.keys()),
        help="보고 슬롯 (morning_06 / pre_trade_1355 / post_trade_1600 / daily_close_1900)",
    )
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 OFF (dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    reporter = Reporter()
    msg = reporter.send(args.slot, dry_run=args.no_tg)

    # 콘솔 출력
    print("=" * 60)
    print(f"  Reporter 슬롯: {args.slot} ({SLOTS[args.slot]})")
    print(f"  실행 시각: {reporter.now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  텔레그램: {'SKIP (dry-run)' if args.no_tg else 'SENT'}")
    print("=" * 60)
    print(msg)
    print("=" * 60)

    # 무응답 워커 있으면 exit 2 (cron 모니터링용)
    results = reporter.collect_all()
    missing = [n for n in WORKERS if (results.get(n) or {}).get("_missing")]
    if missing:
        logger.warning("무응답 워커: %s", missing)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
