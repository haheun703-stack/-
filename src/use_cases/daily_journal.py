"""
장마감 일일 업무일지 생성기

16:30 스케줄: 오늘의 거래 내역, 포지션 현황, 스케줄러 실행 결과를 HTML로 기록.
- 보유 포지션 현황 + P&L
- 오늘 매수/매도 거래 내역
- 스케줄러 Phase별 실행 결과
- 내일 매수 후보 시그널
- 오늘의 교훈/메모

의존성:
  - data/positions.json (보유 포지션)
  - data/trades_history.json (거래 이력)
  - data/sector_rotation/krx_sector_scan.json (매수 후보)
  - logs/scheduler.log (스케줄러 로그)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.use_cases.position_tracker import PositionTracker

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/daily")
TEMPLATES_DIR = Path("templates")
TRADES_FILE = Path("data/trades_history.json")
SECTOR_SCAN_FILE = Path("data/sector_rotation/krx_sector_scan.json")
SCHEDULER_LOG = Path("logs/scheduler.log")


@dataclass
class JournalData:
    """일일 업무일지 데이터"""

    date: str = ""
    day_type: str = ""  # "거래일", "휴일"
    kospi_str: str = ""
    total_eval: int = 0
    daily_pnl: int = 0
    daily_pnl_pct: float = 0.0
    trade_count: int = 0
    phases: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    generated_time: str = ""


class DailyJournalWriter:
    """장마감 일일 업무일지 생성기"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def generate(self) -> Path | None:
        """
        일일 업무일지 생성 + HTML 저장.

        Returns:
            저장된 HTML 파일 경로
        """
        today = datetime.now().strftime("%Y-%m-%d")
        journal = JournalData(
            date=today,
            day_type="거래일",
            generated_time=datetime.now().strftime("%H:%M"),
        )

        # 1. 포지션 현황
        self._load_positions(journal)

        # 2. 오늘 거래 내역
        self._load_trades(journal, today)

        # 3. 스케줄러 실행 이력
        self._load_phase_results(journal, today)

        # 4. 내일 매수 후보
        self._load_signals(journal)

        # 5. 오늘의 교훈 자동 생성
        self._generate_lessons(journal)

        # 6. HTML 렌더링 + 저장
        html = self._render_html(journal)

        save_dir = REPORTS_DIR / today
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "eod_journal.html"
        save_path.write_text(html, encoding="utf-8")
        logger.info("[업무일지] 저장: %s", save_path)

        return save_path

    # ──────────────────────────────────────────
    # 데이터 수집
    # ──────────────────────────────────────────

    def _load_positions(self, journal: JournalData) -> None:
        """포지션 현황 로드"""
        try:
            tracker = PositionTracker(self.config)
            summary = tracker.get_summary()
            journal.total_eval = summary.get("total_eval", 0)
            journal.daily_pnl = summary.get("total_pnl", 0)
            journal.daily_pnl_pct = summary.get("total_pnl_pct", 0)

            today = datetime.now().strftime("%Y-%m-%d")
            for pos_data in summary.get("positions", []):
                try:
                    hold_days = 0
                    # find actual position for entry_date
                    for p in tracker.positions:
                        if p.ticker == pos_data["ticker"]:
                            hold_days = (
                                datetime.strptime(today, "%Y-%m-%d")
                                - datetime.strptime(p.entry_date, "%Y-%m-%d")
                            ).days
                            break
                except Exception:
                    hold_days = 0

                journal.positions.append({
                    "ticker": pos_data.get("ticker", ""),
                    "name": pos_data.get("name", ""),
                    "shares": pos_data.get("shares", 0),
                    "entry_price": pos_data.get("entry_price", 0),
                    "current_price": pos_data.get("current_price", 0),
                    "pnl_pct": pos_data.get("pnl_pct", 0),
                    "grade": pos_data.get("grade", "?"),
                    "hold_days": hold_days,
                })
        except Exception as e:
            logger.error("[업무일지] 포지션 로드 실패: %s", e)

    def _load_trades(self, journal: JournalData, today: str) -> None:
        """오늘 거래 내역 로드"""
        if not TRADES_FILE.exists():
            return

        try:
            trades = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
            today_trades = [t for t in trades if t.get("exit_date", "").startswith(today)]

            for t in today_trades:
                direction = "sell"  # trades_history는 매도 기록
                journal.trades.append({
                    "time": t.get("exit_date", "").split(" ")[-1] if " " in t.get("exit_date", "") else "",
                    "name": t.get("name", t.get("ticker", "")),
                    "direction": direction,
                    "direction_label": "매도" if direction == "sell" else "매수",
                    "shares": t.get("shares", 0),
                    "price": t.get("exit_price", 0),
                    "reason": t.get("exit_reason", ""),
                })

            journal.trade_count = len(today_trades)
        except Exception as e:
            logger.error("[업무일지] 거래 내역 로드 실패: %s", e)

    def _load_phase_results(self, journal: JournalData, today: str) -> None:
        """스케줄러 실행 결과 파싱"""
        phase_defs = [
            ("00:00", "Phase 1", "일일 리셋"),
            ("07:00", "Phase 2", "매크로 수집"),
            ("07:10", "Phase 3", "뉴스/리포트"),
            ("08:20", "Phase 4", "매매 준비"),
            ("08:25", "Phase 4.5", "장전 리포트"),
            ("09:02", "Phase 5", "매수 실행"),
            ("09:10", "Phase 6", "장중 모니터링"),
            ("15:25", "Phase 7", "매도 실행"),
            ("15:35", "Phase 8", "장마감 파이프라인"),
        ]

        # 로그에서 Phase 실행 결과 파싱
        log_phases = self._parse_scheduler_log(today)

        for time_str, name, desc in phase_defs:
            phase_key = name.replace(" ", "").lower()
            log_entry = log_phases.get(phase_key, {})

            status = log_entry.get("status", "skip")
            note = log_entry.get("note", "미실행")

            journal.phases.append({
                "time": time_str,
                "name": f"{name}: {desc}",
                "status": status,
                "status_label": {"ok": "완료", "skip": "스킵", "fail": "실패"}.get(status, "?"),
                "note": note,
            })

    def _parse_scheduler_log(self, today: str) -> dict:
        """스케줄러 로그에서 오늘의 Phase 결과 파싱"""
        results = {}
        if not SCHEDULER_LOG.exists():
            return results

        try:
            log_text = SCHEDULER_LOG.read_text(encoding="utf-8", errors="ignore")
            for line in log_text.split("\n"):
                if today not in line:
                    continue

                # [Phase N] 패턴 매칭
                match = re.search(r"\[Phase\s+(\S+)\]\s+(.*?)(?:완료|실패|스킵)", line)
                if match:
                    phase_id = f"phase{match.group(1).split('-')[0]}"
                    if "완료" in line:
                        results[phase_id] = {"status": "ok", "note": match.group(2).strip()}
                    elif "실패" in line:
                        results[phase_id] = {"status": "fail", "note": match.group(2).strip()}
                    elif "스킵" in line:
                        results[phase_id] = {"status": "skip", "note": match.group(2).strip()}
        except Exception as e:
            logger.debug("[업무일지] 로그 파싱 실패: %s", e)

        return results

    def _load_signals(self, journal: JournalData) -> None:
        """내일 매수 후보 시그널 로드 (섹터 로테이션 스캔)"""
        if not SECTOR_SCAN_FILE.exists():
            return

        try:
            import json
            with open(SECTOR_SCAN_FILE, encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("smart_money", [])[:5]:
                journal.signals.append({
                    "ticker": str(item.get("ticker", "")).zfill(6),
                    "name": item.get("name", ""),
                    "grade": "SMART",
                    "zone_score": 0,
                    "trigger": f"BB{item.get('bb_pct', 0):.0f}% RSI{item.get('rsi', 0):.0f}",
                    "entry_price": 0,
                    "stop_loss": item.get("stop_pct", -7),
                    "rr_ratio": 0,
                })
            for item in data.get("theme_money", [])[:5]:
                journal.signals.append({
                    "ticker": str(item.get("ticker", "")).zfill(6),
                    "name": item.get("name", ""),
                    "grade": "THEME",
                    "zone_score": 0,
                    "trigger": f"ADX{item.get('adx', 0):.0f} RSI{item.get('rsi', 0):.0f}",
                    "entry_price": 0,
                    "stop_loss": item.get("stop_pct", -7),
                    "rr_ratio": 0,
                })
        except Exception as e:
            logger.error("[업무일지] 시그널 로드 실패: %s", e)

    def _generate_lessons(self, journal: JournalData) -> None:
        """오늘의 교훈 자동 생성"""
        # 거래 기반 교훈
        for t in journal.trades:
            reason = t.get("reason", "")
            if "stop_loss" in reason:
                journal.lessons.append(f"{t['name']} 손절 — 진입 시 리스크 평가 재점검 필요")
            elif "partial" in reason:
                journal.lessons.append(f"{t['name']} 부분 익절 — 분할 청산 전략 실행 중")
            elif "max_hold" in reason:
                journal.lessons.append(f"{t['name']} 최대 보유일 도달 — 타이밍 재검토")

        # 포지션 기반 교훈
        for pos in journal.positions:
            if pos.get("pnl_pct", 0) <= -5:
                journal.lessons.append(
                    f"{pos['name']} 수익률 {pos['pnl_pct']:+.1f}% — 손절선 확인 필요"
                )
            elif pos.get("pnl_pct", 0) >= 10:
                journal.lessons.append(
                    f"{pos['name']} 수익률 {pos['pnl_pct']:+.1f}% — 이익 보전 전략 점검"
                )

        if not journal.lessons:
            journal.lessons.append("특이사항 없음 — 계획대로 운영 중")

    # ──────────────────────────────────────────
    # HTML 렌더링
    # ──────────────────────────────────────────

    def _render_html(self, journal: JournalData) -> str:
        """Jinja2로 HTML 렌더링"""
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=False,
        )
        template = env.get_template("daily_journal.html")
        return template.render(journal=journal)
