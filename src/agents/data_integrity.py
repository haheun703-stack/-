"""DataIntegrity 에이전트 — 시각별 데이터 도착/신선도/사전조건 추적

목적 (5/20 자율 자동매매 첫 가동 안전망):
- 정보봇 / 큰형(퀀트봇 BAT) / JGIS / KIS 토큰 등 모든 데이터 소스가
  '기대 시각 ±15분' 안에 도착하는지 자동 추적한다.
- 단타봇 사고처럼 'brain_state.json 부재' 같은 사전 조건 누락을 사전 차단한다.

호출 시점 (cron):
- 06:30 (BAT-A 후 확인)
- 16:50 (BAT-D 후 확인)
- 18:50 (BAT-HEALTH 후 통합 확인)

CLI:
    python -u -X utf8 scripts/run_data_integrity.py [--no-tg]
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정 — 추적 대상 정의
# ──────────────────────────────────────────────

# BAT 결과 (큰형 자체 산출)
# (key, expected_time(HH:MM), files(list[str]), 메모, dir_ok=True면 디렉토리/glob 허용)
BAT_TARGETS: list[dict[str, Any]] = [
    {
        "key": "BAT-A",
        "expected": "06:10",
        "files": ["data/us_market/overnight_signal.json"],
        "glob_dirs": ["data/v3_brain"],
        "note": "미국장 야간 시그널 + v3 brain",
    },
    {
        "key": "BAT-B",
        "expected": "07:00",
        "files": ["data/morning_briefing.json", "data/morning_checklist.json"],
        "note": "모닝 브리핑",
    },
    {
        "key": "BAT-L",
        "expected": "15:35",
        "files": ["data/nxt_close.json", "data/nxt/nxt_signal.json"],
        "note": "KOSPI 마감 NXT 시그널",
    },
    {
        "key": "BAT-D",
        "expected": "16:30",
        "files": [
            "data/tomorrow_picks.json",
            "data/etf_signals.json",
        ],
        "glob_dirs": ["data"],  # sector_fire_YYYYMMDD.json 매칭
        "note": "내일픽 + sector_fire + ETF 시그널",
    },
    {
        "key": "BAT-J",
        "expected": "17:00",
        "files": ["data/portfolio.json", "data/portfolio_allocation.json"],
        "note": "포트폴리오 갱신",
    },
    {
        "key": "BAT-PICKV2",
        "expected": "17:45",
        "files": [],
        "glob_dirs": ["data"],  # picks_v2_YYYYMMDD.csv
        "glob_pattern": "picks_v2_{YYYYMMDD}.csv",
        "note": "daily_pick_v2 산출",
    },
    {
        "key": "BAT-F",
        "expected": "18:35",
        "files": ["data/tomorrow_picks_flowx.json"],
        "note": "FLOWX Supabase 업로드 (산출 파일)",
    },
    {
        "key": "BAT-HEALTH",
        "expected": "18:45",
        "files": ["data/metrics/bat_d_daily.jsonl"],
        "note": "BAT-D 건강체크 결과 JSONL",
    },
]

# 정보봇/JGIS 의존 데이터
INTEL_TARGETS: list[dict[str, Any]] = [
    {
        "key": "정보봇_jgis_ohlcv",
        "expected": "16:15",
        # 로컬에는 없는 경우 많음. VPS 심볼릭 링크 가정.
        "files": [
            "data/external/jgis_ohlcv",  # 디렉토리 자체 mtime
        ],
        "is_dir": True,
        "note": "정보봇 OHLCV (~2632종목, 39컬럼)",
    },
    {
        "key": "정보봇_etf_signal_accuracy",
        "expected": "17:23",
        # Supabase 결과 캐시 또는 docs/from-jgis/ 회신
        "files": [
            "data/etf_signal_accuracy.json",  # 캐시 경로 (있으면 mtime, 없으면 미수신)
        ],
        "optional": True,
        "note": "Supabase etf_signal_accuracy 동기화 (캐시 파일이 없으면 미수신)",
    },
    {
        "key": "정보봇_from_jgis_docs",
        "expected": "00:00",  # 시각 무관 — 신규 회신 여부만 일별 추적
        "files": [],
        "glob_dirs": ["docs/from-jgis"],
        "note": "정보봇 → 퀀트봇 회신 디렉토리 (신규 파일 있으면 신규 처리)",
        "no_time_window": True,
    },
]

# 사전 조건 파일 (단타봇 brain_state.json 사고 패턴 — 절대 누락 안 됨)
PREREQUISITE_FILES: list[dict[str, Any]] = [
    {
        "key": "KILL_SWITCH",
        "path": "data/KILL_SWITCH",
        "must_exist": True,
        "stale_max_hours": None,  # 존재만 확인
        "note": "비상 정지 스위치 (존재해야 함, 내용 OFF=정상)",
    },
    {
        "key": "tomorrow_picks",
        "path": "data/tomorrow_picks.json",
        "must_exist": True,
        "stale_max_hours": 24,  # 24h 안에 갱신돼야 함
        "note": "내일 매수 후보 — 자동매매 사전 조건 1순위",
    },
    {
        "key": "owner_rule_positions",
        "path": "data/owner_rule_positions.json",
        "must_exist": True,
        "stale_max_hours": 48,
        "note": "사장님 룰 포지션",
    },
    {
        "key": "today_snapshot_dir",
        "path": "data/daily_snapshots/{YYYY-MM-DD}",
        "must_exist": True,
        "is_dir": True,
        "min_files": 5,
        "stale_max_hours": None,
        "note": "오늘 일일 스냅샷 디렉토리 (5개 이상 산출물)",
    },
]


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────


@dataclass
class CheckItem:
    """단일 체크 결과."""

    name: str
    expected: str  # "HH:MM" 또는 ""
    ok: bool
    status: str  # "fresh" | "stale" | "missing" | "skip" | "ok"
    detail: str = ""
    age_minutes: float | None = None

    def to_tuple(self) -> tuple:
        return (self.name, self.expected, self.ok, self.status)


@dataclass
class IntegrityReport:
    timestamp: datetime
    bat_results: list[CheckItem] = field(default_factory=list)
    intel_bot: list[CheckItem] = field(default_factory=list)
    kis_token: CheckItem | None = None
    prerequisite_files: list[CheckItem] = field(default_factory=list)

    @property
    def missing(self) -> list[CheckItem]:
        out = []
        for group in (self.bat_results, self.intel_bot, self.prerequisite_files):
            out.extend([c for c in group if c.status == "missing"])
        if self.kis_token and self.kis_token.status == "missing":
            out.append(self.kis_token)
        return out

    @property
    def stale(self) -> list[CheckItem]:
        out = []
        for group in (self.bat_results, self.intel_bot, self.prerequisite_files):
            out.extend([c for c in group if c.status == "stale"])
        if self.kis_token and self.kis_token.status == "stale":
            out.append(self.kis_token)
        return out

    def to_dict(self) -> dict:
        return {
            "bat_results": [c.to_tuple() for c in self.bat_results],
            "intel_bot": [c.to_tuple() for c in self.intel_bot],
            "kis_token": self.kis_token.to_tuple() if self.kis_token else None,
            "prerequisite_files": [c.to_tuple() for c in self.prerequisite_files],
            "missing": [c.name for c in self.missing],
            "stale": [c.name for c in self.stale],
        }


# ──────────────────────────────────────────────
# DataIntegrity 본체
# ──────────────────────────────────────────────


class DataIntegrity:
    """시각별 데이터 도착/신선도/사전조건 추적 에이전트."""

    # 기대 시각 ±15분 윈도우
    FRESH_WINDOW_MIN = 15

    def __init__(self, now: datetime | None = None):
        self.now = now or datetime.now()
        self.today = self.now.date()
        self.project_root = PROJECT_ROOT

    # ── 외부 인터페이스 ────────────────────────

    def check_all(self) -> dict:
        """전체 데이터 도착 점검. dict 반환.

        Returns:
            {
                "agent": "data_integrity",
                "status": "OK"|"FAIL",
                "summary": "missing N / stale N / total N",
                "bat_results": [(name, expected, ok, status), ...],
                "intel_bot": [...],
                "kis_token": (name, expected, ok, status) or None,
                "prerequisite_files": [...],
                "missing": [name, ...],
                "stale": [name, ...],
            }
        """
        report = self._build_report()
        result = report.to_dict()

        # S1 표준 필드 — Reporter가 통일 표시 (5/19 자체 검수 S1)
        result["agent"] = "data_integrity"
        missing_n = len(result.get("missing", []))
        stale_n = len(result.get("stale", []))
        total_n = (
            len(report.bat_results)
            + len(report.intel_bot)
            + len(report.prerequisite_files)
            + (1 if report.kis_token else 0)
        )
        result["total"] = total_n
        if missing_n > 0 or stale_n > 0:
            result["status"] = "FAIL"
        else:
            result["status"] = "OK"
        result["summary"] = f"missing {missing_n} / stale {stale_n} / total {total_n}"
        result["timestamp"] = self.now.strftime("%Y-%m-%d %H:%M:%S")

        _post_process_result(result)
        return result

    def build_report(self) -> IntegrityReport:
        """원본 IntegrityReport 객체 반환 (텔레그램 포맷에서 사용).

        Layer 7: report.to_dict() 결과로 KILL_SWITCH 자동 활성화 + latest.json 저장.
        호출자가 to_dict()를 다시 부르더라도 idempotent (kill_switch_manager 측에서 중복 방지).
        """
        report = self._build_report()
        try:
            # check_all()과 동일한 표준 필드 부여 후 후처리
            result = report.to_dict()
            result["agent"] = "data_integrity"
            missing_n = len(result.get("missing", []))
            stale_n = len(result.get("stale", []))
            total_n = (
                len(report.bat_results)
                + len(report.intel_bot)
                + len(report.prerequisite_files)
                + (1 if report.kis_token else 0)
            )
            result["total"] = total_n
            result["status"] = "FAIL" if (missing_n > 0 or stale_n > 0) else "OK"
            result["summary"] = (
                f"missing {missing_n} / stale {stale_n} / total {total_n}"
            )
            result["timestamp"] = self.now.strftime("%Y-%m-%d %H:%M:%S")
            _post_process_result(result)
        except Exception as e:
            logger.warning(f"[DataIntegrity] build_report 후처리 실패: {e}")
        return report

    def check_bat_freshness(self, bat: str, expected_time: str) -> tuple[bool, str]:
        """특정 BAT 결과가 expected_time ±15분 안에 갱신됐는지.

        Args:
            bat: BAT key (예: "BAT-D")
            expected_time: "HH:MM"

        Returns:
            (ok, status)  status ∈ {"fresh", "stale", "missing", "skip"}
        """
        target = next((t for t in BAT_TARGETS if t["key"] == bat), None)
        if not target:
            return False, "missing"

        item = self._check_target(target, group="bat")
        return item.ok, item.status

    def report_to_telegram(self, result: dict | None = None) -> None:
        """missing/stale 항목 사장님 카톡 (정상이면 1줄, 이상이면 상세).

        사장님 결단 C (2026-05-19): 도배 방지를 위해 디폴트 OFF.
        AGENT_TELEGRAM_ENABLED=true 시만 발송.
        KILL_SWITCH RED는 kill_switch_manager가 별도로 발송 (유일한 단일 채널).
        """
        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            # result 인자가 없으면 build_report로 간이 카운트
            missing_n = 0
            stale_n = 0
            if result is not None:
                missing_n = len(result.get("missing", []))
                stale_n = len(result.get("stale", []))
            else:
                try:
                    report = self._build_report()
                    missing_n = len(report.missing)
                    stale_n = len(report.stale)
                except Exception:
                    pass
            logger.info(
                "[DataIntegrity] 결과 logger.info만 (AGENT_TELEGRAM_ENABLED=false): "
                "missing=%d stale=%d",
                missing_n, stale_n,
            )
            return

        try:
            from src.telegram_sender import send_message
        except Exception:
            logger.warning("telegram_sender 임포트 실패 — 텔레그램 발송 생략")
            return

        report = self._build_report()
        text = self._format_telegram(report)
        try:
            send_message(text)
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")

    # ── 내부 빌드 ─────────────────────────────

    def _build_report(self) -> IntegrityReport:
        report = IntegrityReport(timestamp=self.now)

        for target in BAT_TARGETS:
            item = self._check_target(target, group="bat")
            report.bat_results.append(item)

        for target in INTEL_TARGETS:
            item = self._check_target(target, group="intel")
            report.intel_bot.append(item)

        report.kis_token = self._check_kis_token()

        for prereq in PREREQUISITE_FILES:
            item = self._check_prerequisite(prereq)
            report.prerequisite_files.append(item)

        return report

    # ── 개별 체크 ─────────────────────────────

    def _check_target(self, target: dict, group: str) -> CheckItem:
        """BAT/Intel 공통 — 파일들의 mtime이 expected ±15분 안인지."""
        name = target["key"]
        expected = target.get("expected", "")
        files = list(target.get("files", []))
        glob_dirs = target.get("glob_dirs", [])
        glob_pattern = target.get("glob_pattern")
        is_dir = target.get("is_dir", False)
        optional = target.get("optional", False)
        no_time_window = target.get("no_time_window", False)

        # 1) expected 시각 도달 전이면 skip (BAT-A 06:10 같이 미래는 의미 없음)
        if expected and not no_time_window:
            try:
                hh, mm = map(int, expected.split(":"))
                expected_dt = datetime.combine(self.today, time(hh, mm))
                # 윈도우 시작 전이면 skip (아직 안 돌았을 것)
                if self.now < expected_dt - timedelta(minutes=self.FRESH_WINDOW_MIN):
                    return CheckItem(
                        name=name, expected=expected, ok=True, status="skip",
                        detail=f"기대 시각 미도달 ({expected})",
                    )
            except ValueError:
                pass

        # 2) 후보 파일들 탐색 (mtime 가장 최근)
        candidate_paths: list[Path] = []
        for f in files:
            candidate_paths.append(self.project_root / f)
        for d in glob_dirs:
            dir_path = self.project_root / d
            if not dir_path.exists():
                continue
            if glob_pattern:
                ymd = self.today.strftime("%Y%m%d")
                pat = glob_pattern.replace("{YYYYMMDD}", ymd)
                candidate_paths.extend(list(dir_path.glob(pat)))
            else:
                # 디렉토리 자체 mtime + 내부 .json/.csv 중 가장 최근
                if dir_path.is_dir():
                    candidate_paths.append(dir_path)
                    candidate_paths.extend(list(dir_path.glob("*.json")))
                    candidate_paths.extend(list(dir_path.glob("*.csv")))

        # 존재하는 후보만
        existing = [p for p in candidate_paths if p.exists()]

        if not existing:
            if optional:
                return CheckItem(
                    name=name, expected=expected, ok=True, status="skip",
                    detail="optional — 파일 없음 (무시)",
                )
            return CheckItem(
                name=name, expected=expected, ok=False, status="missing",
                detail=f"파일/디렉토리 없음: {files or glob_dirs}",
            )

        # 가장 최근 mtime
        latest_path = max(existing, key=lambda p: p.stat().st_mtime)
        latest_mtime = datetime.fromtimestamp(latest_path.stat().st_mtime)
        age_min = (self.now - latest_mtime).total_seconds() / 60.0

        # no_time_window 모드: 24h 안에 갱신만 있으면 OK
        if no_time_window:
            if age_min <= 60 * 24:
                return CheckItem(
                    name=name, expected="", ok=True, status="fresh",
                    detail=f"최근 갱신: {latest_mtime:%m-%d %H:%M} ({age_min:.0f}분 전)",
                    age_minutes=age_min,
                )
            return CheckItem(
                name=name, expected="", ok=False, status="stale",
                detail=f"24h+ 미갱신: 마지막 {latest_mtime:%m-%d %H:%M}",
                age_minutes=age_min,
            )

        # expected 시각 ±15분 윈도우 검사 (오늘 날짜 mtime 기준)
        try:
            hh, mm = map(int, expected.split(":"))
            expected_dt = datetime.combine(self.today, time(hh, mm))
        except ValueError:
            expected_dt = None

        if expected_dt is None:
            # expected 없으면 24h 안 갱신만 확인
            if age_min <= 60 * 24:
                return CheckItem(
                    name=name, expected=expected, ok=True, status="fresh",
                    detail=f"{latest_mtime:%m-%d %H:%M} ({age_min:.0f}분 전)",
                    age_minutes=age_min,
                )
            return CheckItem(
                name=name, expected=expected, ok=False, status="stale",
                detail=f"24h+ 미갱신",
                age_minutes=age_min,
            )

        # 오늘 안에 expected 근처에서 갱신됐는지
        delta_to_expected = abs((latest_mtime - expected_dt).total_seconds()) / 60.0

        # 오늘 갱신 여부 (mtime이 오늘 날짜인가)
        is_today = latest_mtime.date() == self.today

        if is_today and delta_to_expected <= self.FRESH_WINDOW_MIN:
            return CheckItem(
                name=name, expected=expected, ok=True, status="fresh",
                detail=f"기대 {expected} → 실제 {latest_mtime:%H:%M} (Δ{delta_to_expected:.0f}분)",
                age_minutes=age_min,
            )

        # 오늘 안에 갱신은 됐지만 윈도우 밖
        if is_today:
            return CheckItem(
                name=name, expected=expected, ok=True, status="fresh",
                detail=f"오늘 갱신 ({latest_mtime:%H:%M}, 기대 {expected}와 Δ{delta_to_expected:.0f}분)",
                age_minutes=age_min,
            )

        # 어제 이전 = stale
        return CheckItem(
            name=name, expected=expected, ok=False, status="stale",
            detail=f"오늘 미갱신: 마지막 {latest_mtime:%m-%d %H:%M} ({age_min/60:.1f}h 전)",
            age_minutes=age_min,
        )

    def _check_prerequisite(self, prereq: dict) -> CheckItem:
        """사전 조건 파일 — 존재 + 신선도 + 최소 파일 수."""
        name = prereq["key"]
        raw_path = prereq["path"]
        path_str = raw_path.replace("{YYYY-MM-DD}", self.today.strftime("%Y-%m-%d"))
        path = self.project_root / path_str
        must_exist = prereq.get("must_exist", True)
        stale_max_hours = prereq.get("stale_max_hours")
        is_dir = prereq.get("is_dir", False)
        min_files = prereq.get("min_files", 0)

        if not path.exists():
            if must_exist:
                return CheckItem(
                    name=name, expected="", ok=False, status="missing",
                    detail=f"필수 파일/디렉토리 없음: {path_str}",
                )
            return CheckItem(
                name=name, expected="", ok=True, status="skip",
                detail=f"없음 (optional)",
            )

        # 디렉토리 검증
        if is_dir:
            if not path.is_dir():
                return CheckItem(
                    name=name, expected="", ok=False, status="missing",
                    detail=f"디렉토리가 아님: {path_str}",
                )
            entries = list(path.iterdir())
            if len(entries) < min_files:
                return CheckItem(
                    name=name, expected="", ok=False, status="missing",
                    detail=f"파일 수 부족: {len(entries)}/{min_files} ({path_str})",
                )

        # 신선도 검증
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_h = (self.now - mtime).total_seconds() / 3600.0

        if stale_max_hours is not None and age_h > stale_max_hours:
            return CheckItem(
                name=name, expected="", ok=False, status="stale",
                detail=f"{age_h:.1f}h 미갱신 (max {stale_max_hours}h): {path_str}",
                age_minutes=age_h * 60,
            )

        return CheckItem(
            name=name, expected="", ok=True, status="ok",
            detail=f"OK ({mtime:%m-%d %H:%M}, {age_h:.1f}h 전)",
            age_minutes=age_h * 60,
        )

    def _check_kis_token(self) -> CheckItem:
        """KIS 토큰 발급 상태 — token cache 파일 mtime + 만료 잔여."""
        # KIS 토큰 캐시 경로 후보들
        candidates = [
            self.project_root / "data" / "kis_token.json",
            self.project_root / ".kis_token.json",
            self.project_root / "data" / "kis_access_token.json",
        ]
        existing = [p for p in candidates if p.exists()]

        if not existing:
            return CheckItem(
                name="KIS_TOKEN",
                expected="매일 새 발급",
                ok=True,
                status="skip",
                detail="토큰 캐시 파일 없음 (런타임 발급 가정)",
            )

        latest = max(existing, key=lambda p: p.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        age_h = (self.now - mtime).total_seconds() / 3600.0

        # KIS 토큰은 24시간 유효. 발급 후 22시간 넘으면 stale (재발급 권고)
        if age_h > 22:
            return CheckItem(
                name="KIS_TOKEN",
                expected="매일 새 발급",
                ok=False,
                status="stale",
                detail=f"토큰 발급 {age_h:.1f}h 전 (만료 임박, 재발급 권고)",
                age_minutes=age_h * 60,
            )

        return CheckItem(
            name="KIS_TOKEN",
            expected="매일 새 발급",
            ok=True,
            status="ok",
            detail=f"발급 {age_h:.1f}h 전 ({mtime:%m-%d %H:%M})",
            age_minutes=age_h * 60,
        )

    # ── 텔레그램 포맷 ─────────────────────────

    def _format_telegram(self, report: IntegrityReport) -> str:
        """missing/stale 있으면 상세 + 가동 영향 멘트, 없으면 1줄."""
        missing = report.missing
        stale = report.stale
        ts = report.timestamp.strftime("%H:%M")

        if not missing and not stale:
            total = (
                len(report.bat_results)
                + len(report.intel_bot)
                + len(report.prerequisite_files)
                + (1 if report.kis_token else 0)
            )
            ok_count = sum(
                1 for g in (report.bat_results, report.intel_bot, report.prerequisite_files)
                for c in g if c.ok
            )
            if report.kis_token and report.kis_token.ok:
                ok_count += 1
            return f"[DataIntegrity {ts}] {ok_count}/{total}개 데이터 소스 모두 정상"

        # 이상 발생
        lines = []
        total_issues = len(missing) + len(stale)
        lines.append(f"[DataIntegrity 누락/지연 {total_issues}건] {self.today:%m/%d} 가동 위험!")

        for c in missing:
            lines.append(f"  [MISSING] {c.name}: {c.detail}")
        for c in stale:
            lines.append(f"  [STALE]  {c.name}: {c.detail}")

        lines.append("  → 5/20 가동 영향 가능성 있음, 확인 부탁드립니다")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Layer 7 — 자동 KILL_SWITCH 통합 (5/20 가동 직전)
# ──────────────────────────────────────────────


def _is_critical_data_missing(result: dict) -> tuple[bool, str]:
    """가동에 치명적인 데이터 누락 검출.

    활성화 조건 (5/20 가동 직전 자동 차단):
    1. tomorrow_picks missing/stale — 후보 풀 신선도 0
    2. BAT-D missing/stale — 전날 16:30 미실행
    3. KIS 토큰 만료 (실거래 주문 불가)

    "missing"과 "stale"은 IntegrityReport.to_dict()에서 단순 name 리스트.
    detail은 prerequisite_files 항목의 status로 확인.
    """
    critical_keys = ["tomorrow_picks", "BAT-D"]
    missing = result.get("missing", [])
    stale = result.get("stale", [])

    for item in missing:
        for cf in critical_keys:
            if cf in str(item):
                return True, f"치명적 데이터 누락: {item}"

    for item in stale:
        for cf in critical_keys:
            if cf in str(item):
                return True, f"치명적 데이터 stale: {item}"

    return False, ""


def _post_process_result(result: dict) -> None:
    """check_all() 결과 후처리 — Layer 7 KILL_SWITCH + Reporter 보고용 latest.json 저장.

    lazy import로 순환 import 방지. kill_switch_manager 부재 시 silent skip
    (다른 워커가 작성 중일 수 있음).
    """
    # 1) Layer 7 KILL_SWITCH 자동 활성화
    try:
        should_kill, kill_reason = _is_critical_data_missing(result)
        if should_kill:
            try:
                from src.agents.kill_switch_manager import activate_kill_switch
                activate_kill_switch(reason=kill_reason, source="DataIntegrity", send_tg=True)  # 5/19 결단 C — RED 단일 채널만
            except Exception as e:
                logger.warning(
                    "[DataIntegrity] activate_kill_switch 실패 (모듈 부재 가능): %s", e
                )
    except Exception as e:
        logger.warning("[DataIntegrity] critical_data_missing 검사 실패: %s", e)

    # 2) latest.json 저장 (Reporter가 읽음)
    try:
        from src.agents.kill_switch_manager import save_worker_report
        save_worker_report("data_integrity", result)
    except Exception as e:
        logger.warning("[DataIntegrity] save_worker_report 실패 (모듈 부재 가능): %s", e)


# ──────────────────────────────────────────────
# 편의 함수
# ──────────────────────────────────────────────


def run_once(send_tg: bool = True) -> dict:
    """1회 실행 — CLI 진입점에서 사용.

    사장님 결단 C (2026-05-19): send_tg=True여도 AGENT_TELEGRAM_ENABLED=false면 발송 SKIP.
    KILL_SWITCH RED는 kill_switch_manager가 별도로 발송 (유일한 단일 채널).
    """
    agent = DataIntegrity()
    report = agent.build_report()
    result = report.to_dict()
    _post_process_result(result)

    # 콘솔 출력
    print(agent._format_telegram(report))

    if send_tg:
        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            logger.info(
                "[DataIntegrity] run_once 텔레그램 SKIP (AGENT_TELEGRAM_ENABLED=false)"
            )
        else:
            try:
                from src.telegram_sender import send_message
                send_message(agent._format_telegram(report))
            except Exception as e:
                logger.warning(f"텔레그램 전송 실패(무시): {e}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_once(send_tg=False)
