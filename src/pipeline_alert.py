"""파이프라인 에러 기록 + 텔레그램 알림 공용 모듈.

사용법:
    from src.pipeline_alert import PipelineErrorTracker

    tracker = PipelineErrorTracker("collect_short_selling")
    tracker.record("005930", "JSONDecodeError: ...")
    tracker.finalize(total=100)  # 에러율 5% 이상이면 텔레그램 발송
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ERRORS_PATH = DATA_DIR / "pipeline_errors.json"
MAX_HISTORY = 50


class PipelineErrorTracker:
    """스크립트별 에러 추적기."""

    def __init__(self, script_name: str):
        self.script_name = script_name
        self.errors: list[dict] = []
        self.start_time = datetime.now()

    def record(self, context: str, error: str | Exception):
        """에러 1건 기록."""
        self.errors.append({
            "context": context,
            "error": str(error)[:200],
            "time": datetime.now().strftime("%H:%M:%S"),
        })

    def finalize(self, total: int, alert_threshold: float = 0.05):
        """에러 집계 저장 + 필요 시 텔레그램 알림.

        Args:
            total: 전체 처리 건수
            alert_threshold: 알림 기준 에러율 (기본 5%)
        """
        error_count = len(self.errors)
        error_rate = error_count / total if total > 0 else 0

        entry = {
            "script": self.script_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total": total,
            "errors": error_count,
            "error_rate_pct": round(error_rate * 100, 1),
            "samples": self.errors[:10],
        }
        _append_to_history(entry)

        if error_rate >= alert_threshold and error_count > 0:
            msg = (
                f"⚠️ 파이프라인 에러 알림\n"
                f"스크립트: {self.script_name}\n"
                f"에러: {error_count}/{total} ({error_rate * 100:.1f}%)\n"
                f"대표 에러: {self.errors[0]['error'][:100]}"
            )
            _send_telegram(msg)
            logger.warning("에러율 %.1f%% — 텔레그램 알림 발송", error_rate * 100)
        elif error_count > 0:
            logger.info("에러 %d건 (%.1f%%) — 임계치 미만, 기록만",
                        error_count, error_rate * 100)


def _append_to_history(entry: dict):
    """pipeline_errors.json에 추가 (최근 MAX_HISTORY건 유지)."""
    history: list = []
    if ERRORS_PATH.exists():
        try:
            raw = json.loads(ERRORS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                history = raw
        except Exception:
            history = []

    history.append(entry)
    history = history[-MAX_HISTORY:]

    ERRORS_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _send_telegram(message: str):
    """텔레그램 알림 전송."""
    try:
        from src.telegram_sender import send_message
        send_message(message)
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


def get_recent_error_rate(hours: int = 24) -> dict:
    """최근 N시간 내 에러율 요약 (data_health_check용).

    Returns:
        {"total_runs": int, "total_errors": int, "error_rate_pct": float,
         "scripts_with_errors": list[str]}
    """
    if not ERRORS_PATH.exists():
        return {"total_runs": 0, "total_errors": 0, "error_rate_pct": 0,
                "scripts_with_errors": []}

    try:
        history = json.loads(ERRORS_PATH.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            return {"total_runs": 0, "total_errors": 0, "error_rate_pct": 0,
                    "scripts_with_errors": []}
    except Exception:
        return {"total_runs": 0, "total_errors": 0, "error_rate_pct": 0,
                "scripts_with_errors": []}

    cutoff = datetime.now()
    from datetime import timedelta
    cutoff = cutoff - timedelta(hours=hours)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M")

    total_items = 0
    total_errors = 0
    error_scripts = set()

    for entry in history:
        ts = entry.get("timestamp", "")
        if ts >= cutoff_str:
            total_items += entry.get("total", 0)
            errors = entry.get("errors", 0)
            if isinstance(errors, list):
                errors = len(errors)
            total_errors += errors
            if errors > 0:
                error_scripts.add(entry.get("script", "unknown"))

    rate = total_errors / total_items * 100 if total_items > 0 else 0

    return {
        "total_runs": total_items,
        "total_errors": total_errors,
        "error_rate_pct": round(rate, 1),
        "scripts_with_errors": sorted(error_scripts),
    }
