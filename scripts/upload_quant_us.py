"""
============================================
FLOWX 퀀트봇 — 미국장 매크로 결과 Supabase 업로드
============================================

us_quant_filter 결과를 Supabase quant_us_macro에 upsert.
FLOWX 웹봇 → "미국장 매크로" 패널에 표시.

스케줄: BAT-M_US (08:10) — 정보봇(07:55) 후 실행
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

# ── PYTHONPATH 안전장치 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("upload_quant_us")


def upload(report: dict) -> bool:
    """quant_us_macro 테이블에 upsert."""
    if not report:
        log.error("빈 리포트 — 업로드 스킵")
        return False

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        log.warning("SUPABASE_URL/KEY 미설정 — 업로드 스킵")
        return False

    try:
        from supabase import create_client
        sb = create_client(url, key)

        hold = report.get("hold_days", (5, 10))
        row = {
            "date":               report.get("date", date.today().isoformat()),
            "strategy_mode":      report.get("strategy_mode"),
            "macro_score":        report.get("macro_score"),
            "position_limit":     report.get("position_limit"),
            "hold_days_min":      hold[0] if isinstance(hold, (list, tuple)) else 5,
            "hold_days_max":      hold[1] if isinstance(hold, (list, tuple)) else 10,
            "sector_overweight":  report.get("sector_overweight") or [],
            "sector_underweight": report.get("sector_underweight") or [],
            "etf_momentum":       report.get("etf_momentum") or {},
            "soxx_5d":            report.get("soxx_5d"),
            "yield_signal":       report.get("yield_signal"),
            "yield_level":        report.get("yield_level"),
            "yield_trend":        report.get("yield_trend"),
            "yield_inverted":     report.get("yield_inverted"),
            "yield_impact":       report.get("yield_impact"),
            "dollar_signal":      report.get("dollar_signal"),
            "dxy":                report.get("dxy"),
            "dollar_impact":      report.get("dollar_impact"),
            "vix_env":            report.get("vix_env"),
            "vix":                report.get("vix"),
            "entry_conditions":   report.get("entry_conditions") or {},
            "weekly_outlook":     report.get("weekly_outlook"),
            "risk_flags":         report.get("risk_flags") or [],
        }

        sb.table("quant_us_macro") \
            .upsert(row, on_conflict="date") \
            .execute()

        log.info("[FLOWX] 미국장 매크로 업로드: %s | %s | 점수 %s",
                 row["date"], row["strategy_mode"], row["macro_score"])
        return True

    except Exception as e:
        log.error("업로드 실패: %s", e)
        return False


def main():
    from us_quant_filter import run as run_quant
    log.info("퀀트봇 미국장 매크로 분석 + 업로드")
    report = run_quant()
    if report:
        ok = upload(report)
        if not ok:
            log.warning("업로드 실패 — 로컬 JSON은 저장됨")


if __name__ == "__main__":
    main()
