"""STEP 11: EVENT CALENDAR — 글로벌 이벤트 캘린더

매년 반복되는 구조적 이벤트(FOMC, MSCI, 주총, 어닝시즌 등)를
퀀트봇이 자체적으로 인지하고 시나리오/LENS에 반영한다.

brain.py 수정 ❌, signal_engine.py 수정 ❌
scenario_detector.py와 연동하여 이벤트 임박 시 시나리오 부스트.

실행: scenario_detector.py에서 호출 (BAT-D 11.23단계 내부)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_CALENDAR_PATH = Path("data/scenarios/event_calendar_2026.json")

# 이벤트 임팩트 → 시나리오 부스트 점수
_IMPACT_BOOST = {
    "HIGH": 20,
    "MEDIUM": 10,
    "LOW": 5,
}


class EventCalendar:
    """글로벌 이벤트 캘린더 조회 + 임팩트 스코어링."""

    def __init__(self, calendar_path: Path | None = None):
        path = calendar_path or _CALENDAR_PATH
        self._data = self._load(path)
        self._events = self._data.get("events", [])
        self._recurring = self._data.get("recurring", {})
        self._scenario_map = self._data.get("scenario_event_map", {})

    def get_upcoming(self, days: int = 7, ref_date: date | None = None) -> list[dict]:
        """향후 N일 이내 이벤트 반환 (당일 포함).

        Args:
            days: 조회 기간 (일)
            ref_date: 기준일 (None이면 오늘)

        Returns:
            [{"date": "2026-03-20", "type": "TRIPLE_WITCHING", "name": ..., ...}, ...]
        """
        today = ref_date or date.today()
        end = today + timedelta(days=days)

        upcoming = []
        for ev in self._events:
            try:
                ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            except (ValueError, KeyError):
                continue
            if today <= ev_date <= end:
                upcoming.append(ev)

        # 한국 옵션만기일 (recurring)
        kr_options = self._recurring.get("KR_OPTIONS_EXPIRY", {})
        year = today.year
        for d_str in kr_options.get("dates_2026", []):
            try:
                ev_date = datetime.strptime(f"{year}-{d_str}", "%Y-%m-%d").date()
            except ValueError:
                continue
            if today <= ev_date <= end:
                upcoming.append({
                    "date": ev_date.isoformat(),
                    "type": "KR_OPTIONS_EXPIRY",
                    "name": f"한국 옵션만기일 ({ev_date.month}월)",
                    "country": "KR",
                    "impact": kr_options.get("impact", "LOW"),
                    "scenarios": [],
                    "sectors_hot": [],
                    "sectors_cold": [],
                    "pattern": kr_options.get("pattern", ""),
                })

        # 날짜 순 정렬
        upcoming.sort(key=lambda x: x["date"])
        return upcoming

    def get_scenario_boost(self, scenario_id: str, days: int = 7,
                           ref_date: date | None = None) -> int:
        """특정 시나리오에 대한 이벤트 부스트 점수 계산.

        향후 N일 이내에 해당 시나리오와 관련된 이벤트가 있으면
        임팩트에 따라 가산점을 부여한다.

        Args:
            scenario_id: 시나리오 ID (예: "FED_RATE_CUT")
            days: 조회 기간
            ref_date: 기준일

        Returns:
            부스트 점수 (0 = 관련 이벤트 없음)
        """
        upcoming = self.get_upcoming(days=days, ref_date=ref_date)
        boost = 0

        for ev in upcoming:
            ev_type = ev.get("type", "")
            # 방법 1: 이벤트 자체에 시나리오 명시
            ev_scenarios = ev.get("scenarios", [])
            if scenario_id in ev_scenarios:
                boost += _IMPACT_BOOST.get(ev.get("impact", "LOW"), 5)
                continue

            # 방법 2: scenario_event_map에서 매핑
            mapped = self._scenario_map.get(ev_type, [])
            if scenario_id in mapped:
                boost += _IMPACT_BOOST.get(ev.get("impact", "LOW"), 5)

        return boost

    def get_earnings_season(self, ref_date: date | None = None) -> dict | None:
        """현재 어닝시즌 구간에 해당하는지 확인.

        Returns:
            {"label": "us_1q", "start": date, "end": date} 또는 None
        """
        today = ref_date or date.today()
        seasons = self._recurring.get("EARNINGS_SEASONS", {})

        for label, dates in seasons.items():
            if label == "description":
                continue
            if isinstance(dates, list) and len(dates) == 2:
                try:
                    start = datetime.strptime(dates[0], "%Y-%m-%d").date()
                    end = datetime.strptime(dates[1], "%Y-%m-%d").date()
                except ValueError:
                    continue
                if start <= today <= end:
                    return {"label": label, "start": start, "end": end}

        return None

    def get_seasonality(self, ref_date: date | None = None) -> dict:
        """현재 시즌널리티 정보 반환.

        Returns:
            {"month_type": "strong"/"weak"/"neutral",
             "special": "sell_in_may" / "santa_rally" / "september_effect" / None}
        """
        today = ref_date or date.today()
        season = self._recurring.get("SEASONALITY", {})

        month = today.month
        strong = season.get("strong_months", [])
        weak = season.get("weak_months", [])

        if month in strong:
            month_type = "strong"
        elif month in weak:
            month_type = "weak"
        else:
            month_type = "neutral"

        # 특수 시즌
        special = None
        sell_in_may = season.get("sell_in_may", {})
        if sell_in_may:
            try:
                sim_start = datetime.strptime(
                    f"{today.year}-{sell_in_may['start']}", "%Y-%m-%d").date()
                sim_end = datetime.strptime(
                    f"{today.year}-{sell_in_may['end']}", "%Y-%m-%d").date()
                if sim_start <= today <= sim_end:
                    special = "sell_in_may"
            except (ValueError, KeyError):
                pass

        santa = season.get("santa_rally", {})
        if santa and not special:
            try:
                sr_start = datetime.strptime(
                    f"{today.year}-{santa['start']}", "%Y-%m-%d").date()
                # 1/3은 다음 해
                if today >= sr_start:
                    special = "santa_rally"
            except (ValueError, KeyError):
                pass

        if month == 9 and not special:
            special = "september_effect"

        return {"month_type": month_type, "special": special}

    def format_weekly_briefing(self, days: int = 7,
                               ref_date: date | None = None) -> str | None:
        """이번 주 이벤트 브리핑 텔레그램 메시지 포맷.

        Returns:
            포맷팅된 메시지. 이벤트 없으면 None.
        """
        upcoming = self.get_upcoming(days=days, ref_date=ref_date)
        if not upcoming:
            return None

        today = ref_date or date.today()
        earnings = self.get_earnings_season(ref_date)
        seasonality = self.get_seasonality(ref_date)

        lines = [f"📅 이벤트 캘린더 ({today.isoformat()} 기준, 향후 {days}일)"]

        if earnings:
            lines.append(f"📊 어닝시즌: {earnings['label']} ({earnings['start']}~{earnings['end']})")

        season_str = seasonality["month_type"]
        if seasonality["special"]:
            season_str += f" + {seasonality['special']}"
        lines.append(f"🌡️ 시즌: {season_str}")
        lines.append("")

        for ev in upcoming:
            ev_date = ev["date"]
            impact = ev.get("impact", "?")
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}.get(impact, "⚪")

            # D-N 계산
            try:
                ev_d = datetime.strptime(ev_date, "%Y-%m-%d").date()
                d_minus = (ev_d - today).days
                d_str = f"D-{d_minus}" if d_minus > 0 else "오늘"
            except ValueError:
                d_str = "?"

            lines.append(f"{icon} [{ev_date}] {ev['name']} ({d_str})")

            pattern = ev.get("pattern", "")
            if pattern:
                lines.append(f"   → {pattern}")

            hot = ev.get("sectors_hot", [])
            cold = ev.get("sectors_cold", [])
            if hot or cold:
                parts = []
                if hot:
                    parts.append(f"HOT: {', '.join(hot)}")
                if cold:
                    parts.append(f"COLD: {', '.join(cold)}")
                lines.append(f"   → {' | '.join(parts)}")

        return "\n".join(lines)

    @staticmethod
    def _load(path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("이벤트 캘린더 로드 실패: %s — %s", path, e)
            return {}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    import sys
    from pathlib import Path as P

    PROJECT_ROOT = P(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    days = 14
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            pass

    cal = EventCalendar()

    # 향후 이벤트
    upcoming = cal.get_upcoming(days=days)
    print(f"\n향후 {days}일 이벤트: {len(upcoming)}개")
    for ev in upcoming:
        print(f"  [{ev['date']}] {ev['name']} ({ev.get('impact', '?')})")

    # 시나리오별 부스트
    print("\n시나리오별 이벤트 부스트:")
    for sid in ["FED_RATE_CUT", "FED_RATE_HIKE", "SEMICONDUCTOR_CYCLE_UP",
                "WAR_MIDDLE_EAST", "OIL_SPIKE"]:
        boost = cal.get_scenario_boost(sid, days=days)
        if boost > 0:
            print(f"  {sid}: +{boost}")

    # 어닝시즌
    earnings = cal.get_earnings_season()
    if earnings:
        print(f"\n현재 어닝시즌: {earnings['label']} ({earnings['start']}~{earnings['end']})")
    else:
        print("\n현재 어닝시즌 아님")

    # 시즌널리티
    seasonality = cal.get_seasonality()
    print(f"시즌: {seasonality['month_type']}" +
          (f" + {seasonality['special']}" if seasonality['special'] else ""))

    # 텔레그램 브리핑
    msg = cal.format_weekly_briefing(days=days)
    if msg:
        print(f"\n{'='*50}")
        print(msg)


if __name__ == "__main__":
    main()
