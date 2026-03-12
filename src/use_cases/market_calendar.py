"""시장 이벤트 캘린더 — 네 마녀의 날(쿼드러플 위칭) + 주요 만기일

매 분기(3/6/9/12월) 두 번째 목요일:
  KOSPI200 선물 + KOSPI200 옵션 + 개별주식 선물 + 개별주식 옵션 동시 만기.
  프로그램 매매 폭발 → 변동성 급등 → 매수 주의.

사용처:
  1. BAT-B 아침 브리핑 → 만기일/전일 경고
  2. BAT-D 장마감 후 → 내일 만기일이면 텔레그램 사전 알림
  3. SignalEngine/scan → 만기일 감점 (매수 억제)
  4. scan_tomorrow_picks.py → 만기일 주의 표시
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta


# 분기 만기월
EXPIRY_MONTHS = (3, 6, 9, 12)


def get_quad_witching_date(year: int, month: int) -> date:
    """해당 월의 두 번째 목요일 계산.

    Args:
        year: 연도
        month: 월 (3, 6, 9, 12)

    Returns:
        두 번째 목요일 날짜
    """
    # 해당 월 캘린더
    cal = calendar.monthcalendar(year, month)
    # calendar.THURSDAY = 3 (Mon=0)
    thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY] != 0]
    second_thursday = thursdays[1]
    return date(year, month, second_thursday)


def get_yearly_witching_dates(year: int) -> list[date]:
    """해당 연도의 네 마녀의 날 4개 반환."""
    return [get_quad_witching_date(year, m) for m in EXPIRY_MONTHS]


def check_witching_proximity(target: date | None = None) -> dict:
    """오늘/지정일 기준 만기일 근접도 체크.

    Returns:
        {
            "is_witching_day": bool,      # 오늘이 만기일
            "is_day_before": bool,         # 내일이 만기일 (D-1)
            "is_week_of": bool,            # 만기 주간 (월~금)
            "days_to_next": int,           # 다음 만기일까지 남은 일수
            "next_witching": "2026-06-11", # 다음 만기일
            "current_witching": "2026-03-12" or None,  # 오늘이 만기일이면
            "warning_level": "CRITICAL" | "HIGH" | "MODERATE" | "NONE",
            "message": str,
        }
    """
    today = target or date.today()

    # 올해 + 내년 만기일 (연말 대비)
    dates = get_yearly_witching_dates(today.year)
    dates += get_yearly_witching_dates(today.year + 1)

    result = {
        "is_witching_day": False,
        "is_day_before": False,
        "is_week_of": False,
        "days_to_next": 999,
        "next_witching": "",
        "current_witching": None,
        "warning_level": "NONE",
        "message": "",
    }

    for wd in dates:
        delta = (wd - today).days

        if delta == 0:
            result["is_witching_day"] = True
            result["current_witching"] = wd.isoformat()
            result["days_to_next"] = 0
            result["next_witching"] = wd.isoformat()
            result["warning_level"] = "CRITICAL"
            result["message"] = (
                f"⚠️ 오늘 네 마녀의 날 ({wd.strftime('%m/%d')})\n"
                "선물·옵션 동시 만기 → 프로그램 매매 폭발 예상\n"
                "신규 매수 자제, 장 막판(14:30~) 변동성 주의"
            )
            break

        # D-1: 내일이 만기일 (거래일 기준 — 목요일 전날 = 수요일)
        if delta == 1:
            result["is_day_before"] = True
            result["days_to_next"] = 1
            result["next_witching"] = wd.isoformat()
            result["warning_level"] = "HIGH"
            result["message"] = (
                f"📢 내일 네 마녀의 날 ({wd.strftime('%m/%d')})\n"
                "선물·옵션 동시 만기 예정 → 오늘부터 신규 매수 주의\n"
                "포지션 정리 물량으로 변동성 확대 가능"
            )
            break

        # 같은 주 (만기일 기준 월~금)
        wd_monday = wd - timedelta(days=wd.weekday())
        wd_friday = wd_monday + timedelta(days=4)
        if wd_monday <= today <= wd_friday:
            result["is_week_of"] = True
            result["days_to_next"] = delta
            result["next_witching"] = wd.isoformat()
            result["warning_level"] = "MODERATE"
            result["message"] = (
                f"📅 이번 주 {wd.strftime('%m/%d')}(목) 네 마녀의 날\n"
                f"만기 D-{delta}일 → 프로그램 매매 영향권"
            )
            break

        # 미래 만기일 중 가장 가까운 것
        if delta > 0 and delta < result["days_to_next"]:
            result["days_to_next"] = delta
            result["next_witching"] = wd.isoformat()
            # 3일 이내면 MODERATE
            if delta <= 3:
                result["warning_level"] = "MODERATE"
                result["message"] = (
                    f"📅 {wd.strftime('%m/%d')}(목) 네 마녀의 날 D-{delta}\n"
                    "만기일 접근 중 → 프로그램 매매 영향 가능"
                )

    return result


def get_witching_penalty(target: date | None = None) -> int:
    """만기일 근접도에 따른 매수 감점.

    Returns:
        감점 (0 ~ -10)
        CRITICAL(당일): -10
        HIGH(D-1): -5
        MODERATE(같은 주): -3
        NONE: 0
    """
    info = check_witching_proximity(target)
    penalties = {
        "CRITICAL": -10,
        "HIGH": -5,
        "MODERATE": -3,
        "NONE": 0,
    }
    return penalties.get(info["warning_level"], 0)


def format_2026_schedule() -> str:
    """2026년 네 마녀의 날 일정 출력."""
    dates = get_yearly_witching_dates(2026)
    lines = ["📅 2026년 네 마녀의 날 (선물·옵션 동시 만기)"]
    for d in dates:
        lines.append(f"  • {d.strftime('%m월 %d일')} (목)")
    return "\n".join(lines)


if __name__ == "__main__":
    # 테스트
    print(format_2026_schedule())
    print()
    info = check_witching_proximity()
    print(f"오늘 상태: {info['warning_level']}")
    print(f"메시지: {info['message']}")
    print(f"다음 만기: {info['next_witching']} (D-{info['days_to_next']})")
    print(f"매수 감점: {get_witching_penalty()}")
