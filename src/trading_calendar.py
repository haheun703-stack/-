"""거래일 캘린더 — 한국/미국 시장 휴장일 + 유틸리티.

주요 기능:
  1. is_kr_trading_day(d) — 한국 주식시장 개장일 여부
  2. is_us_trading_day(d) — 미국 주식시장 개장일 여부
  3. last_us_trading_day(d) — d 이전 마지막 미국 거래일
  4. last_kr_trading_day(d) — d 이전 마지막 한국 거래일
  5. should_run_bat(bat_type, d) — BAT 스케줄 실행 여부

사용처:
  - BAT-A us_overnight_signal.py (주말 스킵)
  - BAT-B morning_briefing_generator.py (주말 스킵)
  - BAT-D 장마감 파이프라인 (한국 휴장일 스킵)
  - VPS cron 래퍼 (주말/공휴일 가드)
"""

from __future__ import annotations

from datetime import date, timedelta

# ── 2026년 한국 주식시장 휴장일 ──
# 출처: KRX 공시, 관공서의 공휴일에 관한 규정
KR_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # 신정
    date(2026, 1, 27),  # 설날 연휴
    date(2026, 1, 28),  # 설날
    date(2026, 1, 29),  # 설날 연휴
    date(2026, 3, 1),   # 삼일절
    date(2026, 5, 5),   # 어린이날
    date(2026, 5, 24),  # 석가탄신일
    date(2026, 6, 6),   # 현충일
    date(2026, 8, 15),  # 광복절
    date(2026, 9, 24),  # 추석 연휴
    date(2026, 9, 25),  # 추석
    date(2026, 9, 26),  # 추석 연휴
    date(2026, 10, 3),  # 개천절
    date(2026, 10, 9),  # 한글날
    date(2026, 12, 25),  # 크리스마스
    date(2026, 12, 31),  # 연말 휴장 (KRX 특별)
    # 대체공휴일 (2026년 기준)
    date(2026, 1, 30),  # 설날 대체공휴일 (금)
    date(2026, 10, 5),  # 개천절 대체공휴일 (월)
}

# ── 2025년 한국 (연말~연초 대비) ──
KR_HOLIDAYS_2025 = {
    date(2025, 12, 25),
    date(2025, 12, 31),
}

# ── 2027년 한국 (연초 대비) ──
KR_HOLIDAYS_2027 = {
    date(2027, 1, 1),
}

# ── 2026년 미국 주식시장 휴장일 ──
# 출처: NYSE/NASDAQ 공식 휴장일
US_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day (3rd Mon Jan)
    date(2026, 2, 16),  # Presidents' Day (3rd Mon Feb)
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day (last Mon May)
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day observed (7/4=Sat → 7/3 Fri)
    date(2026, 9, 7),   # Labor Day (1st Mon Sep)
    date(2026, 11, 26), # Thanksgiving (4th Thu Nov)
    date(2026, 12, 25), # Christmas
}

US_HOLIDAYS_2025 = {
    date(2025, 12, 25),
}

US_HOLIDAYS_2027 = {
    date(2027, 1, 1),
}

# 전체 합치기
_KR_ALL = KR_HOLIDAYS_2025 | KR_HOLIDAYS_2026 | KR_HOLIDAYS_2027
_US_ALL = US_HOLIDAYS_2025 | US_HOLIDAYS_2026 | US_HOLIDAYS_2027


def is_weekend(d: date) -> bool:
    """토/일 여부."""
    return d.weekday() >= 5


def is_kr_holiday(d: date) -> bool:
    """한국 시장 공휴일 여부 (주말 제외)."""
    return d in _KR_ALL


def is_us_holiday(d: date) -> bool:
    """미국 시장 공휴일 여부 (주말 제외)."""
    return d in _US_ALL


def is_kr_trading_day(d: date | None = None) -> bool:
    """한국 주식시장 거래일인지."""
    d = d or date.today()
    return not is_weekend(d) and not is_kr_holiday(d)


def is_us_trading_day(d: date | None = None) -> bool:
    """미국 주식시장 거래일인지."""
    d = d or date.today()
    return not is_weekend(d) and not is_us_holiday(d)


def last_kr_trading_day(d: date | None = None) -> date:
    """d 이전 마지막 한국 거래일 (d 포함)."""
    d = d or date.today()
    while not is_kr_trading_day(d):
        d -= timedelta(days=1)
    return d


def prev_kr_trading_day(d: date | None = None) -> date:
    """d 직전 한국 거래일 (d 미포함)."""
    d = d or date.today()
    d -= timedelta(days=1)
    return last_kr_trading_day(d)


def last_us_trading_day(d: date | None = None) -> date:
    """d 이전 마지막 미국 거래일 (d 포함)."""
    d = d or date.today()
    while not is_us_trading_day(d):
        d -= timedelta(days=1)
    return d


def prev_us_trading_day(d: date | None = None) -> date:
    """d 직전 미국 거래일 (d 미포함)."""
    d = d or date.today()
    d -= timedelta(days=1)
    return last_us_trading_day(d)


def should_run_bat(bat_type: str, d: date | None = None) -> bool:
    """BAT 스케줄 실행 여부 판단.

    Args:
        bat_type: "us" (BAT-A/B, 미장 기반) or "kr" (BAT-D/E/K 등, 한국장 기반)
        d: 체크할 날짜 (기본: 오늘)

    Returns:
        True면 실행, False면 스킵
    """
    d = d or date.today()

    if bat_type == "us":
        # BAT-A: 미장 마감 후 → 오늘(KST)의 전날이 미국 거래일이어야 함
        # BAT-A는 KST 06:10에 실행 → 미국 기준 전날 종가 데이터
        # 토요일 06:10 → 금요일 미장 = OK (이미 BAT-A로 금요일에 처리됨)
        # 일요일/월요일은 새로운 미장 데이터 없음 (토요일에 이미 금요일 데이터 처리)
        # → 토요일~일요일은 스킵 (금요일 06:10에 목요일 미장 데이터 처리)
        # 실제로는: KST 아침 = US 전날 밤. US 거래일이었으면 실행.
        us_yesterday = d - timedelta(days=1)
        return is_us_trading_day(us_yesterday)
    elif bat_type == "kr":
        return is_kr_trading_day(d)
    else:
        return is_kr_trading_day(d)


def get_stale_data_warning(signal_date: str, today: date | None = None) -> str | None:
    """overnight_signal.json의 날짜가 오래된 경우 경고 메시지.

    Returns:
        None이면 정상, 문자열이면 경고 (비거래일 데이터 등)
    """
    today = today or date.today()
    try:
        from datetime import datetime
        sig_date = datetime.strptime(signal_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return "날짜 파싱 실패"

    # 시그널 생성일과 오늘의 차이
    gap = (today - sig_date).days
    if gap <= 1:
        return None  # 정상 (어제 또는 오늘 생성)

    # 주말/공휴일로 인한 갭인지 확인
    last_us = last_us_trading_day(today - timedelta(days=1))
    if sig_date >= last_us:
        return None  # 마지막 미국 거래일 이후 생성이면 정상

    return f"시그널 데이터 {gap}일 전 ({signal_date}). 최신 미장 데이터 반영 필요."


if __name__ == "__main__":
    from datetime import datetime

    today = date.today()
    print(f"오늘: {today} ({today.strftime('%A')})")
    print(f"  한국 거래일? {is_kr_trading_day(today)}")
    print(f"  미국 거래일? {is_us_trading_day(today)}")
    print(f"  BAT-A 실행? {should_run_bat('us', today)}")
    print(f"  BAT-D 실행? {should_run_bat('kr', today)}")
    print(f"  마지막 한국 거래일: {last_kr_trading_day(today)}")
    print(f"  마지막 미국 거래일: {last_us_trading_day(today)}")

    # 주말 테스트
    for offset in range(7):
        d = today + timedelta(days=offset)
        kr = "O" if is_kr_trading_day(d) else "X"
        us = "O" if is_us_trading_day(d) else "X"
        bat_a = "O" if should_run_bat("us", d) else "X"
        bat_d = "O" if should_run_bat("kr", d) else "X"
        print(f"  {d} ({d.strftime('%a')}): KR={kr} US={us} BAT-A={bat_a} BAT-D={bat_d}")
