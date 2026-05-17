"""자동매매 일일 누적 추적 모듈 (P0 가드레일 보강).

책임:
  - 매수 발생 시 record_buy 호출 → 일일 누적 금액/횟수 업데이트
  - get_today_volume → 현재 누적 상태 반환
  - 날짜 변경 시 자동 초기화 (전날 파일은 daily_volume_YYYYMMDD.json 으로 백업)

저장 위치:
  data/auto_trading/daily_volume.json (현재일)
  data/auto_trading/daily_volume_YYYYMMDD.json (이력)

JSON 스키마:
  {
    "date": "2026-05-26",
    "total_amount": 200000,
    "total_trades": 2,
    "buys": [
      {"ts": "2026-05-26T09:15:23", "ticker": "487240", "qty": 1, "price": 12500, "amount": 12500}
    ]
  }
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from src.utils.atomic_io import atomic_write_json

VOLUME_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "auto_trading"
VOLUME_FILE = VOLUME_DIR / "daily_volume.json"


def _empty_record(d: date) -> dict:
    return {
        "date": d.isoformat(),
        "total_amount": 0,
        "total_trades": 0,
        "buys": [],
    }


def get_today_volume(today: date | None = None) -> dict:
    """현재일 누적 매수 상태 조회.

    Args:
        today: 기준일 (None이면 date.today()).

    Returns:
        {"date", "total_amount", "total_trades", "buys": [...]}
        파일이 없거나 날짜 다르면 빈 레코드 반환.
    """
    today = today or date.today()
    if not VOLUME_FILE.exists():
        return _empty_record(today)
    try:
        data = json.loads(VOLUME_FILE.read_text(encoding="utf-8"))
        if data.get("date") != today.isoformat():
            return _empty_record(today)
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_record(today)


def record_buy(ticker: str, qty: int, price: int, today: date | None = None) -> dict:
    """매수 발생 기록.

    날짜가 바뀌었으면 이전 파일을 daily_volume_YYYYMMDD.json으로 백업하고
    새 파일을 시작한다.

    Args:
        ticker: 종목 코드 (6자리)
        qty: 매수 수량
        price: 매수 단가
        today: 기준일 (None이면 date.today()) — 테스트용

    Returns:
        업데이트된 레코드
    """
    today = today or date.today()
    amount = qty * price

    VOLUME_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 레코드 로드 (날짜 다르면 백업 후 새 레코드)
    if VOLUME_FILE.exists():
        try:
            existing = json.loads(VOLUME_FILE.read_text(encoding="utf-8"))
            if existing.get("date") != today.isoformat():
                # 날짜 변경 — 백업
                prev_date = existing.get("date", "unknown").replace("-", "")
                backup_path = VOLUME_DIR / f"daily_volume_{prev_date}.json"
                if not backup_path.exists():
                    atomic_write_json(backup_path, existing)
                record = _empty_record(today)
            else:
                record = existing
        except (json.JSONDecodeError, OSError):
            record = _empty_record(today)
    else:
        record = _empty_record(today)

    # 신규 매수 추가
    record["buys"].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "amount": amount,
    })
    record["total_amount"] = sum(b["amount"] for b in record["buys"])
    record["total_trades"] = len(record["buys"])

    atomic_write_json(VOLUME_FILE, record)
    return record


def check_daily_limits(amount_to_add: int, max_amount: int, max_trades: int,
                       today: date | None = None) -> tuple[bool, str]:
    """매수 진행 전 일일 한도 사전 검증.

    Args:
        amount_to_add: 이번 매수 추가될 금액
        max_amount: 일일 최대 금액 한도
        max_trades: 일일 최대 횟수 한도
        today: 기준일

    Returns:
        (통과 여부, 실패 사유). 통과 시 (True, "OK").
    """
    today = today or date.today()
    record = get_today_volume(today)
    new_total = record["total_amount"] + amount_to_add
    new_trades = record["total_trades"] + 1
    if new_total > max_amount:
        return False, (
            f"일일 금액 한도 초과: 누적 {record['total_amount']:,} + 추가 "
            f"{amount_to_add:,} = {new_total:,} > 한도 {max_amount:,}"
        )
    if new_trades > max_trades:
        return False, (
            f"일일 횟수 한도 초과: 누적 {record['total_trades']} + 1 = "
            f"{new_trades} > 한도 {max_trades}"
        )
    return True, "OK"
