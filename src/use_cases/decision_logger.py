"""매매 의사결정 일지 (5/26~5/28 3일 학습 모드).

매 매수/매도/회피/알림 시점에 9개 학습 항목(S1~S6 + D1~D3) + 컨텍스트를
JSON Lines로 저장. 5/29(목) 결단 시 분석 데이터로 활용.

학습 항목:
  S1 signal_strength: 체결강도 (KIS tday_rltv, 0~300)
  S2 volume_ratio: 5일 평균 대비 배수
  S3 bullish_ratio: 양봉 비율 (0~1)
  S4 foreign_inst_buy: 외인+기관 동시 매수 (bool)
  S5 time_slot: 'MORNING' / 'NOON' / 'AFTERNOON'
  S6 c2_combo: C2 시그널 조합 dict
  D1 peak_drop_pct: 천장 대비 하락 %
  D2 trailing_drop_pct: Trailing 고점 대비 꺾임 %
  D3 supply_outflow_days: 수급 이탈 연속 일수

파일: data/decision_log/YYYYMMDD.json (JSON Lines)
환경변수: ADAPTIVE_DAILY_LEARNING_MODE=1 일 때만 저장
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "decision_log"


def _learning_mode_on() -> bool:
    """학습 모드 활성 여부 (env 실시간 평가)."""
    return os.getenv("ADAPTIVE_DAILY_LEARNING_MODE", "0") == "1"


def _time_slot() -> str:
    """현재 시각 → 매수 시간대 라벨.

    P1-4 (5/25): 한국 KOSPI 점심 휴장 없음. 11:30~11:59는 NOON 유지 (정상 거래 시간).
    LUNCH는 12:00~12:59 (실제 거래량 약한 시간대)로 한정.
    """
    now = datetime.now()
    h, m = now.hour, now.minute
    if h < 9:
        return "PRE_OPEN"
    if h == 9:
        return "MORNING"
    if 10 <= h < 12:  # 10:00~11:59 (점심 휴장 없음 — NOON)
        return "NOON"
    if h == 12:  # 12:00~12:59 (실제 점심 시간대, 거래량 약함)
        return "LUNCH"
    if 13 <= h < 15:
        return "AFTERNOON"
    if h == 15 and m < 30:
        return "CLOSE"
    return "AFTER_CLOSE"


def log_decision(
    decision_type: str,  # "BUY" / "SELL" / "SKIP" / "ALERT" / "QUEUE_REGISTER" / "QUEUE_TRIGGER"
    ticker: str,
    name: str = "",
    *,
    current_price: int = 0,
    qty: int = 0,
    amount: int = 0,
    target_price: int = 0,
    # S1~S6 상승 학습
    signal_strength: float = 0.0,
    volume_ratio: float = 0.0,
    bullish_ratio: float = 0.0,
    foreign_inst_buy: bool = False,
    time_slot: Optional[str] = None,
    c2_combo: Optional[dict] = None,
    # D1~D3 하락 학습
    peak_drop_pct: float = 0.0,
    trailing_drop_pct: float = 0.0,
    supply_outflow_days: int = 0,
    # 결과 / 컨텍스트
    pass_reasons: Optional[list[str]] = None,
    fail_reasons: Optional[list[str]] = None,
    skip_reason: str = "",
    error: str = "",
    extra: Optional[dict] = None,
) -> bool:
    """매매 의사결정 1건 저장.

    학습 모드 OFF 시 no-op (False 반환).
    JSON Lines 형식으로 append.

    Returns:
        True 저장 성공 / False 미저장 (모드 OFF or 에러)
    """
    if not _learning_mode_on():
        return False

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = LOG_DIR / f"{today}.json"

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": decision_type,
            "ticker": ticker,
            "name": name,
            "current_price": int(current_price),
            "qty": int(qty),
            "amount": int(amount),
            "target_price": int(target_price),
            "signals": {
                "S1_signal_strength": float(signal_strength),
                "S2_volume_ratio": float(volume_ratio),
                "S3_bullish_ratio": float(bullish_ratio),
                "S4_foreign_inst_buy": bool(foreign_inst_buy),
                "S5_time_slot": time_slot or _time_slot(),
                "S6_c2_combo": c2_combo or {},
                "D1_peak_drop_pct": float(peak_drop_pct),
                "D2_trailing_drop_pct": float(trailing_drop_pct),
                "D3_supply_outflow_days": int(supply_outflow_days),
            },
            "pass_reasons": pass_reasons or [],
            "fail_reasons": fail_reasons or [],
            "skip_reason": skip_reason,
            "error": error,
            "extra": extra or {},
        }

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as e:
        logger.warning("decision_log 저장 실패 [%s/%s]: %s", decision_type, ticker, e)
        return False


def query_today(decision_type: Optional[str] = None, ticker: Optional[str] = None) -> list[dict]:
    """오늘 의사결정 조회 (필터).

    Args:
        decision_type: 'BUY' / 'SELL' / 'SKIP' / 'ALERT' / ... (None = 전체)
        ticker: 종목 코드 필터 (None = 전체)
    """
    today = datetime.now().strftime("%Y%m%d")
    return query_date(today, decision_type=decision_type, ticker=ticker)


def query_date(
    date_str: str,  # 'YYYYMMDD'
    decision_type: Optional[str] = None,
    ticker: Optional[str] = None,
) -> list[dict]:
    """특정 일자 의사결정 조회."""
    log_file = LOG_DIR / f"{date_str}.json"
    if not log_file.exists():
        return []

    records: list[dict] = []
    try:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if decision_type and r.get("type") != decision_type:
                continue
            if ticker and r.get("ticker") != ticker:
                continue
            records.append(r)
    except OSError as e:
        logger.warning("decision_log 조회 실패 [%s]: %s", date_str, e)
        return []

    return records


def summarize_today() -> dict:
    """오늘 의사결정 통계 요약 (type/ticker별 count)."""
    records = query_today()
    summary: dict = {
        "total": len(records),
        "by_type": {},
        "by_ticker": {},
        "buy_count": 0,
        "sell_count": 0,
        "skip_count": 0,
    }
    for r in records:
        t = r.get("type", "UNKNOWN")
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        tk = r.get("ticker", "?")
        summary["by_ticker"][tk] = summary["by_ticker"].get(tk, 0) + 1
        if t == "BUY":
            summary["buy_count"] += 1
        elif t == "SELL":
            summary["sell_count"] += 1
        elif t == "SKIP":
            summary["skip_count"] += 1
    return summary
