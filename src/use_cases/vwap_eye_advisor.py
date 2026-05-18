"""vwap_monitor + intraday_eye → advisory 통합 (2026-05-18 4번 작업)

배경: 5/18 14:23 형이 본 텔레그램 알림 풀세트
- VWAP 알림: LG씨엔에스/ISC/DB/HPSP 눌림 매수
- EYE-07: 252670/HPSP 워치리스트 급등 (HPSP 4번 = 황금 표준)

자비스 텔레그램만 발송 → advisory 미통합 = 사장님 카톡 사각지대

데이터 소스:
- data/vwap_monitor.json (현재 VWAP 상태, 매 분 갱신)
- logs/intraday_eye.log (EYE 이벤트 grep)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VWAP_STATE_PATH = PROJECT_ROOT / "data" / "vwap_monitor.json"
EYE_LOG_PATH = PROJECT_ROOT / "logs" / "intraday_eye.log"


def get_vwap_state(min_dip_dev: float = -1.5, min_overheat_dev: float = 2.5) -> dict:
    """vwap_monitor.json에서 현재 VWAP 상태 추출.

    Args:
        min_dip_dev: 눌림 임계 (VWAP -1.5% 이하)
        min_overheat_dev: 과열 임계 (VWAP +2.5% 이상)

    Returns:
        {
            "updated_at": str,
            "dips": [{ticker, name, vwap_dev_pct, ...}, ...],
            "overheats": [{...}, ...],
            "n_total": int,
        }
    """
    if not VWAP_STATE_PATH.exists():
        return {"updated_at": None, "dips": [], "overheats": [], "n_total": 0}

    try:
        data = json.loads(VWAP_STATE_PATH.read_text(encoding="utf-8"))
        stocks = data.get("stocks", {})
        dips = []
        overheats = []
        for ticker, st in stocks.items():
            dev = st.get("vwap_dev_pct", 0)
            entry = {
                "ticker": ticker,
                "name": st.get("name", ticker),
                "current": st.get("current_price"),
                "vwap": st.get("vwap"),
                "vwap_dev_pct": dev,
                "dip_count": st.get("dip_count", 0),
                "day_high": st.get("day_high"),
                "day_low": st.get("day_low"),
            }
            if dev <= min_dip_dev:
                dips.append(entry)
            elif dev >= min_overheat_dev:
                overheats.append(entry)

        # 정렬
        dips.sort(key=lambda x: x["vwap_dev_pct"])  # 가장 깊은 눌림 먼저
        overheats.sort(key=lambda x: x["vwap_dev_pct"], reverse=True)  # 가장 강한 과열

        return {
            "updated_at": data.get("updated_at"),
            "dips": dips,
            "overheats": overheats,
            "n_total": len(stocks),
        }
    except Exception as e:
        logger.warning("vwap_monitor.json 로드 실패: %s", e)
        return {"updated_at": None, "dips": [], "overheats": [], "n_total": 0}


EYE_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d+,\d+\].*\[EYE-(\d{2})\] (\w+)\(\w+\).*?현재가 ([\d,]+)원 \(([+\-]?\d+\.?\d*)%\).*?사유: (.+?)(?:$|\|)"
)


def get_recent_eye_events(minutes: int = 120, eye_ids: list[str] | None = None) -> list[dict]:
    """intraday_eye.log에서 최근 N분간 EYE 이벤트 추출.

    Args:
        minutes: 최근 N분 (기본 2시간)
        eye_ids: 필터 (예: ['EYE-07']), None이면 전체

    Returns:
        [{time, eye_id, ticker, price, change_pct, reason}, ...]
    """
    if not EYE_LOG_PATH.exists():
        return []

    cutoff = datetime.now() - timedelta(minutes=minutes)
    events = []

    try:
        # 마지막 1000줄만 읽기 (성능)
        with open(EYE_LOG_PATH, encoding="utf-8") as f:
            lines = f.readlines()[-1000:]

        for line in lines:
            m = EYE_PATTERN.search(line)
            if not m:
                continue
            ts, eye_id, ticker, price, chg, reason = m.groups()
            try:
                event_time = datetime.strptime(ts, "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if event_time < cutoff:
                continue
            if eye_ids and f"EYE-{eye_id}" not in eye_ids:
                continue
            events.append({
                "time": ts,
                "eye_id": f"EYE-{eye_id}",
                "ticker": ticker,
                "price": int(price.replace(",", "")),
                "change_pct": float(chg),
                "reason": reason.strip(),
            })
    except Exception as e:
        logger.warning("intraday_eye.log 파싱 실패: %s", e)

    return events


def count_eye_per_ticker(events: list[dict]) -> dict:
    """종목별 EYE 알림 횟수 (HPSP 4번 같은 황금 표준 마커)."""
    counts = {}
    for e in events:
        tk = e["ticker"]
        counts[tk] = counts.get(tk, 0) + 1
    return counts


def format_vwap_for_telegram(state: dict) -> str:
    """텔레그램 1줄."""
    lines = []
    if state["dips"]:
        top_dip = state["dips"][0]
        lines.append(f"⬇️ VWAP 눌림 TOP: {top_dip['name']}({top_dip['ticker']}) {top_dip['vwap_dev_pct']:+.1f}%")
    if state["overheats"]:
        top_oh = state["overheats"][0]
        lines.append(f"⬆️ VWAP 과열 TOP: {top_oh['name']}({top_oh['ticker']}) {top_oh['vwap_dev_pct']:+.1f}%")
    return "\n".join(lines)


def format_eye_for_telegram(events: list[dict], ticker_counts: dict) -> str:
    """텔레그램 1~2줄."""
    if not events:
        return ""
    # 가장 많이 알림 받은 종목 = 황금 표준
    top_ticker = max(ticker_counts, key=ticker_counts.get) if ticker_counts else None
    if top_ticker:
        n = ticker_counts[top_ticker]
        return f"✨ EYE 황금 표준: {top_ticker} {n}번 알림 (최근 2시간)"
    return ""
