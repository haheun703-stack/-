"""AI 밸류체인 동조 발화 시 워치리스트 자동 추가 — 5/26 신규.

배경:
- ai_chain_detector가 4섹터 동시 폭등 감지 시 워치리스트 자동 갱신 필요
- 매번 사용자가 수동으로 settings.yaml 편집 X
- 자동 추가 = 즉시 EYE-07 모니터링 진입 (다음 사이클부터 알림)

룰:
1. AI 동조 발화 + 폭등 종목 (+5% 이상) 발견 시
2. 워치리스트 미등록 + 보호종목 제외 + 이미 보유 제외
3. data/ai_chain_watchlist.json 에 자동 추가 (만료 7일)
4. intraday_eye가 settings.yaml + ai_chain_watchlist.json 모두 읽음
5. 7일 후 자동 만료 (재발화 시 갱신)

저장 형식 (data/ai_chain_watchlist.json):
{
  "added_at": "2026-05-26T11:35:00",
  "tickers": [
    {"ticker": "095340", "name": "ISC", "sector": "AI반도체검사",
     "added_at": "2026-05-26T11:35:00", "expires_at": "2026-06-02T11:35:00",
     "trigger_reason": "AI 4섹터 동조 발화 (16.9%)"}
  ]
}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WATCHLIST_PATH = PROJECT_ROOT / "data" / "ai_chain_watchlist.json"
EXPIRY_DAYS = int(os.getenv("AI_CHAIN_WATCHLIST_EXPIRY_DAYS", "7"))


def load_ai_chain_watchlist() -> dict:
    """ai_chain_watchlist.json 로드. 만료 항목 자동 제외."""
    if not WATCHLIST_PATH.exists():
        return {"added_at": "", "tickers": []}

    try:
        d = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[AI watchlist] 로드 실패: %s", e)
        return {"added_at": "", "tickers": []}

    # 만료 항목 제거
    now = datetime.now()
    valid = []
    for x in d.get("tickers", []):
        try:
            exp = datetime.fromisoformat(x.get("expires_at", ""))
            if exp > now:
                valid.append(x)
        except (ValueError, TypeError):
            continue
    d["tickers"] = valid
    return d


def add_to_ai_chain_watchlist(
    surge_stocks: list[dict],
    protected_tickers: set[str] | None = None,
    held_tickers: set[str] | None = None,
) -> dict:
    """폭등 종목들을 워치리스트에 자동 추가.

    Args:
        surge_stocks: ai_chain_detector.detect_ai_chain_sync() 결과의 surge_stocks
        protected_tickers: 보호 종목 (이미 보유 + 자동 매매 제외)
        held_tickers: 현재 보유 종목 (제외)

    Returns:
        {"added": [...], "skipped": [...], "total": int}
    """
    if not surge_stocks:
        return {"added": [], "skipped": [], "total": 0}

    protected_tickers = protected_tickers or set()
    held_tickers = held_tickers or set()

    current = load_ai_chain_watchlist()
    existing_tickers = {x["ticker"] for x in current.get("tickers", [])}

    now = datetime.now()
    expires = now + timedelta(days=EXPIRY_DAYS)
    added = []
    skipped = []

    for s in surge_stocks:
        tk = str(s.get("ticker", "")).zfill(6)
        if not tk:
            continue
        # 제외 사유 검사
        if tk in protected_tickers:
            skipped.append({"ticker": tk, "reason": "PROTECTED"})
            continue
        if tk in held_tickers:
            skipped.append({"ticker": tk, "reason": "HELD"})
            continue
        if tk in existing_tickers:
            skipped.append({"ticker": tk, "reason": "ALREADY_LISTED"})
            continue

        entry = {
            "ticker": tk,
            "name": s.get("name", "") or tk,
            "sector": s.get("sector", ""),
            "added_at": now.isoformat(timespec="seconds"),
            "expires_at": expires.isoformat(timespec="seconds"),
            "trigger_reason": (
                f"AI 동조 발화 ({s.get('change_pct', 0):+.1f}% — {s.get('sector', '')})"
            ),
            "added_price": s.get("current_price", 0),
        }
        added.append(entry)

    if not added:
        return {"added": [], "skipped": skipped, "total": len(current["tickers"])}

    current["tickers"].extend(added)
    current["added_at"] = now.isoformat(timespec="seconds")

    # 원자적 저장
    try:
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = WATCHLIST_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(WATCHLIST_PATH)
        logger.info(
            "[AI watchlist] %d종목 추가, %d종목 스킵 (총 %d)",
            len(added), len(skipped), len(current["tickers"]),
        )
    except Exception as e:
        logger.error("[AI watchlist] 저장 실패: %s", e)
        return {"added": [], "skipped": skipped, "total": len(current["tickers"]),
                "error": str(e)}

    return {"added": added, "skipped": skipped, "total": len(current["tickers"])}


def get_ai_chain_watchlist_tickers() -> list[str]:
    """현재 활성 (만료 X) AI 동조 워치리스트 ticker 리스트.

    intraday_eye가 settings.yaml watchlist와 합쳐 사용.
    """
    d = load_ai_chain_watchlist()
    return [x["ticker"] for x in d.get("tickers", [])]


def format_added_for_telegram(result: dict) -> str:
    """추가 결과 텔레그램 포맷."""
    added = result.get("added", [])
    if not added:
        return "📭 AI 동조 워치리스트 추가 없음"

    lines = [
        f"📌 [AI 동조 워치리스트 자동 추가] {len(added)}종목",
    ]
    for a in added[:10]:
        lines.append(
            f"  {a['ticker']} {a['name']} ({a['sector']}) — {a['trigger_reason']}"
        )
    lines.append(f"  만료: {EXPIRY_DAYS}일 후")
    return "\n".join(lines)
