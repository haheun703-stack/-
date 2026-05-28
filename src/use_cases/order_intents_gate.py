"""Order Intents Gate — "No Intent, No Order" 강제 가드 (Trading Factory v1)

배경 (5/28 사용자 결단):
  봇들이 각자 떠드는 시스템 → 기관식 매매 공장 전환.
  모든 주문은 `data/order_intents/{bot}_intents_YYYYMMDD.jsonl`에 사전 등록 필수.
  intent 없는 종목 주문 시 RuntimeError → 매매 차단.

사용:
  from src.use_cases.order_intents_gate import assert_order_intent_exists
  intent = assert_order_intent_exists(ticker, side="BUY", mode="paper")
  # intent 없으면 RuntimeError, 있으면 intent dict 반환

연결:
  - docs/01-plan/trading-factory-v1-architecture.md §7 (No Intent, No Order 가드)
  - docs/02-design/quant-runtime-truth-pack.md §3 L9 (10번째 가드 레이어)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDER_INTENTS_DIR = PROJECT_ROOT / "data" / "order_intents"

# 활성 봇 목록 (Trading Factory v1)
ACTIVE_BOTS = ("quant", "day")  # 정보봇/블로그봇은 매매 X


class NoIntentError(RuntimeError):
    """order_intents에 매칭되는 entry 없음 — 매매 차단."""


def _today_date_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _intent_files(date_str: str = None) -> list[Path]:
    """오늘 (또는 지정 날짜) 모든 봇의 order_intents 파일 목록."""
    date_str = date_str or _today_date_str()
    if not ORDER_INTENTS_DIR.exists():
        return []
    files = []
    for bot in ACTIVE_BOTS:
        f = ORDER_INTENTS_DIR / f"{bot}_intents_{date_str}.jsonl"
        if f.exists():
            files.append(f)
    return files


def _load_intents(files: list[Path]) -> list[dict]:
    """파일 list에서 모든 intent 로드."""
    intents = []
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    intents.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("[order_intents_gate] JSON parse error in %s: %s", f.name, e)
        except Exception as e:
            logger.warning("[order_intents_gate] file read error %s: %s", f.name, e)
    return intents


def assert_order_intent_exists(
    ticker: str,
    side: str = "BUY",
    mode: str = "paper",
    horizon_buffer_minutes: int = 60,
) -> dict:
    """모든 매매 주문 함수 진입 시 호출. order_intents 미등록 종목 차단.

    Args:
        ticker: 종목 코드 (6자리 또는 일반)
        side: "BUY" / "SELL"
        mode: "paper" / "live"
        horizon_buffer_minutes: expires_at 이전 버퍼 (시간 여유, default 60분)

    Returns:
        매칭된 intent dict

    Raises:
        NoIntentError: order_intents에 매칭 entry 없음

    환경변수:
        ORDER_INTENTS_GATE_DISABLED=1 → 가드 비활성 (긴급 우회, 권장 X)
    """
    # 긴급 비활성 (테스트/마이그레이션 한정)
    if os.getenv("ORDER_INTENTS_GATE_DISABLED", "0") == "1":
        logger.warning(
            "[order_intents_gate] DISABLED (env ORDER_INTENTS_GATE_DISABLED=1) — "
            "%s %s %s 우회. 권장 X.", ticker, side, mode,
        )
        return {"_bypass": True, "ticker": ticker, "side": side, "mode": mode}

    ticker = str(ticker).strip()
    side = side.upper()
    mode = mode.lower()

    files = _intent_files()
    if not files:
        raise NoIntentError(
            f"[NO_INTENT] order_intents 파일 없음 (오늘 {_today_date_str()}) — "
            f"{ticker} {side} {mode} 차단. "
            f"수동 매매 시 ORDER_INTENTS_GATE_DISABLED=1 임시 우회 가능."
        )

    intents = _load_intents(files)
    if not intents:
        raise NoIntentError(
            f"[NO_INTENT] order_intents 비어있음 ({len(files)}개 파일) — "
            f"{ticker} {side} {mode} 차단."
        )

    now_iso = datetime.now().isoformat()

    for intent in intents:
        if intent.get("ticker") != ticker:
            continue
        if intent.get("side", "").upper() != side:
            continue
        if intent.get("mode", "").lower() != mode:
            continue
        # 유효 시간 체크
        expires_at = intent.get("expires_at", "")
        if expires_at and expires_at < now_iso:
            continue
        # 매칭
        logger.info(
            "[order_intents_gate] PASS — %s %s %s (intent_id=%s, engine=%s)",
            ticker, side, mode,
            intent.get("intent_id", "?"), intent.get("engine", "?"),
        )
        return intent

    # 매칭 실패
    matched_tickers = [i.get("ticker") for i in intents if i.get("side", "").upper() == side]
    raise NoIntentError(
        f"[NO_INTENT] {ticker} {side} {mode} 미등록 — "
        f"오늘 등록된 {side} intent: {len(matched_tickers)}건 "
        f"({matched_tickers[:5]}{'...' if len(matched_tickers) > 5 else ''})."
    )


def list_today_intents(side: str = None, mode: str = None) -> list[dict]:
    """오늘 등록된 intent 전체 또는 필터링 (Codex 검수용)."""
    files = _intent_files()
    intents = _load_intents(files)
    if side:
        intents = [i for i in intents if i.get("side", "").upper() == side.upper()]
    if mode:
        intents = [i for i in intents if i.get("mode", "").lower() == mode.lower()]
    return intents


def register_intent(intent: dict, bot: str = "quant") -> Path:
    """봇이 새 intent 등록 (selector가 호출). append-only jsonl.

    Args:
        intent: intent dict (스키마 검증 후 저장)
        bot: "quant" / "day"
    """
    required_fields = {
        "intent_id", "bot", "engine", "ticker", "side", "mode",
        "score", "created_at", "expires_at",
    }
    missing = required_fields - set(intent.keys())
    if missing:
        raise ValueError(f"[register_intent] 필수 필드 누락: {missing}")

    if intent.get("bot") != bot:
        raise ValueError(f"[register_intent] bot 불일치: intent.bot={intent.get('bot')}, arg bot={bot}")

    if bot not in ACTIVE_BOTS:
        raise ValueError(f"[register_intent] 허용 봇 외: {bot} (허용: {ACTIVE_BOTS})")

    ORDER_INTENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ORDER_INTENTS_DIR / f"{bot}_intents_{_today_date_str()}.jsonl"

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(intent, ensure_ascii=False) + "\n")

    logger.info(
        "[order_intents_gate] register %s %s %s — intent_id=%s engine=%s",
        intent.get("ticker"), intent.get("side"), intent.get("mode"),
        intent.get("intent_id"), intent.get("engine"),
    )
    return out_path
