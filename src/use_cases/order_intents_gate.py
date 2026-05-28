"""Order Intents Gate — "No Intent, No Order" 강제 가드 (Trading Factory v1)

배경 (5/28 사용자 결단):
  봇들이 각자 떠드는 시스템 → 기관식 매매 공장 전환.
  모든 주문은 `data/order_intents/{bot}_intents_YYYYMMDD.jsonl`에 사전 등록 필수.
  intent 없는 종목 주문 시 OrderIntentError → 매매 차단.

코덱스 1차 검수 P0 fix (5/28 13:01):
  P0-1: ORDER_INTENTS_GATE_DISABLED 런타임 우회 제거 (코드 레벨 영구 금지)
  P0-2: mode 기본값 제거 (필수 인자, paper/live 명시)
  P0-3: executor_bot 인자 추가 + intent.bot 매치 검증
  P0-4: expires_at datetime 파싱 강화 + 잘못된 만료값 엄격 거부
  P0-5: HMAC 서명 (위조 방지) — ORDER_INTENTS_HMAC_KEY 환경변수

사용:
  from src.use_cases.order_intents_gate import assert_order_intent_exists
  intent = assert_order_intent_exists(
      ticker="240810", side="BUY", mode="paper", executor_bot="quant"
  )

연결:
  - docs/01-plan/trading-factory-v1-architecture.md §7
  - docs/02-design/quant-runtime-truth-pack.md §3 L9
  - ops/codex_outbox/20260528T130158..._changes-requested.md
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDER_INTENTS_DIR = PROJECT_ROOT / "data" / "order_intents"

# 활성 봇 목록 (Trading Factory v1)
ACTIVE_BOTS = ("quant", "day")  # 정보봇/블로그봇은 매매 X

# 허용 mode (P0-2: 명시 강제, 기본값 X)
ALLOWED_MODES = ("paper", "live")

# HMAC 서명에서 제외할 필드 (서명 자체)
HMAC_EXCLUDED_FIELDS = ("hmac_signature",)


class OrderIntentError(RuntimeError):
    """order_intents 가드 차단 — 매매 거부."""


class NoIntentError(OrderIntentError):
    """order_intents에 매칭되는 entry 없음."""


class IntentSignatureError(OrderIntentError):
    """HMAC 서명 검증 실패 — 위조 의심."""


class IntentExpiredError(OrderIntentError):
    """intent 만료."""


class IntentSchemaError(OrderIntentError):
    """intent 스키마 부적합 (expires_at 파싱 실패 등)."""


# ──────────────────────────────────────────────
# HMAC 서명 (P0-5)
# ──────────────────────────────────────────────
def _get_hmac_key() -> bytes:
    """ORDER_INTENTS_HMAC_KEY 환경변수 → bytes.

    환경변수 미설정 시 IntentSignatureError 발생.
    개발 환경에서도 ORDER_INTENTS_HMAC_KEY 필수 (위조 방지).
    """
    key_str = os.getenv("ORDER_INTENTS_HMAC_KEY", "")
    if not key_str:
        raise IntentSignatureError(
            "[HMAC] ORDER_INTENTS_HMAC_KEY 환경변수 미설정 — intent 서명/검증 불가. "
            ".env에 32+ 문자 무작위 키 설정 필수."
        )
    if len(key_str) < 32:
        raise IntentSignatureError(
            f"[HMAC] ORDER_INTENTS_HMAC_KEY 길이 부족 ({len(key_str)} < 32) — 보안 강화 필요."
        )
    return key_str.encode("utf-8")


def _compute_signature(intent: dict) -> str:
    """intent dict의 HMAC-SHA256 서명 (서명 필드 제외).

    Deterministic JSON (sorted keys) → HMAC-SHA256 → hex digest.
    """
    key = _get_hmac_key()
    payload = {k: v for k, v in intent.items() if k not in HMAC_EXCLUDED_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    sig = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig


def _verify_signature(intent: dict) -> bool:
    """intent의 hmac_signature 검증. 누락 시 False (위조)."""
    stored_sig = intent.get("hmac_signature", "")
    if not stored_sig:
        return False
    expected_sig = _compute_signature(intent)
    return hmac.compare_digest(stored_sig, expected_sig)


# ──────────────────────────────────────────────
# expires_at 파싱 (P0-4 — 코덱스 2차 응답 5/28 13:10 정합)
#
# 코덱스 지시:
#   - timezone-naive 값 거부
#   - Asia/Seoul 또는 timezone-aware ISO만 허용
#   - 모든 비교는 timezone-aware datetime으로 통일
# ──────────────────────────────────────────────
def _parse_expires_at(intent: dict) -> datetime:
    """intent.expires_at → timezone-aware datetime.

    Raises:
        IntentSchemaError:
            - 파싱 실패 또는 누락
            - timezone-naive 값 (코덱스 2차 응답 거부)
    """
    raw = intent.get("expires_at", "")
    if not raw:
        raise IntentSchemaError(
            f"[SCHEMA] expires_at 누락 — intent_id={intent.get('intent_id', '?')}"
        )
    try:
        # ISO 8601 (timezone 필수)
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError) as e:
        raise IntentSchemaError(
            f"[SCHEMA] expires_at 파싱 실패 ({raw}) — intent_id={intent.get('intent_id', '?')}: {e}"
        )

    # P0-4 (5/28 코덱스 2차): timezone-naive 거부
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise IntentSchemaError(
            f"[SCHEMA] expires_at timezone-naive 거부 ({raw}) — "
            f"intent_id={intent.get('intent_id', '?')}. "
            f"Asia/Seoul (예: 2026-05-31T15:30:00+09:00) 또는 timezone-aware ISO만 허용."
        )
    return dt


def _now_aware() -> datetime:
    """현재 시각 — 항상 timezone-aware (UTC 기준, 비교에 사용).

    expires_at도 timezone-aware라 비교 가능. UTC offset 차이 자동 처리.
    """
    return datetime.now(tz=timezone.utc)


# ──────────────────────────────────────────────
# 파일 I/O
# ──────────────────────────────────────────────
def _today_date_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _intent_files(date_str: str = None) -> list[Path]:
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


# ──────────────────────────────────────────────
# 메인 API: 가드 통과 검증
# ──────────────────────────────────────────────
def assert_order_intent_exists(
    ticker: str,
    side: str,
    mode: str,
    executor_bot: str,
) -> dict:
    """모든 매매 주문 함수 진입 시 호출 (No Intent, No Order).

    Args (모두 명시 필수, 기본값 없음 — P0-2):
        ticker: 종목 코드
        side: "BUY" / "SELL"
        mode: "paper" / "live" (P0-2 명시 강제)
        executor_bot: "quant" / "day" (P0-3 — intent.bot과 매치 검증)

    Returns:
        매칭된 intent dict (HMAC 검증 통과)

    Raises:
        OrderIntentError 계열:
            - NoIntentError: 매칭 entry 없음
            - IntentSignatureError: HMAC 검증 실패 (위조)
            - IntentExpiredError: 만료
            - IntentSchemaError: 잘못된 스키마

    Note:
        P0-1: ORDER_INTENTS_GATE_DISABLED 우회 영구 제거 (코드 레벨)
        이 함수가 raise하면 어떤 환경변수로도 우회 불가.
        수동 매매 시 register_intent로 정식 intent 등록 필요.
    """
    # 입력 검증
    ticker = str(ticker).strip()
    side = str(side).upper()
    mode = str(mode).lower()
    executor_bot = str(executor_bot).lower()

    if side not in ("BUY", "SELL"):
        raise OrderIntentError(f"[INPUT] side 잘못: {side} (BUY/SELL만 허용)")
    if mode not in ALLOWED_MODES:
        raise OrderIntentError(f"[INPUT] mode 잘못: {mode} ({ALLOWED_MODES}만 허용)")
    if executor_bot not in ACTIVE_BOTS:
        raise OrderIntentError(
            f"[INPUT] executor_bot 잘못: {executor_bot} ({ACTIVE_BOTS}만 허용)"
        )

    # 파일 로드
    files = _intent_files()
    if not files:
        raise NoIntentError(
            f"[NO_INTENT] order_intents 파일 없음 (오늘 {_today_date_str()}) — "
            f"{ticker} {side} {mode} executor={executor_bot} 차단."
        )

    intents = _load_intents(files)
    if not intents:
        raise NoIntentError(
            f"[NO_INTENT] order_intents 비어있음 ({len(files)}개 파일)."
        )

    # P0-4: 항상 timezone-aware now 사용 (UTC 기준)
    now = _now_aware()

    # 매칭 (P0-3: executor_bot도 매치)
    for intent in intents:
        if intent.get("ticker") != ticker:
            continue
        if str(intent.get("side", "")).upper() != side:
            continue
        if str(intent.get("mode", "")).lower() != mode:
            continue
        # P0-3: intent.bot이 executor_bot과 일치해야 함
        if str(intent.get("bot", "")).lower() != executor_bot:
            continue

        # P0-5: HMAC 서명 검증 (위조 방지)
        if not _verify_signature(intent):
            raise IntentSignatureError(
                f"[HMAC] intent 서명 검증 실패 — intent_id={intent.get('intent_id', '?')} "
                f"위조 의심. register_intent로 정식 재등록 필요."
            )

        # P0-4 (코덱스 2차 정합): expires_at 엄격 파싱 (timezone-aware 강제) + 만료 검증
        # _parse_expires_at이 timezone-naive 거부 → 양쪽 모두 timezone-aware 비교 보장
        try:
            expires_dt = _parse_expires_at(intent)
        except IntentSchemaError:
            raise  # propagate schema error

        if expires_dt < now:
            raise IntentExpiredError(
                f"[EXPIRED] intent 만료 — intent_id={intent.get('intent_id', '?')} "
                f"expires_at={expires_dt.isoformat()}, now={now.isoformat()}"
            )

        # 매칭 통과
        logger.info(
            "[order_intents_gate] PASS — %s %s %s executor=%s (intent_id=%s engine=%s)",
            ticker, side, mode, executor_bot,
            intent.get("intent_id", "?"), intent.get("engine", "?"),
        )
        return intent

    # 매칭 실패
    matched_tickers = [
        i.get("ticker") for i in intents
        if str(i.get("side", "")).upper() == side
        and str(i.get("bot", "")).lower() == executor_bot
    ]
    raise NoIntentError(
        f"[NO_INTENT] {ticker} {side} {mode} executor={executor_bot} 미등록 — "
        f"오늘 등록된 {side}/{executor_bot} intent: {len(matched_tickers)}건 "
        f"({matched_tickers[:5]}{'...' if len(matched_tickers) > 5 else ''})."
    )


# ──────────────────────────────────────────────
# 메인 API: intent 등록 (selector가 호출)
# ──────────────────────────────────────────────
REQUIRED_INTENT_FIELDS = frozenset({
    "intent_id", "bot", "engine", "ticker", "side", "mode",
    "score", "created_at", "expires_at",
})


def register_intent(intent: dict, bot: str) -> Path:
    """selector가 새 intent 등록. HMAC 서명 후 append-only jsonl 저장.

    Args:
        intent: intent dict (스키마 검증 + 서명 추가)
        bot: "quant" / "day" (intent.bot과 매치 필수)

    Raises:
        OrderIntentError: 스키마 부적합 또는 봇 허용 외
        IntentSignatureError: HMAC 키 미설정
    """
    if bot not in ACTIVE_BOTS:
        raise OrderIntentError(f"[REGISTER] 허용 봇 외: {bot} (허용: {ACTIVE_BOTS})")

    missing = REQUIRED_INTENT_FIELDS - set(intent.keys())
    if missing:
        raise OrderIntentError(f"[REGISTER] 필수 필드 누락: {missing}")

    if str(intent.get("bot", "")).lower() != bot:
        raise OrderIntentError(
            f"[REGISTER] bot 불일치: intent.bot={intent.get('bot')}, arg bot={bot}"
        )

    if str(intent.get("mode", "")).lower() not in ALLOWED_MODES:
        raise OrderIntentError(
            f"[REGISTER] mode 잘못: {intent.get('mode')} ({ALLOWED_MODES}만 허용)"
        )

    if str(intent.get("side", "")).upper() not in ("BUY", "SELL"):
        raise OrderIntentError(
            f"[REGISTER] side 잘못: {intent.get('side')} (BUY/SELL만 허용)"
        )

    # Note 1 (5/28 코덱스 3차): quant는 research/selector bot이므로 live intent 등록 금지
    # quant_intents에는 mode="paper"만 허용. live 매매는 별도 executor (day bot)만.
    if bot == "quant" and str(intent.get("mode", "")).lower() == "live":
        raise OrderIntentError(
            "[REGISTER] quant bot은 live mode intent 등록 금지 — research/selector 역할만 수행. "
            "live 매매는 day bot 또는 별도 승인된 executor만 가능."
        )

    # P0-4: expires_at 파싱 가능 여부 검증 (등록 시점에)
    _parse_expires_at(intent)

    # P0-5: HMAC 서명 추가
    intent["hmac_signature"] = _compute_signature(intent)

    # 저장
    ORDER_INTENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ORDER_INTENTS_DIR / f"{bot}_intents_{_today_date_str()}.jsonl"

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(intent, ensure_ascii=False) + "\n")

    logger.info(
        "[order_intents_gate] register %s %s %s — intent_id=%s engine=%s sig=%s",
        intent.get("ticker"), intent.get("side"), intent.get("mode"),
        intent.get("intent_id"), intent.get("engine"),
        intent["hmac_signature"][:8] + "...",
    )
    return out_path


# ──────────────────────────────────────────────
# 검수/감리 API
# ──────────────────────────────────────────────
def list_today_intents(
    side: str = None,
    mode: str = None,
    executor_bot: str = None,
    verify_signatures: bool = True,
) -> list[dict]:
    """오늘 등록된 intent 전체 또는 필터링 (Codex 검수용).

    Args:
        verify_signatures: True → HMAC 검증 실패 entry는 _signature_valid=False 표시
    """
    files = _intent_files()
    intents = _load_intents(files)

    if verify_signatures:
        for i in intents:
            try:
                i["_signature_valid"] = _verify_signature(i)
            except IntentSignatureError:
                i["_signature_valid"] = False

    if side:
        intents = [i for i in intents if str(i.get("side", "")).upper() == side.upper()]
    if mode:
        intents = [i for i in intents if str(i.get("mode", "")).lower() == mode.lower()]
    if executor_bot:
        intents = [i for i in intents if str(i.get("bot", "")).lower() == executor_bot.lower()]
    return intents
