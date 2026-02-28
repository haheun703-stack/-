"""
텔레그램 봇 메시지 송출 모듈

.env에서 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 로드.
텔레그램 Bot API를 직접 호출 (python-telegram-bot 불필요).
"""

import logging
import os
import time as _time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .env 로드
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# 긴급 알림 폴백 로그 경로
ALERT_FALLBACK_PATH = Path(__file__).resolve().parent.parent / "logs" / "emergency_alerts.log"

API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# 텔레그램 메시지 최대 길이
MAX_MESSAGE_LENGTH = 4096


def send_message(
    text: str,
    chat_id: str = None,
    parse_mode: str = None,
    reply_markup: dict = None,
) -> bool:
    """
    텔레그램 메시지 전송.

    Args:
        text: 전송할 메시지 (4096자 초과 시 자동 분할)
        chat_id: 대상 채팅 ID (없으면 .env 값 사용)
        parse_mode: "HTML" / "MarkdownV2" / None (plain text)
        reply_markup: ReplyKeyboardMarkup / InlineKeyboardMarkup dict

    Returns:
        True if all parts sent successfully
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다 (.env 확인)")
        return False

    target_chat = chat_id or TELEGRAM_CHAT_ID
    if not target_chat:
        logger.error("TELEGRAM_CHAT_ID가 설정되지 않았습니다 (.env 확인)")
        return False

    # 메시지 분할 (4096자 제한)
    chunks = _split_message(text, MAX_MESSAGE_LENGTH)

    success = True
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": target_chat,
            "text": chunk,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        # reply_markup은 마지막 청크에만 첨부
        if reply_markup and i == len(chunks) - 1:
            payload["reply_markup"] = reply_markup

        sent = False
        for attempt in range(2):  # 1회 재시도
            try:
                resp = requests.post(f"{API_BASE}/sendMessage", json=payload, timeout=10)
                if resp.status_code == 200 and resp.json().get("ok"):
                    if len(chunks) > 1:
                        logger.debug(f"텔레그램 전송 성공 ({i+1}/{len(chunks)})")
                    sent = True
                    break
                else:
                    logger.error(f"텔레그램 전송 실패: {resp.status_code} {resp.text}")
            except requests.RequestException as e:
                logger.error(f"텔레그램 전송 오류 ({attempt+1}/2): {e}")
            if attempt == 0:
                _time.sleep(3)  # 재시도 전 3초 대기
        if not sent:
            success = False

    if success:
        logger.info(f"텔레그램 메시지 전송 완료 ({len(chunks)}건)")
    else:
        _write_fallback_alert(text)

    return success


def answer_callback_query(
    callback_query_id: str, text: str = "", show_alert: bool = False,
) -> bool:
    """InlineKeyboard 버튼 클릭 응답 (토스트 메시지)."""
    payload: dict = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    if show_alert:
        payload["show_alert"] = True
    try:
        resp = requests.post(f"{API_BASE}/answerCallbackQuery", json=payload, timeout=10)
        return resp.status_code == 200 and resp.json().get("ok", False)
    except requests.RequestException as e:
        logger.error(f"answerCallbackQuery 오류: {e}")
        return False


def edit_message_text(
    chat_id: str,
    message_id: int,
    text: str,
    reply_markup: dict = None,
    parse_mode: str = None,
) -> bool:
    """기존 메시지 텍스트 수정 (확인/취소 후 결과 업데이트)."""
    target_chat = chat_id or TELEGRAM_CHAT_ID
    payload: dict = {
        "chat_id": target_chat,
        "message_id": message_id,
        "text": text,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        resp = requests.post(f"{API_BASE}/editMessageText", json=payload, timeout=10)
        return resp.status_code == 200 and resp.json().get("ok", False)
    except requests.RequestException as e:
        logger.error(f"editMessageText 오류: {e}")
        return False


def send_backtest_report(results: dict, scan_date: str = None) -> bool:
    """
    백테스트 결과를 텔레그램으로 전송.

    Args:
        results: BacktestEngine._compile_results() 반환값
        scan_date: 스캔 날짜
    """
    from .telegram_formatter import format_backtest_report

    message = format_backtest_report(results, scan_date)
    return send_message(message)


def send_trade_alert(signal: dict, action: str = "BUY") -> bool:
    """
    실시간 매매 알림 전송.

    Args:
        signal: 시그널 dict
        action: "BUY" / "SELL"
    """
    from .telegram_formatter import format_trade_alert

    message = format_trade_alert(signal, action)
    return send_message(message)


def send_news_alert(
    ticker: str,
    grade: str,
    action: str,
    reason: str = "",
    news_text: str = "",
    param_overrides: dict = None,
    entry_price: float = 0,
    target_price: float = 0,
    stop_loss: float = 0,
    pipeline_passed: bool = False,
) -> bool:
    """v3.1 뉴스 등급 알림 전송."""
    from .telegram_formatter import format_news_alert

    message = format_news_alert(
        ticker=ticker, grade=grade, action=action, reason=reason,
        news_text=news_text, param_overrides=param_overrides,
        entry_price=entry_price, target_price=target_price,
        stop_loss=stop_loss, pipeline_passed=pipeline_passed,
    )
    return send_message(message)


def send_accumulation_alert(
    ticker: str,
    phase: int,
    confidence: float = 0,
    bonus_score: int = 0,
    inst_streak: int = 0,
    foreign_streak: int = 0,
    obv_divergence: str = "",
    description: str = "",
) -> bool:
    """v3.1 매집 단계 알림 전송."""
    from .telegram_formatter import format_accumulation_alert

    message = format_accumulation_alert(
        ticker=ticker, phase=phase, confidence=confidence,
        bonus_score=bonus_score, inst_streak=inst_streak,
        foreign_streak=foreign_streak, obv_divergence=obv_divergence,
        description=description,
    )
    return send_message(message)


def send_scan_result(
    stats: dict,
    signals: list[dict] = None,
    diagnostic: dict = None,
    scan_date: str = None,
) -> bool:
    """
    스캔 결과 전체 메시지 전송.
    """
    from .telegram_formatter import format_scan_result

    message = format_scan_result(stats, signals, diagnostic, scan_date)
    return send_message(message)


def send_order_result(order, action: str = "BUY", name: str = "") -> bool:
    """주문 체결/실패 알림 전송."""
    from .telegram_formatter import format_order_result

    message = format_order_result(order, action, name=name)
    return send_message(message)


def send_position_summary(positions: list) -> bool:
    """보유종목 현황 전송."""
    from .telegram_formatter import format_position_summary

    message = format_position_summary(positions)
    return send_message(message)


def send_daily_performance(perf) -> bool:
    """일일 성과 리포트 전송."""
    from .telegram_formatter import format_daily_performance

    message = format_daily_performance(perf)
    return send_message(message)


def send_emergency_alert(reason: str) -> bool:
    """긴급 알림 전송 (텔레그램 + 파일 폴백 항상 기록)."""
    from .telegram_formatter import format_emergency_alert

    message = format_emergency_alert(reason)
    # 긴급 알림은 텔레그램 성공 여부와 무관하게 항상 파일에 기록
    _write_fallback_alert(f"[EMERGENCY] {reason}\n{message}")
    return send_message(message)


def send_scheduler_status(phase: str, status: str, detail: str = "") -> bool:
    """스케줄러 상태 전송."""
    from .telegram_formatter import format_scheduler_status

    message = format_scheduler_status(phase, status, detail)
    return send_message(message)


def send_mdd_alert(alert_message: str) -> bool:
    """MDD 모니터 알림 전송."""
    if not alert_message:
        return False
    return send_message(alert_message)


def send_market_analysis(data: dict) -> bool:
    """장시작/장마감 분석 보고서 전송."""
    from .telegram_formatter import format_market_analysis

    message = format_market_analysis(data)
    return send_message(message)


def send_theme_alert(alert) -> bool:
    """v11.0 테마 감지 알림 전송.

    Args:
        alert: ThemeAlert 데이터클래스 또는 dict
    """
    from .telegram_formatter import format_theme_alert

    message = format_theme_alert(alert)
    return send_message(message)


def _split_message(text: str, max_length: int) -> list[str]:
    """
    텔레그램 최대 메시지 길이(4096) 기준 분할.
    줄바꿈 단위로 자연스럽게 분할.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_length:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks


def _write_fallback_alert(text: str) -> None:
    """텔레그램 전송 실패 시 로컬 파일에 알림 기록 (대체 경로).

    logs/emergency_alerts.log에 타임스탬프와 함께 기록.
    """
    try:
        ALERT_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ALERT_FALLBACK_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{timestamp}] 텔레그램 전송 실패 — 폴백 기록\n")
            f.write(f"{'='*60}\n")
            f.write(text)
            f.write("\n")
        logger.warning("[폴백] 알림을 %s에 기록했습니다", ALERT_FALLBACK_PATH)
    except Exception as e:
        logger.error("[폴백] 파일 기록도 실패: %s", e)
