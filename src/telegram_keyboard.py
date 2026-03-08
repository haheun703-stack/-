"""텔레그램 ReplyKeyboardMarkup & InlineKeyboardMarkup 빌더."""

from __future__ import annotations

# ── 메인 키보드 (하단 고정, 2행 8버튼) ──
# v13: 운영 중심 재편 — 매수/매도/청산/체결내역 제거 (수동 운영 전환)

MAIN_KEYBOARD = {
    "keyboard": [
        ["현재잔고", "포트폴리오", "리포트", "분석"],
        ["스캔", "시작", "정지", "도움"],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": False,
}


def build_confirm_keyboard(action: str, ticker: str, detail: str = "") -> dict:
    """매수/매도/청산 확인용 InlineKeyboard [확인 실행] [취소]."""
    cb_data = f"confirm:{action}:{ticker}:{detail}" if detail else f"confirm:{action}:{ticker}"
    # callback_data 64바이트 제한 주의
    if len(cb_data.encode("utf-8")) > 64:
        cb_data = cb_data[:60]
    cancel_data = f"cancel:{action}"
    return {
        "inline_keyboard": [
            [
                {"text": "확인 실행", "callback_data": cb_data},
                {"text": "취소", "callback_data": cancel_data},
            ]
        ]
    }


def build_stock_select_keyboard(matches: list[tuple[str, str]], action: str) -> dict:
    """종목 검색 결과 선택 InlineKeyboard (동명이인 구분)."""
    buttons = []
    for name, ticker in matches[:5]:
        buttons.append([{
            "text": f"{name} ({ticker})",
            "callback_data": f"select:{action}:{ticker}",
        }])
    buttons.append([{"text": "취소", "callback_data": "cancel:select"}])
    return {"inline_keyboard": buttons}


def build_ai_buy_keyboard(ticker: str, qty: int, verdict: str) -> dict:
    """AI 매수 판단 결과에 따른 InlineKeyboard.

    verdict:
      BUY_OK → [매수 실행] [눌림 대기] [취소]
      WAIT   → [눌림 대기] [그래도 매수] [취소]
      SKIP   → [그래도 매수] [취소]
    """
    buy_cb = f"ai_buy:{ticker}:{qty}"
    cancel_cb = "cancel:buy"

    if verdict == "BUY_OK":
        return {"inline_keyboard": [[
            {"text": "✅ 매수 실행", "callback_data": buy_cb},
            {"text": "⏳ 눌림 대기", "callback_data": f"ai_wait:{ticker}:{qty}"},
            {"text": "취소", "callback_data": cancel_cb},
        ]]}
    elif verdict == "WAIT":
        return {"inline_keyboard": [[
            {"text": "⏳ 눌림 대기", "callback_data": f"ai_wait:{ticker}:{qty}"},
            {"text": "그래도 매수", "callback_data": buy_cb},
            {"text": "취소", "callback_data": cancel_cb},
        ]]}
    else:  # SKIP or unknown
        return {"inline_keyboard": [[
            {"text": "⚠️ 그래도 매수", "callback_data": buy_cb},
            {"text": "취소", "callback_data": cancel_cb},
        ]]}


def build_ai_sell_keyboard(ticker: str, qty: int, verdict: str) -> dict:
    """AI 매도 판단 결과에 따른 InlineKeyboard.

    verdict:
      SELL_OK → [매도 실행] [50% 매도] [홀딩]
      PARTIAL → [50% 매도] [전량 매도] [홀딩]
      HOLD    → [그래도 매도] [홀딩]
    """
    sell_cb = f"ai_sell:{ticker}:{qty}"
    partial_cb = f"ai_partial:{ticker}:{qty}"
    hold_cb = f"ai_hold:{ticker}"

    if verdict == "SELL_OK":
        return {"inline_keyboard": [[
            {"text": "🔴 매도 실행", "callback_data": sell_cb},
            {"text": "50% 매도", "callback_data": partial_cb},
            {"text": "홀딩", "callback_data": hold_cb},
        ]]}
    elif verdict == "PARTIAL":
        return {"inline_keyboard": [[
            {"text": "50% 매도", "callback_data": partial_cb},
            {"text": "전량 매도", "callback_data": sell_cb},
            {"text": "홀딩", "callback_data": hold_cb},
        ]]}
    else:  # HOLD or unknown
        return {"inline_keyboard": [[
            {"text": "⚠️ 그래도 매도", "callback_data": sell_cb},
            {"text": "✅ 홀딩", "callback_data": hold_cb},
        ]]}
