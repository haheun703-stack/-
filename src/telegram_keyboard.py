"""텔레그램 ReplyKeyboardMarkup & InlineKeyboardMarkup 빌더."""

from __future__ import annotations

# ── 메인 키보드 (하단 고정, 4행 12버튼) ──

MAIN_KEYBOARD = {
    "keyboard": [
        ["스캔", "리포트", "분석"],
        ["매수", "매도", "청산"],
        ["현재잔고", "포트폴리오", "체결내역"],
        ["시작", "정지", "도움"],
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
