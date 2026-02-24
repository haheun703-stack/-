"""
텔레그램 키보드 버튼 봇 — long-polling 기반.

하단 고정 ReplyKeyboardMarkup (4행 12버튼) + InlineKeyboard 확인/취소.
대화형 상태머신(FSM)으로 매수/매도/분석 파라미터 순차 입력 지원.
기존 /슬래시 명령어도 호환 유지.

DailyScheduler의 백그라운드 스레드로 실행됨.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 매수/매도 지정가 할인/프리미엄 비율
LIMIT_BUY_DISCOUNT = 0.005    # 현재가 -0.5%
LIMIT_SELL_PREMIUM = 0.005    # 현재가 +0.5%


def _tick_round(price: int, reference: int) -> int:
    """호가 단위 맞춤 (KRX 규칙)."""
    if reference < 2000:
        tick = 1
    elif reference < 5000:
        tick = 5
    elif reference < 20000:
        tick = 10
    elif reference < 50000:
        tick = 50
    elif reference < 200000:
        tick = 100
    elif reference < 500000:
        tick = 500
    else:
        tick = 1000
    return (price // tick) * tick


class TelegramCommandBot:
    """텔레그램 키보드 버튼 봇 (백그라운드 스레드)."""

    def __init__(self, scheduler=None):
        self._offset = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._scheduler = scheduler
        self._start_time = datetime.now()

        from src.telegram_conversation import ConversationManager
        self._conv = ConversationManager()

    # ══════════════════════════════════════════
    # 스레드 관리
    # ══════════════════════════════════════════

    def start(self) -> None:
        if not TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN 미설정 — 명령 봇 비활성")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="tg-cmd-bot", daemon=True,
        )
        self._thread.start()
        logger.info("텔레그램 명령 봇 시작 (키보드 버튼)")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("텔레그램 명령 봇 종료")

    def _poll_loop(self) -> None:
        self._flush_old_updates()
        while self._running:
            try:
                updates = self._get_updates(timeout=30)
                for update in updates:
                    self._handle_update(update)
            except requests.ConnectionError:
                logger.warning("[명령봇] 네트워크 연결 실패 — 10초 후 재시도")
                time.sleep(10)
            except Exception as e:
                logger.error("[명령봇] polling 오류: %s", e)
                time.sleep(5)

    def _flush_old_updates(self) -> None:
        try:
            resp = requests.get(
                f"{API_BASE}/getUpdates",
                params={"offset": -1, "timeout": 0},
                timeout=10,
            )
            data = resp.json()
            if data.get("ok") and data.get("result"):
                last_id = data["result"][-1]["update_id"]
                self._offset = last_id + 1
                logger.info("[명령봇] 기존 메시지 %d건 건너뜀", len(data["result"]))
        except Exception as e:
            logger.warning("[명령봇] flush 실패: %s", e)

    def _get_updates(self, timeout: int = 30) -> list[dict]:
        resp = requests.get(
            f"{API_BASE}/getUpdates",
            params={"offset": self._offset, "timeout": timeout},
            timeout=timeout + 10,
        )
        data = resp.json()
        if not data.get("ok"):
            return []
        results = data.get("result", [])
        if results:
            self._offset = results[-1]["update_id"] + 1
        return results

    # ══════════════════════════════════════════
    # 메시지 처리 (핵심 라우터)
    # ══════════════════════════════════════════

    def _handle_update(self, update: dict) -> None:
        # 1) InlineKeyboard callback
        callback = update.get("callback_query")
        if callback:
            self._handle_callback(callback)
            return

        # 2) 일반 메시지
        msg = update.get("message")
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != TELEGRAM_CHAT_ID:
            logger.warning("[명령봇] 미허가 사용자: chat_id=%s", chat_id)
            return

        text = msg.get("text", "").strip()
        if not text:
            return

        logger.info("[명령봇] 수신: %s", text)

        # 대화 타임아웃 체크
        if self._conv.check_timeout():
            self._reply_kb("\u23f0 입력 대기 시간 초과. 처음부터 다시 시도하세요.")

        # "취소" 입력 시 대화 리셋
        if text == "취소":
            self._conv.reset()
            self._reply_kb("\u274c 취소되었습니다.")
            return

        # /슬래시 명령어 (FSM보다 우선)
        if text.startswith("/"):
            self._conv.reset()  # 슬래시 명령 시 대화 초기화
            parts = text.split()
            cmd = parts[0].lower().split("@")[0]
            args = parts[1:]
            handler = SLASH_COMMANDS.get(cmd)
            if handler:
                try:
                    handler(self, args)
                except Exception as e:
                    logger.error("[명령봇] %s 오류: %s", cmd, e)
                    self._reply_kb(f"\u274c 명령 오류: {e}")
            else:
                self._reply_kb(f"\u2753 알 수 없는 명령: {cmd}\n하단 버튼을 사용하세요.")
            return

        # 키보드 버튼 텍스트 (FSM보다 우선 — 첫 단어로 매칭)
        parts = text.split()
        btn_cmd = parts[0]
        btn_args = parts[1:]
        handler = TEXT_COMMANDS.get(btn_cmd)
        if handler:
            self._conv.reset()  # 버튼 명령 시 대화 초기화
            try:
                handler(self, btn_args)
            except Exception as e:
                logger.error("[명령봇] %s 오류: %s", btn_cmd, e)
                self._reply_kb(f"\u274c 오류: {e}")
            return

        # 대화 진행 중이면 FSM으로 위임
        if not self._conv.is_idle():
            self._handle_conversation(text)
            return

        self._reply_kb(f"\u2753 알 수 없는 입력: {text}\n하단 버튼을 사용하세요.")

    # ══════════════════════════════════════════
    # 응답 헬퍼
    # ══════════════════════════════════════════

    def _reply(self, text: str, reply_markup: dict = None) -> None:
        from src.telegram_sender import send_message
        send_message(text, reply_markup=reply_markup)

    def _reply_kb(self, text: str) -> None:
        """메인 키보드와 함께 응답."""
        from src.telegram_keyboard import MAIN_KEYBOARD
        self._reply(text, reply_markup=MAIN_KEYBOARD)

    def _reply_inline(self, text: str, inline_markup: dict) -> None:
        """InlineKeyboard와 함께 응답."""
        self._reply(text, reply_markup=inline_markup)

    # ══════════════════════════════════════════
    # 대화형 FSM 처리
    # ══════════════════════════════════════════

    def _handle_conversation(self, text: str) -> None:
        from src.telegram_conversation import ConvState
        state = self._conv.state
        ctx = self._conv.context

        if state == ConvState.BUY_WAIT_STOCK:
            self._conv_buy_stock(text)
        elif state == ConvState.BUY_WAIT_QTY:
            self._conv_buy_qty(text)
        elif state == ConvState.SELL_WAIT_STOCK:
            self._conv_sell_stock(text)
        elif state == ConvState.SELL_WAIT_QTY:
            self._conv_sell_qty(text)
        elif state == ConvState.ANALYZE_WAIT_STOCK:
            self._conv_analyze_stock(text)
        else:
            self._conv.reset()
            self._reply_kb("\u274c 알 수 없는 상태. 다시 시도하세요.")

    def _conv_buy_stock(self, text: str) -> None:
        from src.telegram_conversation import ConvState
        matches = self._resolve_stock(text)
        if not matches:
            self._reply_kb(f"\u274c '{text}' 종목을 찾을 수 없습니다. 다시 입력하세요.")
            return
        if len(matches) > 1:
            from src.telegram_keyboard import build_stock_select_keyboard
            self._reply_inline(
                f"\U0001f50d '{text}' 검색 결과 — 종목을 선택하세요:",
                build_stock_select_keyboard(matches, "buy"),
            )
            return
        name, ticker = matches[0]
        self._conv.set_state(ConvState.BUY_WAIT_QTY, ticker=ticker, stock_name=name)
        self._reply_kb(f"\U0001f4b0 {name}({ticker})\n매수 수량을 입력하세요. (예: 10)")

    def _conv_buy_qty(self, text: str) -> None:
        try:
            qty = int(text.strip())
            if qty <= 0:
                raise ValueError
        except ValueError:
            self._reply_kb("\u274c 숫자를 입력하세요. (예: 10)")
            return

        ctx = self._conv.context
        ctx.quantity = qty
        from src.telegram_keyboard import build_confirm_keyboard
        self._reply_inline(
            f"\U0001f6d2 매수 확인\n"
            f"  종목: {ctx.stock_name} ({ctx.ticker})\n"
            f"  수량: {qty}주 (지정가 -0.5%)\n"
            f"\n실행하시겠습니까?",
            build_confirm_keyboard("buy", ctx.ticker, str(qty)),
        )

    def _conv_sell_stock(self, text: str) -> None:
        from src.telegram_conversation import ConvState
        matches = self._resolve_stock(text)
        if not matches:
            self._reply_kb(f"\u274c '{text}' 종목을 찾을 수 없습니다.")
            return
        if len(matches) > 1:
            from src.telegram_keyboard import build_stock_select_keyboard
            self._reply_inline(
                f"\U0001f50d '{text}' 검색 결과:",
                build_stock_select_keyboard(matches, "sell"),
            )
            return
        name, ticker = matches[0]
        # 보유 수량 확인
        held = self._get_held_shares(ticker)
        self._conv.set_state(ConvState.SELL_WAIT_QTY, ticker=ticker, stock_name=name)
        held_msg = f" (보유: {held}주)" if held else ""
        self._reply_kb(f"\U0001f4b0 {name}({ticker}){held_msg}\n매도 수량을 입력하세요.")

    def _conv_sell_qty(self, text: str) -> None:
        try:
            qty = int(text.strip())
            if qty <= 0:
                raise ValueError
        except ValueError:
            self._reply_kb("\u274c 숫자를 입력하세요.")
            return

        ctx = self._conv.context
        ctx.quantity = qty
        from src.telegram_keyboard import build_confirm_keyboard
        self._reply_inline(
            f"\U0001f4e4 매도 확인\n"
            f"  종목: {ctx.stock_name} ({ctx.ticker})\n"
            f"  수량: {qty}주 (지정가 +0.5%)\n"
            f"\n실행하시겠습니까?",
            build_confirm_keyboard("sell", ctx.ticker, str(qty)),
        )

    def _conv_analyze_stock(self, text: str) -> None:
        matches = self._resolve_stock(text)
        if not matches:
            self._reply_kb(f"\u274c '{text}' 종목을 찾을 수 없습니다.")
            return
        if len(matches) > 1:
            from src.telegram_keyboard import build_stock_select_keyboard
            self._reply_inline(
                f"\U0001f50d '{text}' 검색 결과:",
                build_stock_select_keyboard(matches, "analyze"),
            )
            return
        name, ticker = matches[0]
        self._conv.reset()
        self._do_analyze(name, ticker)

    # ══════════════════════════════════════════
    # InlineKeyboard callback 처리
    # ══════════════════════════════════════════

    def _handle_callback(self, callback: dict) -> None:
        from src.telegram_sender import answer_callback_query, edit_message_text

        query_id = callback["id"]
        data = callback.get("data", "")
        chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
        message_id = callback.get("message", {}).get("message_id")

        if chat_id != TELEGRAM_CHAT_ID:
            answer_callback_query(query_id, "미허가 사용자")
            return

        parts = data.split(":")
        action = parts[0]

        if action == "confirm" and len(parts) >= 3:
            cmd = parts[1]  # buy / sell / liquidate / start
            ticker = parts[2]
            detail = parts[3] if len(parts) > 3 else ""

            answer_callback_query(query_id, "실행 중...")
            self._conv.reset()

            if cmd == "buy":
                self._execute_buy(ticker, int(detail) if detail else 0, chat_id, message_id)
            elif cmd == "sell":
                self._execute_sell(ticker, int(detail) if detail else 0, chat_id, message_id)
            elif cmd == "liquidate":
                self._execute_liquidate(chat_id, message_id)
            elif cmd == "start":
                self._execute_auto_start(chat_id, message_id)

        elif action == "cancel":
            answer_callback_query(query_id, "취소됨")
            edit_message_text(chat_id, message_id, "\u274c 취소되었습니다.")
            self._conv.reset()

        elif action == "select" and len(parts) >= 3:
            cmd = parts[1]  # buy / sell / analyze
            ticker = parts[2]
            answer_callback_query(query_id)
            from src.stock_name_resolver import ticker_to_name
            from src.telegram_conversation import ConvState

            name = ticker_to_name(ticker)
            edit_message_text(chat_id, message_id, f"\u2705 선택: {name}({ticker})")

            if cmd == "buy":
                self._conv.set_state(ConvState.BUY_WAIT_QTY, ticker=ticker, stock_name=name)
                self._reply_kb(f"\U0001f4b0 {name}({ticker})\n매수 수량을 입력하세요.")
            elif cmd == "sell":
                held = self._get_held_shares(ticker)
                self._conv.set_state(ConvState.SELL_WAIT_QTY, ticker=ticker, stock_name=name)
                held_msg = f" (보유: {held}주)" if held else ""
                self._reply_kb(f"\U0001f4b0 {name}({ticker}){held_msg}\n매도 수량을 입력하세요.")
            elif cmd == "analyze":
                self._conv.reset()
                self._do_analyze(name, ticker)
        else:
            answer_callback_query(query_id, "알 수 없는 작업")

    # ══════════════════════════════════════════
    # 주문 실행 (확인 후)
    # ══════════════════════════════════════════

    def _execute_buy(self, ticker: str, qty: int, chat_id: str, msg_id: int) -> None:
        from src.telegram_sender import edit_message_text
        from src.stock_name_resolver import ticker_to_name
        name = ticker_to_name(ticker)
        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            adapter = KisOrderAdapter()
            # 현재가 조회 → -0.5% 지정가 매수
            price_info = adapter.fetch_current_price(ticker)
            current = price_info.get("current_price", 0)
            if current <= 0:
                edit_message_text(chat_id, msg_id, f"\u274c 매수 실패: 현재가 조회 불가 ({ticker})")
                return
            limit_price = _tick_round(int(current * (1 - LIMIT_BUY_DISCOUNT)), current)
            order = adapter.buy_limit(ticker, limit_price, qty)
            status = getattr(order, "status", "UNKNOWN")
            edit_message_text(
                chat_id, msg_id,
                f"\u2705 매수 주문 접수\n"
                f"  {name}({ticker}) {qty}주\n"
                f"  지정가: {limit_price:,}원 (현재가 {current:,}원 -0.5%)\n"
                f"  상태: {status}",
            )
        except Exception as e:
            edit_message_text(chat_id, msg_id, f"\u274c 매수 실패: {e}")

    def _execute_sell(self, ticker: str, qty: int, chat_id: str, msg_id: int) -> None:
        from src.telegram_sender import edit_message_text
        from src.stock_name_resolver import ticker_to_name
        name = ticker_to_name(ticker)
        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            adapter = KisOrderAdapter()
            # 현재가 조회 → +0.5% 지정가 매도
            price_info = adapter.fetch_current_price(ticker)
            current = price_info.get("current_price", 0)
            if current <= 0:
                edit_message_text(chat_id, msg_id, f"\u274c 매도 실패: 현재가 조회 불가 ({ticker})")
                return
            limit_price = _tick_round(int(current * (1 + LIMIT_SELL_PREMIUM)), current)
            order = adapter.sell_limit(ticker, limit_price, qty)
            status = getattr(order, "status", "UNKNOWN")
            edit_message_text(
                chat_id, msg_id,
                f"\u2705 매도 주문 접수\n"
                f"  {name}({ticker}) {qty}주\n"
                f"  지정가: {limit_price:,}원 (현재가 {current:,}원 +0.5%)\n"
                f"  상태: {status}",
            )
        except Exception as e:
            edit_message_text(chat_id, msg_id, f"\u274c 매도 실패: {e}")

    def _execute_liquidate(self, chat_id: str, msg_id: int) -> None:
        from src.telegram_sender import edit_message_text
        pos_path = PROJECT_ROOT / "data" / "positions.json"
        if not pos_path.exists():
            edit_message_text(chat_id, msg_id, "\u274c 보유 종목 없음")
            return
        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)
        positions = data.get("positions", [])
        if not positions:
            edit_message_text(chat_id, msg_id, "\u274c 보유 종목 없음")
            return

        results = []
        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            adapter = KisOrderAdapter()
            for p in positions:
                ticker = p.get("ticker", "")
                shares = p.get("shares", 0)
                name = p.get("name", ticker)
                if shares > 0:
                    try:
                        adapter.sell_market(ticker, shares)
                        results.append(f"\u2705 {name} {shares}주 매도")
                    except Exception as e:
                        results.append(f"\u274c {name} 실패: {e}")
        except Exception as e:
            edit_message_text(chat_id, msg_id, f"\u274c 청산 실패: {e}")
            return

        edit_message_text(chat_id, msg_id, "\U0001f6d1 전량 청산 결과\n" + "\n".join(results))

    def _execute_auto_start(self, chat_id: str, msg_id: int) -> None:
        from src.telegram_sender import edit_message_text
        stop_path = PROJECT_ROOT / "STOP.signal"
        if stop_path.exists():
            stop_path.unlink()
        if self._scheduler:
            self._scheduler.enabled = True
        edit_message_text(chat_id, msg_id, "\U0001f7e2 자동매매 시작됨 (STOP.signal 삭제)")
        logger.info("[명령봇] 자동매매 시작")

    # ══════════════════════════════════════════
    # 버튼 핸들러 — 분석 그룹
    # ══════════════════════════════════════════

    @staticmethod
    def _grade_stock(item: dict, money_type: str) -> tuple[str, str]:
        """종목 등급 판정. Returns (등급, 이모지)."""
        bb = item.get("bb_pct", 50)
        rsi = item.get("rsi", 50)
        adx = item.get("adx", 0)
        gx = item.get("stoch_golden_recent", False)
        if money_type == "SMART":
            if bb < 30 and rsi < 45:
                return "S", "\U0001f525"
            elif bb < 50 and rsi < 55:
                return "A", "\u2b50"
            else:
                return "B", "\U0001f539"
        else:
            if adx > 50:
                return "S", "\U0001f525"
            elif adx > 40 or gx:
                return "A", "\u2b50"
            else:
                return "B", "\U0001f539"

    def _cmd_scan(self, args: list) -> None:
        """스캔 — 매수 후보 조회 (섹터 로테이션 스캔 기반)."""
        scan_path = PROJECT_ROOT / "data" / "sector_rotation" / "krx_sector_scan.json"
        if not scan_path.exists():
            self._reply_kb("\U0001f50d 스캔 결과 없음 (krx_sector_scan.json)")
            return

        with open(scan_path, "r", encoding="utf-8") as f:
            scan = json.load(f)

        scan_date = scan.get("scan_date", "?")
        smart = scan.get("smart_money", [])
        theme = scan.get("theme_money", [])

        # Smart Money: BB% 낮은 순 (저평가 우선), RSI < 70
        good_smart = [s for s in smart if s.get("rsi", 100) < 70]
        good_smart.sort(key=lambda x: x.get("bb_pct", 100))

        # Theme Money: ADX > 35, RSI < 80
        good_theme = [t for t in theme if t.get("adx", 0) > 35 and t.get("rsi", 100) < 80]
        good_theme.sort(key=lambda x: -x.get("adx", 0))

        lines = [f"\U0001f50d 매수 후보 ({scan_date})", "\u2501" * 24, ""]

        lines.append("\U0001f48e Smart Money (FULL)")
        lines.append("\u2500" * 24)
        if good_smart:
            for s in good_smart[:5]:
                g, ge = self._grade_stock(s, "SMART")
                name = s.get("name", "?")
                ticker = s.get("ticker", "?")
                sector = s.get("etf_sector", "?")
                rsi = s.get("rsi", 0)
                bb = s.get("bb_pct", 0)
                lines.append(f"  {ge} {g}급 {name} ({ticker}) [{sector}]")
                lines.append(f"    RSI {rsi:.0f} | BB {bb:.0f}%")
                lines.append("")
        else:
            lines.append("  해당 없음")
            lines.append("")

        lines.append("\U0001f525 Theme Money (HALF)")
        lines.append("\u2500" * 24)
        if good_theme:
            for t in good_theme[:5]:
                g, ge = self._grade_stock(t, "THEME")
                name = t.get("name", "?")
                ticker = t.get("ticker", "?")
                adx = t.get("adx", 0)
                rsi = t.get("rsi", 0)
                gx = " \u2605GX" if t.get("stoch_golden_recent") else ""
                lines.append(f"  {ge} {g}급 {name} ({ticker}){gx}")
                lines.append(f"    ADX {adx:.0f} | RSI {rsi:.0f}")
                lines.append("")
        else:
            lines.append("  해당 없음")
            lines.append("")

        summary = scan.get("summary", {})
        lines.append(
            f"\U0001f4cb Smart {len(smart)} | Theme {len(theme)} | "
            f"\uc9c4\uc785OK {summary.get('entry_ok', '?')}"
        )
        self._reply_kb("\n".join(lines))

    def _cmd_report(self, args: list) -> None:
        """리포트 — 장전 브리핑."""
        self._reply_kb("\u23f3 브리핑 생성 중...")
        try:
            from scripts.send_market_briefing import build_briefing_message
            msg = build_briefing_message()
            self._reply_kb(msg)
        except Exception as e:
            self._reply_kb(f"\u274c 브리핑 실패: {e}")

    def _cmd_analyze(self, args: list) -> None:
        """분석 — 개별 종목 기술 분석."""
        from src.telegram_conversation import ConvState
        if args:
            # 한 줄 입력: "분석 삼성전자"
            query = " ".join(args)
            matches = self._resolve_stock(query)
            if not matches:
                self._reply_kb(f"\u274c '{query}' 종목을 찾을 수 없습니다.")
                return
            if len(matches) > 1:
                from src.telegram_keyboard import build_stock_select_keyboard
                self._reply_inline(f"\U0001f50d 종목 선택:", build_stock_select_keyboard(matches, "analyze"))
                return
            name, ticker = matches[0]
            self._do_analyze(name, ticker)
        else:
            self._conv.set_state(ConvState.ANALYZE_WAIT_STOCK)
            self._reply_kb("\U0001f4ca 분석할 종목명을 입력하세요. (예: 삼성전자)")

    def _do_analyze(self, name: str, ticker: str) -> None:
        """종목 기술적 분석 실행."""
        lines = [f"\U0001f4ca {name} ({ticker}) 기술적 분석", "\u2500" * 24]
        # parquet에서 최신 지표 조회
        parquet_path = PROJECT_ROOT / "data" / "processed" / f"{ticker}.parquet"
        if parquet_path.exists():
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                last = df.iloc[-1]
                close = last.get("close", 0)
                rsi = last.get("rsi_14", 0)
                macd = last.get("macd", 0)
                macd_sig = last.get("macd_signal", 0)
                adx = last.get("adx", 0)
                ma20 = last.get("ma_20", 0)
                ma60 = last.get("ma_60", 0)
                vol = last.get("volume", 0)
                date = df.index[-1].strftime("%m/%d")

                lines.append(f"  날짜: {date}")
                lines.append(f"  종가: {close:,.0f}원")
                lines.append(f"  RSI(14): {rsi:.1f}")
                macd_status = "양전환" if macd > macd_sig else "음전환"
                lines.append(f"  MACD: {macd:.1f} ({macd_status})")
                lines.append(f"  ADX: {adx:.1f}")
                ma_status = "MA20 위" if close > ma20 else "MA20 아래"
                lines.append(f"  MA20: {ma20:,.0f} ({ma_status})")
                lines.append(f"  MA60: {ma60:,.0f}")
                lines.append(f"  거래량: {vol:,.0f}")
        else:
            lines.append("  parquet 데이터 없음 (유니버스 외)")
        self._reply_kb("\n".join(lines))

    # ══════════════════════════════════════════
    # 버튼 핸들러 — 매매 그룹
    # ══════════════════════════════════════════

    def _cmd_buy(self, args: list) -> None:
        """매수 — 수동 매수 주문."""
        from src.telegram_conversation import ConvState
        if len(args) >= 2:
            # 한 줄: "매수 삼성전자 10"
            query = args[0]
            try:
                qty = int(args[1])
            except ValueError:
                self._reply_kb("\u274c 수량은 숫자로 입력하세요. 예: 매수 삼성전자 10")
                return
            matches = self._resolve_stock(query)
            if not matches:
                self._reply_kb(f"\u274c '{query}' 종목을 찾을 수 없습니다.")
                return
            if len(matches) > 1:
                from src.telegram_keyboard import build_stock_select_keyboard
                self._reply_inline(f"\U0001f50d 종목 선택:", build_stock_select_keyboard(matches, "buy"))
                return
            name, ticker = matches[0]
            from src.telegram_keyboard import build_confirm_keyboard
            self._conv.set_state(ConvState.BUY_WAIT_QTY, ticker=ticker, stock_name=name, quantity=qty)
            self._reply_inline(
                f"\U0001f6d2 매수 확인\n  {name}({ticker}) {qty}주 (지정가 -0.5%)\n\n실행하시겠습니까?",
                build_confirm_keyboard("buy", ticker, str(qty)),
            )
        elif len(args) == 1:
            query = args[0]
            matches = self._resolve_stock(query)
            if not matches:
                self._reply_kb(f"\u274c '{query}' 종목을 찾을 수 없습니다.")
                return
            if len(matches) > 1:
                from src.telegram_keyboard import build_stock_select_keyboard
                self._reply_inline(f"\U0001f50d 종목 선택:", build_stock_select_keyboard(matches, "buy"))
                return
            name, ticker = matches[0]
            self._conv.set_state(ConvState.BUY_WAIT_QTY, ticker=ticker, stock_name=name)
            self._reply_kb(f"\U0001f4b0 {name}({ticker})\n매수 수량을 입력하세요.")
        else:
            self._conv.set_state(ConvState.BUY_WAIT_STOCK, action="buy")
            self._reply_kb("\U0001f4b0 매수할 종목명을 입력하세요. (예: 삼성전자)\n취소: '취소' 입력")

    def _cmd_sell(self, args: list) -> None:
        """매도 — 수동 매도 주문."""
        from src.telegram_conversation import ConvState
        if args:
            query = args[0]
            matches = self._resolve_stock(query)
            if not matches:
                self._reply_kb(f"\u274c '{query}' 종목을 찾을 수 없습니다.")
                return
            if len(matches) > 1:
                from src.telegram_keyboard import build_stock_select_keyboard
                self._reply_inline(f"\U0001f50d 종목 선택:", build_stock_select_keyboard(matches, "sell"))
                return
            name, ticker = matches[0]
            held = self._get_held_shares(ticker)
            if len(args) >= 2:
                try:
                    qty = int(args[1])
                except ValueError:
                    qty = held or 0
            else:
                qty = held or 0
            if qty > 0:
                from src.telegram_keyboard import build_confirm_keyboard
                self._conv.set_state(ConvState.SELL_WAIT_QTY, ticker=ticker, stock_name=name, quantity=qty)
                self._reply_inline(
                    f"\U0001f4e4 매도 확인\n  {name}({ticker}) {qty}주 (지정가 +0.5%)\n\n실행하시겠습니까?",
                    build_confirm_keyboard("sell", ticker, str(qty)),
                )
            else:
                self._conv.set_state(ConvState.SELL_WAIT_QTY, ticker=ticker, stock_name=name)
                self._reply_kb(f"\U0001f4b0 {name}({ticker})\n매도 수량을 입력하세요.")
        else:
            # 보유 종목 보여주기
            holdings_msg = self._build_holdings_summary()
            self._conv.set_state(ConvState.SELL_WAIT_STOCK, action="sell")
            self._reply_kb(f"\U0001f4e4 매도할 종목명을 입력하세요.\n{holdings_msg}\n취소: '취소' 입력")

    def _cmd_liquidate(self, args: list) -> None:
        """청산 — 전 포지션 매도."""
        pos_path = PROJECT_ROOT / "data" / "positions.json"
        if not pos_path.exists():
            self._reply_kb("\u274c 보유 종목 없음")
            return
        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)
        positions = data.get("positions", [])
        if not positions:
            self._reply_kb("\u274c 보유 종목 없음")
            return
        lines = ["\U0001f6d1 전량 청산 확인", "\u2500" * 24]
        for p in positions:
            name = p.get("name", p.get("ticker", "?"))
            shares = p.get("shares", 0)
            pnl = p.get("unrealized_pnl_pct", 0)
            lines.append(f"  {name} {shares}주 ({pnl:+.1f}%)")
        lines.append(f"\n총 {len(positions)}종목 전량 시장가 매도")
        from src.telegram_keyboard import build_confirm_keyboard
        self._reply_inline("\n".join(lines), build_confirm_keyboard("liquidate", "all"))

    # ══════════════════════════════════════════
    # 버튼 핸들러 — 조회 그룹
    # ══════════════════════════════════════════

    def _cmd_balance(self, args: list) -> None:
        """현재잔고 — KIS API 실잔고."""
        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            adapter = KisOrderAdapter()
            bal = adapter.fetch_balance()
            holdings = bal.get("holdings", [])
            cash = bal.get("available_cash", 0)
            total_eval = bal.get("total_eval", 0)
            total_pnl = bal.get("total_pnl", 0)

            lines = ["\U0001f4b0 KIS 실잔고", "\u2500" * 24]
            if holdings:
                for h in holdings:
                    name = h.get("name", h.get("ticker", "?"))
                    qty = h.get("quantity", 0)
                    eval_amt = h.get("eval_amount", 0)
                    pnl_pct = h.get("pnl_pct", 0)
                    emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
                    lines.append(f"{emoji} {name} {qty}주 {eval_amt:,.0f}원 ({pnl_pct:+.1f}%)")
            else:
                lines.append("  보유 종목 없음")
            lines.append("")
            lines.append(f"  예수금: {cash:,.0f}원")
            lines.append(f"  평가액: {total_eval:,.0f}원")
            lines.append(f"  손익: {total_pnl:+,.0f}원")
            self._reply_kb("\n".join(lines))
        except Exception as e:
            self._reply_kb(f"\u274c 잔고 조회 실패: {e}")

    def _cmd_portfolio(self, args: list) -> None:
        """포트폴리오 — 포지션 상세 (손익/손절/목표가)."""
        pos_path = PROJECT_ROOT / "data" / "positions.json"
        if not pos_path.exists():
            self._reply_kb("\U0001f4bc 포지션 없음")
            return
        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)
        positions = data.get("positions", [])
        if not positions:
            self._reply_kb("\U0001f4bc 포지션 없음")
            return
        capital = data.get("capital", 0)
        lines = ["\U0001f4bc 포트폴리오 상세", "\u2500" * 24]
        total_invested = 0
        for p in positions:
            name = p.get("name", p.get("ticker", "?"))
            ticker = p.get("ticker", "?")
            entry = p.get("entry_price", 0)
            current = p.get("current_price", entry)
            shares = p.get("shares", 0)
            pnl_pct = ((current - entry) / entry * 100) if entry else 0
            stop = p.get("stop_loss", 0)
            target = p.get("target_price", 0)
            grade = p.get("grade", "?")
            days = p.get("max_hold_days", 0)
            invested = entry * shares
            total_invested += invested
            emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
            lines.append(f"{emoji} {name}({ticker}) [{grade}]")
            lines.append(f"  진입 {entry:,.0f} | 현재 {current:,.0f} | {pnl_pct:+.1f}%")
            lines.append(f"  손절 {stop:,.0f} | 목표 {target:,.0f} | {shares}주")
            lines.append(f"  투자금 {invested:,.0f}원 | 잔여 {days}일")
            lines.append("")
        lines.append(f"\U0001f4ca 총 투자: {total_invested:,.0f} / 자본: {capital:,.0f}")
        self._reply_kb("\n".join(lines))

    def _cmd_trade_history(self, args: list) -> None:
        """체결내역 — 당일 체결 기록."""
        hist_path = PROJECT_ROOT / "data" / "trades_history.json"
        if not hist_path.exists():
            self._reply_kb("\U0001f4c4 체결 기록 없음")
            return
        with open(hist_path, encoding="utf-8") as f:
            trades = json.load(f)
        if not isinstance(trades, list):
            trades = trades.get("trades", [])
        today = datetime.now().strftime("%Y-%m-%d")
        today_trades = [t for t in trades if t.get("date", "").startswith(today)]
        if not today_trades:
            self._reply_kb(f"\U0001f4c4 오늘({today}) 체결 내역 없음")
            return
        lines = [f"\U0001f4c4 체결내역 ({today})", "\u2500" * 24]
        for t in today_trades[-10:]:
            side = t.get("side", "?")
            name = t.get("name", t.get("ticker", "?"))
            qty = t.get("quantity", 0)
            price = t.get("price", 0)
            emoji = "\U0001f7e2" if side == "BUY" else "\U0001f534"
            lines.append(f"{emoji} {side} {name} {qty}주 @{price:,.0f}")
        self._reply_kb("\n".join(lines))

    # ══════════════════════════════════════════
    # 버튼 핸들러 — 제어 그룹
    # ══════════════════════════════════════════

    def _cmd_auto_start(self, args: list) -> None:
        """시작 — 자동매매 ON."""
        from src.telegram_keyboard import build_confirm_keyboard
        self._reply_inline(
            "\U0001f7e2 자동매매를 시작하시겠습니까?\n"
            "STOP.signal이 삭제되고 매매가 활성화됩니다.",
            build_confirm_keyboard("start", "auto"),
        )

    def _cmd_stop(self, args: list) -> None:
        """정지 — 자동매매 OFF (즉시 실행)."""
        stop_path = PROJECT_ROOT / "STOP.signal"
        stop_path.write_text(
            f"STOPPED by telegram command at {datetime.now().isoformat()}",
            encoding="utf-8",
        )
        self._reply_kb("\U0001f6d1 자동매매 정지됨\nSTOP.signal 생성 완료")
        logger.warning("[명령봇] 자동매매 정지")

    def _cmd_help(self, args: list) -> None:
        """도움 — 명령어 목록."""
        lines = [
            "\U0001f4cb Quantum Master",
            "\u2501" * 24,
            "",
            "\U0001f50d [분석]",
            "  스캔 — 매수 후보 조회",
            "  리포트 — 장전 브리핑",
            "  분석 — 개별 종목 분석",
            "    (예: 분석 삼성전자)",
            "",
            "\U0001f4b0 [매매]",
            "  매수 — 지정가 매수 (-0.5%)",
            "    (예: 매수 삼성전자 10)",
            "  매도 — 지정가 매도 (+0.5%)",
            "  청산 — 전 포지션 시장가 매도",
            "",
            "\U0001f4bc [조회]",
            "  현재잔고 — KIS 실잔고",
            "  포트폴리오 — 포지션 상세",
            "  체결내역 — 당일 체결",
            "",
            "\u2699\ufe0f [제어]",
            "  시작 — 자동매매 ON",
            "  정지 — 자동매매 OFF",
        ]
        self._reply_kb("\n".join(lines))

    # ══════════════════════════════════════════
    # 기존 /슬래시 전용 핸들러
    # ══════════════════════════════════════════

    def _cmd_ping(self, args: list) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        uptime = datetime.now() - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        mins = int((uptime.total_seconds() % 3600) // 60)
        self._reply_kb(f"\U0001f3d3 pong! | {now}\n\u23f1 uptime: {hours}시간 {mins}분")

    def _cmd_status(self, args: list) -> None:
        now = datetime.now()
        lines = [f"\U0001f4e1 스케줄러 상태 | {now.strftime('%H:%M:%S')}", "\u2500" * 24]
        if self._scheduler:
            mode = self._scheduler.mode
            enabled = self._scheduler.enabled
            lines.append(f"  모드: {mode} | 실주문: {'ON' if enabled else 'OFF'}")
            lines.append(f"  공휴일: {'예' if self._scheduler._is_holiday else '아니오'}")
            lines.append(f"  매수후보: {len(self._scheduler._buy_signals)}종목")
        else:
            lines.append("  스케줄러 미연결")
        stop_file = PROJECT_ROOT / "STOP.signal"
        lines.append("  \U0001f6d1 STOP.signal 활성!" if stop_file.exists() else "  \U0001f7e2 정상 운영 중")
        # MDD 모니터 요약
        try:
            from src.mdd_monitor import MDDMonitor
            mdd = MDDMonitor()
            lines.append(f"  {mdd.format_status_line()}")
        except Exception:
            pass
        if self._scheduler:
            lines.append("")
            lines.append("\u23f0 다음 Phase")
            schedule = self._scheduler.schedule
            current_time = now.strftime("%H:%M")
            next_phases = [(t, n) for n, t in sorted(schedule.items(), key=lambda x: x[1]) if t > current_time]
            for t, n in next_phases[:3]:
                lines.append(f"  {t} — {n}")
            if not next_phases:
                lines.append("  오늘 남은 Phase 없음")
        self._reply_kb("\n".join(lines))

    def _cmd_schedule(self, args: list) -> None:
        if not self._scheduler:
            self._reply_kb("\u274c 스케줄러 미연결")
            return
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        schedule = self._scheduler.schedule
        lines = [f"\U0001f4c5 오늘 스케줄 | {now.strftime('%m/%d %H:%M')}", "\u2500" * 24]
        for name, time_str in sorted(schedule.items(), key=lambda x: x[1]):
            marker = "\u25b6" if time_str > current_time else "\u2705"
            lines.append(f"  {marker} {time_str} — {name}")
        self._reply_kb("\n".join(lines))

    def _cmd_supply(self, args: list) -> None:
        if not self._scheduler:
            self._reply_kb("\u274c 스케줄러 미연결")
            return
        self._reply_kb("\u23f3 수급 스냅샷 수집 중...")
        try:
            snap_num = len(self._scheduler._supply_snapshots) + 1
            self._scheduler.phase_supply_snapshot(snap_num)
            self._reply_kb(f"\u2705 수급 스냅샷 {snap_num}차 완료!")
        except Exception as e:
            self._reply_kb(f"\u274c 수급 수집 실패: {e}")

    def _cmd_phase(self, args: list) -> None:
        if not args:
            self._reply_kb(
                "\u2753 사용법: /실행 <번호>\n"
                "예: /실행 3b, /실행 10b\n"
                "가능: 0~11, 3b, snap1~4, 8-2~8-5, 10b"
            )
            return
        if not self._scheduler:
            self._reply_kb("\u274c 스케줄러 미연결")
            return
        phase_key = args[0].lower()
        phases = {
            "0": ("일일 리셋", self._scheduler.phase_daily_reset),
            "1": ("US 데이터 수집", self._scheduler.phase_us_close_collect),
            "2": ("매크로 수집", self._scheduler.phase_macro_collect),
            "3a": ("테마 스캔", self._scheduler.phase_theme_scan),
            "3": ("뉴스 스캔", self._scheduler.phase_news_briefing),
            "3b": ("장전 브리핑", self._scheduler.phase_morning_briefing),
            "4": ("매매 준비", self._scheduler.phase_trade_prep),
            "5": ("매수 실행", self._scheduler.phase_buy_execution),
            "7": ("매도 실행", self._scheduler.phase_sell_execution),
            "8": ("종가 수집", self._scheduler.phase_close_data_collect),
            "8-2": ("CSV 업데이트", self._scheduler.phase_csv_update),
            "8-3": ("parquet 증분", self._scheduler.phase_parquet_update),
            "8-4": ("지표 재계산", self._scheduler.phase_indicator_calc),
            "8-5": ("데이터 검증", self._scheduler.phase_data_verify),
            "9": ("수급 확정", self._scheduler.phase_supply_final),
            "10": ("매수 후보 스캔", self._scheduler.phase_evening_scan),
            "10b": ("장마감 리포트", self._scheduler.phase_evening_briefing),
            "11": ("업무일지", self._scheduler.phase_eod_journal),
        }
        for i in range(1, 5):
            phases[f"snap{i}"] = (f"수급 스냅샷 {i}차", lambda n=i: self._scheduler.phase_supply_snapshot(n))
        if phase_key not in phases:
            self._reply_kb(f"\u274c 없는 Phase: {phase_key}")
            return
        label, func = phases[phase_key]
        self._reply_kb(f"\u23f3 Phase {phase_key}: {label}")
        t = threading.Thread(target=self._run_phase, args=(phase_key, label, func))
        t.start()

    def _run_phase(self, key: str, label: str, func) -> None:
        try:
            func()
            self._reply_kb(f"\u2705 Phase {key} 완료: {label}")
        except Exception as e:
            self._reply_kb(f"\u274c Phase {key} 오류: {e}")

    # ══════════════════════════════════════════
    # 유틸
    # ══════════════════════════════════════════

    def _resolve_stock(self, query: str) -> list[tuple[str, str]]:
        from src.stock_name_resolver import resolve_name
        return resolve_name(query)

    def _get_held_shares(self, ticker: str) -> int:
        pos_path = PROJECT_ROOT / "data" / "positions.json"
        if not pos_path.exists():
            return 0
        try:
            with open(pos_path, encoding="utf-8") as f:
                data = json.load(f)
            for p in data.get("positions", []):
                if p.get("ticker") == ticker:
                    return p.get("shares", 0)
        except Exception:
            pass
        return 0

    def _build_holdings_summary(self) -> str:
        pos_path = PROJECT_ROOT / "data" / "positions.json"
        if not pos_path.exists():
            return "  (보유 종목 없음)"
        try:
            with open(pos_path, encoding="utf-8") as f:
                data = json.load(f)
            positions = data.get("positions", [])
            if not positions:
                return "  (보유 종목 없음)"
            items = []
            for p in positions:
                name = p.get("name", p.get("ticker", "?"))
                shares = p.get("shares", 0)
                items.append(f"{name}({shares}주)")
            return "보유: " + ", ".join(items)
        except Exception:
            return "  (보유 조회 실패)"

    def _get_stock_name(self, ticker: str) -> str:
        from src.stock_name_resolver import ticker_to_name
        return ticker_to_name(ticker)


# ══════════════════════════════════════════
# 라우팅 테이블
# ══════════════════════════════════════════

# 키보드 버튼 텍스트 → 핸들러
TEXT_COMMANDS = {
    "스캔": TelegramCommandBot._cmd_scan,
    "리포트": TelegramCommandBot._cmd_report,
    "분석": TelegramCommandBot._cmd_analyze,
    "매수": TelegramCommandBot._cmd_buy,
    "매도": TelegramCommandBot._cmd_sell,
    "청산": TelegramCommandBot._cmd_liquidate,
    "현재잔고": TelegramCommandBot._cmd_balance,
    "포트폴리오": TelegramCommandBot._cmd_portfolio,
    "체결내역": TelegramCommandBot._cmd_trade_history,
    "시작": TelegramCommandBot._cmd_auto_start,
    "정지": TelegramCommandBot._cmd_stop,
    "도움": TelegramCommandBot._cmd_help,
}

# 기존 /슬래시 명령어 (호환 유지)
SLASH_COMMANDS = {
    "/도움": TelegramCommandBot._cmd_help,
    "/연결": TelegramCommandBot._cmd_ping,
    "/명령어": TelegramCommandBot._cmd_help,
    "/상태": TelegramCommandBot._cmd_status,
    "/잔고": TelegramCommandBot._cmd_portfolio,
    "/후보": TelegramCommandBot._cmd_scan,
    "/스케줄": TelegramCommandBot._cmd_schedule,
    "/브리핑": TelegramCommandBot._cmd_report,
    "/수급": TelegramCommandBot._cmd_supply,
    "/실행": TelegramCommandBot._cmd_phase,
    "/정지": TelegramCommandBot._cmd_stop,
    "/재개": TelegramCommandBot._cmd_auto_start,
}


# ══════════════════════════════════════════
# 독립 실행 (테스트용)
# ══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 50)
    print("  텔레그램 키보드 버튼 봇 (독립 실행)")
    print("  텔레그램에서 키보드 버튼을 눌러보세요")
    print("  Ctrl+C 로 종료")
    print("=" * 50)

    bot = TelegramCommandBot()
    bot._start_time = datetime.now()
    bot.start()

    # 키보드 초기 전송
    from src.telegram_keyboard import MAIN_KEYBOARD
    from src.telegram_sender import send_message
    send_message("\U0001f4cb Quantum Master 키보드 봇 시작!", reply_markup=MAIN_KEYBOARD)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()
        print("\n봇 종료됨")
