"""텔레그램 InlineKeyboard 기반 매매 승인 게이트웨이.

자동매매(SmartEntry / sell_monitor) 시 사용자 승인을 받은 후 주문 실행.
파일 기반 승인 큐로 프로세스 간 통신:
  1. SmartEntry/sell_monitor → 승인 요청 JSON + 텔레그램 메시지
  2. TelegramCommandBot → callback 수신 → JSON 상태 업데이트
  3. SmartEntry/sell_monitor → JSON 폴링 → approved/rejected

TelegramCommandBot 미실행 시 직접 getUpdates fallback.
"""

from __future__ import annotations

import json
import logging
import os
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
APPROVAL_FILE = PROJECT_ROOT / "data" / "trade_approvals.json"

# 승인 상태
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_TIMEOUT = "timeout"


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


class TradeApprovalGateway:
    """텔레그램 InlineKeyboard 기반 매매 승인 게이트웨이."""

    def __init__(self, timeout_sec: int = 300):
        self.timeout = timeout_sec
        self._offset = 0

    # ──────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────

    def request_buy_approval(
        self,
        ticker: str,
        name: str,
        price: int,
        qty: int,
        reasoning: str = "",
        timeout_sec: int | None = None,
    ) -> bool:
        """매수 승인 요청. True=승인, False=거부/타임아웃."""
        timeout = timeout_sec or self.timeout
        approval_id = f"buy_{ticker}_{int(time.time())}"
        invest_amount = price * qty

        text = (
            f"\U0001f514 [\uc790\ub3d9\ub9e4\uc218 \uc2b9\uc778 \uc694\uccad]\n"
            f"\u2501" * 20 + "\n"
            f"  \uc885\ubaa9: {name} ({ticker})\n"
            f"  \uc9c0\uc815\uac00: {price:,}\uc6d0\n"
            f"  \uc218\ub7c9: {qty}\uc8fc\n"
            f"  \ud22c\uc790\uae08: {invest_amount:,}\uc6d0\n"
        )
        if reasoning:
            text += f"\n  \uc0ac\uc720: {reasoning[:100]}\n"
        text += f"\u2501" * 20 + f"\n  \u23f3 {timeout}\ucd08 \ub0b4 \uc751\ub2f5 \ud544\uc694"

        trade_info = {
            "type": "buy",
            "ticker": ticker,
            "name": name,
            "price": price,
            "qty": qty,
            "reasoning": reasoning,
        }

        return self._request_approval(approval_id, text, trade_info, timeout)

    def request_sell_approval(
        self,
        ticker: str,
        name: str,
        qty: int,
        pnl_pct: float = 0.0,
        action: str = "SELL_NOW",
        sell_price: int = 0,
        reasoning: str = "",
        timeout_sec: int | None = None,
    ) -> bool:
        """매도 승인 요청. True=승인, False=거부/타임아웃."""
        timeout = timeout_sec or self.timeout
        approval_id = f"sell_{ticker}_{int(time.time())}"

        action_label = {
            "SELL_NOW": "\uc804\ub7c9 \ub9e4\ub3c4",
            "PARTIAL_SELL": "50% \ubd80\ubd84\ub9e4\ub3c4",
        }.get(action, action)

        text = (
            f"\U0001f514 [\uc790\ub3d9\ub9e4\ub3c4 \uc2b9\uc778 \uc694\uccad]\n"
            f"\u2501" * 20 + "\n"
            f"  \uc885\ubaa9: {name} ({ticker})\n"
            f"  \uc218\ub7c9: {qty}\uc8fc ({action_label})\n"
            f"  \ud604\uc7ac \uc190\uc775: {pnl_pct:+.1f}%\n"
        )
        if sell_price > 0:
            text += f"  \ub9e4\ub3c4\uac00: {sell_price:,}\uc6d0 (+0.5%)\n"
        if reasoning:
            text += f"\n  AI \ud310\ub2e8: {reasoning[:100]}\n"
        text += f"\u2501" * 20 + f"\n  \u23f3 {timeout}\ucd08 \ub0b4 \uc751\ub2f5 \ud544\uc694"

        trade_info = {
            "type": "sell",
            "ticker": ticker,
            "name": name,
            "qty": qty,
            "pnl_pct": pnl_pct,
            "action": action,
            "sell_price": sell_price,
            "reasoning": reasoning,
        }

        return self._request_approval(approval_id, text, trade_info, timeout)

    def update_approval(self, approval_id: str, status: str) -> None:
        """승인 상태 업데이트 (TelegramCommandBot에서 호출)."""
        approvals = self._load_approvals()
        if approval_id in approvals:
            approvals[approval_id]["status"] = status
            approvals[approval_id]["responded_at"] = datetime.now().isoformat()
            self._save_approvals(approvals)
            logger.info("[승인] %s → %s", approval_id, status)

    # ──────────────────────────────────────────
    # 내부 구현
    # ──────────────────────────────────────────

    def _request_approval(
        self, approval_id: str, text: str, trade_info: dict, timeout: int
    ) -> bool:
        """승인 요청 공통 로직."""
        # 1) 승인 파일에 pending 기록
        self._write_pending(approval_id, trade_info)

        # 2) 텔레그램 메시지 + InlineKeyboard 전송
        self._send_approval_message(text, approval_id)

        # 3) 승인 대기 (파일 폴링 → fallback: 직접 getUpdates)
        result = self._poll_approval(approval_id, timeout)

        # 4) 타임아웃 처리
        if result is None:
            self.update_approval(approval_id, STATUS_TIMEOUT)
            logger.warning("[승인] %s 타임아웃 (%d초)", approval_id, timeout)
            return False

        return result

    def _write_pending(self, approval_id: str, trade_info: dict) -> None:
        """승인 대기 상태를 JSON 파일에 기록."""
        approvals = self._load_approvals()
        approvals[approval_id] = {
            "status": STATUS_PENDING,
            "trade_info": trade_info,
            "requested_at": datetime.now().isoformat(),
            "responded_at": None,
        }
        self._save_approvals(approvals)

    def _poll_approval(self, approval_id: str, timeout: int) -> bool | None:
        """JSON 파일 폴링으로 승인/거부 상태 확인.

        Returns:
            True=승인, False=거부, None=타임아웃
        """
        deadline = time.time() + timeout
        poll_interval = 2  # 2초 간격

        while time.time() < deadline:
            approvals = self._load_approvals()
            entry = approvals.get(approval_id)
            if entry:
                status = entry.get("status", STATUS_PENDING)
                if status == STATUS_APPROVED:
                    logger.info("[승인] %s 승인됨!", approval_id)
                    return True
                elif status == STATUS_REJECTED:
                    logger.info("[승인] %s 거부됨", approval_id)
                    return False
            time.sleep(poll_interval)

        # 파일 기반 폴링 실패 → 직접 getUpdates fallback
        remaining = max(0, deadline - time.time())
        if remaining > 0:
            logger.info("[승인] 파일 폴링 실패 → getUpdates fallback (%d초)", remaining)
            return self._fallback_poll_updates(approval_id, int(remaining))

        return None  # 타임아웃

    def _fallback_poll_updates(self, approval_id: str, timeout: int) -> bool | None:
        """TelegramCommandBot 미실행 시 직접 getUpdates 폴링."""
        if not TELEGRAM_BOT_TOKEN:
            return None

        # 기존 업데이트 건너뜀
        try:
            resp = requests.get(
                f"{API_BASE}/getUpdates",
                params={"offset": -1, "timeout": 0},
                timeout=10,
            )
            data = resp.json()
            if data.get("ok") and data.get("result"):
                self._offset = data["result"][-1]["update_id"] + 1
        except Exception:
            pass

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(1, int(deadline - time.time()))
            poll_timeout = min(remaining, 10)

            try:
                resp = requests.get(
                    f"{API_BASE}/getUpdates",
                    params={"offset": self._offset, "timeout": poll_timeout},
                    timeout=poll_timeout + 10,
                )
                data = resp.json()
                if not data.get("ok"):
                    continue

                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1

                    callback = update.get("callback_query")
                    if not callback:
                        continue

                    cb_data = callback.get("data", "")
                    query_id = callback["id"]
                    chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
                    msg_id = callback.get("message", {}).get("message_id")

                    if chat_id != TELEGRAM_CHAT_ID:
                        continue

                    # 승인/거부 매칭
                    if cb_data == f"auto_approve:{approval_id}":
                        self._answer_callback(query_id, "\u2705 \uc2b9\uc778\ub428")
                        self._edit_message(chat_id, msg_id, f"\u2705 \uc2b9\uc778\ub428 ({approval_id})")
                        self.update_approval(approval_id, STATUS_APPROVED)
                        return True
                    elif cb_data == f"auto_reject:{approval_id}":
                        self._answer_callback(query_id, "\u274c \uac70\ubd80\ub428")
                        self._edit_message(chat_id, msg_id, f"\u274c \uac70\ubd80\ub428 ({approval_id})")
                        self.update_approval(approval_id, STATUS_REJECTED)
                        return False

            except requests.ConnectionError:
                time.sleep(5)
            except Exception as e:
                logger.warning("[승인] getUpdates 오류: %s", e)
                time.sleep(3)

        return None

    def _send_approval_message(self, text: str, approval_id: str) -> None:
        """InlineKeyboard 포함 텔레그램 메시지 전송."""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "\u2705 \uc2b9\uc778", "callback_data": f"auto_approve:{approval_id}"},
                    {"text": "\u274c \uac70\ubd80", "callback_data": f"auto_reject:{approval_id}"},
                ]
            ]
        }

        try:
            resp = requests.post(
                f"{API_BASE}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "reply_markup": keyboard,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("ok"):
                logger.info("[승인] 텔레그램 메시지 전송 완료: %s", approval_id)
            else:
                logger.warning("[승인] 텔레그램 전송 실패: %s", data)
        except Exception as e:
            logger.error("[승인] 텔레그램 전송 오류: %s", e)

    def _answer_callback(self, query_id: str, text: str) -> None:
        """콜백 응답."""
        try:
            requests.post(
                f"{API_BASE}/answerCallbackQuery",
                json={"callback_query_id": query_id, "text": text},
                timeout=5,
            )
        except Exception:
            pass

    def _edit_message(self, chat_id: str, msg_id: int, text: str) -> None:
        """메시지 텍스트 수정."""
        try:
            requests.post(
                f"{API_BASE}/editMessageText",
                json={"chat_id": chat_id, "message_id": msg_id, "text": text},
                timeout=5,
            )
        except Exception:
            pass

    # ──────────────────────────────────────────
    # JSON 파일 I/O
    # ──────────────────────────────────────────

    def _load_approvals(self) -> dict:
        """승인 파일 로드. 오래된 항목은 자동 정리."""
        if not APPROVAL_FILE.exists():
            return {}
        try:
            with open(APPROVAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 1시간 이상 된 항목 정리
            cutoff = time.time() - 3600
            cleaned = {}
            for k, v in data.items():
                req_time = v.get("requested_at", "")
                try:
                    req_ts = datetime.fromisoformat(req_time).timestamp()
                    if req_ts > cutoff:
                        cleaned[k] = v
                except (ValueError, TypeError):
                    pass
            return cleaned
        except Exception:
            return {}

    def _save_approvals(self, data: dict) -> None:
        """승인 파일 저장."""
        APPROVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(APPROVAL_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[승인] 파일 저장 실패: %s", e)
