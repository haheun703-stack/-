"""
텔레그램 양방향 명령 봇 — long-polling 기반.

사용자가 텔레그램에서 /명령어, /상태 등 한글 명령을 보내면
봇이 처리하고 결과를 메시지로 응답.

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


class TelegramCommandBot:
    """텔레그램 명령어 수신 + 처리 봇 (백그라운드 스레드)."""

    def __init__(self, scheduler=None):
        self._offset = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._scheduler = scheduler  # DailyScheduler 인스턴스 참조
        self._start_time = datetime.now()

    # ══════════════════════════════════════════
    # 스레드 관리
    # ══════════════════════════════════════════

    def start(self) -> None:
        """백그라운드 polling 시작."""
        if not TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN 미설정 — 명령 봇 비활성")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="tg-cmd-bot", daemon=True,
        )
        self._thread.start()
        logger.info("텔레그램 명령 봇 시작 (long-polling)")

    def stop(self) -> None:
        """polling 중지."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("텔레그램 명령 봇 종료")

    def _poll_loop(self) -> None:
        """메인 polling 루프 — long-polling 30초."""
        # 시작 시 기존 미처리 메시지 건너뛰기
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
        """봇 시작 전 쌓인 메시지 건너뛰기."""
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
        """Telegram getUpdates API (long-polling)."""
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
    # 메시지 처리
    # ══════════════════════════════════════════

    def _handle_update(self, update: dict) -> None:
        """메시지 파싱 + 명령 라우팅."""
        msg = update.get("message")
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != TELEGRAM_CHAT_ID:
            logger.warning("[명령봇] 미허가 사용자: chat_id=%s", chat_id)
            return

        text = msg.get("text", "").strip()
        if not text.startswith("/"):
            return

        parts = text.split()
        cmd = parts[0].lower().split("@")[0]  # /help@botname → /help
        args = parts[1:]

        logger.info("[명령봇] 수신: %s (args=%s)", cmd, args)

        handler = COMMANDS.get(cmd)
        if handler:
            try:
                handler(self, args)
            except Exception as e:
                logger.error("[명령봇] %s 처리 오류: %s", cmd, e)
                self._reply(f"\u274c 명령 오류: {e}")
        else:
            self._reply(
                f"\u2753 알 수 없는 명령: {cmd}\n"
                f"/명령어 로 사용 가능한 명령을 확인하세요."
            )

    def _reply(self, text: str) -> None:
        """텔레그램 메시지 응답."""
        from src.telegram_sender import send_message
        send_message(text)

    # ══════════════════════════════════════════
    # 명령 핸들러
    # ══════════════════════════════════════════

    def _cmd_ping(self, args: list) -> None:
        """/연결 — alive 체크"""
        now = datetime.now().strftime("%H:%M:%S")
        uptime = datetime.now() - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        mins = int((uptime.total_seconds() % 3600) // 60)
        self._reply(f"\U0001f3d3 pong! | {now}\n\u23f1 uptime: {hours}시간 {mins}분")

    def _cmd_help(self, args: list) -> None:
        """/명령어 — 명령어 목록"""
        lines = [
            "\U0001f4cb Quantum Master 명령어",
            "\u2501" * 24,
            "",
            "\U0001f50d [조회]",
            "  /연결 — 봇 연결 확인",
            "  /상태 — 스케줄러 상태",
            "  /잔고 — 보유 종목 현황",
            "  /후보 — 매수 후보 조회",
            "  /스케줄 — 오늘 Phase 시간표",
            "",
            "\U0001f4e4 [실행]",
            "  /브리핑 — 장전 브리핑 즉시",
            "  /수급 — 수급 스냅샷 수집",
            "  /실행 N — Phase 즉시 실행",
            "    (예: /실행 3b, /실행 10b)",
            "",
            "\u26a0\ufe0f [제어]",
            "  /정지 — 매매 중단",
            "  /재개 — 매매 재개",
        ]
        self._reply("\n".join(lines))

    def _cmd_status(self, args: list) -> None:
        """/상태 — 스케줄러 상태 + 다음 Phase"""
        now = datetime.now()
        lines = [
            f"\U0001f4e1 스케줄러 상태 | {now.strftime('%H:%M:%S')}",
            "\u2500" * 24,
        ]

        # 스케줄러 모드
        if self._scheduler:
            mode = self._scheduler.mode
            enabled = self._scheduler.enabled
            lines.append(f"  모드: {mode} | 실주문: {'ON' if enabled else 'OFF'}")
            lines.append(f"  공휴일: {'예' if self._scheduler._is_holiday else '아니오'}")
            lines.append(f"  매수후보: {len(self._scheduler._buy_signals)}종목")
            lines.append(f"  수급스냅샷: {len(self._scheduler._supply_snapshots)}회")
        else:
            lines.append("  스케줄러 미연결 (독립 실행)")

        # STOP.signal 상태
        stop_file = PROJECT_ROOT / "STOP.signal"
        if stop_file.exists():
            lines.append("  \U0001f6d1 STOP.signal 활성!")
        else:
            lines.append("  \U0001f7e2 정상 운영 중")

        # 다음 Phase 예측
        lines.append("")
        lines.append("\u23f0 다음 Phase")
        if self._scheduler:
            schedule = self._scheduler.schedule
            current_time = now.strftime("%H:%M")
            next_phases = []
            for name, time_str in sorted(schedule.items(), key=lambda x: x[1]):
                if time_str > current_time:
                    next_phases.append((time_str, name))
            if next_phases:
                for t, n in next_phases[:3]:
                    lines.append(f"  {t} — {n}")
            else:
                lines.append("  오늘 남은 Phase 없음")

        self._reply("\n".join(lines))

    def _cmd_schedule(self, args: list) -> None:
        """/스케줄 — 오늘 Phase 시간표"""
        if not self._scheduler:
            self._reply("\u274c 스케줄러 미연결 (독립 실행)")
            return

        now = datetime.now()
        current_time = now.strftime("%H:%M")
        schedule = self._scheduler.schedule
        sorted_phases = sorted(schedule.items(), key=lambda x: x[1])

        lines = [
            f"\U0001f4c5 오늘 스케줄 | {now.strftime('%m/%d %H:%M')}",
            "\u2500" * 24,
        ]
        for name, time_str in sorted_phases:
            marker = "\u25b6" if time_str > current_time else "\u2705"
            lines.append(f"  {marker} {time_str} — {name}")

        self._reply("\n".join(lines))

    def _cmd_positions(self, args: list) -> None:
        """/잔고 — 보유 종목 현황"""
        pos_path = PROJECT_ROOT / "data" / "positions.json"
        if not pos_path.exists():
            self._reply("\U0001f4bc 보유 종목 없음 (positions.json 없음)")
            return

        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)

        positions = data.get("positions", [])
        if not positions:
            self._reply("\U0001f4bc 보유 종목 없음")
            return

        lines = [
            "\U0001f4bc 보유 종목 현황",
            "\u2500" * 24,
        ]
        total_pnl = 0
        for p in positions:
            name = p.get("name", p.get("ticker", "?"))
            ticker = p.get("ticker", "?")
            entry = p.get("entry_price", 0)
            current = p.get("current_price", entry)
            pnl_pct = p.get("unrealized_pnl_pct", 0)
            shares = p.get("shares", 0)
            pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
            lines.append(f"{pnl_emoji} {name} ({ticker})")
            lines.append(f"  진입 {entry:,.0f} | 현재 {current:,.0f} | {pnl_pct:+.1f}%")
            total_pnl += pnl_pct

        lines.append("")
        avg_pnl = total_pnl / len(positions) if positions else 0
        lines.append(f"\U0001f4ca 평균 수익률: {avg_pnl:+.1f}% ({len(positions)}종목)")
        self._reply("\n".join(lines))

    def _cmd_scan(self, args: list) -> None:
        """/후보 — 최신 매수 후보 조회"""
        import pandas as pd

        sig_path = PROJECT_ROOT / "results" / "signals_log.csv"
        if not sig_path.exists():
            self._reply("\U0001f50d 스캔 결과 없음 (signals_log.csv 없음)")
            return

        df = pd.read_csv(sig_path)
        if df.empty:
            self._reply("\U0001f50d 스캔 결과 없음")
            return

        if "date" in df.columns:
            latest_date = df["date"].max()
            df = df[df["date"] == latest_date]
        if "zone_score" in df.columns:
            df = df.sort_values("zone_score", ascending=False)

        lines = [
            f"\U0001f50d 매수 후보 ({latest_date if 'date' in df.columns else '?'})",
            "\u2501" * 24,
        ]

        grade_map = {
            0: ("\U0001f525", "S"), 1: ("\u2b50", "A"),
            2: ("\U0001f539", "B"), 3: ("\u26d4", "C"),
        }

        for rank, (_, row) in enumerate(df.head(5).iterrows()):
            emoji, g = grade_map.get(rank, ("\u2796", "D"))
            ticker = str(row.get("ticker", "?")).zfill(6)
            name = self._get_stock_name(ticker)
            entry = row.get("entry_price", 0)
            rr = row.get("rr_ratio", 0)
            zone = row.get("zone_score", 0)
            trigger = row.get("trigger_type", "")
            lines.append(f"{emoji} {g}등급 {name} ({ticker})")
            lines.append(f"  {entry:,.0f}원 | R:R {rr:.1f}x | Zone {zone:.2f}")
            lines.append(f"  트리거: {trigger}")
            lines.append("")

        if df.empty:
            lines.append("  Kill 필터 통과 종목 없음")

        self._reply("\n".join(lines))

    def _cmd_briefing(self, args: list) -> None:
        """/브리핑 — 장전 브리핑 즉시 발송"""
        self._reply("\u23f3 장전 브리핑 생성 중...")
        try:
            from scripts.send_market_briefing import build_briefing_message
            msg = build_briefing_message()
            self._reply(msg)
        except Exception as e:
            self._reply(f"\u274c 브리핑 생성 실패: {e}")

    def _cmd_supply(self, args: list) -> None:
        """/수급 — 수급 스냅샷 즉시 수집"""
        if not self._scheduler:
            self._reply("\u274c 스케줄러 미연결 — 독립 실행 불가")
            return
        self._reply("\u23f3 수급 스냅샷 수집 중...")
        try:
            snap_num = len(self._scheduler._supply_snapshots) + 1
            self._scheduler.phase_supply_snapshot(snap_num)
            self._reply(f"\u2705 수급 스냅샷 {snap_num}차 수집 완료!")
        except Exception as e:
            self._reply(f"\u274c 수급 수집 실패: {e}")

    def _cmd_stop(self, args: list) -> None:
        """/정지 — 매매 중단 (STOP.signal 생성)"""
        stop_path = PROJECT_ROOT / "STOP.signal"
        stop_path.write_text(
            f"STOPPED by telegram command at {datetime.now().isoformat()}",
            encoding="utf-8",
        )
        self._reply(
            "\U0001f6d1 매매 중단됨\n"
            "STOP.signal 생성 완료\n"
            "재개: /재개"
        )
        logger.warning("[명령봇] /정지 명령으로 매매 중단")

    def _cmd_resume(self, args: list) -> None:
        """/재개 — 매매 재개 (STOP.signal 삭제)"""
        stop_path = PROJECT_ROOT / "STOP.signal"
        if stop_path.exists():
            stop_path.unlink()
            self._reply("\U0001f7e2 매매 재개됨\nSTOP.signal 삭제 완료")
            logger.info("[명령봇] /재개 명령으로 매매 재개")
        else:
            self._reply("\U0001f7e2 이미 정상 운영 중 (STOP.signal 없음)")

    def _cmd_phase(self, args: list) -> None:
        """/실행 N — 특정 Phase 즉시 실행"""
        if not args:
            self._reply(
                "\u2753 사용법: /실행 <번호>\n"
                "예: /실행 3b, /실행 10b, /실행 1\n"
                "가능: 0~11, 3b, snap1~4, 8-2~8-5, 10b"
            )
            return

        if not self._scheduler:
            self._reply("\u274c 스케줄러 미연결 — Phase 실행 불가")
            return

        phase_key = args[0].lower()
        phases = {
            "0": ("일일 리셋", self._scheduler.phase_daily_reset),
            "1": ("US 데이터 수집", self._scheduler.phase_us_close_collect),
            "2": ("매크로 수집", self._scheduler.phase_macro_collect),
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
        # snap1~4
        for i in range(1, 5):
            phases[f"snap{i}"] = (
                f"수급 스냅샷 {i}차",
                lambda n=i: self._scheduler.phase_supply_snapshot(n),
            )

        if phase_key not in phases:
            available = ", ".join(sorted(phases.keys()))
            self._reply(f"\u274c 없는 Phase: {phase_key}\n사용 가능: {available}")
            return

        label, func = phases[phase_key]
        self._reply(f"\u23f3 Phase {phase_key} 실행 중: {label}")

        try:
            # 별도 스레드에서 실행 (blocking 방지)
            t = threading.Thread(target=self._run_phase, args=(phase_key, label, func))
            t.start()
        except Exception as e:
            self._reply(f"\u274c Phase {phase_key} 실행 실패: {e}")

    def _run_phase(self, key: str, label: str, func) -> None:
        """Phase를 별도 스레드에서 실행하고 완료 알림."""
        try:
            func()
            self._reply(f"\u2705 Phase {key} 완료: {label}")
        except Exception as e:
            self._reply(f"\u274c Phase {key} 오류: {e}")

    # ──── 유틸 ────

    def _get_stock_name(self, ticker: str) -> str:
        """ticker → 종목명 조회."""
        stock_dir = PROJECT_ROOT / "stock_data_daily"
        if stock_dir.exists():
            for csv in stock_dir.glob(f"*_{ticker}.csv"):
                return csv.stem.rsplit("_", 1)[0]
        return ticker


# ══════════════════════════════════════════
# 명령 라우팅 테이블
# ══════════════════════════════════════════

COMMANDS = {
    # ── 조회 ──
    "/연결": TelegramCommandBot._cmd_ping,
    "/명령어": TelegramCommandBot._cmd_help,
    "/상태": TelegramCommandBot._cmd_status,
    "/잔고": TelegramCommandBot._cmd_positions,
    "/후보": TelegramCommandBot._cmd_scan,
    "/스케줄": TelegramCommandBot._cmd_schedule,
    # ── 실행 ──
    "/브리핑": TelegramCommandBot._cmd_briefing,
    "/수급": TelegramCommandBot._cmd_supply,
    "/실행": TelegramCommandBot._cmd_phase,
    # ── 제어 ──
    "/정지": TelegramCommandBot._cmd_stop,
    "/재개": TelegramCommandBot._cmd_resume,
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
    print("  텔레그램 명령 봇 (독립 실행)")
    print("  텔레그램에서 /연결 을 보내 연결을 확인하세요")
    print("  Ctrl+C 로 종료")
    print("=" * 50)

    bot = TelegramCommandBot()
    bot._start_time = datetime.now()
    bot.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()
        print("\n봇 종료됨")
