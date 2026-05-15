"""장중 실시간 학습 엔진 (Phase 12a)

09:00~15:30 KIS WebSocket으로 40종목 구독하여 분단위 패턴 수집.

구독 종목 (총 40 이내):
- ETF 화이트리스트 12종 (필수, 자동매매 대상)
- picks_v2 TOP 15 (당일 급등 후보)
- bluechip TOP 13 (모태 시총 대형주)

수집 항목 (분단위 집계):
- OHLC + 체결량
- 체결강도 평균 (KIS field[18])
- 매수/매도 체결 건수
- 누적 거래량

저장:
- data/intraday/intraday_minute_{YYYYMMDD}.db (SQLite, 1분당 1행)

운영:
- cron: 평일 08:55 시작, 15:35 자동 종료
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.kis_websocket import KISWebSocketClient

DATA_DIR = PROJECT_ROOT / "data"
INTRA_DIR = DATA_DIR / "intraday"
INTRA_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = PROJECT_ROOT / "logs" / "intraday_learner.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("intraday_learner")

# ETF 화이트리스트 (kis_order_adapter와 동기화)
ETF_WHITELIST: Set[str] = {
    "122630", "069500", "114800", "252670",  # JARVIS 4모드
    "251340", "233160", "243880",             # 추가 인버스/레버리지
    "395160", "466920", "228810", "401470", "337160",  # theme TOP 5
}

# bluechip TOP 13 (시총 대형주, 5/16 기준)
BLUECHIP_TOP: Set[str] = {
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "373220",  # LG에너지솔루션
    "207940",  # 삼성바이오로직스
    "005380",  # 현대차
    "000270",  # 기아
    "068270",  # 셀트리온
    "035420",  # NAVER
    "035720",  # 카카오
    "051910",  # LG화학
    "012330",  # 현대모비스
    "066570",  # LG전자
    "028260",  # 삼성물산
}

MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(15, 35)  # 15:30 정규 + 5분 grace


def select_subscriptions(today_str: str) -> List[str]:
    """40종목 이내 구독 리스트 결정."""
    codes: Set[str] = set()
    codes.update(ETF_WHITELIST)
    codes.update(BLUECHIP_TOP)

    # picks_v2 TOP 15 (전일 BAT 결과)
    today_compact = today_str.replace("-", "")
    picks_path = DATA_DIR / f"picks_v2_{today_compact}.csv"
    if picks_path.exists():
        try:
            df = pd.read_csv(picks_path, encoding="utf-8-sig")
            if "code" in df.columns:
                picks = df.head(15)["code"].astype(str).str.zfill(6).tolist()
                codes.update(picks)
                logger.info(f"[select] picks_v2 {len(picks)}종목 추가: {picks[:5]}...")
        except Exception as e:
            logger.warning(f"[select] picks_v2 로드 실패: {e}")
    else:
        logger.info(f"[select] picks_v2 파일 없음 ({picks_path.name}), ETF+blue만 구독")

    result = list(codes)[:40]
    logger.info(f"[select] 총 {len(result)}종목 구독 (ETF {len(ETF_WHITELIST)} + blue {len(BLUECHIP_TOP)} + picks)")
    return result


def init_db(today_str: str) -> Path:
    """오늘 DB 초기화."""
    today_compact = today_str.replace("-", "")
    db_path = INTRA_DIR / f"intraday_minute_{today_compact}.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS intraday_minute (
            date TEXT,
            code TEXT,
            minute TEXT,
            open INTEGER,
            high INTEGER,
            low INTEGER,
            close INTEGER,
            volume INTEGER,
            cum_volume INTEGER,
            strength_avg REAL,
            buy_count INTEGER,
            sell_count INTEGER,
            tick_count INTEGER,
            PRIMARY KEY (date, code, minute)
        );
        CREATE INDEX IF NOT EXISTS idx_code_minute ON intraday_minute(code, minute);
        """
    )
    conn.commit()
    conn.close()
    logger.info(f"[db] 초기화: {db_path}")
    return db_path


class MinuteAggregator:
    """종목별 1분 OHLC + 체결강도 집계기."""

    def __init__(self, db_path: Path, today_str: str):
        self.db_path = db_path
        self.today_str = today_str
        self._bucket = defaultdict(lambda: defaultdict(lambda: {
            "open": None,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
            "cum_volume": 0,
            "strength_sum": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "tick_count": 0,
        }))
        self._flushed_minutes = set()

    def on_tick(self, code: str, price: int, volume: int, time_hhmmss: str, fields: list):
        """KIS H0STCNT0 tick 처리.

        fields 표준 인덱스 (H0STCNT0):
        - 0: 종목코드, 1: 체결시간, 2: 현재가
        - 12: 체결거래량, 13: 누적거래량
        - 15: 매도체결건수, 16: 매수체결건수
        - 18: 체결강도 (매수체결량/매도체결량 비율)
        """
        if not time_hhmmss or len(time_hhmmss) < 6:
            return
        minute = time_hhmmss[:4]  # HHMM
        b = self._bucket[code][minute]

        # OHLC
        if b["open"] is None:
            b["open"] = price
            b["high"] = price
            b["low"] = price
        b["high"] = max(b["high"], price)
        b["low"] = min(b["low"], price) if b["low"] else price
        b["close"] = price
        b["volume"] += volume

        # KIS 표준 인덱스
        def _f(idx, default="0"):
            return fields[idx] if idx < len(fields) else default

        try:
            b["cum_volume"] = int(_f(13)) or b["cum_volume"]
            b["sell_count"] = int(_f(15)) or b["sell_count"]
            b["buy_count"] = int(_f(16)) or b["buy_count"]
            strength = float(_f(18, "0"))
            b["strength_sum"] += strength
        except (ValueError, IndexError):
            pass

        b["tick_count"] += 1

    def flush_minute(self, current_minute: str):
        """이전 분들을 DB에 flush (현재 분은 제외)."""
        conn = sqlite3.connect(self.db_path)
        rows = []
        for code, minutes in list(self._bucket.items()):
            for minute, b in list(minutes.items()):
                if minute >= current_minute:
                    continue
                if (code, minute) in self._flushed_minutes:
                    continue
                if b["tick_count"] == 0:
                    continue
                strength_avg = b["strength_sum"] / b["tick_count"]
                rows.append((
                    self.today_str, code, minute,
                    b["open"], b["high"], b["low"], b["close"],
                    b["volume"], b["cum_volume"], strength_avg,
                    b["buy_count"], b["sell_count"], b["tick_count"],
                ))
                self._flushed_minutes.add((code, minute))
        if rows:
            conn.executemany(
                """INSERT OR REPLACE INTO intraday_minute
                   (date, code, minute, open, high, low, close, volume, cum_volume,
                    strength_avg, buy_count, sell_count, tick_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            logger.info(f"[flush] {current_minute} 이전 분 {len(rows)}행 저장")
        conn.close()


async def run_intraday_session(today_str: str, force: bool = False):
    """장중 세션 실행."""
    db_path = init_db(today_str)
    codes = select_subscriptions(today_str)
    agg = MinuteAggregator(db_path, today_str)

    ws = KISWebSocketClient(max_subscriptions=40)
    await ws.connect()

    async def on_tick(code, price, volume, ts, fields):
        agg.on_tick(code, price, volume, ts, fields)

    await ws.subscribe(codes, on_tick=on_tick, tr_id="H0STCNT0")
    logger.info(f"[session] 구독 완료: {len(codes)}종목")

    # 수신 루프 + 매 분 flush
    receiver_task = asyncio.create_task(ws.run_forever())
    last_flushed = ""
    try:
        while True:
            now = datetime.now()
            if not force and now.time() >= MARKET_CLOSE:
                logger.info(f"[session] 장마감 시간 {now.time()} → 종료")
                break

            current_minute = now.strftime("%H%M")
            if current_minute != last_flushed:
                agg.flush_minute(current_minute)
                last_flushed = current_minute

            await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass
    finally:
        # 최종 flush
        future_min = (datetime.now() + timedelta(minutes=1)).strftime("%H%M")
        agg.flush_minute(future_min)
        await ws.stop()
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass
        logger.info(f"[session] 종료, DB={db_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (기본: today)")
    parser.add_argument("--force", action="store_true", help="장마감 시간 무시")
    parser.add_argument("--wait-open", action="store_true", help="09:00까지 대기")
    args = parser.parse_args()

    today_str = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.wait_open:
        while datetime.now().time() < MARKET_OPEN:
            now = datetime.now()
            wait = (datetime.combine(now.date(), MARKET_OPEN) - now).total_seconds()
            logger.info(f"[wait] 09:00 대기 중... {wait:.0f}초 남음")
            import time as _time
            _time.sleep(min(60, wait))

    asyncio.run(run_intraday_session(today_str, force=args.force))


if __name__ == "__main__":
    main()
