"""KRX 국적별 외국인 거래 수집기 — data.krx.co.kr HARD API

매일 장마감 후 실행:
  1. KRX 로그인 (MDCCOMS001D1.cmd)
  2. 종목별 국적별 거래량 수집 (MDCHARD05302)
  3. SQLite DB에 일별 저장

데이터 특성:
  - ACC_TRDVOL = 일별 누적 거래량 (매수+매도 합계)
  - 순매수/순매도는 직접 제공 X → KIS 외국인 순매수와 교차 분석 필요
  - T+1 지연: 당일 데이터는 다음날부터 조회 가능
  - 주말/공휴일 데이터 없음
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "krx_nationality"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "nationality.db"

load_dotenv(PROJECT_ROOT / ".env")


# ═══════════════════════════════════════════════════
# SQLite 스키마
# ═══════════════════════════════════════════════════

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nationality_daily (
    date       TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    name       TEXT NOT NULL,
    country    TEXT NOT NULL,
    trade_vol  INTEGER NOT NULL DEFAULT 0,  -- 일별 거래량 (매수+매도)
    PRIMARY KEY (date, ticker, country)
);

CREATE INDEX IF NOT EXISTS idx_nd_ticker_date
    ON nationality_daily(ticker, date);

CREATE INDEX IF NOT EXISTS idx_nd_date
    ON nationality_daily(date);

CREATE TABLE IF NOT EXISTS collect_log (
    date        TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    stock_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'OK',
    PRIMARY KEY (date)
);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB 초기화 + 스키마 생성."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════
# KRX 세션 관리
# ═══════════════════════════════════════════════════

class KRXSession:
    """KRX Data Marketplace 인증 세션."""

    LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    MAIN_URL = "https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd"
    HARD_URL = "https://data.krx.co.kr/contents/MDC/HARD/hardController/MDCHARD053.cmd"
    DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd?bld=dbms/MDC/HARD/MDCHARD05302"
    SEARCH_URL = "https://data.krx.co.kr/comm/util/SearchEngine/isuCore.cmd"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ko-KR,ko;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
        })
        self._logged_in = False

    def login(self) -> bool:
        """KRX 로그인 (skipDup=Y)."""
        krx_id = os.getenv("KRX_DATA_ID")
        krx_pw = os.getenv("KRX_DATA_PW")
        if not krx_id or not krx_pw:
            logger.error("KRX_DATA_ID / KRX_DATA_PW 미설정")
            return False

        try:
            # 세션 쿠키 획득
            self.session.get(self.MAIN_URL, timeout=15)

            # 로그인
            r = self.session.post(
                self.LOGIN_URL,
                data={
                    "mbrId": krx_id,
                    "pw": krx_pw,
                    "skipDup": "Y",
                },
                headers={
                    "Referer": "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc",
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                },
                timeout=15,
            )
            resp = r.json()
            code = resp.get("_error_code", "")
            msg = resp.get("_error_message", "")

            if code == "CD001":
                logger.info(f"KRX 로그인 성공: {msg}")
                # HARD 페이지 방문 (세션 컨텍스트 설정)
                self.session.get(self.HARD_URL, timeout=15)
                self._logged_in = True
                return True
            else:
                logger.error(f"KRX 로그인 실패: {code} / {msg}")
                return False

        except Exception as e:
            logger.error(f"KRX 로그인 에러: {e}")
            return False

    def find_isin(self, keyword: str) -> tuple[str | None, str | None]:
        """종목 검색 → (ISIN, 종목명)."""
        try:
            r = self.session.post(
                f"{self.SEARCH_URL}?solrType=isu&solrIsuType=STK"
                f"&solrKeyword={requests.utils.quote(keyword)}&rows=3&start=0",
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
                timeout=10,
            )
            data = r.json()
            if data.get("result"):
                item = data["result"][0]
                isin = item["isu_cd"]
                name = item["isu_abbrv"]
                if isinstance(isin, list):
                    isin = isin[0]
                if isinstance(name, list):
                    name = name[0]
                return isin, name
        except Exception as e:
            logger.warning(f"ISIN 검색 실패 [{keyword}]: {e}")
        return None, None

    def fetch_nationality(
        self, isin: str, date: str
    ) -> dict[str, int]:
        """단일 종목의 국적별 거래량 조회.

        Args:
            isin: ISIN 코드 (예: KR7005930003)
            date: 조회 날짜 YYYYMMDD

        Returns:
            {"영국": 7533072, "중국": 138071, ...}
        """
        try:
            r = self.session.post(
                self.DATA_URL,
                data={
                    "locale": "ko_KR",
                    "isuCd": isin,
                    "isuCd2": isin,
                    "strtDd": date,
                    "endDd": date,
                    "order": "DESC",
                },
                headers={
                    "Referer": self.HARD_URL,
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                },
                timeout=15,
            )
            block = r.json().get("block1", [])
            countries: dict[str, int] = {}
            for row in block:
                for suffix in ["A", "B", "C"]:
                    cn = row.get(f"CNTR_NM_{suffix}", "")
                    vol = row.get(f"ACC_TRDVOL_{suffix}", "0")
                    if cn and vol:
                        countries[cn] = int(vol.replace(",", ""))
            return countries
        except Exception as e:
            logger.warning(f"국적별 데이터 실패 [{isin}]: {e}")
            return {}


# ═══════════════════════════════════════════════════
# 수집 대상 종목
# ═══════════════════════════════════════════════════

# 보유 + 워치 + 주요 대형주 + 방산/조선/반도체 핵심
DEFAULT_TARGETS = [
    # 보유 + 관심
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("012450", "한화에어로스페이스"),
    ("064350", "현대로템"),
    ("009540", "HD한국조선해양"),
    ("329180", "HD현대중공업"),
    ("047810", "한국항공우주"),
    ("000270", "기아"),
    ("005380", "현대차"),
    ("373220", "LG에너지솔루션"),
    ("071050", "한국금융지주"),
    ("207940", "삼성바이오로직스"),
    # 워치리스트
    ("307950", "산일전기"),
    ("272210", "한화시스템"),
    ("103140", "풍산"),
    ("090430", "아모레퍼시픽"),
    ("051900", "LG생활건강"),
    # 추가 대형주 / 섹터 대표
    ("105560", "KB금융"),
    ("055550", "신한지주"),
    ("068270", "셀트리온"),
    ("006400", "삼성SDI"),
    ("035420", "NAVER"),
    ("035720", "카카오"),
    ("028260", "삼성물산"),
    ("010130", "고려아연"),
    ("402340", "SK스퀘어"),
    ("180640", "한진칼"),
    ("042700", "한미반도체"),
    ("096770", "SK이노베이션"),
    ("000990", "DB하이텍"),
    ("352820", "하이브"),
]


def get_target_isins(krx: KRXSession) -> list[tuple[str, str, str]]:
    """대상 종목들의 ISIN을 일괄 조회.

    Returns:
        [(ticker, name, isin), ...]
    """
    results = []
    for ticker, name in DEFAULT_TARGETS:
        isin, found_name = krx.find_isin(name)
        if isin:
            results.append((ticker, found_name or name, isin))
        else:
            logger.warning(f"ISIN 못 찾음: {ticker} {name}")
        time.sleep(0.1)
    return results


# ═══════════════════════════════════════════════════
# 일별 수집 + DB 저장
# ═══════════════════════════════════════════════════

def collect_and_store(
    date: str | None = None,
    db_path: Path = DB_PATH,
    delay: float = 0.3,
) -> dict:
    """하루치 국적별 외국인 거래 데이터 수집 → DB 저장.

    Args:
        date: YYYYMMDD (None이면 전 거래일 자동 계산)
        db_path: SQLite 경로
        delay: API 요청 간격 (초)

    Returns:
        {"date": "20260310", "stocks": 30, "rows": 450, "status": "OK"}
    """
    if date is None:
        # 전 거래일 (T-1) 자동 계산
        today = datetime.now()
        if today.hour < 18:
            # 18시 이전이면 T-2 (오늘 데이터 아직 없을 수 있음)
            d = today - timedelta(days=1)
        else:
            d = today
        # 주말 스킵
        while d.weekday() >= 5:  # 5=토, 6=일
            d -= timedelta(days=1)
        # 하루 더 빼기 (T+1 지연)
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        date = d.strftime("%Y%m%d")

    conn = init_db(db_path)

    # 이미 수집했는지 확인
    existing = conn.execute(
        "SELECT stock_count FROM collect_log WHERE date = ?", (date,)
    ).fetchone()
    if existing and existing[0] > 0:
        logger.info(f"{date} 이미 수집 완료 ({existing[0]}종목)")
        conn.close()
        return {"date": date, "stocks": existing[0], "rows": 0, "status": "SKIP"}

    # KRX 로그인
    krx = KRXSession()
    if not krx.login():
        conn.close()
        return {"date": date, "stocks": 0, "rows": 0, "status": "LOGIN_FAIL"}

    # ISIN 조회
    targets = get_target_isins(krx)
    logger.info(f"수집 대상: {len(targets)}종목, 날짜: {date}")

    total_rows = 0
    success_count = 0

    for ticker, name, isin in targets:
        countries = krx.fetch_nationality(isin, date)
        if not countries:
            continue

        success_count += 1
        for country, vol in countries.items():
            conn.execute(
                "INSERT OR REPLACE INTO nationality_daily "
                "(date, ticker, name, country, trade_vol) VALUES (?, ?, ?, ?, ?)",
                (date, ticker, name, country, vol),
            )
            total_rows += 1

        time.sleep(delay)

    # 수집 로그
    conn.execute(
        "INSERT OR REPLACE INTO collect_log (date, collected_at, stock_count, status) "
        "VALUES (?, ?, ?, ?)",
        (date, datetime.now().isoformat(), success_count, "OK"),
    )
    conn.commit()
    conn.close()

    logger.info(f"수집 완료: {date} / {success_count}종목 / {total_rows}행")
    return {
        "date": date,
        "stocks": success_count,
        "rows": total_rows,
        "status": "OK",
    }


def backfill(days: int = 10, db_path: Path = DB_PATH) -> list[dict]:
    """과거 N 거래일 백필."""
    results = []
    d = datetime.now() - timedelta(days=1)

    filled = 0
    attempts = 0
    while filled < days and attempts < days * 2:
        if d.weekday() < 5:  # 평일만
            date_str = d.strftime("%Y%m%d")
            r = collect_and_store(date_str, db_path)
            results.append(r)
            if r["status"] == "OK":
                filled += 1
        d -= timedelta(days=1)
        attempts += 1

    return results
