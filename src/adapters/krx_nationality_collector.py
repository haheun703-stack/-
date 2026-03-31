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
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # 동시 읽기/쓰기 허용
    conn.execute("PRAGMA busy_timeout=10000")  # 10초 대기
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

# 보유 + 워치 + 주요 대형주 + 방산/조선/반도체 핵심 (항상 추적)
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


def _load_dynamic_universe() -> list[tuple[str, str]]:
    """데이터 파일에서 추적 대상 종목을 동적으로 로드.

    소스:
      1. event_catalyst.json — 이벤트 촉매 종목
      2. tomorrow_picks.json — 추천 후보 TOP
      3. tomorrow_picks_flowx.json — FLOWX 추천 TOP
      4. accumulation_tracker.json — 매집 추적 종목
      5. whale_detect.json — 세력 감지 종목
      6. volume_spike_watchlist.json — 눌림목/수급폭발
      7. relay/relay_signal.json — 릴레이 발화 종목
      8. portfolio_allocation.json — 보유 종목

    Returns:
        [(ticker, name), ...] — 중복 제거됨
    """
    seen: set[str] = set()
    result: list[tuple[str, str]] = []

    def _add(ticker: str, name: str):
        if ticker and ticker not in seen and len(ticker) == 6:
            seen.add(ticker)
            result.append((ticker, name or ticker))

    # 1. 이벤트 촉매
    try:
        ec = json.loads((DATA_DIR.parent / "event_catalyst.json").read_text(encoding="utf-8"))
        for s in ec.get("stocks", []):
            _add(s.get("ticker", ""), s.get("name", ""))
    except Exception:
        pass

    # 2~3. 추천 TOP (일반 + FLOWX)
    for fname in ["tomorrow_picks.json", "tomorrow_picks_flowx.json"]:
        try:
            tp = json.loads((DATA_DIR.parent / fname).read_text(encoding="utf-8"))
            for p in tp.get("picks", []):
                if isinstance(p, dict):
                    _add(p.get("ticker", ""), p.get("name", ""))
        except Exception:
            pass

    # 4. 매집 추적
    try:
        at = json.loads((DATA_DIR.parent / "accumulation_tracker.json").read_text(encoding="utf-8"))
        tracking = at.get("tracking", at.get("stocks", {}))
        if isinstance(tracking, dict):
            for tk, info in tracking.items():
                name = info.get("name", "") if isinstance(info, dict) else ""
                _add(tk, name)
        elif isinstance(tracking, list):
            for item in tracking:
                if isinstance(item, dict):
                    _add(item.get("ticker", ""), item.get("name", ""))
    except Exception:
        pass

    # 5. 세력 감지
    try:
        wd = json.loads((DATA_DIR.parent / "whale_detect.json").read_text(encoding="utf-8"))
        for key in ["detected", "signals", "stocks"]:
            items = wd.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        _add(item.get("ticker", ""), item.get("name", ""))
            elif isinstance(items, dict):
                for tk, info in items.items():
                    name = info.get("name", "") if isinstance(info, dict) else ""
                    _add(tk, name)
    except Exception:
        pass

    # 6. 눌림목/수급폭발
    for fname in ["volume_spike_watchlist.json", "smallcap_explosion.json"]:
        try:
            vs = json.loads((DATA_DIR.parent / fname).read_text(encoding="utf-8"))
            for key in ["watchlist", "candidates", "signals", "stocks"]:
                items = vs.get(key, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            _add(item.get("ticker", item.get("code", "")),
                                 item.get("name", ""))
        except Exception:
            pass

    # 7. 릴레이 발화 종목
    try:
        rs = json.loads((DATA_DIR.parent / "relay" / "relay_signal.json").read_text(encoding="utf-8"))
        for sector in rs.get("fired_sectors", rs.get("sectors", [])):
            if isinstance(sector, dict):
                for stock in sector.get("stocks", sector.get("picks", [])):
                    if isinstance(stock, dict):
                        _add(stock.get("ticker", ""), stock.get("name", ""))
    except Exception:
        pass

    # 8. 보유 종목
    try:
        pa = json.loads((DATA_DIR.parent / "portfolio_allocation.json").read_text(encoding="utf-8"))
        for h in pa.get("holdings", pa.get("positions", [])):
            if isinstance(h, dict):
                _add(h.get("ticker", h.get("code", "")), h.get("name", ""))
    except Exception:
        pass

    logger.info(f"동적 유니버스: {len(result)}종목 로드")
    return result


def build_full_universe() -> list[tuple[str, str]]:
    """DEFAULT_TARGETS + 동적 유니버스 합산 (중복 제거)."""
    seen: set[str] = set()
    merged: list[tuple[str, str]] = []

    # 기본 대상 우선
    for ticker, name in DEFAULT_TARGETS:
        if ticker not in seen:
            seen.add(ticker)
            merged.append((ticker, name))

    # 동적 추가
    for ticker, name in _load_dynamic_universe():
        if ticker not in seen:
            seen.add(ticker)
            merged.append((ticker, name))

    logger.info(f"전체 유니버스: 기본 {len(DEFAULT_TARGETS)} + 동적 {len(merged) - len(DEFAULT_TARGETS)} = {len(merged)}종목")
    return merged


def get_target_isins(krx: KRXSession) -> list[tuple[str, str, str]]:
    """대상 종목들의 ISIN을 일괄 조회 (전체 유니버스).

    Returns:
        [(ticker, name, isin), ...]
    """
    universe = build_full_universe()
    results = []
    for ticker, name in universe:
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
        # 전 거래일 (T-1) 자동 계산: 항상 어제 (주말 스킵)
        # 사고 이력: T-2 로직이 과도하여 이미 SKIP된 날짜만 계산 → 매일 0건 (2026-03-25~)
        d = datetime.now() - timedelta(days=1)
        while d.weekday() >= 5:  # 5=토, 6=일
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
    consecutive_fail = 0

    for ticker, name, isin in targets:
        countries = krx.fetch_nationality(isin, date)
        if not countries:
            consecutive_fail += 1
            # 재발방지: 5연속 실패 시 재로그인 (403/세션만료 대응)
            if consecutive_fail >= 5 and success_count == 0:
                logger.warning("5연속 실패 — 세션 재로그인 시도")
                if krx.login():
                    consecutive_fail = 0
                else:
                    logger.error("재로그인 실패 — 수집 중단")
                    break
            continue

        success_count += 1
        consecutive_fail = 0
        for country, vol in countries.items():
            conn.execute(
                "INSERT OR REPLACE INTO nationality_daily "
                "(date, ticker, name, country, trade_vol) VALUES (?, ?, ?, ?, ?)",
                (date, ticker, name, country, vol),
            )
            total_rows += 1

        time.sleep(delay)

    # 0종목이면 API 에러 상태 반환 (재발방지: 403 무음 실패 방지)
    if success_count == 0 and len(targets) > 0:
        status = "API_ERROR"
        logger.error(f"수집 실패: {date} — {len(targets)}종목 시도했으나 0종목 성공 (API 오류 의심)")
    else:
        status = "OK"

    # 수집 로그
    conn.execute(
        "INSERT OR REPLACE INTO collect_log (date, collected_at, stock_count, status) "
        "VALUES (?, ?, ?, ?)",
        (date, datetime.now().isoformat(), success_count, status),
    )
    conn.commit()
    conn.close()

    logger.info(f"수집 완료: {date} / {success_count}종목 / {total_rows}행 / {status}")
    return {
        "date": date,
        "stocks": success_count,
        "rows": total_rows,
        "status": status,
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
