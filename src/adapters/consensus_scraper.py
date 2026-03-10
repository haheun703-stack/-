"""wisereport 컨센서스 데이터 스크래핑 어댑터

navercomp.wisereport.co.kr에서 종목별 컨센서스 수집:
  - 평균 목표주가, 투자의견(5점), 추정기관수
  - Forward EPS/BPS/PER/PBR, 배당 추정

Usage:
    from src.adapters.consensus_scraper import ConsensusScraper
    scraper = ConsensusScraper()
    data = scraper.fetch_one('000660')  # SK하이닉스
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://navercomp.wisereport.co.kr/",
}

BASE_URL = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx"


def _parse_num(text: str) -> float | None:
    """쉼표/공백 제거 후 숫자 파싱. 실패 시 None."""
    if not text:
        return None
    s = str(text).replace(",", "").replace(" ", "").strip()
    if not s or s in ("N/A", "-", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_int(text: str) -> int | None:
    v = _parse_num(text)
    return int(v) if v is not None else None


class ConsensusScraper:
    """wisereport에서 종목별 컨센서스 데이터 수집."""

    def __init__(self, max_retries: int = 3, timeout: int = 10):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_one(self, ticker: str) -> dict | None:
        """단일 종목 컨센서스 수집.

        Returns:
            dict with keys: ticker, name, target_price, opinion_score,
            analyst_count, forward_eps, forward_bps, forward_per,
            forward_pbr, dividend_est, fetched_at
            or None if no consensus data available.
        """
        url = f"{BASE_URL}?cmp_cd={ticker}"

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait = (attempt + 1) * 2
                    logger.warning(f"  {ticker} 재시도 {attempt+1}/{self.max_retries} ({wait}초 대기): {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"  {ticker} HTTP 실패: {e}")
                    return None

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            return self._parse_page(soup, ticker)
        except Exception as e:
            logger.error(f"  {ticker} 파싱 실패: {e}")
            return None

    def fetch_batch(
        self, tickers: list[str], delay: float = 0.5, progress: bool = True
    ) -> list[dict]:
        """복수 종목 순차 수집.

        Args:
            tickers: 종목코드 리스트
            delay: 요청 간 대기 (초)
            progress: 진행률 로깅 여부

        Returns:
            성공한 종목의 dict 리스트 (None 제외)
        """
        results = []
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            if progress and (i + 1) % 50 == 0:
                logger.info(f"  [{i+1}/{total}] 수집 중...")

            data = self.fetch_one(ticker)
            if data is not None:
                results.append(data)

            if i < total - 1:
                time.sleep(delay)

        if progress:
            logger.info(f"  수집 완료: {len(results)}/{total}개 성공")

        return results

    def _parse_page(self, soup: BeautifulSoup, ticker: str) -> dict | None:
        """wisereport 종목 페이지에서 컨센서스 데이터 추출.

        wisereport 테이블 구조 (2026-03 확인):
          Table 5: 주요지표 — PER/PBR/EPS/BPS/DPS (헤더: 실적/컨센서스)
          Table 11: 투자의견 컨센서스 — 의견점수/목표가/EPS/PER/기관수
        """

        # 종목명 추출
        name = ""
        name_tag = soup.find("span", class_="name")
        if name_tag:
            name = name_tag.get_text(strip=True)
        if not name:
            title = soup.find("title")
            if title:
                name = title.get_text(strip=True).split("-")[0].strip()

        result = {
            "ticker": ticker,
            "name": name,
            "target_price": None,
            "opinion_score": None,
            "analyst_count": None,
            "forward_eps": None,
            "forward_bps": None,
            "forward_per": None,
            "forward_pbr": None,
            "dividend_est": None,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        tables = soup.find_all("table")

        # ── 1) 투자의견 컨센서스 테이블 (목표주가/의견/기관수) ──
        for table in tables:
            text = table.get_text()
            if "투자의견" in text and "목표주가" in text and "추정기관수" in text:
                self._parse_opinion_table(table, result)
                break

        # ── 2) 주요지표 테이블 (EPS/BPS/PER/PBR/DPS) ──
        for table in tables:
            text = table.get_text()
            if "주요지표" in text and "EPS" in text and "BPS" in text:
                self._parse_financial_table(table, result)
                break

        # 목표주가 없으면 컨센서스 미제공 종목
        if result["target_price"] is None:
            return None

        return result

    def _parse_opinion_table(self, table, result: dict):
        """투자의견 컨센서스 테이블 파싱.

        실제 구조 (Table 11):
          Row 0: ['4.00', '투자의견', '목표주가(원)', 'EPS(원)', 'PER(배)', '추정기관수']
          Row 1: ['4.00', '1,308,000', '177,217', '4.72', '25']
        """
        rows = table.find_all("tr")

        # 헤더 행에서 컬럼 매핑 찾기
        header_row = None
        data_row = None

        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if "목표주가" in " ".join(cells):
                header_row = cells
            elif header_row is not None and data_row is None:
                # 헤더 다음 행이 데이터
                data_row = cells

        if header_row and data_row:
            # 헤더에서 각 항목 위치 찾기
            for hi, h in enumerate(header_row):
                if "목표주가" in h:
                    # 데이터 행에서 대응 인덱스 (헤더와 오프셋 있을 수 있음)
                    for val_text in data_row:
                        v = _parse_int(val_text)
                        if v and v > 10000:  # 목표주가는 만원 이상
                            result["target_price"] = v
                            break
                    break

            # 추정기관수: 데이터 행 마지막 숫자 (보통 끝)
            for val_text in reversed(data_row):
                v = _parse_int(val_text)
                if v and 1 <= v <= 100:
                    result["analyst_count"] = v
                    break

        # 투자의견 점수: 첫 행 첫 셀에 x.xx 형태로 있는 경우
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            for txt in cells:
                v = _parse_num(txt)
                if v and 1.0 <= v <= 5.0:
                    result["opinion_score"] = round(v, 2)
                    return  # 첫 매칭이 투자의견

    def _parse_financial_table(self, table, result: dict):
        """주요지표 테이블에서 Forward EPS/BPS/PER/PBR/DPS 추출.

        실제 구조 (Table 5):
          Row 0 (헤더): ['주요지표', '2024/12(A)', '2025/12(E)']
          Row 1: ['PER', '30.76', '13.80']
          Row 2: ['PBR', '7.79', '4.68']
          ...
          Row 5: ['EPS', '27,182원', '55,583원']
          Row 6: ['BPS', '107,256원', '163,942원']
          Row 8: ['현금DPS', '2,204원', '1,788원']
        """
        rows = table.find_all("tr")
        if len(rows) < 3:
            return

        # 헤더에서 컨센서스(E) 열 인덱스 찾기
        header_cells = rows[0].find_all(["td", "th"])
        header_texts = [c.get_text(strip=True) for c in header_cells]

        consensus_col = None
        for ci, ht in enumerate(header_texts):
            if "(E)" in ht:
                consensus_col = ci
                break

        if consensus_col is None:
            # 마지막 열 사용
            consensus_col = len(header_texts) - 1

        if consensus_col < 1:
            return

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= consensus_col:
                continue

            label = cells[0].get_text(strip=True)
            value_text = cells[consensus_col].get_text(strip=True)
            # "55,583원" → "55583"
            value_text = value_text.replace("원", "").replace("억원", "").replace("%", "")

            if label == "EPS":
                v = _parse_num(value_text)
                if v:
                    result["forward_eps"] = v
            elif label == "BPS":
                v = _parse_num(value_text)
                if v:
                    result["forward_bps"] = v
            elif label == "PER":
                v = _parse_num(value_text)
                if v and result["forward_per"] is None:
                    result["forward_per"] = v
            elif label == "PBR":
                v = _parse_num(value_text)
                if v and result["forward_pbr"] is None:
                    result["forward_pbr"] = v
            elif "DPS" in label:
                v = _parse_num(value_text)
                if v:
                    result["dividend_est"] = v
