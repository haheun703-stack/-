"""DART OpenAPI 어댑터 — 금융감독원 전자공시 재무제표 연동

클린 아키텍처 Adapter 계층:
- FundamentalDataPort 인터페이스 구현
- DART API로 매출, 영업이익, 순이익, PER, ROE 등 조회
- 캐싱으로 API 호출 최소화 (일일 10,000건 제한)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# DART API 베이스 URL
DART_BASE_URL = "https://opendart.fss.or.kr/api"

# 보고서 코드
REPRT_CODES = {
    "annual": "11011",   # 사업보고서
    "q3": "11014",       # 3분기보고서
    "half": "11012",     # 반기보고서
    "q1": "11013",       # 1분기보고서
}

# 최신 보고서 순서 (가장 최근 → 과거)
REPRT_ORDER = ["11014", "11012", "11013", "11011"]


class DartAdapter:
    """DART OpenAPI 재무 데이터 어댑터"""

    def __init__(self, api_key: str | None = None, cache_dir: str = "data/dart_cache"):
        self.api_key = api_key or os.getenv("DART_API_KEY", "")
        if not self.api_key:
            logger.warning("DART_API_KEY 미설정 — DART 재무 데이터 사용 불가")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 고유번호 매핑 (ticker → corp_code)
        self._corp_code_map: dict[str, str] = {}
        self._load_corp_codes()

        # 메모리 캐시 (세션 내 중복 호출 방지)
        self._finstate_cache: dict[str, pd.DataFrame] = {}

        # API 호출 카운터 (일일 제한 관리)
        self._api_calls = 0

    @property
    def is_available(self) -> bool:
        """DART API 사용 가능 여부"""
        return bool(self.api_key)

    # ──────────────────────────────────────────────
    # 고유번호 매핑 (ticker ↔ corp_code)
    # ──────────────────────────────────────────────

    def _load_corp_codes(self):
        """고유번호 매핑 로드 (캐시 파일 또는 API)"""
        cache_file = self.cache_dir / "corp_codes.csv"

        if cache_file.exists():
            df = pd.read_csv(cache_file, dtype=str)
            self._corp_code_map = dict(
                zip(df["stock_code"].str.zfill(6), df["corp_code"].str.zfill(8))
            )
            logger.info(f"DART 고유번호 로드: {len(self._corp_code_map)}개")
            return

        if not self.is_available:
            return

        self._download_corp_codes()

    def _download_corp_codes(self):
        """DART API에서 고유번호 파일 다운로드 및 파싱"""
        import io
        import zipfile

        url = f"{DART_BASE_URL}/corpCode.xml"
        params = {"crtfc_key": self.api_key}

        try:
            resp = requests.get(url, params=params, timeout=30)
            self._api_calls += 1

            if resp.status_code != 200:
                logger.error(f"DART 고유번호 다운로드 실패: {resp.status_code}")
                return

            # ZIP 파일 해제 → XML 파싱
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                xml_name = zf.namelist()[0]
                with zf.open(xml_name) as f:
                    df = pd.read_xml(f)

            # stock_code가 있는 상장사만 필터
            df = df[df["stock_code"].notna() & (df["stock_code"] != " ")]
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
            df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)

            # 캐시 저장
            df[["corp_code", "corp_name", "stock_code"]].to_csv(
                self.cache_dir / "corp_codes.csv", index=False
            )
            self._corp_code_map = dict(zip(df["stock_code"], df["corp_code"]))
            logger.info(f"DART 고유번호 다운로드 완료: {len(self._corp_code_map)}개 상장사")

        except Exception as e:
            logger.error(f"DART 고유번호 다운로드 오류: {e}")

    def get_corp_code(self, ticker: str) -> str | None:
        """종목코드(6자리) → DART 고유번호(8자리) 변환"""
        return self._corp_code_map.get(ticker.zfill(6))

    # ──────────────────────────────────────────────
    # 재무제표 조회
    # ──────────────────────────────────────────────

    def fetch_financial_statement(
        self,
        ticker: str,
        year: int,
        reprt_code: str = "11011",
    ) -> pd.DataFrame | None:
        """
        단일회사 주요계정 재무제표 조회.

        Args:
            ticker: 종목코드 (예: "005930")
            year: 사업연도 (예: 2024)
            reprt_code: 보고서코드 (11011=연간, 11012=반기, 11013=Q1, 11014=Q3)

        Returns:
            DataFrame with columns: account_nm, thstrm_amount, ...
        """
        if not self.is_available:
            return None

        cache_key = f"{ticker}_{year}_{reprt_code}"
        if cache_key in self._finstate_cache:
            return self._finstate_cache[cache_key]

        # 로컬 캐시 파일 확인
        cache_file = self.cache_dir / f"finstate_{cache_key}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            self._finstate_cache[cache_key] = df
            return df

        corp_code = self.get_corp_code(ticker)
        if not corp_code:
            logger.debug(f"DART 고유번호 없음: {ticker}")
            return None

        url = f"{DART_BASE_URL}/fnlttSinglAcnt.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
        }

        try:
            time.sleep(0.1)  # Rate limiting
            resp = requests.get(url, params=params, timeout=15)
            self._api_calls += 1
            data = resp.json()

            if data.get("status") != "000":
                logger.debug(
                    f"DART API 응답: {data.get('status')} - {data.get('message')} "
                    f"(ticker={ticker}, year={year}, reprt={reprt_code})"
                )
                return None

            df = pd.DataFrame(data["list"])

            # 캐시 저장
            df.to_csv(cache_file, index=False)
            self._finstate_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"DART 재무제표 조회 오류: {ticker} - {e}")
            return None

    def fetch_latest_financial(self, ticker: str, year: int) -> pd.DataFrame | None:
        """가장 최신 보고서를 찾아서 반환 (Q3 → 반기 → Q1 → 연간 순)"""
        for reprt_code in REPRT_ORDER:
            df = self.fetch_financial_statement(ticker, year, reprt_code)
            if df is not None and len(df) > 0:
                return df

        # 올해 없으면 전년도 사업보고서
        df = self.fetch_financial_statement(ticker, year - 1, "11011")
        return df

    # ──────────────────────────────────────────────
    # 핵심 재무 지표 추출
    # ──────────────────────────────────────────────

    def get_revenue(self, ticker: str, year: int) -> float | None:
        """매출액(원) 조회"""
        df = self.fetch_latest_financial(ticker, year)
        return self._extract_account(df, "매출액")

    def get_operating_income(self, ticker: str, year: int) -> float | None:
        """영업이익(원) 조회"""
        df = self.fetch_latest_financial(ticker, year)
        return self._extract_account(df, "영업이익")

    def get_net_income(self, ticker: str, year: int) -> float | None:
        """당기순이익(원) 조회"""
        df = self.fetch_latest_financial(ticker, year)
        return self._extract_account(df, "당기순이익")

    def get_key_financials(self, ticker: str, year: int) -> dict:
        """
        핵심 재무지표 일괄 조회.

        Returns:
            {
                "revenue": 매출액(억원),
                "operating_income": 영업이익(억원),
                "net_income": 당기순이익(억원),
                "operating_margin": 영업이익률(%),
                "profitable": 영업이익 > 0 여부,
                "reprt_code": 보고서 코드,
                "year": 사업연도,
            }
        """
        result = {
            "revenue": None,
            "operating_income": None,
            "net_income": None,
            "operating_margin": None,
            "profitable": None,
            "reprt_code": None,
            "year": year,
        }

        if not self.is_available:
            return result

        df = self.fetch_latest_financial(ticker, year)
        if df is None:
            return result

        revenue = self._extract_account(df, "매출액")
        op_income = self._extract_account(df, "영업이익")
        net_income = self._extract_account(df, "당기순이익")

        # 원 → 억원 변환
        if revenue is not None:
            result["revenue"] = round(revenue / 1e8, 1)
        if op_income is not None:
            result["operating_income"] = round(op_income / 1e8, 1)
        if net_income is not None:
            result["net_income"] = round(net_income / 1e8, 1)

        # 영업이익률
        if revenue and op_income and revenue > 0:
            result["operating_margin"] = round(op_income / revenue * 100, 2)

        # 흑자 여부
        if op_income is not None:
            result["profitable"] = op_income > 0

        # 보고서 코드
        if "reprt_code" in df.columns:
            result["reprt_code"] = df["reprt_code"].iloc[0]

        return result

    def check_consecutive_profit(self, ticker: str, year: int, quarters: int = 2) -> bool:
        """
        연속 N분기 흑자 확인.

        최신 보고서부터 역순으로 영업이익 > 0인지 검증.
        """
        if not self.is_available:
            return True  # API 없으면 통과 (기존 동작 유지)

        profit_count = 0
        check_year = year
        for reprt_code in REPRT_ORDER:
            df = self.fetch_financial_statement(ticker, check_year, reprt_code)
            if df is None:
                continue

            op = self._extract_account(df, "영업이익")
            if op is not None and op > 0:
                profit_count += 1
                if profit_count >= quarters:
                    return True
            elif op is not None:
                return False  # 적자 분기 발견

        # 올해 분기 부족 시 전년도 확인
        if profit_count < quarters:
            for reprt_code in REPRT_ORDER:
                df = self.fetch_financial_statement(ticker, check_year - 1, reprt_code)
                if df is None:
                    continue
                op = self._extract_account(df, "영업이익")
                if op is not None and op > 0:
                    profit_count += 1
                    if profit_count >= quarters:
                        return True
                elif op is not None:
                    return False

        return profit_count >= quarters

    # ──────────────────────────────────────────────
    # L3 QoQ 턴어라운드 감지
    # ──────────────────────────────────────────────

    def get_qoq_turnaround(self, ticker: str, year: int) -> dict:
        """분기별 영업이익 비교 → 적자→흑자 / 감익→증익 감지

        Returns:
            {
                "turnaround": bool,      # 적자→흑자 전환
                "qoq_oi_growth": float,  # 분기 영업이익 성장률 (%)
                "current_oi": float,     # 현재 영업이익 (억원)
                "prev_oi": float,        # 직전 분기 영업이익 (억원)
            }
        """
        result = {
            "turnaround": False,
            "qoq_oi_growth": 0.0,
            "current_oi": None,
            "prev_oi": None,
        }

        if not self.is_available:
            return result

        # 최신 2개 분기 영업이익 비교
        quarters = [
            (year, "11014"),    # Q3
            (year, "11012"),    # 반기
            (year, "11013"),    # Q1
            (year - 1, "11011"),  # 전년 연간
            (year - 1, "11014"),  # 전년 Q3
        ]

        oi_list = []  # (year, reprt_code, operating_income)
        for yr, rc in quarters:
            df = self.fetch_financial_statement(ticker, yr, rc)
            oi = self._extract_account(df, "영업이익") if df is not None else None
            if oi is not None:
                oi_list.append((yr, rc, oi))
            if len(oi_list) >= 2:
                break

        if len(oi_list) < 2:
            return result

        current_oi = oi_list[0][2]
        prev_oi = oi_list[1][2]

        result["current_oi"] = round(current_oi / 1e8, 1)
        result["prev_oi"] = round(prev_oi / 1e8, 1)

        # 적자→흑자 전환
        if prev_oi <= 0 and current_oi > 0:
            result["turnaround"] = True

        # QoQ 성장률
        if prev_oi != 0:
            result["qoq_oi_growth"] = round(
                (current_oi - prev_oi) / abs(prev_oi) * 100, 2
            )

        return result

    # ──────────────────────────────────────────────
    # 다중회사 일괄 조회 (스캐너용)
    # ──────────────────────────────────────────────

    def fetch_multi_financials(
        self,
        tickers: list[str],
        year: int,
        reprt_code: str = "11011",
    ) -> pd.DataFrame | None:
        """
        다중회사 주요계정 일괄 조회 (최대 100개).
        스캐너에서 전종목 필터링 시 효율적.
        """
        if not self.is_available or not tickers:
            return None

        corp_codes = []
        for t in tickers[:100]:
            cc = self.get_corp_code(t)
            if cc:
                corp_codes.append(cc)

        if not corp_codes:
            return None

        url = f"{DART_BASE_URL}/fnlttMultiAcnt.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": ",".join(corp_codes),
            "bsns_year": str(year),
            "reprt_code": reprt_code,
        }

        try:
            time.sleep(0.1)
            resp = requests.get(url, params=params, timeout=30)
            self._api_calls += 1
            data = resp.json()

            if data.get("status") != "000":
                logger.debug(f"DART 다중회사 조회: {data.get('message')}")
                return None

            return pd.DataFrame(data["list"])

        except Exception as e:
            logger.error(f"DART 다중회사 조회 오류: {e}")
            return None

    # ──────────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────────

    def _extract_account(self, df: pd.DataFrame | None, account_name: str) -> float | None:
        """재무제표에서 특정 계정과목 금액 추출 (연결 우선)"""
        if df is None or len(df) == 0:
            return None

        # 연결재무제표 우선, 없으면 개별
        for fs_div in ["CFS", "OFS"]:
            if "fs_div" in df.columns:
                subset = df[
                    (df["account_nm"] == account_name) & (df["fs_div"] == fs_div)
                ]
            else:
                subset = df[df["account_nm"] == account_name]

            if len(subset) > 0:
                amt_str = subset.iloc[0].get("thstrm_amount", "")
                return self._parse_amount(amt_str)

        return None

    @staticmethod
    def _parse_amount(value) -> float | None:
        """금액 문자열 → float 변환 (쉼표 제거)"""
        if pd.isna(value) or value == "" or value is None:
            return None
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return None

    def get_api_calls_count(self) -> int:
        """현재 세션 API 호출 횟수"""
        return self._api_calls
