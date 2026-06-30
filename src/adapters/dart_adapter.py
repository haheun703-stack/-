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

        # API 호출 카운터 (일일 제한 관리) — _load_corp_codes보다 먼저 초기화
        self._api_calls = 0

        # 메모리 캐시 (세션 내 중복 호출 방지)
        self._finstate_cache: dict[str, pd.DataFrame] = {}

        # 고유번호 매핑 (ticker → corp_code)
        self._corp_code_map: dict[str, str] = {}
        self._load_corp_codes()

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

            raw_list = data.get("list")
            if not raw_list or not isinstance(raw_list, list):
                logger.warning("DART 재무제표 응답에 list 필드 없음 (ticker=%s)", ticker)
                return None
            df = pd.DataFrame(raw_list)

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
    # 분기 영업이익 TTM-YoY 시계열 (주도주 사이클 델타 게이트)
    # ──────────────────────────────────────────────

    # 보고서코드 → (분기키, 인덱스, 분기말 월일)
    _REPRT_QMAP = {
        "11013": ("Q1", 1, (3, 31)),
        "11012": ("H1", 2, (6, 30)),   # 반기 누적
        "11014": ("9M", 3, (9, 30)),   # 3분기 누적
        "11011": ("FY", 4, (12, 31)),  # 사업보고서(연간 누적)
    }
    # 분기말 → 실제 공시(가용)일 추정 지연(일). 한국 공시기한: 분기/반기 45일, 사업보고서 90일.
    # 시계열을 '공시가용일'로 색인해야 백테스트(as_of) 시 미래실적 누설(lookahead)을 막는다.
    _AVAIL_LAG_DAYS = {1: 45, 2: 45, 3: 45, 4: 90}

    def get_op_growth_series(
        self,
        ticker: str,
        *,
        years_back: int = 4,
        end_year: int | None = None,
    ) -> pd.Series | None:
        """분기 영업이익 TTM(최근 12개월) YoY 성장률 시계열.

        주도주 사이클 진단(leader_cycle_diagnosis)의 델타 게이트 입력.
        DART 분기보고서는 '누적값'(Q1→반기→3분기→연간)이므로 단일분기로 환산한 뒤
        TTM(직전 4분기 합) YoY를 계산한다.

        TTM을 쓰는 이유:
          - 한국 분기 실적의 계절성·일회성 노이즈를 흡수(단일분기 YoY는 부호폭발·결측 다발).
          - 같은 12개월 창끼리 비교 → 델타(=YoY의 1차 차분 = 영업이익 성장의 2차 미분)가
            진짜 '성장 둔화/가속'을 의미하게 됨.

        Args:
            ticker: 종목코드(6자리).
            years_back: 조회 시작 연도 오프셋(end_year - years_back ~ end_year). TTM-YoY는
                과거 2년 단일분기가 필요하므로 기본 4(=5개년)면 최근 수분기 YoY 산출 가능.
            end_year: 마지막 조회 연도(기본=올해). 백테스트 시 과거연도 지정 가능.

        Returns:
            pd.Series(index=분기말 Timestamp, value=TTM YoY %), 오름차순.
            데이터 부족(< 2점)이거나 DART 미사용 시 None.
        """
        if not self.is_available:
            return None

        from datetime import datetime

        if end_year is None:
            end_year = datetime.now().year
        start_year = end_year - years_back

        # 1) 누적 영업이익 수집: cum[(year, qidx)] = 영업이익(원)
        cum: dict[tuple[int, int], float] = {}
        for year in range(start_year, end_year + 1):
            for reprt_code, (_qkey, qidx, _md) in self._REPRT_QMAP.items():
                df = self.fetch_financial_statement(ticker, year, reprt_code)
                oi = self._extract_account(df, "영업이익") if df is not None else None
                if oi is not None:
                    cum[(year, qidx)] = float(oi)

        if len(cum) < 6:  # 단일분기→TTM→YoY 최소 데이터 가드
            return None

        # 2) 단일분기 환산 (연내 누적 차감). 연중 결측이 끼면 그 이후 분기는 환산 불가 → 중단.
        single: dict[int, float] = {}   # 절대분기인덱스(year*4 + (q-1)) → 단일분기 영업이익
        for year in range(start_year, end_year + 1):
            prev_cum = 0.0
            ok_prev = True   # 직전 누적(Q0=0)이 신뢰 가능한가
            for qidx in (1, 2, 3, 4):
                key = (year, qidx)
                if key in cum and ok_prev:
                    abs_q = year * 4 + (qidx - 1)
                    single[abs_q] = cum[key] - prev_cum
                    prev_cum = cum[key]
                    ok_prev = True
                else:
                    ok_prev = False   # 연내 구멍 → 잔여 분기 환산 포기(차감 오염 방지)

        # 3) TTM(직전 4분기 합) → 4분기 전(전년 동분기) 대비 YoY%
        out: dict[pd.Timestamp, float] = {}
        for abs_q in single:
            window = [single.get(abs_q - k) for k in range(4)]      # 현재 포함 직전 4분기
            prev_window = [single.get(abs_q - 4 - k) for k in range(4)]  # 1년 전 같은 4분기
            if any(v is None for v in window) or any(v is None for v in prev_window):
                continue
            ttm = sum(window)
            ttm_prev = sum(prev_window)
            if ttm_prev <= 0:   # 전년 TTM 적자/0 → YoY 무의미(부호폭발) → 스킵
                continue
            year, q = divmod(abs_q, 4)
            qnum = q + 1
            mm, dd = self._REPRT_QMAP[{1: "11013", 2: "11012", 3: "11014", 4: "11011"}[qnum]][2]
            # 분기말 + 공시지연 = 가용일로 색인(point-in-time). 엔진의 as_of 필터가 미공시분 자동 제외.
            avail = pd.Timestamp(year=year, month=mm, day=dd) + pd.Timedelta(days=self._AVAIL_LAG_DAYS[qnum])
            # TTM이 0 근처(적자 턴어라운드)를 지나면 YoY가 폭발 → ±999%로 클램프(노이즈 정화,
            # 델타 게이트는 부호만 사용하므로 무영향. 실 성장률은 이 범위 내).
            yoy = (ttm - ttm_prev) / ttm_prev * 100
            out[avail] = round(max(-999.0, min(999.0, yoy)), 1)

        if len(out) < 2:
            return None
        return pd.Series(out).sort_index()

    # ──────────────────────────────────────────────
    # 공시 목록 조회 (촉매 분류용)
    # ──────────────────────────────────────────────

    def fetch_recent_disclosures(
        self,
        ticker: str,
        days: int = 30,
    ) -> list[dict]:
        """최근 N일간 공시 목록 조회.

        Args:
            ticker: 종목코드 (6자리)
            days: 조회 기간 (기본 30일)

        Returns:
            [{"title": "...", "date": "2026-02-15", "rcept_no": "...", "type": "..."}, ...]
        """
        if not self.is_available:
            return []

        corp_code = self.get_corp_code(ticker)
        if not corp_code:
            return []

        from datetime import date, timedelta

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        url = f"{DART_BASE_URL}/list.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": start_date.strftime("%Y%m%d"),
            "end_de": end_date.strftime("%Y%m%d"),
            "page_no": "1",
            "page_count": "30",
        }

        try:
            time.sleep(0.1)
            resp = requests.get(url, params=params, timeout=15)
            self._api_calls += 1
            data = resp.json()

            if data.get("status") != "000":
                logger.debug(
                    f"DART 공시 목록: {data.get('message')} (ticker={ticker})"
                )
                return []

            filings = []
            for item in data.get("list", []):
                dt = item.get("rcept_dt", "")
                filings.append({
                    "title": item.get("report_nm", ""),
                    "date": f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) == 8 else dt,
                    "rcept_no": item.get("rcept_no", ""),
                    "filer": item.get("flr_nm", ""),
                    "remark": item.get("rm", ""),
                })
            return filings

        except Exception as e:
            logger.error(f"DART 공시 목록 조회 오류: {ticker} - {e}")
            return []

    # ──────────────────────────────────────────────
    # 타법인 출자현황 (지주사 NAV — 자회사 지분율)
    # ──────────────────────────────────────────────

    def fetch_other_corp_investments(
        self,
        ticker: str,
        year: int | None = None,
        reprt_code: str = "11011",
    ) -> list[dict]:
        """타법인 출자현황(otrCprStkInvstmtSttus) 조회 — 지주사의 자회사별 기말 지분율.

        사업보고서(11011)에만 기재되는 항목. 비상장 포함 전 출자 법인이 나오므로
        호출측에서 inv_name → stock_code 매핑으로 상장사만 필터한다.

        Args:
            ticker: 지주사 종목코드(6자리)
            year: 사업연도. None이면 직전 연도(최근 사업보고서).
            reprt_code: 보고서코드(기본 11011=사업보고서)

        Returns:
            [{"inv_name": 법인명, "stake_pct": 기말지분율(%), "book_value": 장부가(원),
              "net_income": 당기순이익(원), "total_assets": 총자산(원)}, ...]
        """
        if not self.is_available:
            return []
        corp_code = self.get_corp_code(ticker)
        if not corp_code:
            return []
        if year is None:
            from datetime import date
            year = date.today().year - 1  # 최근 사업보고서 = 직전 사업연도

        url = f"{DART_BASE_URL}/otrCprInvstmntSttus.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
        }
        try:
            time.sleep(0.1)
            resp = requests.get(url, params=params, timeout=15)
            self._api_calls += 1
            data = resp.json()
            if data.get("status") != "000":
                logger.debug(
                    f"DART 타법인 출자현황: {data.get('status')} {data.get('message')} "
                    f"(ticker={ticker}, year={year})"
                )
                return []
            out: list[dict] = []
            for item in data.get("list", []):
                out.append({
                    "inv_name": (item.get("inv_prm") or "").strip(),
                    "stake_pct": self._parse_amount(item.get("trmend_blce_qota_rt")),
                    "book_value": self._parse_amount(item.get("trmend_blce_acntbk_amount")),
                    "net_income": self._parse_amount(
                        item.get("recent_bsns_year_fnnr_sttus_thstrm_ntincm")),
                    "total_assets": self._parse_amount(
                        item.get("recent_bsns_year_fnnr_sttus_tot_assets")),
                })
            return out
        except Exception as e:  # noqa: BLE001
            logger.error(f"DART 타법인 출자현황 조회 오류: {ticker} - {e}")
            return []

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

            raw_list = data.get("list")
            if not raw_list or not isinstance(raw_list, list):
                logger.warning("DART 다중회사 응답에 list 필드 없음")
                return None
            return pd.DataFrame(raw_list)

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
