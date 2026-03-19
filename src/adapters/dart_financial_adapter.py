"""Alpha Engine V2 — DART 확장 재무 파이프라인 (STEP 3-1)

퀄리티(Q)+밸류(V) 팩터용 재무 데이터 수집 어댑터.
기존 DartAdapter를 활용하여 추가 재무 항목을 수집한다.

수집 항목:
  BS: 자산총계, 부채총계, 자본총계 → ROE, 부채비율
  IS: 매출액, 영업이익, 당기순이익 (기존 fnlttSinglAcnt)
  CF: 영업CF, CAPEX, 배당금 → Accruals, FCF, 배당성향

사용법:
  adapter = DartFinancialAdapter()
  results = adapter.collect_universe(tickers)
  adapter.save(results, "data/v2_migration/financial_quarterly.json")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.adapters.dart_adapter import DART_BASE_URL, DartAdapter

logger = logging.getLogger(__name__)

# 8분기 조회 정의 (최신 → 과거)
# (year, reprt_code, label, annualize_factor)
# annualize_factor: cumulative→연간환산 배수 (Q1×4, H1×2, 9M×4/3, Annual×1)
QUARTER_REPORTS = [
    (2025, "11014", "2025Q3", 4 / 3),
    (2025, "11012", "2025Q2", 2),
    (2025, "11013", "2025Q1", 4),
    (2024, "11011", "2024Q4", 1),
    (2024, "11014", "2024Q3", 4 / 3),
    (2024, "11012", "2024Q2", 2),
    (2024, "11013", "2024Q1", 4),
    (2023, "11011", "2023Q4", 1),
]

# CF 항목 키워드 매핑 (DART 계정과목명 → 내부 키)
_CF_KEYWORDS = {
    "operating_cf": [("영업활동", "현금흐름")],
    "capex_tangible": [("유형자산", "취득")],
    "capex_intangible": [("무형자산", "취득")],
    "dividends_paid": [("배당금", "지급"), ("배당금", "납부")],
}


def _parse_amount(value) -> float | None:
    """금액 문자열 → float (쉼표 제거)"""
    if pd.isna(value) or value == "" or value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


class DartFinancialAdapter:
    """V2 퀄리티/밸류 팩터용 확장 DART 재무 데이터 어댑터"""

    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv()

        self.dart = DartAdapter()
        self.api_key = self.dart.api_key
        self.cache_dir = Path("data/dart_cache/v2")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_calls = 0

    @property
    def is_available(self) -> bool:
        return self.dart.is_available

    # ═══════════════════════════════════════════════
    # BS 데이터 수집 (fnlttMultiAcnt — 100종목 일괄)
    # ═══════════════════════════════════════════════

    def collect_bs_all(
        self,
        tickers: list[str],
        periods: list[tuple] | None = None,
    ) -> dict[str, dict]:
        """전체 종목 × 전체 분기 BS+IS 데이터 수집.

        Returns:
            {ticker: {period_label: {equity, total_assets, total_debt,
                                     net_income_cum, op_income_cum}}}
        """
        if periods is None:
            periods = QUARTER_REPORTS

        all_data: dict[str, dict] = {}
        batch_size = 100

        for year, reprt_code, label, _ann in periods:
            logger.info("BS 수집: %s (year=%d, reprt=%s)", label, year, reprt_code)
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                batch_result = self._fetch_bs_batch(batch, year, reprt_code)

                for ticker, items in batch_result.items():
                    if ticker not in all_data:
                        all_data[ticker] = {}
                    all_data[ticker][label] = items

            logger.info(
                "  %s 완료 — %d종목 수집",
                label,
                sum(1 for t in all_data if label in all_data[t]),
            )

        return all_data

    def _fetch_bs_batch(
        self, tickers: list[str], year: int, reprt_code: str
    ) -> dict[str, dict]:
        """다중회사 API로 BS 항목 일괄 조회."""
        df = self.dart.fetch_multi_financials(tickers, year, reprt_code)
        if df is None or df.empty:
            return {}

        results: dict[str, dict] = {}

        # CFS(연결) 우선, OFS(개별) 폴백
        for fs_div in ["CFS", "OFS"]:
            subset = df[df["fs_div"] == fs_div] if "fs_div" in df.columns else df

            for stock_code, group in subset.groupby("stock_code"):
                ticker = str(stock_code).zfill(6)
                if ticker in results:
                    continue  # CFS 이미 있으면 스킵

                items = self._extract_bs_from_group(group)
                if items:
                    results[ticker] = items

        return results

    @staticmethod
    def _extract_bs_from_group(group: pd.DataFrame) -> dict:
        """DataFrame 그룹에서 BS+IS 항목 추출."""
        items: dict[str, float | None] = {}

        for _, row in group.iterrows():
            name = str(row.get("account_nm", ""))
            amt = _parse_amount(row.get("thstrm_amount", ""))

            if name == "자산총계":
                items["total_assets"] = amt
            elif name == "부채총계":
                items["total_debt"] = amt
            elif "자본총계" in name and "비지배" not in name:
                items["equity"] = amt
            elif "당기순이익" in name:
                items["net_income_cum"] = amt
            elif name in ("영업이익", "영업이익(손실)"):
                items["op_income_cum"] = amt
            elif name == "매출액" or name == "수익(매출액)":
                items["revenue_cum"] = amt

        return items

    # ═══════════════════════════════════════════════
    # CF 데이터 수집 (fnlttSinglAcntAll — 개별 종목)
    # ═══════════════════════════════════════════════

    def collect_cf_all(
        self,
        tickers: list[str],
        period: tuple | None = None,
    ) -> dict[str, dict]:
        """전체 종목 CF 데이터 수집 (최신 조회 가능 기간).

        Returns:
            {ticker: {period, operating_cf, capex_tangible,
                      capex_intangible, dividends_paid}}
        """
        if period is None:
            period = QUARTER_REPORTS[0]  # 최신 분기

        year, reprt_code, label, _ = period
        all_cf: dict[str, dict] = {}

        total = len(tickers)
        for idx, ticker in enumerate(tickers, 1):
            if idx % 50 == 0 or idx == total:
                logger.info("CF 수집: %d/%d (%s)", idx, total, label)

            cf = self._fetch_cf_single(ticker, year, reprt_code)
            if cf:
                all_cf[ticker] = {"period": label, **cf}

        return all_cf

    def _fetch_cf_single(
        self, ticker: str, year: int, reprt_code: str
    ) -> dict | None:
        """단일 종목 CF 데이터 조회 (fnlttSinglAcntAll.json)."""
        full_df = self._fetch_full_statement(ticker, year, reprt_code)
        if full_df is None or full_df.empty:
            return None

        # CF(현금흐름표) 섹션 필터
        if "sj_div" in full_df.columns:
            cf_df = full_df[full_df["sj_div"] == "CF"]
        else:
            return None

        if cf_df.empty:
            return None

        result: dict[str, float | None] = {
            "operating_cf": None,
            "capex_tangible": None,
            "capex_intangible": None,
            "dividends_paid": None,
        }

        for _, row in cf_df.iterrows():
            name = str(row.get("account_nm", ""))
            amt = _parse_amount(row.get("thstrm_amount", ""))

            for key, keyword_pairs in _CF_KEYWORDS.items():
                for keywords in keyword_pairs:
                    if all(kw in name for kw in keywords):
                        if result[key] is None:  # 첫 매칭만
                            result[key] = amt
                        break

        # 하나라도 수집되었으면 반환
        if any(v is not None for v in result.values()):
            return result
        return None

    def _fetch_full_statement(
        self, ticker: str, year: int, reprt_code: str
    ) -> pd.DataFrame | None:
        """fnlttSinglAcntAll.json — 전체 재무제표 조회 (캐시 포함)."""
        for fs_div in ["CFS", "OFS"]:
            cache_key = f"full_{ticker}_{year}_{reprt_code}_{fs_div}"
            cache_file = self.cache_dir / f"{cache_key}.csv"

            if cache_file.exists():
                return pd.read_csv(cache_file)

            corp_code = self.dart.get_corp_code(ticker)
            if not corp_code:
                return None

            url = f"{DART_BASE_URL}/fnlttSinglAcntAll.json"
            params = {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "fs_div": fs_div,
            }

            try:
                time.sleep(0.15)
                resp = requests.get(url, params=params, timeout=15)
                self._api_calls += 1
                data = resp.json()

                if data.get("status") == "000":
                    df = pd.DataFrame(data["list"])
                    df.to_csv(cache_file, index=False)
                    return df
                # CFS 실패 시 OFS 시도
                continue

            except Exception as e:
                logger.error("전체 재무제표 조회 오류: %s %d %s - %s", ticker, year, reprt_code, e)
                return None

        return None

    # ═══════════════════════════════════════════════
    # Quality Metrics 계산
    # ═══════════════════════════════════════════════

    def compute_quality_metrics(
        self,
        bs_data: dict[str, dict],
        cf_data: dict[str, dict],
    ) -> dict[str, dict]:
        """BS+CF 데이터에서 퀄리티 메트릭 계산.

        Returns:
            {ticker: {
                roe_annualized: [8개 분기 연환산 ROE],
                roe_mean, roe_std, roe_stability,
                debt_ratio, debt_health,
                accruals_ratio,
                dividend_payout,
                fcf, operating_cf,
            }}
        """
        results: dict[str, dict] = {}

        for ticker, quarters in bs_data.items():
            # ── Q1: ROE 안정성 (연환산) ──
            roe_list: list[float] = []
            for year, _rc, label, ann_factor in QUARTER_REPORTS:
                q = quarters.get(label, {})
                equity = q.get("equity")
                net_cum = q.get("net_income_cum")

                if equity and equity > 0 and net_cum is not None:
                    roe_ann = (net_cum / equity) * ann_factor
                    roe_list.append(round(roe_ann, 6))

            roe_mean = float(np.mean(roe_list)) if roe_list else None
            roe_std = float(np.std(roe_list)) if len(roe_list) >= 4 else None
            roe_stability = None
            if roe_mean is not None and roe_std is not None and roe_std > 1e-8:
                roe_stability = round(roe_mean / roe_std, 4)

            # ── Q2: 부채 건전성 (최신 분기) ──
            latest_q = None
            for _, _, label, _ in QUARTER_REPORTS:
                if label in quarters and quarters[label].get("total_assets"):
                    latest_q = quarters[label]
                    break

            debt_ratio = None
            if latest_q:
                total_debt = latest_q.get("total_debt")
                total_assets = latest_q.get("total_assets")
                if total_debt is not None and total_assets and total_assets > 0:
                    debt_ratio = round(total_debt / total_assets, 6)

            # ── Q3: Accruals Ratio (영업CF / 순이익) ──
            cf = cf_data.get(ticker, {})
            operating_cf = cf.get("operating_cf")
            net_income_cum = latest_q.get("net_income_cum") if latest_q else None
            accruals_ratio = None
            if (
                operating_cf is not None
                and net_income_cum is not None
                and net_income_cum > 0
            ):
                accruals_ratio = round(operating_cf / net_income_cum, 4)

            # ── Q4: 배당 성향 ──
            dividends_paid = cf.get("dividends_paid")
            dividend_payout = None
            if (
                dividends_paid is not None
                and net_income_cum is not None
                and net_income_cum > 0
            ):
                dividend_payout = round(abs(dividends_paid) / net_income_cum, 4)

            # ── V팩터용: FCF ──
            capex = 0.0
            if cf.get("capex_tangible") is not None:
                capex += abs(cf["capex_tangible"])
            if cf.get("capex_intangible") is not None:
                capex += abs(cf["capex_intangible"])
            fcf = (operating_cf - capex) if operating_cf is not None else None

            # ── EBITDA 근사 (V팩터용) ──
            # 영업이익 + 대략적 감가상각비 추정
            # 정확한 감가상각비는 전체 재무제표에서 추출 필요 → 향후 확장
            ebitda_approx = None
            op_cum = latest_q.get("op_income_cum") if latest_q else None
            if op_cum is not None and operating_cf is not None and net_income_cum is not None:
                # 조정 항목 ≈ 비현금 비용 (감가상각 포함)
                # EBITDA ≈ operating_cf + 세금 + 이자 (근사)
                # 또는 간단히: EBITDA ≈ 영업이익 × 1.15~1.25
                ebitda_approx = round(op_cum * 1.2, 0)

            results[ticker] = {
                "roe_annualized": roe_list,
                "roe_mean": round(roe_mean, 6) if roe_mean is not None else None,
                "roe_std": round(roe_std, 6) if roe_std is not None else None,
                "roe_stability": roe_stability,
                "debt_ratio": debt_ratio,
                "debt_health": round(1 - debt_ratio, 6) if debt_ratio is not None else None,
                "accruals_ratio": accruals_ratio,
                "dividend_payout": dividend_payout,
                "fcf": round(fcf, 0) if fcf is not None else None,
                "operating_cf": operating_cf,
                "ebitda_approx": ebitda_approx,
                "data_points": len(roe_list),
            }

        return results

    # ═══════════════════════════════════════════════
    # 통합 수집 + 저장
    # ═══════════════════════════════════════════════

    def collect_universe(
        self,
        tickers: list[str],
        collect_cf: bool = True,
    ) -> dict:
        """전체 유니버스 재무 데이터 수집 + 메트릭 계산.

        Args:
            tickers: 종목코드 리스트
            collect_cf: CF 데이터도 수집할지 (False면 BS만)

        Returns:
            {
                "meta": {collected_at, ticker_count, ...},
                "bs_data": {ticker: {period: {items}}},
                "cf_data": {ticker: {items}},
                "quality": {ticker: {metrics}},
            }
        """
        from datetime import datetime

        logger.info("=== V2 재무 데이터 수집 시작: %d종목 ===", len(tickers))

        # 1. BS 수집 (multi-company API, 효율적)
        bs_data = self.collect_bs_all(tickers)
        logger.info("BS 수집 완료: %d종목", len(bs_data))

        # 2. CF 수집 (개별 API, 느리지만 필수)
        cf_data: dict = {}
        if collect_cf:
            # 최신 분기 + 최신 연간 둘 다 시도
            cf_data = self.collect_cf_all(tickers, QUARTER_REPORTS[0])  # 최신 분기
            logger.info("CF 수집 완료: %d종목", len(cf_data))

            # CF 미수집 종목 → 최신 연간으로 재시도
            missing_cf = [t for t in tickers if t not in cf_data]
            if missing_cf:
                annual_period = QUARTER_REPORTS[3]  # 2024 Q4 (연간)
                cf_fallback = self.collect_cf_all(missing_cf, annual_period)
                cf_data.update(cf_fallback)
                logger.info("CF 연간 폴백: %d종목 추가", len(cf_fallback))

        # 3. Quality Metrics 계산
        quality = self.compute_quality_metrics(bs_data, cf_data)
        logger.info("Quality 메트릭 계산 완료: %d종목", len(quality))

        return {
            "meta": {
                "collected_at": datetime.now().isoformat(),
                "ticker_count": len(tickers),
                "bs_count": len(bs_data),
                "cf_count": len(cf_data),
                "quality_count": len(quality),
                "api_calls": self._api_calls,
                "quarters": [q[2] for q in QUARTER_REPORTS],
            },
            "bs_data": self._serialize_bs(bs_data),
            "cf_data": cf_data,
            "quality": quality,
        }

    @staticmethod
    def _serialize_bs(bs_data: dict) -> dict:
        """BS 데이터를 JSON 직렬화 가능하게 변환."""
        serialized = {}
        for ticker, quarters in bs_data.items():
            serialized[ticker] = {}
            for label, items in quarters.items():
                serialized[ticker][label] = {
                    k: round(v, 0) if isinstance(v, float) else v
                    for k, v in items.items()
                }
        return serialized

    @staticmethod
    def save(data: dict, output_path: str):
        """결과를 JSON 파일로 저장."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("저장 완료: %s (%.1fKB)", path, path.stat().st_size / 1024)
