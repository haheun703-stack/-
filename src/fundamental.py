"""
Step 3: fundamental.py — 재무 데이터 관리

v6.5: DART OpenAPI 연동으로 실제 매출/영업이익 데이터 사용.
DART_API_KEY가 없으면 기존 sector_map 기반 fallback 유지.

백테스트에서는 Look-ahead bias 방지를 위해:
- Forward PER → pykrx Trailing PER로 대체 (과거 시점 Forward PER 확보 어려움)
- EPS 리비전 → EPS 3개월 변화율로 근사
- 매출액/영업이익 → DART API (실시간) 또는 가장 최근 공시 기준

실시간에서는 DART API로 분기별 재무제표 직접 조회.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FundamentalEngine:
    """재무 데이터 로딩 + 밸류에이션 점수 계산"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.sector_per = self.config.get("sector_per", {})
        self.universe_dir = Path("data/universe")

        # 종목-업종 매핑 로드
        self.sector_map = self._load_sector_map()

        # DART 재무 캐시 (fundamentals_all.csv)
        self._fund_cache = self._load_fundamentals_cache()

        # DART 어댑터 (Lazy 초기화)
        self._dart = None
        self._dart_initialized = False

    def _load_fundamentals_cache(self) -> dict:
        """
        data/dart_cache/fundamentals_all.csv 로딩.
        ticker → {revenue_억, op_income_억, net_income_억, op_margin_pct, profitable}
        """
        cache_file = Path("data/dart_cache/fundamentals_all.csv")
        if not cache_file.exists():
            return {}

        try:
            df = pd.read_csv(cache_file, dtype={"ticker": str})
            df["ticker"] = df["ticker"].str.zfill(6)
            cache = {}
            for _, row in df.iterrows():
                cache[row["ticker"]] = {
                    "revenue": row.get("revenue_억"),
                    "operating_income": row.get("op_income_억"),
                    "net_income": row.get("net_income_억"),
                    "operating_margin": row.get("op_margin_pct"),
                    "profitable": row.get("profitable"),
                }
            logger.info(f"DART 재무 캐시 로드: {len(cache)}종목 (fundamentals_all.csv)")
            return cache
        except Exception as e:
            logger.warning(f"DART 재무 캐시 로드 실패: {e}")
            return {}

    @property
    def dart(self):
        """DART 어댑터 Lazy 로딩 (API 키 없으면 None)"""
        if not self._dart_initialized:
            self._dart_initialized = True
            try:
                from src.adapters.dart_adapter import DartAdapter
                adapter = DartAdapter()
                if adapter.is_available:
                    self._dart = adapter
                    logger.info("DART API 어댑터 초기화 완료")
                else:
                    logger.info("DART_API_KEY 미설정 — sector_map fallback 사용")
            except Exception as e:
                logger.warning(f"DART 어댑터 로드 실패: {e}")
        return self._dart

    def _load_sector_map(self) -> dict:
        """ticker → sector 매핑"""
        sector_file = self.universe_dir / "sector_map.csv"
        if sector_file.exists():
            df = pd.read_csv(sector_file)
            if "sector" in df.columns:
                return dict(zip(df["ticker"].astype(str).str.zfill(6), df["sector"]))
        return {}

    def get_sector(self, ticker: str) -> str:
        """종목의 업종 반환"""
        return self.sector_map.get(ticker, "기타")

    def get_sector_avg_per(self, ticker: str) -> float:
        """업종 평균 PER 반환"""
        sector = self.get_sector(ticker)
        return self.sector_per.get(sector, 11.0)  # 기본값 11.0 (KOSPI 평균)

    def calc_trailing_value_score(self, current_per: float, sector_avg_per: float) -> float:
        """
        Trailing PER 기반 밸류 점수 (Forward PER 대용)
        value_ratio = current_per / sector_avg_per
        """
        if pd.isna(current_per) or current_per <= 0 or pd.isna(sector_avg_per):
            return 0.3  # 데이터 없으면 중립

        ratio = current_per / sector_avg_per

        ranges = self.config["strategy"]["forward_value_ranges"]
        if ratio <= ranges["deep_discount"][1]:
            return 1.0
        elif ratio <= ranges["discount"][1]:
            return 1.0
        elif ratio <= ranges["fair"][1]:
            return 0.5
        elif ratio <= ranges["premium"][1]:
            return 0.2
        else:
            return 0.0

    def calc_eps_revision_score(self, df: pd.DataFrame, idx: int,
                                lookback_days: int = 60) -> float:
        """
        EPS 리비전 점수 (3개월 전 대비 현재 EPS 변화율)

        상향(>5%):  1.0
        유지(-5~5%): 0.7
        하향(<-5%): 0.3
        적자 전환:   0.0
        """
        if "fund_EPS" not in df.columns:
            return 0.5  # 데이터 없으면 중립

        current_eps = df["fund_EPS"].iloc[idx]
        prev_idx = max(0, idx - lookback_days)
        prev_eps = df["fund_EPS"].iloc[prev_idx]

        if pd.isna(current_eps) or pd.isna(prev_eps):
            return 0.5

        # 적자 전환
        if current_eps <= 0 and prev_eps > 0:
            return 0.0
        if current_eps <= 0:
            return 0.1

        # 리비전 계산
        if prev_eps != 0:
            revision_pct = (current_eps - prev_eps) / abs(prev_eps) * 100
        else:
            revision_pct = 0

        if revision_pct > 5:
            return 1.0   # 상향
        elif revision_pct >= -5:
            return 0.7   # 유지
        else:
            return 0.3   # 하향

    def calc_combined_value_score(self, per_score: float, eps_score: float) -> float:
        """밸류 종합 점수 = PER 비율 × 0.6 + EPS 리비전 × 0.4"""
        return per_score * 0.6 + eps_score * 0.4

    # ──────────────────────────────────────────────
    # Pre-screening 필터 (DART 연동)
    # ──────────────────────────────────────────────

    def check_revenue_filter(self, ticker: str,
                              min_revenue_억: float = 1000) -> bool:
        """
        매출 필터: 매출 >= min_revenue_억 (기본 1,000억원)

        우선순위:
        1. fundamentals_all.csv 캐시 (API 호출 없이 즉시)
        2. DART API 실시간 조회
        3. Fallback: sector_map에 있으면 통과 (기존 동작)
        """
        # 1. CSV 캐시에서 조회 (가장 빠름)
        cached = self._fund_cache.get(ticker.zfill(6))
        if cached and pd.notna(cached.get("revenue")):
            revenue = cached["revenue"]
            passed = revenue >= min_revenue_억
            if not passed:
                logger.debug(
                    f"{ticker}: 매출 {revenue:.0f}억 < {min_revenue_억}억 → 필터 차단 (캐시)"
                )
            return passed

        # 2. DART API 실시간
        if self.dart is not None:
            year = datetime.now().year
            financials = self.dart.get_key_financials(ticker, year)
            revenue = financials.get("revenue")
            if revenue is not None:
                passed = revenue >= min_revenue_억
                if not passed:
                    logger.debug(
                        f"{ticker}: 매출 {revenue:.0f}억 < {min_revenue_억}억 → 필터 차단"
                    )
                return passed

        # 3. Fallback: sector_map 기반
        if ticker in self.sector_map:
            return True
        return False

    def check_profitability(self, df: pd.DataFrame, idx: int,
                            ticker: str | None = None) -> bool:
        """
        수익성 필터: 영업이익 > 0

        우선순위:
        1. CSV 캐시 profitable 필드
        2. Fallback: EPS > 0으로 근사 (기존 백테스트 동작)
        """
        # 1. CSV 캐시 (ticker가 넘어온 경우)
        if ticker:
            cached = self._fund_cache.get(ticker.zfill(6))
            if cached and cached.get("profitable") is not None:
                return bool(cached["profitable"])

        # 2. Fallback: EPS 기반 (백테스트 및 DART 없을 때)
        if "fund_EPS" not in df.columns:
            return True

        current_eps = df["fund_EPS"].iloc[idx]
        prev_eps = df["fund_EPS"].iloc[max(0, idx - 60)]

        if pd.isna(current_eps) or pd.isna(prev_eps):
            return True

        return current_eps > 0 and prev_eps > 0

    def check_profitability_dart(self, ticker: str, quarters: int = 2) -> bool | None:
        """
        DART API 기반 연속 흑자 확인 (실시간 스캐너용).

        Returns:
            True/False: 판정 결과
            None: DART 데이터 없음 (fallback 필요)
        """
        if self.dart is None:
            return None

        year = datetime.now().year
        return self.dart.check_consecutive_profit(ticker, year, quarters)

    # ──────────────────────────────────────────────
    # DART 재무 데이터 직접 접근 (스캐너/리포트용)
    # ──────────────────────────────────────────────

    def get_financials(self, ticker: str, year: int | None = None) -> dict:
        """
        종목 핵심 재무지표 조회.

        우선순위: CSV 캐시 → DART API → fallback

        Returns:
            {
                "revenue": 매출(억원),
                "operating_income": 영업이익(억원),
                "net_income": 순이익(억원),
                "operating_margin": 영업이익률(%),
                "profitable": 흑자 여부,
                "source": "cache" | "dart" | "fallback",
            }
        """
        if year is None:
            year = datetime.now().year

        # 1. CSV 캐시
        cached = self._fund_cache.get(ticker.zfill(6))
        if cached and pd.notna(cached.get("revenue")):
            return {
                "revenue": cached["revenue"],
                "operating_income": cached["operating_income"],
                "net_income": cached["net_income"],
                "operating_margin": cached["operating_margin"],
                "profitable": cached["profitable"],
                "source": "cache",
            }

        # 2. DART API 실시간
        if self.dart is not None:
            result = self.dart.get_key_financials(ticker, year)
            if result.get("revenue") is not None:
                result["source"] = "dart"
                return result

        # 3. Fallback
        return {
            "revenue": None,
            "operating_income": None,
            "net_income": None,
            "operating_margin": None,
            "profitable": ticker in self.sector_map,
            "source": "fallback",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = FundamentalEngine()
    print(f"업종 PER 맵: {engine.sector_per}")
    print(f"종목-업종 매핑: {len(engine.sector_map)}종목")
    print(f"DART API: {'연결됨' if engine.dart else '미연결'}")
