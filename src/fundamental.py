"""
Step 3: fundamental.py — 재무 데이터 관리

백테스트에서는 Look-ahead bias 방지를 위해:
- Forward PER → pykrx Trailing PER로 대체 (과거 시점 Forward PER 확보 어려움)
- EPS 리비전 → EPS 3개월 변화율로 근사
- 매출액/영업이익 → 가장 최근 공시 기준

실시간(Q2)에서는 FnGuide 크롤링으로 Forward PER 사용 예정.
"""

import logging
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

    def check_revenue_filter(self, ticker: str,
                              min_revenue_억: float = 1000) -> bool:
        """
        매출 1,000억 이상 필터.
        샘플 데이터에서는 대형주 리스트에 포함된 것만 통과.
        실전에서는 FnGuide/DART 크롤링으로 실제 매출 확인.
        """
        # 대형주 리스트에 있으면 통과 (샘플 데이터용)
        if ticker in self.sector_map:
            return True
        return False

    def check_profitability(self, df: pd.DataFrame, idx: int) -> bool:
        """최근 2분기 연속 영업이익 > 0 (EPS > 0으로 근사)"""
        if "fund_EPS" not in df.columns:
            return True  # 데이터 없으면 통과 (보수적이지 않지만 백테스트 진행 위해)

        current_eps = df["fund_EPS"].iloc[idx]
        prev_eps = df["fund_EPS"].iloc[max(0, idx - 60)]  # ~3개월 전

        if pd.isna(current_eps) or pd.isna(prev_eps):
            return True

        return current_eps > 0 and prev_eps > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = FundamentalEngine()
    print(f"업종 PER 맵: {engine.sector_per}")
    print(f"종목-업종 매핑: {len(engine.sector_map)}종목")
