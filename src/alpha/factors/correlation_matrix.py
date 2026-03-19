"""상관관계 매트릭스 — L3 SIZE 엔진용 (STEP 5-1)

84종목 60일 롤링 상관계수 행렬 계산.
동일 포트폴리오 내 고상관 종목 감지 → 사이즈 감산에 활용.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class CorrelationMatrix:
    """포트폴리오 상관관계 매트릭스"""

    def __init__(self, lookback: int = 60):
        self._lookback = lookback
        self._corr_matrix: pd.DataFrame | None = None
        self._ret_df: pd.DataFrame | None = None

    def build(self) -> pd.DataFrame:
        """data/processed/*.parquet에서 상관관계 행렬 구축."""
        parquet_dir = PROJECT_ROOT / "data" / "processed"
        files = sorted(parquet_dir.glob("*.parquet"))

        all_ret: dict[str, pd.Series] = {}
        for f in files:
            ticker = f.stem
            if not ticker[-1].isdigit() or ticker[-1] == "5":
                continue
            try:
                df = pd.read_parquet(f, columns=["ret1"])
                if df.empty or len(df) < self._lookback:
                    continue
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                all_ret[ticker] = df["ret1"]
            except Exception:
                continue

        self._ret_df = pd.DataFrame(all_ret)

        # 최근 lookback일만 사용
        recent = self._ret_df.tail(self._lookback)

        # 최소 30일 데이터 있는 종목만
        valid_cols = recent.columns[recent.notna().sum() >= 30]
        recent = recent[valid_cols]

        self._corr_matrix = recent.corr()
        logger.info(
            "상관관계 행렬 구축: %d종목, %d일",
            len(valid_cols), len(recent),
        )

        return self._corr_matrix

    def get_correlation(self, ticker_a: str, ticker_b: str) -> float:
        """두 종목 간 상관계수 반환."""
        if self._corr_matrix is None:
            self.build()
        if (
            ticker_a not in self._corr_matrix.index
            or ticker_b not in self._corr_matrix.columns
        ):
            return 0.0  # 데이터 없으면 독립 가정
        val = self._corr_matrix.loc[ticker_a, ticker_b]
        return float(val) if not np.isnan(val) else 0.0

    def get_portfolio_avg_corr(self, tickers: list[str]) -> float:
        """포트폴리오 내 평균 상관계수."""
        if len(tickers) < 2:
            return 0.0
        if self._corr_matrix is None:
            self.build()

        corrs = []
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                corrs.append(self.get_correlation(t1, t2))
        return float(np.mean(corrs)) if corrs else 0.0

    def get_max_corr_with_portfolio(
        self, candidate: str, portfolio: list[str]
    ) -> tuple[float, str]:
        """후보 종목과 기존 포트폴리오 종목들 간 최대 상관계수.

        Returns:
            (max_corr, most_correlated_ticker)
        """
        if not portfolio:
            return 0.0, ""
        if self._corr_matrix is None:
            self.build()

        max_corr = -1.0
        max_ticker = ""
        for t in portfolio:
            c = self.get_correlation(candidate, t)
            if c > max_corr:
                max_corr = c
                max_ticker = t

        return max_corr, max_ticker

    def save(self, path: str | None = None):
        """상관관계 행렬 JSON 저장."""
        if self._corr_matrix is None:
            self.build()

        if path is None:
            path = str(
                PROJECT_ROOT / "data" / "v2_migration" / "correlation_matrix.json"
            )

        # 상위 삼각 행렬만 저장 (용량 절감)
        result = {}
        tickers = list(self._corr_matrix.index)
        for i, t1 in enumerate(tickers):
            pairs = {}
            for t2 in tickers[i + 1:]:
                val = self._corr_matrix.loc[t1, t2]
                if not np.isnan(val) and abs(val) > 0.3:  # 약한 상관만 저장
                    pairs[t2] = round(float(val), 4)
            if pairs:
                result[t1] = pairs

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"n_stocks": len(tickers), "lookback": self._lookback, "pairs": result},
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("상관관계 저장: %s (%d 고상관 페어)", path, sum(len(v) for v in result.values()))
