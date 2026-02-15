"""
⑦ Confluence Scorer — 3D 구조: N팩터 조합 적중률 분석

수학적 근거: (a+b+c)³ 전개에서 3중 교차항 6abc
  → 개별 팩터보다 3팩터 동시 충족의 예측력이 최대 6배

핵심 기능:
  1. 7개 팩터를 이진화 (True/False)
  2. C(7,3)=35개 트리플 조합의 과거 적중률 DB 구축
  3. 현재 시점에서 활성화된 트리플과 적중률 반환

의존성: parquet 일봉 데이터 (indicators.py 결과)
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 7개 기본 팩터 정의 ────────────────────────────

FACTOR_DEFS: dict[str, dict] = {
    "rsi_oversold": {
        "label": "RSI과매도",
        "condition": lambda row: row.get("rsi_14", 50) < 40,
    },
    "adx_trending": {
        "label": "ADX추세",
        "condition": lambda row: row.get("adx_14", 0) > 20,
    },
    "bb_lower": {
        "label": "BB하단",
        "condition": lambda row: row.get("bb_position", 0.5) < 0.25,
    },
    "ou_undervalued": {
        "label": "OU저평가",
        "condition": lambda row: row.get("ou_z", 0) < -1.0,
    },
    "volume_surge": {
        "label": "거래량급증",
        "condition": lambda row: row.get("volume_surge_ratio", 1.0) > 1.5,
    },
    "macd_recovering": {
        "label": "MACD회복",
        "condition": lambda row: (
            row.get("macd_histogram", 0) < 0
            and row.get("macd_histogram", 0) > row.get("macd_histogram_prev", -999)
        ),
    },
    "smart_money_buy": {
        "label": "스마트머니매수",
        "condition": lambda row: row.get("smart_z", 0) > 1.0,
    },
}


class ConfluenceScorer:
    """N-팩터 조합 적중률 분석기"""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("geometry", {}).get("confluence", {})
        self.n_factors = cfg.get("n_factors", 3)
        self.min_samples = cfg.get("min_samples", 10)
        self.min_hit_rate = cfg.get("min_hit_rate", 0.60)
        self.forward_days = cfg.get("forward_days", 5)
        self.min_return = cfg.get("min_return", 0.03)

        # 적중률 DB: {("rsi_oversold","bb_lower","ou_undervalued"): {"hits": 18, "total": 23, "rate": 0.78}}
        self.hit_db: dict[tuple, dict] = {}

    # ─── 팩터 이진화 ──────────────────────────────

    @staticmethod
    def binarize_factors(row: dict | pd.Series) -> dict[str, bool]:
        """행 데이터에서 각 팩터의 True/False를 결정"""
        if isinstance(row, pd.Series):
            row = row.to_dict()
        return {
            name: defn["condition"](row)
            for name, defn in FACTOR_DEFS.items()
        }

    # ─── 적중률 DB 구축 ──────────────────────────

    def build_hit_rate_db(self, df: pd.DataFrame) -> dict[tuple, dict]:
        """
        과거 데이터에서 모든 C(7, n_factors) 조합의 적중률을 계산.

        Parameters:
            df: indicators 컬럼이 포함된 일봉 DataFrame (index=DatetimeIndex)

        Returns:
            hit_db: {조합 튜플: {"hits": int, "total": int, "rate": float, "avg_return": float}}
        """
        if len(df) < self.forward_days + 20:
            logger.warning("데이터 부족: %d rows, 최소 %d 필요", len(df), self.forward_days + 20)
            return {}

        # forward return 계산
        df = df.copy()
        df["_fwd_return"] = df["close"].shift(-self.forward_days) / df["close"] - 1

        # macd_histogram_prev 없으면 생성
        if "macd_histogram_prev" not in df.columns and "macd_histogram" in df.columns:
            df["macd_histogram_prev"] = df["macd_histogram"].shift(1)

        # 각 행의 팩터 이진화 (마지막 forward_days는 제외)
        analysis_rows = df.iloc[:-self.forward_days] if self.forward_days > 0 else df
        factor_names = list(FACTOR_DEFS.keys())
        combos = list(combinations(factor_names, self.n_factors))

        # 팩터 매트릭스 구축 (벡터화)
        factor_matrix = {}
        for name, defn in FACTOR_DEFS.items():
            try:
                factor_matrix[name] = analysis_rows.apply(
                    lambda r, c=defn["condition"]: c(r), axis=1
                ).values
            except Exception:
                factor_matrix[name] = np.zeros(len(analysis_rows), dtype=bool)

        fwd_returns = analysis_rows["_fwd_return"].values

        hit_db = {}
        for combo in combos:
            # 모든 팩터가 True인 날 찾기
            mask = np.ones(len(analysis_rows), dtype=bool)
            for fname in combo:
                mask &= factor_matrix[fname]

            total = int(mask.sum())
            if total < self.min_samples:
                continue

            combo_returns = fwd_returns[mask]
            valid = ~np.isnan(combo_returns)
            if valid.sum() == 0:
                continue

            valid_returns = combo_returns[valid]
            hits = int((valid_returns >= self.min_return).sum())
            avg_ret = float(np.mean(valid_returns))

            hit_db[combo] = {
                "hits": hits,
                "total": int(valid.sum()),
                "rate": hits / valid.sum(),
                "avg_return": avg_ret,
            }

        self.hit_db = hit_db
        logger.info("Confluence DB 구축: %d개 조합 (min_samples=%d)", len(hit_db), self.min_samples)
        return hit_db

    # ─── 현재 시점 분석 ──────────────────────────

    def score_current(self, row: dict | pd.Series) -> dict:
        """
        현재 행에서 활성화된 트리플과 적중률 반환.

        Returns:
            {
                "active_factors": ["rsi_oversold", "bb_lower", ...],
                "active_triples": [
                    {"combo": ("rsi_oversold", "bb_lower", "ou_undervalued"),
                     "labels": "RSI과매도×BB하단×OU저평가",
                     "hit_rate": 0.78, "total": 23, "avg_return": 0.045},
                ],
                "best_hit_rate": 0.78,
                "triple_count": 2,
            }
        """
        factors = self.binarize_factors(row)
        active = [name for name, val in factors.items() if val]

        if len(active) < self.n_factors:
            return {
                "active_factors": active,
                "active_triples": [],
                "best_hit_rate": 0.0,
                "triple_count": 0,
            }

        # 활성 팩터에서 가능한 조합
        active_combos = list(combinations(sorted(active), self.n_factors))

        triples = []
        for combo in active_combos:
            key = tuple(sorted(combo))
            if key in self.hit_db:
                info = self.hit_db[key]
                if info["rate"] >= self.min_hit_rate:
                    labels = "×".join(FACTOR_DEFS[f]["label"] for f in combo)
                    triples.append({
                        "combo": combo,
                        "labels": labels,
                        "hit_rate": info["rate"],
                        "total": info["total"],
                        "avg_return": info["avg_return"],
                    })

        triples.sort(key=lambda x: x["hit_rate"], reverse=True)

        return {
            "active_factors": active,
            "active_triples": triples,
            "best_hit_rate": triples[0]["hit_rate"] if triples else 0.0,
            "triple_count": len(triples),
        }

    # ─── 프롬프트 텍스트 ─────────────────────────

    @staticmethod
    def to_prompt_text(result: dict) -> str:
        """Claude API 입력용 텍스트 변환"""
        lines = ["[3D 교차 분석]"]
        if not result["active_triples"]:
            lines.append("  활성 트리플 없음 (조건 미충족)")
            return "\n".join(lines)

        lines.append(f"  활성 트리플 {result['triple_count']}개:")
        for t in result["active_triples"][:3]:  # 상위 3개만
            lines.append(
                f"    {t['labels']} → 적중률 {t['hit_rate']:.0%} "
                f"(과거 {t['total']}건, 평균수익 {t['avg_return']:+.1%})"
            )
        lines.append(f"  최고 적중률: {result['best_hit_rate']:.0%}")
        return "\n".join(lines)
