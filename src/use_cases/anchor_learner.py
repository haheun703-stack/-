"""
v5.0 AnchorLearner — Sci-CoE 앵커 학습기

Sci-CoE 논문의 Stage 1 (Anchored Learning) 개념 적용:
  - 백테스트 거래 결과에서 성공/실패 사례 추출
  - 특성 벡터 중심점(centroid) 계산
  - JSON 직렬화로 앵커 DB 저장/로드

"400개 라벨만으로도 부트스트랩 가능" — Sci-CoE Table 3

클린 아키텍처: entities만 import.
"""

import json
import logging
import os
from typing import Optional

from src.entities.consensus_models import AnchorCase, AnchorDatabase

logger = logging.getLogger(__name__)


class AnchorLearner:
    """백테스트 거래 결과에서 앵커 사례를 학습"""

    def __init__(
        self,
        db_path: str = "data/anchors.json",
        min_success_pnl_pct: float = 3.0,
        min_failure_pnl_pct: float = -2.0,
    ):
        self.db_path = db_path
        self.min_success_pnl_pct = min_success_pnl_pct
        self.min_failure_pnl_pct = min_failure_pnl_pct
        self.db = AnchorDatabase()

    def learn_from_trades(self, trades: list) -> AnchorDatabase:
        """
        거래 결과 리스트에서 앵커 사례를 추출.

        Args:
            trades: Trade 객체 또는 dict 리스트.
                    필수 필드: ticker, entry_date, pnl_pct, trigger_type, grade
                    선택 필드: features (레이어별 confidence dict)

        Returns:
            업데이트된 AnchorDatabase
        """
        for trade in trades:
            if isinstance(trade, dict):
                pnl = trade.get("pnl_pct", 0)
                ticker = trade.get("ticker", "")
                date = str(trade.get("entry_date", ""))
                trigger = trade.get("trigger_type", "")
                grade = trade.get("grade", "")
                features = trade.get("features", {})
            else:
                pnl = getattr(trade, "pnl_pct", 0)
                ticker = getattr(trade, "ticker", "")
                date = str(getattr(trade, "entry_date", ""))
                trigger = getattr(trade, "trigger_type", "")
                grade = getattr(trade, "grade", "")
                features = getattr(trade, "features", {})

            if pnl >= self.min_success_pnl_pct:
                outcome = "success"
            elif pnl <= self.min_failure_pnl_pct:
                outcome = "failure"
            else:
                continue  # 중립 구간 → 앵커에 포함하지 않음

            case = AnchorCase(
                ticker=ticker,
                date=date,
                outcome=outcome,
                pnl_pct=pnl,
                trigger_type=trigger,
                grade=grade,
                features=features,
            )
            self.db.cases.append(case)

        self._calc_centroids()
        return self.db

    def _calc_centroids(self) -> None:
        """성공/실패 사례의 특성 벡터 중심점 계산"""
        success_features = []
        failure_features = []

        for case in self.db.cases:
            if not case.features:
                continue
            if case.outcome == "success":
                success_features.append(case.features)
            elif case.outcome == "failure":
                failure_features.append(case.features)

        self.db.success_centroid = self._mean_features(success_features)
        self.db.failure_centroid = self._mean_features(failure_features)

    @staticmethod
    def _mean_features(feature_list: list) -> dict:
        """특성 딕셔너리 리스트의 평균 계산"""
        if not feature_list:
            return {}

        all_keys = set()
        for f in feature_list:
            all_keys.update(f.keys())

        centroid = {}
        for key in all_keys:
            values = []
            for f in feature_list:
                val = f.get(key)
                if val is not None and isinstance(val, (int, float)):
                    values.append(float(val))
            if values:
                centroid[key] = round(sum(values) / len(values), 4)

        return centroid

    def save(self) -> None:
        """앵커 DB를 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        data = {
            "cases": [
                {
                    "ticker": c.ticker,
                    "date": c.date,
                    "outcome": c.outcome,
                    "pnl_pct": c.pnl_pct,
                    "trigger_type": c.trigger_type,
                    "grade": c.grade,
                    "features": c.features,
                }
                for c in self.db.cases
            ],
            "success_centroid": self.db.success_centroid,
            "failure_centroid": self.db.failure_centroid,
        }

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "앵커 DB 저장: %d cases (%s)",
            len(self.db.cases), self.db_path,
        )

    def load(self) -> Optional[AnchorDatabase]:
        """앵커 DB를 JSON 파일에서 로드"""
        if not os.path.exists(self.db_path):
            logger.info("앵커 DB 파일 없음: %s", self.db_path)
            return None

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.db = AnchorDatabase(
                cases=[
                    AnchorCase(**c) for c in data.get("cases", [])
                ],
                success_centroid=data.get("success_centroid", {}),
                failure_centroid=data.get("failure_centroid", {}),
            )

            logger.info(
                "앵커 DB 로드: %d cases (%s)",
                len(self.db.cases), self.db_path,
            )
            return self.db

        except Exception as e:
            logger.warning("앵커 DB 로드 실패: %s", e)
            return None
