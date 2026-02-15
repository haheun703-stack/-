"""
Master Controller — 서브시스템 가중합산 컨트롤러
================================================
7-Layer Pipeline, TGCI, 수급, 레짐, 기하학 엔진의 결과를
가중 합산하여 최종 master_score(0-100)와 판정(Action)을 산출한다.

순수 계산 클래스 — sub_scores dict만 받으며 외부 의존성 없음.
"""
from enum import Enum
from typing import Dict, Any


class Action(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    HOLD = "HOLD"


class EntryMode(str, Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


DEFAULT_WEIGHTS = {
    "pipeline": 0.40,
    "tgci": 0.20,
    "smart_money": 0.15,
    "regime": 0.15,
    "geometric": 0.10,
}

DEFAULT_THRESHOLDS = {
    "strong_buy": 75,
    "buy_conservative": 65,
    "buy_aggressive": 50,
    "watch": 40,
}

DEFAULT_MIN_CONTRIBUTING = {
    "conservative": 3,
    "aggressive": 2,
}


class MasterController:
    """서브시스템 가중합산 → 최종 판정."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        mc_cfg = cfg.get("master_controller", {})
        self.weights = mc_cfg.get("weights", DEFAULT_WEIGHTS)
        self.thresholds = mc_cfg.get("thresholds", DEFAULT_THRESHOLDS)
        self.min_contributing = mc_cfg.get(
            "min_contributing_systems", DEFAULT_MIN_CONTRIBUTING
        )

    def evaluate(self, sub_scores: Dict[str, float]) -> Dict[str, Any]:
        """서브시스템 점수를 가중 합산하여 최종 판정.

        Args:
            sub_scores: {
                "pipeline": 0-100,
                "tgci": 0-100,
                "smart_money": 0-100,
                "regime": 0-100,
                "geometric": 0-100,
            }
            키가 없으면 0으로 처리.

        Returns:
            {
                "master_score": int(0-100),
                "action": Action,
                "entry_mode": EntryMode | None,
                "contributing_systems": int,
                "weighted_breakdown": dict,
            }
        """
        weighted = {}
        total = 0.0
        for key, weight in self.weights.items():
            raw = max(0, min(100, sub_scores.get(key, 0)))
            contribution = raw * weight
            weighted[key] = round(contribution, 1)
            total += contribution

        master_score = int(round(total))
        master_score = max(0, min(100, master_score))

        contributing = sum(
            1 for key in self.weights if sub_scores.get(key, 0) > 50
        )

        tgci_above_50 = sub_scores.get("tgci", 0) > 50

        action, entry_mode = self._decide(
            master_score, contributing, tgci_above_50
        )

        return {
            "master_score": master_score,
            "action": action,
            "entry_mode": entry_mode,
            "contributing_systems": contributing,
            "weighted_breakdown": weighted,
        }

    def _decide(
        self, score: int, contributing: int, tgci_above_50: bool
    ) -> tuple:
        """판정 로직."""
        th = self.thresholds
        mc = self.min_contributing

        if score >= th["strong_buy"] and contributing >= 4:
            return Action.STRONG_BUY, EntryMode.CONSERVATIVE

        if score >= th["buy_conservative"] and contributing >= mc["conservative"]:
            return Action.BUY, EntryMode.CONSERVATIVE

        if (
            score >= th["buy_aggressive"]
            and contributing >= mc["aggressive"]
            and tgci_above_50
        ):
            return Action.BUY, EntryMode.AGGRESSIVE

        if score >= th["watch"]:
            return Action.WATCH, None

        return Action.HOLD, None
