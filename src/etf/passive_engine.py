"""패시브 ETF 엔진 — 금/소형주/채권/달러 통합 관리

각 ARM당 1개 ETF를 보유하는 단순 패시브 전략.
regime_allocation에서 비중 > 0이면 보유, 0이면 미보유.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PassiveETFEngine:
    """패시브 ETF 관리 (금/소형주/채권/달러).

    regime_allocation에서 비중 > 0이면 보유, 0이면 미보유.
    각 ARM당 1개 ETF만 관리.
    """

    ETF_MAP = {
        "gold":      {"code": "132030", "name": "KODEX 골드선물(H)"},
        "small_cap": {"code": "229200", "name": "KODEX 코스닥150"},
        "bonds":     {"code": "114820", "name": "KODEX 국고채10년"},
        "dollar":    {"code": "261240", "name": "KODEX 미국달러선물"},
    }

    ARM_LABELS = {
        "gold": "금",
        "small_cap": "소형주",
        "bonds": "채권",
        "dollar": "달러",
    }

    def run(self, arm_name: str, allocation_pct: float, regime: str) -> dict:
        """단일 패시브 ARM 시그널 생성.

        Args:
            arm_name: "gold" | "small_cap" | "bonds" | "dollar"
            allocation_pct: BRAIN이 결정한 비중 (%)
            regime: 현재 레짐

        Returns:
            dict with signal, etf_code, etf_name, allocation_pct
        """
        etf = self.ETF_MAP.get(arm_name)
        label = self.ARM_LABELS.get(arm_name, arm_name)

        if not etf:
            logger.warning("알 수 없는 패시브 ARM: %s", arm_name)
            return {
                "arm": arm_name,
                "signal": "NONE",
                "reason": f"미등록 ARM: {arm_name}",
            }

        if allocation_pct <= 0:
            return {
                "arm": arm_name,
                "signal": "SELL",
                "etf_code": etf["code"],
                "etf_name": etf["name"],
                "allocation_pct": 0,
                "reason": f"{label} ETF 비중 0% ({regime})",
            }

        return {
            "arm": arm_name,
            "signal": "BUY",
            "etf_code": etf["code"],
            "etf_name": etf["name"],
            "allocation_pct": allocation_pct,
            "reason": f"{label} ETF {allocation_pct:.0f}% ({regime})",
        }

    def run_all(self, allocation: dict, regime: str) -> dict:
        """4개 패시브 ARM 일괄 실행.

        Args:
            allocation: {"gold": 5, "small_cap": 10, "bonds": 0, "dollar": 0, ...}
            regime: 현재 레짐

        Returns:
            dict[arm_name] → 시그널 결과
        """
        results = {}
        for arm_name in self.ETF_MAP:
            alloc = float(allocation.get(arm_name, 0))
            results[arm_name] = self.run(arm_name, alloc, regime)

        active = [k for k, v in results.items() if v["signal"] == "BUY"]
        if active:
            labels = [self.ARM_LABELS.get(a, a) for a in active]
            logger.info("패시브 ETF 활성: %s", ", ".join(labels))
        else:
            logger.info("패시브 ETF: 전체 미보유")

        return results
