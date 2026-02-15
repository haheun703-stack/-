"""
⑨ Divergence Detector — 3D 구조: 팩터간 모순/발산 감지

수학적 근거: "회전하면 보이는 구조"
  → 같은 데이터를 다른 각도에서 보면 다른 것이 보임
  → Price-Volume, Price-Flow, Volume-Flow 쌍이 모순될 때 감지

핵심 기능:
  1. 4개 축의 방향 분류 (-1, 0, +1)
  2. 6개 쌍 (C(4,2)=6)의 정합성 검사
  3. 위험 발산 / 기회 발산 분류

의존성: indicators.py 결과 (일봉) 또는 실시간 데이터
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ─── 발산 패턴 정의 ─────────────────────────────

DIVERGENCE_PATTERNS = [
    {
        "name": "허약한 상승",
        "condition": lambda d: d["price"] > 0 and d["volume"] < 0,
        "type": "risk",
        "description": "가격↑ + 거래량↓ = 동력 없는 상승",
    },
    {
        "name": "상승 피로",
        "condition": lambda d: d["price"] > 0 and d["momentum"] < 0,
        "type": "risk",
        "description": "가격↑ + 모멘텀↓ = 추세 약화 징후",
    },
    {
        "name": "과열 경고",
        "condition": lambda d: d["price"] > 0 and d["volume"] > 0 and d["flow"] < 0,
        "type": "risk",
        "description": "가격↑ + 거래량↑ + 수급↓ = 개인 과열 (기관 이탈)",
    },
    {
        "name": "매집 중",
        "condition": lambda d: d["price"] <= 0 and d["flow"] > 0,
        "type": "opportunity",
        "description": "가격↓/횡보 + 수급↑ = 스마트머니 매집",
    },
    {
        "name": "바닥 다지기",
        "condition": lambda d: d["momentum"] < 0 and d["flow"] > 0,
        "type": "opportunity",
        "description": "모멘텀↓ + 수급↑ = 에너지 축적 중",
    },
    {
        "name": "거래량 확인 상승",
        "condition": lambda d: d["price"] > 0 and d["volume"] > 0 and d["flow"] > 0,
        "type": "confirm",
        "description": "가격↑ + 거래량↑ + 수급↑ = 건전한 상승",
    },
    {
        "name": "투매 징후",
        "condition": lambda d: d["price"] < 0 and d["volume"] > 0 and d["flow"] < 0,
        "type": "risk",
        "description": "가격↓ + 거래량↑ + 수급↓ = 패닉 매도",
    },
    {
        "name": "조용한 이탈",
        "condition": lambda d: d["price"] <= 0 and d["volume"] < 0 and d["flow"] < 0,
        "type": "risk",
        "description": "가격↓ + 거래량↓ + 수급↓ = 관심 소멸",
    },
]


class DivergenceDetector:
    """팩터간 모순/발산 감지기"""

    def __init__(self, config: dict | None = None):
        self.enabled = (config or {}).get("geometry", {}).get("divergence", {}).get("enabled", True)

    # ─── 방향 분류 ───────────────────────────────

    @staticmethod
    def classify_direction(
        price_change: float = 0.0,
        volume_ratio: float = 1.0,
        foreign_net: float = 0.0,
        inst_net: float = 0.0,
        macd_histogram: float = 0.0,
    ) -> dict[str, int]:
        """
        4개 축의 방향을 -1, 0, +1로 분류.

        Parameters:
            price_change: 등락률 (%)
            volume_ratio: 거래량/20MA 비율
            foreign_net: 외인 순매수 (주)
            inst_net: 기관 순매수 (주)
            macd_histogram: MACD 히스토그램

        Returns:
            {"price": 1, "volume": -1, "flow": 1, "momentum": 0}
        """
        def sign_with_threshold(val, threshold=0):
            if val > threshold:
                return 1
            elif val < -threshold:
                return -1
            return 0

        return {
            "price": sign_with_threshold(price_change, 0.3),       # ±0.3% 이상이면 방향 있음
            "volume": sign_with_threshold(volume_ratio - 1.0, 0.2), # ±20% 이상이면 방향 있음
            "flow": sign_with_threshold(foreign_net + inst_net),
            "momentum": sign_with_threshold(macd_histogram),
        }

    # ─── 발산 감지 ───────────────────────────────

    def detect(
        self,
        price_change: float = 0.0,
        volume_ratio: float = 1.0,
        foreign_net: float = 0.0,
        inst_net: float = 0.0,
        macd_histogram: float = 0.0,
    ) -> dict:
        """
        현재 데이터에서 발산 패턴 감지.

        Returns:
            {
                "directions": {"price": 1, "volume": -1, ...},
                "divergences": [
                    {"name": "허약한 상승", "type": "risk", "description": "..."},
                ],
                "risk_count": 1,
                "opportunity_count": 0,
                "confirm_count": 0,
                "net_signal": -0.5,  # (기회 - 위험) / 전체
            }
        """
        dirs = self.classify_direction(
            price_change, volume_ratio, foreign_net, inst_net, macd_histogram,
        )

        if not self.enabled:
            return {
                "directions": dirs,
                "divergences": [],
                "risk_count": 0,
                "opportunity_count": 0,
                "confirm_count": 0,
                "net_signal": 0.0,
            }

        found = []
        for pattern in DIVERGENCE_PATTERNS:
            try:
                if pattern["condition"](dirs):
                    found.append({
                        "name": pattern["name"],
                        "type": pattern["type"],
                        "description": pattern["description"],
                    })
            except Exception:
                continue

        risk_count = sum(1 for d in found if d["type"] == "risk")
        opp_count = sum(1 for d in found if d["type"] == "opportunity")
        confirm_count = sum(1 for d in found if d["type"] == "confirm")

        total = risk_count + opp_count + confirm_count
        if total > 0:
            net_signal = (opp_count + confirm_count - risk_count) / total
        else:
            net_signal = 0.0

        return {
            "directions": dirs,
            "divergences": found,
            "risk_count": risk_count,
            "opportunity_count": opp_count,
            "confirm_count": confirm_count,
            "net_signal": round(net_signal, 2),
        }

    # ─── 행 데이터에서 직접 분석 ──────────────────

    def detect_from_row(self, row: dict) -> dict:
        """딕셔너리(row)에서 필요한 값을 추출하여 detect 호출"""
        return self.detect(
            price_change=row.get("change_pct", row.get("ret1", 0.0)) or 0.0,
            volume_ratio=row.get("volume_surge_ratio", row.get("volume_ratio", 1.0)) or 1.0,
            foreign_net=row.get("foreign_net_buy", row.get("foreign_net", 0.0)) or 0.0,
            inst_net=row.get("inst_net_buy", row.get("inst_net", 0.0)) or 0.0,
            macd_histogram=row.get("macd_histogram", 0.0) or 0.0,
        )

    # ─── 프롬프트 텍스트 ─────────────────────────

    @staticmethod
    def to_prompt_text(result: dict) -> str:
        """Claude API 입력용 텍스트"""
        lines = ["[발산 감지]"]
        if not result["divergences"]:
            lines.append("  발산 없음 (팩터 방향 일치)")
            return "\n".join(lines)

        type_icons = {"risk": "⚠", "opportunity": "✅", "confirm": "💪"}
        for d in result["divergences"]:
            icon = type_icons.get(d["type"], "•")
            lines.append(f"  {icon} {d['name']}: {d['description']}")

        # 종합
        net = result["net_signal"]
        if net > 0.3:
            conclusion = "긍정 우세"
        elif net < -0.3:
            conclusion = "위험 우세"
        else:
            conclusion = "중립"
        lines.append(
            f"  종합: 위험 {result['risk_count']}건, "
            f"기회 {result['opportunity_count']}건, "
            f"확인 {result['confirm_count']}건 → {conclusion}"
        )
        return "\n".join(lines)
